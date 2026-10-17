"""Decider lifecycle: health probes, auto-restart after a crash or a hang with exponential backoff (never under
memory pressure, bounded per hour), breaker recovery, and a warm-up before the deployment counts as ready."""
from types import SimpleNamespace

import pytest

from dgxmm.decider.service import Decider
from dgxmm.decider.supervisor import DeciderSupervisor
from dgxmm.settings_store import SettingsStore
from tests.test_decider_engine import d3, deploy_decider  # noqa: F401  (fixture)


class Clock:
    def __init__(self):
        self.t = 10_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def sup(d3):  # noqa: F811
    store = SettingsStore(d3.db, d3.settings)
    decider = Decider(db=d3.db, settings=d3.settings, store=store, deployer=lambda: d3.deployer,
                      http=lambda: d3.deployer.http)
    clock = Clock()
    watchdog = SimpleNamespace(level="ok")
    s = DeciderSupervisor(decider=decider, deployer=lambda: d3.deployer, watchdog=lambda: watchdog,
                          settings=d3.settings, db=d3.db, clock=clock)
    return SimpleNamespace(stack=d3, decider=decider, sup=s, clock=clock, watchdog=watchdog)


async def running_decider(t, ready=True):
    view = await deploy_decider(t.stack)
    t.decider.set_deployment_id(view["id"])
    t.decider.set_enabled(True)
    if ready:
        t.stack.http.healthy.add(view["port"])
        assert await t.stack.deployer.wait_ready(view["id"], 5) == "ready"
    return view


async def crash(t, view):
    v = t.stack.deployer.view(view["id"])
    c = t.stack.sdk.container(v["container_id"])
    c.status, c.exit_code = "exited", 1
    if v["state"] in ("starting", "loading"):
        await t.stack.deployer.wait_ready(view["id"], 5)  # the start-up monitor sees the exit
    else:
        await t.stack.deployer.check_liveness()
    assert t.stack.deployer.view(view["id"])["state"] == "failed"


async def test_disabled_or_absent_decider_is_left_alone(sup):
    assert await sup.sup.tick() == "idle"
    view = await running_decider(sup)
    sup.decider.set_enabled(False)
    await crash(sup, view)
    assert await sup.sup.tick() == "idle"
    assert sup.stack.deployer.view(view["id"])["state"] == "failed"


async def test_healthy_probe_and_breaker_recovery(sup):
    await running_decider(sup)
    assert await sup.sup.tick() == "probe_ok"
    st = sup.sup.status()
    assert st["probe"]["ok"] is True and st["probe"]["latency_ms"] is not None and st["restarts_last_hour"] == 0
    c = sup.decider.client()
    for _ in range(3):
        c.breaker.failure()
    assert c.breaker.is_open() and not sup.decider.healthy()
    assert await sup.sup.tick() == "probe_ok"
    assert not c.breaker.is_open() and sup.decider.healthy()  # a good probe closes the breaker early


async def test_crash_restarts_with_exponential_backoff(sup):
    view = await running_decider(sup)
    await crash(sup, view)
    assert await sup.sup.tick() == "restarted"
    assert sup.stack.deployer.view(view["id"])["state"] in ("starting", "loading")
    await crash(sup, view)
    assert await sup.sup.tick() == "backoff"  # too soon after the last restart
    st = sup.sup.status()
    assert st["next_attempt_in_s"] == pytest.approx(sup.stack.settings.decider_restart_backoff_s, abs=1)
    sup.clock.t += sup.stack.settings.decider_restart_backoff_s + 1
    assert await sup.sup.tick() == "restarted"
    assert sup.sup.status()["next_attempt_in_s"] == pytest.approx(2 * sup.stack.settings.decider_restart_backoff_s,
                                                                  abs=1)
    assert "exited" in sup.sup.status()["last_reason"]


async def test_never_restarts_under_memory_pressure(sup):
    view = await running_decider(sup)
    await crash(sup, view)
    sup.watchdog.level = "warning"
    assert await sup.sup.tick() == "memory_pressure"
    assert sup.stack.deployer.view(view["id"])["state"] == "failed"
    sup.watchdog.level = "ok"
    sup.stack.probe.available = 1 * 1024**3  # does not fit: restart refused by the normal fit check
    assert await sup.sup.tick() == "memory_pressure"
    assert sup.stack.deployer.view(view["id"])["state"] == "failed"
    assert "memory" in sup.sup.status()["last_reason"]


async def test_user_stop_is_respected(sup):
    view = await running_decider(sup)
    await sup.stack.deployer.stop(view["id"], 1)
    assert await sup.sup.tick() == "idle"
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"


async def test_hung_decider_is_restarted_after_consecutive_failed_probes(sup):
    view = await running_decider(sup)
    sup.stack.http.healthy.discard(view["port"])  # container still running but no longer answering
    n = sup.stack.settings.decider_hang_probes
    for _ in range(n - 1):
        assert await sup.sup.tick() == "probe_failed"
    assert await sup.sup.tick() == "restarted"
    assert "not answering" in sup.sup.status()["last_reason"]


async def test_gives_up_after_too_many_restarts_in_an_hour(sup):
    view = await running_decider(sup)
    limit = sup.stack.settings.decider_max_restarts_per_hour
    for _ in range(limit):
        await crash(sup, view)
        sup.clock.t += (sup.sup.status()["next_attempt_in_s"] or 0) + 1  # just past the backoff, within the hour
        assert await sup.sup.tick() == "restarted"
    await crash(sup, view)
    sup.clock.t += sup.sup.status()["next_attempt_in_s"] + 1
    assert await sup.sup.tick() == "gave_up"
    assert sup.sup.status()["gave_up"] is True
    sup.decider.set_enabled(False)
    await sup.sup.tick()
    sup.decider.set_enabled(True)  # re-enabling (admin action) resets the give-up
    sup.sup.reset()
    assert sup.sup.status()["gave_up"] is False


async def test_backoff_resets_after_a_stable_period(sup):
    view = await running_decider(sup)
    await crash(sup, view)
    assert await sup.sup.tick() == "restarted"
    sup.stack.http.healthy.add(sup.stack.deployer.view(view["id"])["port"])
    assert await sup.stack.deployer.wait_ready(view["id"], 5) == "ready"
    sup.clock.t += sup.stack.settings.decider_stable_s + 1
    assert await sup.sup.tick() == "probe_ok"
    assert sup.sup.status()["backoff_s"] == sup.stack.settings.decider_restart_backoff_s


async def test_warmup_runs_realistic_questions_before_ready(d3):  # noqa: F811
    view = await deploy_decider(d3)
    d3.http.healthy.add(view["port"])
    assert await d3.deployer.wait_ready(view["id"], 5) == "ready"
    posts = [c for c in d3.http.calls if c[2] == "/v1/systemone"]
    assert len(posts) >= 3  # the probe plus warm-up questions shaped like the features' (many options, long state)


async def test_failed_warmup_keeps_it_starting(d3):  # noqa: F811
    from dgxmm.engines.decision import DecisionEngine, warmup_requests
    assert len(warmup_requests()) >= 2
    eng: DecisionEngine = d3.deployer.engines["decision"]
    view = await deploy_decider(d3)
    d3.http.healthy.add(view["port"])
    d3.http.fail_paths.add("/v1/systemone")
    assert await eng.health(d3.deployer.http, d3.deployer.base_url(view["name"], view["port"]), "d3-mini") is False


def test_status_endpoint_reports_breaker_load_and_supervisor(api_stack):
    from tests.test_api_deployments import admin
    admin(api_stack)
    st = api_stack.get("/dgxmm/api/decider/status").json()
    assert st["load"]["max_in_flight"] == api_stack.ctx.settings.decider_max_in_flight
    assert "supervisor" in st and "breaker" in st


# ---------- fix round 1 ----------
async def hang(t, view):
    t.stack.http.healthy.discard(view["port"])
    for _ in range(t.stack.settings.decider_hang_probes - 1):
        assert await t.sup.tick() == "probe_failed"


async def test_supervisor_stop_then_failed_restart_is_retried_not_taken_for_an_admin_stop(sup):
    view = await running_decider(sup)
    await hang(sup, view)
    real_restart = sup.stack.deployer.restart

    async def no_memory(*a, **kw):
        from dgxmm.deployer.service import InsufficientMemory
        from dgxmm.planner.fit import check_fit
        prep = sup.stack.deployer.prepare("decision", view["model_ref"], {})
        sup.stack.probe.available = 1 * 1024**3
        raise InsufficientMemory(check_fit(prep.estimate, await sup.stack.budget.refresh()))
    sup.stack.deployer.restart = no_memory
    assert await sup.sup.tick() == "memory_pressure"
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"
    st = sup.sup.status()
    assert st["pending_restart"] and "not answering" in st["pending_restart"]
    sup.stack.deployer.restart = real_restart
    sup.stack.probe.available = 100 * 1024**3
    assert await sup.sup.tick() == "backoff"  # retried, with backoff
    sup.clock.t += sup.sup.status()["next_attempt_in_s"] + 1
    assert await sup.sup.tick() == "restarted"
    assert sup.stack.deployer.view(view["id"])["state"] in ("starting", "loading")
    assert sup.sup.status()["pending_restart"] is None


async def test_admin_disable_during_the_supervisors_stop_wins(sup):
    view = await running_decider(sup)
    await hang(sup, view)
    real_stop = sup.stack.deployer.stop

    async def stop_and_disable(*a, **kw):
        out = await real_stop(*a, **kw)
        sup.decider.set_enabled(False)  # an admin pressed Disable meanwhile
        return out
    sup.stack.deployer.stop = stop_and_disable
    assert await sup.sup.tick() == "idle"
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"
    assert sup.sup.status()["pending_restart"] is None
    assert await sup.sup.tick() == "idle"


async def test_a_switched_model_that_fails_rolls_back_to_the_previous_one(sup, fake_hf_repo):
    from tests.test_decider_models import NANO, add_models
    add_models(fake_hf_repo, sup.stack.inv, which=(NANO,))
    view = await running_decider(sup)
    mini = view["model_ref"]
    # what /api/decider/enable does for a model change
    await sup.stack.deployer.stop(view["id"], 1)
    sup.decider.set_model(NANO)
    await sup.stack.deployer.restart(view["id"], 1, model_ref=NANO)
    sup.decider.set_switch_from(mini)
    await crash(sup, view)  # the new model fails to start
    assert await sup.sup.tick() == "rolled_back"
    v = sup.stack.deployer.view(view["id"])
    assert v["model_ref"] == mini and v["state"] in ("starting", "loading")
    assert sup.decider.configured_model == mini and sup.decider.switch_from is None
    assert "rolled back to" in sup.sup.status()["last_reason"]


async def test_switch_marker_clears_once_the_new_model_is_ready(sup, fake_hf_repo):
    from tests.test_decider_models import NANO, add_models
    add_models(fake_hf_repo, sup.stack.inv, which=(NANO,))
    view = await running_decider(sup)
    await sup.stack.deployer.stop(view["id"], 1)
    sup.decider.set_model(NANO)
    v = await sup.stack.deployer.restart(view["id"], 1, model_ref=NANO)
    sup.decider.set_switch_from(view["model_ref"])
    sup.stack.http.healthy.add(v["port"])
    assert await sup.stack.deployer.wait_ready(view["id"], 5) == "ready"
    assert await sup.sup.tick() == "probe_ok"
    assert sup.decider.switch_from is None
