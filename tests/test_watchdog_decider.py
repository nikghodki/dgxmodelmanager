"""M3: a runaway decision model is stopped by the memory watchdog before any user deployment, and the supervisor
restarts it only after a backoff (never in a loop, never under memory pressure)."""
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import AuditLog
from dgxmm.deployer.service import DeployRequest
from dgxmm.memory.budget import DeploymentUsage, compute_budget
from dgxmm.memory.hostmem import HostMemory
from dgxmm.watchdog import OVERUSE_SAMPLES, Watchdog
from tests.test_decider_engine import d3  # noqa: F401  (fixture)
from tests.test_decider_supervisor import Clock, running_decider, sup  # noqa: F401  (fixture)
from tests.test_watchdog import SMALL


def usage_budget(settings, view, actual_gib, avail_gib=60):
    u = DeploymentUsage(view["id"], view["name"], "ready", view["reserved_bytes"], int(actual_gib * GIB),
                        view.get("container_id"))
    return compute_budget(HostMemory(121 * GIB, int(avail_gib * GIB), 0, 0, 0, 0), [u], settings)


def watchdog_for(t, clock=None):
    w = Watchdog(t.stack.settings, t.stack.deployer, t.stack.events, t.stack.db, clock=clock or Clock())
    w.system_stop_listeners.append(t.sup.on_watchdog_stop)
    return w


async def test_runaway_decider_is_stopped_and_restarted_only_after_backoff(sup):  # noqa: F811
    view = await running_decider(sup)
    assert view["reserved_bytes"] == 13 * GIB
    w = watchdog_for(sup)
    over = 13 + sup.stack.settings.decider_overuse_margin_gib + 1
    for _ in range(OVERUSE_SAMPLES - 1):
        await w.check(usage_budget(sup.stack.settings, view, over))
        assert sup.stack.deployer.view(view["id"])["state"] == "ready"  # one sample is not enough
    await w.check(usage_budget(sup.stack.settings, view, over))
    v = sup.stack.deployer.view(view["id"], with_events=True)
    assert v["state"] == "stopped"
    assert any("over its 13.0 GiB reservation" in (e.get("message") or "") for e in v["events"])
    assert "decision model used 16.0 GiB" in w.state()["action"]
    with sup.stack.db.session() as s:
        a = s.scalars(select(AuditLog).where(AuditLog.action == "watchdog_stop")).one()
        assert a.target == view["name"]
    st = sup.sup.status()
    assert st["last_action"] == "watchdog_stopped" and "over its" in st["last_reason"]
    assert st["pending_restart"] and st["next_attempt_in_s"] > 0
    # not restarted at once (that would loop): backoff first, then a normal restart
    assert await sup.sup.tick() == "backoff"
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"
    sup.clock.t += st["next_attempt_in_s"] + 1
    assert await sup.sup.tick() == "restarted"
    assert sup.sup.status()["backoff_s"] >= 4 * sup.stack.settings.decider_restart_backoff_s  # doubled twice


async def test_within_margin_or_a_single_spike_is_left_alone(sup):  # noqa: F811
    view = await running_decider(sup)
    w = watchdog_for(sup)
    within = 13 + sup.stack.settings.decider_overuse_margin_gib - 0.5
    for _ in range(5):
        await w.check(usage_budget(sup.stack.settings, view, within))
    over = 13 + sup.stack.settings.decider_overuse_margin_gib + 1
    for _ in range(5):  # spikes separated by normal samples never add up
        await w.check(usage_budget(sup.stack.settings, view, over))
        await w.check(usage_budget(sup.stack.settings, view, 12))
    assert sup.stack.deployer.view(view["id"])["state"] == "ready"


async def test_no_restart_while_memory_is_under_pressure(sup):  # noqa: F811
    view = await running_decider(sup)
    w = watchdog_for(sup)
    for _ in range(OVERUSE_SAMPLES):
        await w.check(usage_budget(sup.stack.settings, view, 30))
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"
    sup.clock.t += 10_000
    sup.watchdog.level = "warning"
    assert await sup.sup.tick() == "memory_pressure"
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"


async def test_auto_protect_picks_the_decider_before_any_user_deployment(sup):  # noqa: F811
    st = sup.stack
    view = await running_decider(sup)
    st.http.all_healthy = True
    lo = await st.deployer.deploy(DeployRequest(name="lo", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL,
                                                priority=0), 1)
    await st.deployer.wait_ready(lo["id"], 5)
    st.settings.auto_protect = True
    now = [1000.0]
    w = watchdog_for(sup, clock=lambda: now[0])
    critical = compute_budget(HostMemory(121 * GIB, 3 * GIB, 0, 0, 0, 0), [], st.settings)
    await w.check(critical)
    assert st.deployer.view(view["id"])["state"] == "stopped"
    assert st.deployer.view(lo["id"])["state"] == "ready"
    assert sup.sup.status()["last_action"] == "watchdog_stopped"
    now[0] += 61
    await w.check(critical)  # no decider left to sacrifice: the lowest-priority user deployment
    assert st.deployer.view(lo["id"])["state"] == "stopped"


def test_wire_features_connects_the_supervisor_to_the_watchdog(stack):
    from dgxmm.context import AppContext
    from dgxmm.decider.features import wire_features
    from tests.fakes import FakeDecider
    w = Watchdog(stack.settings, stack.deployer, stack.events, stack.db)
    fake = FakeDecider()
    fake.supervisor = None
    fake.enabled = False
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, inventory=stack.inv,
                     budget=stack.budget, deployer=stack.deployer, docker=stack.deployer.docker,
                     engines=stack.deployer.engines, probe=stack.probe, decider=fake, watchdog=w)
    wire_features(ctx)
    assert w.system_stop_listeners == [fake.supervisor.on_watchdog_stop]
