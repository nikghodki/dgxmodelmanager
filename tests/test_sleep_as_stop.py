"""vLLM sleep on unified memory (GB10): level-1 sleep backs the weights up into the same RAM, so it frees nothing and
vLLM's own check fails ("Memory usage increased after sleeping"). ``vllm_sleep: stop`` (the default) makes a
single-node vLLM "sleep" a stop of the deployment, like a cluster's (SLEEP_AS_STOP_LABEL semantics), and a wake starts
it again; ``vllm_sleep: level1`` opts back into vLLM's sleep. A failed engine sleep never leaves a deployment "ready":
it is restarted (or ends failed)."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from dgxmm.config import Settings
from dgxmm.db import DeploymentEvent
from dgxmm.deployer.service import DeployError, DeployRequest

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048, "enable_sleep_mode": True}
SINGLE_LABEL = "sleep (stops the model)"


async def ready(stack, name="a", **cfg):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, alias=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={**SMALL, **cfg}), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return stack.deployer.view(d["id"])


def events(stack, dep_id):
    with stack.db.session() as s:
        return [(e.from_state, e.to_state, e.actor, e.message) for e in s.scalars(
            select(DeploymentEvent).where(DeploymentEvent.deployment_id == dep_id).order_by(DeploymentEvent.id))]


def engine_paths(stack):
    return [p for _, _, p in stack.http.calls]


def test_vllm_sleep_setting_defaults_to_stop_and_is_validated():
    assert Settings().vllm_sleep == "stop"
    assert Settings(vllm_sleep="level1").vllm_sleep == "level1"
    with pytest.raises(ValidationError):
        Settings(vllm_sleep="level2")


async def test_stop_mode_runs_without_sleep_mode_but_keeps_the_stored_config(stack):
    v = await ready(stack)
    c = stack.sdk.container(v["container_id"])
    assert "--enable-sleep-mode" not in c.command and "VLLM_SERVER_DEV_MODE" not in c.env
    # an existing (or new) deployment configured with sleep mode keeps it: no validation error, opt-in later works
    assert stack.deployer.view(v["id"])["config"]["enable_sleep_mode"] is True


async def test_stop_mode_view_labels_sleep_as_a_stop(stack):
    v = await ready(stack)
    assert v["sleep_capable"] is False and v["sleep_label"] == SINGLE_LABEL


async def test_no_label_without_sleep_mode(stack):
    v = await ready(stack, enable_sleep_mode=False)
    assert v["sleep_capable"] is False and v["sleep_label"] is None


async def test_sleep_stops_the_deployment_and_wake_starts_it(stack):
    v = await ready(stack)
    out = await stack.deployer.sleep(v["id"])
    assert out["state"] == "stopped"
    assert "/sleep" not in engine_paths(stack)
    assert stack.sdk.containers.list() == []  # the container is gone: its memory is released
    evs = events(stack, v["id"])
    assert ("ready", "stopping", "sleep", f"{SINGLE_LABEL}: stopping the container") in evs
    woke = await stack.deployer.wake(v["id"])
    assert woke["state"] in ("starting", "loading", "ready")
    assert await stack.deployer.wait_ready(v["id"], 5) == "ready"
    assert "/wake_up" not in engine_paths(stack)


async def test_wake_does_not_start_a_deployment_the_user_stopped(stack):
    v = await ready(stack)
    await stack.deployer.stop(v["id"])
    with pytest.raises(DeployError) as e:
        await stack.deployer.wake(v["id"])
    assert e.value.code == "not_sleeping"


async def test_apply_actions_sleep_as_stop_is_undone_by_a_start(stack):
    from dgxmm.planner.plans import PlanAction
    v = await ready(stack)
    applied = await stack.deployer.apply_actions([PlanAction(type="sleep", deployment_id=v["id"])])
    assert applied == [("stop", v["id"])]
    assert await stack.deployer.undo_actions(applied) == []
    assert await stack.deployer.wait_ready(v["id"], 5) == "ready"


async def test_level1_opt_in_keeps_vllms_sleep(stack):
    stack.settings.vllm_sleep = "level1"
    v = await ready(stack)
    c = stack.sdk.container(v["container_id"])
    assert "--enable-sleep-mode" in c.command and c.env.get("VLLM_SERVER_DEV_MODE") == "1"
    assert v["sleep_capable"] is True and v["sleep_label"] is None
    assert (await stack.deployer.sleep(v["id"]))["state"] == "sleeping"
    assert "/sleep" in engine_paths(stack)


async def test_a_deployment_asleep_before_the_switch_to_stop_still_wakes(stack):
    stack.settings.vllm_sleep = "level1"
    v = await ready(stack)
    await stack.deployer.sleep(v["id"])
    stack.settings.vllm_sleep = "stop"  # upgraded / reconfigured while it sleeps (it runs with sleep mode)
    assert (await stack.deployer.wake(v["id"]))["state"] == "ready"
    assert "/wake_up" in engine_paths(stack)


async def test_on_demand_idle_reaper_stops_without_an_engine_sleep(stack, settings, db):
    from dgxmm.ondemand.controller import OnDemandController
    v = await ready(stack)
    stack.deployer.set_mode(v["id"], "on_demand", 60)
    now = [0.0]
    ctx = SimpleNamespace(settings=settings, db=db, deployer=stack.deployer, metrics=None, budget=stack.budget,
                          litellm=SimpleNamespace(trigger=lambda: None))
    ctl = OnDemandController(ctx, clock=lambda: now[0], wall=lambda: 1_700_000_000.0 + now[0])
    ctl.attach(stack.deployer)
    assert await ctl.reap_once() == []  # first seen: its idle timer starts now
    now[0] += 61
    assert await ctl.reap_once() == [v["id"]]
    assert stack.deployer.view(v["id"])["state"] == "stopped"
    assert "/sleep" not in engine_paths(stack)


# ---------- a failed engine sleep (vllm_sleep: level1) ----------
async def test_failed_sleep_restarts_the_deployment(stack):
    stack.settings.vllm_sleep = "level1"
    v = await ready(stack)
    stack.http.fail_paths.add("/sleep")  # e.g. vLLM's "Memory usage increased after sleeping." (HTTP 500)
    with pytest.raises(DeployError) as e:
        await stack.deployer.sleep(v["id"])
    assert e.value.code == "engine_error" and "restart" in e.value.message
    after = stack.deployer.view(v["id"])
    assert after["state"] in ("queued", "starting", "loading", "ready")  # never "ready" on the old (asleep?) engine
    assert after["container_id"] != v["container_id"]
    assert v["container_id"] not in {c.id for c in stack.sdk.containers.list()}
    assert "sleep failed" in (after["last_error"] or "")
    assert await stack.deployer.wait_ready(v["id"], 5) == "ready"
    assert "sleep failed" in (stack.deployer.view(v["id"])["last_error"] or "")
    assert any(to == "stopping" and "sleep failed" in (msg or "") for _, to, _, msg in events(stack, v["id"]))


async def test_failed_sleep_with_any_error_restarts(stack, monkeypatch):
    stack.settings.vllm_sleep = "level1"
    v = await ready(stack)

    async def boom(*a, **kw):
        raise RuntimeError("engine went away")
    monkeypatch.setattr(stack.deployer.engines["vllm"], "sleep", boom)
    with pytest.raises(DeployError):
        await stack.deployer.sleep(v["id"])
    assert await stack.deployer.wait_ready(v["id"], 5) == "ready"
    assert "engine went away" in stack.deployer.view(v["id"])["last_error"]


async def test_failed_sleep_whose_restart_fails_ends_failed(stack, monkeypatch):
    stack.settings.vllm_sleep = "level1"
    v = await ready(stack)
    stack.http.fail_paths.add("/sleep")

    async def no_restart(*a, **kw):
        raise DeployError("insufficient_memory", "does not fit", 409)
    monkeypatch.setattr(stack.deployer, "restart", no_restart)
    with pytest.raises(DeployError):
        await stack.deployer.sleep(v["id"])
    after = stack.deployer.view(v["id"])
    assert after["state"] == "failed"
    assert "sleep failed" in after["last_error"] and "does not fit" in after["last_error"]
