import asyncio

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name="tiny", ref="Qwen/Qwen3-Tiny", **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg})


async def become_ready(stack, dep):
    stack.http.healthy.add(dep["port"])
    return await stack.deployer.wait_ready(dep["id"], 5)


async def test_deploy_reaches_ready_with_computed_util(stack):
    dep = await stack.deployer.deploy(req(), user_id=1)
    assert dep["state"] in ("starting", "loading") and dep["port"] == 8100
    assert await become_ready(stack, dep) == "ready"
    call = stack.sdk.run_calls[0]
    cmd = call["command"]
    util = float(cmd[cmd.index("--gpu-memory-utilization") + 1])
    assert util == pytest.approx(dep["reserved_bytes"] / stack.probe.total, abs=0.011)
    with stack.db.session() as s:
        states = [e.to_state for e in s.scalars(select(DeploymentEvent).order_by(DeploymentEvent.id))]
    assert states == ["starting", "loading", "ready"]


async def test_refuses_when_not_enough_memory(stack):
    stack.probe.available = 20 * GIB  # 101 GiB used by something external
    with pytest.raises(InsufficientMemory) as e:
        await stack.deployer.deploy(req(ref="org/big-model"), user_id=1)
    assert e.value.fit.required > e.value.fit.free_for_new
    assert stack.sdk.run_calls == []


async def test_concurrent_deploys_cannot_both_claim_memory(stack):
    # free_for_new = 121 - 12 - 21 external = 88 GiB; each big deploy needs ~50 GiB
    results = await asyncio.gather(stack.deployer.deploy(req("a", "org/big-model").model_copy(update={"alias": "a"}), 1),
                                   stack.deployer.deploy(req("b", "org/big-model").model_copy(update={"alias": "b"}), 1),
                                   return_exceptions=True)
    assert sum(isinstance(r, InsufficientMemory) for r in results) == 1
    assert len(stack.sdk.run_calls) == 1


async def test_exit_137_marks_failed_likely_oom(stack):
    dep = await stack.deployer.deploy(req(), 1)
    c = stack.sdk.container(dep["container_id"])
    c.log_text = "CUDA out of memory"
    c.status, c.exit_code = "exited", 137
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    v = stack.deployer.view(dep["id"], with_events=True)
    assert v["exit_code"] == 137 and "likely out of memory" in v["last_error"]
    assert "CUDA out of memory" in v["events"][0]["log_tail"]


async def test_memory_drop_during_load_aborts(stack):
    dep = await stack.deployer.deploy(req(), 1)
    stack.probe.available = 4 * GIB
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    assert "safety margin" in stack.deployer.view(dep["id"])["last_error"]
    assert stack.sdk._containers == {}


async def test_startup_timeout(stack):
    stack.settings.startup_timeout_s = 0
    dep = await stack.deployer.deploy(req(), 1)
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    assert "timed out" in stack.deployer.view(dep["id"])["last_error"]


async def test_stop_then_restart_reuses_row(stack):
    dep = await stack.deployer.deploy(req(), 1)
    await become_ready(stack, dep)
    stopped = await stack.deployer.stop(dep["id"], 1)
    assert stopped["state"] == "stopped" and stopped["container_id"] is None and stack.sdk._containers == {}
    again = await stack.deployer.restart(dep["id"], 1, config_override={**SMALL, "max_model_len": 2048})
    assert again["state"] in ("starting", "loading") and again["config"]["max_model_len"] == 2048
    assert await become_ready(stack, again) == "ready"


async def test_input_validation(stack):
    with pytest.raises(ValidationError):
        DeployRequest(name="bad name;", engine="vllm", model_ref="Qwen/Qwen3-Tiny")
    for ref, code in (("../../etc", "invalid_model_ref"), ("nope/nope", "model_not_found"),
                      ("org/partial", "model_incomplete")):
        with pytest.raises(DeployError) as e:
            await stack.deployer.deploy(req(ref=ref), 1)
        assert e.value.code == code
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(extra_args=["--seed", "1;rm"]), 1)
    assert e.value.code == "invalid_config" and e.value.status == 422
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(engine_typo=1), 1)
    assert e.value.code == "invalid_config"


async def test_name_in_use_and_port_allocation(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b").model_copy(update={"alias": "b"}), 1)  # distinct LiteLLM model names (same model twice)
    assert (a["port"], b["port"]) == (8100, 8101)
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req("a"), 1)
    assert e.value.code == "name_in_use"


async def test_docker_failure_marks_failed(stack):
    stack.sdk.fail_run = True
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(), 1)
    assert e.value.code == "docker_error"
    with stack.db.session() as s:
        assert s.scalar(select(Deployment)).state == "failed"


async def test_preview_reports_util_and_fit(stack):
    p = await stack.deployer.preview("vllm", "Qwen/Qwen3-Tiny", SMALL)
    assert p["fit"]["fits"] and 0 < p["gpu_memory_utilization"] < 1
    assert p["served_name"] == "qwen3-tiny"


async def test_state_hooks_called(stack):
    seen = []

    async def hook(dep_id, a, b):
        seen.append(b)

    stack.deployer.state_hooks.append(hook)
    dep = await stack.deployer.deploy(req(), 1)
    await become_ready(stack, dep)
    await asyncio.sleep(0.05)
    assert seen[-1] == "ready"
