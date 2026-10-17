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


async def test_exit_137_oom_killed_marks_failed_out_of_memory(stack):
    dep = await stack.deployer.deploy(req(), 1)
    c = stack.sdk.container(dep["container_id"])
    c.log_text = "CUDA out of memory"
    c.status, c.exit_code, c.oom_killed = "exited", 137, True
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    v = stack.deployer.view(dep["id"], with_events=True)
    assert v["exit_code"] == 137 and "out of memory: Docker reports" in v["last_error"]
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


async def test_port_held_by_a_host_process_is_skipped(stack):
    """DGXMM runs in a container and cannot see host listeners; docker reports the conflict at start. Real case:
    `failed to bind host port 127.0.0.1:8100/tcp: address already in use` (tests/fixtures/real_failures)."""
    lo = stack.settings.port_range[0]
    stack.sdk.busy_host_ports = {lo, lo + 1}
    dep = await stack.deployer.deploy(req(), 1)
    assert dep["state"] == "starting" and dep["port"] == lo + 2
    assert [c["ports"] for c in stack.sdk.run_calls][-1] == {f"{lo + 2}/tcp": ("127.0.0.1", lo + 2)}
    assert "--port" in stack.sdk.run_calls[-1]["command"] and str(lo + 2) in stack.sdk.run_calls[-1]["command"]


async def test_port_conflicts_give_up_after_a_few_ports(stack):
    lo, hi = stack.settings.port_range
    stack.sdk.busy_host_ports = set(range(lo, hi + 1))
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(), 1)
    assert e.value.code == "docker_error" and "address already in use" in e.value.message
    assert len(stack.sdk.run_calls) == 4
    # docker created the container before the start failed: nothing may stay behind (real leftover on the DGX)
    assert stack.sdk._containers == {}


async def test_reclassified_model_new_deploy_refused_existing_deployment_restarts(stack, fake_hf_repo):
    """A *ForCausalLM decision checkpoint (decider_config.json) is now kind `decision`: no NEW vLLM chat deployment,
    but a deployment created before the reclassification keeps restarting and reconfiguring."""
    from tests.conftest import QWEN_TINY
    fake_hf_repo("example/decision-checkpoint", config=QWEN_TINY,
                 files={"model.safetensors": GIB, "decider_config.json": 10})
    stack.inv.refresh()
    assert stack.inv.get("example/decision-checkpoint").kind == "decision"
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req("new", "example/decision-checkpoint"), 1)
    assert e.value.code == "engine_unsupported"
    cfg = stack.deployer.engines["vllm"].validate_config(
        {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}).model_dump(mode="json")
    with stack.db.session() as s:
        row = Deployment(name="old", engine="vllm", model_ref="example/decision-checkpoint", config=cfg,
                         state="stopped", port=None)
        s.add(row)
        s.flush()
        dep_id = row.id
    stack.http.all_healthy = True
    v = await stack.deployer.restart(dep_id, 1)
    assert v["state"] in ("starting", "loading") and await stack.deployer.wait_ready(dep_id, 5) == "ready"
    v = await stack.deployer.reconfigure(dep_id, {**cfg, "max_num_seqs": 1})
    assert v["config"]["max_num_seqs"] == 1 and v["model_ref"] == "example/decision-checkpoint"
