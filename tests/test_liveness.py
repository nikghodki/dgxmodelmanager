"""I2: ready/sleeping/unknown deployments are re-checked; crashed engines go to failed."""
import asyncio

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployRequest
from dgxmm.memory.calibration import Calibration

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def ready(stack, name, **cfg):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={**SMALL, **cfg}), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return stack.deployer.view(d["id"])


async def test_container_exit_after_ready_fails_and_records_oom(stack):
    cal = Calibration(stack.db)
    stack.deployer.calibration = cal
    seen = []

    async def hook(dep_id, frm, to):
        seen.append((dep_id, frm, to))
    stack.deployer.state_hooks.append(hook)
    d = await ready(stack, "a")
    c = stack.sdk.container(d["container_id"])
    c.status, c.exit_code = "exited", 137  # runtime OOM kill
    await stack.deployer.check_liveness()
    v = stack.deployer.view(d["id"])
    assert v["state"] == "failed" and v["exit_code"] == 137 and "out of memory" in v["last_error"]
    assert v["container_id"] is None
    await asyncio.sleep(0)
    assert (d["id"], "ready", "failed") in seen  # LiteLLM sync deregisters via this hook
    assert any(r["oom"] for r in cal.summary("vllm", "Qwen/Qwen3-Tiny"))


async def test_missing_container_of_sleeping_deployment_fails(stack):
    d = await ready(stack, "a")
    await stack.deployer.sleep(d["id"])
    stack.sdk.container(d["container_id"]).remove(force=True)
    await stack.deployer.check_liveness()
    v = stack.deployer.view(d["id"])
    assert v["state"] == "failed" and v["last_error"] == "container disappeared"


async def test_running_container_untouched_and_errors_tolerated(stack):
    d = await ready(stack, "a")
    await stack.deployer.check_liveness()
    assert stack.deployer.view(d["id"])["state"] == "ready"
    stack.sdk.fail_get_count = 1  # transient Docker API error: skip, do not fail the deployment
    await stack.deployer.check_liveness()
    assert stack.deployer.view(d["id"])["state"] == "ready"


async def test_budget_loop_runs_liveness(stack):
    """The production budget loop wires the liveness check as a budget listener."""
    from dgxmm.context import AppContext, budget_loop
    from dgxmm.events import EventBroker
    d = await ready(stack, "a")
    c = stack.sdk.container(d["container_id"])
    c.status, c.exit_code = "exited", 1
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, budget=stack.budget,
                     deployer=stack.deployer)
    from dgxmm.context import liveness_listener
    ctx.budget_listeners.append(liveness_listener(stack.deployer))
    task = asyncio.create_task(budget_loop(ctx))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if stack.deployer.view(d["id"])["state"] == "failed":
            break
    task.cancel()
    assert stack.deployer.view(d["id"])["state"] == "failed"
