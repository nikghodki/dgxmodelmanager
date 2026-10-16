import pytest

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name, ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg}, alias=alias)


async def ready(stack, name, **kw):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req(name, **kw), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return d


async def test_sleep_and_wake(stack):
    d = await ready(stack, "a")
    assert (await stack.deployer.sleep(d["id"]))["state"] == "sleeping"
    assert ("POST", d["port"], "/sleep") in stack.http.calls
    assert (await stack.deployer.wake(d["id"]))["state"] == "ready"


async def test_wake_refused_when_memory_short(stack):
    d = await ready(stack, "a")
    await stack.deployer.sleep(d["id"])
    stack.probe.available = 9 * GIB
    with pytest.raises(InsufficientMemory):
        await stack.deployer.wake(d["id"])


async def test_sleep_refused_when_disabled(stack):
    d = await ready(stack, "a", enable_sleep_mode=False)
    with pytest.raises(DeployError) as e:
        await stack.deployer.sleep(d["id"])
    assert e.value.code == "sleep_unsupported"


async def test_switch_moves_alias_and_sleeps_old(stack):
    a = await ready(stack, "a", alias="chat")
    b = await stack.deployer.switch(a["id"], req("b"), [], 1)
    assert b["state"] == "ready" and b["alias"] == "chat"
    old = stack.deployer.view(a["id"])
    assert old["state"] == "sleeping" and old["alias"] is None


async def test_switch_failure_rolls_back(stack):
    a = await ready(stack, "a", alias="chat")
    stack.sdk.exit_on_start["dgxmm-b"] = 137
    with pytest.raises(DeployError):
        await stack.deployer.switch(a["id"], req("b"), [], 1)
    old = stack.deployer.view(a["id"])
    assert old["state"] == "ready" and old["alias"] == "chat"
    # I5: the failed new deployment is removed so its name can be reused
    assert "b" not in {v["name"] for v in stack.deployer.list_views()}
    assert not any(c.name == "dgxmm-b" for c in stack.sdk._containers.values())
    stack.sdk.exit_on_start.clear()
    b = await stack.deployer.switch(a["id"], req("b"), [], 1)
    assert b["state"] == "ready" and b["alias"] == "chat"


async def test_reconfigure_success_and_rollback(stack):
    a = await ready(stack, "a")
    out = await stack.deployer.reconfigure(a["id"], {**SMALL, "max_model_len": 2048})
    assert out["id"] == a["id"] and out["state"] == "ready" and out["config"]["max_model_len"] == 2048
    stack.sdk.exit_on_start["dgxmm-a"] = 1
    original = {**out["config"]}
    with pytest.raises(DeployError):
        await stack.deployer.reconfigure(a["id"], {**SMALL, "max_model_len": 1024})
    stack.sdk.exit_on_start.clear()
    # rollback restarts the previous config; the fake container now starts fine
    st = await stack.deployer.wait_ready(a["id"], 5)
    v = stack.deployer.view(a["id"])
    assert v["config"]["max_model_len"] == original["max_model_len"]
    assert st in ("ready", "failed")


async def test_apply_actions_undo_on_partial_failure(stack):
    a = await ready(stack, "a")
    with pytest.raises(DeployError):
        await stack.deployer.apply_actions([{"type": "sleep", "deployment_id": a["id"]},
                                            {"type": "stop", "deployment_id": 999}], 1)
    assert stack.deployer.view(a["id"])["state"] == "ready"


async def test_switch_docker_error_removes_new_row(stack):
    """I5: even when deploy() itself raises after creating the row, rollback removes it."""
    a = await ready(stack, "a", alias="chat")
    stack.sdk.fail_run = True
    with pytest.raises(DeployError):
        await stack.deployer.switch(a["id"], req("b"), [], 1)
    stack.sdk.fail_run = False
    assert "b" not in {v["name"] for v in stack.deployer.list_views()}
    assert stack.deployer.view(a["id"])["state"] == "ready"


async def test_switch_name_in_use_never_removes_existing_row(stack):
    a = await ready(stack, "a")
    other = await ready(stack, "b", alias="b")  # distinct LiteLLM model names (same model twice)
    with pytest.raises(DeployError):
        await stack.deployer.switch(a["id"], req("b"), [], 1)
    assert stack.deployer.view(other["id"])["state"] == "ready"
