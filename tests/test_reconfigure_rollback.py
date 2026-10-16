"""Test reconfigure rollback when stop fails."""
import pytest

from dgxmm.deployer.service import DeployError, DeployRequest

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name, ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg}, alias=alias)


async def ready(stack, name, ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req(name, ref=ref, alias=alias, **cfg), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return d


async def test_reconfigure_rollback_when_stop_fails(stack):
    """When rollback stop() raises, old config is still restored and undo_actions runs."""
    d = await ready(stack, "a")
    old_config = dict(d["config"])

    # Inject failure into stop during rollback
    original_stop = stack.deployer.stop
    stop_calls = []

    async def stop_that_tracks_and_fails(dep_id, user_id=None, reason="user"):
        stop_calls.append(reason)
        if reason == "reconfigure rollback":
            raise DeployError("test_error", "stop failed in rollback", 502)
        return await original_stop(dep_id, user_id, reason)

    stack.deployer.stop = stop_that_tracks_and_fails

    # Make wait_ready timeout so rollback tries to stop a running container
    original_wait = stack.deployer.wait_ready

    async def wait_that_never_finishes(dep_id, timeout):
        # Return "starting" so reconfigure fails and rollback is triggered
        return "starting"

    stack.deployer.wait_ready = wait_that_never_finishes

    # Reconfigure should fail and trigger rollback with stop
    new_config = {**SMALL, "max_model_len": 8192}
    with pytest.raises(DeployError):
        await stack.deployer.reconfigure(d["id"], new_config)

    # Even though rollback stop failed, config should be restored
    restored = stack.deployer.view(d["id"])
    assert restored["config"] == old_config
    # Verify stop was attempted during rollback
    assert "reconfigure rollback" in stop_calls


async def test_reconfigure_inactive_restores_reserved_estimate_port(stack):
    """When reconfigure fails on inactive deployment, reserved_bytes/estimate/port restored."""
    # Create stopped deployment
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req("a"), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    await stack.deployer.stop(d["id"])

    stopped = stack.deployer.view(d["id"])
    old_config = dict(stopped["config"])
    old_reserved = stopped["reserved_bytes"]
    old_estimate = dict(stopped["estimate"])
    old_port = stopped["port"]

    # Make restart fail to trigger rollback
    stack.sdk.exit_on_start["dgxmm-a"] = 1

    # Reconfigure should fail and rollback
    new_config = {**SMALL, "max_model_len": 8192}
    with pytest.raises(DeployError):
        await stack.deployer.reconfigure(d["id"], new_config)

    # Config AND estimate/reserved/port should be restored (was_active=False)
    restored = stack.deployer.view(d["id"])
    assert restored["config"] == old_config
    assert restored["reserved_bytes"] == old_reserved
    assert restored["estimate"] == old_estimate
    assert restored["port"] == old_port
