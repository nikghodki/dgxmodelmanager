"""Tests for reconcile and deployer recovery edge cases."""
import asyncio

import docker.errors
import pytest
from sqlalchemy import select

from dgxmm.db import Deployment
from dgxmm.deployer.reconcile import reconcile
from dgxmm.deployer.service import DeployError, DeployRequest

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def deploy(stack, name):
    return await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL,
                                                     alias=name), 1)  # distinct LiteLLM model names (same model twice)


async def test_reconcile_handles_interrupted_stop_and_continues(stack):
    """Defect 1: reconcile must handle a row in 'stopping' state and continue processing later rows."""
    stack.settings.max_concurrent_startups = 0  # two containers at once (the start-up gate is tested elsewhere)
    a = await deploy(stack, "a")
    b = await deploy(stack, "b")
    await stack.deployer.aclose()

    # Simulate interrupted stop: manually set 'a' to stopping state
    with stack.db.session() as s:
        row = s.get(Deployment, a["id"])
        row.state = "stopping"

    # Reconcile should handle this and still process 'b'
    notes = await reconcile(stack.deployer)

    # 'a' should be stopped, 'b' should be resumed
    assert any("finishing interrupted stop" in n and "a" in n for n in notes)
    assert any("resumed" in n and "b" in n for n in notes)
    assert stack.deployer.view(a["id"])["state"] == "stopped"


async def test_reconcile_continues_after_row_error(stack):
    """Defect 1: if one row raises, reconcile should log and continue with remaining rows and orphans."""
    a = await deploy(stack, "a")
    await stack.deployer.aclose()

    # Break container 'a' to cause an error
    stack.sdk._containers.pop(a["container_id"])
    with stack.db.session() as s:
        row = s.get(Deployment, a["id"])
        row.state = "stopping"  # Will try to stop but container missing

    # Add an orphan that should still be adopted
    from dgxmm.engines.base import managed_labels
    stack.sdk.add_unmanaged("dgxmm-orphan", labels=managed_labels(999, "vllm"),
                            command=["--model", "test"], ports=[(8150, "127.0.0.1")])

    notes = await reconcile(stack.deployer)

    # Orphan should still be adopted despite earlier error
    with stack.db.session() as s:
        rows = list(s.scalars(select(Deployment)))
    assert any(r.name == "orphan" for r in rows)


async def test_restart_with_docker_failure_records_error(stack):
    """Defect 2: restart docker.run failure should transition to failed with last_error."""
    dep = await deploy(stack, "a")
    await stack.deployer.stop(dep["id"], 1)

    stack.sdk.fail_run = True
    with pytest.raises(DeployError) as e:
        await stack.deployer.restart(dep["id"], 1)

    assert e.value.code == "docker_error"
    v = stack.deployer.view(dep["id"])
    assert v["state"] == "failed"
    assert v["last_error"] and "docker error" in v["last_error"]


async def test_restart_reallocates_when_port_taken(stack):
    """Defect 3: restart must reallocate port if another deployment now holds it."""
    a = await deploy(stack, "a")
    port_a = a["port"]
    await stack.deployer.stop(a["id"], 1)

    # Deploy 'b' which will get the same port
    b = await deploy(stack, "b")
    assert b["port"] == port_a  # b took a's old port

    # Restart 'a' should get a different port
    a2 = await stack.deployer.restart(a["id"], 1)
    assert a2["port"] != port_a
    assert a2["port"] == port_a + 1  # Next available


async def test_monitor_recovers_from_transient_docker_errors(stack):
    """Defect 4: monitor should recover from transient docker.get errors."""
    dep = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL), 1)

    # Now make monitor get() fail 3 times then succeed
    stack.sdk.fail_get_count = 3

    # Wait a bit for failures
    await asyncio.sleep(0.1)

    # Now let it succeed
    stack.http.healthy.add(dep["port"])
    assert await stack.deployer.wait_ready(dep["id"], 5) == "ready"


async def test_monitor_fails_after_persistent_errors(stack):
    """Defect 4: monitor should fail deployment after 5 consecutive errors."""
    stack.sdk.skip_gets = 1  # Allow remove_by_name to succeed
    stack.sdk.always_fail_get = True

    dep = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL), 1)

    state = await stack.deployer.wait_ready(dep["id"], 5)
    assert state == "failed"
    v = stack.deployer.view(dep["id"])
    assert "monitor error" in v["last_error"]
