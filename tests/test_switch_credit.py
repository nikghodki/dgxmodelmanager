"""Test switch auto-escalates sleep to stop when memory insufficient."""
import pytest

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name, ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg}, alias=alias)


async def ready(stack, name, ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req(name, ref=ref, alias=alias, **cfg), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return d


async def test_switch_escalates_to_stop_when_sleep_insufficient(stack):
    """A is sleep-capable but B needs more memory than sleep frees → switch stops A (not sleeps)."""
    a = await ready(stack, "a", enable_sleep_mode=True, alias="chat")

    # Set managed_actual so budget calculates external correctly
    a_cid = stack.deployer.view(a["id"])["container_id"]
    stack.probe.managed_actual[a_cid] = 3 * GIB

    # Constrain memory: make sleep insufficient but stop sufficient
    # A: reserved ~7 GiB, weights ~3 GiB → sleep frees ~4 GiB, stop frees ~7 GiB
    # B: needs ~7 GiB
    # Set available so: free_for_new + sleep_frees < 7 GiB, but free_for_new + stop_frees >= 7 GiB
    stack.probe.available = 20 * GIB  # Give enough headroom for B after stopping A

    # Switch should auto-escalate to stop (not sleep) A
    b = await stack.deployer.switch(a["id"], req("b", ref="Qwen/Qwen3-Tiny"), [], 1)
    assert b["state"] == "ready" and b["alias"] == "chat"

    old = stack.deployer.view(a["id"])
    assert old["state"] == "stopped"  # not "sleeping"


async def test_switch_raises_insufficient_memory_before_touching_anything(stack):
    """If even stopping A won't free enough for B, switch raises before applying actions."""
    a = await ready(stack, "a", enable_sleep_mode=True)

    # Make memory so constrained that even stopping A won't help
    stack.probe.available = 5 * GIB

    # B needs ~20 GiB, A reserves ~20 GiB → after stopping A we have 25 GiB
    # But margin is 8 GiB → we need 28 GiB → still short
    # Actually let's make it more extreme to ensure failure
    # Set available to 1 GiB, so after stopping A (frees 20) we have 21 GiB
    # But B needs 20 + 8 margin + 4 OS = 32 GiB
    stack.probe.available = 1 * GIB

    with pytest.raises(InsufficientMemory) as exc:
        await stack.deployer.switch(a["id"], req("b", ref="Qwen/Qwen3-Tiny"), [], 1)

    # A should still be ready (nothing was touched)
    assert stack.deployer.view(a["id"])["state"] == "ready"

    # The 409 should include plans
    assert exc.value.fit is not None
