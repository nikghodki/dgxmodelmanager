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
    # GB10: `used` includes A's full reservation. Leave less than B needs even after sleeping A
    # (sleep frees almost nothing for this tiny model), but enough once A is stopped.
    from tests.fakes import free_on_stop
    free_on_stop(stack.deployer, stack.probe)
    stack.probe.available = 15 * GIB

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


async def test_switch_credits_user_chosen_stop_of_third_deployment(stack):
    """I1: an explicit plan action (stop C) must count toward the fit, in addition to the old deployment."""
    from tests.fakes import free_on_stop
    a = await ready(stack, "a", enable_sleep_mode=False, alias="chat")
    c = await ready(stack, "c", enable_sleep_mode=False)
    free_on_stop(stack.deployer, stack.probe)
    need = stack.deployer.prepare("vllm", "Qwen/Qwen3-Tiny", SMALL).estimate.total
    ra, rc = a["reserved_bytes"], c["reserved_bytes"]
    usable = stack.probe.total - stack.settings.margin_bytes - stack.settings.os_reserve_bytes
    # GB10 accounting: both ready reservations are inside `used`; free_for_new = usable - used.
    # Stopping A alone is not enough; stopping A and C is.
    stack.probe.available = stack.probe.total - (usable - (need - ra - rc // 2))
    b = await stack.budget.refresh()
    assert b.free_for_new + ra < need <= b.free_for_new + ra + rc
    out = await stack.deployer.switch(a["id"], req("b"), [{"type": "stop", "deployment_id": c["id"]}], 1)
    assert out["state"] == "ready" and out["alias"] == "chat"
    assert stack.deployer.view(a["id"])["state"] == "stopped"
    assert stack.deployer.view(c["id"])["state"] == "stopped"


async def test_switch_agrees_with_preview_for_sleeping_old(stack):
    """I1: preview credits a sleeping old deployment's measured memory; switch must do the same."""
    from tests.fakes import free_on_stop
    a = await ready(stack, "a", enable_sleep_mode=True)
    await stack.deployer.sleep(a["id"])
    cid = stack.deployer.view(a["id"])["container_id"]
    stack.probe.managed_actual[cid] = 4 * GIB
    free_on_stop(stack.deployer, stack.probe)
    need = stack.deployer.prepare("vllm", "Qwen/Qwen3-Tiny", SMALL).estimate.total
    usable = stack.probe.total - stack.settings.margin_bytes - stack.settings.os_reserve_bytes
    stack.probe.available = stack.probe.total - (usable - (need - 2 * GIB))  # short by 2 GiB until A is stopped
    p = await stack.deployer.preview("vllm", "Qwen/Qwen3-Tiny", SMALL, replacing=a["id"])
    assert p["fit"]["fits"] and not (await stack.budget.refresh()).free_for_new >= need
    out = await stack.deployer.switch(a["id"], req("b"), [], 1)
    assert out["state"] == "ready" and stack.deployer.view(a["id"])["state"] == "stopped"
