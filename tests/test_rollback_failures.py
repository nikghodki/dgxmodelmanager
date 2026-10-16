"""Test rollback failures are reported and audited."""
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


async def test_switch_rollback_incomplete_when_wake_fails(stack):
    """When undo_actions wake fails, switch raises rollback_incomplete with failure list."""
    a = await ready(stack, "a", enable_sleep_mode=True, alias="chat")

    # Inject wake failure
    original_wake = stack.deployer.wake

    async def wake_that_fails(dep_id, user_id=None):
        raise DeployError("test_error", "wake failed", 500)

    stack.deployer.wake = wake_that_fails

    # Make wait_ready timeout so switch fails and triggers rollback
    original_wait = stack.deployer.wait_ready

    async def wait_that_never_finishes(dep_id, timeout):
        return "starting"

    stack.deployer.wait_ready = wait_that_never_finishes

    # Switch should fail with rollback_incomplete
    with pytest.raises(DeployError) as exc:
        await stack.deployer.switch(a["id"], req("b"), [], 1)

    assert exc.value.code == "rollback_incomplete"
    assert "wake failed" in exc.value.message or "sleep deployment" in exc.value.message


async def test_switch_api_audits_failure(api_stack, make_user):
    """API audits switch_failed with error code."""
    a = await ready(api_stack.stack, "a", alias="chat")

    # Make wait_ready timeout
    async def wait_that_never_finishes(dep_id, timeout):
        return "starting"

    api_stack.stack.deployer.wait_ready = wait_that_never_finishes

    # Login
    make_user("admin", "admin123", "admin")
    resp = api_stack.post("/dgxmm/api/auth/login", json={"username": "admin", "password": "admin123"})
    csrf = resp.json()["csrf_token"]

    # Switch should fail
    resp = api_stack.post(f"/dgxmm/api/deployments/{a['id']}/switch", json={
        "request": {"engine": "vllm", "name": "b", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL},
        "actions": []
    }, headers={"X-CSRF-Token": csrf})
    assert resp.status_code in (500, 502)

    # Check audit log for switch_failed
    resp = api_stack.get("/dgxmm/api/audit?limit=5", headers={"X-CSRF-Token": csrf})
    assert resp.status_code == 200
    audit_entries = resp.json()
    switch_failed = [e for e in audit_entries if e["action"] == "switch_failed"]
    assert len(switch_failed) == 1
    assert "code" in switch_failed[0]["detail"]
