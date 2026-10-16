"""Test malformed actions return 422, not 500."""
import pytest

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def test_deploy_with_malformed_actions_422(api_stack, make_user):
    """POST /deployments with malformed actions returns 422, not 500."""
    # Login
    make_user("admin", "admin123", "admin")
    resp = api_stack.post("/dgxmm/api/auth/login", json={"username": "admin", "password": "admin123"})
    csrf = resp.json()["csrf_token"]

    # Deploy with malformed action (unknown type)
    resp = api_stack.post("/dgxmm/api/deployments", json={
        "engine": "vllm",
        "name": "test",
        "model_ref": "Qwen/Qwen3-Tiny",
        "config": SMALL,
        "actions": [{"type": "nuke", "deployment_id": 1}]
    }, headers={"X-CSRF-Token": csrf})

    # Should return 422 (validation error), not 500
    assert resp.status_code == 422


async def test_switch_with_malformed_actions_422(api_stack, make_user):
    """POST /deployments/{id}/switch with malformed actions returns 422."""
    from dgxmm.deployer.service import DeployRequest

    api_stack.stack.http.all_healthy = True

    # Create deployment first
    req = DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL)
    d = await api_stack.stack.deployer.deploy(req, 1)
    await api_stack.stack.deployer.wait_ready(d["id"], 5)

    # Login
    make_user("admin", "admin123", "admin")
    resp = api_stack.post("/dgxmm/api/auth/login", json={"username": "admin", "password": "admin123"})
    csrf = resp.json()["csrf_token"]

    # Switch with malformed action
    resp = api_stack.post(f"/dgxmm/api/deployments/{d['id']}/switch", json={
        "request": {"engine": "vllm", "name": "b", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL},
        "actions": [{"type": "invalid", "deployment_id": 1}]
    }, headers={"X-CSRF-Token": csrf})

    # Should return 422 (validation error), not 500
    assert resp.status_code == 422

    # Original deployment should still be ready
    view = api_stack.ctx.deployer.view(d["id"])
    assert view["state"] == "ready"
