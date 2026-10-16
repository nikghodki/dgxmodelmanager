from dgxmm.config import GIB

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def admin(c, make_user=None):
    from dgxmm.auth.passwords import hash_password
    from dgxmm.db import User
    with c.ctx.db.session() as s:
        s.add(User(username="root", pw_hash=hash_password("root-password-123"), role="admin"))
        s.add(User(username="vic", pw_hash=hash_password("viewer-password-1"), role="viewer"))
    r = c.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def test_inventory_host_engines(api_stack):
    h = admin(api_stack)
    inv = api_stack.get("/dgxmm/api/inventory").json()
    tiny = next(a for a in inv if a["ref"] == "Qwen/Qwen3-Tiny")
    assert tiny["engines"] == ["vllm"] and tiny["complete"]
    host = api_stack.get("/dgxmm/api/host").json()
    assert host["budget"]["total"] == 121 * GIB and host["alert"] == {"level": "ok"}
    eng = api_stack.get("/dgxmm/api/engines").json()
    assert eng[0]["name"] == "vllm" and eng[0]["defaults"]["max_model_len"] == 32768


def test_preview_deploy_stop_remove(api_stack):
    h = admin(api_stack)
    p = api_stack.post("/dgxmm/api/deployments/preview", headers=h,
                       json={"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    assert p["fit"]["fits"]
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 201, r.text
    dep = r.json()
    assert api_stack.get("/dgxmm/api/deployments").json()[0]["name"] == "tiny"
    assert api_stack.get(f"/dgxmm/api/deployments/{dep['id']}").json()["events"]
    assert api_stack.post(f"/dgxmm/api/deployments/{dep['id']}/stop", headers=h).json()["state"] == "stopped"
    assert api_stack.delete(f"/dgxmm/api/deployments/{dep['id']}", headers=h).status_code == 204


def test_insufficient_memory_is_409_with_fit(api_stack):
    h = admin(api_stack)
    api_stack.stack.probe.available = 10 * GIB
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "big", "engine": "vllm", "model_ref": "org/big-model", "config": SMALL})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "insufficient_memory" and d["fit"]["required"] > d["fit"]["free_for_new"]


def test_hostile_inputs_rejected_before_docker(api_stack):
    h = admin(api_stack)
    cases = [
        {"name": "x;rm", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL},
        {"name": "x", "engine": "vllm", "model_ref": "../../etc/passwd", "config": SMALL},
        {"name": "x", "engine": "vllm", "model_ref": "/etc/passwd.gguf", "config": SMALL},
        {"name": "x", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {**SMALL, "extra_args": ["--model", "evil"]}},
        {"name": "x", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {**SMALL, "gpu_memory_utilization": 0.99}},
    ]
    for body in cases:
        r = api_stack.post("/dgxmm/api/deployments", headers=h, json=body)
        assert r.status_code == 422, (body, r.text)
    assert api_stack.stack.sdk.run_calls == []


def test_viewer_cannot_deploy(api_stack):
    admin(api_stack)
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    h = {"X-CSRF-Token": r.json()["csrf_token"]}
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 403
    assert api_stack.post("/dgxmm/api/deployments/preview", headers=h,
                          json={"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).status_code == 200


def test_unknown_api_route_is_json_404(api_stack):
    admin(api_stack)
    r = api_stack.get("/dgxmm/api/nope")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found"
