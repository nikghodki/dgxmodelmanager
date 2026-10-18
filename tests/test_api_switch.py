from tests.test_api_deployments import SMALL, admin


def deploy(c, h, name, **extra):
    c.stack.http.all_healthy = True
    r = c.post("/dgxmm/api/deployments", headers=h,
               json={"name": name, "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def wait(c, dep_id, want):
    import time
    for _ in range(200):
        if c.get(f"/dgxmm/api/deployments/{dep_id}").json()["state"] == want:
            return True
        time.sleep(0.02)
    return False


def test_sleep_wake_switch_reconfigure_audit(api_stack):
    h = admin(api_stack)
    a = deploy(api_stack, h, "a", alias="chat")
    assert wait(api_stack, a["id"], "ready")
    assert api_stack.post(f"/dgxmm/api/deployments/{a['id']}/sleep", headers=h).json()["state"] == "sleeping"
    assert api_stack.post(f"/dgxmm/api/deployments/{a['id']}/wake", headers=h).json()["state"] == "ready"
    r = api_stack.post(f"/dgxmm/api/deployments/{a['id']}/switch", headers=h,
                       json={"request": {"name": "b", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL},
                             "actions": []})
    assert r.status_code == 200, r.text
    assert r.json()["alias"] == "chat"
    r = api_stack.post(f"/dgxmm/api/deployments/{r.json()['id']}/reconfigure", headers=h,
                       json={"config": {**SMALL, "max_model_len": 2048}, "actions": []})
    assert r.status_code == 200 and r.json()["config"]["max_model_len"] == 2048
    actions = [e["action"] for e in api_stack.get("/dgxmm/api/audit").json()]
    assert {"deploy", "sleep", "wake", "switch", "reconfigure", "login"} <= set(actions)


def test_adopt_routes(api_stack):
    h = admin(api_stack)
    api_stack.stack.sdk.add_unmanaged("vllm-external", command=["--model", "Qwen/Qwen3-Tiny", "--max-model-len", "4096"])
    rows = api_stack.get("/dgxmm/api/containers/adoptable").json()
    assert [r["name"] for r in rows] == ["vllm-external"]
    r = api_stack.post("/dgxmm/api/containers/vllm-external/adopt", headers=h, json={"name": "qwen-tiny"})
    assert r.status_code == 201 and r.json()["state"] == "stopped"


def test_audit_requires_admin(api_stack):
    admin(api_stack)
    api_stack.post("/dgxmm/api/auth/logout", headers={"X-CSRF-Token": api_stack.get("/dgxmm/api/auth/me").json()["csrf_token"]})
    api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert api_stack.get("/dgxmm/api/audit").status_code == 403
