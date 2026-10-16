"""HTTP API for weight deletion and incomplete cleanup."""
from pathlib import Path

from sqlalchemy import select

from dgxmm.db import AuditLog, Deployment, Download
from tests.test_api_deployments import admin

API = "/dgxmm/api/inventory"


def hub(c) -> Path:
    return Path(c.ctx.settings.hf_cache.local) / "hub"


def viewer(c):
    r = c.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def audits(c, action):
    with c.ctx.db.session() as s:
        return [(a.target, a.detail) for a in s.scalars(select(AuditLog).where(AuditLog.action == action))]


def add_dep(c, name, ref, state):
    with c.ctx.db.session() as s:
        s.add(Deployment(name=name, engine="vllm", model_ref=ref, state=state, config={}))


def test_preview_and_delete_hf_model(api_stack):
    h = admin(api_stack)
    p = api_stack.get(f"{API}/delete-preview", params={"ref": "Qwen/Qwen3-Tiny"})
    assert p.status_code == 200, p.text
    p = p.json()
    assert p["paths"] == ["/host/hf/hub/models--Qwen--Qwen3-Tiny"] and p["bytes"] > 0 and not p["blocked"]
    r = api_stack.delete(API, params={"ref": "Qwen/Qwen3-Tiny"}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["freed_bytes"] == p["bytes"]
    assert not (hub(api_stack) / "models--Qwen--Qwen3-Tiny").exists()
    refs = [a["ref"] for a in api_stack.get(API).json()]
    assert "Qwen/Qwen3-Tiny" not in refs and "org/big-model" in refs
    [(target, detail)] = audits(api_stack, "model_delete")
    assert target == "Qwen/Qwen3-Tiny" and detail["bytes"] == p["bytes"]


def test_delete_accepts_ref_in_body(api_stack):
    h = admin(api_stack)
    r = api_stack.request("DELETE", API, json={"ref": "Qwen/Qwen3-Tiny"}, headers=h)
    assert r.status_code == 200, r.text


def test_delete_in_use_is_409_with_blocking_deployments(api_stack):
    h = admin(api_stack)
    add_dep(api_stack, "tiny", "Qwen/Qwen3-Tiny", "ready")
    p = api_stack.get(f"{API}/delete-preview", params={"ref": "Qwen/Qwen3-Tiny"}).json()
    assert p["blocked"] and p["deployments_blocking"][0]["name"] == "tiny"
    r = api_stack.delete(API, params={"ref": "Qwen/Qwen3-Tiny"}, headers=h)
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "model_in_use" and d["deployments"][0]["name"] == "tiny"
    assert (hub(api_stack) / "models--Qwen--Qwen3-Tiny").is_dir()
    assert audits(api_stack, "model_delete") == []


def test_delete_with_active_download_is_409(api_stack):
    h = admin(api_stack)
    with api_stack.ctx.db.session() as s:
        s.add(Download(repo_id="org/partial", state="running", files=[]))
    r = api_stack.delete(API, params={"ref": "org/partial"}, headers=h)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "download_active"


def test_delete_unknown_ref_404_and_missing_ref_422(api_stack):
    h = admin(api_stack)
    assert api_stack.delete(API, params={"ref": "nope/nope"}, headers=h).status_code == 404
    assert api_stack.delete(API, headers=h).status_code == 422
    assert api_stack.get(f"{API}/delete-preview", params={"ref": "nope/nope"}).status_code == 404


def test_viewers_and_csrf_less_requests_cannot_delete(api_stack):
    admin(api_stack)
    v = viewer(api_stack)
    assert api_stack.get(f"{API}/delete-preview", params={"ref": "Qwen/Qwen3-Tiny"}).status_code == 403
    assert api_stack.delete(API, params={"ref": "Qwen/Qwen3-Tiny"}, headers=v).status_code == 403
    assert api_stack.get(f"{API}/cleanup-incomplete/preview").status_code == 403
    assert api_stack.post(f"{API}/cleanup-incomplete", json={}, headers=v).status_code == 403
    assert (hub(api_stack) / "models--Qwen--Qwen3-Tiny").is_dir()


def test_delete_without_csrf_is_refused(api_stack):
    admin(api_stack)
    assert api_stack.delete(API, params={"ref": "Qwen/Qwen3-Tiny"}).status_code == 403
    assert (hub(api_stack) / "models--Qwen--Qwen3-Tiny").is_dir()


def age(c):
    import os
    import time
    t = time.time() - 3600
    for b in hub(c).glob("models--*/blobs/*.incomplete"):
        os.utime(b, (t, t))


def test_cleanup_incomplete_preview_and_run(api_stack):
    h = admin(api_stack)
    age(api_stack)
    p = api_stack.get(f"{API}/cleanup-incomplete/preview").json()
    assert [i["ref"] for i in p["items"]] == ["org/partial"] and p["count"] == 1 and p["bytes"] > 0
    r = api_stack.post(f"{API}/cleanup-incomplete", json={}, headers=h)
    assert r.status_code == 200, r.text
    out = r.json()
    assert [d["ref"] for d in out["deleted"]] == ["org/partial"] and out["freed_bytes"] == p["bytes"]
    assert not (hub(api_stack) / "models--org--partial").exists()
    assert (hub(api_stack) / "models--Qwen--Qwen3-Tiny").is_dir()
    [(_, detail)] = audits(api_stack, "incomplete_cleanup")
    assert detail["deleted"] == ["org/partial"] and detail["bytes"] == p["bytes"]
    assert [t for t, _ in audits(api_stack, "model_delete")] == ["org/partial"]


def test_cleanup_with_refs_reports_skips(api_stack):
    h = admin(api_stack)
    age(api_stack)
    add_dep(api_stack, "p", "org/partial", "loading")
    out = api_stack.post(f"{API}/cleanup-incomplete", json={"refs": ["org/partial", "Qwen/Qwen3-Tiny"]},
                         headers=h).json()
    assert out["deleted"] == []
    assert {s["ref"]: s["code"] for s in out["skipped"]} == {"org/partial": "model_in_use",
                                                             "Qwen/Qwen3-Tiny": "not_incomplete"}
    assert (hub(api_stack) / "models--org--partial").is_dir()


def test_refused_delete_is_audited(api_stack):
    h = admin(api_stack)
    add_dep(api_stack, "tiny", "Qwen/Qwen3-Tiny", "ready")
    assert api_stack.delete(API, params={"ref": "Qwen/Qwen3-Tiny"}, headers=h).status_code == 409
    [(target, detail)] = audits(api_stack, "model_delete_refused")
    assert target == "Qwen/Qwen3-Tiny" and detail["reason"] == "model_in_use"


def test_api_wires_the_being_deleted_guard_into_deployer(api_stack):
    admin(api_stack)
    api_stack.get(f"{API}/delete-preview", params={"ref": "Qwen/Qwen3-Tiny"})
    assert api_stack.ctx.deployer.delete_guard == api_stack.ctx.weights.is_being_deleted
