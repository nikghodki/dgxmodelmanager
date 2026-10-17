"""H3: the decider API boundary — access matrix (anon / viewer / admin x every route), rate limits, request size
limits, audit coverage of every admin action, and bearer-key rotation (API + admin CLI)."""
import json

import pytest
from sqlalchemy import select

from dgxmm.db import AuditLog
from tests.test_api_decider import B, api, secret, viewer  # noqa: F401  (fixture)


def anon(c):
    c.cookies.clear()
    return {}


def admin_login(c):
    r = c.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def routes(dep_id):
    """(method, path, json body, viewer allowed) for every decider-related route."""
    return [
        ("GET", f"{B}/status", None, True),
        ("GET", f"{B}/image", None, True),
        ("GET", f"/dgxmm/api/deployments/{dep_id}/diagnosis", None, True),
        ("GET", f"/dgxmm/api/deployments/{dep_id}/workload", None, True),
        ("PUT", f"{B}/settings", {"features": {"plan": {"threshold": 0.7}}}, False),
        ("POST", f"{B}/enable", {}, False),
        ("POST", f"{B}/disable", {}, False),
        ("POST", f"{B}/image/build", None, False),
        ("GET", f"{B}/decisions", None, False),
        ("POST", f"{B}/decisions/1/feedback", {"accepted": True}, False),
        ("POST", f"{B}/suggest-config", {"model_ref": "Qwen/Qwen3-Tiny", "engine": "vllm"}, False),
        ("POST", f"{B}/rotate-key", None, False),
        ("POST", "/dgxmm/api/deployments/preview", {"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                                     "config": {"max_model_len": 4096}}, False),
    ]


def _tiny(api):
    from tests.test_api_deployments import SMALL
    r = api.post("/dgxmm/api/deployments", headers=api.h, json={"name": "tiny", "engine": "vllm",
                                                                 "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code in (200, 201, 202), r.text
    return r.json()["id"]


def test_access_matrix_every_route(api):
    dep_id = _tiny(api)
    for method, path, body, viewer_ok in routes(dep_id):
        h = anon(api)
        r = api.request(method, path, headers=h, json=body)
        assert r.status_code == 401, (method, path, "anon", r.status_code)
        vh = viewer(api)
        r = api.request(method, path, headers=vh, json=body)
        if viewer_ok:
            assert r.status_code == 200, (method, path, "viewer", r.status_code, r.text)
        else:
            assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required", (method, path, r.text)
        ah = admin_login(api)
        r = api.request(method, path, headers=ah, json=body)
        assert r.status_code not in (401, 403, 429), (method, path, "admin", r.status_code, r.text)
        # a state-changing request without the CSRF header is refused even for an admin
        if method != "GET":
            r = api.request(method, path, json=body)
            assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf_failed", (method, path)


def test_every_admin_action_is_audited_without_secrets(api):
    h = api.h
    assert api.put(f"{B}/settings", headers=h, json={"features": {"kind": {"enabled": False}}}).status_code == 200
    r = api.post(f"{B}/enable", headers=h, json={})
    assert r.status_code == 202, r.text
    with api.ctx.db.session() as s:
        from dgxmm.db import DecisionRecord
        s.add(DecisionRecord(feature="plan", subject="preview:x", source="model", choice="kv", confidence=0.9))
    did = api.get(f"{B}/decisions", headers=h).json()[0]["id"]
    assert api.post(f"{B}/decisions/{did}/feedback", headers=h, json={"accepted": False}).status_code == 200
    assert api.post(f"{B}/rotate-key", headers=h).status_code == 200
    assert api.post(f"{B}/disable", headers=h, json={}).status_code == 200
    assert api.post(f"{B}/image/build", headers=h).status_code == 202
    key = secret(api)
    with api.ctx.db.session() as s:
        rows = list(s.scalars(select(AuditLog)))
    actions = {r.action for r in rows}
    for a in ("decider_settings", "decider_enable", "decider_feedback", "decider_rotate_key", "decider_disable",
              "decider_image_build"):
        assert a in actions, a
    assert all(key not in json.dumps(r.detail) and key not in r.target for r in rows)


def test_failed_enable_is_audited(api):
    api.stack.probe.available = 1  # nothing fits
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 409
    with api.ctx.db.session() as s:
        row = s.scalar(select(AuditLog).where(AuditLog.action == "decider_enable_failed"))
    assert row is not None and row.detail["code"] == "insufficient_memory"


# ---------- rate limits ----------
def test_suggest_config_is_rate_limited_per_user(api):
    from dgxmm.api.decider import RATE_LIMITS
    n = RATE_LIMITS["decider_suggest"][0]
    body = {"model_ref": "Qwen/Qwen3-Tiny", "engine": "vllm"}
    codes = [api.post(f"{B}/suggest-config", headers=api.h, json=body).status_code for _ in range(n + 1)]
    assert codes[:n] == [200] * n and codes[n] == 429
    r = api.post(f"{B}/suggest-config", headers=api.h, json=body)
    assert r.json()["detail"]["code"] == "rate_limited" and r.headers.get("retry-after")


def test_admin_mutations_share_a_tight_limit(api):
    from dgxmm.api.decider import RATE_LIMITS
    n = RATE_LIMITS["decider_admin"][0]
    body = {"features": {"plan": {"threshold": 0.7}}}
    codes = [api.put(f"{B}/settings", headers=api.h, json=body).status_code for _ in range(n)]
    assert codes == [200] * n
    for method, path, b in (("PUT", f"{B}/settings", body), ("POST", f"{B}/enable", {}),
                            ("POST", f"{B}/disable", {}), ("POST", f"{B}/image/build", None),
                            ("POST", f"{B}/rotate-key", None)):
        assert api.request(method, path, headers=api.h, json=b).status_code == 429, path


def test_feedback_is_rate_limited(api):
    from dgxmm.api.decider import RATE_LIMITS
    n = RATE_LIMITS["decider_feedback"][0]
    codes = [api.post(f"{B}/decisions/999/feedback", headers=api.h, json={"accepted": True}).status_code
             for _ in range(n + 1)]
    assert codes[:n] == [404] * n and codes[n] == 429


def test_viewers_get_403_not_429_and_do_not_consume_admin_budget(api):
    vh = viewer(api)
    for _ in range(30):
        assert api.post(f"{B}/rotate-key", headers=vh).status_code == 403
    ah = admin_login(api)
    assert api.post(f"{B}/rotate-key", headers=ah).status_code == 200


# ---------- request size ----------
def test_oversized_bodies_are_refused_at_the_boundary(api):
    big = {"model_ref": "Qwen/Qwen3-Tiny", "engine": "vllm", "pad": "x" * (70 * 1024)}
    r = api.post(f"{B}/suggest-config", headers=api.h, json=big)
    assert r.status_code == 413 and r.json()["detail"]["code"] == "request_too_large"
    r = api.post(f"{B}/suggest-config", headers=api.h, json={"model_ref": "x" * 1025, "engine": "vllm"})
    assert r.status_code == 422
    r = api.post(f"{B}/suggest-config", headers=api.h, json={"model_ref": "Qwen/Qwen3-Tiny", "engine": "vllm",
                                                               "extra": 1})
    assert r.status_code == 422  # unknown fields are refused
    acts = [{"type": "stop", "deployment_id": i} for i in range(40)]
    assert api.post(f"{B}/enable", headers=api.h, json={"actions": acts}).status_code == 422
    feats = {f"f{i}": {"enabled": True} for i in range(40)}
    assert api.put(f"{B}/settings", headers=api.h, json={"features": feats}).status_code == 422


def test_chunked_oversized_body_is_refused(api):
    def gen():
        for _ in range(80):
            yield b"x" * 1024
    r = api.post(f"{B}/suggest-config", headers={**api.h, "content-type": "application/json"}, content=gen())
    assert r.status_code == 413


# ---------- key rotation ----------
def test_rotate_key_without_a_decider_just_rotates(api):
    from dgxmm.decider.secret import ensure_key
    old = ensure_key(api.ctx.settings)
    r = api.post(f"{B}/rotate-key", headers=api.h)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["rotated"] is True and out["restarted"] is False
    new = secret(api)
    assert new and new != old and old not in r.text and new not in r.text


def test_rotate_key_restarts_a_running_decider_with_the_new_key(api):
    dep_id = api.post(f"{B}/enable", headers=api.h, json={}).json()["deployment_id"]
    sdk = api.stack.sdk
    old_cid = api.get(f"/dgxmm/api/deployments/{dep_id}").json()["container_id"]
    old = secret(api)
    r = api.post(f"{B}/rotate-key", headers=api.h)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["restarted"] is True and out["deployment_id"] == dep_id
    view = api.get(f"/dgxmm/api/deployments/{dep_id}").json()
    new = secret(api)
    assert new != old and view["container_id"] != old_cid and view["state"] in ("starting", "loading", "ready")
    # the new container got the new key as its secret file, the old key is gone everywhere
    assert sdk.secret_writes[-1] == (view["container_id"], "/run/dgxmm-decider/api_key", new.encode())
    assert old not in str(sdk.container(view["container_id"]).attrs)
    assert api.get(f"{B}/status").json()["enabled"] is True


def test_rotation_request_file_is_processed_by_the_app(api):
    from dgxmm.decider import rotation
    old = secret(api) or ""
    nonce = rotation.request_rotation(api.ctx.settings)
    assert rotation.request_path(api.ctx.settings).exists()
    api.portal.call(rotation.process_pending, api.ctx)  # what the app's background watcher runs
    result = rotation.read_result(api.ctx.settings, nonce)
    assert result is not None and result["ok"] is True and result["nonce"] == nonce
    assert not rotation.request_path(api.ctx.settings).exists()
    assert secret(api) != old
    with api.ctx.db.session() as s:
        row = s.scalar(select(AuditLog).where(AuditLog.action == "decider_rotate_key"))
    assert row is not None and row.detail["via"] == "cli"


def test_cli_rotate_decider_key_waits_for_the_app(tmp_path, settings, db, monkeypatch):
    import yaml

    from dgxmm import admin
    from dgxmm.decider import rotation
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": settings.data_dir}))
    seen = {}

    def fake_wait(s, nonce, timeout):
        seen["nonce"] = nonce
        assert rotation.request_path(s).read_text().strip() == nonce
        return {"nonce": nonce, "status": "done", "ok": True, "restarted": True, "state": "starting", "error": None}

    monkeypatch.setattr(rotation, "wait_result", fake_wait)
    assert admin.main(["--config", str(cfg), "rotate-decider-key", "--wait", "5"]) == 0
    assert seen["nonce"]

    monkeypatch.setattr(rotation, "wait_result", lambda s, n, t: None)
    assert admin.main(["--config", str(cfg), "rotate-decider-key", "--wait", "1"]) == 2  # app did not pick it up


def test_rotated_key_file_is_private_and_atomic(settings):
    import os
    import stat

    from dgxmm.decider.secret import ensure_key, rotate_key
    a = ensure_key(settings)
    rotate_key(settings)
    p = settings.decider_key_path
    assert p.read_text() != a and stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert not [x for x in p.parent.iterdir() if x.name.endswith(".tmp")]


@pytest.mark.parametrize("bad", ["../x", "a b"])
def test_feature_names_are_validated(api, bad):
    r = api.put(f"{B}/settings", headers=api.h, json={"features": {bad: {"enabled": True}}})
    assert r.status_code == 422


# ---------- fix round 1: rotation vs enable/restart, CLI handshake ----------
def test_restart_and_enable_are_refused_while_a_rotation_runs(api):
    from dgxmm.decider import rotation
    from dgxmm.deployer.service import DeployError
    dep_id = api.post(f"{B}/enable", headers=api.h, json={}).json()["deployment_id"]
    deployer = api.ctx.deployer
    seen = {}
    real_stop = deployer.stop

    async def stop_then_race(dep, *a, **kw):
        out = await real_stop(dep, *a, **kw)
        try:  # someone presses Start while the key is being rotated
            await deployer.restart(dep)
        except DeployError as e:
            seen["restart"] = (e.code, e.status)
        seen["in_progress"] = rotation.in_progress(api.ctx)
        return out

    deployer.stop = stop_then_race
    r = api.post(f"{B}/rotate-key", headers=api.h)
    deployer.stop = real_stop
    assert r.status_code == 200 and r.json()["restarted"] is True, r.text
    assert seen == {"restart": ("rotation_in_progress", 409), "in_progress": True}
    assert dep_id not in deployer.rotating and not rotation.in_progress(api.ctx)


def test_enable_answers_409_during_a_rotation(api):
    from dgxmm.decider import rotation
    lock = api.portal.call(rotation._lock_of, api.ctx)
    api.portal.call(lock.acquire)
    try:
        r = api.post(f"{B}/enable", headers=api.h, json={})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "rotation_in_progress"
    finally:
        api.portal.call(_release, lock)


async def _release(lock):
    lock.release()


def test_cli_request_is_not_duplicated_and_reports_in_progress(api):
    from dgxmm.decider import rotation
    s = api.ctx.settings
    n1 = rotation.request_rotation(s)
    assert rotation.request_rotation(s) == n1  # a pending request is reused, never queued twice
    # the app has picked it up and is rotating: the result says so, a re-run waits on the same rotation
    rotation.request_path(s).unlink()
    rotation._write_private(rotation.result_path(s), __import__("json").dumps(
        {"nonce": n1, "status": "in_progress", "started_at": __import__("time").time()}))
    assert rotation.request_rotation(s) == n1
    assert not rotation.request_path(s).exists()
    res = rotation.wait_result(s, n1, 0.2)
    assert res["status"] == "in_progress"
    # processing writes "in_progress" first and "done" at the end
    n2 = "fresh"
    rotation.result_path(s).unlink()
    rotation._write_private(rotation.request_path(s), n2 + "\n")
    out = api.portal.call(rotation.process_pending, api.ctx)
    assert out["status"] == "done" and out["ok"] is True
    assert rotation.read_result(s, n2)["status"] == "done"


def test_cli_exit_codes(tmp_path, settings, db, monkeypatch, capsys):
    import yaml

    from dgxmm import admin
    from dgxmm.decider import rotation
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": settings.data_dir}))
    waits = []
    monkeypatch.setattr(rotation, "wait_result",
                        lambda s, n, t: waits.append(t) or {"nonce": n, "status": "in_progress"})
    assert admin.main(["--config", str(cfg), "rotate-decider-key"]) == 3
    assert waits == [90.0] and "still in progress" in capsys.readouterr().err


# ---------- fix round 1: upgrade detection ----------
def test_status_flags_a_pre_hardening_decider_and_restart_fixes_it(api):
    dep_id = api.post(f"{B}/enable", headers=api.h, json={}).json()["deployment_id"]
    st = api.get(f"{B}/status").json()
    assert st["upgrade"] == {"needed": False, "reasons": []}
    # what a decider started by the POC build looks like
    cid = api.get(f"/dgxmm/api/deployments/{dep_id}").json()["container_id"]
    c = api.stack.sdk.container(cid)
    old_key = "old-env-key-value-1234567890"
    c.image_name, c.user, c.read_only = "dgxmm-decider:poc", "", False
    c.env = {**c.env, "DECISION_API_KEY": old_key}
    c.ports, c.networks = [(8101, "127.0.0.1")], ["gateway"]
    r = api.get(f"{B}/status")
    up = r.json()["upgrade"]
    assert up["needed"] is True and old_key not in r.text
    text = " ".join(up["reasons"])
    for needle in ("dgxmm-decider:poc", "environment", "root", "publishes", "internal network", "writable"):
        assert needle in text, (needle, text)
    # viewers see the warning too (the status is viewer-readable), without details that leak secrets
    viewer(api)
    assert api.get(f"{B}/status").json()["upgrade"]["needed"] is True
    h = admin_login(api)
    # the fix: restart with the current image and spec (the rotate action does stop -> new key -> start)
    assert api.post(f"{B}/rotate-key", headers=h).json()["restarted"] is True
    assert api.get(f"{B}/status").json()["upgrade"] == {"needed": False, "reasons": []}


def test_status_upgrade_is_quiet_without_a_decider(api):
    assert api.get(f"{B}/status").json()["upgrade"] == {"needed": False, "reasons": []}
