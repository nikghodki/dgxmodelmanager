"""/api/decider: status, settings, enable/disable via the normal fit/plan flow, image build, decisions, feedback."""
import time

import pytest
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import AuditLog
from dgxmm.engines.decision import DECIDER_REPO, DECIDER_REVISION
from tests.test_api_deployments import SMALL, admin
from tests.test_decider_engine import D3_CONFIG

B = "/dgxmm/api/decider"


@pytest.fixture
def api(api_stack, fake_hf_repo):
    fake_hf_repo(DECIDER_REPO, rev=DECIDER_REVISION, config=D3_CONFIG,
                 files={"model-00001-of-00002.safetensors": 5 * GIB, "model-00002-of-00002.safetensors": 4 * GIB})
    api_stack.stack.inv.refresh()
    api_stack.h = admin(api_stack)
    return api_stack


def viewer(c):
    r = c.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def secret(c):
    p = c.ctx.settings.decider_key_path
    return p.read_text() if p.exists() else None


def test_status_defaults_viewer_readable(api):
    st = api.get(f"{B}/status").json()
    assert st["enabled"] is False and st["healthy"] is False and st["deployment_id"] is None and st["state"] is None
    assert st["model"] == "vllm-sr/d3-mini" and st["revision"] == DECIDER_REVISION
    assert st["features"]["diagnosis"] == {"enabled": True, "threshold": 0.8}
    assert st["features"]["parser"]["threshold"] == 0.8 and st["features"]["plan"]["threshold"] == 0.6
    assert st["features"]["workload"]["threshold"] == 0.75 and st["features"]["kind"]["threshold"] == 0.8
    assert st["image"] == {"present": True, "building": False, "error": None}
    assert st["decisions_24h"] == 0 and st["p50_ms"] is None and st["accepted_rate"] is None
    viewer(api)
    assert api.get(f"{B}/status").status_code == 200


def test_settings_update_admin_only_validated_and_audited(api):
    r = api.put(f"{B}/settings", headers=api.h, json={"features": {"plan": {"threshold": 0.75}, "kind": {"enabled": False}}})
    assert r.status_code == 200, r.text
    f = r.json()["features"]
    assert f["plan"]["threshold"] == 0.75 and f["kind"]["enabled"] is False
    for bad in ({"features": {"nope": {"enabled": True}}}, {"features": {"plan": {"threshold": 1.5}}},
                {"features": {"plan": {"enabled": "yes"}}}, {"enabled": True}):
        assert api.put(f"{B}/settings", headers=api.h, json=bad).status_code == 422, bad
    with api.ctx.db.session() as s:
        assert s.scalar(select(AuditLog).where(AuditLog.action == "decider_settings")) is not None
    vh = viewer(api)
    assert api.put(f"{B}/settings", headers=vh, json={"features": {"plan": {"threshold": 0.7}}}).status_code == 403


def test_enable_deploys_system_decider_then_disable_stops_it(api):
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 202, r.text
    dep_id = r.json()["deployment_id"]
    view = api.get(f"/dgxmm/api/deployments/{dep_id}").json()
    assert view["system"] and view["engine"] == "decision" and view["model_ref"] == DECIDER_REPO
    st = api.get(f"{B}/status").json()
    assert st["enabled"] and st["deployment_id"] == dep_id and st["state"] in ("starting", "loading", "ready")
    # idempotent: enabling again keeps the same deployment
    assert api.post(f"{B}/enable", headers=api.h, json={}).json()["deployment_id"] == dep_id
    assert len([v for v in api.get("/dgxmm/api/deployments").json() if v["engine"] == "decision"]) == 1
    r = api.post(f"{B}/disable", headers=api.h, json={})
    assert r.status_code == 200, r.text
    assert api.get(f"/dgxmm/api/deployments/{dep_id}").json()["state"] == "stopped"
    assert api.get(f"{B}/status").json()["enabled"] is False
    # enable again restarts the stopped deployment instead of creating another
    assert api.post(f"{B}/enable", headers=api.h, json={}).json()["deployment_id"] == dep_id
    assert api.get(f"/dgxmm/api/deployments/{dep_id}").json()["state"] in ("starting", "loading", "ready")
    r = api.post(f"{B}/disable", headers=api.h, json={"remove": True})
    assert r.status_code == 200 and api.get(f"/dgxmm/api/deployments/{dep_id}").status_code == 404
    assert api.get(f"{B}/status").json()["deployment_id"] is None
    with api.ctx.db.session() as s:
        actions = [a.action for a in s.scalars(select(AuditLog))]
    assert actions.count("decider_enable") == 3 and actions.count("decider_disable") == 2


def test_enable_insufficient_memory_returns_plans_then_accepts_actions(api):
    api.stack.http.all_healthy = True
    r = api.post("/dgxmm/api/deployments", headers=api.h,
                 json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    tiny = r.json()
    from tests.fakes import free_on_stop
    free_on_stop(api.stack.deployer, api.stack.probe)
    reserve = api.ctx.settings.margin_bytes + api.ctx.settings.os_reserve_bytes
    api.stack.probe.available = 13 * GIB + reserve - 1 * GIB
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 409, r.text
    d = r.json()["detail"]
    assert d["code"] == "insufficient_memory" and d["fit"]["required"] == 13 * GIB
    stop = next(p for p in d["plans"] if p["id"] == f"stop:{tiny['id']}")
    assert stop["actions"] == [{"type": "stop", "deployment_id": tiny["id"]}]
    assert api.get(f"{B}/status").json()["enabled"] is False
    r = api.post(f"{B}/enable", headers=api.h, json={"actions": stop["actions"]})
    assert r.status_code == 202, r.text
    assert api.get(f"/dgxmm/api/deployments/{tiny['id']}").json()["state"] == "stopped"


def test_enable_missing_model_is_404(api_stack):
    h = admin(api_stack)
    r = api_stack.post(f"{B}/enable", headers=h, json={})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "model_not_found"


def test_image_missing_then_build_with_progress(api):
    api.stack.sdk.missing_images.add("dgxmm-decider:3")
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "image_missing"
    assert r.json()["detail"]["build_url"] == "/dgxmm/api/decider/image/build"
    assert api.get(f"{B}/image").json()["present"] is False
    q = api.ctx.events.subscribe()
    assert api.post(f"{B}/image/build", headers=api.h).status_code == 202
    deadline = time.time() + 5
    while api.get(f"{B}/image").json()["building"] and time.time() < deadline:
        time.sleep(0.02)
    img = api.get(f"{B}/image").json()
    assert img["present"] is True and img["error"] is None and img["last_build"]["ok"] is True
    [call] = api.stack.sdk.api.build_calls
    assert call["tag"] == "dgxmm-decider:3" and sorted(call["files"]) == sorted(["Dockerfile", "entrypoint.sh", "launch.py", "requirements.lock", f"d3-mini-{DECIDER_REVISION}.sha256",
                                                                         "d3-nano-aeb70d4dd53ed25b855c3b0a66394b8fe265db7f.sha256"])
    assert call["decode"] is True and call["custom_context"] is True
    phases = []
    while not q.empty():
        topic, data = q.get_nowait()
        if topic == "decider":
            phases.append(data["phase"])
    assert phases[0] == "started" and "progress" in phases and phases[-1] == "done"
    assert api.post(f"{B}/enable", headers=api.h, json={}).status_code == 202


def test_image_build_failure_is_reported(api):
    api.stack.sdk.missing_images.add("dgxmm-decider:3")
    api.stack.sdk.api.build_chunks = [{"stream": "Step 1\n"}, {"error": "pip install failed"}]
    assert api.post(f"{B}/image/build", headers=api.h).status_code == 202
    deadline = time.time() + 5
    while api.get(f"{B}/image").json()["building"] and time.time() < deadline:
        time.sleep(0.02)
    img = api.get(f"{B}/image").json()
    assert img["present"] is False and "pip install failed" in img["error"]
    assert api.get(f"{B}/status").json()["image"]["error"] == img["error"]


def test_decisions_list_and_feedback_admin_only(api):
    dec = api.ctx.decider.record_rules(feature="diagnosis", subject="deployment:1", choice="gpu_oom", reason="OOM")
    rows = api.get(f"{B}/decisions", headers=api.h).json()
    assert rows[0]["id"] == dec.decision_id and rows[0]["source"] == "rules" and rows[0]["feedback"] is None
    assert api.get(f"{B}/decisions?feature=kind", headers=api.h).json() == []
    r = api.post(f"{B}/decisions/{dec.decision_id}/feedback", headers=api.h, json={"accepted": False})
    assert r.status_code == 200 and r.json()["feedback"] == "rejected"
    assert api.post(f"{B}/decisions/999/feedback", headers=api.h, json={"accepted": True}).status_code == 404
    assert api.post(f"{B}/decisions/{dec.decision_id}/feedback", headers=api.h, json={"accepted": "y"}).status_code == 422
    assert api.get(f"{B}/status").json()["accepted_rate"] == 0.0
    vh = viewer(api)
    assert api.get(f"{B}/decisions").status_code == 403
    assert api.post(f"{B}/decisions/{dec.decision_id}/feedback", headers=vh, json={"accepted": True}).status_code == 403
    assert api.post(f"{B}/enable", headers=vh, json={}).status_code == 403
    assert api.post(f"{B}/disable", headers=vh, json={}).status_code == 403
    assert api.post(f"{B}/image/build", headers=vh).status_code == 403


def test_secret_never_in_responses_or_audit(api, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    bodies = [api.post(f"{B}/enable", headers=api.h, json={}).text]
    key = secret(api)
    assert key
    dep_id = api.get(f"{B}/status").json()["deployment_id"]
    bodies += [api.get(f"{B}/status").text, api.get(f"/dgxmm/api/deployments/{dep_id}").text,
               api.get("/dgxmm/api/deployments").text, api.get(f"{B}/decisions", headers=api.h).text,
               api.get("/dgxmm/api/engines").text, api.post(f"{B}/disable", headers=api.h, json={}).text]
    for b in bodies:
        assert key not in b
    with api.ctx.db.session() as s:
        assert all(key not in str(a.detail) for a in s.scalars(select(AuditLog)))
    assert key not in caplog.text


def test_docker_image_check_runs_off_the_event_loop(api, monkeypatch):
    import asyncio
    on_loop = []
    orig = api.ctx.decider_image.present

    def present():
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:  # a worker thread: no running loop
            on_loop.append(False)
        return orig()

    monkeypatch.setattr(api.ctx.decider_image, "present", present)
    api.get(f"{B}/status")
    api.get(f"{B}/image")
    api.post(f"{B}/enable", headers=api.h, json={})
    assert on_loop and not any(on_loop)
