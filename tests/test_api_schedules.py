"""Feature 7 — schedule API: GET/POST /api/deployments/{id}/schedules, PUT/DELETE /api/schedules/{id}. Viewers read,
admins write (CSRF, rate limit, body cap, audit); validation of days, HH:MM, IANA zone and the engine's actions."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from dgxmm.db import AuditLog, Deployment

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
B = "/dgxmm/api"


def users(c):
    from dgxmm.auth.passwords import hash_password
    from dgxmm.db import User
    with c.ctx.db.session() as s:
        s.add(User(username="root", pw_hash=hash_password("root-password-123"), role="admin"))
        s.add(User(username="vic", pw_hash=hash_password("viewer-password-1"), role="viewer"))


def login(c, who="root"):
    pw = "root-password-123" if who == "root" else "viewer-password-1"
    r = c.post(f"{B}/auth/login", json={"username": who, "password": pw})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


@pytest.fixture
def api(api_stack):
    users(api_stack)
    h = login(api_stack)
    api_stack.stack.http.all_healthy = True
    r = api_stack.post(f"{B}/deployments", headers=h, json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                                          "config": {**SMALL, "enable_sleep_mode": True}})
    assert r.status_code == 201, r.text
    api_stack.dep = r.json()
    api_stack.h = h
    return api_stack


PAIR = [{"action": "sleep", "days": 31, "at_local": "19:00", "timezone": "Europe/Berlin"},
        {"action": "wake", "days": 31, "at_local": "07:45", "timezone": "Europe/Berlin"}]


def post(api, items, h=None):
    return api.post(f"{B}/deployments/{api.dep['id']}/schedules", headers=api.h if h is None else h,
                    json={"schedules": items})


def test_create_list_update_delete_with_audit(api):
    r = post(api, PAIR)
    assert r.status_code == 201, r.text
    sleep, wake = r.json()["schedules"]
    assert sleep["action"] == "sleep" and sleep["days_label"] == "Mon–Fri" and sleep["next_run_at"]
    lst = api.get(f"{B}/deployments/{api.dep['id']}/schedules").json()
    assert [s["action"] for s in lst["schedules"]] == ["wake", "sleep"]  # by time of day
    assert lst["allowed_actions"] == ["sleep", "wake", "stop", "start"] and lst["timezone"]
    assert lst["paused_all"] is False
    r = api.put(f"{B}/schedules/{sleep['id']}", headers=api.h, json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False and r.json()["next_run_at"] is None
    r = api.put(f"{B}/schedules/{sleep['id']}", headers=api.h, json={"at_local": "20:30", "days": 127, "enabled": True})
    assert r.json()["at_local"] == "20:30" and r.json()["days_label"] == "every day"
    assert api.delete(f"{B}/schedules/{wake['id']}", headers=api.h).status_code == 204
    assert api.delete(f"{B}/schedules/{wake['id']}", headers=api.h).status_code == 404
    with api.ctx.db.session() as s:
        acts = [a.action for a in s.scalars(select(AuditLog).order_by(AuditLog.id))]
    assert acts[-4:] == ["schedule_create", "schedule_update", "schedule_update", "schedule_delete"]


def test_viewer_reads_but_cannot_write(api):
    post(api, PAIR[:1])
    sid = api.get(f"{B}/deployments/{api.dep['id']}/schedules").json()["schedules"][0]["id"]
    assert api.get(f"{B}/deployments/{api.dep['id']}/schedules").json()["schedules"][0]["created_by"] is not None
    v = login(api, "vic")
    r = api.get(f"{B}/deployments/{api.dep['id']}/schedules")
    assert r.status_code == 200
    assert all("created_by" not in s for s in r.json()["schedules"])  # user ids are not shown to viewers
    assert post(api, PAIR[1:], h=v).status_code == 403
    assert api.put(f"{B}/schedules/{sid}", headers=v, json={"enabled": False}).status_code == 403
    assert api.delete(f"{B}/schedules/{sid}", headers=v).status_code == 403


def test_csrf_required(api):
    r = api.post(f"{B}/deployments/{api.dep['id']}/schedules", json={"schedules": PAIR})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf_failed"


@pytest.mark.parametrize("patch,code", [
    ({"days": 0}, 422), ({"days": 128}, 422), ({"days": True}, 422), ({"at_local": "24:00"}, 422),
    ({"at_local": "7:45"}, 422), ({"at_local": "07:45:00"}, 422), ({"timezone": "Mars/Olympus"}, 422),
    ({"timezone": "../../etc/passwd"}, 422), ({"timezone": "localtime"}, 422), ({"action": "remove"}, 422),
    ({"extra": 1}, 422),
])
def test_validation(api, patch, code):
    r = post(api, [{**PAIR[0], **patch}])
    assert r.status_code == code, r.text


def test_stop_start_for_a_deployment_without_sleep_mode(api):
    h = api.h
    r = api.post(f"{B}/deployments", headers=h, json={"name": "plain", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                                    "alias": "plain", "config": {**SMALL, "enable_sleep_mode": False}})
    dep = r.json()
    url = f"{B}/deployments/{dep['id']}/schedules"
    assert api.post(url, headers=h, json={"schedules": [{**PAIR[0], "action": "sleep"}]}).status_code == 422
    r = api.post(url, headers=h, json={"schedules": [{**PAIR[0], "action": "stop"}, {**PAIR[1], "action": "start"}]})
    assert r.status_code == 201
    assert api.get(url).json()["allowed_actions"] == ["stop", "start"]


def test_conflicts_and_system_deployments(api):
    assert post(api, PAIR).status_code == 201
    r = post(api, [{**PAIR[1], "action": "sleep", "days": 1}])
    assert r.status_code == 409 and r.json()["detail"]["code"] == "schedule_conflict"
    r = post(api, [PAIR[0], {**PAIR[0], "action": "wake"}])
    assert r.status_code == 409
    with api.ctx.db.session() as s:
        s.get(Deployment, api.dep["id"]).system = True
    r = post(api, [{**PAIR[0], "at_local": "01:00"}])
    assert r.status_code == 409 and r.json()["detail"]["code"] == "system_deployment"


def test_unknown_deployment_and_list_size(api):
    r = api.post(f"{B}/deployments/999/schedules", headers=api.h, json={"schedules": PAIR})
    assert r.status_code == 404
    assert api.get(f"{B}/deployments/999/schedules").status_code == 404
    assert post(api, []).status_code == 422
    assert post(api, PAIR * 3).status_code == 422  # at most 2 per request


def test_body_cap(api):
    r = api.post(f"{B}/deployments/{api.dep['id']}/schedules", headers={**api.h, "content-type": "application/json"},
                 content=b'{"schedules": [' + b" " * 9000 + b"]}")
    assert r.status_code == 413
    r = api.put(f"{B}/schedules/1", headers={**api.h, "content-type": "application/json"}, content=b"{" + b" " * 9000 + b"}")
    assert r.status_code == 413


def test_rate_limited(api):
    codes = [api.get(f"{B}/deployments/{api.dep['id']}/schedules").status_code for _ in range(70)]
    assert 429 in codes and codes[0] == 200


def test_removing_the_deployment_removes_its_schedules(api):
    from dgxmm.decider.features import wire_schedules
    wire_schedules(api.ctx)
    assert post(api, PAIR).status_code == 201
    assert api.post(f"{B}/deployments/{api.dep['id']}/stop", headers=api.h).status_code == 200
    assert api.delete(f"{B}/deployments/{api.dep['id']}", headers=api.h).status_code == 204
    from dgxmm.db import DeploymentSchedule
    with api.ctx.db.session() as s:
        assert s.scalars(select(DeploymentSchedule)).all() == []
        a = s.scalars(select(AuditLog).where(AuditLog.action == "schedule_delete").order_by(AuditLog.id.desc())).first()
    assert a is not None and a.target == str(api.dep["id"])  # removed with the deployment: audited too
    assert a.detail["reason"] == "deployment removed" and len(a.detail["schedule_ids"]) == 2


def test_pause_all_setting(api):
    url = f"{B}/schedules/settings"
    assert api.get(url).json()["paused_all"] is False
    r = api.put(url, headers=api.h, json={"paused_all": True})
    assert r.status_code == 200 and r.json()["paused_all"] is True
    assert api.get(f"{B}/deployments/{api.dep['id']}/schedules").json()["paused_all"] is True
    assert api.put(url, json={"paused_all": False}).status_code == 403  # CSRF
    assert api.put(url, headers=api.h, json={"paused_all": "no"}).status_code == 422
    with api.ctx.db.session() as s:
        assert [a.action for a in s.scalars(select(AuditLog).order_by(AuditLog.id))][-1] == "schedules_pause_all"
    v = login(api, "vic")
    assert api.get(url).json()["paused_all"] is True
    assert api.put(url, headers=v, json={"paused_all": False}).status_code == 403
