"""/api/integrations/litellm: settings, status, sync, models and virtual keys."""
import logging
from pathlib import Path

import pytest

from dgxmm.db import AuditLog
from dgxmm.litellm import LiteLLMClient, LiteLLMSync
from tests.fakes import FakeLiteLLM
from tests.test_api_deployments import admin
from tests.test_api_switch import deploy, wait

MASTER = "sk-master-TOPSECRET-4242"
BASE = "/dgxmm/api/integrations/litellm"


@pytest.fixture
def lite(api_stack, settings):
    Path(settings.litellm_key_file).write_text(MASTER)
    fake = FakeLiteLLM(master_key=MASTER)
    http = fake.client()
    c = api_stack.ctx
    c.litellm = LiteLLMSync(LiteLLMClient(http, c.settings_store.litellm), c.deployer, c.db, c.events)
    api_stack.fake = fake
    api_stack.h = admin(api_stack)
    yield api_stack
    api_stack.portal.call(http.aclose)


def as_viewer(c):
    c.post("/dgxmm/api/auth/logout", headers=c.h)
    r = c.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def audit_rows(c):
    with c.ctx.db.session() as s:
        return [(a.action, a.target, a.detail) for a in s.query(AuditLog).order_by(AuditLog.id)]


# ---------- settings + status ----------

def test_get_settings_shape(lite):
    r = lite.get(BASE)
    assert r.status_code == 200, r.text
    j = r.json()
    st = j.pop("status")
    assert j == {"enabled": True, "url": "http://litellm-proxy:4000", "public_url": "",
                 "endpoint": "http://litellm-proxy:4000/v1", "master_key_set": True, "master_key_source": "secret_file",
                 "master_key_hint": None, "default_url": "http://litellm-proxy:4000"}
    assert st["reachable"] is True and st["version"] == "1.100.0" and st["db"] == "connected"
    assert st["error"] is None and st["checked_at"]
    assert MASTER not in r.text


def test_version_from_readiness_when_present(lite):
    lite.fake.readiness_version, lite.fake.version = True, "1.55.0"
    assert lite.get(BASE).json()["status"]["version"] == "1.55.0"
    assert ("GET", "/openapi.json") not in lite.fake.calls


def test_get_settings_viewer_readable_and_unreachable(lite):
    as_viewer(lite)
    lite.fake.down = True
    r = lite.get(BASE)
    assert r.status_code == 200
    st = r.json()["status"]
    assert st["reachable"] is False and "litellm-proxy:4000" in st["error"] and st["version"] is None


def test_put_settings_admin_only_and_csrf(lite):
    assert lite.put(BASE, json={"public_url": "https://x.example"}).status_code == 403  # no CSRF header
    vh = as_viewer(lite)
    r = lite.put(BASE, headers=vh, json={"public_url": "https://x.example"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required"


def sent_key_to(fake, host):
    return [a for h, a in zip(fake.hosts, fake.auth) if h == host and a]


def test_file_key_never_follows_url_change(lite):
    """Reviewer's bypass: point the URL elsewhere with a garbage key, then clear the key. The secret-file key
    must not be sent to the other host (it is only ever used with the configured default URL)."""
    r = lite.put(BASE, headers=lite.h, json={"url": "http://evil.example:4000", "master_key": "sk-garbage"})
    assert r.status_code == 200
    r = lite.put(BASE, headers=lite.h, json={"master_key": ""})
    assert r.status_code == 200
    j = r.json()
    assert j["master_key_set"] is False and j["master_key_source"] == "none" and "re-enter" in j["master_key_hint"]
    lite.post(BASE + "/test", headers=lite.h)
    lite.post(BASE + "/sync", headers=lite.h)
    lite.get(BASE + "/keys")
    assert all(MASTER not in a for a in sent_key_to(lite.fake, "evil.example"))
    assert [a for a in sent_key_to(lite.fake, "evil.example") if a != "Bearer sk-garbage"] == []


def test_url_change_unbinds_stored_key(lite):
    lite.fake.master_key = "sk-db-key-1"
    assert lite.put(BASE, headers=lite.h, json={"master_key": "sk-db-key-1"}).json()["master_key_source"] == "settings"
    r = lite.put(BASE, headers=lite.h, json={"url": "http://evil.example:4000"})  # key not re-entered
    assert r.status_code == 200
    assert r.json()["master_key_set"] is False and r.json()["master_key_hint"]
    lite.post(BASE + "/test", headers=lite.h)
    lite.post(BASE + "/sync", headers=lite.h)
    assert sent_key_to(lite.fake, "evil.example") == []
    # back to the URL the key was entered for: usable again
    r = lite.put(BASE, headers=lite.h, json={"url": "http://litellm-proxy:4000"})
    assert r.json()["master_key_source"] == "settings" and r.json()["master_key_hint"] is None


def test_key_entered_with_url_is_bound_to_it(lite):
    lite.fake.master_key = "sk-new"
    r = lite.put(BASE, headers=lite.h, json={"url": "http://new-litellm:4000", "master_key": "sk-new"})
    assert r.json()["master_key_source"] == "settings"
    assert lite.post(BASE + "/test", headers=lite.h).json()["auth_ok"] is True
    assert sent_key_to(lite.fake, "new-litellm") and set(sent_key_to(lite.fake, "new-litellm")) == {"Bearer sk-new"}


def test_put_without_changes_is_not_audited(lite):
    lite.put(BASE, headers=lite.h, json={})
    lite.put(BASE, headers=lite.h, json={"url": "http://litellm-proxy:4000/"})  # = effective default
    lite.put(BASE, headers=lite.h, json={"public_url": ""})
    from dgxmm.settings_store import LITELLM_URL
    assert lite.ctx.settings_store.get(LITELLM_URL) is None
    assert not [a for a, _, _ in audit_rows(lite) if a == "litellm_settings_update"]


def test_status_is_cached(lite):
    first = lite.get(BASE).json()["status"]
    second = lite.get(BASE).json()["status"]
    assert first["checked_at"] == second["checked_at"]
    assert sum(1 for _, p in lite.fake.calls if p == "/health/liveliness") == 1
    lite.put(BASE, headers=lite.h, json={"public_url": "https://x.example"})  # a settings change re-probes
    assert sum(1 for _, p in lite.fake.calls if p == "/health/liveliness") == 2


def test_viewer_status_reads_are_unauthenticated(lite):
    as_viewer(lite)
    lite.fake.auth.clear()
    lite.get(BASE)
    lite.get(BASE)
    assert all(a is None for a in lite.fake.auth)


def test_impostor_endpoint(lite):
    lite.fake.impostor = True
    st = lite.get(BASE).json()["status"]
    assert st["reachable"] is False and "not a LiteLLM endpoint" in st["error"]


def test_key_patch_blocked_must_be_bool(lite):
    tok = lite.fake.add_key("x")
    r = lite.patch(f"{BASE}/keys/{tok}", headers=lite.h, json={"blocked": "yes"})
    assert r.status_code == 422


def test_put_settings_round_trip(lite):
    r = lite.put(BASE, headers=lite.h, json={"url": "http://other-litellm:4000/", "public_url": "https://llm.example.com/"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["url"] == "http://other-litellm:4000" and j["public_url"] == "https://llm.example.com"
    assert j["endpoint"] == "https://llm.example.com/v1"
    assert lite.fake.hosts[-1] == "other-litellm"  # read at call time, no restart
    r = lite.put(BASE, headers=lite.h, json={"url": "", "public_url": ""})  # empty → back to defaults
    assert r.json()["url"] == "http://litellm-proxy:4000" and r.json()["public_url"] == ""
    r = lite.put(BASE, headers=lite.h, json={"enabled": False})
    assert r.json()["enabled"] is False
    actions = [a for a, _, _ in audit_rows(lite)]
    assert actions.count("litellm_settings_update") == 3


@pytest.mark.parametrize("body,field", [
    ({"url": "ftp://x"}, "url"), ({"url": "not a url"}, "url"), ({"public_url": "javascript:alert(1)"}, "public_url"),
    ({"url": "http://"}, "url"), ({"enabled": "yes"}, "enabled"), ({"master_key": "has space"}, "master_key"),
    ({"master_key": 42}, "master_key"), ({"nope": 1}, "nope"),
])
def test_put_settings_validation(lite, body, field):
    r = lite.put(BASE, headers=lite.h, json=body)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "invalid_settings" and r.json()["detail"]["field"] == field


def test_put_validation_error_never_echoes_master_key(lite):
    r = lite.put(BASE, headers=lite.h, json={"master_key": "sk-new-SECRET-key", "url": "ftp://bad"})
    assert r.status_code == 422 and "sk-new-SECRET-key" not in r.text
    r = lite.put(BASE, headers=lite.h, json={"master_key": "sk-new-SECRET-key " * 3})
    assert r.status_code == 422 and "sk-new-SECRET-key" not in r.text


def test_master_key_write_only(lite):
    lite.fake.master_key = "sk-db-SECRET-9999"
    r = lite.put(BASE, headers=lite.h, json={"master_key": "sk-db-SECRET-9999"})
    assert r.status_code == 200
    j = r.json()
    assert j["master_key_set"] is True and j["master_key_source"] == "settings"
    assert "sk-db-SECRET-9999" not in r.text
    t = lite.post(BASE + "/test", headers=lite.h).json()
    assert t["auth_ok"] is True
    assert lite.fake.auth[-1] == "Bearer sk-db-SECRET-9999"
    r = lite.put(BASE, headers=lite.h, json={"master_key": ""})  # clear → secret file fallback
    assert r.json()["master_key_source"] == "secret_file"
    Path(lite.ctx.settings.litellm_key_file).write_text("")
    r = lite.get(BASE)
    assert r.json()["master_key_set"] is False and r.json()["master_key_source"] == "none"
    assert r.json()["enabled"] is False  # default: enabled only when a key is available
    for _, _, detail in audit_rows(lite):
        assert "sk-db-SECRET-9999" not in str(detail) and MASTER not in str(detail)
    assert any(d.get("master_key") == "set" for a, _, d in audit_rows(lite) if a == "litellm_settings_update")
    assert any(d.get("master_key") == "cleared" for a, _, d in audit_rows(lite) if a == "litellm_settings_update")


def test_connection_test(lite):
    r = lite.post(BASE + "/test", headers=lite.h)
    assert r.status_code == 200
    assert r.json() == {**r.json(), "reachable": True, "auth_ok": True, "version": "1.100.0", "db": "connected",
                        "error": None, "models": 1}
    lite.fake.master_key = "sk-something-else"
    j = lite.post(BASE + "/test", headers=lite.h).json()
    assert j["reachable"] is True and j["auth_ok"] is False and "master key" in j["error"]
    assert MASTER not in str(j) and "ab" * 32 not in str(j)
    lite.fake.down = True
    j = lite.post(BASE + "/test", headers=lite.h).json()
    assert j["reachable"] is False and j["auth_ok"] is None and j["error"]
    vh = as_viewer(lite)
    assert lite.post(BASE + "/test", headers=vh).status_code == 403


def test_test_without_master_key(lite):
    Path(lite.ctx.settings.litellm_key_file).write_text("")
    j = lite.post(BASE + "/test", headers=lite.h).json()
    assert j["reachable"] is True and j["auth_ok"] is False and "no master key" in j["error"]


# ---------- sync + models ----------

def test_sync_and_models(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    r = lite.post(BASE + "/sync", headers=lite.h)
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "status": "synced", "registered": 1, "removed": 0, "error": None}
    assert lite.post(BASE + "/sync", headers=lite.h).json()["registered"] == 0
    models = lite.get(BASE + "/models").json()
    by = {m["model_name"]: m for m in models}
    assert by["chat"]["managed_by_dgxmm"] is True
    assert by["chat"]["deployment"] == {"id": a["id"], "name": "a", "state": "ready"}
    assert by["chat"]["model"] == "openai/qwen3-tiny" and by["chat"]["api_base"].endswith("/v1")
    assert by["hand-written"]["managed_by_dgxmm"] is False and by["hand-written"]["deployment"] is None
    as_viewer(lite)
    models = lite.get(BASE + "/models").json()
    assert all(m["api_base"] is None for m in models)


def test_sync_failure_and_disabled(lite):
    lite.fake.down = True
    r = lite.post(BASE + "/sync", headers=lite.h)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "litellm_unreachable"
    lite.fake.down = False
    lite.put(BASE, headers=lite.h, json={"enabled": False})
    r = lite.post(BASE + "/sync", headers=lite.h)
    assert r.status_code == 200 and r.json()["status"] == "disabled"
    r = lite.get(BASE + "/models")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "litellm_disabled"


def test_settings_change_triggers_sync(lite):
    lite.ctx.litellm._wake.clear()
    lite.put(BASE, headers=lite.h, json={"public_url": "https://llm.example.com"})
    assert lite.ctx.litellm._wake.is_set()


# ---------- virtual keys ----------

def test_key_lifecycle(lite):
    r = lite.post(BASE + "/keys", headers=lite.h, json={
        "key_alias": "team-a", "models": ["chat"], "max_budget": 5, "budget_duration": "30d", "rpm_limit": 10,
        "tpm_limit": 1000, "duration": "30d", "metadata": {"team": "a", "created_by": "spoofed"}})
    assert r.status_code == 201, r.text
    created = r.json()
    full = created["key"]
    assert full.startswith("sk-") and len(created["token"]) == 64
    assert created["key_alias"] == "team-a" and created["models"] == ["chat"] and created["rpm_limit"] == 10
    assert created["expires"] and created["created_by_dgxmm"] is True
    sent = next(b for b in lite.fake.bodies if b.get("key_alias") == "team-a")
    assert sent["metadata"] == {"team": "a", "created_by": "dgxmm", "dgxmm_user": "root"}
    assert sent["duration"] == "30d" and sent["max_budget"] == 5

    lite.fake.add_key("external", ["all-proxy-models"])
    r = lite.get(BASE + "/keys")
    assert r.status_code == 200
    keys = {k["key_alias"]: k for k in r.json()}
    assert set(keys) == {"team-a", "external"}
    k = keys["team-a"]
    assert set(k) == {"token", "key_name", "key_alias", "models", "spend", "max_budget", "budget_duration", "rpm_limit",
                      "tpm_limit", "expires", "created_at", "blocked", "metadata", "created_by_dgxmm"}
    assert k["key_name"].startswith("sk-...") and keys["external"]["created_by_dgxmm"] is False
    assert full not in r.text

    tok = created["token"]
    r = lite.patch(f"{BASE}/keys/{tok}", headers=lite.h, json={"models": [], "blocked": True, "rpm_limit": 20,
                                                               "key_alias": "team-a2"})
    assert r.status_code == 200, r.text
    assert r.json()["blocked"] is True and r.json()["models"] == [] and r.json()["key_alias"] == "team-a2"
    assert full not in r.text
    assert lite.delete(f"{BASE}/keys/{tok}", headers=lite.h).status_code == 204
    assert tok not in lite.fake.keys
    r = lite.delete(f"{BASE}/keys/{tok}", headers=lite.h)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "litellm_key_not_found"
    r = lite.patch(f"{BASE}/keys/{tok}", headers=lite.h, json={"blocked": False})
    assert r.status_code == 404

    rows = audit_rows(lite)
    acts = [(a, t) for a, t, _ in rows if a.startswith("litellm_key")]
    assert acts == [("litellm_key_create", "team-a"), ("litellm_key_update", "team-a2"), ("litellm_key_delete", "team-a2")]
    create_detail = next(d for a, _, d in rows if a == "litellm_key_create")
    assert create_detail["models"] == ["chat"]
    assert all(full not in str(d) for _, _, d in rows)


def test_key_inputs_validated(lite):
    bad = [{"models": []}, {"key_alias": "", "models": []}, {"key_alias": "x" * 200}, {"key_alias": "ok", "models": "chat"},
           {"key_alias": "ok", "duration": "forever"}, {"key_alias": "ok", "rpm_limit": -1},
           {"key_alias": "ok", "max_budget": -5}, {"key_alias": "ok", "models": ["bad name!"]}]
    for body in bad:
        r = lite.post(BASE + "/keys", headers=lite.h, json=body)
        assert r.status_code == 422, (body, r.text)
    r = lite.patch(BASE + "/keys/sk-notahash", headers=lite.h, json={"blocked": True})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_token"
    assert lite.delete(BASE + "/keys/xyz", headers=lite.h).status_code == 422


def test_keys_admin_only(lite):
    tok = lite.fake.add_key("x")
    vh = as_viewer(lite)
    assert lite.get(BASE + "/keys").status_code == 403
    assert lite.post(BASE + "/keys", headers=vh, json={"key_alias": "v"}).status_code == 403
    assert lite.patch(f"{BASE}/keys/{tok}", headers=vh, json={"blocked": True}).status_code == 403
    assert lite.delete(f"{BASE}/keys/{tok}", headers=vh).status_code == 403


def test_keys_no_db_is_409(lite):
    lite.fake.no_db = True
    for r in (lite.get(BASE + "/keys"), lite.post(BASE + "/keys", headers=lite.h, json={"key_alias": "a"}),
              lite.patch(f"{BASE}/keys/{'a' * 64}", headers=lite.h, json={"blocked": True}),
              lite.delete(f"{BASE}/keys/{'a' * 64}", headers=lite.h)):
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["code"] == "litellm_no_db" and "database" in r.json()["detail"]["message"]


def test_keys_no_db_detected_from_error_message(lite):
    """Even if readiness claims a DB, LiteLLM's 'No connected db' error maps to 409."""
    lite.fake.no_db = True
    lite.fake.readiness_db = "connected"
    r = lite.get(BASE + "/keys")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "litellm_no_db"


def test_keys_unreachable_and_bad_master(lite):
    lite.fake.down = True
    r = lite.get(BASE + "/keys")
    assert r.status_code == 502 and r.json()["detail"]["code"] == "litellm_unreachable"
    assert "litellm-proxy:4000" in r.json()["detail"]["message"]
    lite.fake.down = False
    lite.fake.master_key = "sk-other"
    r = lite.get(BASE + "/keys")
    assert r.status_code == 502 and r.json()["detail"]["code"] == "litellm_auth_failed"
    assert MASTER not in r.text
    Path(lite.ctx.settings.litellm_key_file).write_text("")
    r = lite.get(BASE + "/keys")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "litellm_not_configured"


def test_key_list_paginates(lite):
    for i in range(130):
        lite.fake.add_key(f"k{i}")
    assert len(lite.get(BASE + "/keys").json()) == 130


# ---------- secrets never leak ----------

def test_master_key_never_in_responses_audit_or_logs(lite, caplog):
    caplog.set_level(logging.DEBUG)
    texts = []
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    texts.append(lite.put(BASE, headers=lite.h, json={"master_key": MASTER, "public_url": "https://llm.example.com"}).text)
    for call in (lambda: lite.get(BASE), lambda: lite.post(BASE + "/test", headers=lite.h),
                 lambda: lite.post(BASE + "/sync", headers=lite.h), lambda: lite.get(BASE + "/models"),
                 lambda: lite.post(BASE + "/keys", headers=lite.h, json={"key_alias": "k", "models": ["chat"]}),
                 lambda: lite.get(BASE + "/keys"), lambda: lite.get(f"/dgxmm/api/deployments/{a['id']}/litellm"),
                 lambda: lite.get("/dgxmm/api/audit")):
        texts.append(call().text)
    lite.fake.master_key = "sk-rotated"  # LiteLLM now rejects our key: error paths must not echo it either
    for call in (lambda: lite.post(BASE + "/test", headers=lite.h), lambda: lite.post(BASE + "/sync", headers=lite.h),
                 lambda: lite.get(BASE + "/keys"), lambda: lite.get(f"/dgxmm/api/deployments/{a['id']}/litellm")):
        texts.append(call().text)
    lite.fake.down = True
    texts += [lite.post(BASE + "/sync", headers=lite.h).text, lite.get(BASE + "/keys").text]
    for t in texts:
        assert MASTER not in t
    for _, _, d in audit_rows(lite):
        assert MASTER not in str(d)
    for rec in caplog.records:
        assert MASTER not in rec.getMessage() and MASTER not in str(rec.args)


def test_full_virtual_key_only_in_create_response(lite, caplog):
    caplog.set_level(logging.DEBUG)
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    full = lite.post(BASE + "/keys", headers=lite.h, json={"key_alias": "k", "models": ["chat"]}).json()["key"]
    tok = next(iter(lite.fake.keys))
    texts = [lite.get(BASE + "/keys").text, lite.get(f"/dgxmm/api/deployments/{a['id']}/litellm").text,
             lite.patch(f"{BASE}/keys/{tok}", headers=lite.h, json={"rpm_limit": 3}).text,
             lite.get("/dgxmm/api/audit").text]
    for t in texts:
        assert full not in t
    for rec in caplog.records:
        assert full not in rec.getMessage()


def test_public_url_seeded_from_config(lite):
    from dgxmm.settings_store import LITELLM_PUBLIC_URL
    lite.ctx.settings.litellm_public_url = "http://10.0.0.5:4000"
    j = lite.get(BASE).json()
    assert j["public_url"] == "http://10.0.0.5:4000" and j["endpoint"] == "http://10.0.0.5:4000/v1"
    # re-sending the config default stores nothing and is not a change
    r = lite.put(BASE, headers=lite.h, json={"public_url": "http://10.0.0.5:4000/"})
    assert r.status_code == 200 and lite.ctx.settings_store.get(LITELLM_PUBLIC_URL) is None
    assert "litellm_settings_update" not in [a for a, _, _ in audit_rows(lite)]
    # a stored value wins over the config default
    r = lite.put(BASE, headers=lite.h, json={"public_url": "https://llm.example.com"})
    assert r.json()["public_url"] == "https://llm.example.com"
    # clearing stays cleared (the endpoint falls back to the internal URL), it does not revert to the seed
    r = lite.put(BASE, headers=lite.h, json={"public_url": ""})
    assert r.json()["public_url"] == "" and r.json()["endpoint"] == "http://litellm-proxy:4000/v1"
    assert lite.get(BASE).json()["public_url"] == ""
    # setting it back to the config default unsets the stored override
    lite.put(BASE, headers=lite.h, json={"public_url": "http://10.0.0.5:4000"})
    assert lite.ctx.settings_store.get(LITELLM_PUBLIC_URL) is None
    assert lite.get(BASE).json()["public_url"] == "http://10.0.0.5:4000"
