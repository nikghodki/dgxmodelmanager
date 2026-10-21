"""PATCH /deployments/{id} (alias = LiteLLM model name), model-name uniqueness, GET /deployments/{id}/litellm."""
import pytest

from tests.test_api_deployments import SMALL
from tests.test_api_integrations import BASE, MASTER, as_viewer, audit_rows, lite  # noqa: F401  (fixture)
from tests.test_api_switch import deploy, wait

DEP = "/dgxmm/api/deployments"


def taken(r, holder_name):
    assert r.status_code == 409, r.text
    d = r.json()["detail"]
    assert d["code"] == "model_name_taken" and d["holder"]["name"] == holder_name, d
    return d


def test_view_has_model_name(lite):
    a = deploy(lite, lite.h, "a")
    assert a["model_name"] == "qwen3-tiny"
    b = deploy(lite, lite.h, "b", alias="chat")
    assert b["model_name"] == "chat"


def test_patch_alias_validation_and_uniqueness(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    b = deploy(lite, lite.h, "b")
    assert wait(lite, b["id"], "ready")
    for bad in ("has space", "x" * 65, "semi;colon", "ünï"):
        r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": bad})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_alias", bad
    d = taken(lite.patch(f"{DEP}/{b['id']}", headers=lite.h, json={"alias": "chat"}), "a")
    assert d["holder"] == {"id": a["id"], "name": "a", "state": d["holder"]["state"]}
    # cannot take the model name another active deployment is registered under (its served name)
    taken(lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "qwen3-tiny"}), "b")
    # clearing the alias would make a's model name qwen3-tiny, which b holds
    taken(lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": None}), "b")
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "org/Chat-v2.1:latest"})
    assert r.status_code == 200, r.text
    assert r.json()["alias"] == "org/Chat-v2.1:latest" and r.json()["model_name"] == "org/Chat-v2.1:latest"
    assert lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "org/Chat-v2.1:latest"}).status_code == 200
    lite.post(f"{DEP}/{b['id']}/stop", headers=lite.h)
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": None})
    assert r.status_code == 200 and r.json()["alias"] is None and r.json()["model_name"] == "qwen3-tiny"
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": ""})
    assert r.status_code == 200 and r.json()["alias"] is None
    assert lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={}).status_code == 422
    assert lite.patch(f"{DEP}/999", headers=lite.h, json={"alias": "z"}).status_code == 404
    rows = [(a_, t, d) for a_, t, d in audit_rows(lite) if a_ == "deployment_alias"]
    assert rows[0][:2] == ("deployment_alias", "a")
    assert rows[0][2]["from"] == "chat" and rows[0][2]["to"] == "org/Chat-v2.1:latest"


def test_deploy_rejects_taken_model_names(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    body = {"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}
    taken(lite.post(DEP, headers=lite.h, json={**body, "name": "b", "alias": "chat"}), "a")
    deploy(lite, lite.h, "c")  # served name qwen3-tiny is free (a is registered as "chat")
    taken(lite.post(DEP, headers=lite.h, json={**body, "name": "d"}), "c")  # same served name, no alias
    # a served name may not equal another deployment's alias either, even a stopped one
    lite.post(f"{DEP}/{a['id']}/stop", headers=lite.h)
    taken(lite.post(DEP, headers=lite.h, json={**body, "name": "e", "config": {**SMALL, "served_model_name": "chat"}}), "a")


def test_start_rejects_taken_model_name(lite):
    a = deploy(lite, lite.h, "a")
    assert wait(lite, a["id"], "ready")
    lite.post(f"{DEP}/{a['id']}/stop", headers=lite.h)
    b = deploy(lite, lite.h, "b")  # takes qwen3-tiny while a is stopped
    assert wait(lite, b["id"], "ready")
    taken(lite.post(f"{DEP}/{a['id']}/start", headers=lite.h), "b")
    lite.post(f"{DEP}/{b['id']}/stop", headers=lite.h)
    assert lite.post(f"{DEP}/{a['id']}/start", headers=lite.h).status_code == 200


def test_wake_rejects_taken_model_name(lite):
    a = deploy(lite, lite.h, "a", config={**SMALL, "enable_sleep_mode": True})
    assert wait(lite, a["id"], "ready")
    assert lite.post(f"{DEP}/{a['id']}/sleep", headers=lite.h).json()["state"] == "sleeping"
    b = deploy(lite, lite.h, "b")
    assert wait(lite, b["id"], "ready")
    taken(lite.post(f"{DEP}/{a['id']}/wake", headers=lite.h), "b")


def test_removed_deployment_frees_its_alias(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    lite.post(f"{DEP}/{a['id']}/stop", headers=lite.h)
    assert lite.delete(f"{DEP}/{a['id']}", headers=lite.h).status_code == 204
    assert deploy(lite, lite.h, "b", alias="chat")["model_name"] == "chat"


async def test_concurrent_deploys_cannot_share_a_name(stack):
    import asyncio
    from dgxmm.deployer.service import DeployError, DeployRequest
    stack.http.all_healthy = True
    reqs = [DeployRequest(name=n, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, alias="chat") for n in "xy"]
    out = await asyncio.gather(*(stack.deployer.deploy(r, 1) for r in reqs), return_exceptions=True)
    errors = [o for o in out if isinstance(o, DeployError)]
    assert len(errors) == 1 and errors[0].code == "model_name_taken"


def test_patch_alias_admin_only(lite):
    a = deploy(lite, lite.h, "a")
    vh = as_viewer(lite)
    assert lite.patch(f"{DEP}/{a['id']}", headers=vh, json={"alias": "x"}).status_code == 403


def test_alias_change_reregisters_via_api(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    lite.post(BASE + "/sync", headers=lite.h)
    lite.ctx.litellm._wake.clear()
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat2"})
    assert r.json()["affected_keys"] == [] and r.json()["keys_rewritten"] == 0
    assert lite.ctx.litellm._wake.is_set()
    r = lite.post(BASE + "/sync", headers=lite.h).json()
    assert (r["registered"], r["removed"]) == (1, 1)
    mine = [m["model_name"] for m in lite.fake.models if m["model_info"].get("dgxmm_id")]
    assert mine == ["chat2"]


def test_alias_change_reports_and_rewrites_scoped_keys(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    scoped = lite.fake.add_key("scoped", ["chat", "other"])
    lite.fake.add_key("all", [])
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat2"})
    assert r.status_code == 200
    j = r.json()
    assert [k["key_alias"] for k in j["affected_keys"]] == ["scoped"] and j["keys_rewritten"] == 0
    assert j["affected_keys"][0] == {"token": scoped, "key_alias": "scoped", "models": ["chat", "other"],
                                     "rewritten": False}
    assert lite.fake.keys[scoped]["models"] == ["chat", "other"]  # untouched without rewrite_keys
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat3", "rewrite_keys": False})
    assert r.json()["affected_keys"] == []  # the key no longer names chat2
    lite.fake.keys[scoped]["models"] = ["chat3", "other"]
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat4", "rewrite_keys": True})
    j = r.json()
    assert j["keys_rewritten"] == 1 and lite.fake.keys[scoped]["models"] == ["chat4", "other"]
    assert j["affected_keys"][0]["rewritten"] is True
    row = [d for a_, _, d in audit_rows(lite) if a_ == "deployment_alias"][-1]
    assert row["keys_rewritten"] == 1 and row["rewritten_keys"] == ["scoped"]


def test_partial_key_rewrite_is_reported_and_audited(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    k1 = lite.fake.add_key("k1", ["chat"])
    k2 = lite.fake.add_key("k2", ["chat"])
    lite.fake.fail_update_tokens.add(k2)
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat2", "rewrite_keys": True})
    assert r.status_code == 200
    j = r.json()
    assert {k["key_alias"]: k["rewritten"] for k in j["affected_keys"]} == {"k1": True, "k2": False}
    assert j["keys_rewritten"] == 1 and j["keys_error"]["code"] == "litellm_error"
    assert lite.fake.keys[k1]["models"] == ["chat2"] and lite.fake.keys[k2]["models"] == ["chat"]
    row = [d for a_, _, d in audit_rows(lite) if a_ == "deployment_alias"][-1]
    assert row["keys_rewritten"] == 1 and row["rewritten_keys"] == ["k1"]


def test_rewrite_does_not_duplicate_new_name(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    k = lite.fake.add_key("both", ["chat", "chat2", "x"])
    lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat2", "rewrite_keys": True})
    assert lite.fake.keys[k]["models"] == ["chat2", "x"]


async def test_rename_waits_for_name_reservation_lock(stack):
    import asyncio
    from dgxmm.deployer.service import DeployRequest
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL), 1)
    async with stack.deployer.mem_lock:  # held by deploy/wake/start while they check and claim names
        t = asyncio.create_task(stack.deployer.rename(d["id"], "chat"))
        await asyncio.sleep(0.05)
        assert not t.done()
    assert (await t)["alias"] == "chat"


def test_alias_change_keys_unavailable(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    lite.fake.no_db = True
    j = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat2"}).json()
    assert j["alias"] == "chat2" and j["affected_keys"] == [] and j["keys_error"]["code"] == "litellm_no_db"


def test_deployment_litellm_info(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    other = deploy(lite, lite.h, "b")
    assert wait(lite, a["id"], "ready") and wait(lite, other["id"], "ready")
    lite.put(BASE, headers=lite.h, json={"public_url": "https://llm.example.com"})
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["registered"] is None and j["model_name"] == "chat"  # nothing synced yet
    lite.post(BASE + "/sync", headers=lite.h)
    lite.fake.add_key("for-chat", ["chat"])
    lite.fake.add_key("all-1", [])
    lite.fake.add_key("all-2", ["all-proxy-models"])
    lite.fake.add_key("for-other", ["qwen3-tiny"])
    lite.ctx.litellm.client.invalidate_keys()
    r = lite.get(f"{DEP}/{a['id']}/litellm")
    assert r.status_code == 200, r.text
    j = r.json()
    keys = sorted(k["key_alias"] for k in j.pop("keys"))
    assert keys == ["all-1", "all-2", "for-chat"]
    assert j == {"deployment_id": a["id"], "model_name": "chat", "endpoint": "https://llm.example.com/v1",
                 "registered": True, "status": "synced", "enabled": True, "error": None,
                 "keys_visible": True, "keys_error": None}
    lite.fake.calls.clear()
    as_viewer(lite)
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["keys"] == [] and j["keys_visible"] is False and j["registered"] is True
    assert lite.fake.calls == []  # viewers never trigger LiteLLM calls here


def test_deployment_litellm_keys_cached(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    lite.get(f"{DEP}/{a['id']}/litellm")
    lite.fake.add_key("late", [])
    assert lite.get(f"{DEP}/{a['id']}/litellm").json()["keys"] == []  # cached ~15 s
    lite.post(BASE + "/keys", headers=lite.h, json={"key_alias": "via-api"})  # API writes invalidate
    keys = sorted(k["key_alias"] for k in lite.get(f"{DEP}/{a['id']}/litellm").json()["keys"])
    assert keys == ["late", "via-api"]


def test_deployment_litellm_info_degrades(lite):
    a = deploy(lite, lite.h, "a")
    lite.fake.down = True
    lite.post(BASE + "/sync", headers=lite.h)
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["registered"] is None and j["error"]["code"] == "litellm_unreachable"
    assert j["keys_error"]["code"] == "litellm_unreachable"
    lite.fake.down = False
    lite.fake.no_db = True
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["keys"] == [] and j["keys_error"]["code"] == "litellm_no_db"
    lite.put(BASE, headers=lite.h, json={"enabled": False})
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["enabled"] is False and j["registered"] is None and j["error"]["code"] == "litellm_disabled"
    assert lite.get(f"{DEP}/999/litellm").status_code == 404


def test_name_conflict_backstop(lite):
    """If two ready deployments end up with one model name anyway, only one is registered; the other is
    marked litellm_status=error with a message."""
    from dgxmm.db import Deployment
    a = deploy(lite, lite.h, "a", alias="chat")
    b = deploy(lite, lite.h, "b", alias="chat-b")
    assert wait(lite, a["id"], "ready") and wait(lite, b["id"], "ready")
    with lite.ctx.db.session() as s:
        s.get(Deployment, b["id"]).alias = "chat"  # bypass validation
    lite.post(BASE + "/sync", headers=lite.h)
    mine = [(m["model_name"], m["model_info"]["dgxmm_id"]) for m in lite.fake.models if m["model_info"].get("dgxmm_id")]
    assert mine == [("chat", str(a["id"]))]
    assert lite.get(f"{DEP}/{b['id']}").json()["litellm_status"] == "error"
    j = lite.get(f"{DEP}/{b['id']}/litellm").json()
    assert j["registered"] is False and j["error"]["code"] == "model_name_conflict" and "a" in j["error"]["message"]


@pytest.mark.parametrize("alias", ["a", "A.b_c-d:e/f", "x" * 64])
def test_alias_pattern_accepts(lite, alias):
    a = deploy(lite, lite.h, "a")
    assert lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": alias}).status_code == 200


# ---------- per-deployment litellm_error in the deployment views ----------

def by_id(c, dep_id):
    return next(d for d in c.get(DEP).json() if d["id"] == dep_id)


def test_list_has_litellm_error_for_name_conflict(lite):
    from dgxmm.db import Deployment
    a = deploy(lite, lite.h, "a", alias="chat")
    b = deploy(lite, lite.h, "b", alias="chat-b")
    assert wait(lite, a["id"], "ready") and wait(lite, b["id"], "ready")
    lite.post(BASE + "/sync", headers=lite.h)
    assert by_id(lite, a["id"])["litellm_error"] is None and by_id(lite, b["id"])["litellm_error"] is None
    with lite.ctx.db.session() as s:
        s.get(Deployment, b["id"]).alias = "chat"  # bypass validation
    lite.post(BASE + "/sync", headers=lite.h)
    err = by_id(lite, b["id"])["litellm_error"]
    assert err["code"] == "model_name_conflict" and "deployment a" in err["message"]
    assert by_id(lite, a["id"])["litellm_error"] is None
    assert lite.get(f"{DEP}/{b['id']}").json()["litellm_error"] == err  # detail view and SSE payloads too


def test_list_has_litellm_error_for_failed_sync_on_ready_deployments_only(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    b = deploy(lite, lite.h, "b", alias="other")
    assert wait(lite, a["id"], "ready") and wait(lite, b["id"], "ready")
    lite.post(f"{DEP}/{b['id']}/stop", headers=lite.h)
    assert wait(lite, b["id"], "stopped")
    lite.fake.down = True
    lite.post(BASE + "/sync", headers=lite.h)
    err = by_id(lite, a["id"])["litellm_error"]
    assert err["code"] == "litellm_unreachable" and err["message"]
    assert by_id(lite, b["id"])["litellm_error"] is None  # not registered anyway: no noise
    lite.fake.down = False
    lite.post(BASE + "/sync", headers=lite.h)
    assert by_id(lite, a["id"])["litellm_error"] is None


def test_litellm_error_never_contains_master_key(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    lite.fake.master_key = "sk-rotated-elsewhere"  # LiteLLM now rejects DGXMM's key
    lite.post(BASE + "/sync", headers=lite.h)
    r = lite.get(DEP)
    err = by_id(lite, a["id"])["litellm_error"]
    assert err["code"] == "litellm_auth_failed"
    assert MASTER not in r.text


def test_deploy_and_rename_refuse_a_name_litellm_has_from_elsewhere(lite):
    """A static config.yaml entry (no dgxmm_id) holds a name: 409 model_name_taken without a holder deployment,
    with a message the deploy wizard / rename form show as-is."""
    lite.fake.models.append({"model_name": "chat", "litellm_params": {"model": "openai/chat", "api_base": "http://old-engine:8000/v1"},
                             "model_info": {"id": "static-1", "db_model": False}})
    body = {"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL, "name": "a", "alias": "chat"}
    r = lite.post(DEP, headers=lite.h, json=body)
    assert r.status_code == 409, r.text
    d = r.json()["detail"]
    assert d["code"] == "model_name_taken" and d["holder"] is None and d["external"]["host"] == "old-engine"
    assert d["message"].startswith("LiteLLM already has a model named chat that DGXMM didn't create")
    assert MASTER not in r.text
    a = deploy(lite, lite.h, "a", alias="mine")
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat"})
    assert r.status_code == 409 and r.json()["detail"]["holder"] is None


def test_litellm_tab_reports_name_conflict(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    assert lite.post(BASE + "/sync", headers=lite.h).status_code == 200
    lite.fake.models.append({"model_name": "chat", "litellm_params": {"model": "openai/chat", "api_base": "http://old-engine:8000/v1"},
                             "model_info": {"id": "static-1", "db_model": False}})
    assert lite.post(BASE + "/sync", headers=lite.h).status_code == 200
    j = lite.get(f"{DEP}/{a['id']}/litellm", headers=lite.h).json()
    assert j["status"] == "conflict" and j["registered"] is True
    assert j["error"]["code"] == "litellm_name_conflict" and "old-engine" in j["error"]["message"]
    ins = lite.get("/dgxmm/api/insights", headers=lite.h).json()
    items = ins["items"] if isinstance(ins, dict) else ins
    assert [i["kind"] for i in items if i["status"] == "open"] == ["litellm_conflict"]
