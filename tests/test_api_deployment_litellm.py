"""PATCH /deployments/{id} (alias = LiteLLM model name) and GET /deployments/{id}/litellm."""
import pytest

from tests.test_api_integrations import BASE, MASTER, as_viewer, audit_rows, lite  # noqa: F401  (fixture)
from tests.test_api_switch import deploy, wait

DEP = "/dgxmm/api/deployments"


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
    r = lite.patch(f"{DEP}/{b['id']}", headers=lite.h, json={"alias": "chat"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "alias_in_use"
    # cannot take the model name another active deployment is registered under (its served name)
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "qwen3-tiny"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "alias_in_use"
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "org/Chat-v2.1:latest"})
    assert r.status_code == 200, r.text
    assert r.json()["alias"] == "org/Chat-v2.1:latest" and r.json()["model_name"] == "org/Chat-v2.1:latest"
    assert lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "org/Chat-v2.1:latest"}).status_code == 200
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": None})
    assert r.json()["alias"] is None and r.json()["model_name"] == "qwen3-tiny"
    r = lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": ""})
    assert r.status_code == 200 and r.json()["alias"] is None
    assert lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={}).status_code == 422
    assert lite.patch(f"{DEP}/999", headers=lite.h, json={"alias": "z"}).status_code == 404
    rows = [(a_, t, d) for a_, t, d in audit_rows(lite) if a_ == "deployment_alias"]
    assert rows[0] == ("deployment_alias", "a", {"from": "chat", "to": "org/Chat-v2.1:latest"})


def test_deploy_rejects_duplicate_alias(lite):
    deploy(lite, lite.h, "a", alias="chat")
    r = lite.post(DEP, headers=lite.h, json={"name": "b", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                             "config": {"max_model_len": 4096, "max_num_seqs": 2,
                                                        "max_num_batched_tokens": 2048}, "alias": "chat"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "alias_in_use"


def test_patch_alias_admin_only(lite):
    a = deploy(lite, lite.h, "a")
    vh = as_viewer(lite)
    assert lite.patch(f"{DEP}/{a['id']}", headers=vh, json={"alias": "x"}).status_code == 403


def test_alias_change_reregisters_via_api(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    assert wait(lite, a["id"], "ready")
    lite.post(BASE + "/sync", headers=lite.h)
    lite.ctx.litellm._wake.clear()
    lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": "chat2"})
    assert lite.ctx.litellm._wake.is_set()
    r = lite.post(BASE + "/sync", headers=lite.h).json()
    assert (r["registered"], r["removed"]) == (1, 1)
    mine = [m["model_name"] for m in lite.fake.models if m["model_info"].get("dgxmm_id")]
    assert mine == ["chat2"]


def test_deployment_litellm_info(lite):
    a = deploy(lite, lite.h, "a", alias="chat")
    other = deploy(lite, lite.h, "b")
    assert wait(lite, a["id"], "ready") and wait(lite, other["id"], "ready")
    lite.put(BASE, headers=lite.h, json={"public_url": "https://llm.example.com"})
    r = lite.get(f"{DEP}/{a['id']}/litellm")
    j = r.json()
    assert j["registered"] is False and j["model_name"] == "chat"
    lite.post(BASE + "/sync", headers=lite.h)
    lite.fake.add_key("for-chat", ["chat"])
    lite.fake.add_key("all-1", [])
    lite.fake.add_key("all-2", ["all-proxy-models"])
    lite.fake.add_key("for-other", ["qwen3-tiny"])
    r = lite.get(f"{DEP}/{a['id']}/litellm")
    assert r.status_code == 200, r.text
    j = r.json()
    keys = sorted(k["key_alias"] for k in j.pop("keys"))
    assert keys == ["all-1", "all-2", "for-chat"]
    assert j == {"deployment_id": a["id"], "model_name": "chat", "endpoint": "https://llm.example.com/v1",
                 "registered": True, "status": "synced", "enabled": True, "error": None,
                 "keys_visible": True, "keys_error": None}
    as_viewer(lite)
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["keys"] == [] and j["keys_visible"] is False and j["registered"] is True


def test_deployment_litellm_info_degrades(lite):
    a = deploy(lite, lite.h, "a")
    lite.fake.no_db = True
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["keys"] == [] and j["keys_error"]["code"] == "litellm_no_db"
    lite.fake.down = True
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["registered"] is None and j["error"]["code"] == "litellm_unreachable"
    assert j["keys_error"]["code"] == "litellm_unreachable"
    lite.fake.down = False
    lite.put(BASE, headers=lite.h, json={"enabled": False})
    j = lite.get(f"{DEP}/{a['id']}/litellm").json()
    assert j["enabled"] is False and j["registered"] is None and j["error"]["code"] == "litellm_disabled"
    assert lite.get(f"{DEP}/999/litellm").status_code == 404


@pytest.mark.parametrize("alias", ["a", "A.b_c-d:e/f", "x" * 64])
def test_alias_pattern_accepts(lite, alias):
    a = deploy(lite, lite.h, "a")
    assert lite.patch(f"{DEP}/{a['id']}", headers=lite.h, json={"alias": alias}).status_code == 200
