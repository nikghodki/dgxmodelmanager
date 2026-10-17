"""Release fixes for the decision model: generic actions on the System decider, failed restarts and the hourly
give-up, the decider's memory in plans, and a blob store symlinked outside the HF cache."""
import json
import os
import shutil
import time

import pytest

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployError, DeployRequest
from dgxmm.engines.decision import DECIDER_MODELS
from tests.test_decider_engine import D3_CONFIG, d3, deploy_decider  # noqa: F401  (fixture)
from tests.test_decider_models import NANO, _xet_layout, api  # noqa: F401  (fixture)
from tests.test_decider_supervisor import crash, running_decider, sup  # noqa: F401  (fixture)

B = "/dgxmm/api/decider"
D = "/dgxmm/api/deployments"


def enabled_decider(api) -> int:  # noqa: F811
    r = api.post(f"{B}/enable", headers=api.h, json={})
    assert r.status_code == 202, r.text
    dep_id = r.json()["deployment_id"]
    for _ in range(300):
        if api.get(f"{D}/{dep_id}").json()["state"] == "ready":
            break
        time.sleep(0.01)
    return dep_id


# ---------- generic actions on the System decider ----------

def test_switch_and_reconfigure_refuse_the_system_deployment(api):  # noqa: F811
    dep_id = enabled_decider(api)
    req = {"name": "other", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {}}
    r = api.post(f"{D}/{dep_id}/switch", headers=api.h, json={"request": req, "actions": []})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "system_deployment", r.text
    assert "Decision model page" in r.json()["detail"]["message"]
    r = api.post(f"{D}/{dep_id}/reconfigure", headers=api.h, json={"config": {"batch_size": 4}})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "system_deployment", r.text
    view = api.get(f"{D}/{dep_id}").json()
    assert view["state"] == "ready" and view["config"].get("batch_size", 8) == 8  # untouched


def test_start_of_a_disabled_decider_is_refused_and_points_at_enable(api):  # noqa: F811
    dep_id = enabled_decider(api)
    assert api.post(f"{B}/disable", headers=api.h, json={}).status_code == 200
    assert api.get(f"{D}/{dep_id}").json()["state"] == "stopped"
    r = api.post(f"{D}/{dep_id}/start", headers=api.h)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "decider_disabled", r.text
    assert api.get(f"{D}/{dep_id}").json()["state"] == "stopped"  # nothing loaded for no feature
    # enabled but stopped (an admin's Stop): Start is a plain restart
    assert api.post(f"{B}/enable", headers=api.h, json={}).status_code == 202
    for _ in range(300):
        if api.get(f"{D}/{dep_id}").json()["state"] == "ready":
            break
        time.sleep(0.01)
    assert api.post(f"{D}/{dep_id}/stop", headers=api.h).status_code == 200
    r = api.post(f"{D}/{dep_id}/start", headers=api.h)
    assert r.status_code == 200, r.text


def test_removing_the_decider_disables_it_and_clears_its_deployment_id(api):  # noqa: F811
    dep_id = enabled_decider(api)
    assert api.post(f"{D}/{dep_id}/stop", headers=api.h).status_code == 200
    assert api.delete(f"{D}/{dep_id}", headers=api.h).status_code == 204
    st = api.get(f"{B}/status").json()
    assert st["enabled"] is False and st["deployment_id"] is None
    assert api.ctx.decider.deployment_id is None
    rows = api.get("/dgxmm/api/audit", headers=api.h).json()
    items = rows["items"] if isinstance(rows, dict) else rows
    assert any(a["action"] == "decider_disable" for a in items)


# ---------- supervisor: failed restarts count toward the hourly give-up ----------

async def test_failed_restarts_count_toward_the_hourly_limit_and_give_up(sup, monkeypatch):  # noqa: F811
    view = await running_decider(sup)
    await crash(sup, view)

    async def broken_restart(*a, **kw):
        raise DeployError("docker_error", "could not start container: image dgxmm-decider:3 not found", 502)
    monkeypatch.setattr(sup.stack.deployer, "restart", broken_restart)
    limit = sup.stack.settings.decider_max_restarts_per_hour
    for _ in range(limit):
        assert await sup.sup.tick() == "restart_failed"
        sup.clock.t = sup.sup.next_attempt_at + 1  # each retry as soon as its backoff allows
    st = sup.sup.status()
    assert st["restarts_last_hour"] == limit
    assert await sup.sup.tick() == "gave_up"
    st = sup.sup.status()
    assert st["gave_up"] is True and "image dgxmm-decider:3 not found" in st["last_reason"]
    sup.clock.t += 1000
    assert await sup.sup.tick() == "gave_up"  # no more retries (and no more failed events) until re-enabled
    sup.sup.reset()  # what enable (the Retry action) does
    assert sup.sup.status()["gave_up"] is False


# ---------- plans: the decider's memory ----------

async def test_preview_that_does_not_fit_names_the_decider_memory(d3):  # noqa: F811
    view = await deploy_decider(d3)
    d3.http.healthy.add(view["port"])
    assert await d3.deployer.wait_ready(view["id"], 5) == "ready"
    d3.probe.available = 20 * GIB
    out = await d3.deployer.preview("vllm", "org/big-model", {"max_model_len": 4096})
    assert out["fit"]["fits"] is False
    assert out["notes"] == ["The decision model holds 13 GiB — disable it on the Decision model page to free it"]
    assert not any(a.get("deployment_id") == view["id"] for p in out["plans"] for a in p["actions"])
    d3.probe.available = 500 * GIB
    out = await d3.deployer.preview("vllm", "Qwen/Qwen3-Tiny", {"max_model_len": 4096})
    assert out["fit"]["fits"] is True and out.get("notes", []) == []


async def test_preview_without_a_running_decider_has_no_note(d3):  # noqa: F811
    d3.probe.available = 20 * GIB
    out = await d3.deployer.preview("vllm", "org/big-model", {"max_model_len": 4096})
    assert out["fit"]["fits"] is False and out["notes"] == []


# ---------- a blob store symlinked outside the HF cache ----------

async def test_blob_store_symlinked_outside_the_cache_is_a_clear_deploy_error(stack, tmp_path):
    root, _snap = _xet_layout(stack, NANO, DECIDER_MODELS[NANO].revision,
                              {"config.json": json.dumps(D3_CONFIG).encode(), "model.safetensors": b"w" * 64,
                               "d3_server.py": b"print()"})
    outside = tmp_path / "elsewhere" / "blobs"
    outside.parent.mkdir()
    shutil.move(str(root / "blobs"), str(outside))
    os.symlink(outside, root / "blobs")
    stack.inv.refresh()
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(DeployRequest(name="decider", engine="decision", model_ref=NANO, config={},
                                                  priority=100), 1)
    assert e.value.code == "invalid_mounts" and e.value.status == 422
    assert "hub/blobs" in e.value.message and "outside the HF cache" in e.value.message
    v = next(x for x in stack.deployer.list_views() if x["engine"] == "decision")
    assert v["state"] == "failed" and "outside the HF cache" in (v["last_error"] or "")


# ---------- the Diagnosis card remembers the admin's vote ----------

def test_latest_diagnosis_carries_the_feedback_given_on_its_decision(db):
    from types import SimpleNamespace

    from dgxmm.db import DecisionRecord, Deployment, DiagnosisRow
    from dgxmm.decider.diagnosis import DiagnosisService
    with db.session() as s:
        for i in (77, 78):
            s.add(Deployment(id=i, name=f"d{i}", engine="vllm", model_ref="org/m", state="failed"))
        s.flush()
        rec = DecisionRecord(feature="diagnosis", source="rules", choice="port_in_use", feedback="rejected")
        s.add(rec)
        s.flush()
        s.add(DiagnosisRow(deployment_id=77, cause="port_in_use", source="rules", confidence=1.0, summary="s",
                           fix_hint="h", patch=None, log_excerpt="log", decision_id=rec.id))
        s.add(DiagnosisRow(deployment_id=78, cause="port_in_use", source="rules", confidence=1.0, summary="s",
                           fix_hint="h", patch=None, log_excerpt="log"))
    svc = DiagnosisService(SimpleNamespace(db=db, settings=None))
    assert svc.latest(77)["feedback"] == "rejected"
    assert svc.latest(78)["feedback"] is None
