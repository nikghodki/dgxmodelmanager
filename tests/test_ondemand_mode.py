"""On-demand models, part 1: the deployment mode (on_demand | always_on), its idle timeout, settings, migration 40,
the PATCH /mode API and the installer-generated gateway key (docs/superpowers/plans/2026-10-24-on-demand-models.md)."""
from __future__ import annotations

import logging
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from dgxmm import migrations
from dgxmm.config import Settings, load_ondemand_key
from dgxmm.db import AuditLog, Database
from dgxmm.deployer.service import DeployError, DeployRequest
from tests.test_api_deployments import admin
from tests.test_decider_engine import d3, deploy_decider  # noqa: F401  (d3 is a fixture)
from tests.test_install_script import FIRST_RUN, Box, _copy_litellm_files, mode as file_mode
from tests.test_migrations import _make, _versions

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
TINY = {"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}
IDLE_STATUS = {"inflight": 0, "queued": 0, "last_request_at": None, "phase": "idle"}


def _viewer(c) -> dict:
    r = c.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


# --- settings and request model -------------------------------------------------------------------

def test_settings_defaults_and_bounds():
    s = Settings()
    assert (s.ondemand_idle_timeout_s, s.ondemand_queue_timeout_s) == (300, 900)
    assert s.ondemand_key_file == "/run/secrets/dgxmm_ondemand_key"
    for bad in ({"ondemand_idle_timeout_s": 59}, {"ondemand_idle_timeout_s": 86401},
                {"ondemand_queue_timeout_s": 3601}):
        with pytest.raises(ValidationError):
            Settings(**bad)


def test_queue_timeout_below_300_is_clamped_with_a_warning(caplog):
    """The queue limit includes the model's start-up (about 110 s for a small model), so below 300 s every cold
    start would fail. An older config (minimum was 60) still loads: the value is raised to 300 with a warning."""
    for low in (60, 299, 0):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert Settings(ondemand_queue_timeout_s=low).ondemand_queue_timeout_s == 300
        assert any("ondemand_queue_timeout_s" in r.getMessage() and "300" in r.getMessage()
                   for r in caplog.records), low
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert Settings(ondemand_queue_timeout_s=300).ondemand_queue_timeout_s == 300
    assert not caplog.records


def test_load_ondemand_key(tmp_path):
    path = tmp_path / "dgxmm_ondemand_key"
    s = Settings(ondemand_key_file=str(path))
    with pytest.raises(RuntimeError):
        load_ondemand_key(s)  # missing: fail closed, never an empty key
    path.write_text("")
    with pytest.raises(RuntimeError):
        load_ondemand_key(s)
    path.write_text("sk-abc123\n")
    assert load_ondemand_key(s) == "sk-abc123"


def test_idle_timeout_bounds(api_stack):
    assert DeployRequest(**TINY).idle_timeout_s is None
    assert DeployRequest(**TINY, idle_timeout_s=60).idle_timeout_s == 60
    assert DeployRequest(**TINY, idle_timeout_s=86400).idle_timeout_s == 86400
    for bad in (59, 86401):
        with pytest.raises(ValidationError):
            DeployRequest(**TINY, idle_timeout_s=bad)
    with pytest.raises(ValidationError):
        DeployRequest(**TINY, mode="sometimes")
    h = admin(api_stack)
    for bad in (59, 86401):
        r = api_stack.post("/dgxmm/api/deployments", headers=h, json={**TINY, "idle_timeout_s": bad})
        assert r.status_code == 422, r.text
    assert api_stack.stack.sdk.run_calls == []
    dep = api_stack.post("/dgxmm/api/deployments", headers=h, json=TINY).json()
    for bad in (59, 86401):
        r = api_stack.patch(f"/dgxmm/api/deployments/{dep['id']}/mode", headers=h,
                            json={"mode": "on_demand", "idle_timeout_s": bad})
        assert r.status_code == 422, r.text
    r = api_stack.patch(f"/dgxmm/api/deployments/{dep['id']}/mode", headers=h, json={"mode": "sometimes"})
    assert r.status_code == 422, r.text


# --- deploy ---------------------------------------------------------------------------------------

def test_new_deploy_defaults_always_on(api_stack):
    """The API default is always on: on demand needs LiteLLM (it starts models on requests that come through it), so
    a script that deploys without a mode keeps the old behaviour and gets on demand only when it asks for it."""
    assert DeployRequest(name="x", engine="vllm", model_ref="org/x").mode == "always_on"
    h = admin(api_stack)
    r = api_stack.post("/dgxmm/api/deployments", headers=h, json={**TINY, "name": "always", "alias": "tiny-always"})
    assert r.status_code == 201, r.text
    always = r.json()
    assert always["mode"] == "always_on" and always["idle_timeout_s"] is None and always["pending_mode"] is None
    assert always["ondemand"] is None  # not managed by the controller
    r = api_stack.post("/dgxmm/api/deployments", headers=h, json={**TINY, "mode": "on_demand", "idle_timeout_s": 600})
    assert r.status_code == 201, r.text
    dep = r.json()
    assert dep["mode"] == "on_demand" and dep["idle_timeout_s"] == 600
    assert dep["ondemand"] == IDLE_STATUS  # filled by the controller (ctx.ondemand.status)
    listed = {v["name"]: v for v in api_stack.get("/dgxmm/api/deployments").json()}
    assert listed["tiny"]["mode"] == "on_demand" and listed["always"]["mode"] == "always_on"


async def test_system_deployment_forced_always_on(d3):  # noqa: F811
    from dgxmm.engines.decision import DECIDER_REPO
    dep = await d3.deployer.deploy(DeployRequest(name="decider", engine="decision", model_ref=DECIDER_REPO, config={},
                                                 priority=100, mode="on_demand", idle_timeout_s=600), 1)
    assert dep["system"] is True and dep["mode"] == "always_on" and dep["idle_timeout_s"] is None
    with pytest.raises(DeployError) as e:
        d3.deployer.set_mode(dep["id"], "on_demand", None)
    assert e.value.code == "system_deployment" and e.value.status == 409
    assert d3.deployer.view(dep["id"])["mode"] == "always_on"


async def test_system_deployment_mode_patch_refused(api_stack, fake_hf_repo):
    from tests.test_decider_engine import D3_CONFIG
    from dgxmm.config import GIB
    from dgxmm.engines.decision import DECIDER_REPO, DECIDER_REVISION
    fake_hf_repo(DECIDER_REPO, rev=DECIDER_REVISION, config=D3_CONFIG,
                 files={"model.safetensors": 5 * GIB, "d3_server.py": 100})
    api_stack.stack.inv.refresh()
    h = admin(api_stack)
    dep = await deploy_decider(api_stack.stack)
    r = api_stack.patch(f"/dgxmm/api/deployments/{dep['id']}/mode", headers=h, json={"mode": "on_demand"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "system_deployment"
    assert api_stack.get(f"/dgxmm/api/deployments/{dep['id']}").json()["mode"] == "always_on"


async def test_switch_keeps_the_old_mode_unless_given(stack):
    stack.http.all_healthy = True
    a = await stack.deployer.deploy(DeployRequest(**{**TINY, "name": "a"}, mode="always_on", idle_timeout_s=900), 1)
    assert await stack.deployer.wait_ready(a["id"], 5) == "ready"
    b = await stack.deployer.switch(a["id"], DeployRequest(**{**TINY, "name": "b"}), [], 1)
    assert b["mode"] == "always_on" and b["idle_timeout_s"] == 900
    c = await stack.deployer.switch(b["id"], DeployRequest(**{**TINY, "name": "c"}, mode="on_demand"), [], 1)
    assert c["mode"] == "on_demand" and c["idle_timeout_s"] is None


# --- PATCH /mode ----------------------------------------------------------------------------------

def test_patch_mode_admin_only_csrf_audited(api_stack):
    h = admin(api_stack)
    dep = api_stack.post("/dgxmm/api/deployments", headers=h, json={**TINY, "mode": "on_demand"}).json()
    url = f"/dgxmm/api/deployments/{dep['id']}/mode"
    body = {"mode": "always_on", "idle_timeout_s": 600}
    assert api_stack.patch(url, json=body).status_code == 403  # admin session without the CSRF header
    api_stack.cookies.clear()
    hv = _viewer(api_stack)
    assert api_stack.patch(url, headers=hv, json=body).status_code == 403  # viewer
    api_stack.cookies.clear()
    h = {"X-CSRF-Token": api_stack.post("/dgxmm/api/auth/login",
                                        json={"username": "root", "password": "root-password-123"}).json()["csrf_token"]}
    assert api_stack.get(url.removesuffix("/mode")).json()["mode"] == "on_demand"  # unchanged so far
    r = api_stack.patch(url, headers=h, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "always_on" and r.json()["idle_timeout_s"] == 600
    # idle_timeout_s omitted: kept; null: back to the global default
    r = api_stack.patch(url, headers=h, json={"mode": "on_demand"})
    assert r.json()["mode"] == "on_demand" and r.json()["idle_timeout_s"] == 600
    r = api_stack.patch(url, headers=h, json={"mode": "on_demand", "idle_timeout_s": None})
    assert r.json()["idle_timeout_s"] is None and r.json()["ondemand"] == IDLE_STATUS
    assert api_stack.patch("/dgxmm/api/deployments/999/mode", headers=h, json=body).status_code == 404
    with api_stack.ctx.db.session() as s:
        rows = s.scalars(select(AuditLog).where(AuditLog.action == "ondemand_mode").order_by(AuditLog.id)).all()
        assert [(r.target, r.detail) for r in rows] == [
            ("tiny", {"from": {"mode": "on_demand", "idle_timeout_s": None},
                      "to": {"mode": "always_on", "idle_timeout_s": 600}}),
            ("tiny", {"from": {"mode": "always_on", "idle_timeout_s": 600},
                      "to": {"mode": "on_demand", "idle_timeout_s": 600}}),
            ("tiny", {"from": {"mode": "on_demand", "idle_timeout_s": 600},
                      "to": {"mode": "on_demand", "idle_timeout_s": None}}),
        ]
        assert all(r.user_id is not None for r in rows)


def test_patch_mode_goes_through_the_controller(api_stack):
    h = admin(api_stack)
    dep = api_stack.post("/dgxmm/api/deployments", headers=h, json=TINY).json()
    calls = []
    real = api_stack.ctx.ondemand.set_mode

    async def spy(dep_id, mode, idle_timeout_s, user_id):
        calls.append((dep_id, mode, idle_timeout_s, user_id))
        return await real(dep_id, mode, idle_timeout_s, user_id)
    api_stack.ctx.ondemand.set_mode = spy
    r = api_stack.patch(f"/dgxmm/api/deployments/{dep['id']}/mode", headers=h, json={"mode": "always_on"})
    assert r.status_code == 200, r.text
    assert calls == [(dep["id"], "always_on", None, calls[0][3])] and calls[0][3] is not None


# --- migration 40 ---------------------------------------------------------------------------------

def test_existing_rows_migrate_always_on(tmp_path):
    path = _make(tmp_path / "main.sqlite3", "main-7c055b7")
    con = sqlite3.connect(path)
    con.execute("INSERT INTO deployments (id, name, engine, model_ref, config, state, priority, reserved_bytes, "
                "estimate, litellm_status, created_at, updated_at) VALUES (2, 'r', 'vllm', 'Qwen/Qwen3-0.6B', '{}', "
                "'ready', 0, 0, '{}', 'none', '2026-10-01 00:00:00', '2026-10-01 00:00:00')")
    con.commit()
    cols = {r[1] for r in con.execute("PRAGMA table_info(deployments)")}
    con.close()
    assert "mode" not in cols and "idle_timeout_s" not in cols  # a main-era database
    for _ in range(2):  # idempotent: the second start changes nothing
        db = Database(f"sqlite:///{path}")
        db.engine.dispose()
        con = sqlite3.connect(path)
        assert con.execute("SELECT name, mode, idle_timeout_s FROM deployments ORDER BY id").fetchall() == [
            ("q8", "always_on", None), ("r", "always_on", None)]
        con.close()
    assert 40 in _versions(path) and migrations.LATEST >= 40
    m = next(m for m in migrations.MIGRATIONS if m.version == 40)
    assert m.name == "deployments: on-demand mode"


def test_migration_40_adds_missing_columns_and_is_idempotent(tmp_path):
    path = _make(tmp_path / "raw.sqlite3", "main-7c055b7")
    con = sqlite3.connect(path, isolation_level=None)
    step = next(m for m in migrations.MIGRATIONS if m.version == 40).apply
    for _ in range(2):  # on its own (no add_missing_columns before it), twice
        step(con)
        assert con.execute("SELECT mode, idle_timeout_s FROM deployments").fetchall() == [("always_on", None)]
    con.execute("UPDATE deployments SET mode = 'on_demand'")
    step(con)  # a value that is set is never rewritten
    assert con.execute("SELECT mode FROM deployments").fetchall() == [("on_demand",)]
    con.close()


def test_fresh_rows_without_a_mode_read_as_always_on(db):
    """Rows written by an older DGXMM after the upgrade (it does not know the column) have mode NULL: they are
    treated as always_on (nothing changes for them until switched)."""
    from dgxmm.db import Deployment
    with db.session() as s:
        d = Deployment(name="old", engine="vllm", model_ref="Qwen/Qwen3-0.6B", config={}, state="stopped")
        s.add(d)
        s.flush()
        assert d.mode == "always_on"
        s.execute(Deployment.__table__.update().values(mode=None))
    with db.session() as s:
        assert s.scalar(select(Deployment.mode)) is None


# --- installer ------------------------------------------------------------------------------------

def test_installer_generates_ondemand_key(tmp_path):
    box = Box(tmp_path)
    p = box.run(FAKE_LOGS=FIRST_RUN)
    assert p.returncode == 0, p.out
    key_file = box.repo / "secrets" / "dgxmm_ondemand_key"
    assert key_file.is_file() and file_mode(key_file) == 0o600
    key = key_file.read_text()
    assert len(key) >= 32 and key not in p.out
    p2 = box.run(FAKE_LOGS="", FAKE_NETWORK="exists", FAKE_DGXMM_STATE="running",
                 FAKE_DGXMM_WORKDIR=str(box.repo.resolve()))
    assert p2.returncode == 0, p2.out
    assert key_file.read_text() == key and file_mode(key_file) == 0o600  # never overwritten on rerun
    assert "kept existing secrets/dgxmm_ondemand_key" in p2.out


def test_installer_generates_ondemand_key_with_bundled_litellm(tmp_path):
    box = Box(tmp_path)
    _copy_litellm_files(box)
    p = box.run("--with-litellm", FAKE_LOGS=FIRST_RUN)
    assert p.returncode == 0, p.out
    key_file = box.repo / "secrets" / "dgxmm_ondemand_key"
    assert key_file.is_file() and file_mode(key_file) == 0o600 and key_file.read_text()


def test_installer_dry_run_writes_no_ondemand_key(tmp_path):
    box = Box(tmp_path)
    p = box.run("--dry-run")
    assert p.returncode == 0, p.out
    assert not (box.repo / "secrets" / "dgxmm_ondemand_key").exists()
    assert "secrets/dgxmm_ondemand_key" in p.out


def test_compose_mounts_the_ondemand_key():
    import yaml
    from tests.test_install_script import ROOT
    compose = yaml.safe_load((ROOT / "docker-compose.yaml").read_text())
    assert "dgxmm_ondemand_key" in compose["services"]["dgxmm"]["secrets"]
    assert compose["secrets"]["dgxmm_ondemand_key"] == {"file": "./secrets/dgxmm_ondemand_key"}


# --- global defaults for the UI -------------------------------------------------------------------
def test_host_payload_exposes_ondemand_defaults_to_viewers(api_stack):
    from dgxmm.api.host import host_payload
    admin(api_stack)
    api_stack.cookies.clear()
    _viewer(api_stack)
    api_stack.ctx.settings.ondemand_idle_timeout_s = 600
    api_stack.ctx.settings.ondemand_queue_timeout_s = 1200
    body = api_stack.get("/dgxmm/api/host").json()
    assert body["ondemand"] == {"idle_timeout_s": 600, "queue_timeout_s": 1200, "litellm_ready": False}
    # the SSE "budget" event carries the same payload (it replaces the UI's host query data)
    assert host_payload(api_stack.ctx, api_stack.ctx.budget.current())["ondemand"] == body["ondemand"]


def test_host_payload_says_whether_litellm_can_start_on_demand_models(api_stack):
    """``ondemand.litellm_ready``: the integration is on and its last sync succeeded (the deploy wizard then
    defaults to on demand). Known already, no LiteLLM call."""
    from dgxmm.api.host import host_payload
    ctx = api_stack.ctx
    h = admin(api_stack)
    assert api_stack.get("/dgxmm/api/host", headers=h).json()["ondemand"]["litellm_ready"] is False
    b = ctx.budget.current()
    for active, last, want in [(True, {"ok": True, "status": "synced"}, True),
                               (True, {"ok": False, "status": "pending"}, False),
                               (True, {"ok": None, "status": None}, False),
                               (False, {"ok": True, "status": "disabled"}, False)]:
        ctx.litellm = SimpleNamespace(active=lambda a=active: a, last=last)
        assert host_payload(ctx, b)["ondemand"]["litellm_ready"] is want, (active, last)
    ctx.litellm = None
    assert host_payload(ctx, b)["ondemand"]["litellm_ready"] is False
