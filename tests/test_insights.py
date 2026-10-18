"""Insights feed (dgxmm/decider/insights.py, dgxmm/api/insights.py): dedupe, resolve, user actions, retention,
deletion with the deployment, action descriptor validation, the API's role / CSRF / rate-limit / size checks, the
SSE topic, and the schema upgrade from a main-era database."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from dgxmm.db import Database, Insight, utcnow
from dgxmm.decider.insights import (HEALTH_CLEAR_GRACE, InsightClosed, InsightService, deployment_subject,
                                    validate_action)
from dgxmm.events import EventBroker


class Clock:
    def __init__(self):
        self.now = utcnow()

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def svc(db):
    clock = Clock()
    events = EventBroker()
    q = events.subscribe()
    s = InsightService(db, events, clock=clock)
    s.clock_ = clock
    s.q = q
    return s


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def raise_kv(svc, dep=1, severity="warning", title="KV cache pressure", **kw):
    return svc.upsert("health", "health", deployment_subject(dep), severity, title, "templated body",
                      dedupe_key=f"health:kv:{dep}", source="rules", **kw)


# ---------------- service ----------------
async def test_upsert_updates_one_open_insight_in_place(svc):
    a = raise_kv(svc)
    b = raise_kv(svc, title="KV cache pressure (worse)", confidence=0.9)
    assert a.id == b.id
    rows = svc.list()
    assert len(rows) == 1 and rows[0]["title"] == "KV cache pressure (worse)" and rows[0]["confidence"] == 0.9
    assert drain(svc.q) == [("insights", {"id": a.id}), ("insights", {"id": a.id})]


async def test_unchanged_upsert_publishes_nothing(svc):
    raise_kv(svc)
    drain(svc.q)
    raise_kv(svc)
    assert drain(svc.q) == []


async def test_dismissed_stays_dismissed_until_resolved_unless_escalated(svc):
    a = raise_kv(svc)
    svc.dismiss(a.id)
    assert raise_kv(svc).status == "dismissed" and len(svc.list()) == 1
    c = raise_kv(svc, severity="critical")
    assert c.id == a.id and c.status == "open"
    svc.dismiss(a.id)
    assert svc.resolve("health:kv:1") == 1
    d = raise_kv(svc)
    assert d.id != a.id and d.status == "open"
    assert [r["status"] for r in svc.list()] == ["open", "expired"]


async def test_resolve_with_grace_keeps_it_visible_then_expires(svc):
    a = raise_kv(svc)
    svc.resolve("health:kv:1", grace=HEALTH_CLEAR_GRACE)
    assert svc.list()[0]["status"] == "open" and svc.list()[0]["expires_at"]
    svc.clock_.advance(minutes=30)
    raise_kv(svc)  # the condition came back: the pending expiry is cancelled
    assert svc.get(a.id).expires_at is None
    svc.resolve("health:kv:1", grace=HEALTH_CLEAR_GRACE)
    svc.clock_.advance(minutes=61)
    assert svc.list()[0]["status"] == "expired"
    assert svc.resolve("health:kv:1") == 0


async def test_snooze_hides_until_it_ends(svc):
    a = raise_kv(svc)
    svc.snooze(a.id, 2)
    assert svc.list(status=["open"]) == []
    raise_kv(svc, title="still there")  # updated in place, still snoozed
    assert svc.get(a.id).status == "snoozed"
    svc.clock_.advance(hours=3)
    assert [r["id"] for r in svc.list(status=["open"])] == [a.id]
    with pytest.raises(ValueError):
        svc.snooze(a.id, 0)
    with pytest.raises(ValueError):
        svc.snooze(a.id, 169)


async def test_applied_closes_and_next_upsert_opens_a_new_one(svc):
    a = raise_kv(svc)
    assert svc.mark_applied(a.id).status == "applied"
    with pytest.raises(InsightClosed):
        svc.dismiss(a.id)
    assert raise_kv(svc).id != a.id
    assert svc.dismiss(999) is None


async def test_expires_at_and_retention(svc, db):
    svc.upsert("proposal", "schedule", deployment_subject(2), "info", "Sleep at night?", "b", dedupe_key="p:2",
               source="model", confidence=0.91, expires_at=svc.clock_.now + timedelta(days=7))
    a = raise_kv(svc)
    svc.dismiss(a.id)
    svc.clock_.advance(days=8)
    out = svc.prune()
    assert out["insights_expired"] == 1 and out["insights_closed"] == 0
    svc.clock_.advance(days=31)
    assert svc.prune()["insights_closed"] == 2
    with db.session() as s:
        assert s.scalars(select(Insight)).all() == []


async def test_deleted_with_the_deployment(svc):
    raise_kv(svc, dep=1)
    raise_kv(svc, dep=2)
    svc.upsert("routing", "routing", "routing", "info", "Route code to x", "b", dedupe_key="routing:code",
               source="model", confidence=0.9, action={"type": "reconfigure", "deployment_id": 1,
                                                        "patch": {"enable_prefix_caching": True}})
    assert svc.on_deployment_removed(1) == 2
    assert [r["subject"] for r in svc.list()] == ["deployment:2"]


async def test_publish_from_a_worker_thread_goes_through_the_loop(svc):
    svc.bind_loop(asyncio.get_running_loop())
    a = await asyncio.to_thread(raise_kv, svc)
    await asyncio.sleep(0.01)
    assert drain(svc.q) == [("insights", {"id": a.id})]


@pytest.mark.parametrize("bad", [
    dict(kind="Bad Kind"), dict(feature="nope"), dict(severity="fatal"), dict(source="llm"), dict(title=""),
    dict(title="x" * 257), dict(body="x" * 4001), dict(dedupe_key=""), dict(confidence=1.5),
    dict(confidence=float("nan")), dict(action={"type": "exec", "cmd": "rm -rf /"}),
])
async def test_upsert_rejects_invalid_input(svc, bad):
    args = dict(kind="health", feature="health", subject="deployment:1", severity="warning", title="t", body="b",
                dedupe_key="k", source="rules")
    args.update(bad)
    with pytest.raises(ValueError):
        svc.upsert(args.pop("kind"), args.pop("feature"), args.pop("subject"), args.pop("severity"),
                   args.pop("title"), args.pop("body"), **args)


# ---------------- action descriptors ----------------
@pytest.mark.parametrize("action", [
    {"type": "reconfigure", "deployment_id": 3, "patch": {"kv_cache_dtype": "fp8", "max_num_seqs": 64}},
    {"type": "schedule", "deployment_id": 3, "proposal": {"action": "sleep", "days": 31, "at_local": "22:00"}},
    {"type": "threshold", "feature": "plan", "value": 0.7},
    {"type": "copy", "text": "model_list:\n  - model_name: dgxmm-code", "label": "Copy snippet"},
    {"type": "open", "path": "/litellm"},
])
def test_valid_actions(action):
    assert validate_action(action) == action


@pytest.mark.parametrize("action", [
    {"type": "exec"}, {"type": "reconfigure", "deployment_id": 0, "patch": {"a": 1}},
    {"type": "reconfigure", "deployment_id": 1, "patch": {}},
    {"type": "reconfigure", "deployment_id": "1", "patch": {"a": 1}},
    {"type": "reconfigure", "deployment_id": 1, "patch": {"bad key": 1}},
    {"type": "reconfigure", "deployment_id": 1, "patch": {"a": 1}, "extra": True},
    {"type": "threshold", "feature": "parser_tool", "value": 0.7}, {"type": "threshold", "feature": "plan", "value": 2},
    {"type": "threshold", "feature": "plan", "value": True},
    {"type": "copy", "text": ""}, {"type": "copy", "text": "x" * 5000},
    {"type": "open", "path": "https://evil.example/"}, {"type": "open", "path": "//evil.example"},
    {"type": "open", "path": "/x\" onclick=\"y"}, {"type": "open", "path": "javascript:alert(1)"},
    {"type": "schedule", "deployment_id": 1, "proposal": {"x": float("inf")}},
    "reconfigure", [], {"type": "copy", "text": "x", "label": ""},
])
def test_invalid_actions(action):
    with pytest.raises(ValueError):
        validate_action(action)


# ---------------- API ----------------
B = "/dgxmm/api/insights"


@pytest.fixture
def api(client, admin_csrf, make_user, ctx):
    client.h = {"X-CSRF-Token": admin_csrf}
    client.ctx = ctx
    make_user("viewer", "viewer-password-123", "viewer")
    return client


def as_viewer(c):
    c.cookies.clear()
    r = c.post("/dgxmm/api/auth/login", json={"username": "viewer", "password": "viewer-password-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def as_admin(c):
    c.cookies.clear()
    r = c.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def seed(ctx):
    a = ctx.insights.upsert("health", "health", "deployment:1", "warning", "KV cache pressure", "b",
                            dedupe_key="h:1", source="rules",
                            action={"type": "reconfigure", "deployment_id": 1, "patch": {"kv_cache_dtype": "fp8"}})
    b = ctx.insights.upsert("routing", "routing", "routing", "info", "Route code", "b", dedupe_key="r:code",
                            source="model", confidence=0.93, action={"type": "copy", "text": "snippet"})
    return a, b


def test_list_filters_and_viewer_view(api):
    a, b = seed(api.ctx)
    rows = api.get(B, headers=api.h).json()
    assert [r["id"] for r in rows] == [a.id, b.id] and rows[0]["action"]["type"] == "reconfigure"
    assert [r["id"] for r in api.get(f"{B}?subject=deployment:1").json()] == [a.id]
    assert [r["id"] for r in api.get(f"{B}?kind=routing").json()] == [b.id]
    assert api.get(f"{B}?status=dismissed").json() == []
    assert api.get(f"{B}?status=bogus").status_code == 422
    as_viewer(api)
    rows = api.get(B).json()  # routing is admin-only (viewer_visible=False); no actions for viewers
    assert [r["id"] for r in rows] == [a.id] and rows[0]["action"] is None


def test_actions_need_admin_and_csrf_and_are_audited(api):
    from dgxmm.db import AuditLog
    a, _ = seed(api.ctx)
    vh = as_viewer(api)
    assert api.post(f"{B}/{a.id}/dismiss", headers=vh).status_code == 403
    h = as_admin(api)
    assert api.post(f"{B}/{a.id}/dismiss").status_code == 403  # no CSRF header
    r = api.post(f"{B}/{a.id}/snooze", headers=h, json={"hours": 4})
    assert r.status_code == 200 and r.json()["status"] == "snoozed"
    assert api.post(f"{B}/{a.id}/snooze", headers=h, json={"hours": 500}).status_code == 422
    assert api.post(f"{B}/{a.id}/snooze", headers=h, json={"hours": 2, "x": 1}).status_code == 422
    assert api.post(f"{B}/{a.id}/applied", headers=h).json()["status"] == "applied"
    assert api.post(f"{B}/{a.id}/dismiss", headers=h).status_code == 409
    assert api.post(f"{B}/9999/dismiss", headers=h).status_code == 404
    with api.ctx.db.session() as s:
        acts = [r.action for r in s.scalars(select(AuditLog))]
    assert "insight_snooze" in acts and "insight_applied" in acts


def test_anonymous_gets_401(api):
    api.cookies.clear()
    assert api.get(B).status_code == 401
    assert api.post(f"{B}/1/dismiss").status_code == 401


def test_body_cap(api):
    a, _ = seed(api.ctx)
    r = api.post(f"{B}/{a.id}/snooze", headers=api.h, content=b'{"hours": 1, "pad": "' + b"x" * 9000 + b'"}',
                 )
    assert r.status_code == 413


def test_rate_limited(api):
    from dgxmm.api.ratelimit import RATE_LIMITS
    a, _ = seed(api.ctx)
    n = RATE_LIMITS["insights_write"][0]
    codes = [api.post(f"{B}/{a.id}/dismiss", headers=api.h).status_code for _ in range(n + 1)]
    assert codes[-1] == 429 and 429 not in codes[:-1]


def test_insights_topic_is_viewer_visible():
    from dgxmm.api.stream import ADMIN_ONLY_TOPICS, VIEWER_TOPICS, visible
    assert "insights" in VIEWER_TOPICS and "insights" not in ADMIN_ONLY_TOPICS and visible("insights", False)


# ---------------- schema upgrade ----------------
def test_upgrade_from_a_main_era_database(tmp_path):
    """A database created by main (7c055b7, schema_version 1-2) gains the insights table and migration 3, keeps its
    data, and a second start is a no-op."""
    fix = Path(__file__).resolve().parent / "fixtures" / "db" / "main-7c055b7.schema.sql"
    path = tmp_path / "main.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(fix.read_text())
    assert "insights" not in {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.execute("INSERT INTO schema_version VALUES (1, 'baseline', '2026-10-17'), (2, 'diagnoses', '2026-10-17')")
    con.execute("INSERT INTO app_settings (key, value, updated_at) VALUES ('decider.enabled', 'false', "
                "'2026-10-17 00:00:00')")
    con.commit()
    con.close()
    for _ in range(2):
        db = Database(f"sqlite:///{path}")
        db.engine.dispose()
    con = sqlite3.connect(path)
    from dgxmm import migrations
    assert [r[0] for r in con.execute("SELECT version FROM schema_version ORDER BY version")] == \
        [m.version for m in migrations.MIGRATIONS]  # 1-3 plus the feature lanes' steps
    idx = {r[1]: r[2] for r in con.execute("PRAGMA index_list(insights)")}
    assert idx.get("uq_insights_live_dedupe") == 1
    assert con.execute("SELECT value FROM app_settings WHERE key='decider.enabled'").fetchone() == ("false",)
    con.close()
    db = Database(f"sqlite:///{path}")
    InsightService(db).upsert("health", "health", "deployment:1", "info", "t", "b", dedupe_key="k", source="rules")
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):  # the database enforces one live insight per key
        with db.session() as s:
            s.add(Insight(kind="health", feature="health", subject="", severity="info", title="t", body="",
                          source="rules", dedupe_key="k", status="open"))
    db.engine.dispose()


def test_migration_3_collapses_duplicates_of_a_table_without_the_index(tmp_path):
    from dgxmm import migrations
    path = tmp_path / "dup.sqlite3"
    Database(f"sqlite:///{path}").engine.dispose()
    con = sqlite3.connect(path)
    con.execute("DROP INDEX uq_insights_live_dedupe")
    for i in (1, 2):
        con.execute("INSERT INTO insights (id, kind, feature, subject, severity, title, body, source, status, "
                    "dedupe_key, created_at, updated_at) VALUES (?, 'health', 'health', '', 'info', 't', '', "
                    "'rules', 'open', 'k', '2026-10-17', '2026-10-17')", (i,))
    con.execute("DELETE FROM schema_version WHERE version = 3")
    con.commit()
    con.close()
    db = Database(f"sqlite:///{path}")
    with db.session() as s:
        assert [(r.id, r.status) for r in s.scalars(select(Insight).order_by(Insight.id))] == [(1, "expired"),
                                                                                              (2, "open")]
    db.engine.dispose()
    assert migrations.LATEST >= 3


async def test_concurrent_upserts_of_one_key_converge(svc):
    import threading
    barrier = threading.Barrier(4)

    def one(i):
        barrier.wait()
        return raise_kv(svc, title=f"KV cache pressure {i}").id
    ids = await asyncio.gather(*(asyncio.to_thread(one, i) for i in range(4)))
    assert len(set(ids)) == 1 and len(svc.list()) == 1


# ---------------- fix round 1 ----------------
async def test_severity_rise_reopens_a_snoozed_insight(svc):
    a = raise_kv(svc)
    svc.snooze(a.id, 168)
    assert raise_kv(svc).status == "snoozed"  # same severity: stays snoozed
    b = raise_kv(svc, severity="critical")
    assert b.id == a.id and b.status == "open" and b.snooze_until is None
    assert [r["id"] for r in svc.list(status=["open"])] == [a.id]


async def test_a_dismissed_insight_whose_condition_persists_survives_retention(svc):
    a = raise_kv(svc)
    svc.dismiss(a.id)
    for _ in range(8):  # the condition is re-reported every few days for 40 days
        svc.clock_.advance(days=5)
        raise_kv(svc)
        svc.prune()
    assert svc.get(a.id).status == "dismissed"  # never pruned, never resurfaced as a new open insight
    assert len(svc.list()) == 1
    svc.clock_.advance(days=31)  # no longer reported for 30 days: pruned
    svc.prune()
    assert svc.list() == []


@pytest.mark.parametrize("key", ["model_ref", "trust_remote_code"])
def test_reconfigure_patches_cannot_switch_the_model_or_enable_remote_code(key):
    with pytest.raises(ValueError):
        validate_action({"type": "reconfigure", "deployment_id": 1, "patch": {key: True}})


def test_list_route_runs_off_the_event_loop(api, monkeypatch):
    import threading
    seen = []
    real = api.ctx.insights.list

    def spy(**kw):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(**kw)
    monkeypatch.setattr(api.ctx.insights, "list", spy)
    from dgxmm.decider import insights as mod
    on_loop = []
    real_publish = mod.InsightService._publish

    def where(self, ids):
        try:
            _ = __import__("asyncio").get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real_publish(self, ids)
    assert api.get(B, headers=api.h).status_code == 200
    assert seen == [False]  # list() ran in a worker thread
    monkeypatch.setattr(mod.InsightService, "_publish", where)
    a, _ = seed(api.ctx)
    on_loop.clear()
    assert api.post(f"{B}/{a.id}/dismiss", headers=api.h).status_code == 200
    assert on_loop == [False]  # the action's DB work ran in a worker thread too


def test_the_client_rejects_the_same_patch_keys_as_the_server():
    """Security review L6: frontend parseAction mirrors FORBIDDEN_PATCH_KEYS."""
    import re
    from pathlib import Path

    from dgxmm.decider.insights import FORBIDDEN_PATCH_KEYS
    ts = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "insights.ts").read_text()
    m = re.search(r"FORBIDDEN_PATCH_KEYS = new Set\(\[([^\]]*)\]\)", ts)
    assert m and set(re.findall(r'"([^"]+)"', m.group(1))) == set(FORBIDDEN_PATCH_KEYS)
