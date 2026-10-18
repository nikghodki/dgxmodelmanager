"""Versioned schema migrations (dgxmm/migrations.py) on top of create_all + add_missing_columns.

Upgrades are tested from the real schema shapes of earlier releases (tests/fixtures/db/*.schema.sql, dumped from
databases created by those commits): main before the decision model (b193e9c), the first decision-model build
(11ce098, diagnoses without the unique constraint) and the current live install (e3ca780). Set
DGXMM_LIVE_DB_COPY to a *copy* of a real database file to also upgrade that copy."""
from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import select

from dgxmm import migrations
from dgxmm.db import BenchmarkRun, Database, Deployment, DiagnosisRow

FIX = Path(__file__).resolve().parent / "fixtures" / "db"
SHAPES = ["main-b193e9c", "poc-11ce098", "live-e3ca780"]


def _make(path: Path, shape: str, rows: bool = True) -> Path:
    con = sqlite3.connect(path)
    con.executescript((FIX / f"{shape}.schema.sql").read_text())
    if rows:
        con.execute("INSERT INTO users (id, username, pw_hash, role, must_change_pw, created_at) "
                    "VALUES (1, 'admin', 'x', 'admin', 0, '2026-10-01 00:00:00')")
        con.execute("INSERT INTO deployments (id, name, engine, model_ref, config, state, priority, reserved_bytes, "
                    "estimate, litellm_status, created_at, updated_at) VALUES (1, 'q8', 'vllm', 'Qwen/Qwen3-8B', '{}', "
                    "'failed', 0, 0, '{}', 'none', '2026-10-01 00:00:00', '2026-10-01 00:00:00')")
        con.execute("INSERT INTO app_settings (key, value, updated_at) VALUES ('litellm.url', 'http://x:4000', "
                    "'2026-10-01 00:00:00')")
    con.commit()
    con.close()
    return path


def _indexes(path: Path, table: str) -> dict[str, tuple[bool, list[str]]]:
    con = sqlite3.connect(path)
    out = {}
    for _, name, unique, *_ in con.execute(f"PRAGMA index_list({table})"):
        cols = [r[2] for r in con.execute(f"PRAGMA index_info({name})")]
        out[name] = (bool(unique), cols)
    con.close()
    return out


def _has_unique(path: Path, table: str, cols: list[str]) -> bool:
    return any(u and c == cols for u, c in _indexes(path, table).values())


def _versions(path: Path) -> list[int]:
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute("SELECT version FROM schema_version ORDER BY version")]
    finally:
        con.close()


def _open(path: Path) -> Database:
    return Database(f"sqlite:///{path}")


def test_migrations_are_ordered_unique_and_named():
    versions = [m.version for m in migrations.MIGRATIONS]
    assert versions == sorted(versions) and len(set(versions)) == len(versions) and versions[0] == 1
    assert all(m.name for m in migrations.MIGRATIONS)
    assert migrations.LATEST == versions[-1]


def test_fresh_database_records_every_version(tmp_path):
    db = _open(tmp_path / "new.sqlite3")
    db.engine.dispose()
    assert _versions(tmp_path / "new.sqlite3") == [m.version for m in migrations.MIGRATIONS]
    assert _has_unique(tmp_path / "new.sqlite3", "diagnoses", ["deployment_id", "event_id"])


@pytest.mark.parametrize("shape", SHAPES)
def test_upgrade_from_earlier_schema_keeps_data_and_is_idempotent(tmp_path, shape):
    path = _make(tmp_path / f"{shape}.sqlite3", shape)
    for _ in range(2):  # the second start is a no-op
        db = _open(path)
        with db.session() as s:
            assert s.scalar(select(DiagnosisRow.id)) is None
        db.engine.dispose()
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("SELECT name, state FROM deployments").fetchall() == [("q8", "failed")]
    assert con.execute("SELECT value FROM app_settings WHERE key = 'litellm.url'").fetchone() == ("http://x:4000",)
    cols = {r[1] for r in con.execute("PRAGMA table_info(deployments)")}
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    con.close()
    assert "system" in cols
    assert {"decisions", "diagnoses", "artifact_kinds", "schema_version"} <= tables
    assert _has_unique(path, "diagnoses", ["deployment_id", "event_id"])
    assert _versions(path) == [m.version for m in migrations.MIGRATIONS]


def test_upgrade_enforces_diagnoses_uniqueness_and_drops_older_duplicates(tmp_path):
    """The first decision-model build had no unique (deployment_id, event_id): duplicates may exist."""
    path = _make(tmp_path / "poc.sqlite3", "poc-11ce098")
    con = sqlite3.connect(path)
    for i, (event, summary) in enumerate([(7, "old"), (7, "new"), (8, "other"), (None, "a"), (None, "b")], start=1):
        con.execute("INSERT INTO diagnoses (id, deployment_id, event_id, created_at, source, summary, fix_hint, "
                    "log_excerpt) VALUES (?, 1, ?, '2026-10-16 00:00:00', 'rules', ?, '', '')", (i, event, summary))
    con.commit()
    con.close()
    db = _open(path)
    with db.session() as s:
        rows = [(r.event_id, r.summary) for r in s.scalars(select(DiagnosisRow).order_by(DiagnosisRow.id))]
    assert rows == [(7, "new"), (8, "other"), (None, "a"), (None, "b")]  # latest kept; NULL events are distinct
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        with db.session() as s:
            s.add(DiagnosisRow(deployment_id=1, event_id=8, source="rules", summary="dup"))
    db.engine.dispose()


def test_missing_metadata_indexes_are_created_on_existing_tables(tmp_path):
    """create_all only indexes tables it creates: an index declared later must reach existing databases too."""
    path = _make(tmp_path / "m.sqlite3", "main-b193e9c")
    con = sqlite3.connect(path)
    con.execute("DROP INDEX ix_audit_log_ts")
    con.commit()
    con.close()
    _open(path).engine.dispose()
    assert "ix_audit_log_ts" in _indexes(path, "audit_log")


def test_newer_database_is_left_alone(tmp_path, caplog):
    """Rollback: an older DGXMM opening a database a newer one migrated must keep working (and say so)."""
    path = tmp_path / "n.sqlite3"
    _open(path).engine.dispose()
    con = sqlite3.connect(path)
    con.execute("INSERT INTO schema_version (version, name, applied_at) VALUES (999, 'from the future', "
                "'2026-12-01 00:00:00')")
    con.commit()
    con.close()
    with caplog.at_level("WARNING"):
        _open(path).engine.dispose()
    assert 999 in _versions(path)
    assert any("newer DGXMM" in r.message for r in caplog.records)


def test_failed_migration_rolls_back_and_is_retried(tmp_path, monkeypatch):
    path = _make(tmp_path / "f.sqlite3", "main-b193e9c")
    calls = []

    def boom(conn):
        calls.append(1)
        conn.execute("CREATE TABLE half_done (id INTEGER)")
        raise RuntimeError("disk on fire")
    bad = migrations.Migration(migrations.LATEST + 1, "explodes", boom)
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, bad])
    with pytest.raises(RuntimeError, match="disk on fire"):
        _open(path)
    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    con.close()
    assert "half_done" not in tables  # the failed step left nothing behind
    assert bad.version not in _versions(path)
    assert migrations.LATEST in _versions(path)  # earlier steps stay applied
    monkeypatch.setattr(migrations, "MIGRATIONS", [m for m in migrations.MIGRATIONS if m is not bad])
    _open(path).engine.dispose()


def test_concurrent_starts_apply_each_migration_once(tmp_path, monkeypatch):
    """The app and an admin CLI may open the same database at once."""
    path = _make(tmp_path / "c.sqlite3", "main-b193e9c")
    applied = []
    lock = threading.Lock()

    def count(conn):
        with lock:
            applied.append(1)
    extra = migrations.Migration(migrations.LATEST + 1, "counted", count)
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, extra])
    errors = []

    def start():
        try:
            _open(path).engine.dispose()
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)
    threads = [threading.Thread(target=start) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and applied == [1]


@pytest.mark.skipif(not os.environ.get("DGXMM_LIVE_DB_COPY"), reason="set DGXMM_LIVE_DB_COPY to a copy of a real DB")
def test_upgrade_a_copy_of_a_real_database(tmp_path):
    src = Path(os.environ["DGXMM_LIVE_DB_COPY"])
    path = tmp_path / "live.sqlite3"
    shutil.copy2(src, path)  # never the original
    before = sqlite3.connect(path)
    counts = {t: before.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
              for (t,) in before.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    before.close()
    _open(path).engine.dispose()
    after = sqlite3.connect(path)
    assert after.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    for t, n in counts.items():
        if t not in ("diagnoses", "schema_version", "sqlite_sequence"):  # dedupe; the versions/sequence applied
            assert after.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] == n, t
    after.close()
    assert _versions(path) == [m.version for m in migrations.MIGRATIONS]


def test_original_error_survives_a_failing_rollback(tmp_path, monkeypatch):
    path = _make(tmp_path / "r.sqlite3", "main-b193e9c")

    def ends_transaction_then_fails(conn):
        conn.execute("COMMIT")  # leaves no transaction, so ROLLBACK itself raises
        raise RuntimeError("the real cause")
    bad = migrations.Migration(migrations.LATEST + 1, "fails oddly", ends_transaction_then_fails)
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, bad])
    with pytest.raises(RuntimeError, match="the real cause"):
        _open(path)


def _insert_dep(con, dep_id, name):
    con.execute("INSERT INTO deployments (id, name, engine, model_ref, config, state, priority, reserved_bytes, "
                "estimate, litellm_status, created_at, updated_at) VALUES (?, ?, 'vllm', 'Qwen/Qwen3-0.6B', '{}', "
                "'stopped', 0, 0, '{}', 'none', '2026-10-01 00:00:00', '2026-10-01 00:00:00')", (dep_id, name))


def _new_deployment(db, name):
    from dgxmm.db import Deployment
    with db.session() as s:
        d = Deployment(name=name, engine="vllm", model_ref="Qwen/Qwen3-0.6B", config={}, state="stopped")
        s.add(d)
        s.flush()
        return d.id


@pytest.mark.parametrize("shape", [*SHAPES, "main-7c055b7"])
def test_deployment_ids_are_never_reused_after_upgrade_from_a_database_with_the_bug(tmp_path, shape):
    """Live finding: SQLite reuses the highest rowid after a delete, so a new deployment took the id of a removed
    one and showed its benchmark runs (and readings, rollups, insights keyed by the id). After the upgrade the id
    sequence starts above every id still referenced anywhere, and removed ids are never handed out again; rows of
    other tables survive the table rebuild (no cascade) and foreign keys still cascade afterwards."""
    path = _make(tmp_path / f"{shape}.sqlite3", shape)
    con = sqlite3.connect(path)
    _insert_dep(con, 2, "kept")
    con.execute("INSERT INTO deployment_events (deployment_id, ts, from_state, to_state, message) "
                "VALUES (2, '2026-10-01 00:00:00', 'ready', 'stopped', '')")
    # deployment 4 was removed (its rows stayed behind); 5 and 7 only survive in rollups / the decision log
    con.execute("INSERT INTO benchmark_runs (deployment_id, engine, model_ref, config_hash, levels, status, "
                "started_at) VALUES (4, 'vllm', 'Qwen/Qwen3-0.6B', 'h', '[]', 'done', '2026-10-01 00:00:00')")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "metric_rollups" in tables:
        cols = {r[1]: r for r in con.execute("PRAGMA table_info(metric_rollups)")}
        names = [c for c in cols if c != "id"]
        vals = ["5" if c == "deployment_id" else ("0" if cols[c][3] else "NULL") for c in names]  # NOT NULL -> 0
        con.execute(f"INSERT INTO metric_rollups ({', '.join(names)}) VALUES ({', '.join(vals)})")
    want = 6 if "metric_rollups" in tables else 5
    if "decisions" in tables:  # the decision log names deployment 7 too
        cols = {r[1]: r for r in con.execute("PRAGMA table_info(decisions)")}
        vals = {"created_at": "'2026-10-01 00:00:00'", "feature": "'health'", "subject": "'deployment:7'",
                "source": "'rules'", "probabilities": "'{}'"}
        con.execute(f"INSERT INTO decisions ({', '.join(vals)}) VALUES ({', '.join(vals.values())})")
        want = 8
    con.commit()
    con.close()
    for _ in range(2):  # the second start is a no-op
        db = _open(path)
        db.engine.dispose()
    db = _open(path)
    new = _new_deployment(db, "fresh")
    assert new == want  # above every id still referenced (benchmark run of 4, rollup of 5, decision about 7)
    with db.session() as s:
        assert s.scalars(select(BenchmarkRun).where(BenchmarkRun.deployment_id == new)).all() == []
    with db.session() as s:
        s.delete(s.get(Deployment, new))
    assert _new_deployment(db, "again") == new + 1  # the removed id is not handed out again
    with db.session() as s:
        s.delete(s.get(Deployment, 2))
    db.engine.dispose()
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    assert con.execute("SELECT COUNT(*) FROM deployment_events").fetchone()[0] == 0  # cascaded with deployment 2
    assert con.execute("SELECT name FROM deployments ORDER BY id").fetchall() == [("q8",), ("again",)]
    with pytest.raises(sqlite3.IntegrityError):
        _insert_dep(con, 99, "q8")  # the name is still unique
    con.close()
    assert _versions(path) == [m.version for m in migrations.MIGRATIONS]


def test_fresh_database_never_reuses_a_deployment_id(tmp_path):
    db = _open(tmp_path / "new.sqlite3")
    first = _new_deployment(db, "a")
    with db.session() as s:
        s.delete(s.get(Deployment, first))
    assert _new_deployment(db, "b") == first + 1
    db.engine.dispose()
