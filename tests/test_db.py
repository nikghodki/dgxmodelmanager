import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dgxmm.db import Deployment, DeploymentEvent, User


def test_roundtrip_user_and_deployment(db):
    with db.session() as s:
        s.add(User(username="admin", pw_hash="x", role="admin"))
        s.add(Deployment(name="q8b", engine="vllm", model_ref="Qwen/Qwen3-8B", config={"max_model_len": 4096},
                         state="planned", port=8100, reserved_bytes=10, estimate={}))
    with db.session() as s:
        dep = s.scalar(select(Deployment).where(Deployment.name == "q8b"))
        assert dep.config == {"max_model_len": 4096}
        assert dep.litellm_status == "none"
        s.add(DeploymentEvent(deployment_id=dep.id, from_state="planned", to_state="starting", message="go"))
    with db.session() as s:
        assert len(s.scalars(select(DeploymentEvent)).all()) == 1


def test_unique_username_and_rollback(db):
    with db.session() as s:
        s.add(User(username="a", pw_hash="x", role="viewer"))
    with pytest.raises(IntegrityError):
        with db.session() as s:
            s.add(User(username="a", pw_hash="y", role="viewer"))
    with db.session() as s:
        assert len(s.scalars(select(User)).all()) == 1


def test_add_missing_columns_upgrades_old_schema_idempotently(tmp_path):
    """No migration framework: new nullable columns are added to existing SQLite tables at startup."""
    import sqlite3

    from dgxmm.db import Database, MemoryCalibration
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE memory_calibration (id INTEGER PRIMARY KEY, engine VARCHAR(16), repo VARCHAR(512), "
                "config_hash VARCHAR(32), estimated_bytes BIGINT, measured_bytes BIGINT, oom BOOLEAN, ts DATETIME)")
    con.execute("INSERT INTO memory_calibration VALUES (1, \"vllm\", \"r\", \"h\", 10, 4, 0, \"2026-10-01 00:00:00\")")
    con.commit()
    con.close()
    for _ in range(2):  # second open is a no-op
        db = Database(f"sqlite:///{path}")
        with db.session() as s:
            row = s.get(MemoryCalibration, 1)
            assert row.measured_bytes == 4 and row.method is None and row.quality is None and row.detail is None
        db.engine.dispose()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(memory_calibration)")}
    assert {"method", "quality", "detail"} <= cols


def test_add_missing_columns_tolerates_concurrent_add(tmp_path, monkeypatch):
    """Two processes starting at once: the other one added the column between our inspect and ALTER."""
    import sqlalchemy

    import dgxmm.db as dbmod
    path = tmp_path / "c.sqlite3"
    db = dbmod.Database(f"sqlite:///{path}")
    real_inspect = sqlalchemy.inspect

    class StaleInspector:
        def __init__(self, engine):
            self._i = real_inspect(engine)

        def get_table_names(self):
            return self._i.get_table_names()

        def get_columns(self, table):
            cols = self._i.get_columns(table)
            return [c for c in cols if not (table == "memory_calibration" and c["name"] == "method")]
    monkeypatch.setattr(dbmod, "inspect", StaleInspector)
    assert dbmod.add_missing_columns(db.engine) == []
