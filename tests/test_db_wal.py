"""Switching a database to WAL while another process opens it (the app and an admin CLI starting together, CI on a
busy runner): SQLite answers "database is locked" at once, without its busy handler, so the switch is retried."""
import sqlite3
import threading

import pytest

from dgxmm import db


class FlakyCursor:
    def __init__(self, failures: int, message: str = "database is locked"):
        self.failures, self.message, self.calls = failures, message, []

    def execute(self, sql):
        self.calls.append(sql)
        if sql.startswith("PRAGMA journal_mode") and self.failures:
            self.failures -= 1
            raise sqlite3.OperationalError(self.message)


def test_wal_switch_retries_while_locked(monkeypatch):
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    cur = FlakyCursor(failures=3)
    db.enable_wal(cur)
    assert cur.calls == ["PRAGMA journal_mode=WAL"] * 4


def test_wal_switch_gives_up_after_the_deadline(monkeypatch):
    clock = iter(range(0, 1000))
    monkeypatch.setattr(db.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        db.enable_wal(FlakyCursor(failures=10_000), timeout_s=5)


def test_wal_switch_does_not_retry_other_errors(monkeypatch):
    monkeypatch.setattr(db.time, "sleep", lambda s: pytest.fail("no retry"))
    cur = FlakyCursor(failures=1, message="disk I/O error")
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        db.enable_wal(cur)
    assert len(cur.calls) == 1


def test_concurrent_first_opens_of_a_rollback_journal_database(tmp_path):
    """Regression (GitHub runner): four threads open a fresh rollback-journal database at once, many times."""
    for i in range(15):
        path = tmp_path / f"r{i}.sqlite3"
        sqlite3.connect(path).close()
        errors = []

        def start():
            try:
                db.Database(f"sqlite:///{path}").engine.dispose()
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)
        threads = [threading.Thread(target=start) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
