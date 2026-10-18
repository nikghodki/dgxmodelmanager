"""Versioned, ordered, idempotent schema migrations for the SQLite database.

Startup (``Database.__init__``) runs, in order:

1. ``Base.metadata.create_all``: creates tables that do not exist yet (with all their constraints and indexes);
2. ``add_missing_columns``: adds nullable columns that a newer model declares on an existing table;
3. ``add_missing_indexes``: creates non-unique indexes a newer model declares on an existing table (``create_all``
   only indexes tables it creates);
4. ``migrate``: the versioned steps below, each applied at most once, in version order, recorded in the
   ``schema_version`` table. These are the changes the three automatic steps cannot make safely: unique
   constraints/indexes (existing rows may violate them), data rewrites, renames.

Each versioned step runs in its own ``BEGIN IMMEDIATE`` transaction (SQLite has transactional DDL): it takes the
write lock, re-checks whether another process (the app and an admin CLI may start together) applied it meanwhile,
applies it and records it, or rolls back completely and raises. Steps must be idempotent on their own as well
(``IF NOT EXISTS``, checks before changes), so a database whose ``schema_version`` row was lost converges.

Rollback compatibility: migrations are forward-only. Older DGXMM versions ignore ``schema_version`` and tables
they do not know; every step here must keep the schema readable and writable by the previous release (add, never
drop or rename, columns and tables). A database migrated by a newer DGXMM opened by an older one logs a warning and
is otherwise left alone. Migration 30 rebuilds ``deployments`` (same columns and constraints plus AUTOINCREMENT,
which older versions use transparently). Tested downgrades: main (b193e9c, before the decision model), dec-int
(e3ca780) and main (7c055b7, for migration 30) read and write a migrated database. Partly supported: the first
decision-model builds before 4a6a215 (schema 11ce098, diagnosis service abc4af7) check for an existing diagnosis
before inserting but do not handle the IntegrityError that migration 2's unique index raises when two diagnoses of
the same failure event race; that background task then fails and logs (one duplicate diagnosis is not stored);
everything else works.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import CreateIndex

log = logging.getLogger(__name__)
TABLE = "schema_version"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]
    # rebuilds a parent table: run with foreign key enforcement off (SQLite's documented table rebuild; otherwise
    # dropping the old table cascades to its children) and checked for new foreign key violations before commit
    rebuilds: bool = False


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None


def _has_unique(conn: sqlite3.Connection, table: str, cols: list[str]) -> bool:
    """A unique index or UNIQUE constraint on exactly `cols` (in order) exists."""
    for row in conn.execute(f'PRAGMA index_list("{table}")').fetchall():
        name, unique = row[1], row[2]
        if unique and [r[2] for r in conn.execute(f'PRAGMA index_info("{name}")').fetchall()] == cols:
            return True
    return False


def _baseline(conn: sqlite3.Connection) -> None:
    """Version 1: the schema as created by create_all + add_missing_columns (nothing else to do)."""


def _diagnoses_unique_event(conn: sqlite3.Connection) -> None:
    """Version 2: one diagnosis per failure event. The first decision-model build created ``diagnoses`` without
    UNIQUE (deployment_id, event_id); keep the newest row of any duplicate pair, then add the unique index
    (SQLite cannot add a constraint to an existing table; a unique index enforces the same rule). Rows with a
    NULL event_id are distinct by SQL semantics and are kept."""
    if not _table_exists(conn, "diagnoses") or _has_unique(conn, "diagnoses", ["deployment_id", "event_id"]):
        return
    conn.execute("DELETE FROM diagnoses WHERE event_id IS NOT NULL AND id NOT IN "
                 "(SELECT MAX(id) FROM diagnoses WHERE event_id IS NOT NULL GROUP BY deployment_id, event_id)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_diagnoses_event ON diagnoses (deployment_id, event_id)")


def _insights_live_dedupe(conn: sqlite3.Connection) -> None:
    """Version 3: at most one live (open / snoozed / dismissed) insight per dedupe_key. ``create_all`` creates the
    ``insights`` table with this partial unique index on upgrade from main (7c055b7) and on fresh installs; this step
    covers a table created without it: duplicates keep their newest row live and the older ones become ``expired``
    (closed rows are pruned by retention), then the index is added."""
    if not _table_exists(conn, "insights"):
        return
    live = "status IN ('open', 'snoozed', 'dismissed')"
    conn.execute(f"UPDATE insights SET status = 'expired' WHERE {live} AND id NOT IN "
                 f"(SELECT MAX(id) FROM insights WHERE {live} GROUP BY dedupe_key)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_insights_live_dedupe ON insights (dedupe_key) "
                 "WHERE status IN ('open', 'snoozed', 'dismissed')")


def _traffic_profiles_unique_hour(conn: sqlite3.Connection) -> None:
    """Version 20: one traffic-profile row per (deployment, hour of week). ``create_all`` creates the table with this
    constraint; this step covers a table created without it: duplicates keep their newest row, then a unique index
    is added. Profiles are rebuilt nightly, so dropping a duplicate loses nothing."""
    if not _table_exists(conn, "traffic_profiles") or _has_unique(conn, "traffic_profiles",
                                                                   ["deployment_id", "hour_of_week"]):
        return
    conn.execute("DELETE FROM traffic_profiles WHERE id NOT IN "
                 "(SELECT MAX(id) FROM traffic_profiles GROUP BY deployment_id, hour_of_week)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_traffic_profiles_hour ON traffic_profiles "
                 "(deployment_id, hour_of_week)")


_SUBJECT_TABLES = ("insights", "decisions")  # subject "deployment:<id>"


def _deployment_ids_never_reused(conn: sqlite3.Connection) -> None:
    """Version 30: deployment ids are never handed out again. Without AUTOINCREMENT SQLite reuses the highest rowid
    after a delete, so a new deployment took the id of a removed one and showed its benchmark runs, readings,
    rollups and insights (all keyed by the id; benchmark results themselves are per model and config, by design).
    Rebuild ``deployments`` with AUTOINCREMENT (columns copied by name; ``create_all`` already creates it so on a
    fresh install) and start the sequence above every deployment id still referenced anywhere."""
    from sqlalchemy.dialects import sqlite as sqlite_dialect
    from sqlalchemy.schema import CreateTable

    from dgxmm.db import Deployment
    if not _table_exists(conn, "deployments"):
        return
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'deployments'").fetchone()[0]
    if "AUTOINCREMENT" not in sql.upper():
        table = Deployment.__table__
        new = str(CreateTable(table).compile(dialect=sqlite_dialect.dialect()))
        new = new.replace("CREATE TABLE deployments ", "CREATE TABLE deployments__rebuild ", 1)
        if "AUTOINCREMENT" not in new or "deployments__rebuild" not in new:  # pragma: no cover - model changed
            raise RuntimeError("unexpected DDL for the deployments rebuild")
        old_cols = [r[1] for r in conn.execute('PRAGMA table_info("deployments")')]
        indexes = [r[0] for r in conn.execute("SELECT sql FROM sqlite_master WHERE type = 'index' AND "
                                               "tbl_name = 'deployments' AND sql IS NOT NULL")]
        conn.execute("DROP TABLE IF EXISTS deployments__rebuild")
        conn.execute(new)
        known = set(table.columns.keys())
        for row in conn.execute('PRAGMA table_info("deployments")').fetchall():  # columns of a newer DGXMM: kept
            if row[1] not in known:
                conn.execute(f'ALTER TABLE deployments__rebuild ADD COLUMN "{row[1]}" {row[2] or ""}')
        cols = ", ".join(f'"{c}"' for c in old_cols)
        conn.execute(f"INSERT INTO deployments__rebuild ({cols}) SELECT {cols} FROM deployments")
        conn.execute("DROP TABLE deployments")
        conn.execute("ALTER TABLE deployments__rebuild RENAME TO deployments")
        for ix in indexes:
            conn.execute(ix)
    high = [conn.execute("SELECT COALESCE(MAX(id), 0) FROM deployments").fetchone()[0]]
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')}
        if "deployment_id" in cols:
            high.append(conn.execute(f'SELECT COALESCE(MAX(deployment_id), 0) FROM "{name}"').fetchone()[0])
        if name in _SUBJECT_TABLES and "subject" in cols:
            high.append(conn.execute(
                f"SELECT COALESCE(MAX(CAST(SUBSTR(subject, 12) AS INTEGER)), 0) FROM \"{name}\" "
                "WHERE subject LIKE 'deployment:%'").fetchone()[0])
    seq = max(int(h or 0) for h in high)
    have = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'deployments'").fetchone()
    if have is None:
        conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('deployments', ?)", (seq,))
    elif have[0] < seq:
        conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'deployments'", (seq,))


# Version numbers reserved for the parallel feature lanes of the advanced decision-model features, so their
# branches merge without renumbering: 4-9 deploy-time features, 10-19 operations features, 20-29 schedules/commands;
# fixes after them from 30 on.
MIGRATIONS: list[Migration] = [
    Migration(1, "baseline (create_all + add_missing_columns)", _baseline),
    Migration(2, "diagnoses: unique (deployment_id, event_id)", _diagnoses_unique_event),
    Migration(3, "insights: one live insight per dedupe_key", _insights_live_dedupe),
    Migration(20, "traffic_profiles: unique (deployment_id, hour_of_week)", _traffic_profiles_unique_hour),
    Migration(30, "deployments: ids are never reused (AUTOINCREMENT)", _deployment_ids_never_reused, rebuilds=True),
]
LATEST = max(m.version for m in MIGRATIONS)


def add_missing_indexes(engine: Engine, metadata) -> list[str]:
    """Create non-unique indexes declared in `metadata` that an existing table lacks (idempotent). Unique indexes
    are never added here: existing rows may violate them, so they need a versioned migration."""
    added: list[str] = []
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in tables:
                continue
            have = {ix["name"] for ix in insp.get_indexes(table.name)}
            for index in table.indexes:
                if index.unique or index.name in have:
                    continue
                try:
                    conn.exec_driver_sql(str(CreateIndex(index, if_not_exists=True).compile(dialect=engine.dialect)))
                except OperationalError as e:  # pragma: no cover - IF NOT EXISTS makes this very unlikely
                    if "already exists" not in str(e):
                        raise
                    continue
                added.append(index.name)
    return added


def _applied(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute(f"SELECT version FROM {TABLE}").fetchall()}


def migrate(engine: Engine) -> list[int]:
    """Apply pending versioned migrations; returns the versions applied by this call."""
    path = engine.url.database
    if not path or path == ":memory:":
        raise ValueError("migrations need a file-backed SQLite database")
    # Our own connection in autocommit mode, so BEGIN IMMEDIATE / COMMIT are explicit and DDL is transactional.
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    applied_now: list[int] = []
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                     "applied_at TEXT NOT NULL)")
        done = _applied(conn)
        newer = sorted(v for v in done if v > LATEST)
        if newer:
            log.warning("database schema version %s was written by a newer DGXMM (this one knows up to %s); "
                        "continuing without changing it", newer[-1], LATEST)
        for m in sorted(MIGRATIONS, key=lambda m: m.version):
            if m.version in done:
                continue
            if m.rebuilds:
                conn.execute("PRAGMA foreign_keys=OFF")  # a no-op inside a transaction: set before BEGIN
            conn.execute("BEGIN IMMEDIATE")
            try:
                if m.version in _applied(conn):  # another process applied it while we waited for the lock
                    conn.execute("ROLLBACK")
                    continue
                violations = conn.execute("PRAGMA foreign_key_check").fetchall() if m.rebuilds else []
                m.apply(conn)
                if m.rebuilds and conn.execute("PRAGMA foreign_key_check").fetchall() != violations:
                    raise RuntimeError(f"schema migration {m.version} would break foreign keys")
                conn.execute(f"INSERT INTO {TABLE} (version, name, applied_at) VALUES (?, ?, ?)",
                             (m.version, m.name, datetime.now(timezone.utc).isoformat()))
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # no transaction left (or the connection broke): keep the original error
                    log.exception("rolling back schema migration %s failed", m.version)
                log.error("schema migration %s (%s) failed; the database was left at the previous version",
                          m.version, m.name)
                raise
            finally:
                if m.rebuilds:
                    conn.execute("PRAGMA foreign_keys=ON")
            applied_now.append(m.version)
            log.info("applied schema migration %s: %s", m.version, m.name)
    finally:
        conn.close()
    return applied_now
