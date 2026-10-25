"""SQLite persistence (SQLAlchemy 2 ORM)."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint, create_engine, event, inspect, text)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_aware(dt: datetime) -> datetime:
    """SQLite drops tzinfo; every stored datetime is UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    pw_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16))  # admin | viewer
    must_change_pw: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserSession(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(token)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    csrf_token: Mapped[str] = mapped_column(String(64))
    ip: Mapped[str] = mapped_column(String(64), default="")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CliToken(Base):
    """Short-lived admin token for the local admin CLI (dgxmm/auth/cli_tokens.py). Only the SHA-256 is stored."""
    __tablename__ = "cli_tokens"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(token)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    scope: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Deployment(Base):
    __tablename__ = "deployments"
    # ids are never reused: rows of other tables keyed by a removed deployment's id (benchmark runs, readings,
    # rollups, insights) must not attach to a new one (migration 30 rebuilds older tables)
    __table_args__ = {"sqlite_autoincrement": True}
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    engine: Mapped[str] = mapped_column(String(16))
    model_ref: Mapped[str] = mapped_column(String(512))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(16))
    container_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    alias: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reserved_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    estimate: Mapped[dict] = mapped_column(JSON, default=dict)
    litellm_status: Mapped[str] = mapped_column(String(16), default="none")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The last exit was an out-of-memory kill (Docker's OOMKilled, or the kernel OOM killer's line in the log).
    oom_killed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    adopted_from: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # System deployment (the decision model): never a plan stop/sleep candidate or LiteLLM model; the watchdog stops it
    # first under memory pressure, and whenever its own memory runs away (dgxmm/watchdog.py).
    # A real column, not a config key: config is user-visible and editable. NULL on rows from older versions.
    system: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # "on_demand" (started by the first request, stopped after idle_timeout_s without one; dgxmm/ondemand) or
    # "always_on". Nullable for add_missing_columns; migration 40 sets "always_on" on older rows, and a NULL
    # (a row written later by an older DGXMM) reads as "always_on".
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True, default="always_on")
    idle_timeout_s: Mapped[int | None] = mapped_column(Integer, nullable=True)  # NULL = ondemand_idle_timeout_s
    # A switch from on demand to always on that waits for the deployment to be ready: until then it stays on demand
    # (registered through the gateway), and the deployer applies this mode when it reaches ready, failed or stopped.
    pending_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DeploymentEvent(Base):
    __tablename__ = "deployment_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    from_state: Mapped[str] = mapped_column(String(16))
    to_state: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text, default="")
    log_tail: Mapped[str | None] = mapped_column(Text, nullable=True)
    # who asked for a stop ("user", "schedule:<id>", ...); NULL on other events and on rows of older versions
    actor: Mapped[str | None] = mapped_column(String(64), nullable=True)


class MemoryCalibration(Base):
    __tablename__ = "memory_calibration"
    id: Mapped[int] = mapped_column(primary_key=True)
    engine: Mapped[str] = mapped_column(String(16))
    repo: Mapped[str] = mapped_column(String(512), index=True)
    config_hash: Mapped[str] = mapped_column(String(32))
    estimated_bytes: Mapped[int] = mapped_column(BigInteger)
    measured_bytes: Mapped[int] = mapped_column(BigInteger)  # 0 for OOM records
    oom: Mapped[bool] = mapped_column(Boolean, default=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # How `measured_bytes` was obtained. NULL = legacy row (cgroup + NVML: misses CUDA memory on GB10) and is
    # ignored by calibration; "engine_log" = engine startup accounting (+ host-side cgroup memory); "oom".
    method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quality: Mapped[str | None] = mapped_column(String(16), nullable=True)  # verified | disputed | unverified
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Reading label (dgxmm/decider/reading.py); when set, quality is verified (plausible) or disputed (not learned
    # from). The MemAvailable check's own verdict moves to reading.delta_quality; unlabelled rows are used as before.
    reading: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    engine: Mapped[str] = mapped_column(String(16))
    model_ref: Mapped[str] = mapped_column(String(512), index=True)
    config_hash: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    concurrency: Mapped[int] = mapped_column(Integer)
    input_len: Mapped[int] = mapped_column(Integer)
    output_len: Mapped[int] = mapped_column(Integer)
    requests: Mapped[int] = mapped_column(Integer)
    errors: Mapped[int | None] = mapped_column(Integer, nullable=True)  # failed requests at this level
    ttft_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    tps_total: Mapped[float] = mapped_column(Float)
    tps_per_req: Mapped[float] = mapped_column(Float)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Reading label (dgxmm/decider/reading.py): quality verified (plausible) | disputed (left out of advice);
    # NULL = not labelled (undecided or feature off): used as before
    quality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reading: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class BenchmarkRun(Base):
    """One benchmark invocation. Levels with zero successful requests are NOT stored as `Benchmark` rows
    (they would read as tps=0 results); the failure is recorded here so the UI can say where it failed."""
    __tablename__ = "benchmark_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    engine: Mapped[str] = mapped_column(String(16))
    model_ref: Mapped[str] = mapped_column(String(512), index=True)
    config_hash: Mapped[str] = mapped_column(String(32))
    levels: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | done | failed
    failed_concurrency: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MetricRollup(Base):
    __tablename__ = "metric_rollups"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)  # NULL = host
    minute: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[dict] = mapped_column(JSON)


class Download(Base):
    __tablename__ = "downloads"
    id: Mapped[int] = mapped_column(primary_key=True)
    repo_id: Mapped[str] = mapped_column(String(256))
    revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    files: Mapped[list] = mapped_column(JSON, default=list)
    bytes_done: Mapped[int] = mapped_column(BigInteger, default=0)
    bytes_total: Mapped[int] = mapped_column(BigInteger, default=0)
    state: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed|cancelled
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Favourite(Base):
    __tablename__ = "favourites"
    ref: Mapped[str] = mapped_column(String(512), primary_key=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(256), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class DecisionRecord(Base):
    """One decider decision (rules, model or abstained). Never stores raw untrusted input: only its hash."""
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    feature: Mapped[str] = mapped_column(String(32), index=True)
    subject: Mapped[str] = mapped_column(String(256), default="")
    source: Mapped[str] = mapped_column(String(8))  # rules | model | none
    choice: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    probabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(String(64), nullable=True)  # error code when the model call failed
    feedback: Mapped[str | None] = mapped_column(String(8), nullable=True)  # accepted | rejected
    feedback_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AppSetting(Base):
    """Runtime settings editable from the UI (take precedence over config.yaml/env)."""
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DiagnosisRow(Base):
    """Why a deployment failed (decision-model feature 1). The latest row per deployment is shown."""
    __tablename__ = "diagnoses"
    __table_args__ = (UniqueConstraint("deployment_id", "event_id", name="uq_diagnoses_event"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="CASCADE"), index=True)
    event_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    cause: Mapped[str | None] = mapped_column(String(64), nullable=True)  # NULL = could not diagnose confidently
    source: Mapped[str] = mapped_column(String(8))  # rules | model | none
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    fix_hint: Mapped[str] = mapped_column(Text, default="")
    patch: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log_excerpt: Mapped[str] = mapped_column(Text, default="")  # redacted


class ArtifactKindRow(Base):
    """Cache of model-derived artifact kinds (decision-model feature 4), keyed by ref + snapshot mtime."""
    __tablename__ = "artifact_kinds"
    ref: Mapped[str] = mapped_column(String(1024), primary_key=True)
    mtime: Mapped[float] = mapped_column(Float)
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)  # model's top choice (even below threshold)
    source: Mapped[str] = mapped_column(String(8))  # rules | model | none
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


# Insight statuses that hold their dedupe_key: at most one such row per key (open/snoozed are updated in place; a
# dismissed one stays dismissed until its condition clears). applied / expired are closed for good.
INSIGHT_LIVE = ("open", "snoozed", "dismissed")


class Insight(Base):
    """A proactive finding of a decision-model feature (spec "Insights feed"): templated title/body, an optional
    action descriptor the UI turns into an existing dialog (never executed by the server). dgxmm/decider/insights.py
    """
    __tablename__ = "insights"
    __table_args__ = (Index("uq_insights_live_dedupe", "dedupe_key", unique=True,
                            sqlite_where=text("status IN ('open', 'snoozed', 'dismissed')")),)
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    feature: Mapped[str] = mapped_column(String(32), index=True)
    subject: Mapped[str] = mapped_column(String(256), default="", index=True)  # e.g. deployment:<id>
    severity: Mapped[str] = mapped_column(String(8))  # info | warning | critical
    title: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(8))  # rules | model
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open|dismissed|snoozed|applied|expired
    snooze_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dedupe_key: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CompatCacheRow(Base):
    """Results of the pre-download compatibility check (dgxmm/decider/compat.py) by (repo, commit sha): verdicts,
    reason keys and flags only, never file contents. Pruned after 7 days (dgxmm/decider/retention.py)."""
    __tablename__ = "compat_cache"
    repo: Mapped[str] = mapped_column(String(256), primary_key=True)
    sha: Mapped[str] = mapped_column(String(64), primary_key=True)
    result: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class DeploymentSchedule(Base):
    """A user-owned sleep/wake (or stop/start) schedule of a deployment (decision-model feature 7), run by the
    schedule runner (dgxmm/decider/schedule.py) through the deployer's normal paths, audited as ``schedule:<id>``."""
    __tablename__ = "deployment_schedules"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(8))  # sleep | wake | stop | start
    days: Mapped[int] = mapped_column(Integer)  # bitmask: bit 0 = Monday .. bit 6 = Sunday
    at_local: Mapped[str] = mapped_column(String(5))  # HH:MM wall time in `timezone`
    timezone: Mapped[str] = mapped_column(String(64))  # IANA name
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Changed by the user only (create / edit / pause / resume), never by a run: slots before it are not "missed".
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_run_date: Mapped[str | None] = mapped_column(String(10), nullable=True)  # local date of the last slot handled
    last_result: Mapped[str | None] = mapped_column(String(256), nullable=True)


class TrafficProfile(Base):
    """A deployment's request activity per hour of the week (0 = Monday 00:00 local) over the last 30 days of
    metric rollups, rebuilt nightly (decision-model feature 7)."""
    __tablename__ = "traffic_profiles"
    __table_args__ = (UniqueConstraint("deployment_id", "hour_of_week", name="uq_traffic_profiles_hour"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="CASCADE"), index=True)
    hour_of_week: Mapped[int] = mapped_column(Integer)
    requests_per_min_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    requests_per_min_p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    active_fraction: Mapped[float] = mapped_column(Float)  # share of observed minutes with request activity
    samples: Mapped[int] = mapped_column(Integer)  # observed minutes
    weeks: Mapped[int | None] = mapped_column(Integer, nullable=True)  # weeks with this hour observed
    idle_weeks: Mapped[int | None] = mapped_column(Integer, nullable=True)  # of those, weeks it was idle (<= 5 %)
    days: Mapped[int | None] = mapped_column(Integer, nullable=True)  # local days with data (the whole profile)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def add_missing_columns(engine) -> list[str]:
    """Idempotent schema upgrade step: `create_all` creates new tables but never alters existing ones, so add any
    nullable model column missing from an existing table. Constraints, unique indexes and data changes are
    versioned migrations instead (dgxmm/migrations.py)."""
    added: list[str] = []
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                if not col.nullable:
                    # SQLite can only ADD a NOT NULL column with a constant default, and existing rows would need a
                    # real value: that is a data migration, not something to guess at startup. Fail loudly instead.
                    raise RuntimeError(f"cannot auto-add NOT NULL column {table.name}.{col.name}")
                ddl = col.type.compile(dialect=engine.dialect)
                try:
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))
                except OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
                    continue  # another process (e.g. a concurrent startup) added it first
                added.append(f"{table.name}.{col.name}")
    return added


def enable_wal(cur, timeout_s: float = 5.0) -> None:
    """Switch the database to WAL. Converting a rollback-journal database takes an exclusive lock, and when another
    connection converts or opens it at the same moment (concurrent startups) SQLite answers "database is locked" at
    once instead of waiting in its busy handler, so retry with a short backoff for up to `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    delay = 0.01
    while True:
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() >= deadline:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 0.2)


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _pragmas(conn, _):  # noqa: ANN001
            cur = conn.cursor()
            cur.execute("PRAGMA busy_timeout=5000")  # first: switching to WAL needs a lock another process may hold
            enable_wal(cur)
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        from dgxmm import migrations  # imports this module's Base lazily (no cycle)
        # Concurrent startups (the app and an admin CLI) can race on CREATE TABLE: every lost race means another
        # process created a table, so re-checking converges within one round per table.
        for attempt in range(len(Base.metadata.tables) + 1):
            try:
                Base.metadata.create_all(self.engine)
                break
            except OperationalError as e:
                if "already exists" not in str(e) or attempt == len(Base.metadata.tables):
                    raise
        add_missing_columns(self.engine)
        migrations.add_missing_indexes(self.engine, Base.metadata)
        migrations.migrate(self.engine)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._factory()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()
