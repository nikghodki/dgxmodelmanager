"""SQLite persistence (SQLAlchemy 2 ORM)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event,
                        inspect, text)
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


class Deployment(Base):
    __tablename__ = "deployments"
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
    adopted_from: Mapped[str | None] = mapped_column(String(128), nullable=True)
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


class AppSetting(Base):
    """Runtime settings editable from the UI (take precedence over config.yaml/env)."""
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


def add_missing_columns(engine) -> list[str]:
    """Tiny idempotent schema upgrade (there is no migration framework): `create_all` creates new tables but
    never alters existing ones, so add any nullable model column missing from an existing table."""
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


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _pragmas(conn, _):  # noqa: ANN001
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

        Base.metadata.create_all(self.engine)
        add_missing_columns(self.engine)
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
