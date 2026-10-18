"""Background insight engine (spec "Background engine"): one scheduler for the decision-model features' periodic
jobs (health checks, traffic profiles, schedule proposals and runner, routing advice, threshold tuning, ...).

- ``register_job(name, feature, schedule, fn, budget_calls)``; ``schedule`` is an interval in seconds or a wall-clock
  spec ``"daily HH:MM"`` / ``"weekly <mon..sun> HH:MM"`` in the box's time zone (``local_zone()``: setting ``tz`` /
  ``DGXMM_TZ``, which install.sh sets from the host; else ``TZ``; else /etc/localtime; else UTC). DST: a wall time
  that does not exist (spring gap) runs at the same offset after the gap (02:30 -> 03:30); one that occurs twice
  (autumn) runs once, at the first occurrence. ``trigger(name)`` asks for a run as soon as possible;
  ``trigger(name, settle_s)`` (a change notice) once no other notice came for ``settle_s``, and at the latest
  ``SETTLE_MAX_FACTOR * settle_s`` after the first one, so a burst of changes is one run.
- A job whose feature is switched off does not run (``rules_enabled``; with ``requires_decider=True`` the decision
  model must be enabled too); pending triggers are dropped and the next run is re-planned when it is switched on.
  A job is skipped while its previous run is still active (``skipped`` counts the missed slots).
- ``fn(run: JobRun)``: a coroutine function, or a plain function run in the engine's own small thread pool
  (``JOB_THREADS`` workers, separate from asyncio's default pool). Async jobs put CPU work through
  ``run.to_thread``. A run, including any thread it started, counts as running until that thread really ended, so a
  timed-out run never overlaps the next one. The return value (a short summary, or None) is the last result.
- Budget: while a job runs, every ``Decider.decide`` in its context (``run.decider`` = ``ctx.decider``, feature
  services, ...) uses the client's background lane and counts against ``budget_calls``
  (``dgxmm.decider.service.JobBudget`` / ``job_context``); beyond it calls abstain without asking. Calls the decider
  would not make anyway (disabled, unavailable) do not count.
- A run that raises or exceeds ``timeout_s`` is a failure: retried with exponential backoff (1 min doubling, at most
  1 h; never sooner than the job's interval); after ``DEGRADED_AFTER`` consecutive failures one deduplicated
  "feature degraded" insight is raised (``degraded:<job>``, expires ``DEGRADED_TTL`` after the last failure), and
  every success resolves it. Errors are logged, never shown (the insight body is a template).
- ``status()``: per job last run, duration, result and next run (admin part of ``GET /api/decider/status``).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone, tzinfo
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dgxmm.db import utcnow
from dgxmm.decider import registry
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.service import JobBudget, job_context

log = logging.getLogger(__name__)
TICK_S = 5.0                 # how often due jobs are checked (trigger() wakes the loop at once)
FIRST_RUN_MAX_S = 60.0       # interval jobs first run after min(interval, this) once DGXMM is up
DEFAULT_TIMEOUT_S = 600.0
DEGRADED_AFTER = 3           # consecutive failures before the "feature degraded" insight
BACKOFF_BASE_S, BACKOFF_MAX_S = 60.0, 3600.0
DEGRADED_TTL = timedelta(hours=6)  # a degraded insight expires this long after the last failure
SETTLE_MAX_FACTOR = 4        # a settling trigger runs at most this many settle_s after the burst's first notice
JOB_THREADS = 2              # the engine's own pool for sync jobs and run.to_thread
LOCALTIME = "/etc/localtime"
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_AT = re.compile(r"^(\d{2}):(\d{2})$")


@dataclass(frozen=True)
class Cadence:
    """Every ``interval_s`` seconds, or daily / weekly at a local ``hh:mm`` (``weekday`` 0 = Monday)."""
    interval_s: float | None = None
    hh: int = 0
    mm: int = 0
    weekday: int | None = None  # None with interval_s None = daily

    @staticmethod
    def parse(spec: float | int | str) -> "Cadence":
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            if spec < 1:
                raise ValueError("interval must be at least 1 s")
            return Cadence(interval_s=float(spec))
        parts = str(spec).split()
        if len(parts) == 2 and parts[0] == "daily":
            hh, mm = _hhmm(parts[1])
            return Cadence(hh=hh, mm=mm)
        if len(parts) == 3 and parts[0] == "weekly" and parts[1] in DAYS:
            hh, mm = _hhmm(parts[2])
            return Cadence(hh=hh, mm=mm, weekday=DAYS.index(parts[1]))
        raise ValueError(f"schedule {spec!r}: seconds, 'daily HH:MM' or 'weekly <mon..sun> HH:MM'")

    def describe(self) -> str:
        if self.interval_s is not None:
            return f"every {self.interval_s:g} s"
        at = f"{self.hh:02d}:{self.mm:02d}"
        return f"daily {at}" if self.weekday is None else f"weekly {DAYS[self.weekday]} {at}"

    def next_after(self, now: datetime, tz: tzinfo | None) -> datetime:
        """The next run strictly after ``now`` (aware), as an aware UTC datetime. Wall-clock specs are read in
        ``tz`` (None = UTC); see the module docstring for DST gaps and overlaps."""
        utc = now.astimezone(timezone.utc)
        if self.interval_s is not None:
            return utc + timedelta(seconds=self.interval_s)
        zone = tz or timezone.utc
        day = utc.astimezone(zone).date()
        for i in range(9):
            d = day + timedelta(days=i)
            if self.weekday is not None and d.weekday() != self.weekday:
                continue
            at = _wall(d, self.hh, self.mm, zone)
            if at > utc:
                return at
        raise AssertionError("unreachable: a slot exists within 8 days")


def _wall(d: date, hh: int, mm: int, zone: tzinfo) -> datetime:
    """The UTC instant of wall time d hh:mm in ``zone``: fold=0, i.e. the first occurrence in an autumn overlap; in a
    spring gap the pre-transition offset, which lands just after the gap."""
    return datetime.combine(d, dtime(hh, mm), tzinfo=zone).astimezone(timezone.utc)


def local_zone(name: str | None = None) -> tzinfo:
    """The box's time zone: ``name`` (setting ``tz`` / DGXMM_TZ), else $TZ, else /etc/localtime, else UTC."""
    for cand in (name, os.environ.get("TZ", "").lstrip(":") or None):
        if cand:
            try:
                return ZoneInfo(cand)
            except (ZoneInfoNotFoundError, ValueError, OSError):
                log.warning("unknown time zone %r; trying the next source", cand)
    try:
        with open(LOCALTIME, "rb") as f:
            return ZoneInfo.from_file(f, key="localtime")
    except (OSError, ValueError):
        return timezone.utc


def _hhmm(s: str) -> tuple[int, int]:
    m = _AT.match(s)
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"time {s!r}: HH:MM")
    return int(m.group(1)), int(m.group(2))


@dataclass
class JobRun:
    """What a job function gets: the app context, the decider (every ``decide`` in a job counts against its budget,
    whatever path it takes), the insight service, and ``to_thread`` for CPU work in the engine's thread pool."""
    ctx: Any
    name: str
    feature: str
    decider: Any
    insights: Any
    started_at: datetime
    budget: JobBudget
    _pool: concurrent.futures.Executor | None = None
    _threads: list = field(default_factory=list)

    def _submit(self, fn: Callable, *args, **kw) -> concurrent.futures.Future:
        cf = self._pool.submit(contextvars.copy_context().run, fn, *args, **kw)
        self._threads.append(cf)
        return cf

    async def to_thread(self, fn: Callable, *args, **kw):
        return await asyncio.wrap_future(self._submit(fn, *args, **kw))


@dataclass
class Job:
    name: str
    feature: str
    cadence: Cadence
    fn: Callable[[JobRun], Awaitable[str | None] | str | None]
    budget_calls: int = 0
    requires_decider: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    gate: Callable[[], bool] | None = None  # replaces the feature toggle as the on/off switch (e.g. user schedules)
    # state
    next_run: datetime | None = None
    task: asyncio.Task | None = None
    last_run_at: datetime | None = None
    last_duration_ms: float | None = None
    last_result: str | None = None
    failures: int = 0
    runs: int = 0
    skipped: int = 0
    calls_used: int = 0
    calls_refused: int = 0
    pending: bool = False
    pending_at: datetime | None = None    # a settling trigger: not before this (None = as soon as possible)
    settle_until: datetime | None = None  # ... and never later than this (the burst's first notice + the cap)
    enabled: bool = True

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class InsightEngine:
    def __init__(self, ctx, *, clock: Callable[[], datetime] = utcnow, tz: tzinfo | None = None,
                 tick_s: float = TICK_S):
        """``tz``: the zone of daily/weekly schedules; default ``local_zone(settings.tz)``."""
        if tz is None:
            tz = local_zone(getattr(getattr(ctx, "settings", None), "tz", None))
        self.ctx, self.clock, self.tz, self.tick_s = ctx, clock, tz, tick_s
        self._pool = concurrent.futures.ThreadPoolExecutor(JOB_THREADS, thread_name_prefix="insight-job")
        self._jobs: dict[str, Job] = {}
        self._started = clock()
        self._wake = asyncio.Event()

    # ---------- registration ----------
    def register_job(self, name: str, feature: str, schedule: float | int | str,
                     fn: Callable[[JobRun], Awaitable[str | None] | str | None], budget_calls: int = 0, *,
                     requires_decider: bool = False, timeout_s: float = DEFAULT_TIMEOUT_S,
                     gate: Callable[[], bool] | None = None) -> Job:
        """Add a job (ValueError: duplicate name, unknown feature, bad schedule or budget). Call before or after the
        engine started; it is picked up at the next tick. ``gate``: the job runs while it returns True, whatever
        the feature's toggle (for work the user owns, like the schedule runner)."""
        if name in self._jobs or not re.match(r"^[a-z][a-z0-9_]{0,47}$", name):
            raise ValueError(f"job name {name!r} is taken or invalid")
        if registry.find(feature) is None:
            raise ValueError(f"unknown feature {feature!r}")
        if int(budget_calls) < 0:
            raise ValueError("budget_calls must be >= 0")
        cadence = Cadence.parse(schedule)
        job = Job(name=name, feature=feature, cadence=cadence, fn=fn, budget_calls=int(budget_calls),
                  requires_decider=requires_decider, timeout_s=float(timeout_s), gate=gate)
        job.next_run = self._first_run(cadence)
        self._jobs[name] = job
        return job

    def _first_run(self, c: Cadence) -> datetime:
        now = self.clock()
        if c.interval_s is not None:
            return max(now, self._started) + timedelta(seconds=min(c.interval_s, FIRST_RUN_MAX_S))
        return c.next_after(now, self.tz)

    def trigger(self, name: str, settle_s: float = 0.0) -> None:
        """Run ``name`` as soon as possible (after its current run, if one is active). KeyError if unknown.
        ``settle_s`` > 0: a change notice — run once no further notice came for ``settle_s`` seconds, but at the
        latest ``SETTLE_MAX_FACTOR * settle_s`` after the first notice of the burst; never delays an earlier
        immediate trigger."""
        job = self._jobs[name]
        if settle_s <= 0:
            job.pending, job.pending_at, job.settle_until = True, None, None
        elif not job.pending or job.pending_at is not None:
            now = self.clock()
            if not job.pending:
                job.settle_until = now + timedelta(seconds=SETTLE_MAX_FACTOR * settle_s)
            job.pending = True
            job.pending_at = min(now + timedelta(seconds=settle_s), job.settle_until)
        self._wake.set()

    # ---------- scheduling ----------
    def _enabled(self, job: Job) -> bool:
        d = decider_of(self.ctx)
        try:
            if job.gate is not None:
                return bool(job.gate())
            if job.requires_decider:
                return bool(d.feature_enabled(job.feature))
            fn = getattr(d, "rules_enabled", None)
            return bool(fn(job.feature)) if fn is not None else True
        except Exception:
            log.exception("insight engine: could not read the toggle of %s", job.feature)
            return False

    async def tick(self) -> list[str]:
        """Start every due, enabled, idle job; returns the names started."""
        now = self.clock()
        started = []
        for job in self._jobs.values():
            was = job.enabled
            job.enabled = self._enabled(job)
            if not job.enabled:
                job.pending, job.pending_at = False, None  # a trigger does not outlive a switch-off
                continue
            if not was:  # switched (back) on: plan afresh instead of running a slot missed while off
                job.next_run = self._first_run(job.cadence)
                continue
            slot_due = job.next_run is not None and job.next_run <= now
            triggered = job.pending and (job.pending_at is None or job.pending_at <= now)
            if not (triggered or slot_due):
                continue
            if job.running:
                if slot_due:  # one missed slot: counted once, next slot planned
                    job.skipped += 1
                    job.next_run = job.cadence.next_after(now, self.tz)
                continue
            job.pending, job.pending_at = False, None
            job.task = asyncio.create_task(self._run(job), name=f"insight-job-{job.name}")
            started.append(job.name)
        return started

    async def wait_idle(self) -> None:
        """Tests / shutdown: wait for every running job to finish."""
        tasks = [j.task for j in self._jobs.values() if j.running]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self) -> None:
        """The background loop (ctx.background)."""
        insights = getattr(self.ctx, "insights", None)
        if insights is not None and hasattr(insights, "bind_loop"):
            insights.bind_loop(asyncio.get_running_loop())
        try:
            while True:
                try:
                    await self.tick()
                except Exception:
                    log.exception("insight engine tick failed")
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), self.tick_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            for j in self._jobs.values():
                if j.running:
                    j.task.cancel()
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def _run(self, job: Job) -> None:
        start, t0 = self.clock(), time.perf_counter()
        budget = JobBudget(job.name, job.budget_calls)
        run = JobRun(ctx=self.ctx, name=job.name, feature=job.feature, decider=decider_of(self.ctx),
                     insights=getattr(self.ctx, "insights", None), started_at=start, budget=budget, _pool=self._pool)
        ok, result = False, None
        try:
            with job_context(budget):  # copied into the job's threads by run._submit
                if inspect.iscoroutinefunction(job.fn):
                    res = await asyncio.wait_for(job.fn(run), job.timeout_s)
                else:
                    res = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(run._submit(job.fn, run))),
                                                 job.timeout_s)
            ok, result = True, "ok" if res in (None, "") else f"ok: {str(res)[:200]}"
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            result = "error: timeout"
            log.warning("insight job %s timed out after %ss; waiting for its thread to end", job.name, job.timeout_s)
        except Exception as e:
            result = f"error: {type(e).__name__}"
            log.exception("insight job %s failed", job.name)
        # Still running threads (a timed-out sync job, to_thread work of a cancelled async job): the run lasts until
        # they end, so the next one never overlaps it.
        busy = [cf for cf in run._threads if not cf.done()]
        if busy:
            await asyncio.gather(*(asyncio.wrap_future(cf) for cf in busy), return_exceptions=True)
        job.runs += 1
        job.last_run_at, job.last_result = start, result
        job.last_duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        job.calls_used, job.calls_refused = budget.used, budget.refused
        degraded = self._after(job, ok)
        try:  # database work off the event loop
            await asyncio.to_thread(self._raise_degraded if degraded else self._resolve_degraded, job)
        except Exception:
            log.exception("insight engine: could not update the degraded insight of %s", job.name)

    def _after(self, job: Job, ok: bool) -> bool:
        """Plan the next run; True when the job is degraded."""
        now = self.clock()
        nxt = job.cadence.next_after(now, self.tz)
        if ok:
            job.failures = 0
        else:
            job.failures += 1
            backoff = timedelta(seconds=min(BACKOFF_BASE_S * 2 ** (job.failures - 1), BACKOFF_MAX_S))
            if job.cadence.interval_s is not None:
                nxt = now + max(backoff, timedelta(seconds=job.cadence.interval_s))
            else:
                nxt = min(nxt, now + backoff)
        job.next_run = nxt
        return job.failures >= DEGRADED_AFTER

    def _raise_degraded(self, job: Job) -> None:
        ins = getattr(self.ctx, "insights", None)
        if ins is None:
            return
        spec = registry.find(job.feature)
        label = spec.label if spec else job.feature
        ins.upsert("degraded", job.feature, f"job:{job.name}", "warning", f"{label} is degraded",
                   f"The background job {job.name} failed {DEGRADED_AFTER} or more times in a row. DGXMM keeps "
                   f"retrying with a growing delay; until it succeeds, {label.lower()} findings may be missing "
                   "or stale. The DGXMM log has the details.",
                   dedupe_key=f"degraded:{job.name}", source="rules", expires_at=self.clock() + DEGRADED_TTL)

    def _resolve_degraded(self, job: Job) -> None:
        """Every success: close a degraded insight (also one left from before a restart); a no-op when none."""
        ins = getattr(self.ctx, "insights", None)
        if ins is not None:
            ins.resolve(f"degraded:{job.name}")

    # ---------- status ----------
    def status(self) -> list[dict]:
        iso = lambda d: d.isoformat() if d else None  # noqa: E731
        out = []
        for j in self._jobs.values():
            out.append({"name": j.name, "feature": j.feature, "schedule": j.cadence.describe(),
                        "enabled": j.enabled, "running": j.running, "last_run_at": iso(j.last_run_at),
                        "last_duration_ms": j.last_duration_ms, "last_result": j.last_result,
                        "next_run_at": iso(j.next_run), "failures": j.failures, "runs": j.runs,
                        "skipped": j.skipped, "budget_calls": j.budget_calls, "calls_used": j.calls_used,
                        "calls_refused": j.calls_refused})
        return out
