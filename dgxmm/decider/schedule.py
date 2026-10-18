"""Feature 7 — usage-pattern sleep/wake suggestions and user-owned schedules (spec §7).

Three insight-engine jobs (registered in ``features.wire_features``):

- ``traffic_profiles`` (nightly): ``traffic_profiles`` rows per deployment and hour of the week (0 = Monday 00:00 in
  the box's time zone, ``local_zone``) from the last 30 days of ``metric_rollups``: observed minutes, the share of them
  with request activity, requests per minute (p50 / p95, Little's law as in workload.py) and, per hour, in how many
  observed weeks it was idle (activity in at most 5 % of its minutes). Minutes without a valid rollup (the
  deployment was stopped, asleep or not scraped) are not observations: missing data is never read as idle.
- ``schedule_proposals`` (nightly, after the profiles): for deployments with at least ``MIN_DAYS`` days of data and
  no schedule yet, rules find candidate windows — runs of at least ``MIN_WINDOW_H`` hours that were idle in at least
  90 % of their observed weeks — and turn them into a sleep time (the window's start) and a wake time
  (``WAKE_LEAD_MIN`` before the first usually-active hour), grouped by time of day into one sleep/wake pair with a
  days bitmask each. The decision model decides ``propose`` / ``skip`` per candidate (state: the profile summary, the
  deployment's priority and the host's memory-pressure history) unless the rules skip it (seen in one week only,
  or every window shorter than 4 h); at or above threshold("schedule") the candidate
  becomes a proposal insight with a ``schedule`` action (``proposal`` below). With the decision model unavailable,
  only strong candidates (idle in every one of at least three observed weeks, every window at least 4 h, priority
  below 90) are proposed by the rules.
- ``schedule_runner`` (every minute, no model): runs due schedules through the deployer's normal methods (per
  deployment lock, fit check, start-up gate, protections) and never evicts anything. See ``ScheduleRunner``. The
  schedules are the user's: the feature toggle (and the decision model) only stop new profiles and proposals; the
  admin setting ``schedules.paused_all`` ("pause all schedules") stops the runner.

Proposal (``ScheduleAction.proposal``): ``{"sleep_action": "sleep"|"stop", "sleep_at": "HH:MM", "sleep_days": mask,
"wake_action": "wake"|"start", "wake_at": "HH:MM", "wake_days": mask, "timezone": IANA, "idle_hours_per_week": h,
"weeks": n}``. Accepting it (admin, the schedule dialog) creates the two schedules through
``POST /api/deployments/{id}/schedules``; the server never applies it by itself.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from functools import lru_cache
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from sqlalchemy import delete, select

from dgxmm.audit import audit
from dgxmm.db import (Deployment, DeploymentEvent, DeploymentSchedule, MetricRollup, TrafficProfile, as_aware,
                      utcnow)
from dgxmm.decider.common import arecord_rules, busy_minute, gib, percentile, plural, quoted_block, safe_decide
from dgxmm.decider.insight_engine import LOCALTIME, _wall, local_zone
from dgxmm.decider.insights import PROPOSAL_TTL, deployment_subject
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
FEATURE = "schedule"
ACTIONS = ("sleep", "wake", "stop", "start")
WAKE_OF = {"sleep": "wake", "stop": "start"}
DOWN_ACTIONS = ("sleep", "stop")
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ALL_DAYS = 0b1111111
MAX_PER_DEPLOYMENT = 16
_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_NOT_ZONES = frozenset({"localtime", "posixrules", "Factory"})  # files of some tz databases, not IANA zones
_TZ = re.compile(r"^[A-Za-z0-9_+-]{1,64}(/[A-Za-z0-9_+-]{1,64}){0,3}$")

# runner
CATCH_UP = timedelta(minutes=15)  # a slot missed by at most this much (e.g. DGXMM restarting) still runs
QUIET_MIN = 10                    # a sleep/stop needs no request activity for this many minutes
LIVE_FRESH_S = 90.0               # ... and a live metrics sample at most this old showing nothing running/waiting
FIT_INSIGHT_TTL = timedelta(hours=24)
RETRY_WINDOW = timedelta(minutes=60)  # a deferred slot (traffic, metrics, a transition) is retried this long
ENGINE_FAILURES = frozenset({"engine_error", "docker_error"})
STATE_CHANGED = frozenset({"not_ready", "not_stopped", "not_sleeping", "not_found"})  # someone acted meanwhile
PENDING_WAKE_LIMIT = timedelta(minutes=15)
PENDING_MARGIN = timedelta(minutes=10)  # added to the start-up queue + start-up timeouts: a pending start is
# reported overdue only after that limit (_pending_limit); a pending wake uses PENDING_WAKE_LIMIT
PAUSE_KEY = "schedules.paused_all"  # admin setting: no schedule runs while it is set
STABLE = {"sleep": ("ready",), "stop": ("ready", "sleeping"), "wake": ("sleeping",), "start": ("stopped", "failed")}
TARGET = {"sleep": ("sleeping", "stopped"), "stop": ("stopped",), "wake": ("ready",), "start": ("ready",)}
TRANSIENT = ("planned", "queued", "starting", "loading", "stopping", "unknown")

# profiles and proposals
PROFILE_DAYS = 30
MIN_DAYS = 7                 # proposals need at least this many local days with data
MIN_WEEK_MINUTES = 30        # an hour counts as observed in a week with at least this many observed minutes
IDLE_ACTIVE_MAX = 0.05       # an hour is idle in a week when at most this share of its minutes had activity
IDLE_WEEKS_MIN = 0.9         # a candidate hour was idle in at least this share of its observed weeks
MIN_WINDOW_H = 2
WAKE_LEAD_MIN = 15
MAX_CANDIDATES = 3           # per deployment and night
RECUR_MIN_WEEKS = 2          # rules skip a candidate seen in fewer weeks (one week says nothing about recurrence)
WORTH_MIN_WINDOW_H = 4       # ... or whose windows are all shorter (a sleep of under ~4 h is not worth a schedule)
STRONG_MIN_WEEKS = 3         # rules-only proposals (decision model unavailable): idle in every one of 3+ weeks,
STRONG_MIN_WINDOW_H = 4      # every window at least 4 h, priority below STRONG_MAX_PRIORITY
STRONG_MAX_PRIORITY = 90
PRESSURE_SHARE = 0.15        # host "under pressure": MemAvailable below this share of total memory
PRESSURE_DAYS = 7
QUESTION = ("Should DGXMM suggest this sleep/wake schedule to the admin? Suggest it only when the deployment is "
            "reliably idle in these windows week after week, so sleeping it then frees memory without "
            "disturbing anyone.")
OPTIONS = [Option("propose", "propose", "Suggest the schedule: the windows were idle in every observed week, with at "
                                        "least two weeks of data, each window lasts several hours, and the "
                                        "deployment is not one that must stay available (priority below 90)"),
           Option("skip", "skip", "Do not suggest it: only one week of data, activity in some of the weeks, windows "
                                  "of only a couple of hours, or a high-priority deployment that should stay "
                                  "available")]


class ScheduleError(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# ---------------- small helpers ----------------
def parse_hhmm(s: str) -> tuple[int, int]:
    m = _HHMM.match(s or "")
    if not m:
        raise ValueError(f"time {s!r}: HH:MM (00:00-23:59)")
    return int(m.group(1)), int(m.group(2))


@lru_cache(maxsize=1)
def _zones() -> frozenset[str]:
    return frozenset(available_timezones())


def valid_timezone(name: str) -> bool:
    """An IANA time zone name of the tz database (no paths, no "localtime")."""
    if not isinstance(name, str) or not _TZ.match(name) or name not in _zones() or name in _NOT_ZONES \
            or name.startswith(("posix/", "right/")):
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False
    return True


def zone_name(tz: tzinfo) -> str:
    """The IANA name of a zone (local_zone() of /etc/localtime has key "localtime": UTC is the safe answer then)."""
    key = getattr(tz, "key", None)
    return key if isinstance(key, str) and valid_timezone(key) else "UTC"


def box_zone(ctx) -> tzinfo:
    """The box's time zone as a *named* IANA zone, so profiles (local hours) and the schedules proposed from them
    (``timezone``) agree. ``local_zone`` may read /etc/localtime as a file (key "localtime"): its name is then taken
    from the symlink target; if there is none, UTC is used for both."""
    z = local_zone(getattr(getattr(ctx, "settings", None), "tz", None))
    key = getattr(z, "key", None)
    if isinstance(key, str) and valid_timezone(key):
        return z
    if key == "localtime":
        target = os.path.realpath(LOCALTIME)
        name = target.split("zoneinfo/", 1)[1] if "zoneinfo/" in target else None
        if name and valid_timezone(name):
            return ZoneInfo(name)
    return ZoneInfo("UTC")


def describe_days(mask: int) -> str:
    """"Mon–Fri", "Sat, Sun", "every day"."""
    days = [i for i in range(7) if mask >> i & 1]
    if len(days) == 7:
        return "every day"
    if not days:
        return "no day"
    runs: list[list[int]] = []
    for d in days:
        if runs and d == runs[-1][-1] + 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    return ", ".join(f"{DAY_NAMES[r[0]]}–{DAY_NAMES[r[-1]]}" if len(r) >= 3 else
                     ", ".join(DAY_NAMES[x] for x in r) for r in runs)


def slot(s: DeploymentSchedule | dict, d: date) -> datetime:
    """The UTC instant of the schedule's wall time on local date d (DST: a time in the spring gap runs just after
    it; a time that occurs twice runs at its first occurrence)."""
    get = s.get if isinstance(s, dict) else lambda k: getattr(s, k)
    hh, mm = parse_hhmm(get("at_local"))
    return _wall(d, hh, mm, ZoneInfo(get("timezone")))


def next_run(s: DeploymentSchedule | dict, now: datetime) -> datetime | None:
    """The next slot strictly after ``now`` (UTC), skipping a slot already handled today; None when paused."""
    get = s.get if isinstance(s, dict) else lambda k: getattr(s, k)
    if not get("enabled") or not get("days"):
        return None
    zone = ZoneInfo(get("timezone"))
    today = now.astimezone(zone).date()
    for i in range(-1, 9):
        d = today + timedelta(days=i)
        if not get("days") >> d.weekday() & 1 or get("last_run_date") == d.isoformat():
            continue
        at = slot(s, d)
        if at > now:
            return at
    return None


def schedule_out(s: DeploymentSchedule, now: datetime | None = None) -> dict:
    now = now or utcnow()
    iso = lambda d: as_aware(d).isoformat() if d else None  # noqa: E731
    nxt = next_run(s, now)
    return {"id": s.id, "deployment_id": s.deployment_id, "action": s.action, "days": s.days,
            "days_label": describe_days(s.days), "at_local": s.at_local, "timezone": s.timezone,
            "enabled": bool(s.enabled), "created_by": s.created_by, "created_at": iso(s.created_at),
            "updated_at": iso(s.updated_at), "last_run_at": iso(s.last_run_at), "last_result": s.last_result,
            "deferred": (s.last_result or "").startswith("deferred:"), "pending": (s.last_result or "").startswith("pending:"),
            "next_run_at": nxt.isoformat() if nxt else None}


def allowed_actions(view: dict) -> tuple[str, ...]:
    """What a schedule may do to this deployment: sleep/wake (and stop/start) where the engine runs it with sleep
    mode, else stop/start only. System deployments (the decision model) get none."""
    if view.get("system") or view.get("engine") == "decision":
        return ()
    return ("sleep", "wake", "stop", "start") if view.get("sleep_capable") else ("stop", "start")


# ---------------- CRUD ----------------
@dataclass
class ScheduleIn:
    action: str
    days: int
    at_local: str
    timezone: str
    enabled: bool = True


def _validate(item: ScheduleIn, view: dict) -> None:
    if item.action not in ACTIONS:
        raise ScheduleError("invalid_action", f"action must be one of {', '.join(ACTIONS)}")
    if isinstance(item.days, bool) or not isinstance(item.days, int) or not 1 <= item.days <= ALL_DAYS:
        raise ScheduleError("invalid_days", "days is a bitmask of weekdays (1 = Monday .. 64 = Sunday), 1-127")
    try:
        parse_hhmm(item.at_local)
    except ValueError as e:
        raise ScheduleError("invalid_time", str(e)) from None
    if not valid_timezone(item.timezone):
        raise ScheduleError("invalid_timezone", f"unknown time zone {item.timezone!r} (an IANA name, e.g. "
                                                "Europe/Berlin)")
    allowed = allowed_actions(view)
    if not allowed:
        raise ScheduleError("system_deployment", f"{view['name']} is a system deployment; schedules never act on "
                                                 "it", 409)
    if item.action not in allowed:
        hint = "its engine does not run it with sleep mode: use stop / start"
        raise ScheduleError("action_not_allowed", f"{item.action} is not possible for {view['name']}: {hint}")


def _overlaps(a, b) -> bool:
    return a.timezone == b.timezone and a.at_local == b.at_local and bool(a.days & b.days)


class ScheduleService:
    """Schedules (CRUD + validation), the runner, traffic profiles and proposals. ``ctx.schedules``."""

    def __init__(self, ctx, *, clock: Callable[[], datetime] = utcnow):
        self.ctx, self.clock = ctx, clock

    @property
    def db(self):
        return self.ctx.db

    # ---------- reads ----------
    def list(self, dep_id: int) -> list[dict]:
        now = self.clock()
        with self.db.session() as s:
            rows = s.scalars(select(DeploymentSchedule).where(DeploymentSchedule.deployment_id == dep_id)
                             .order_by(DeploymentSchedule.at_local, DeploymentSchedule.id))
            return [schedule_out(r, now) for r in rows]

    def get(self, sid: int) -> DeploymentSchedule | None:
        with self.db.session() as s:
            return s.get(DeploymentSchedule, sid)

    # ---------- writes (the API audits them) ----------
    def create(self, view: dict, items: list[ScheduleIn], user_id: int | None) -> list[dict]:
        """Create one or more schedules of a deployment atomically (a proposal is a sleep plus a wake row)."""
        for it in items:
            _validate(it, view)
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if _overlaps(a, b):
                    raise ScheduleError("schedule_conflict", f"two schedules at {a.at_local} on the same day", 409)
        now = self.clock()
        with self.db.session() as s:
            existing = list(s.scalars(select(DeploymentSchedule)
                                      .where(DeploymentSchedule.deployment_id == view["id"])))
            if len(existing) + len(items) > MAX_PER_DEPLOYMENT:
                raise ScheduleError("too_many_schedules", f"at most {MAX_PER_DEPLOYMENT} schedules per deployment",
                                    409)
            for it in items:  # paused schedules do not collide (e.g. stop/start replacing paused sleep/wake)
                clash = next((e for e in existing if e.enabled and it.enabled and _overlaps(e, it)), None)
                if clash is not None:
                    raise ScheduleError("schedule_conflict", f"schedule {clash.id} already runs at {it.at_local} "
                                                             f"({describe_days(clash.days & it.days)})", 409)
            rows = [DeploymentSchedule(deployment_id=view["id"], action=it.action, days=it.days,
                                       at_local=it.at_local, timezone=it.timezone, enabled=bool(it.enabled),
                                       created_by=user_id, created_at=now, updated_at=now) for it in items]
            s.add_all(rows)
            s.flush()
            out = [schedule_out(r, now) for r in rows]
        self._publish(view["id"])
        return out

    def update(self, sid: int, view: dict, patch: dict) -> dict:
        now = self.clock()
        with self.db.session() as s:
            row = s.get(DeploymentSchedule, sid)
            if row is None:
                raise ScheduleError("not_found", f"schedule {sid} not found", 404)
            item = ScheduleIn(action=patch.get("action", row.action), days=patch.get("days", row.days),
                              at_local=patch.get("at_local", row.at_local),
                              timezone=patch.get("timezone", row.timezone),
                              enabled=patch.get("enabled", row.enabled))
            _validate(item, view)
            others = s.scalars(select(DeploymentSchedule).where(DeploymentSchedule.deployment_id == row.deployment_id,
                                                                DeploymentSchedule.id != sid))
            clash = next((e for e in others if e.enabled and item.enabled and _overlaps(e, item)), None)
            if clash is not None:
                raise ScheduleError("schedule_conflict", f"schedule {clash.id} already runs at {item.at_local}", 409)
            row.action, row.days, row.at_local, row.timezone, row.enabled = (item.action, item.days, item.at_local,
                                                                            item.timezone, bool(item.enabled))
            row.updated_at = now  # last_run_date is kept: a slot that ran today does not run again at a new time
            s.flush()
            out = schedule_out(row, now)
        self._publish(view["id"])
        return out

    def delete(self, sid: int) -> DeploymentSchedule | None:
        with self.db.session() as s:
            row = s.get(DeploymentSchedule, sid)
            if row is None:
                return None
            s.delete(row)
        self._publish(row.deployment_id)
        self._resolve_fit(row.id)
        return row

    def before_deployment_removed(self, dep_id: int) -> None:
        """Deployer before-removal hook: audit the schedules that go with the deployment (the row's deletion
        cascades to them), like a delete by hand."""
        with self.db.session() as s:
            ids = list(s.scalars(select(DeploymentSchedule.id).where(DeploymentSchedule.deployment_id == int(dep_id))))
        if ids:
            audit(self.db, None, "schedule_delete", str(dep_id),
                  {"actor": "deployment_removal", "schedule_ids": ids, "reason": "deployment removed"})

    def on_deployment_removed(self, dep_id: int) -> int:
        """Deployer removal hook: a removed deployment's schedules and traffic profile go with it."""
        with self.db.session() as s:
            n = s.execute(delete(DeploymentSchedule).where(DeploymentSchedule.deployment_id == int(dep_id))).rowcount
            s.execute(delete(TrafficProfile).where(TrafficProfile.deployment_id == int(dep_id)))
        return n

    # ---------- pause ----------
    def paused_all(self) -> bool:
        store = getattr(self.ctx, "settings_store", None)
        return bool(store is not None and store.get_bool(PAUSE_KEY, False))

    def set_paused_all(self, value: bool) -> None:
        self.ctx.settings_store.set_bool(PAUSE_KEY, bool(value))

    def pause_sleep_schedules(self, dep_id: int) -> list[DeploymentSchedule]:
        """Pause a deployment's enabled sleep / wake schedules (after a failed scheduled sleep or wake)."""
        now = self.clock()
        with self.db.session() as s:
            rows = list(s.scalars(select(DeploymentSchedule).where(DeploymentSchedule.deployment_id == dep_id,
                                                                   DeploymentSchedule.action.in_(("sleep", "wake")),
                                                                   DeploymentSchedule.enabled.is_(True))))
            for r in rows:
                r.enabled, r.updated_at = False, now
            s.flush()
            for r in rows:
                s.expunge(r)
        if rows:
            log.warning("paused %d sleep/wake schedule(s) of deployment %s after a failed scheduled run", len(rows),
                        dep_id)
            audit(self.db, None, "schedule_pause", str(dep_id),
                  {"actor": "schedule:runner", "schedule_ids": [r.id for r in rows], "reason": "failed sleep or wake"})
        return rows

    def _publish(self, dep_id: int) -> None:
        """SSE ``deployment`` (the Deployment detail page re-reads its schedules with the deployment)."""
        dep = getattr(self.ctx, "deployer", None)
        events = getattr(self.ctx, "events", None)
        if dep is None or events is None:
            return
        try:
            view = dep.view(dep_id)
        except Exception:
            return
        try:
            asyncio.get_running_loop()
            events.publish("deployment", view)
        except RuntimeError:  # a worker thread: the next SSE event or page load shows it
            pass

    def _resolve_fit(self, sid: int) -> None:
        ins = getattr(self.ctx, "insights", None)
        if ins is not None:
            try:
                ins.resolve(f"schedule_fit:{sid}")
                ins.resolve(f"schedule_failed:{sid}")
            except Exception:
                log.exception("could not resolve the fit insight of schedule %s", sid)

    # ---------- runner ----------
    async def run_due(self, now: datetime | None = None) -> str:
        """One runner pass (the ``schedule_runner`` job, every minute)."""
        return await ScheduleRunner(self).run(now or self.clock())

    # ---------- profiles ----------
    def build_profiles(self, now: datetime | None = None) -> dict:
        """Rebuild every non-system deployment's traffic profile (runs in a worker thread)."""
        now = now or self.clock()
        tz = box_zone(self.ctx)
        since = now - timedelta(days=PROFILE_DAYS)
        built = 0
        with self.db.session() as s:
            deps = [(d.id, bool(d.system) or d.engine == "decision") for d in s.scalars(select(Deployment))]
        for dep_id, system in deps:
            if system:
                continue
            with self.db.session() as s:
                rows = s.execute(select(MetricRollup.minute, MetricRollup.data)
                                 .where(MetricRollup.deployment_id == dep_id, MetricRollup.minute >= since)).all()
            prof = profile_from_rollups(((as_aware(m), d) for m, d in rows), tz)
            with self.db.session() as s:
                s.execute(delete(TrafficProfile).where(TrafficProfile.deployment_id == dep_id))
                s.add_all([TrafficProfile(deployment_id=dep_id, hour_of_week=h, updated_at=now, days=prof.days,
                                          **{k: v for k, v in p.items()}) for h, p in sorted(prof.hours.items())])
            built += 1 if prof.hours else 0
        return {"profiles": built}

    def profile(self, dep_id: int) -> Profile:
        with self.db.session() as s:
            rows = list(s.scalars(select(TrafficProfile).where(TrafficProfile.deployment_id == dep_id)))
        return Profile(days=max((r.days or 0 for r in rows), default=0),
                       hours={r.hour_of_week: {"requests_per_min_p50": r.requests_per_min_p50,
                                               "requests_per_min_p95": r.requests_per_min_p95,
                                               "active_fraction": r.active_fraction, "samples": r.samples,
                                               "weeks": r.weeks or 0, "idle_weeks": r.idle_weeks or 0}
                              for r in rows})

    # ---------- proposals ----------
    async def propose(self, run=None) -> str:
        """The ``schedule_proposals`` job: candidates -> rules / model -> proposal insights."""
        deployer = getattr(self.ctx, "deployer", None)
        if deployer is None:
            return "no deployer"
        decider = decider_of(self.ctx)
        tz = box_zone(self.ctx)
        views = await asyncio.to_thread(deployer.list_views)
        proposed = considered = 0
        for v in views:
            if not allowed_actions(v) or v["state"] not in ("ready", "sleeping"):
                continue
            prof, has_sched, pressure, sleep_ok = await asyncio.to_thread(self._facts, v)
            if has_sched or prof.days < MIN_DAYS:
                continue
            for cand in candidates(prof, tz)[:MAX_CANDIDATES]:
                considered += 1
                if await self._consider(decider, v, prof, cand, pressure, tz, sleep_ok):
                    proposed += 1
        return f"{proposed} proposed of {plural(considered, 'candidate')}"

    def _facts(self, v: dict) -> tuple[Profile, bool, dict, bool]:
        prof = self.profile(v["id"])
        with self.db.session() as s:
            has = s.scalar(select(DeploymentSchedule.id).where(DeploymentSchedule.deployment_id == v["id"])
                           .limit(1)) is not None
        return prof, has, memory_pressure(self.db, self.clock(), v.get("reserved_bytes") or 0), sleep_proven(self.db, v)

    async def _consider(self, decider, v: dict, prof: Profile, cand: Candidate, pressure: dict, tz,
                        sleep_ok: bool = False) -> bool:
        subject = deployment_subject(v["id"])
        key = f"schedule:{v['id']}:{cand.key}"
        source, confidence, decision_id = None, None, None
        veto = rule_skip(cand)
        model_up = veto is None and decider.feature_enabled(FEATURE) and await decider.available()
        if veto is not None:
            await arecord_rules(decider, feature=FEATURE, subject=subject, choice="skip", reason=veto)
        elif model_up:
            d = await safe_decide(decider, feature=FEATURE, subject=subject,
                                  state=build_state(v, prof, cand, pressure, zone_name(tz)), question=QUESTION,
                                  options=OPTIONS, timeout_s=10.0)
            if d.source == "model" and d.choice == "propose":
                source, confidence, decision_id = "model", d.confidence, d.decision_id
        elif cand.strong and int(v.get("priority") or 0) < STRONG_MAX_PRIORITY:
            d = await arecord_rules(decider, feature=FEATURE, subject=subject, choice="propose",
                                    reason=f"idle in every observed week ({cand.min_weeks}+), "
                                           f"{cand.idle_hours:g} h a week")
            source, confidence, decision_id = "rules", 1.0, d.decision_id
        ins = getattr(self.ctx, "insights", None)
        if ins is None:
            return False
        if source is None:
            await asyncio.to_thread(ins.resolve, key)
            return False
        proposal = cand.proposal(v, zone_name(tz), sleep_ok)
        await asyncio.to_thread(
            ins.upsert, "schedule_proposal", FEATURE, subject, "info", f"Sleep {v['name']} on a schedule?",
            proposal_body(v, cand, zone_name(tz), proposal["sleep_action"]), dedupe_key=key, confidence=confidence, source=source,
            decision_id=decision_id, action={"type": "schedule", "deployment_id": v["id"], "proposal": proposal,
                                             "label": "Review schedule"},
            expires_at=self.clock() + PROPOSAL_TTL)
        return True


# ---------------- runner ----------------
@dataclass
class Due:
    sched: DeploymentSchedule
    day: date
    intended: datetime
    missed: bool = False
    gave_up: bool = False  # a deferred slot whose retry window ended


class Defer(Exception):
    """A slot to retry at the next pass (traffic, unknown metrics, a transition in progress)."""


class ScheduleRunner:
    """One pass over the enabled schedules at ``now`` (nothing at all while every schedule is paused, ``paused_all``):

    - A schedule's slot on a local date is handled once (``last_run_date``), in its first pass at or after the slot
      and at most ``CATCH_UP`` later (e.g. after a DGXMM restart); a slot missed by more is skipped and logged. Slots
      before the schedule was created / last edited are not missed; editing a time does not re-arm today.
    - A slot that cannot run *yet* — requests running, waiting or recent, metrics that cannot tell, a transition or
      start-up queue in progress — is *deferred*: retried every pass for up to ``RETRY_WINDOW``, then given up for
      the day. The first deferral and the final outcome are events and audit rows; retries in between are not.
    - Several slots due for one deployment in the same pass collide: only the latest-intended one runs (the others
      are recorded as superseded); two different actions intended for the same instant cancel each other.
    - Never on system deployments, never when the deployment already is in the target state. A wake does not start a
      deployment that was stopped by someone else since its scheduled sleep.
    - Sleep / stop re-check the traffic inside the deployer's lock (``precheck``).
    - Wake / start use the deployer's fit check and start-up gate and never evict: if it does not fit, the run is
      skipped and a warning insight raised. A wake / start still waiting for its start-up slot is *pending* and its
      real outcome is recorded by a later pass.
    - A failed run raises a warning insight (``schedule_failed:<id>``). A failed sleep or wake also pauses the
      deployment's sleep / wake schedules and offers stop / start instead.
    - Only a definite "not found" deletes a deployment's schedules; any other error reading it is logged and retried.
    - Every handled slot is a deployment event and an audit row with actor ``schedule:<id>``.
    """

    def __init__(self, svc: ScheduleService):
        self.svc, self.ctx = svc, svc.ctx

    async def run(self, now: datetime) -> str:
        deployer = getattr(self.ctx, "deployer", None)
        if deployer is None:
            return "no deployer"
        if await asyncio.to_thread(self.svc.paused_all):
            return "paused (all schedules)"
        counts: dict[str, int] = {}
        for outcome in await self._check_pending(now):
            counts[outcome] = counts.get(outcome, 0) + 1
        due = await asyncio.to_thread(self._due, now)
        by_dep: dict[int, list[Due]] = {}
        for d in due:
            by_dep.setdefault(d.sched.deployment_id, []).append(d)
        for dep_id, items in sorted(by_dep.items()):
            for d in (x for x in items if x.missed):
                await self._record(d, now, "skipped", f"missed by {_mins(now - d.intended)} min (DGXMM was not "
                                                      "running at the time); runs again at its next slot")
                counts["missed"] = counts.get("missed", 0) + 1
            for d in (x for x in items if x.gave_up):
                why = (d.sched.last_result or "").removeprefix("deferred: ").split(" (retrying")[0]
                await self._record(d, now, "skipped", f"gave up after {_mins(RETRY_WINDOW)} min: {why}")
                counts["gave up"] = counts.get("gave up", 0) + 1
            live = sorted((d for d in items if not d.missed and not d.gave_up), key=lambda d: (d.intended, d.sched.id))
            if not live:
                continue
            last = live[-1]
            rivals = [d for d in live if d.intended == last.intended and _kind(d.sched.action) != _kind(last.sched.action)]
            if rivals:
                for d in live:
                    other = ", ".join(f"#{x.sched.id}" for x in live if x is not d)
                    await self._record(d, now, "skipped", f"conflicts with schedule {other} at the same time; "
                                                          "nothing was done")
                counts["conflict"] = counts.get("conflict", 0) + len(live)
                continue
            for d in live[:-1]:
                await self._record(d, now, "skipped", f"superseded by schedule #{last.sched.id} "
                                                      f"({last.sched.action} at {last.sched.at_local})")
                counts["superseded"] = counts.get("superseded", 0) + 1
            if await asyncio.to_thread(self.svc.paused_all):  # paused while this pass ran: stop here
                counts["paused"] = counts.get("paused", 0) + 1
                break
            outcome = await self._execute(last, now)
            counts[outcome] = counts.get(outcome, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing due"

    # ---------- what is due ----------
    def _due(self, now: datetime) -> list[Due]:
        out: list[Due] = []
        with self.svc.db.session() as s:
            rows = list(s.scalars(select(DeploymentSchedule).where(DeploymentSchedule.enabled.is_(True))))
            for r in rows:
                s.expunge(r)
        for r in rows:
            try:
                zone = ZoneInfo(r.timezone)
            except (ZoneInfoNotFoundError, ValueError, OSError):
                log.warning("schedule %s has an unknown time zone %r; skipped", r.id, r.timezone)
                continue
            armed = as_aware(r.updated_at or r.created_at)
            today = now.astimezone(zone).date()
            for d in (today - timedelta(days=1), today):  # a slot just before local midnight is still catchable
                if not r.days >> d.weekday() & 1 or r.last_run_date == d.isoformat():
                    continue
                if r.last_run_date is not None and r.last_run_date > d.isoformat():
                    continue
                try:
                    at = slot(r, d)
                except ValueError:
                    continue
                if at > now or at < armed:
                    continue
                deferred = (r.last_result or "").startswith("deferred:") and r.last_run_at is not None \
                    and as_aware(r.last_run_at) >= at
                if deferred:
                    out.append(Due(r, d, at, gave_up=now - at > RETRY_WINDOW))
                else:
                    out.append(Due(r, d, at, missed=now - at > CATCH_UP))
        return out

    # ---------- doing it ----------
    async def _execute(self, d: Due, now: datetime) -> str:
        from dgxmm.deployer.service import DeployError
        deployer = self.ctx.deployer
        s = d.sched
        try:
            view = await asyncio.to_thread(deployer.view, s.deployment_id)
        except DeployError as e:
            if e.code != "not_found":
                log.warning("schedule %s: could not read deployment %s (%s); retrying", s.id, s.deployment_id, e.code)
                return "error"
            await asyncio.to_thread(self.svc.on_deployment_removed, s.deployment_id)
            log.info("schedule %s: deployment %s no longer exists; its schedules were deleted", s.id, s.deployment_id)
            return "removed"
        except Exception:  # e.g. "database is locked": never a reason to delete the user's schedules
            log.warning("schedule %s: could not read deployment %s; retrying at the next pass", s.id,
                        s.deployment_id, exc_info=True)
            return "error"
        if not allowed_actions(view):
            await self._record(d, now, "skipped", f"{view['name']} is a system deployment; schedules never act on it",
                               view)
            return "skipped"
        state = view["state"]
        if s.action == "wake" and state in ("stopped", "failed"):
            if state == "failed" or not await asyncio.to_thread(self._stopped_by_schedule, view["id"]):
                await self._record(d, now, "skipped", f"{'failed' if state == 'failed' else 'stopped by user'} since "
                                                      "the last scheduled sleep; not started", view)
                return "skipped"
        action = effective_action(s.action, view)
        if state in TARGET[action]:
            await self._record(d, now, "skipped", f"already {state}; nothing to do", view)
            return "skipped"
        busy = self._in_transition(view)
        if busy:
            return await self._defer(d, now, busy, view)
        if state not in STABLE[action]:
            await self._record(d, now, "skipped", f"{view['name']} is {state}; cannot {action} it", view)
            return "skipped"
        if action in DOWN_ACTIONS and state == "ready":
            why = await asyncio.to_thread(self._traffic_reason, view, now)
            if why:
                return await self._defer(d, now, why, view)
        return await self._apply(d, now, action, view)

    async def _defer(self, d: Due, now: datetime, why: str, view: dict) -> str:
        until = (d.intended + RETRY_WINDOW).astimezone(ZoneInfo(d.sched.timezone))
        await self._record(d, now, "deferred", f"{why} (retrying until {until:%H:%M})", view)
        return "deferred"

    def _stopped_by_schedule(self, dep_id: int) -> bool:
        """The deployment's latest stop was requested by a schedule: the deployer records who asked for a stop in
        the event's ``actor`` ("schedule:3", "user", ...); message text is never parsed. Stops recorded before the
        column existed (actor NULL) count as not by a schedule, so a wake never starts them."""
        with self.svc.db.session() as sess:
            actor = sess.scalar(select(DeploymentEvent.actor).where(
                DeploymentEvent.deployment_id == dep_id, DeploymentEvent.to_state.in_(("stopping", "stopped")),
                DeploymentEvent.actor.is_not(None)).order_by(DeploymentEvent.id.desc()).limit(1))
        return bool(actor and actor.startswith("schedule:"))

    def _in_transition(self, view: dict) -> str | None:
        dep = self.ctx.deployer
        i = view["id"]
        if view["state"] in TRANSIENT:
            return f"{view['name']} is {view['state']} (mid-transition)"
        if view.get("startup_queue"):
            return f"{view['name']} is waiting for a start-up slot"
        waking = (getattr(dep, "_waking", None) or {}).get(i)
        if waking is not None and not waking.done():
            return f"{view['name']} is waking up"
        queued = (getattr(dep, "_queued", None) or {}).get(i)
        if queued is not None and not queued.done():
            return f"{view['name']} is queued to start"
        if i in (getattr(dep, "launching", None) or {}):
            return f"{view['name']} is being launched"
        if i in (getattr(dep, "rotating", None) or set()):
            return f"{view['name']} is being restarted for a key rotation"
        return None

    def _traffic_reason(self, view: dict, now: datetime) -> str | None:
        """Why a sleep/stop must not run now (traffic), or None when the deployment is demonstrably quiet."""
        from dgxmm.decider.plans import traffic
        store = getattr(self.ctx, "metrics", None)
        if store is None:
            return "no metrics: cannot tell whether requests are arriving"
        ts = now.timestamp()
        try:
            ring = store.series(view["id"])
        except Exception:
            ring = []
        latest = ring[-1] if ring else None
        if latest is None or ts - (latest.get("ts") or 0) > LIVE_FRESH_S or latest.get("running") is None:
            return "no fresh metrics: cannot tell whether requests are running"
        if (latest.get("running") or 0) > 0 or (latest.get("waiting") or 0) > 0:
            return f"{int(latest.get('running') or 0)} request(s) running, {int(latest.get('waiting') or 0)} waiting"
        t = traffic(store, view["id"], ts)
        if t.running_now:
            return "requests are running or waiting"
        if t.last_min is not None and t.last_min < QUIET_MIN:
            return f"a request arrived {max(int(t.last_min), 0)} min ago (needs {QUIET_MIN} quiet minutes)"
        if not t.known:
            return "not enough recent metrics to tell whether it is idle"
        return None

    async def _apply(self, d: Due, now: datetime, action: str, view: dict) -> str:
        from dgxmm.deployer.service import DeployError, InsufficientMemory
        dep = self.ctx.deployer
        s = d.sched

        async def precheck():  # under the deployer's lock: a request may have arrived since the check above
            return await asyncio.to_thread(self._traffic_reason, view, now)
        try:
            if action == "sleep":
                out = await dep.sleep(s.deployment_id, None, precheck=precheck)
                result = "put to sleep"
            elif action == "stop":
                out = await dep.stop(s.deployment_id, None, reason=f"schedule:{s.id}", precheck=precheck)
                result = "stopped"
            elif action == "wake":
                getattr(dep, "wake_failures", {}).pop(s.deployment_id, None)  # only this wake-up's outcome counts
                out = await dep.wake(s.deployment_id, None)
                result = "woken up"
            else:
                out = await dep.restart(s.deployment_id, None)
                result = "started"
        except InsufficientMemory as e:
            fit = e.fit
            need, free = getattr(fit, "required", None), getattr(fit, "free_for_new", None)
            msg = "does not fit in free memory" + (f" (needs {gib(need)}, {gib(free)} free)"
                                                   if need is not None and free is not None else "")
            await self._record(d, now, "skipped", f"{msg}; nothing was stopped to make room", view)
            await asyncio.to_thread(self._raise_fit, s, view, need, free)
            return "skipped"
        except DeployError as e:
            if e.code == "precheck":
                return await self._defer(d, now, e.message, view)
            if e.code in STATE_CHANGED:  # the user (or a plan) changed it meanwhile: no warning
                await self._record(d, now, "skipped", f"state changed: {e.message}"[:200], view)
                return "skipped"
            detail = f"{action} refused: {e.message}"[:200]
            await self._record(d, now, "failed", detail, view)
            await asyncio.to_thread(self._failed, s, view, action, detail, e.code in ENGINE_FAILURES)
            return "failed"
        except Exception as e:  # never let one schedule stop the pass
            log.exception("schedule %s: %s of deployment %s failed", s.id, action, s.deployment_id)
            detail = f"{action} failed: {type(e).__name__}"
            await self._record(d, now, "failed", detail, view)
            await asyncio.to_thread(self._failed, s, view, action, detail, True)
            return "failed"
        out = out if isinstance(out, dict) else view
        if action in ("wake", "start") and out.get("state") != "ready" and \
                (out.get("startup_queue") or out.get("state") in ("queued", "starting", "loading", "sleeping")):
            what = "waiting for a start-up slot" if out.get("startup_queue") or out.get("state") in ("queued",
                                                                                                  "sleeping") \
                else f"{out.get('state')}"
            await self._record(d, now, "pending", f"{'waking up' if action == 'wake' else 'starting'} ({what})", out,
                               before=view)
            return "pending"
        await self._record(d, now, "ok", result, out, before=view)
        await asyncio.to_thread(self.svc._resolve_fit, s.id)
        return "ran"

    def _pending_limit(self, action: str) -> timedelta:
        """How long a pending wake / start may wait before it is reported overdue (it is still followed after)."""
        st = getattr(self.ctx, "settings", None)
        if action == "wake":
            return PENDING_WAKE_LIMIT
        return timedelta(seconds=float(getattr(st, "startup_queue_timeout_s", 1800)) +
                         float(getattr(st, "startup_timeout_s", 900))) + PENDING_MARGIN

    async def _check_pending(self, now: datetime) -> list[str]:
        """The real outcome of wakes / starts that an earlier pass left waiting (persisted as ``pending:``, so a
        DGXMM restart re-checks them). Failures are classified like the immediate path: only engine errors of a
        wake pause the sleep schedules; a full start-up queue is a plain failure; the user acting meanwhile is
        "state changed"; a wake lost with a restart is retried within the retry window."""
        from dgxmm.deployer.service import DeployError
        def pending_rows():
            with self.svc.db.session() as sess:
                rows = list(sess.scalars(select(DeploymentSchedule)
                                         .where(DeploymentSchedule.last_result.like("pending:%"))))
                for r in rows:
                    sess.expunge(r)
            return rows
        rows = await asyncio.to_thread(pending_rows)
        outcomes = []
        dep = self.ctx.deployer
        for s in rows:
            try:
                view = await asyncio.to_thread(dep.view, s.deployment_id)
            except DeployError:
                continue
            except Exception:
                log.warning("schedule %s: could not check its pending run; retrying", s.id, exc_info=True)
                continue
            day = date.fromisoformat(s.last_run_date) if s.last_run_date else now.date()
            d = Due(s, day, slot(s, day))
            overdue = "overdue" in (s.last_result or "")
            state = view["state"]
            failure = getattr(dep, "wake_failures", {}).pop(view["id"], None) if s.action == "wake" else None
            if state == "ready":
                await self._record(d, now, "ok", ("woken up" if s.action == "wake" else "started")
                                   + " (after waiting for a start-up slot)", view)
                await asyncio.to_thread(self.svc._resolve_fit, s.id)
                outcomes.append("ran")
                continue
            if failure is None and (state in ("queued", "starting", "loading") or self._in_transition(view)):
                # still on its way: never declared failed while queued / starting; overdue is reported once
                if state not in ("queued", "starting", "loading") and not overdue and s.last_run_at is not None \
                        and now - as_aware(s.last_run_at) > self._pending_limit(s.action):
                    await self._record(d, now, "pending", f"overdue: still waiting after "
                                                          f"{_mins(self._pending_limit(s.action))} min", view)
                    await asyncio.to_thread(self._failed, s, view, s.action, "still waiting for a start-up slot",
                                            False)
                    outcomes.append("overdue")
                continue
            verb = "wake" if s.action == "wake" else "start"
            if failure is not None:
                code, msg = failure
                if code == "insufficient_memory":
                    await self._record(d, now, "skipped", f"does not fit in free memory ({msg}); nothing was "
                                                          "stopped to make room", view)
                    await asyncio.to_thread(self._raise_fit, s, view, None, None)
                    outcomes.append("skipped")
                elif code in STATE_CHANGED:
                    await self._record(d, now, "skipped", f"state changed: {msg}"[:200], view)
                    outcomes.append("skipped")
                elif code == "startup_busy":
                    await self._record(d, now, "failed", f"queue: {msg}"[:200], view)
                    await asyncio.to_thread(self._failed, s, view, verb, f"queue: {msg}", False)
                    outcomes.append("failed")
                else:
                    await self._record(d, now, "failed", f"{verb} failed: {msg}"[:200], view)
                    await asyncio.to_thread(self._failed, s, view, verb, msg, code in ENGINE_FAILURES or
                                            code == "error")
                    outcomes.append("failed")
                continue
            if s.action == "wake" and state == "sleeping":  # no task, no failure: lost (e.g. DGXMM restarted)
                if now - d.intended <= RETRY_WINDOW:
                    log.info("schedule %s: its pending wake-up was lost (DGXMM restarted?); waking again", s.id)
                    outcomes.append(await self._apply(d, now, "wake", view))
                else:
                    await self._record(d, now, "failed", "wake-up lost (DGXMM restarted) and the retry window "
                                                         "passed", view)
                    await asyncio.to_thread(self._failed, s, view, verb, "the wake-up was lost", False)
                    outcomes.append("failed")
                continue
            if s.action != "wake" and state == "failed":
                why = view.get("last_error") or "the start failed"
                await self._record(d, now, "failed", f"start failed: {why}"[:200], view)
                await asyncio.to_thread(self._failed, s, view, verb, why, False)
                outcomes.append("failed")
                continue
            await self._record(d, now, "skipped", f"state changed: {view['name']} is {state}", view)
            outcomes.append("skipped")
        return outcomes

    def _failed(self, s: DeploymentSchedule, view: dict, action: str, detail: str, engine: bool) -> None:
        """A warning insight for a failed run; a failed sleep / wake (engine trouble) also pauses the deployment's
        sleep / wake schedules and offers the same times as stop / start."""
        ins = getattr(self.ctx, "insights", None)
        paused: list[DeploymentSchedule] = []
        if engine and action in ("sleep", "wake"):
            paused = self.svc.pause_sleep_schedules(view["id"])
        if ins is None:
            return
        body = f"Schedule #{s.id} ({s.action} at {s.at_local}, {s.timezone}) failed: {detail}."
        act: dict = {"type": "open", "path": f"/deployments/{view['id']}"}
        if paused:
            body += (f" Its sleep / wake schedules ({len(paused)}) were paused so the deployment is not left asleep "
                     "and unusable. Stopping and starting it at the same times frees the same memory without sleep "
                     "mode.")
            sl = next((p for p in paused if p.action == "sleep"), None)
            wk = next((p for p in paused if p.action == "wake"), None)
            if sl is not None and wk is not None and sl.timezone == wk.timezone:
                act = {"type": "schedule", "deployment_id": view["id"], "label": "Use stop / start",
                       "proposal": {"sleep_action": "stop", "sleep_at": sl.at_local, "sleep_days": sl.days,
                                    "wake_action": "start", "wake_at": wk.at_local, "wake_days": wk.days,
                                    "timezone": sl.timezone}}
        try:
            ins.upsert("schedule_failed", FEATURE, deployment_subject(view["id"]), "warning",
                       f"Schedule could not {action} {view['name']}", body[:4000], dedupe_key=f"schedule_failed:{s.id}",
                       source="rules", action=act, expires_at=self.svc.clock() + FIT_INSIGHT_TTL)
        except Exception:
            log.exception("could not raise the failure insight of schedule %s", s.id)

    def _raise_fit(self, s: DeploymentSchedule, view: dict, need, free) -> None:
        ins = getattr(self.ctx, "insights", None)
        if ins is None:
            return
        verb = "wake" if s.action == "wake" else "start"
        body = (f"Schedule #{s.id} should {verb} {view['name']} at {s.at_local} ({s.timezone}), but it did not fit in "
                f"free memory" + (f" (it needs {gib(need)}, {gib(free)} were free)" if need is not None and
                                  free is not None else "") +
                ". Schedules never stop other deployments to make room: free memory and start it by hand, or "
                "change the schedule. The next slot is tried as usual.")
        try:
            ins.upsert("schedule_skipped", FEATURE, deployment_subject(view["id"]), "warning",
                       f"Schedule could not {verb} {view['name']}", body, dedupe_key=f"schedule_fit:{s.id}",
                       source="rules", action={"type": "open", "path": f"/deployments/{view['id']}"},
                       expires_at=self.svc.clock() + FIT_INSIGHT_TTL)
        except Exception:
            log.exception("could not raise the fit insight of schedule %s", s.id)

    async def _record(self, d: Due, now: datetime, status: str, detail: str, view: dict | None = None,
                      before: dict | None = None) -> None:
        await asyncio.to_thread(self._write, d, now, status, detail, view, before)
        if view is not None:
            try:
                self.ctx.events.publish("deployment", await asyncio.to_thread(self.ctx.deployer.view, view["id"]))
            except Exception:
                pass

    def _write(self, d: Due, now: datetime, status: str, detail: str, view: dict | None, before: dict | None) -> None:
        s = d.sched
        result = f"{status}: {detail}"[:256]
        local = d.intended.astimezone(ZoneInfo(s.timezone))
        msg = f"Schedule #{s.id} ({s.action} at {s.at_local} {s.timezone}, {local:%a %Y-%m-%d}): {result}"
        actor = f"schedule:{s.id}"
        retry = status == "deferred" and (s.last_result or "").startswith("deferred:") and s.last_run_at is not None \
            and as_aware(s.last_run_at) >= d.intended  # a retry of an already deferred slot: no new event / audit
        if not retry:
            (log.info if status in ("ok", "pending", "deferred") else log.warning)("%s", msg)
        with self.svc.db.session() as sess:
            row = sess.get(DeploymentSchedule, s.id)
            if row is not None:
                row.last_run_at, row.last_result = now, result
                if status != "deferred":
                    row.last_run_date = d.day.isoformat()
            if retry:
                return
            dep = sess.get(Deployment, s.deployment_id)
            if dep is not None:
                frm = (before or view or {}).get("state") or dep.state
                sess.add(DeploymentEvent(deployment_id=dep.id, from_state=frm, to_state=dep.state, message=msg[:2000]))
        name = (view or {}).get("name") or str(s.deployment_id)
        audit(self.svc.db, None, "schedule_run", name,
              {"actor": actor, "schedule_id": s.id, "deployment_id": s.deployment_id, "action": s.action,
               "status": status, "result": detail[:200], "slot": d.intended.isoformat()})


def _kind(action: str) -> str:
    return "down" if action in DOWN_ACTIONS else "up"


def effective_action(action: str, view: dict) -> str:
    """Sleep where the deployment runs with sleep mode, else stop; wake a sleeping one, else start (the runner
    starts a stopped one only when a schedule stopped it)."""
    if action == "sleep":
        return "sleep" if view.get("sleep_capable") else "stop"
    if action == "wake":
        return "wake" if view.get("state") == "sleeping" or view.get("sleep_capable") and view["state"] == "ready" \
            else "start"
    return action


def _mins(td: timedelta) -> int:
    return max(int(td.total_seconds() // 60), 0)


# ---------------- profiles ----------------
@dataclass
class Profile:
    days: int
    hours: dict[int, dict] = field(default_factory=dict)  # hour_of_week -> row fields


def _rpm(r: dict) -> float | None:
    """Requests per minute of one rollup minute (Little's law; workload.per_request); 0 when idle."""
    if not busy_minute(r):
        return 0.0
    in_system = (r.get("running") or 0) + (r.get("waiting") or 0)
    e2e = r.get("e2e_p50")
    if not in_system or not e2e:
        return None
    return in_system / e2e * 60


def profile_from_rollups(rows, tz: tzinfo) -> Profile:
    """Profile of (aware minute, data) rollups. A minute is observed when its rollup has samples and a reading of
    ``running`` (the plan safety filter's definition of coverage)."""
    acc: dict[int, dict] = {}
    days: set[date] = set()
    for minute, data in rows:
        if not isinstance(data, dict) or not data.get("samples") or data.get("running") is None:
            continue
        loc = minute.astimezone(tz)
        how = loc.weekday() * 24 + loc.hour
        week = loc.date() - timedelta(days=loc.weekday())
        a = acc.setdefault(how, {"obs": 0, "act": 0, "rpm": [], "weeks": {}})
        busy = busy_minute(data)
        a["obs"] += 1
        a["act"] += busy
        r = _rpm(data)
        if r is not None:
            a["rpm"].append(r)
        w = a["weeks"].setdefault(week, [0, 0])
        w[0] += 1
        w[1] += busy
        days.add(loc.date())
    hours = {}
    for how, a in acc.items():
        weeks = [w for w in a["weeks"].values() if w[0] >= MIN_WEEK_MINUTES]
        hours[how] = {"requests_per_min_p50": percentile(a["rpm"], 0.5, digits=3),
                      "requests_per_min_p95": percentile(a["rpm"], 0.95, digits=3),
                      "active_fraction": round(a["act"] / a["obs"], 4), "samples": a["obs"], "weeks": len(weeks),
                      "idle_weeks": sum(1 for o, b in weeks if b / o <= IDLE_ACTIVE_MAX)}
    return Profile(days=len(days), hours=hours)


def memory_pressure(db, now: datetime, reserved: int) -> dict:
    """The host's memory history over the last PRESSURE_DAYS days (host rollups): share of minutes under pressure
    (MemAvailable < 15 % of total) and share where the free memory for a new deployment was below this
    deployment's reservation."""
    since = now - timedelta(days=PRESSURE_DAYS)
    with db.session() as s:
        rows = [d for (d,) in s.execute(select(MetricRollup.data).where(MetricRollup.deployment_id.is_(None),
                                                                       MetricRollup.minute >= since))]
    n = low = short = 0
    for d in rows:
        used, avail = (d or {}).get("used"), (d or {}).get("available")
        if used is None or avail is None or used + avail <= 0:
            continue
        n += 1
        low += avail < PRESSURE_SHARE * (used + avail)
        ffn = d.get("free_for_new")
        short += ffn is not None and reserved > 0 and ffn < reserved
    return {"minutes": n, "pressure_share": round(low / n, 4) if n else None,
            "short_share": round(short / n, 4) if n else None}


# ---------------- candidates ----------------
@dataclass
class Candidate:
    sleep_at: str
    wake_at: str
    sleep_days: int
    wake_days: int
    windows: list[tuple[int, int]]  # (start hour of week, length h)
    idle_hours: float               # sleeping hours a week (sleep -> wake)
    min_weeks: int
    all_idle: bool                  # idle in every observed week of every hour
    max_active: float               # highest active_fraction of a window hour
    busy_rpm_p50: float | None      # typical request rate in the deployment's active hours

    @property
    def key(self) -> str:
        return f"{self.sleep_at}-{self.wake_at}-{self.sleep_days}-{self.wake_days}"

    @property
    def strong(self) -> bool:
        return self.all_idle and self.min_weeks >= STRONG_MIN_WEEKS and \
            all(ln >= STRONG_MIN_WINDOW_H for _, ln in self.windows)

    def proposal(self, view: dict, tz_name: str, sleep_ok: bool = False) -> dict:
        """Sleep/wake only where sleep mode is on and proven on this deployment (``sleep_proven``), else
        stop/start."""
        down = "sleep" if view.get("sleep_capable") and sleep_ok else "stop"
        return {"sleep_action": down, "sleep_at": self.sleep_at, "sleep_days": self.sleep_days,
                "wake_action": WAKE_OF[down], "wake_at": self.wake_at, "wake_days": self.wake_days,
                "timezone": tz_name, "idle_hours_per_week": round(self.idle_hours, 1), "weeks": self.min_weeks}


def sleep_proven(db, view: dict) -> bool:
    """Sleep mode is known to work for this deployment: it runs with --enforce-eager, or it went to sleep and woke
    up again before (deployment events). Some engine images crash or wedge on sleep with CUDA graphs, so a schedule
    proposes stop/start otherwise."""
    if not view.get("sleep_capable"):
        return False
    if "--enforce-eager" in ((view.get("config") or {}).get("extra_args") or []):
        return True
    with db.session() as s:
        evs = [(e.from_state, e.to_state) for e in s.scalars(
            select(DeploymentEvent).where(DeploymentEvent.deployment_id == view["id"]).order_by(DeploymentEvent.id))]
    slept = False
    for frm, to in evs:
        if frm == "ready" and to == "sleeping":
            slept = True
        elif slept and frm == "sleeping" and to == "ready":
            return True
    return False


def rule_skip(c: Candidate) -> str | None:
    """Why the rules skip a candidate without asking the model (None: the model decides)."""
    if c.min_weeks < RECUR_MIN_WEEKS:
        return "seen in one week only"
    if max(ln for _, ln in c.windows) < WORTH_MIN_WINDOW_H:
        return f"windows shorter than {WORTH_MIN_WINDOW_H} h"
    return None


def _idle(h: dict | None) -> bool:
    return bool(h and h["weeks"] >= 1 and h["idle_weeks"] / h["weeks"] >= IDLE_WEEKS_MIN)


def candidates(prof: Profile, tz: tzinfo | None = None) -> list[Candidate]:
    """Idle windows (>= MIN_WINDOW_H consecutive idle hours of the week, wrapping around Sunday night) grouped by
    their sleep and wake times of day into sleep/wake pairs, longest first. A deployment idle all week gets none
    (a schedule would never wake it: stopping it is the better advice)."""
    idle = [_idle(prof.hours.get(h)) for h in range(168)]
    if all(idle) or not any(idle):
        return []
    start = next(h for h in range(168) if not idle[h])  # begin the scan at an active hour: no run is split
    runs: list[tuple[int, int]] = []
    cur = None
    for i in range(1, 169):
        h = (start + i) % 168
        if idle[h]:
            cur = (h, 1) if cur is None else (cur[0], cur[1] + 1)
        elif cur is not None:
            runs.append(cur)
            cur = None
    runs = [r for r in runs if r[1] >= MIN_WINDOW_H]
    busy = [prof.hours[h]["requests_per_min_p50"] for h in range(168) if not idle[h] and h in prof.hours
            and prof.hours[h]["requests_per_min_p50"] is not None]
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for st, ln in runs:
        end_min = ((st + ln) * 60 - WAKE_LEAD_MIN) % (168 * 60)
        sleep_key = f"{st % 24:02d}:00"
        wake_key = f"{end_min % 1440 // 60:02d}:{end_min % 60:02d}"
        groups.setdefault((sleep_key, wake_key), []).append((st, ln))
    out = []
    for (sa, wa), wins in groups.items():
        sdays = wdays = 0
        hours_in = []
        for st, ln in wins:
            sdays |= 1 << (st // 24)
            end_min = ((st + ln) * 60 - WAKE_LEAD_MIN) % (168 * 60)
            wdays |= 1 << (end_min // 1440)
            hours_in += [(st + k) % 168 for k in range(ln)]
        rows = [prof.hours[h] for h in hours_in]
        out.append(Candidate(sleep_at=sa, wake_at=wa, sleep_days=sdays, wake_days=wdays, windows=sorted(wins),
                             idle_hours=round(sum(ln for _, ln in wins) - WAKE_LEAD_MIN / 60 * len(wins), 2),
                             min_weeks=min(r["weeks"] for r in rows),
                             all_idle=all(r["idle_weeks"] == r["weeks"] for r in rows),
                             max_active=max(r["active_fraction"] for r in rows),
                             busy_rpm_p50=percentile(busy, 0.5, digits=3)))
    out.sort(key=lambda c: (-c.idle_hours, c.sleep_at))
    return out


def _hour(h: int) -> str:
    return f"{DAY_NAMES[h // 24]} {h % 24:02d}:00"


def build_state(view: dict, prof: Profile, cand: Candidate, pressure: dict, tz_name: str) -> str:
    """The candidate in words (numbers only, plus the quoted, capped deployment name): the evidence first."""
    rows = [prof.hours[(st + k) % 168] for st, ln in cand.windows for k in range(ln)]
    weeks_seen, idle_seen = sum(r["weeks"] for r in rows), sum(r["idle_weeks"] for r in rows)
    busy_weeks = max(r["weeks"] - r["idle_weeks"] for r in rows)
    active = [h for h in range(168) if h in prof.hours and not _idle(prof.hours[h])]
    unobserved = 168 - len(prof.hours)
    shortest = min(ln for _, ln in cand.windows)
    pr = int(view.get("priority") or 0)
    ps, ss = pressure.get("pressure_share"), pressure.get("short_share")
    pct = lambda v: "unknown" if v is None else f"{v:.0%}"  # noqa: E731
    lines = [
        f"- Weeks of data: the window hours were seen in {cand.min_weeks} to {max(r['weeks'] for r in rows)} "
        f"week(s) each ({prof.days} days of metrics in all; {unobserved} of the 168 hours of the week have no data).",
        f"- Idle record: idle in {idle_seen} of {weeks_seen} observed window-hour-weeks ({idle_seen / max(weeks_seen, 1):.0%}); "
        f"{'no week' if busy_weeks == 0 else f'{busy_weeks} week(s)'} with activity in a window hour; highest share "
        f"of active minutes in a window hour {cand.max_active:.0%}.",
        f"- Window length: {len(cand.windows)} window(s) a week, the shortest {shortest} h, "
        f"{cand.idle_hours:g} hours asleep a week in all.",
        f"- Priority: {pr} of 100 ({'high: meant to stay available' if pr >= 90 else 'normal'}).",
        f"- Proposed schedule ({tz_name}): sleep at {cand.sleep_at} on {describe_days(cand.sleep_days)}, wake at "
        f"{cand.wake_at} on {describe_days(cand.wake_days)}.",
        "- Windows: " + "; ".join(f"{_hour(st)} for {ln} h" for st, ln in cand.windows) + ".",
        f"- Outside the windows it is active in {len(active)} hours of the week; typical rate then "
        f"{'unknown' if cand.busy_rpm_p50 is None else f'{cand.busy_rpm_p50:.1f}'} requests per minute.",
        f"- Memory: the host was under memory pressure in {pct(ps)} of the last {PRESSURE_DAYS} days' minutes; free "
        f"memory was below this deployment's reservation in {pct(ss)} of them. Sleeping frees about "
        f"{gib(view.get('reserved_bytes') or 0)} ({'sleep mode' if view.get('sleep_capable') else 'stop and start'}).",
    ]
    return "\n".join([f"Engine: {view.get('engine')}", quoted_block("DEPLOYMENT", view.get("name"), max_chars=64),
                      "Traffic profile (from the engine's minute metrics):", *lines])


def proposal_body(view: dict, cand: Candidate, tz_name: str, sleep_action: str = "stop") -> str:
    down = "put it to sleep" if sleep_action == "sleep" else "stop it"
    up = "wake it" if sleep_action == "sleep" else "start it again"
    same = "window" if len(cand.windows) == 1 else f"{len(cand.windows)} windows"
    return (f"{view['name']} was idle in the same {same} every week for {cand.min_weeks}+ "
            f"observed weeks. A schedule would {down} at {cand.sleep_at} ({describe_days(cand.sleep_days)}) and "
            f"{up} at {cand.wake_at} ({describe_days(cand.wake_days)}), {tz_name}: about "
            f"{plural(cand.idle_hours, 'hour')} a week, freeing about {gib(view.get('reserved_bytes') or 0)}. It never acts while requests arrive, "
            "and you can pause or delete it on the deployment's page.")


# ---------------- evaluation adapter (tools/eval_feature.py) ----------------
def eval_case(case: dict):
    """A corpus case: ``profile`` ({"days": n, "hours": {how: {...}}}), ``view`` (engine, name, priority,
    reserved_bytes, sleep_capable), ``pressure`` and ``candidate`` (its key; default: the first candidate)."""
    from tools.eval_feature import EvalCase  # noqa: PLC0415 - only for the harness
    prof = Profile(days=int(case["profile"]["days"]),
                   hours={int(k): v for k, v in case["profile"]["hours"].items()})
    cands = candidates(prof)
    cand = next((c for c in cands if c.key == case.get("candidate")), cands[0] if cands else None)
    if cand is None or prof.days < MIN_DAYS:
        return EvalCase(rules="skip", state="", question=QUESTION, options=OPTIONS)
    state = build_state(case["view"], prof, cand, case.get("pressure") or {}, "UTC")
    return EvalCase(rules="skip" if rule_skip(cand) else None, state=state, question=QUESTION, options=OPTIONS)

