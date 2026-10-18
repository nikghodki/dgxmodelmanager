"""Insights feed (spec "Insights feed"): proactive findings of the decision-model features — health, schedule
proposals, reading flags, tuning proposals, routing advice, "feature degraded" notices.

- One live insight per ``dedupe_key``: ``upsert`` updates an open / snoozed one in place (never a duplicate); a
  dismissed one stays dismissed (and unchanged) until its condition clears (``resolve``) unless the severity rises,
  which re-opens it. Applied and expired insights are closed for good; the next ``upsert`` opens a new one.
- ``resolve(dedupe_key)`` closes the live insight when its condition is gone (open / snoozed / dismissed ->
  expired), or with ``grace`` keeps it visible until ``now + grace`` (health: 1 h after the condition clears).
- Title and body are written by code from templates (never model text); they are shown to viewers for features
  whose ``FeatureSpec.viewer_visible`` is set. The ``action`` descriptor (``validate_action``) is what the UI turns
  into one of its existing dialogs; the server never executes it, and viewers never receive it.
- Retention (``prune``, in the metrics loop's retention pass): closed insights are deleted 30 days after their last
  change (a dismissed one whose condition is still reported is touched by every upsert, so it is kept); open ones whose ``expires_at`` passed become expired; snoozed ones whose ``snooze_until`` passed re-open.
- A deployment's insights are deleted with it (``on_deployment_removed``, a deployer removal hook): subject
  ``deployment:<id>`` or an action naming that deployment.
- Every change publishes SSE ``insights`` with ``{"id": <id>}`` only (viewer-visible topic; clients re-fetch).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StringConstraints, TypeAdapter, \
    field_validator
from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from dgxmm.db import INSIGHT_LIVE, Insight, as_aware, utcnow
from dgxmm.decider import registry

log = logging.getLogger(__name__)
TOPIC = "insights"
SEVERITIES = ("info", "warning", "critical")
SOURCES = ("rules", "model")
STATUSES = ("open", "dismissed", "snoozed", "applied", "expired")
CLOSED_RETENTION = timedelta(days=30)
PROPOSAL_TTL = timedelta(days=7)          # expires_at for proposals (schedule, tuning, routing)
HEALTH_CLEAR_GRACE = timedelta(hours=1)  # resolve(..., grace=) for health insights
MAX_SNOOZE_HOURS = 168
MAX_TITLE, MAX_BODY, MAX_KEY = 256, 4000, 256
_KIND = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def deployment_subject(dep_id: int) -> str:
    """The subject of insights about one deployment (the Deployment detail card filters on it)."""
    return f"deployment:{int(dep_id)}"


# ---------------- action descriptors ----------------
MAX_ACTION_JSON = 8192
_ConfigKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]
_Label = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: _Label | None = None  # button text; the UI has a default per type


FORBIDDEN_PATCH_KEYS = frozenset({"model_ref", "trust_remote_code"})  # a model switch / remote code: own flows


class ReconfigureAction(_Action):
    """Open the deployment's Reconfigure dialog prefilled with ``patch`` (the user reviews and confirms). The patch
    may not switch the model or enable trust_remote_code (FORBIDDEN_PATCH_KEYS)."""
    type: Literal["reconfigure"]
    deployment_id: StrictInt = Field(ge=1)
    patch: dict[_ConfigKey, Any] = Field(min_length=1, max_length=32)

    @field_validator("patch")
    @classmethod
    def _allowed(cls, v: dict) -> dict:
        bad = sorted(FORBIDDEN_PATCH_KEYS & set(v))
        if bad:
            raise ValueError(f"an insight may not change {', '.join(bad)}")
        return v


class ScheduleAction(_Action):
    """Open the schedule dialog for a proposed sleep/wake schedule (lane A3 defines ``proposal``)."""
    type: Literal["schedule"]
    deployment_id: StrictInt = Field(ge=1)
    proposal: dict[_ConfigKey, Any] = Field(max_length=32)


class ThresholdAction(_Action):
    """Open a confirmation to set a feature's threshold through PUT /api/decider/settings."""
    type: Literal["threshold"]
    feature: str = Field(max_length=32)
    value: StrictFloat | StrictInt = Field(ge=0.0, le=1.0)

    @field_validator("feature")
    @classmethod
    def _known(cls, v: str) -> str:
        s = registry.find(v)
        if s is None or s.group != s.name:
            raise ValueError(f"unknown feature group {v!r}")
        return v


class CopyAction(_Action):
    """Show a text (e.g. a LiteLLM model_list snippet) with a copy button."""
    type: Literal["copy"]
    text: str = Field(min_length=1, max_length=MAX_ACTION_JSON // 2)


class OpenAction(_Action):
    """Navigate to an app page (relative path inside the SPA, e.g. ``/litellm``)."""
    type: Literal["open"]
    path: Annotated[str, StringConstraints(pattern=r"^/([A-Za-z0-9_.~?=&%:-][A-Za-z0-9/_.~?=&%:-]{0,254})?$")]


ACTION_TYPES = ("reconfigure", "schedule", "threshold", "copy", "open")
_ACTION = TypeAdapter(Annotated[Union[ReconfigureAction, ScheduleAction, ThresholdAction, CopyAction, OpenAction],
                                Field(discriminator="type")])


def validate_action(action: dict | None) -> dict | None:
    """The validated, normalised descriptor (ValueError if it is not one of ACTION_TYPES' schemas)."""
    if action is None:
        return None
    if not isinstance(action, dict):
        raise ValueError("an action is a JSON object")
    try:
        raw = json.dumps(action, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"action is not plain JSON: {e}") from None
    if len(raw) > MAX_ACTION_JSON:
        raise ValueError("action too large")
    try:
        out = _ACTION.validate_python(action).model_dump(exclude_none=True)
    except Exception as e:  # pydantic ValidationError (a ValueError subclass) or a discriminator error
        raise ValueError(f"invalid action: {e}") from None
    if "value" in out:
        out["value"] = float(out["value"])
    return out


# ---------------- service ----------------
class InsightClosed(Exception):
    """dismiss / snooze / applied on an insight that is no longer open or snoozed."""


def insight_out(r: Insight, *, admin: bool) -> dict:
    """API view of an insight; viewers never get the action descriptor."""
    iso = lambda d: as_aware(d).isoformat() if d else None  # noqa: E731
    return {"id": r.id, "kind": r.kind, "feature": r.feature, "subject": r.subject, "severity": r.severity,
            "title": r.title, "body": r.body, "confidence": r.confidence, "source": r.source,
            "decision_id": r.decision_id, "action": r.action if admin else None, "status": r.status,
            "snooze_until": iso(r.snooze_until), "dedupe_key": r.dedupe_key, "created_at": iso(r.created_at),
            "updated_at": iso(r.updated_at), "expires_at": iso(r.expires_at)}


def _check(kind, feature, subject, severity, title, body, dedupe_key, confidence, source) -> None:
    if not isinstance(kind, str) or not _KIND.match(kind):
        raise ValueError(f"invalid insight kind {kind!r}")
    if registry.find(feature) is None:
        raise ValueError(f"unknown feature {feature!r}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if not isinstance(title, str) or not title or len(title) > MAX_TITLE:
        raise ValueError("title: 1..256 characters")
    if not isinstance(body, str) or len(body) > MAX_BODY:
        raise ValueError("body: at most 4000 characters")
    if not isinstance(subject, str) or len(subject) > 256:
        raise ValueError("subject: at most 256 characters")
    if not isinstance(dedupe_key, str) or not dedupe_key or len(dedupe_key) > MAX_KEY:
        raise ValueError("dedupe_key: 1..256 characters")
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                   or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be None or a number in [0, 1]")


class InsightService:
    def __init__(self, db, events=None, *, clock: Callable[[], datetime] = utcnow):
        self.db, self.events, self.clock = db, events, clock
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """The event loop SSE events are handed to when a change happens in a worker thread (retention, jobs'
        to_thread work). Called by the insight engine at start."""
        self._loop = loop

    def _publish(self, ids) -> None:
        if self.events is None:
            return
        for i in ids:
            payload = {"id": i}
            try:
                self._loop = asyncio.get_running_loop()  # remembered for publishes from worker threads
                self.events.publish("insights", payload)
            except RuntimeError:  # a worker thread: hand it to the loop (EventBroker is not thread-safe)
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self.events.publish, "insights", payload)

    # ---------- producers (features, jobs) ----------
    def upsert(self, kind: str, feature: str, subject: str, severity: str, title: str, body: str, *,
               dedupe_key: str, confidence: float | None = None, source: str, decision_id: int | None = None,
               action: dict | None = None, expires_at: datetime | None = None) -> Insight:
        """Raise or refresh the insight for ``dedupe_key`` (see the module docstring). ValueError on invalid input
        (a bug in the caller: titles/bodies are templates). Returns the row (detached)."""
        _check(kind, feature, subject, severity, title, body, dedupe_key, confidence, source)
        action = validate_action(action)
        args = (kind, feature, subject, severity, title, body, dedupe_key, confidence, source, decision_id, action,
                expires_at)
        try:
            return self._upsert(*args)
        except IntegrityError:  # a concurrent upsert of the same key created the row first: update that one
            return self._upsert(*args)

    def _upsert(self, kind, feature, subject, severity, title, body, dedupe_key, confidence, source, decision_id,
                action, expires_at) -> Insight:
        now = self.clock()
        changed = False
        with self.db.session() as s:
            row = s.scalar(select(Insight).where(Insight.dedupe_key == dedupe_key, Insight.status.in_(INSIGHT_LIVE)))
            escalated = row is not None and _SEV_RANK[severity] > _SEV_RANK[row.severity]
            if row is not None and row.status == "dismissed" and not escalated:
                row.updated_at = now  # still reported: retention keeps the dismissal (no resurfacing)
                return row  # the user dismissed it; it stays dismissed until the condition clears
            if row is None:
                row = Insight(kind=kind, feature=feature, subject=subject, dedupe_key=dedupe_key, status="open",
                              created_at=now, severity=severity, title=title, body=body, source=source)
                s.add(row)
                changed = True
            elif row.status in ("dismissed", "snoozed") and escalated:  # a rise in severity re-opens it
                row.status, row.snooze_until, changed = "open", None, True
            new = {"kind": kind, "feature": feature, "subject": subject, "severity": severity, "title": title,
                   "body": body, "confidence": None if confidence is None else float(confidence), "source": source,
                   "decision_id": decision_id, "action": action, "expires_at": expires_at}
            for k, v in new.items():
                cur = getattr(row, k)
                if k == "expires_at" and cur is not None and v is not None:
                    same = as_aware(cur) == as_aware(v)
                else:
                    same = cur == v
                if not same:
                    setattr(row, k, v)
                    changed = changed or k not in ("confidence", "decision_id", "expires_at")
            if row.status == "snoozed" and row.snooze_until and as_aware(row.snooze_until) <= now:
                row.status, row.snooze_until, changed = "open", None, True
            row.updated_at = now
            s.flush()
            rid = row.id
        if changed:
            self._publish([rid])
        return row

    def resolve(self, dedupe_key: str, *, grace: timedelta | None = None) -> int:
        """The condition behind ``dedupe_key`` is gone: its live insight becomes expired (or, with ``grace``, stays
        open until now + grace; a later upsert cancels that). Returns the number of insights changed."""
        now = self.clock()
        with self.db.session() as s:
            row = s.scalar(select(Insight).where(Insight.dedupe_key == dedupe_key, Insight.status.in_(INSIGHT_LIVE)))
            if row is None:
                return 0
            if grace and row.status != "dismissed":
                until = now + grace
                if row.expires_at is not None and as_aware(row.expires_at) <= until:
                    return 0
                row.expires_at = until
            else:
                row.status = "expired"
            row.updated_at = now
            rid = row.id
        self._publish([rid])
        return 1

    # ---------- readers ----------
    def get(self, insight_id: int) -> Insight | None:
        with self.db.session() as s:
            return s.get(Insight, insight_id)

    def list(self, *, status: list[str] | None = None, subject: str | None = None, kind: str | None = None,
             feature: str | None = None, admin: bool = True, limit: int = 200) -> list[dict]:
        """Insights, live ones first (most severe, then most recently changed). Viewers (``admin=False``) see only
        features with ``viewer_visible`` and never the action."""
        self.refresh()
        q = select(Insight)
        if status:
            q = q.where(Insight.status.in_(status))
        if subject is not None:
            q = q.where(Insight.subject == subject)
        if kind is not None:
            q = q.where(Insight.kind == kind)
        if feature is not None:
            q = q.where(Insight.feature == feature)
        if not admin:
            q = q.where(Insight.feature.in_([f.name for f in registry.all_features() if f.viewer_visible]))
        live_first = case((Insight.status == "open", 0), (Insight.status == "snoozed", 1), else_=2)
        sev = case((Insight.severity == "critical", 0), (Insight.severity == "warning", 1), else_=2)
        q = q.order_by(live_first, sev, Insight.updated_at.desc(), Insight.id.desc()).limit(limit)
        with self.db.session() as s:
            return [insight_out(r, admin=admin) for r in s.scalars(q)]

    # ---------- user actions (admin; the API audits them) ----------
    def _transition(self, insight_id: int, status: str, snooze_until: datetime | None = None) -> Insight | None:
        now = self.clock()
        with self.db.session() as s:
            row = s.get(Insight, insight_id)
            if row is None:
                return None
            if row.status not in ("open", "snoozed"):
                if row.status == status and status == "dismissed":
                    return row  # idempotent
                raise InsightClosed(row.status)
            row.status, row.snooze_until, row.updated_at = status, snooze_until, now
            s.flush()
        self._publish([insight_id])
        return row

    def dismiss(self, insight_id: int) -> Insight | None:
        return self._transition(insight_id, "dismissed")

    def snooze(self, insight_id: int, hours: int) -> Insight | None:
        if not 1 <= int(hours) <= MAX_SNOOZE_HOURS:
            raise ValueError(f"hours: 1..{MAX_SNOOZE_HOURS}")
        return self._transition(insight_id, "snoozed", self.clock() + timedelta(hours=int(hours)))

    def mark_applied(self, insight_id: int) -> Insight | None:
        """The user applied the action through the normal endpoint (the UI calls this after it succeeded)."""
        return self._transition(insight_id, "applied")

    # ---------- maintenance ----------
    def refresh(self) -> dict[str, int]:
        """Re-open snoozed insights whose snooze ended; expire open/snoozed ones whose expires_at passed."""
        now = self.clock()
        with self.db.session() as s:
            woke = list(s.scalars(select(Insight.id).where(Insight.status == "snoozed",
                                                           Insight.snooze_until <= now)))
            if woke:
                s.execute(update(Insight).where(Insight.id.in_(woke))
                          .values(status="open", snooze_until=None, updated_at=now))
            gone = list(s.scalars(select(Insight.id).where(Insight.status.in_(("open", "snoozed")),
                                                           Insight.expires_at <= now)))
            if gone:
                s.execute(update(Insight).where(Insight.id.in_(gone)).values(status="expired", updated_at=now))
        self._publish(woke + gone)
        return {"insights_unsnoozed": len(woke), "insights_expired": len(gone)}

    def prune(self) -> dict[str, int]:
        """Retention pass (runs in a worker thread): refresh, then delete closed insights older than 30 days."""
        out = self.refresh()
        cutoff = self.clock() - CLOSED_RETENTION
        with self.db.session() as s:
            out["insights_closed"] = s.execute(delete(Insight).where(
                Insight.status.in_(("dismissed", "applied", "expired")), Insight.updated_at < cutoff)).rowcount
        if any(out.values()):
            log.info("insights retention: %s", out)
        return out

    def on_deployment_removed(self, dep_id: int) -> int:
        """Deployer removal hook: delete the deployment's insights (its subject, or an action naming it)."""
        with self.db.session() as s:
            ids = list(s.scalars(select(Insight.id).where(or_(
                Insight.subject == deployment_subject(dep_id),
                func.json_extract(Insight.action, "$.deployment_id") == int(dep_id)))))
            if ids:
                s.execute(delete(Insight).where(Insight.id.in_(ids)))
        self._publish(ids)
        return len(ids)
