"""Decider: the one entry point features use to consult the decision model. Never raises to callers.

Rules first, model second, abstain when unsure: callers run their deterministic rules and only ask ``decide`` where
the rules have no answer. A model answer below the feature's threshold is returned as ``source="none"`` (with the
probabilities, for display) so callers fall back to today's behaviour. Disabled / not deployed / unhealthy /
circuit open / timeout / error -> ``source="none"`` as well.

Logging: every rules decision (``record_rules``) and every model consultation (answered, abstained or failed) is a
``decisions`` row; short-circuits while the decider is unavailable are not logged (they would only be noise).
Rows never contain the state text: only its hash.
"""
from __future__ import annotations

import logging
import statistics
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from sqlalchemy import func, select

from dgxmm.db import Database, DecisionRecord, utcnow
from dgxmm.decider import registry
from dgxmm.decider.client import (PRIORITY_BACKGROUND, PRIORITY_INTERACTIVE, PRIORITY_LANE, DeciderError,
                                  SystemOneClient)
from dgxmm.decider.common import finite, finite_probs
from dgxmm.decider.secret import read_key
from dgxmm.decider.types import Decision, Option
from dgxmm.engines.decision import DECIDER_MODELS, DEFAULT_MODEL

log = logging.getLogger(__name__)

DECIDER_ENABLED = "decider.enabled"
DECIDER_DEPLOYMENT_ID = "decider.deployment_id"
DECIDER_SWITCH_FROM = "decider.switch_from"  # set while a model switch is not confirmed ready
DECIDER_MODEL = "decider.model"  # which pinned decision model (engines/decision.py DECIDER_MODELS) to run
# Settings-level features (UI toggles/thresholds) and their default thresholds: the feature registry's groups
# (dgxmm/decider/registry.py; the same dict object, so later registrations show up here). Calibration: protocol.py.
FEATURES: dict[str, float] = registry.DEFAULT_THRESHOLDS
# Decision features (as logged) -> settings feature (registry members, e.g. parser_tool -> parser).
FEATURE_GROUP = registry.GROUP_OF
UNLOGGED_ERRORS = {"circuit_open", "unavailable", "busy"}  # counted (load()), not logged: bursts would flood the log
@dataclass
class JobBudget:
    """Decider calls one insight-engine job run may make (dgxmm/decider/insight_engine.py)."""
    job: str
    left: int
    used: int = 0
    refused: int = 0


# Set for the duration of a job run (``job_context``). Every ``decide`` in that context — through ``ctx.decider``,
# a feature service or anything else — uses the client's background lane (lowest priority, never every slot) and
# counts against the run's budget; once it is used up, calls abstain without asking.
LANE: ContextVar[JobBudget | None] = ContextVar("decider_job", default=None)


@contextmanager
def job_context(budget: JobBudget):
    token = LANE.set(budget)
    try:
        yield budget
    finally:
        LANE.reset(token)


def take_budget() -> bool:
    """One model call is about to be made: False if the current job's budget is used up (counted as refused).
    Outside a job: always True. Called only once the decider would really be asked."""
    b = LANE.get()
    if b is None:
        return True
    if b.left <= 0:
        b.refused += 1
        return False
    b.left -= 1
    b.used += 1
    return True


def feature_group(feature: str) -> str:
    return registry.group_of(feature)


def call_priority(feature: str) -> int:
    """Client priority of a model call: the insight lane, else background features (registry ``background``) queue
    behind interactive (request-path) questions."""
    if LANE.get() is not None:
        return PRIORITY_LANE
    return PRIORITY_BACKGROUND if registry.is_background(feature) else PRIORITY_INTERACTIVE


def enabled_key(group: str) -> str:
    return f"decider.feature.{group}.enabled"


def threshold_key(group: str) -> str:
    return f"decider.feature.{group}.threshold"


class Decider:
    def __init__(self, *, db: Database, settings, store, deployer: Callable[[], Any] = lambda: None,
                 http: Callable[[], Any] = lambda: None, clock: Callable[[], float] = time.monotonic):
        self.db, self.settings, self.store = db, settings, store
        self._deployer, self._http, self.clock = deployer, http, clock
        self._client: SystemOneClient | None = None
        self._client_revision: str | None = None
        self.supervisor = None  # DeciderSupervisor (dgxmm/decider/supervisor.py), wired in features.wire_features

    # ---------- model ----------
    @property
    def configured_model(self) -> str:
        """The model the decider should run (setting `decider.model`; unknown values fall back to the default)."""
        raw = self.store.get(DECIDER_MODEL)
        return raw if raw in DECIDER_MODELS else DEFAULT_MODEL

    def set_model(self, repo: str) -> None:
        if repo not in DECIDER_MODELS:
            raise KeyError(repo)
        self.store.set(DECIDER_MODEL, None if repo == DEFAULT_MODEL else repo)

    @property
    def switch_from(self) -> str | None:
        """The model a decider was just switched away from (rolled back to if the new one fails to start)."""
        raw = self.store.get(DECIDER_SWITCH_FROM)
        return raw if raw in DECIDER_MODELS else None

    def set_switch_from(self, repo: str | None) -> None:
        self.store.set(DECIDER_SWITCH_FROM, repo)

    @property
    def model(self) -> str:
        """The model actually answering: the decider deployment's model, else the configured one."""
        view = self.deployment()
        ref = view.get("model_ref") if view else None
        return ref if ref in DECIDER_MODELS else self.configured_model

    @property
    def revision(self) -> str:
        return DECIDER_MODELS[self.model].revision

    # ---------- settings ----------
    @property
    def enabled(self) -> bool:
        return self.store.get_bool(DECIDER_ENABLED, False)

    def set_enabled(self, value: bool) -> None:
        self.store.set_bool(DECIDER_ENABLED, value)

    @property
    def deployment_id(self) -> int | None:
        raw = self.store.get(DECIDER_DEPLOYMENT_ID)
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def set_deployment_id(self, dep_id: int | None) -> None:
        self.store.set(DECIDER_DEPLOYMENT_ID, None if dep_id is None else str(dep_id))

    def on_deployment_removed(self, dep_id: int) -> None:
        """Deployer removal hook: removing the decider's deployment (from any page) turns the decider off, so
        neither the settings nor the supervisor point at a deleted deployment."""
        if self.deployment_id == dep_id:
            self.set_enabled(False)
            self.set_deployment_id(None)

    def threshold(self, feature: str) -> float:
        group = feature_group(feature)
        raw = self.store.get(threshold_key(group))
        try:
            if raw is not None:
                v = float(raw)
                if 0.0 <= v <= 1.0:
                    return v
        except ValueError:
            pass
        return FEATURES.get(group, 0.8)

    def feature_enabled(self, feature: str) -> bool:
        """The decider is enabled and this feature is switched on (default on): the model may be asked."""
        return self.enabled and self.store.get_bool(enabled_key(feature_group(feature)), True)

    def rules_enabled(self, feature: str) -> bool:
        """This feature is switched on (default on), whether or not the decider is: its rules may run (rules-only
        paths work with the decision model disabled; spec "Feature registry")."""
        return self.store.get_bool(enabled_key(feature_group(feature)), True)

    def features(self) -> dict[str, dict]:
        return {g: {"enabled": self.store.get_bool(enabled_key(g), True), "threshold": self.threshold(g)}
                for g in FEATURES}

    def set_feature(self, group: str, enabled: bool | None = None, threshold: float | None = None) -> None:
        if group not in FEATURES:
            raise KeyError(group)
        if enabled is not None:
            self.store.set_bool(enabled_key(group), enabled)
        if threshold is not None:
            self.store.set(threshold_key(group), repr(float(threshold)))

    # ---------- deployment / client ----------
    def deployment(self) -> dict | None:
        deployer, dep_id = self._deployer(), self.deployment_id
        if deployer is None or dep_id is None:
            return None
        try:
            return deployer.view(dep_id)
        except Exception:
            return None

    def _base_url(self) -> str | None:
        view = self.deployment()
        if not view or view["state"] != "ready" or not view.get("port"):
            return None
        return self._deployer().base_url(view["name"], view["port"], view.get("engine"))

    def client(self) -> SystemOneClient | None:
        rev = self.revision
        if self._client is not None and self._client_revision != rev:
            self._client = None  # another model answers now: new cache keys (revision) and a fresh breaker
        if self._client is None:
            http = self._http()
            if http is None:
                return None
            self._client = SystemOneClient(http, base_url=self._base_url, api_key=lambda: read_key(self.settings),
                                           revision=rev, clock=self.clock,
                                           max_in_flight=self.settings.decider_max_in_flight,
                                           queue_timeout_s=self.settings.decider_queue_timeout_s)
            self._client_revision = rev
        return self._client

    def reset_client(self) -> None:
        """Forget the client (cache + circuit breaker), e.g. after the key was rotated."""
        self._client = None

    def healthy(self) -> bool:
        """The decider deployment is ready and the circuit breaker is closed (no network call)."""
        c = self.client()
        return c is not None and self._base_url() is not None and not c.breaker.is_open()

    async def available(self) -> bool:
        return self.enabled and self.healthy()

    def load(self) -> dict:
        """Client-side load: requests at the decider, waiting, and shed (answered by rules) since start."""
        c = self._client
        if c is None:
            return {"max_in_flight": self.settings.decider_max_in_flight, "in_flight": 0, "queued": 0, "shed": 0}
        return c.limiter.stats()

    # ---------- decisions ----------
    async def decide(self, *, feature: str, subject: str, state: str, question: str, options: list[Option],
                     threshold: float | None = None, timeout_s: float = 2.0, log: bool = True) -> Decision:
        none = Decision(feature=feature, subject=subject, source="none", choice=None, confidence=None)
        try:
            if not options or not self.feature_enabled(feature) or not await self.available():
                return none
            if not take_budget():  # inside an insight-engine job whose budget is used up
                return none
            client = self.client()
            th = self.threshold(feature) if threshold is None else threshold
            input_hash = client.input_hash(feature, state, question, options)
            try:
                ans = await client.choose(feature=feature, state=state, question=question, options=options,
                                          timeout_s=timeout_s,
                                          priority=call_priority(feature))
            except DeciderError as e:
                if log and e.code not in UNLOGGED_ERRORS:
                    try:
                        none.decision_id = self._write(none, input_hash=input_hash, error=e.code)
                    except Exception:
                        _log_exc("decider: could not log a failed %s decision", feature)
                return none
            accepted = ans.confidence >= th
            d = Decision(feature=feature, subject=subject, source="model" if accepted else "none",
                         choice=ans.choice if accepted else None, confidence=ans.confidence,
                         probabilities=dict(ans.probabilities), latency_ms=None if ans.cached else ans.latency_ms)
            if log:
                try:
                    d.decision_id = self._write(d, input_hash=input_hash)
                except Exception:  # an answer we cannot log is still an answer
                    _log_exc("decider: could not log a %s decision", feature)
            return d
        except Exception:
            _log_exc("decider: decide(%s) failed", feature)
            return none

    def record_rules(self, *, feature: str, subject: str, choice: str, reason: str | None,
                     options: list[Option] | None = None) -> Decision:
        d = Decision(feature=feature, subject=subject, source="rules", choice=choice, confidence=1.0,
                     probabilities={choice: 1.0}, reason=reason)
        try:
            d.decision_id = self._write(d)
        except Exception:
            _log_exc("decider: could not log a rules decision for %s", feature)
        return d

    def _write(self, d: Decision, *, input_hash: str | None = None, error: str | None = None) -> int:
        with self.db.session() as s:
            model = self.model if d.source != "rules" else None
            row = DecisionRecord(feature=d.feature[:32], subject=d.subject[:256], source=d.source, choice=d.choice,
                                 confidence=d.confidence, probabilities=d.probabilities, latency_ms=d.latency_ms,
                                 model=model, revision=DECIDER_MODELS[model].revision if model else None,
                                 input_hash=input_hash, error=error)
            s.add(row)
            s.flush()
            return row.id

    # ---------- admin views ----------
    def stats(self) -> dict:
        since = utcnow() - timedelta(hours=24)
        with self.db.session() as s:
            count = s.scalar(select(func.count()).select_from(DecisionRecord)
                             .where(DecisionRecord.created_at >= since)) or 0
            lat = [v for v in s.scalars(select(DecisionRecord.latency_ms)
                                        .where(DecisionRecord.created_at >= since, DecisionRecord.latency_ms > 0)
                                        .order_by(DecisionRecord.id.desc()).limit(1000))]
            fb = dict(s.execute(select(DecisionRecord.feedback, func.count()).where(DecisionRecord.feedback.is_not(None))
                                .group_by(DecisionRecord.feedback)).all())
        judged = fb.get("accepted", 0) + fb.get("rejected", 0)
        return {"p50_ms": round(statistics.median(lat), 1) if lat else None, "decisions_24h": count,
                "accepted_rate": round(fb.get("accepted", 0) / judged, 3) if judged else None}

    def recent(self, feature: str | None = None, limit: int = 100) -> list[dict]:
        q = select(DecisionRecord).order_by(DecisionRecord.id.desc()).limit(limit)
        if feature:
            q = q.where(DecisionRecord.feature == feature)
        with self.db.session() as s:
            return [row_out(r) for r in s.scalars(q)]

    def feedback(self, decision_id: int, accepted: bool, user_id: int | None) -> dict | None:
        with self.db.session() as s:
            row = s.get(DecisionRecord, decision_id)
            if row is None:
                return None
            row.feedback = "accepted" if accepted else "rejected"
            row.feedback_by, row.feedback_at = user_id, utcnow()
            s.flush()
            return row_out(row)


def row_out(r: DecisionRecord) -> dict:
    return {"id": r.id, "created_at": r.created_at.isoformat() if r.created_at else None, "feature": r.feature,
            "subject": r.subject, "source": r.source, "choice": r.choice, "confidence": finite(r.confidence),
            "probabilities": finite_probs(r.probabilities), "latency_ms": finite(r.latency_ms), "model": r.model,
            "revision": r.revision, "error": r.error, "feedback": r.feedback,
            "feedback_at": r.feedback_at.isoformat() if r.feedback_at else None}


def _log_exc(msg: str, *args) -> None:
    log.exception(msg, *args)
