"""Feature 3 — plan recommendation when a model doesn't fit.

1. Deterministic safety filter over the PlanOptions from `build_plans`; a plan is NOT recommendable (gets
   `unsafe_reason`) when it
   - does not cover the shortfall;
   - stops or sleeps a deployment that is ACTIVE: starting/loading, state unknown, ready for less than
     ACTIVE_WINDOW_MIN minutes (latest `ready` event), or ready with traffic (running/waiting requests or
     prompt/generation tokens) in the last ACTIVE_WINDOW_MIN minutes; sleeping deployments serve nothing;
   - stops or sleeps a ready deployment whose traffic is UNKNOWN: metrics (live samples, or rollups with
     samples > 0 and a non-null `running`) cover fewer than COVERAGE_MIN of the last ACTIVE_WINDOW_MIN minutes.
     The scraper only records successful scrapes, so missing data is never read as "idle";
   - stops or sleeps a DEPENDENCY: a deployment another active deployment uses (its model is the other one's
     speculative draft, or the other one's config names its container);
   - stops or sleeps a SYSTEM deployment (`system` flag / engine "decision");
   - stops or sleeps a deployment with a HIGHER priority than the request (deploy: the request's priority;
     switch / reconfigure: the replaced deployment's own priority);
   - lowers a user-stated requirement: context length / concurrent sequences the user set explicitly to a
     non-default value in the request.
   Unsafe plans stay in the list (and selectable as before) — they are only annotated.
2. Exactly one allowed plan -> recommended, source "rules". Two or more -> the decision model ranks the allowed
   plans only (timeout 2 s, results cached by input); at/above threshold("plan") -> recommended, source "model".
   Otherwise nothing is recommended (today's behaviour).
The reason is a deterministic template from the plan's facts, never model text.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field

from dgxmm.decider.common import abstain, quoted_block, safe_decide, safe_record_rules
from dgxmm.decider.types import Decision, Option

log = logging.getLogger(__name__)
GIB = 1024**3
ACTIVE_WINDOW_MIN = 15
COVERAGE_MIN = 0.8  # fraction of the window's minutes that must have metrics for traffic to count as known
CACHE_TTL_S = 120.0
CACHE_MAX = 128
TIMEOUT_S = 2.0
QUESTION = ("Which plan frees enough memory for the new model with the least disruption to the deployments that "
            "are running?")


@dataclass
class DepFacts:
    id: int
    name: str
    state: str
    priority: int = 0
    system: bool = False
    dependents: list[str] = field(default_factory=list)  # names of active deployments that use this one
    last_traffic_min: float | None = None  # minutes since the last request activity; None = never recorded
    running_now: bool = False
    traffic_known: bool = False  # metrics cover >= COVERAGE_MIN of the last ACTIVE_WINDOW_MIN minutes
    uncovered_min: int = ACTIVE_WINDOW_MIN  # minutes of that window without metrics
    ready_for_min: float | None = None  # minutes since the latest transition to ready (None = unknown)

    def active_reason(self) -> str | None:
        if self.state in ("starting", "loading"):
            return f"{self.name} is {self.state}"
        if self.state == "unknown":
            return f"{self.name} is in an unknown state"
        if self.state == "ready":
            if self.running_now:
                return f"{self.name} is serving requests"
            if self.last_traffic_min is not None and self.last_traffic_min < ACTIVE_WINDOW_MIN:
                return f"{self.name} served requests {max(int(self.last_traffic_min), 0)} min ago"
            if self.ready_for_min is not None and self.ready_for_min < ACTIVE_WINDOW_MIN:
                return f"{self.name} became ready {max(int(self.ready_for_min), 0)} min ago"
        return None

    def unknown_reason(self) -> str | None:
        if self.state == "ready" and not self.traffic_known:
            return f"{self.name}: traffic unknown (no metrics for {self.uncovered_min} min)"
        return None


def _gib(n: int | float) -> str:
    return f"{n / GIB:.1f} GiB"


def idle_text(d: DepFacts) -> str:
    if d.state == "sleeping":
        return f"{d.name} is asleep"
    m = d.last_traffic_min
    if not d.traffic_known and d.state == "ready":
        return f"{d.name} traffic unknown"
    if m is None:
        return f"{d.name} no traffic in the last {ACTIVE_WINDOW_MIN}+ min"
    if m >= 2 * 24 * 60:
        return f"{d.name} idle {m / 1440:.0f} days"
    if m >= 120:
        return f"{d.name} idle {m / 60:.0f} h"
    return f"{d.name} idle {m:.0f} min"


def unsafe_reason(plan: dict, deps: dict[int, DepFacts], *, raw_cfg: dict, defaults: dict, context_key: str,
                  seqs_key: str, request_priority: int = 0) -> str | None:
    if not plan.get("fits"):
        return "does not free enough memory"
    for a in plan.get("actions") or []:
        d = deps.get(a["deployment_id"])
        verb = "stop" if a["type"] == "stop" else "put to sleep"
        if d is None:
            return f"would {verb} an unknown deployment"
        if d.system:
            return f"would {verb} {d.name}, a system deployment"
        if d.priority > request_priority:
            return f"would {verb} {d.name}, which has a higher priority ({d.priority}) than this request ({request_priority})"
        if d.dependents:
            return f"would {verb} {d.name}, which {', '.join(d.dependents)} depend(s) on"
        if (r := d.active_reason()) is not None:
            return f"would {verb} {d.name}: {r.removeprefix(d.name + ' ')}"
        if (r := d.unknown_reason()) is not None:
            return f"would {verb} {r}"
    for key in (context_key, seqs_key):
        patch = plan.get("patch") or {}
        if key in patch and key in raw_cfg and raw_cfg[key] != defaults.get(key):
            try:
                if int(patch[key]) < int(raw_cfg[key]):
                    what = "context length" if key == context_key else "concurrent sequences"
                    return f"lowers the {what} you asked for ({int(raw_cfg[key]):,})"
            except (TypeError, ValueError):
                continue
    return None


def reason_for(plan: dict, deps: dict[int, DepFacts]) -> str:
    acts = plan.get("actions") or []
    if acts:
        parts = [f"frees {_gib(plan.get('frees') or 0)}"]
        parts += [idle_text(deps[a["deployment_id"]]) for a in acts if a["deployment_id"] in deps]
        return "; ".join(parts)
    if plan.get("model_ref"):
        return f"uses {plan['model_ref']} (already on disk); needs {_gib(plan.get('new_required') or 0)}; " \
               "keeps other deployments running"
    patch = plan.get("patch") or {}
    sets = ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in patch.items() if v is not None)
    return f"needs {_gib(plan.get('new_required') or 0)}; keeps other deployments running" + (f"; sets {sets}" if sets else "")


def _option(plan: dict, deps: dict[int, DepFacts]) -> Option:
    desc = f"{plan.get('detail', '')} Needs {_gib(plan.get('new_required') or 0)}."
    if plan.get("actions"):
        desc += " Affects: " + "; ".join(
            f"{deps[a['deployment_id']].name} ({a['type']}, {deps[a['deployment_id']].state}, priority "
            f"{deps[a['deployment_id']].priority}, {idle_text(deps[a['deployment_id']])})"
            for a in plan["actions"] if a["deployment_id"] in deps)
    else:
        desc += " Changes only the new deployment."
    return Option(plan["id"], plan["title"], desc)


def build_state(*, shortfall: int, model_ref: str, engine: str, cfg: dict, deps: dict[int, DepFacts]) -> str:
    lines = [f"Memory shortfall: {_gib(shortfall)}", f"Engine: {engine}",
             quoted_block("NEW MODEL", model_ref, max_chars=300),
             "New deployment config: " + json.dumps({k: v for k, v in sorted(cfg.items())
                                                     if isinstance(v, (int, float, bool, str)) or v is None},
                                                    default=str)[:800],
             "Running deployments:"]
    for d in sorted(deps.values(), key=lambda d: d.id):
        lines.append(f"- {d.name}: state {d.state}, priority {d.priority}, {idle_text(d)}"
                     + (f", used by {', '.join(d.dependents)}" if d.dependents else ""))
    return "\n".join(lines)


class _Cache:
    def __init__(self):
        self._d: dict[str, tuple[float, Decision]] = {}

    def get(self, key: str) -> Decision | None:
        hit = self._d.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        self._d.pop(key, None)
        return None

    def put(self, key: str, d: Decision) -> None:
        if len(self._d) >= CACHE_MAX:
            self._d.pop(next(iter(self._d)))
        self._d[key] = (time.monotonic() + CACHE_TTL_S, d)


CACHE = _Cache()


def _defaults(plans: list[dict]) -> list[dict]:
    return [{**p, "recommended": False, "recommendation": None, "unsafe_reason": p.get("unsafe_reason")} for p in plans]


async def recommend(plans: list[dict], *, deps: dict[int, DepFacts], decider, shortfall: int, model_ref: str,
                    engine: str, cfg: dict, raw_cfg: dict, defaults: dict, context_key: str, seqs_key: str,
                    request_priority: int = 0, cache: _Cache | None = None, consult=None) -> list[dict]:
    """`consult()` is asked right before the decision model would be (a cache miss with two or more allowed plans);
    False (the caller's rate limit) -> no model ranking, the plans keep their safety annotations only."""
    out = _defaults(plans)
    for p in out:
        p["unsafe_reason"] = unsafe_reason(p, deps, raw_cfg=raw_cfg, defaults=defaults, context_key=context_key,
                                           seqs_key=seqs_key, request_priority=request_priority)
    allowed = [p for p in out if p["unsafe_reason"] is None]
    subject = "preview:" + hashlib.sha256(json.dumps([model_ref, cfg, [p["id"] for p in out]], sort_keys=True,
                                                     default=str).encode()).hexdigest()[:16]
    if not allowed:
        return out
    cache = CACHE if cache is None else cache
    if len(allowed) == 1:
        p = allowed[0]
        reason = reason_for(p, deps)
        rkey = "rules:" + hashlib.sha256(json.dumps([subject, p["id"], reason]).encode()).hexdigest()
        d = cache.get(rkey)  # one `decisions` row per distinct answer and cache period, not one per preview
        if d is None:
            d = safe_record_rules(decider, feature="plan", subject=subject, choice=p["id"], reason=reason)
            cache.put(rkey, d)
        p["recommended"] = True
        p["recommendation"] = {"source": "rules", "confidence": 1.0, "reason": reason, "decision_id": d.decision_id}
        return out
    options = [_option(p, deps) for p in allowed]
    state = build_state(shortfall=shortfall, model_ref=model_ref, engine=engine, cfg=cfg, deps=deps)
    key = hashlib.sha256(json.dumps([state, [(o.key, o.label, o.description) for o in options]]).encode()).hexdigest()
    d = cache.get(key)
    if d is None and consult is not None and not consult():
        d = abstain("plan", subject)  # over the caller's decider rate limit: rules only (not cached)
    elif d is None:
        d = await safe_decide(decider, feature="plan", subject=subject, state=state, question=QUESTION,
                              options=options, timeout_s=TIMEOUT_S)
        if d.source == "model" or d.probabilities:
            cache.put(key, d)
    chosen = next((p for p in allowed if p["id"] == d.choice), None) if d.source == "model" else None
    if chosen is not None:  # belt and braces: only ever an allowed plan
        chosen["recommended"] = True
        chosen["recommendation"] = {"source": "model", "confidence": d.confidence, "reason": reason_for(chosen, deps),
                                    "decision_id": d.decision_id}
    return out


# ---------- facts from the live system ----------
ACTIVE_VIEW_STATES = ("ready", "sleeping", "unknown", "loading", "starting")


def _busy(s: dict) -> bool:
    return any((s.get(k) or 0) > 0 for k in ("running", "waiting", "gen_tps", "prompt_tps"))


@dataclass
class Traffic:
    last_min: float | None  # minutes since the last request activity (None = none recorded)
    running_now: bool
    covered_min: int        # minutes of the last ACTIVE_WINDOW_MIN with metrics

    @property
    def known(self) -> bool:
        return self.covered_min >= COVERAGE_MIN * ACTIVE_WINDOW_MIN


def traffic(store, dep_id: int, now: float | None = None) -> Traffic:
    """Request activity and metrics coverage of a deployment (live ring + 1-minute rollups)."""
    now = time.time() if now is None else now
    if store is None:
        return Traffic(None, False, 0)
    cur = int(now // 60)
    window = set(range(cur - ACTIVE_WINDOW_MIN + 1, cur + 1))
    covered: set[int] = set()
    last: float | None = None
    running_now = False
    try:
        ring = store.series(dep_id)
    except Exception:
        ring = []
    for smp in ring:
        ts = smp.get("ts") or 0
        if smp.get("running") is not None and int(ts // 60) in window:
            covered.add(int(ts // 60))
        if _busy(smp):
            last = ts if last is None else max(last, ts)
    if ring:
        latest = ring[-1]
        running_now = now - (latest.get("ts") or 0) <= 60 and ((latest.get("running") or 0) > 0
                                                              or (latest.get("waiting") or 0) > 0)
    busy_rollup, rollup_minutes = _rollup_facts(store, dep_id, now)
    covered |= rollup_minutes & window
    if busy_rollup is not None:
        last = busy_rollup if last is None else max(last, busy_rollup)
    return Traffic((now - last) / 60 if last is not None else None, running_now, len(covered))


def _rollup_facts(store, dep_id: int, now: float) -> tuple[float | None, set[int]]:
    """(end of the latest busy rollup minute, minutes of the window with a valid rollup)."""
    db = getattr(store, "db", None)
    if db is None:
        return None, set()
    from datetime import datetime, timezone

    from sqlalchemy import func, or_, select

    from dgxmm.db import MetricRollup, as_aware
    j = lambda k: func.coalesce(func.json_extract(MetricRollup.data, f"$.{k}"), 0)  # noqa: E731
    since = datetime.fromtimestamp((int(now // 60) - ACTIVE_WINDOW_MIN) * 60, timezone.utc)
    try:
        with db.session() as s:
            m = s.scalar(select(func.max(MetricRollup.minute)).where(
                MetricRollup.deployment_id == dep_id,
                or_(j("running") > 0, j("waiting") > 0, j("gen_tps") > 0, j("prompt_tps") > 0)))
            rows = s.execute(select(MetricRollup.minute).where(
                MetricRollup.deployment_id == dep_id, MetricRollup.minute >= since,
                func.json_extract(MetricRollup.data, "$.samples") > 0,
                func.json_extract(MetricRollup.data, "$.running").is_not(None))).scalars().all()
    except Exception:
        log.warning("could not read rollups for deployment %s", dep_id, exc_info=True)
        return None, set()
    minutes = {int(as_aware(r).timestamp() // 60) for r in rows}
    if m is None:
        return None, minutes
    if isinstance(m, str):  # func.max over a DateTime column comes back as text in SQLite
        m = datetime.fromisoformat(m)
    return as_aware(m).timestamp() + 60, minutes  # end of that minute


def _ready_since(db, dep_ids: list[int]) -> dict[int, float]:
    """Latest transition to `ready` per deployment (epoch seconds)."""
    if db is None or not dep_ids:
        return {}
    from sqlalchemy import func, select

    from dgxmm.db import DeploymentEvent, as_aware
    try:
        with db.session() as s:
            rows = s.execute(select(DeploymentEvent.deployment_id, func.max(DeploymentEvent.ts))
                             .where(DeploymentEvent.deployment_id.in_(dep_ids), DeploymentEvent.to_state == "ready")
                             .group_by(DeploymentEvent.deployment_id)).all()
    except Exception:
        log.warning("could not read ready events", exc_info=True)
        return {}
    out = {}
    for dep_id, ts in rows:
        if isinstance(ts, str):
            from datetime import datetime
            ts = datetime.fromisoformat(ts)
        out[dep_id] = as_aware(ts).timestamp()
    return out


def facts_for(views: list[dict], store, exclude: int | None = None, now: float | None = None,
              db=None) -> dict[int, DepFacts]:
    """Facts for every active deployment. Does DB work: call it in a thread from async code."""
    from dgxmm.engines.base import container_name
    now = time.time() if now is None else now
    active = [v for v in views if v["state"] in ACTIVE_VIEW_STATES and v["id"] != exclude]
    ready = _ready_since(db if db is not None else getattr(store, "db", None), [v["id"] for v in active])
    out: dict[int, DepFacts] = {}
    for v in active:
        t = traffic(store, v["id"], now)
        r = ready.get(v["id"])
        out[v["id"]] = DepFacts(id=v["id"], name=v["name"], state=v["state"], priority=int(v.get("priority") or 0),
                                system=bool(v.get("system")) or v.get("engine") == "decision",
                                last_traffic_min=t.last_min, running_now=t.running_now, traffic_known=t.known,
                                uncovered_min=ACTIVE_WINDOW_MIN - t.covered_min,
                                ready_for_min=(now - r) / 60 if r is not None else None)
    for v in active:
        for w in active:
            if w["id"] == v["id"]:
                continue
            cfg = w.get("config") or {}
            spec = cfg.get("speculative") if isinstance(cfg.get("speculative"), dict) else None
            named = re.search(r"(?<![\w-])" + re.escape(container_name(v["name"])) + r"(?![\w-])",
                              json.dumps(cfg, default=str))
            uses = (spec is not None and spec.get("model") == v["model_ref"]) or named is not None
            if uses and w["name"] not in out[v["id"]].dependents:
                out[v["id"]].dependents.append(w["name"])
    return out
