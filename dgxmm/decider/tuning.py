"""Feature 9 — threshold tuning from feedback (rules only; spec "9. Threshold tuning").

Weekly, for every settings group whose features ask the model: take the model decisions of the last 30 days that an
admin judged (``decisions.feedback`` accepted / rejected; rules decisions are not the model's and are left out) and,
with at least MIN_FEEDBACK of them, compute the precision of the model's answers at each candidate threshold
(0.50..0.95 step 0.05) with its Wilson 95 % lower bound (``common.wilson``). The proposal is the lowest candidate
whose lower bound reaches the feature's target (TARGETS: 0.95, command 0.9), if it differs from the current
threshold by at least MIN_CHANGE. It is an insight with the evidence table in its (templated) body and a
``threshold`` action; applying goes through PUT /api/decider/settings (admin, audited). Nothing is changed here.

When no candidate qualifies, or the best one is the current threshold, an open proposal is resolved.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import select

from dgxmm.db import DecisionRecord, utcnow
from dgxmm.decider import registry
from dgxmm.decider.common import plural, wilson
from dgxmm.decider.insights import PROPOSAL_TTL

log = logging.getLogger(__name__)
JOB = "threshold_tuning"
SCHEDULE = "weekly sun 04:30"
WINDOW = timedelta(days=30)
MIN_FEEDBACK = 20
CANDIDATES = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
DEFAULT_TARGET = 0.95
TARGETS = {"command": 0.90}
MIN_CHANGE = 0.05 - 1e-9


def target_of(group: str) -> float:
    return TARGETS.get(group, DEFAULT_TARGET)


def tunable_groups() -> list[str]:
    """Settings groups whose features ask the model (tuning itself is rules only)."""
    return [s.name for s in registry.groups() if s.needs_model]


def judged(db, group: str, *, now=None) -> list[tuple[float, bool]]:
    """(confidence, accepted) of the judged model decisions of ``group`` (its members included) in the window."""
    now = now or utcnow()
    names = [s.name for s in registry.all_features() if s.group == group]
    with db.session() as s:
        rows = s.execute(select(DecisionRecord.confidence, DecisionRecord.feedback).where(
            DecisionRecord.feature.in_(names), DecisionRecord.source.in_(("model", "none")),
            DecisionRecord.confidence.is_not(None), DecisionRecord.feedback.in_(("accepted", "rejected")),
            DecisionRecord.created_at >= now - WINDOW)).all()
    out = []
    for conf, fb in rows:
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if 0.0 <= c <= 1.0:  # NaN fails both comparisons
            out.append((c, fb == "accepted"))
    return out


def evidence(rows: list[tuple[float, bool]]) -> list[dict]:
    """Per candidate threshold: judged answers at or above it, accepted ones, precision and the Wilson bounds."""
    out = []
    for t in CANDIDATES:
        kept = [ok for c, ok in rows if c >= t - 1e-9]
        k, n = sum(kept), len(kept)
        lo, hi = wilson(k, n)
        out.append({"threshold": t, "n": n, "accepted": k, "precision": round(k / n, 4) if n else None,
                    "wilson_lo": None if lo is None else round(lo, 4),
                    "wilson_hi": None if hi is None else round(hi, 4)})
    return out


def propose(rows: list[tuple[float, bool]], current: float, target: float) -> dict | None:
    """{value, evidence} for the lowest candidate whose Wilson lower bound >= target, if it differs from
    ``current`` by >= MIN_CHANGE; None otherwise (too little feedback, nothing qualifies, or no change)."""
    if len(rows) < MIN_FEEDBACK:
        return None
    table = evidence(rows)
    best = next((r for r in table if r["n"] and r["wilson_lo"] is not None and r["wilson_lo"] >= target), None)
    if best is None or abs(best["threshold"] - current) < MIN_CHANGE:
        return None
    return {"value": best["threshold"], "evidence": table}


def _f(v, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f} %" if pct else f"{v:.2f}"


def render_body(label: str, current: float, value: float, target: float, n: int, table: list[dict]) -> str:
    """The insight body: a fixed template with numbers only (no model or user text)."""
    direction = "Lowering" if value < current else "Raising"
    lines = [f"{direction} the {label} threshold from {current:.2f} to {value:.2f} is supported by "
             f"{plural(n, 'judged model answer')} of the last {WINDOW.days} days: at {value:.2f} the Wilson 95 % "
             f"lower bound of their precision reaches the target of {target * 100:.0f} %.",
             ""]
    cells = [["threshold", "answers", "accepted", "precision", "Wilson 95 % lower bound"]]
    for r in table:
        mark = " (proposed)" if abs(r["threshold"] - value) < 1e-9 else (
            " (current)" if abs(r["threshold"] - current) < 1e-9 else "")
        cells.append([f"{r['threshold']:.2f}{mark}", str(r["n"]), str(r["accepted"]), _f(r["precision"], True),
                      _f(r["wilson_lo"], True)])
    # An aligned plain-text table (the UI renders "a | b" rows with a "--- | ---" rule as a table).
    widths = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
    fmt = lambda row: " | ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()  # noqa: E731
    lines += [fmt(cells[0]), " | ".join("-" * w for w in widths), *(fmt(row) for row in cells[1:])]
    lines += ["", "Apply sets the threshold through the Decision model settings (audited); nothing changes until "
                  "then."]
    return "\n".join(lines)


def dedupe_key(group: str) -> str:
    return f"tuning:{group}"


def tune(ctx, *, now=None) -> dict:
    """One pass over every tunable feature: raise / refresh / resolve proposals. Sync (runs in a job thread)."""
    decider, insights = ctx.decider, ctx.insights
    out = {"proposed": [], "checked": 0}
    for group in tunable_groups():
        rows = judged(ctx.db, group, now=now)
        current = float(decider.threshold(group))
        target = target_of(group)
        p = propose(rows, current, target)
        out["checked"] += 1
        if p is None:
            insights.resolve(dedupe_key(group))
            continue
        label = registry.get(group).label
        body = render_body(label, current, p["value"], target, len(rows), p["evidence"])
        insights.upsert("tuning", "tuning", f"feature:{group}", "info",
                        f"{label}: threshold {current:.2f} → {p['value']:.2f} suggested", body,
                        dedupe_key=dedupe_key(group), source="rules",
                        action={"type": "threshold", "feature": group, "value": p["value"],
                                "label": f"Set threshold to {p['value']:.2f}"},
                        expires_at=(now or utcnow()) + PROPOSAL_TTL)
        out["proposed"].append(group)
    return out


def job(run) -> str:
    """Insight-engine job (sync: the engine runs it in its thread pool)."""
    res = tune(run.ctx)
    return f"{plural(res['checked'], 'feature')} checked, {plural(len(res['proposed']), 'proposal')}"
