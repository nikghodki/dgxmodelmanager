"""Feature 9 — threshold tuning from feedback (dgxmm/decider/tuning.py): Wilson lower bounds on synthetic feedback
histories, the minimum count, the target per feature, the change step, templated insight with a threshold action,
the weekly job (rules only: runs with the decision model disabled)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from dgxmm.db import DecisionRecord, Insight, utcnow
from dgxmm.decider import tuning
from dgxmm.decider.common import wilson
from dgxmm.decider.insights import InsightService, validate_action


def feedback(db, feature, rows, *, days_ago=1, source="model"):
    """rows: [(confidence, accepted)]"""
    with db.session() as s:
        for conf, ok in rows:
            s.add(DecisionRecord(feature=feature, subject="x", source=source, choice="a", confidence=conf,
                                 created_at=utcnow() - timedelta(days=days_ago),
                                 feedback="accepted" if ok else "rejected"))


def history(n_high=120, n_mid=40, mid_ok=0.7, n_low=10):
    """Answers >= 0.9 always right, 0.7-0.9 right at mid_ok, below 0.7 mostly wrong."""
    rows = [(0.9 + 0.09 * (i % 10) / 10, True) for i in range(n_high)]
    rows += [(0.7 + 0.19 * (i % 10) / 10, i % 10 < mid_ok * 10) for i in range(n_mid)]
    rows += [(0.5 + 0.19 * (i % 10) / 10, i % 3 == 0) for i in range(n_low)]
    return rows


def test_evidence_table_has_wilson_bounds_per_candidate():
    rows = history()
    table = tuning.evidence(rows)
    assert [r["threshold"] for r in table] == tuning.CANDIDATES
    top = next(r for r in table if r["threshold"] == 0.9)
    assert top["n"] == 120 and top["accepted"] == 120 and top["precision"] == 1.0
    assert top["wilson_lo"] == pytest.approx(wilson(120, 120)[0], abs=1e-4)
    assert table[0]["n"] == len(rows)


def test_proposes_the_lowest_threshold_whose_lower_bound_reaches_the_target():
    rows = history()
    p = tuning.propose(rows, current=0.8, target=0.95)
    # 0.9: 120/120 -> lower bound 0.969; 0.85 already includes wrong answers
    assert p["value"] == 0.9
    lo_85 = next(r for r in p["evidence"] if r["threshold"] == 0.85)["wilson_lo"]
    assert lo_85 < 0.95


def test_no_proposal_below_the_minimum_feedback_or_without_change_or_qualifier():
    assert tuning.propose([(0.99, True)] * (tuning.MIN_FEEDBACK - 1), 0.8, 0.95) is None
    assert tuning.propose(history(), 0.9, 0.95) is None            # already there
    assert tuning.propose(history(), 0.88, 0.95) is None           # differs by < 0.05
    assert tuning.propose([(0.99, i % 2 == 0) for i in range(60)], 0.8, 0.95) is None  # nothing reaches 95 %
    # 30 perfect answers: lower bound 0.886 < 0.95 -> no proposal yet, even though precision is 1.0
    assert tuning.propose([(0.97, True)] * 30, 0.8, 0.95) is None


def test_command_has_a_lower_target():
    assert tuning.target_of("command") == 0.9 and tuning.target_of("health") == 0.95
    rows = [(0.85, True)] * 40 + [(0.6, False)] * 10
    assert tuning.propose(rows, 0.7, tuning.target_of("command"))["value"] == 0.65
    assert tuning.propose(rows, 0.7, tuning.target_of("health")) is None  # 40/40 -> 0.912 < 0.95


def test_can_also_propose_raising():
    rows = [(0.75, False)] * 10 + [(0.95, True)] * 100
    assert tuning.propose(rows, 0.7, 0.95)["value"] == 0.8


def test_judged_only_model_answers_with_feedback_in_window_and_members_count(db):
    feedback(db, "parser_tool", [(0.95, True)] * 3)
    feedback(db, "parser_reasoning", [(0.95, False)] * 2)
    feedback(db, "parser_tool", [(0.95, True)] * 5, days_ago=40)     # too old
    feedback(db, "parser_tool", [(1.0, True)] * 4, source="rules")    # not the model's
    with db.session() as s:
        s.add(DecisionRecord(feature="parser_tool", subject="x", source="model", choice="a", confidence=0.9))  # no fb
        s.add(DecisionRecord(feature="parser_tool", subject="x", source="model", choice="a", confidence=float("nan"),
                             feedback="accepted"))
    rows = tuning.judged(db, "parser")
    assert sorted(rows) == sorted([(0.95, True)] * 3 + [(0.95, False)] * 2)


class Th:
    def __init__(self, values=None):
        self.values = values or {}

    def threshold(self, feature):
        return self.values.get(feature, 0.8)


def tctx(ctx, thresholds=None):
    ctx.decider = Th(thresholds)
    ctx.insights = InsightService(ctx.db)
    return ctx


def test_tune_raises_a_templated_proposal_with_a_threshold_action(ctx):
    feedback(ctx.db, "health", history())
    tctx(ctx)
    res = tuning.tune(ctx)
    assert res["proposed"] == ["health"] and res["checked"] == len(tuning.tunable_groups())
    assert "tuning" not in tuning.tunable_groups()
    [ins] = ctx.insights.list(status=["open"])
    assert ins["kind"] == "tuning" and ins["feature"] == "tuning" and ins["subject"] == "feature:health"
    assert ins["action"] == validate_action({"type": "threshold", "feature": "health", "value": 0.9,
                                             "label": "Set threshold to 0.90"})
    assert "0.80 to 0.90" in ins["body"] and "0.90 (proposed)" in ins["body"] and "0.80 (current)" in ins["body"]
    assert "Wilson 95 % lower bound" in ins["body"] and len(ins["body"]) <= 4000
    assert ins["expires_at"] is not None and ins["source"] == "rules"
    # weekly again: refreshed in place, not duplicated
    tuning.tune(ctx)
    with ctx.db.session() as s:
        assert s.query(Insight).count() == 1


def test_tune_resolves_a_proposal_once_it_no_longer_applies(ctx):
    feedback(ctx.db, "health", history())
    tctx(ctx)
    tuning.tune(ctx)
    ctx.decider = Th({"health": 0.9})  # the admin applied it
    tuning.tune(ctx)
    assert ctx.insights.list(status=["open"]) == []


async def test_weekly_job_is_rules_only_and_runs_with_the_decider_disabled(ctx):
    from datetime import timezone

    from dgxmm.decider.features import wire_features
    from dgxmm.decider.insight_engine import InsightEngine
    feedback(ctx.db, "kind", history())
    assert not ctx.decider.enabled
    ctx.insight_engine = InsightEngine(ctx, tz=timezone.utc)
    wire_features(ctx)
    job = next(j for j in ctx.insight_engine.status() if j["name"] == tuning.JOB)
    assert job["schedule"] == "weekly sun 04:30" and job["budget_calls"] == 0 and job["feature"] == "tuning"
    ctx.insight_engine.trigger(tuning.JOB)
    assert tuning.JOB in await ctx.insight_engine.tick()
    await ctx.insight_engine.wait_idle()
    st = next(j for j in ctx.insight_engine.status() if j["name"] == tuning.JOB)
    assert st["last_result"].startswith("ok:") and st["calls_used"] == 0
    assert [i["subject"] for i in ctx.insights.list(status=["open"])] == ["feature:kind"]
    # switched off: the job does not run
    ctx.decider.set_feature("tuning", enabled=False)
    ctx.insight_engine.trigger(tuning.JOB)
    assert tuning.JOB not in await ctx.insight_engine.tick()


def test_tuning_insights_are_admin_only(ctx):
    feedback(ctx.db, "health", history())
    tctx(ctx)
    tuning.tune(ctx)
    assert ctx.insights.list(admin=False) == []
