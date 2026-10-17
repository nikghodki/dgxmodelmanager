"""M4: retention of `decisions` (30 days / 50 000 rows, configurable) and `diagnoses` (latest 20 per deployment,
deleted with the deployment), pruned from the metrics loop."""
from datetime import timedelta

from sqlalchemy import func, insert, select

from dgxmm.db import DecisionRecord, DiagnosisRow, utcnow
from dgxmm.decider import retention
from dgxmm.decider.retention import DIAGNOSES_PER_DEPLOYMENT, prune
from tests.fakes import FakeDecider
from tests.test_decider_diagnosis import LOGS, fail_with, make_ctx


def _decisions(db, n, age_days=0.0):
    at = utcnow() - timedelta(days=age_days)
    with db.session() as s:
        s.execute(insert(DecisionRecord), [{"feature": "plan", "subject": f"s{i}", "source": "rules", "choice": "x",
                                            "confidence": 1.0, "probabilities": {}, "created_at": at}
                                           for i in range(n)])


def _count(db, model, *where):
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where))


def test_defaults_are_30_days_and_50000_rows(settings):
    assert settings.decider_retention_days == 30 and settings.decider_max_rows == 50_000


def test_decisions_older_than_the_retention_are_deleted(db, settings):
    _decisions(db, 7, age_days=31)
    _decisions(db, 5, age_days=29)
    out = prune(db, settings)
    assert out["decisions_age"] == 7 and _count(db, DecisionRecord) == 5


def test_decisions_are_capped_at_max_rows_keeping_the_newest(db, settings):
    settings.decider_max_rows = 1000
    _decisions(db, 1250)
    with db.session() as s:
        newest = s.scalar(select(func.max(DecisionRecord.id)))
    prune(db, settings)
    assert _count(db, DecisionRecord) == 1000
    with db.session() as s:
        assert s.scalar(select(func.max(DecisionRecord.id))) == newest
        assert s.scalar(select(func.min(DecisionRecord.id))) == newest - 999


async def test_diagnoses_keep_the_latest_20_per_deployment_and_go_with_it(stack):
    ctx = make_ctx(stack, FakeDecider())
    dep_id = await fail_with(stack, ctx, "Loading...\n" + LOGS["disk_full"])
    for i in range(DIAGNOSES_PER_DEPLOYMENT + 5):  # insert-time pruning
        ctx.diagnosis._insert(deployment_id=dep_id, event_id=10_000 + i, cause="disk_full", source="rules",
                              confidence=1.0, decision_id=None, summary=f"s{i}", fix_hint="", patch=None,
                              log_excerpt="")
    assert _count(stack.db, DiagnosisRow, DiagnosisRow.deployment_id == dep_id) == DIAGNOSES_PER_DEPLOYMENT
    assert ctx.diagnosis.latest(dep_id)["summary"] == f"s{DIAGNOSES_PER_DEPLOYMENT + 4}"
    # rows written past the insert-time rule (e.g. by an older build) are pruned by the periodic pass
    with stack.db.session() as s:
        s.execute(insert(DiagnosisRow), [{"deployment_id": dep_id, "event_id": 20_000 + i, "source": "none",
                                          "summary": "", "fix_hint": "", "log_excerpt": ""} for i in range(7)])
    assert prune(stack.db, stack.settings)["diagnoses"] == 7
    assert _count(stack.db, DiagnosisRow, DiagnosisRow.deployment_id == dep_id) == DIAGNOSES_PER_DEPLOYMENT
    await stack.deployer.remove(dep_id)
    assert _count(stack.db, DiagnosisRow, DiagnosisRow.deployment_id == dep_id) == 0


async def test_the_metrics_loop_runs_the_retention_pass(stack, monkeypatch):
    from dgxmm.metrics.scraper import MetricsScraper
    from dgxmm.metrics.store import MetricsStore
    calls = []
    sc = MetricsScraper(stack.settings, stack.deployer, None, MetricsStore(stack.db), stack.events, stack.budget)
    sc.retention.append(lambda: calls.append(prune(stack.db, stack.settings)))
    sc.retention.insert(0, lambda: (_ for _ in ()).throw(RuntimeError("boom")))  # one failing pass stops nothing
    sc._rollup()
    assert len(calls) == 1


def test_build_context_wires_the_retention_into_the_metrics_loop():
    import inspect

    from dgxmm import context
    src = inspect.getsource(context)
    assert "scraper.retention.append" in src and "prune_decider_tables(db, settings)" in src
    assert retention.prune is prune
