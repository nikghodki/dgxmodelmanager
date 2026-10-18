"""Retention of the decision log and the stored diagnoses (docs/operations/decision-model.md, "Data retention").

- ``decisions``: rows older than ``decider_retention_days`` (30) are deleted, and at most ``decider_max_rows``
  (50 000) of the newest rows are kept. Rows with feedback (what threshold tuning learns from) are exempt from both:
  they are deleted only after ``decider_feedback_retention_days`` (180; never fewer than
  ``decider_retention_days``), so a flood of new rows cannot push them out.
- ``compat_cache``: pre-download compatibility results older than 7 days (dgxmm/decider/compat.py).
- ``diagnoses``: the latest DIAGNOSES_PER_DEPLOYMENT (20) per deployment are kept (also enforced at every insert),
  and a deployment's diagnoses are deleted with it.

Runs in the metrics loop's rollup step (once a minute, in a worker thread; dgxmm/metrics/scraper.py)."""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import delete, func, select

from dgxmm.db import Deployment, DecisionRecord, DiagnosisRow, utcnow

log = logging.getLogger(__name__)
DIAGNOSES_PER_DEPLOYMENT = 20


def prune_diagnoses_of(s, dep_id: int, keep: int = DIAGNOSES_PER_DEPLOYMENT) -> int:
    """Delete all but the newest `keep` diagnoses of one deployment (inside an open session)."""
    cut = s.scalar(select(DiagnosisRow.id).where(DiagnosisRow.deployment_id == dep_id)
                   .order_by(DiagnosisRow.id.desc()).offset(keep - 1).limit(1))
    if cut is None:
        return 0
    return s.execute(delete(DiagnosisRow).where(DiagnosisRow.deployment_id == dep_id, DiagnosisRow.id < cut)).rowcount


def prune(db, settings) -> dict[str, int]:
    """One retention pass; returns how many rows each rule deleted."""
    out = {"decisions_age": 0, "decisions_rows": 0, "diagnoses": 0, "compat_cache": 0}
    cutoff = utcnow() - timedelta(days=settings.decider_retention_days)
    with db.session() as s:
        # never shorter than the plain rows' retention (a lower setting would delete the labelled rows first)
        fb_days = max(settings.decider_feedback_retention_days, settings.decider_retention_days)
        fb_cutoff = utcnow() - timedelta(days=fb_days)
        plain = DecisionRecord.feedback.is_(None)
        out["decisions_age"] = s.execute(delete(DecisionRecord).where(
            ((DecisionRecord.created_at < cutoff) & plain) | (DecisionRecord.created_at < fb_cutoff))).rowcount
        cut = s.scalar(select(DecisionRecord.id).where(plain).order_by(DecisionRecord.id.desc())
                       .offset(settings.decider_max_rows - 1).limit(1))
        if cut is not None:
            out["decisions_rows"] = s.execute(delete(DecisionRecord).where(DecisionRecord.id < cut, plain)).rowcount
        ranked = select(DiagnosisRow.id, func.row_number().over(partition_by=DiagnosisRow.deployment_id,
                                                                order_by=DiagnosisRow.id.desc()).label("rn")).subquery()
        n = s.execute(delete(DiagnosisRow).where(DiagnosisRow.id.in_(
            select(ranked.c.id).where(ranked.c.rn > DIAGNOSES_PER_DEPLOYMENT)))).rowcount
        n += s.execute(delete(DiagnosisRow).where(DiagnosisRow.deployment_id.not_in(select(Deployment.id)))).rowcount
        out["diagnoses"] = n
    from dgxmm.decider.compat import prune_cache
    out["compat_cache"] = prune_cache(db)
    if any(out.values()):
        log.info("decider retention: deleted %s", out)
    return out
