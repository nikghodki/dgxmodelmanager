"""Learn from measured memory: exact config → measured × 1.05, same model → mean ratio (spec §4.4)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.db import Database, MemoryCalibration
from dgxmm.memory.estimate import MemoryEstimate

log = logging.getLogger(__name__)
GIB = 1024**3


class Calibration:
    def __init__(self, db: Database):
        self.db = db

    def record(self, engine: str, repo: str, config_hash: str, estimated: int, measured: int) -> None:
        with self.db.session() as s:
            s.add(MemoryCalibration(engine=engine, repo=repo, config_hash=config_hash, estimated_bytes=estimated,
                                    measured_bytes=measured, oom=False))

    def record_oom(self, engine: str, repo: str, config_hash: str, estimated: int) -> None:
        with self.db.session() as s:
            s.add(MemoryCalibration(engine=engine, repo=repo, config_hash=config_hash, estimated_bytes=estimated,
                                    measured_bytes=0, oom=True))

    def _rows(self, engine: str, repo: str) -> list[MemoryCalibration]:
        with self.db.session() as s:
            return list(s.scalars(select(MemoryCalibration).where(MemoryCalibration.engine == engine,
                                                                  MemoryCalibration.repo == repo)
                                  .order_by(MemoryCalibration.id.desc())))

    def summary(self, engine: str, repo: str) -> list[dict]:
        return [{"config_hash": r.config_hash, "estimated": r.estimated_bytes, "measured": r.measured_bytes,
                 "oom": r.oom, "ts": r.ts.isoformat()} for r in self._rows(engine, repo)]

    def adjust(self, engine: str, repo: str, config_hash: str, est: MemoryEstimate) -> MemoryEstimate:
        rows = self._rows(engine, repo)
        if not rows:
            return est
        same = [r for r in rows if r.config_hash == config_hash]
        oom_pending = bool(same) and same[0].oom
        exact = next((r for r in same if not r.oom and r.measured_bytes > 0), None)
        if exact:
            # Controller ruling: never lower the estimate
            new_total = max(int(exact.measured_bytes * 1.05), est.total)
            est = est.with_total(new_total, "measured",
                                 f"measured {exact.measured_bytes / GIB:.1f} GiB with this exact configuration")
        else:
            ratios = [r.measured_bytes / r.estimated_bytes for r in rows if not r.oom and r.estimated_bytes > 0][:5]
            if ratios:
                # Controller ruling: ratio clamped to [1.0, 1.5] not [0.9, 1.5]
                ratio = min(max(sum(ratios) / len(ratios), 1.0), 1.5)
                est = est.with_total(int(est.total * ratio), "calibrated",
                                     f"scaled ×{ratio:.2f} from {len(ratios)} measurement(s) of this model")
        if oom_pending:
            est = est.with_total(int(est.total * 1.10), est.confidence, "last attempt was OOM-killed; added 10%")
        # Never swing back to (or below) a size this exact configuration was OOM-killed at
        oom_sizes = [r.estimated_bytes for r in same if r.oom]
        if oom_sizes:
            floor = int(max(oom_sizes) * 1.10)
            if est.total < floor:
                est = est.with_total(floor, est.confidence,
                                     f"OOM-killed before at {max(oom_sizes) / GIB:.1f} GiB; floored at +10%")
        return est


class CalibrationRecorder:
    """After a deployment becomes ready (and KV is allocated), store estimated vs measured memory."""

    def __init__(self, deployer, calibration: Calibration, delay_s: float = 15.0):
        self.deployer, self.calibration, self.delay_s = deployer, calibration, delay_s

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        if to != "ready" or frm == "sleeping":
            return
        await asyncio.sleep(self.delay_s)
        dep = self.deployer._get(dep_id)
        if dep is None or dep.state != "ready":
            return
        budget = await self.deployer.budget.refresh()
        actual = next((u.actual for u in budget.deployments if u.deployment_id == dep_id), 0)
        if actual <= 0:
            return
        from dgxmm.deployer.service import config_hash
        raw = dep.estimate.get("total", dep.reserved_bytes) - dep.estimate.get("margin", 0) if dep.estimate else dep.reserved_bytes
        self.calibration.record(dep.engine, dep.model_ref, config_hash(dep.config), int(raw), int(actual))
        log.info("calibration: %s estimated %.1f GiB (formula) measured %.1f GiB", dep.name, raw / GIB, actual / GIB)
