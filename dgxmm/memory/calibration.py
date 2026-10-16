"""Learn from measured memory: exact config → measured × 1.05, same model → mean ratio (spec §4.4).

Only *reliable* measurements are learned from. On GB10 unified memory the cgroup + NVML number
(`budget.actual`) does not include CUDA allocations, so it is NOT a measurement of the footprint and is
never recorded. Instead the recorder takes the engine's own startup accounting from its logs
(`footprint.py`: weights, KV cache, CUDA graphs, activations, ...) and adds the container's host-side
memory (cgroup anon + kernel, which on GB10 holds the Python/C++ heap but not the CUDA allocations).
Every term is a real allocation, so the result is a lower bound on the true footprint.

The MemAvailable delta across startup (snapshot when the container starts vs. when it is ready) is used
only as a cross-check and stored as `quality`:
  verified   - the delta agrees with the log-based figure (within max(15 %, 1 GiB))
  disputed   - the delta disagrees (other host activity, or the engine log missed/double-counted something)
  unverified - no clean delta (another deployment changed state during the window, or no snapshot)
The delta is never the recorded figure: unrelated processes allocating or freeing memory on the host make
it noisy in both directions.

Calibration stays conservative: estimates are only ever raised, never lowered below the formula, as
before (exact config: max(measured × 1.05, estimate); same model: mean ratio clamped to [1.0, 1.5]).
Rows without a reliable method (legacy rows recorded from cgroup + NVML before this change, method NULL)
are ignored, except OOM records, which stay authoritative.
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.db import Database, MemoryCalibration
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.memory.footprint import parse_engine_log

log = logging.getLogger(__name__)
GIB = 1024**3
RELIABLE_METHODS = frozenset({"engine_log"})
DELTA_TOLERANCE = 0.15
DELTA_TOLERANCE_MIN = 1 * GIB
LOG_TAIL = 10000


def _usable(r: MemoryCalibration) -> bool:
    return not r.oom and r.method in RELIABLE_METHODS and r.measured_bytes > 0


class Calibration:
    def __init__(self, db: Database):
        self.db = db

    def record(self, engine: str, repo: str, config_hash: str, estimated: int, measured: int, *, method: str,
               quality: str | None = None, detail: dict | None = None) -> None:
        with self.db.session() as s:
            s.add(MemoryCalibration(engine=engine, repo=repo, config_hash=config_hash, estimated_bytes=estimated,
                                    measured_bytes=measured, oom=False, method=method, quality=quality, detail=detail))

    def record_oom(self, engine: str, repo: str, config_hash: str, estimated: int) -> None:
        with self.db.session() as s:
            s.add(MemoryCalibration(engine=engine, repo=repo, config_hash=config_hash, estimated_bytes=estimated,
                                    measured_bytes=0, oom=True, method="oom"))

    def _rows(self, engine: str, repo: str) -> list[MemoryCalibration]:
        with self.db.session() as s:
            return list(s.scalars(select(MemoryCalibration).where(MemoryCalibration.engine == engine,
                                                                  MemoryCalibration.repo == repo)
                                  .order_by(MemoryCalibration.id.desc())))

    def summary(self, engine: str, repo: str) -> list[dict]:
        return [{"config_hash": r.config_hash, "estimated": r.estimated_bytes, "measured": r.measured_bytes,
                 "oom": r.oom, "method": r.method, "quality": r.quality, "detail": r.detail,
                 "used": r.oom or _usable(r), "ts": r.ts.isoformat()} for r in self._rows(engine, repo)]

    def adjust(self, engine: str, repo: str, config_hash: str, est: MemoryEstimate) -> MemoryEstimate:
        rows = [r for r in self._rows(engine, repo) if r.oom or _usable(r)]  # drop unreliable/legacy measurements
        if not rows:
            return est
        same = [r for r in rows if r.config_hash == config_hash]
        oom_pending = bool(same) and same[0].oom
        exact = next((r for r in same if not r.oom), None)
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


def delta_quality(measured: int, delta: int | None) -> str:
    if delta is None:
        return "unverified"
    return "verified" if abs(delta - measured) <= max(DELTA_TOLERANCE * measured, DELTA_TOLERANCE_MIN) else "disputed"


class CalibrationRecorder:
    """After a deployment becomes ready (and KV is allocated), store estimated vs measured memory."""

    def __init__(self, deployer, calibration: Calibration, delay_s: float = 15.0):
        self.deployer, self.calibration, self.delay_s = deployer, calibration, delay_s
        # dep_id -> {"before": MemAvailable at start, "noisy": another deployment changed state since}
        self._windows: dict[int, dict] = {}

    async def _available(self) -> int | None:
        try:
            return (await asyncio.to_thread(self.deployer.budget.probe.meminfo)).available
        except Exception:
            log.debug("calibration: MemAvailable snapshot failed", exc_info=True)
            return None

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        for other, w in self._windows.items():
            if other != dep_id:
                w["noisy"] = True  # memory moved for a reason other than this deployment's startup
        if to == "starting":
            before = await self._available()
            self._windows[dep_id] = {"before": before, "noisy": False}
            return
        if to != "ready":
            if to != "loading":
                self._windows.pop(dep_id, None)
            return
        if frm == "sleeping":
            self._windows.pop(dep_id, None)
            return
        try:
            await asyncio.sleep(self.delay_s)
            await self._measure(dep_id)
        finally:
            self._windows.pop(dep_id, None)

    async def _measure(self, dep_id: int) -> None:
        dep = self.deployer._get(dep_id)
        if dep is None or dep.state != "ready":
            return
        logs = await asyncio.to_thread(self.deployer.docker.logs, dep.container_id, LOG_TAIL)
        fp = parse_engine_log(dep.engine, logs or "")
        if fp is None:
            # cgroup + NVML alone misses CUDA memory on GB10: not a measurement, so record nothing
            log.info("calibration: %s skipped - no engine memory accounting found in its log", dep.name)
            return
        window = self._windows.get(dep_id)
        budget = await self.deployer.budget.refresh()
        after = budget.available
        actual = next((u.actual for u in budget.deployments if u.deployment_id == dep_id), 0)
        sample = getattr(self.deployer.budget, "last_sample", None)
        gpu_part = (getattr(sample, "managed_gpu", {}) or {}).get(dep.container_id, 0) if sample else 0
        host = max(actual - gpu_part, 0)  # cgroup anon + kernel: host-side heap, not in the engine's accounting
        measured = fp.total + host
        delta = None
        if window and not window["noisy"] and window["before"] is not None:
            delta = window["before"] - after
        quality = delta_quality(measured, delta)
        from dgxmm.deployer.service import config_hash
        raw = dep.estimate.get("total", dep.reserved_bytes) - dep.estimate.get("margin", 0) if dep.estimate else dep.reserved_bytes
        detail = {"engine_bytes": fp.total, "host_bytes": host, "complete": fp.complete, "parts": fp.parts,
                  "memavailable_delta": delta, "cgroup_nvml_bytes": actual}
        self.calibration.record(dep.engine, dep.model_ref, config_hash(dep.config), int(raw), int(measured),
                                method="engine_log", quality=quality, detail=detail)
        log.info("calibration: %s estimated %.1f GiB (formula) measured %.1f GiB (engine log %.1f + host %.1f; "
                 "MemAvailable delta %s) quality=%s", dep.name, raw / GIB, measured / GIB, fp.total / GIB, host / GIB,
                 f"{delta / GIB:.1f} GiB" if delta is not None else "n/a", quality)
