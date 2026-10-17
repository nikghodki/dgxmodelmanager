"""Live samples (ring buffer, ~1 h at 5 s) and 1-minute rollups in SQLite for 30 days (spec §8)."""
from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import delete, select

from dgxmm.db import Database, MetricRollup, as_aware
from dgxmm.engines.base import EngineMetrics
from dgxmm.engines.prom import delta_buckets, histogram_quantile

NUMERIC_KEYS = ("running", "waiting", "kv_usage", "prompt_tps", "gen_tps", "ttft_p50", "ttft_p95", "itl_p50",
                "itl_p95", "e2e_p50", "prefix_hit_rate", "spec_accept_rate", "preemptions")
HOST_KEYS = ("used", "available", "free_for_new", "external", "committed", "gpu_util", "temp_c", "power_w")
RETENTION = timedelta(days=30)


def _finite(v):
    """NaN/Inf are not valid JSON (FastAPI would answer 500): store them as missing."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _clean(sample: dict) -> dict:
    return {k: _finite(v) for k, v in sample.items()}


def _delta(cur: EngineMetrics, prev: EngineMetrics, key: str) -> float | None:
    a, b = cur.counters.get(key), prev.counters.get(key)
    if a is None or b is None or a < b:
        return None
    return a - b


class MetricsStore:
    def __init__(self, db: Database, keep_points: int = 720, clock: Callable[[], float] = time.time):
        self.db, self.keep, self.clock = db, keep_points, clock
        self._rings: dict[int | str, deque] = {}
        self._prev: dict[int, tuple[float, EngineMetrics]] = {}
        self._rolled: dict[int | str, int] = {}

    def _ring(self, key: int | str) -> deque:
        return self._rings.setdefault(key, deque(maxlen=self.keep))

    def add(self, dep_id: int, ts: float, m: EngineMetrics) -> dict:
        s: dict = {k: None for k in NUMERIC_KEYS}
        s.update(ts=ts, running=m.gauges.get("running"), waiting=m.gauges.get("waiting"), kv_usage=m.gauges.get("kv_usage"))
        prev = self._prev.get(dep_id)
        if prev:
            pts, pm = prev
            dt = ts - pts
            if dt > 0:
                for out_key, counter in (("prompt_tps", "prompt_tokens"), ("gen_tps", "generation_tokens")):
                    d = _delta(m, pm, counter)
                    s[out_key] = d / dt if d is not None else None
            for hist in ("ttft", "itl", "e2e"):
                if hist in m.histograms:
                    db = delta_buckets(m.histograms[hist], pm.histograms.get(hist))
                    s[f"{hist}_p50"] = histogram_quantile(db, 0.5)
                    if hist != "e2e":
                        s[f"{hist}_p95"] = histogram_quantile(db, 0.95)
            for out_key, num, den in (("prefix_hit_rate", "prefix_hits", "prefix_queries"),
                                      ("spec_accept_rate", "spec_accepted", "spec_draft")):
                dn, dd = _delta(m, pm, num), _delta(m, pm, den)
                s[out_key] = dn / dd if dn is not None and dd else None
            s["preemptions"] = _delta(m, pm, "preemptions")
        self._prev[dep_id] = (ts, m)
        s = _clean(s)
        self._ring(dep_id).append(s)
        return s

    def add_host(self, ts: float, budget, gpu: dict | None) -> dict:
        gpu = gpu or {}
        s = {"ts": ts, "used": budget.total - budget.available, "available": budget.available,
             "free_for_new": budget.free_for_new, "external": budget.external, "committed": budget.committed,
             "gpu_util": gpu.get("util"), "temp_c": gpu.get("temp_c"), "power_w": gpu.get("power_w")}
        s = _clean(s)
        self._ring("host").append(s)
        return s

    def series(self, key: int | str) -> list[dict]:
        return list(self._rings.get(key, ()))

    def forget(self, dep_id: int) -> None:
        """Drop a removed deployment's live samples and rollups (SQLite reuses deployment ids, so a later
        deployment must not inherit them)."""
        for d in (self._rings, self._prev, self._rolled):
            d.pop(dep_id, None)
        with self.db.session() as sess:
            sess.execute(delete(MetricRollup).where(MetricRollup.deployment_id == int(dep_id)))
            sess.commit()

    def rollup(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        current_minute = int(now // 60) * 60
        written = 0
        rolled_keys = []
        with self.db.session() as sess:
            for key, ring in self._rings.items():
                last = self._rolled.get(key, 0)
                groups: dict[int, list[dict]] = {}
                for s in ring:
                    minute = int(s["ts"] // 60) * 60
                    if last <= minute < current_minute:
                        groups.setdefault(minute, []).append(s)
                keys = HOST_KEYS if key == "host" else NUMERIC_KEYS
                for minute, samples in sorted(groups.items()):
                    data = {}
                    for k in keys:
                        vals = [x[k] for x in samples if x.get(k) is not None]
                        data[k] = sum(vals) / len(vals) if vals else None
                    if key != "host":
                        waits = [x["waiting"] for x in samples if x.get("waiting") is not None]
                        data["waiting_max"] = max(waits) if waits else None
                        data["samples"] = len(samples)
                    sess.add(MetricRollup(deployment_id=None if key == "host" else int(key),
                                          minute=datetime.fromtimestamp(minute, timezone.utc), data=data))
                    written += 1
                if groups:
                    rolled_keys.append(key)
            sess.execute(delete(MetricRollup).where(MetricRollup.minute < datetime.now(timezone.utc) - RETENTION))
        for key in rolled_keys:
            self._rolled[key] = current_minute
        return written

    def history(self, key: int | str, hours: float) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.db.session() as sess:
            q = select(MetricRollup).where(MetricRollup.minute >= since).order_by(MetricRollup.minute)
            q = q.where(MetricRollup.deployment_id.is_(None)) if key == "host" else q.where(MetricRollup.deployment_id == int(key))
            return [{"ts": as_aware(r.minute).timestamp(), **r.data} for r in sess.scalars(q)]
