"""Prometheus text helpers shared by engines."""
from __future__ import annotations

import math
from collections import defaultdict

from prometheus_client.parser import text_string_to_metric_families

Samples = dict[str, list[tuple[dict[str, str], float]]]


def parse_samples(text: str) -> Samples:
    out: Samples = defaultdict(list)
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out[s.name].append((dict(s.labels), float(s.value)))
    return out


def gauge(samples: Samples, *names: str) -> float | None:
    for n in names:
        if n in samples:
            return sum(v for _, v in samples[n])
    return None


def counter(samples: Samples, *names: str) -> float | None:
    for n in names:
        for candidate in (n + "_total", n):
            if candidate in samples:
                return sum(v for _, v in samples[candidate])
    return None


def buckets(samples: Samples, *names: str) -> list[tuple[float, float]]:
    for n in names:
        rows = samples.get(n + "_bucket")
        if rows:
            agg: dict[float, float] = defaultdict(float)
            for labels, v in rows:
                le = labels.get("le", "+Inf")
                agg[math.inf if le in ("+Inf", "inf") else float(le)] += v
            return sorted(agg.items())
    return []


def histogram_quantile(bkts: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style linear interpolation over cumulative (le, count) buckets."""
    if not bkts or bkts[-1][1] <= 0:
        return None
    rank = q * bkts[-1][1]
    prev_le, prev_count = 0.0, 0.0
    for le, count in bkts:
        if count >= rank:
            if math.isinf(le):
                return prev_le
            if count == prev_count:
                return le
            return prev_le + (le - prev_le) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return prev_le


def delta_buckets(cur: list[tuple[float, float]], prev: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    if not prev:
        return cur
    p = dict(prev)
    return [(le, max(c - p.get(le, 0.0), 0.0)) for le, c in cur]
