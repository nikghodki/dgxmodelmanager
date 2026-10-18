"""Feature 8 — benchmark / calibration reading labels (spec "8. Reading labels").

Each new benchmark level (``benchmarks`` row) and each new engine-log memory measurement (``memory_calibration``
row) is labelled ``plausible``, ``contended``, ``cold_cache`` or ``broken`` shortly after it is stored: the stores
call ``ReadingService.notify``, which triggers the insight-engine job ``reading_labels`` (it also runs every
SWEEP_S as a safety net and picks up any unlabelled reading of the last LOOKBACK).

Rules, against the prior readings of the same measurement (benchmark: model, engine, config, concurrency and
lengths; calibration: engine, model and config), compared by median and MAD (robust z):
- ``broken``: the reading is not a measurement (no successful request, zero throughput, many failed requests,
  an incomplete or empty engine log, a footprint far from the formula);
- within Z_OK of the priors: ``plausible`` (host activity during it did not move it);
- off by more, with host contention at the time (benchmarks: another deployment changing state, other programs'
  memory moving by CONTENTION_BYTES or more; calibration: another deployment changing state during the startup, or
  MemAvailable dropping by more than the engine accounts for): ``contended``; a benchmark that is the first one after the deployment
  started and slower: ``cold_cache``;
- no priors and no contention signal: ``plausible``. Everything else is inconclusive and only then is the decision
  model asked; below its threshold the reading stays unlabelled (the store's own quality stays as it was).

The label maps onto the existing ``quality`` column: ``plausible`` -> ``verified``, every other label ->
``disputed``. Calibration (``Calibration.adjust``) and the benchmark advice skip disputed readings. Readings the
rules leave undecided and the model does not answer confidently stay unlabelled and are used exactly as before
(fail open); the label never upgrades calibration confidence (only the MemAvailable check claims "measured"). With the
feature switched off nothing is labelled and calibration keeps its own MemAvailable check and the
two-consecutive-readings rule for unexplained excess (which keeps working with labels too: the original delta
quality is kept in ``reading.delta_quality``). A ``broken`` reading raises an insight.
"""
from __future__ import annotations

import logging
import statistics
from datetime import timedelta
from types import SimpleNamespace

from sqlalchemy import select

from dgxmm.db import Benchmark, BenchmarkRun, DeploymentEvent, MemoryCalibration, MetricRollup, as_aware, utcnow
from dgxmm.decider.common import plural, quoted_block, safe_decide, safe_record_rules
from dgxmm.decider.insights import PROPOSAL_TTL, deployment_subject
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
JOB = "reading_labels"
SWEEP_S = 900
LOOKBACK = timedelta(hours=24)
BUDGET_CALLS = 8
MAX_PER_RUN = 64
TIMEOUT_S = 10.0
GIB = 1024**3

MIN_PRIORS = 2
PRIORS = 8                  # newest prior readings compared
Z_OK = 3.0                  # robust z within this: consistent with the priors
REL_FLOOR = {"benchmark": 0.05, "calibration": 0.03}  # scale floor (share of the median) for tight priors
# Other programs' memory ("external" in the host rollups) moving by this much during a benchmark level. On GB10 the
# benchmarked deployment's own CUDA allocations partly show up as external, so quiet benchmark windows on the box
# moved it by up to 3.9 GiB (tests/fixtures/decider_corpus/reading); another deployment starting moves it by more.
CONTENTION_BYTES = 5 * GIB
MAX_ERROR_SHARE = 0.2
CAL_FORMULA_RANGE = (0.25, 4.0)  # observed / formula outside this: not a measurement of this config
BROKEN_FACTOR = 10.0        # a benchmark this many times faster than its priors is not a real reading
QUALITY = {"plausible": "verified"}  # every other label -> "disputed"

LABELS: list[tuple[str, str]] = [
    ("plausible", "A normal reading: consistent with earlier readings of the same measurement, or nothing "
                  "suggests it was disturbed"),
    ("contended", "Disturbed by other work on the host at the time (another deployment starting or stopping, "
                  "other programs allocating memory or using the GPU)"),
    ("cold_cache", "The first measurement after the deployment started: caches and compiled kernels were still "
                   "warming up, so it is slower than later ones"),
    ("broken", "Not a real measurement: failed requests, zero or impossible values, incomplete accounting"),
]
OPTIONS = [Option(k, k.replace("_", " "), d) for k, d in LABELS]
QUESTION = "Which label fits this new reading?"


def quality_of(label: str) -> str:
    return QUALITY.get(label, "disputed")


# ---------------- statistics ----------------
def robust_z(value: float, priors: list[float], kind: str) -> dict:
    """{n, median, mad, scale, z}; z is None with fewer than MIN_PRIORS priors."""
    vals = [float(v) for v in priors if isinstance(v, (int, float))]
    if len(vals) < MIN_PRIORS:
        return {"n": len(vals), "median": statistics.median(vals) if vals else None, "mad": None, "scale": None,
                "z": None}
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    scale = max(1.4826 * mad, REL_FLOOR[kind] * abs(med), 1e-9)
    return {"n": len(vals), "median": med, "mad": mad, "scale": scale, "z": round((value - med) / scale, 3)}


# ---------------- facts of one reading (pure data, also what the corpus stores) ----------------
def benchmark_broken(f: dict) -> str | None:
    req, err = f.get("requests") or 0, f.get("errors") or 0
    if req <= 0:
        return "no successful request"
    if (f.get("tps_total") or 0) <= 0 or (f.get("tps_per_req") or 0) <= 0:
        return "zero throughput"
    if err and err / (req + err) > MAX_ERROR_SHARE:
        return f"{err} of {req + err} requests failed"
    if f.get("ttft_p50") is None:
        return "no time to first token"
    med = f.get("prior_median")
    if med and f["tps_total"] > BROKEN_FACTOR * med:
        return "throughput far above any earlier reading"
    return None


def calibration_broken(f: dict) -> str | None:
    if f.get("complete") is False:
        return "the engine log's memory accounting is incomplete"
    obs = f.get("value") or 0
    if obs <= 0:
        return "no memory accounted"
    est = f.get("estimated") or 0
    if est > 0 and not CAL_FORMULA_RANGE[0] <= obs / est <= CAL_FORMULA_RANGE[1]:
        return f"{obs / GIB:.1f} GiB is far from the formula's {est / GIB:.1f} GiB"
    return None


def rules_label(f: dict) -> tuple[str | None, str]:
    """(label or None = ask the model, reason) for the facts of one reading (``facts_*``)."""
    kind = f["kind"]
    why = benchmark_broken(f) if kind == "benchmark" else calibration_broken(f)
    if why:
        return "broken", why
    st = robust_z(f["value"], f.get("priors") or [], kind)
    contention = f.get("contention") or []
    z = st["z"]
    if z is not None:
        if abs(z) <= Z_OK:
            return "plausible", f"within {Z_OK:g} MAD of {st['n']} earlier readings (z {z:+.1f})"
        worse = z < 0 if kind == "benchmark" else True
        if worse and contention:
            return "contended", f"z {z:+.1f} against {st['n']} earlier readings while " + "; ".join(contention)
        if kind == "benchmark" and z < 0 and f.get("first_after_start"):
            return "cold_cache", f"first benchmark after the deployment started, z {z:+.1f}"
        return None, f"z {z:+.1f} against {st['n']} earlier readings, no host contention seen"
    if not contention and not f.get("first_after_start"):
        return "plausible", "no earlier readings to compare with and no host contention seen"
    return None, "no earlier readings, " + ("; ".join(contention) if contention else "first run after start")


def _n(v, unit="", digits=1) -> str:
    return "unknown" if v is None else f"{v:,.{digits}f}{unit}"


def build_state(f: dict) -> str:
    st = robust_z(f["value"], f.get("priors") or [], f["kind"])
    if f["kind"] == "benchmark":
        what = (f"Benchmark level: concurrency {f.get('concurrency')}, {f.get('input_len')} prompt / "
                f"{f.get('output_len')} output tokens, engine {f.get('engine')}.")
        val = (f"- This reading: {_n(f.get('tps_total'))} tokens/s in total, {_n(f.get('tps_per_req'))} tokens/s "
               f"per request, time to first token p50 {_n(f.get('ttft_p50'), ' s', 3)}, {f.get('requests')} "
               f"requests succeeded, {f.get('errors') or 0} failed.")
        unit, scale = " tokens/s", 1.0
    else:
        what = f"Memory measurement of a deployment's startup (engine {f.get('engine')})."
        val = (f"- This reading: {_n((f.get('value') or 0) / GIB, ' GiB', 2)} observed (engine log + host), formula "
               f"estimate {_n((f.get('estimated') or 0) / GIB, ' GiB', 2)}, MemAvailable dropped by "
               f"{_n(f['delta'] / GIB, ' GiB', 2) if f.get('delta') is not None else 'unknown'}.")
        unit, scale = " GiB", 1 / GIB
    prior = ("- Earlier readings of the same measurement: none." if not st["n"] else
             f"- Earlier readings of the same measurement: {st['n']}, median {_n(st['median'] * scale, unit, 2)}"
             + (f", MAD {_n(st['mad'] * scale, unit, 2)}, robust z of this reading {st['z']:+.1f}."
                if st["z"] is not None else "."))
    cont = f.get("contention") or []
    rel = (f"- Compared with the earlier median: {f['value'] / st['median'] * 100:.0f} %."
           if st["median"] else "- Compared with the earlier median: no earlier readings.")
    if f["kind"] == "benchmark":
        when = ("- When: the first benchmark since the deployment started (caches may still be cold)."
                if f.get("first_after_start") else
                "- When: not the first benchmark since the deployment started (earlier ones already ran, so its "
                "caches were warm).")
    else:
        when = "- When: during the deployment's startup."
    lines = [what, val, prior, rel, when,
             "- Host contention seen by DGXMM during the reading: " + ("; ".join(cont) if cont else
                                                                     "none (other GPU work is not visible here)") + "."]
    return "\n".join([quoted_block("MODEL", f.get("model_ref", ""), max_chars=300), *lines])


def eval_case(case: dict):
    """Evaluation adapter (tools/eval_feature.py): a case holds the reading's facts (as ``facts_*`` returns)."""
    label, _ = rules_label(case)
    return SimpleNamespace(rules=label, state=build_state(case), question=QUESTION, options=OPTIONS)


# ---------------- gathering facts from the database ----------------
def _host_contention(s, start, end) -> list[str]:
    out = []
    rows = list(s.scalars(select(MetricRollup).where(MetricRollup.deployment_id.is_(None),
                                                     MetricRollup.minute >= start - timedelta(minutes=1),
                                                     MetricRollup.minute <= end + timedelta(minutes=1))))
    ext = [r.data.get("external") for r in rows if isinstance((r.data or {}).get("external"), (int, float))]
    if len(ext) >= 2 and max(ext) - min(ext) >= CONTENTION_BYTES:
        out.append(f"other programs' memory moved by {(max(ext) - min(ext)) / GIB:.1f} GiB")
    return out


def _other_state_changes(s, dep_id, start, end) -> list[str]:
    q = select(DeploymentEvent).where(DeploymentEvent.ts >= start, DeploymentEvent.ts <= end)
    if dep_id is not None:
        q = q.where(DeploymentEvent.deployment_id != dep_id)
    n = sum(1 for e in s.scalars(q) if e.from_state != e.to_state)
    return [f"{n} state change{'s' if n != 1 else ''} of other deployments"] if n else []


def facts_benchmark(s, b: Benchmark) -> dict:
    prior_rows = list(s.scalars(select(Benchmark).where(
        Benchmark.id < b.id, Benchmark.model_ref == b.model_ref, Benchmark.engine == b.engine,
        Benchmark.config_hash == b.config_hash, Benchmark.concurrency == b.concurrency,
        Benchmark.input_len == b.input_len, Benchmark.output_len == b.output_len, Benchmark.requests > 0,
        Benchmark.tps_total > 0).order_by(Benchmark.id.desc()).limit(PRIORS * 2)))
    priors = [p.tps_total for p in prior_rows if p.quality != "disputed"][:PRIORS]
    end = as_aware(b.ts)
    prev = s.scalar(select(Benchmark).where(Benchmark.id < b.id, Benchmark.deployment_id == b.deployment_id)
                    .order_by(Benchmark.id.desc()).limit(1)) if b.deployment_id is not None else None
    run = s.scalar(select(BenchmarkRun).where(BenchmarkRun.deployment_id == b.deployment_id,
                                              BenchmarkRun.started_at <= b.ts)
                   .order_by(BenchmarkRun.id.desc()).limit(1)) if b.deployment_id is not None else None
    start = as_aware(run.started_at) if run is not None else end - timedelta(minutes=2)
    if prev is not None and as_aware(prev.ts) > start:
        start = as_aware(prev.ts)  # this level ran after the previous one
    first = False
    if b.deployment_id is not None:
        ready = s.scalar(select(DeploymentEvent.ts).where(DeploymentEvent.deployment_id == b.deployment_id,
                                                          DeploymentEvent.to_state == "ready",
                                                          DeploymentEvent.ts <= b.ts)
                         .order_by(DeploymentEvent.ts.desc()).limit(1))
        if ready is not None:
            first = not s.scalar(select(Benchmark.id).where(Benchmark.deployment_id == b.deployment_id,
                                                            Benchmark.id < b.id, Benchmark.ts >= ready).limit(1))
    contention = _other_state_changes(s, b.deployment_id, start, end) + _host_contention(s, start, end)
    return {"kind": "benchmark", "id": b.id, "deployment_id": b.deployment_id, "model_ref": b.model_ref,
            "engine": b.engine, "concurrency": b.concurrency, "input_len": b.input_len, "output_len": b.output_len,
            "requests": b.requests, "errors": b.errors, "tps_total": b.tps_total, "tps_per_req": b.tps_per_req,
            "ttft_p50": b.ttft_p50, "value": b.tps_total, "priors": priors,
            "prior_median": statistics.median(priors) if priors else None, "contention": contention,
            "first_after_start": first}


def _observed(r: MemoryCalibration) -> float:
    d = r.detail or {}
    return float(d.get("observed_bytes") or r.measured_bytes or 0)


def facts_calibration(s, r: MemoryCalibration) -> dict:
    prior_rows = list(s.scalars(select(MemoryCalibration).where(
        MemoryCalibration.id < r.id, MemoryCalibration.engine == r.engine, MemoryCalibration.repo == r.repo,
        MemoryCalibration.config_hash == r.config_hash, MemoryCalibration.method == "engine_log",
        MemoryCalibration.oom.is_(False)).order_by(MemoryCalibration.id.desc()).limit(PRIORS * 2)))
    priors = [_observed(p) for p in prior_rows if p.quality != "disputed" and _observed(p) > 0][:PRIORS]
    d = r.detail or {}
    delta_q = (r.reading or {}).get("delta_quality", r.quality)
    contention = []
    if d.get("noisy"):
        contention.append("another deployment changed state during the startup")
    if delta_q == "disputed_high" and d.get("memavailable_delta") is not None:
        extra = d["memavailable_delta"] - d.get("observed_bytes", 0)
        contention.append(f"MemAvailable dropped {extra / GIB:.1f} GiB more than the engine accounts for")
    # (no host-rollup signal here: during a startup the starting deployment's own CUDA memory reads as external)
    return {"kind": "calibration", "id": r.id, "model_ref": r.repo, "engine": r.engine, "value": _observed(r),
            "estimated": r.estimated_bytes, "delta": d.get("memavailable_delta"), "complete": d.get("complete"),
            "priors": priors, "contention": contention, "delta_quality": delta_q}


# ---------------- the service / job ----------------
class ReadingService:
    def __init__(self, ctx, clock=utcnow):
        self.ctx, self.clock = ctx, clock
        self._abstained: set[str] = set()  # "kind:id" the model was asked about and did not answer confidently

    def notify(self, *_ids) -> None:
        """A new reading was stored (benchmark level or calibration row): label it soon. Never raises."""
        eng = getattr(self.ctx, "insight_engine", None)
        try:
            if eng is not None:
                eng.trigger(JOB)
        except Exception:
            log.debug("reading labels: trigger failed", exc_info=True)

    def pending(self) -> list[tuple[str, dict]]:
        """Facts of the unlabelled readings of the last LOOKBACK (oldest first, at most MAX_PER_RUN)."""
        since = self.clock() - LOOKBACK
        out = []
        with self.ctx.db.session() as s:
            # declined readings are excluded in the query, before the limit, so they never starve newer ones
            skip_b = [int(k.split(":")[1]) for k in self._abstained if k.startswith("benchmark:")]
            skip_c = [int(k.split(":")[1]) for k in self._abstained if k.startswith("calibration:")]
            for b in s.scalars(select(Benchmark).where(Benchmark.reading.is_(None), Benchmark.ts >= since,
                                                       Benchmark.id.not_in(skip_b))
                               .order_by(Benchmark.id).limit(MAX_PER_RUN)):
                out.append(("benchmark", facts_benchmark(s, b)))
            for r in s.scalars(select(MemoryCalibration).where(
                    MemoryCalibration.reading.is_(None), MemoryCalibration.ts >= since,
                    MemoryCalibration.method == "engine_log", MemoryCalibration.oom.is_(False),
                    MemoryCalibration.id.not_in(skip_c))
                    .order_by(MemoryCalibration.id).limit(MAX_PER_RUN)):
                out.append(("calibration", facts_calibration(s, r)))
        return out[:MAX_PER_RUN]

    def store(self, f: dict, label: str, source: str, confidence, reason: str, decision_id) -> None:
        reading = {"label": label, "source": source, "confidence": confidence, "reason": reason[:300],
                   "decision_id": decision_id, "labelled_at": self.clock().isoformat()}
        with self.ctx.db.session() as s:
            if f["kind"] == "benchmark":
                row = s.get(Benchmark, f["id"])
            else:
                row = s.get(MemoryCalibration, f["id"])
                reading["delta_quality"] = f.get("delta_quality")
            if row is None:
                return
            row.reading, row.quality = reading, quality_of(label)
        if label == "broken":
            self._flag(f, reason)

    def _flag(self, f: dict, reason: str) -> None:
        if f["kind"] == "benchmark":
            subject = deployment_subject(f["deployment_id"]) if f.get("deployment_id") else "benchmark"
            title = f"Broken benchmark reading at concurrency {f.get('concurrency')}"
            body = (f"The benchmark level at concurrency {f.get('concurrency')} did not produce a real measurement "
                    f"({reason}). It is kept on the Benchmarks tab, marked broken, and not used for advice. "
                    "Re-run the benchmark when the deployment is idle.")
        else:
            subject = "calibration"
            title = f"Broken memory measurement ({f.get('engine')})"
            body = (f"A startup memory measurement was not usable ({reason}). It is ignored by calibration; the "
                    "estimate stays at the formula and earlier measurements.")
        action = {"type": "open", "path": f"/deployments/{int(f['deployment_id'])}", "label": "Open deployment"} \
            if f["kind"] == "benchmark" and f.get("deployment_id") else None
        self.ctx.insights.upsert("reading", "reading", subject[:256], "warning", title[:256], body[:4000],
                                 dedupe_key=f"reading:{f['kind']}:{f['id']}", source="rules", action=action,
                                 expires_at=self.clock() + PROPOSAL_TTL)

    async def run(self, run) -> str:
        items = await run.to_thread(self.pending)
        done = asked = 0
        for kind, f in items:
            label, reason = rules_label(f)
            subject = f"{kind}:{f['id']}"
            if label is not None:
                d = await run.to_thread(safe_record_rules, run.decider, feature="reading", subject=subject,
                                        choice=label, reason=reason, options=OPTIONS)
                await run.to_thread(self.store, f, label, "rules", None, reason, d.decision_id)
                done += 1
                continue
            asked += 1
            d = await safe_decide(run.decider, feature="reading", subject=subject, state=build_state(f),
                                  question=QUESTION, options=OPTIONS, timeout_s=TIMEOUT_S)
            if d.source == "model" and d.choice:
                await run.to_thread(self.store, f, d.choice, "model", d.confidence,
                                    f"decision model ({reason})", d.decision_id)
                done += 1
            elif d.probabilities:  # answered below the threshold: not asked again (the reading stays unlabelled)
                self._abstained.add(subject)
        return f"{done} of {plural(len(items), 'reading')} labelled ({asked} asked the model)"


def label_out(reading: dict | None) -> dict:
    """The API view of a reading label (benchmark rows)."""
    r = reading or {}
    return {"label": r.get("label"), "label_source": r.get("source"), "label_confidence": r.get("confidence"),
            "label_reason": r.get("reason")}
