"""Feature 5 — workload-aware advice.

For a ready deployment with at least MIN_MINUTES of 1-minute rollups in the last hour, summarise the traffic
numerically; rules label clear cases (idle, batch throughput), the decision model labels the rest. Cached 15 min
per deployment. The advisor's rules still produce the advice set; the workload only orders it (`relevance`).
"""
from __future__ import annotations

import asyncio
import logging
import time

from dgxmm.decider.common import busy_minute, last_span, quoted_block, safe_decide, safe_record_rules
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
WINDOW_H = 1.0
MIN_MINUTES = 30
CACHE_TTL_S = 15 * 60
RETRY_TTL_S = 60  # decider unavailable: try again sooner
# Option descriptions: what each profile looks like in the engine metrics, as measured on real traffic of a test
# deployment (tests/fixtures/workload, docs/decider-evaluation.md).
LABELS: list[tuple[str, str]] = [
    ("single_user_chat", "One person chatting: well under one request in flight on average, a few requests per "
                         "minute with pauses between turns, prompts of hundreds of tokens, answers of hundreds of "
                         "tokens, no queue"),
    ("agent_tool_use", "An agent / tool-calling loop: one or two requests in flight almost all the time, many "
                       "requests per minute back to back, prompts of thousands of tokens that repeat earlier "
                       "prefixes (prefix-cache hit rate above 80 %), prompt tokens several times the output tokens"),
    ("batch_throughput", "Batch or many concurrent clients: the engine is full (running near its sequence limit) "
                         "and requests wait in the queue, throughput matters more than latency"),
    ("long_context_rag", "Long-context / RAG: one or two clients sending very long, mostly unique prompts (many "
                         "thousands of tokens each, low prefix-cache reuse) with short answers; prompt tokens far "
                         "exceed output tokens and time to first token is high"),
    ("idle", "Almost no traffic"),
    ("mixed", "Several of these patterns alternate in the window with no dominant one (e.g. chat plus batch bursts "
              "and occasional long-document queries)"),
]
OPTIONS = [Option(k, k.replace("_", " "), d) for k, d in LABELS]
QUESTION = "Which workload best describes this deployment's traffic over the last hour?"

# Advice ids each workload makes more (+) or less (-) relevant. Warnings always stay first.
BOOST: dict[str, tuple[set[str], set[str]]] = {
    "idle": ({"shrink_reservation"}, {"raise_seqs", "moe_batch", "spec_decode", "batched_tokens_small"}),
    "batch_throughput": ({"raise_seqs", "moe_batch", "kv_pressure", "batched_tokens_small", "nvfp4_variant"},
                         {"spec_decode", "try_llamacpp"}),
    "long_context_rag": ({"fp8_kv", "kv_pressure", "llamacpp_quant_kv", "chunked_prefill_off", "batched_tokens_small"},
                         {"prefix_useless"}),
    "agent_tool_use": ({"spec_decode", "spec_accept_low", "nvfp4_variant", "batched_tokens_small"},
                       {"prefix_useless", "try_llamacpp"}),
    "single_user_chat": ({"spec_decode", "try_llamacpp", "nvfp4_variant", "spec_accept_low"},
                         {"raise_seqs", "moe_batch"}),
    "mixed": (set(), set()),
}
SEVERITY_BASE = {"warning": 3.0, "suggestion": 2.0, "info": 1.0}


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _max(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _per_minute(samples: list[dict]) -> list[dict]:
    """5-second live samples averaged into minute rows (the unit of the rollups); waiting keeps its peak."""
    groups: dict[int, list[dict]] = {}
    for smp in samples:
        groups.setdefault(int((smp.get("ts") or 0) // 60) * 60, []).append(smp)
    out = []
    for minute, grp in sorted(groups.items()):
        keys = {k for smp in grp for k in smp} - {"ts"}
        row = {"ts": minute, **{k: _mean(smp.get(k) for smp in grp) for k in keys}}
        row["waiting_max"] = _max([smp.get("waiting_max", smp.get("waiting")) for smp in grp])
        out.append(row)
    return out


def summarize(rows: list[dict], live: list[dict] | None = None) -> dict:
    """Numeric summary of minute rollups (plus live samples newer than the last rollup)."""
    rows = list(rows)
    if live:
        last = rows[-1]["ts"] + 60 if rows else 0
        rows += _per_minute([s for s in live if (s.get("ts") or 0) >= last])
    busy = [r for r in rows if busy_minute(r)]
    n = len(rows)
    prompt, gen = _mean(r.get("prompt_tps") for r in busy), _mean(r.get("gen_tps") for r in busy)
    out = {
        "minutes": n,
        "idle_fraction": round(1 - len(busy) / n, 4) if n else None,
        "mean_running": _mean(r.get("running") for r in busy),
        "peak_running": _max(r.get("running") for r in rows),
        "mean_waiting": _mean(r.get("waiting") for r in busy),
        "peak_waiting": _max([r.get("waiting_max", r.get("waiting")) for r in rows]),
        "queued_fraction": round(sum(1 for r in busy if (r.get("waiting") or 0) > 0) / len(busy), 4) if busy else None,
        "prompt_tps": prompt,
        "gen_tps": gen,
        "prompt_gen_ratio": round(prompt / gen, 2) if prompt is not None and gen else None,
        "prefix_hit_rate": _mean(r.get("prefix_hit_rate") for r in busy),
        "spec_accept_rate": _mean(r.get("spec_accept_rate") for r in busy),
        "kv_usage_mean": _mean(r.get("kv_usage") for r in busy),
        "kv_usage_peak": _max(r.get("kv_usage") for r in rows),
        "ttft_p50": _mean(r.get("ttft_p50") for r in busy),
        "e2e_p50": _mean(r.get("e2e_p50") for r in busy),
        **_variation([r.get("running") for r in busy]),
    }
    out.update(per_request(out))
    return out


def _variation(vals: list) -> dict:
    """How much the per-minute concurrency moved: bursty / alternating traffic has a wide range and a high
    std/mean, steady traffic (one chat user, a saturated batch) a narrow one."""
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return {"running_min": None, "running_max": None, "running_cv": None}
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return {"running_min": round(min(v), 2), "running_max": round(max(v), 2),
            "running_cv": round(sd / m, 2) if m else None}


def per_request(s: dict) -> dict:
    """Request rate and tokens per request from Little's law (requests in the system = rate x latency): the
    engines export no per-request counters DGXMM keeps. On real traffic these estimates were within ~20 % of the
    client-side counts (docs/decider-evaluation.md)."""
    in_system = (s.get("mean_running") or 0) + (s.get("mean_waiting") or 0)
    e2e = s.get("e2e_p50")
    if not in_system or not e2e:
        return {"requests_per_min": None, "prompt_tokens_per_request": None, "output_tokens_per_request": None}
    rate = in_system / e2e  # requests per second while busy
    tok = lambda v: round(v / rate) if v is not None else None  # noqa: E731
    return {"requests_per_min": round(rate * 60, 1), "prompt_tokens_per_request": tok(s.get("prompt_tps")),
            "output_tokens_per_request": tok(s.get("gen_tps"))}


def rule_label(s: dict, cfg: dict | None = None) -> tuple[str, str] | None:
    if s.get("idle_fraction") is not None and s["idle_fraction"] >= 0.9:
        return "idle", f"no traffic in {s['idle_fraction']:.0%} of {last_span(s['minutes'], 'minute')}"
    mr, q = s.get("mean_running") or 0, s.get("queued_fraction") or 0
    if mr >= 16 or (mr >= 8 and q >= 0.5):
        return "batch_throughput", f"{mr:.0f} requests running on average, queued {q:.0%} of the busy time"
    # Saturation against the engine's own limit: real batch traffic on max_num_seqs 8 averaged 7.96 running with a
    # queue every minute (tests/fixtures/workload). Engines with fewer than 4 slots are left to the model: two chat
    # users or an agent plus a user already fill 1-3 slots and queue without being batch traffic.
    slots = (cfg or {}).get("max_num_seqs") or (cfg or {}).get("parallel")
    if isinstance(slots, int) and slots >= 4 and q >= 0.5 and mr >= 0.9 * slots:
        return "batch_throughput", (f"engine saturated: {mr:.1f} of {slots} sequences busy on average, requests "
                                    f"queued {q:.0%} of the busy time")
    return None


def _n(v, digits: int = 1, pct: bool = False) -> str:
    if v is None:
        return "unknown"
    return f"{v * 100:.0f} %" if pct else (f"{v:,.{digits}f}" if digits else f"{v:,.0f}")


def build_state(summary: dict, engine: str, model_ref: str, cfg: dict) -> str:
    """The traffic summary in words (measured: the model labels real traffic more accurately from sentences
    with per-request sizes than from a bare key/value dump; docs/decider-evaluation.md)."""
    s = summary
    flags = {k: cfg.get(k) for k in ("max_model_len", "max_num_seqs", "ctx_size", "parallel", "tool_call_parser",
                                     "reasoning_parser", "enable_prefix_caching") if k in cfg}
    flags["speculative"] = bool(cfg.get("speculative"))
    busy = None if s.get("idle_fraction") is None else 1 - s["idle_fraction"]
    lines = [
        f"- Activity: requests in flight during {_n(busy, pct=True)} of the {s.get('minutes')} minutes.",
        f"- Concurrency while busy: {_n(s.get('mean_running'), 2)} requests running on average (peak "
        f"{_n(s.get('peak_running'), 0)}); queue {_n(s.get('mean_waiting'), 1)} waiting on average, non-empty in "
        f"{_n(s.get('queued_fraction'), pct=True)} of busy minutes (peak {_n(s.get('peak_waiting'), 0)}).",
        f"- Minute-to-minute variation: running ranged from {_n(s.get('running_min'), 2)} to "
        f"{_n(s.get('running_max'), 2)} requests (std/mean {_n(s.get('running_cv'), 2)}).",
        f"- Request rate (estimated): about {_n(s.get('requests_per_min'))} requests per minute.",
        f"- Size per request (estimated): about {_n(s.get('prompt_tokens_per_request'), 0)} prompt tokens in and "
        f"{_n(s.get('output_tokens_per_request'), 0)} tokens out (prompt/output token ratio "
        f"{_n(s.get('prompt_gen_ratio'), 2)}).",
        f"- Throughput while busy: {_n(s.get('prompt_tps'))} prompt tokens/s, {_n(s.get('gen_tps'))} generated "
        f"tokens/s.",
        f"- Prefix-cache hit rate: {_n(s.get('prefix_hit_rate'), pct=True)} (share of prompt tokens repeating an "
        f"earlier prompt's prefix).",
        f"- Latency while busy: time to first token p50 {_n(s.get('ttft_p50'), 2)} s; whole request p50 "
        f"{_n(s.get('e2e_p50'), 1)} s.",
        f"- KV cache usage: mean {_n(s.get('kv_usage_mean'), pct=True)}, peak {_n(s.get('kv_usage_peak'), pct=True)}.",
    ]
    if s.get("spec_accept_rate") is not None:
        lines.append(f"- Speculative decoding: {_n(s['spec_accept_rate'], pct=True)} of draft tokens accepted.")
    return "\n".join([f"Engine: {engine}", quoted_block("MODEL", model_ref, max_chars=300), f"Config: {flags}",
                      f"Traffic over the last {s.get('minutes')} minutes (from the engine's metrics, per-minute "
                      f"averages):", *lines])


def rank(advice: list[dict], workload: dict | None) -> list[dict]:
    """Add `relevance` (severity base 3/2/1 plus or minus the workload adjustment) and `workload_fit` ("boost",
    "demote" or None: whether the workload label favours or disfavours the item) to each advice item and order by
    relevance (stable). Without a workload label the order is the advisor's own (severity)."""
    label = (workload or {}).get("label")
    conf = (workload or {}).get("confidence") or (1.0 if label else 0.0)
    up, down = BOOST.get(label, (set(), set())) if label else (set(), set())
    out = []
    for i, a in enumerate(advice):
        r = SEVERITY_BASE.get(a.get("severity"), 1.0)
        fit = None
        if a.get("id") in up:
            r += 0.9 * conf
            fit = "boost"
        elif a.get("id") in down:
            r -= 0.5 * conf
            fit = "demote"
        out.append({**a, "relevance": round(r, 4), "workload_fit": fit})
    return [a for _, a in sorted(enumerate(out), key=lambda ia: (-ia[1]["relevance"], ia[0]))]


class WorkloadService:
    def __init__(self, ctx, clock=time.monotonic):
        self.ctx, self.clock = ctx, clock
        self._cache: dict[int, tuple[float, dict | None]] = {}
        self._inflight: dict[int, asyncio.Task] = {}

    def _cached(self, dep_id: int) -> tuple[bool, dict | None]:
        hit = self._cache.get(dep_id)
        if hit and hit[0] > self.clock():
            return True, hit[1]
        return False, None

    def _start(self, dep_id: int) -> asyncio.Task:
        """Single flight: at most one computation per deployment at a time."""
        task = self._inflight.get(dep_id)
        if task is None or task.done():
            task = asyncio.get_running_loop().create_task(self._refresh(dep_id))
            self._inflight[dep_id] = task
            task.add_done_callback(lambda t, d=dep_id: self._inflight.pop(d, None) if self._inflight.get(d) is t
                                   else None)
        return task

    def cached(self, dep_id: int) -> dict | None:
        """The cached workload, fresh or not; never computes (callers inside insight-engine jobs)."""
        hit = self._cache.get(dep_id)
        return hit[1] if hit else None

    def peek(self, dep_id: int) -> dict | None:
        """The cached workload (never waits). On a miss, schedules one background computation and returns
        None; callers (the advice endpoint) pick the label up on their next poll."""
        fresh, value = self._cached(dep_id)
        if not fresh:
            self._start(dep_id)
        return value if fresh else (self._cache.get(dep_id, (0, None))[1])

    async def get(self, dep_id: int) -> dict | None:
        """{label, source, confidence, summary} | None (not ready / not enough data). Waits for the (shared)
        computation, which is bounded by the decider timeout. Never raises."""
        fresh, value = self._cached(dep_id)
        if fresh:
            return value
        try:
            await asyncio.shield(self._start(dep_id))
        except Exception:
            pass
        return self._cache.get(dep_id, (0, None))[1]

    def forget(self, dep_id: int) -> None:
        """A removed deployment: drop its cached workload (its id may be reused by a new deployment)."""
        self._cache.pop(dep_id, None)
        task = self._inflight.pop(dep_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def drain(self) -> None:  # tests
        while self._inflight:
            await asyncio.gather(*list(self._inflight.values()), return_exceptions=True)
            await asyncio.sleep(0)  # let done-callbacks remove finished tasks

    async def _refresh(self, dep_id: int) -> None:
        try:
            result, ttl = await self._compute(dep_id)
        except Exception:
            log.exception("workload classification of deployment %s failed", dep_id)
            result, ttl = None, RETRY_TTL_S
        self._cache[dep_id] = (self.clock() + ttl, result)

    def _load(self, dep_id: int):
        v = self.ctx.deployer.view(dep_id)
        store = self.ctx.metrics
        if v["state"] != "ready" or store is None:
            return v, None
        rows = store.history(dep_id, WINDOW_H)
        if len(rows) < MIN_MINUTES:
            return v, None
        return v, summarize(rows, store.series(dep_id))

    async def _compute(self, dep_id: int) -> tuple[dict | None, float]:
        v, summary = await asyncio.to_thread(self._load, dep_id)
        if summary is None:
            return None, RETRY_TTL_S
        decider = decider_of(self.ctx)
        subject = f"deployment:{dep_id}"
        hit = rule_label(summary, v["config"] or {})
        if hit is not None:
            d = safe_record_rules(decider, feature="workload", subject=subject, choice=hit[0], reason=hit[1])
            return {"label": hit[0], "source": "rules", "confidence": 1.0, "summary": summary,
                    "reason": hit[1], "decision_id": d.decision_id}, CACHE_TTL_S
        d = await safe_decide(decider, feature="workload", subject=subject,
                              state=build_state(summary, v["engine"], v["model_ref"], v["config"] or {}),
                              question=QUESTION, options=OPTIONS, timeout_s=2.0)
        if d.source == "model" and d.choice:
            return {"label": d.choice, "source": "model", "confidence": d.confidence, "summary": summary,
                    "decision_id": d.decision_id}, CACHE_TTL_S
        ttl = CACHE_TTL_S if d.probabilities else RETRY_TTL_S
        return {"label": None, "source": "none", "confidence": d.confidence, "summary": summary,
                "decision_id": d.decision_id}, ttl
