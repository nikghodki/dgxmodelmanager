"""Advanced feature 2 — deploy-time config advice: which sizing suits the deployment's goal?

Rules generate the candidates, and only from DGXMM's memory calculator (``Deployer.prepare`` -> the engine's
estimate, with calibration): every combination of context length (native, 128k, 64k, 32k, 16k, 8k — at most the
native one), KV-cache dtype (auto / fp8; llama.cpp f16 / q8_0), concurrency (8, 32, 128, 256; llama.cpp slots
1, 4, 8, 16) and prefix caching (vLLM), each with a KV pool of at least PER_SEQ_TOKENS per sequence. Only the ones
that fit the memory free right now are kept, then a Pareto cut (context, concurrency, memory, KV precision, prefix
caching) leaves at most MAX_OPTIONS, each described by its trade-off. Nothing that does not fit is ever offered.

The decision model ranks the candidates for the goal (chat, agent_tool_use, batch_throughput, long_context). At or
above the threshold the wizard shows a "Suggested config" card that fills untouched fields; below it (or with the
decider disabled) the top three are shown with their trade-offs (model order, else a deterministic goal order),
nothing is recommended.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

from dgxmm.config import GIB
from dgxmm.decider.common import quoted_block, safe_decide, threshold_of
from dgxmm.decider.compat import rules_on
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
FEATURE = "config"
GOALS: dict[str, str] = {
    "chat": "interactive chat: a few users at a time, multi-turn conversations, quick first tokens",
    "agent_tool_use": "agents calling tools: long system prompts with tool schemas, repeated prefixes, "
                      "multi-step conversations",
    "batch_throughput": "offline batch jobs: as many requests in parallel as possible, total tokens per second "
                        "matters more than latency",
    "long_context": "long documents or repositories in one request: the longest context matters most",
}
CONTEXTS = (131072, 65536, 32768, 16384, 8192)
VLLM_SEQS = (8, 32, 128, 256)
LLAMACPP_SLOTS = (1, 4, 8, 16)
PER_SEQ_TOKENS = 4096  # KV pool per concurrent sequence at least (vLLM shares one pool across sequences)
MAX_OPTIONS = 8
TOP = 3
TIMEOUT_S = 4.0
NEVER = 1.01
QUESTION = "Which of these serving configurations suits the deployment's goal best?"


@dataclass
class Candidate:
    key: str
    config: dict           # the engine settings this candidate sets (the wizard fills these keys)
    context: int           # tokens per request
    seqs: int              # concurrent sequences (llama.cpp: slots)
    kv: str                # KV cache dtype
    prefix: bool           # prefix caching (vLLM)
    pool_tokens: int       # KV pool (all sequences)
    required: int          # bytes, from the memory calculator
    headroom: int          # bytes left after it
    extra: dict = field(default_factory=dict)

    @property
    def quality(self) -> int:
        return 1 if self.kv in ("auto", "f16") else 0

    @property
    def summary(self) -> str:
        bits = [f"{_k(self.context)} context", f"{self.seqs} concurrent", f"{self.kv} KV"]
        if self.prefix:
            bits.append("prefix caching")
        return (" · ".join(bits) + f" — needs {self.required / GIB:.1f} GiB, leaves {self.headroom / GIB:.1f} GiB "
                f"free")

    def as_dict(self) -> dict:
        return {"key": self.key, "config": dict(self.config), "summary": self.summary,
                "required_bytes": self.required, "headroom_bytes": self.headroom, "context": self.context,
                "seqs": self.seqs, "kv_cache_dtype": self.kv, "prefix_caching": self.prefix,
                "kv_pool_tokens": self.pool_tokens}


def _k(n: int) -> str:
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def context_choices(native: int) -> list[int]:
    return [native] + [c for c in CONTEXTS if c < native]


def pool_tokens(context: int, seqs: int) -> int:
    return min(context * seqs, max(2 * context, seqs * min(context, PER_SEQ_TOKENS)))


def grid(engine: str, native: int) -> list[tuple[str, dict, int, int, str, bool, int]]:
    """(key, config, context, seqs, kv, prefix, pool) of every combination before the memory check."""
    out = []
    for ctx in context_choices(native):
        if engine == "vllm":
            for kv in ("auto", "fp8"):
                for seqs in VLLM_SEQS:
                    for prefix in (True, False):
                        pool = pool_tokens(ctx, seqs)
                        cfg = {"max_model_len": ctx, "max_num_seqs": seqs, "kv_cache_dtype": kv,
                               "enable_prefix_caching": prefix, "kv_tokens": pool}
                        key = f"ctx{_k(ctx)}-seq{seqs}-{kv}{'-pc' if prefix else ''}"
                        out.append((key, cfg, ctx, seqs, kv, prefix, pool))
        else:
            for kv in ("f16", "q8_0"):
                for slots in LLAMACPP_SLOTS:
                    if ctx * slots > 1_048_576:
                        continue
                    cfg = {"ctx_size": ctx * slots, "parallel": slots, "cache_type_k": kv, "cache_type_v": kv}
                    out.append((f"ctx{_k(ctx)}-slots{slots}-{kv}", cfg, ctx, slots, kv, False, ctx * slots))
    return out


def candidates(engine: str, native: int, free: int, required_of) -> list[Candidate]:
    """The grid, sized by `required_of(config) -> bytes | None` (the memory calculator), keeping what fits."""
    out = []
    for key, cfg, ctx, seqs, kv, prefix, pool in grid(engine, native):
        req = required_of(cfg)
        if req is None or req > free:
            continue
        out.append(Candidate(key, cfg, ctx, seqs, kv, prefix, pool, req, free - req))
    return out


def _dominates(a: Candidate, b: Candidate) -> bool:
    ge = (a.context >= b.context and a.seqs >= b.seqs and a.required <= b.required and a.quality >= b.quality
          and a.prefix >= b.prefix)
    gt = (a.context > b.context or a.seqs > b.seqs or a.required < b.required or a.quality > b.quality
          or a.prefix > b.prefix)
    return ge and gt


def pareto(cands: list[Candidate]) -> list[Candidate]:
    return [c for c in cands if not any(_dominates(o, c) for o in cands if o is not c)]


def cut(front: list[Candidate], n: int = MAX_OPTIONS) -> list[Candidate]:
    """At most `n` spread-out options: the longest context, the most concurrency, the smallest footprint, then
    round-robin over the context lengths (most concurrency first, full-precision KV before fp8)."""
    if len(front) <= n:
        return sorted(front, key=lambda c: (-c.context, -c.seqs, -c.quality))
    chosen: list[Candidate] = []

    def take(c: Candidate | None) -> None:
        if c is not None and c not in chosen and len(chosen) < n:
            chosen.append(c)
    take(max(front, key=lambda c: (c.context, c.seqs, c.quality, -c.required)))
    take(max(front, key=lambda c: (c.seqs, c.context, c.quality, -c.required)))
    take(min(front, key=lambda c: (c.required, -c.context, -c.seqs)))
    by_ctx: dict[int, list[Candidate]] = {}
    for c in sorted(front, key=lambda c: (-c.seqs, -c.quality, c.required)):
        by_ctx.setdefault(c.context, []).append(c)
    while len(chosen) < n and any(by_ctx.values()):
        for ctx in sorted(by_ctx, reverse=True):
            rest = [c for c in by_ctx[ctx] if c not in chosen]
            by_ctx[ctx] = rest
            if rest:
                take(rest[0])
    return sorted(chosen, key=lambda c: (-c.context, -c.seqs, -c.quality))


def goal_order(cands: list[Candidate], goal: str) -> list[Candidate]:
    """Deterministic order for a goal (shown when the model does not answer; never a recommendation)."""
    def lg(x: int) -> float:
        return math.log2(max(x, 1))
    if goal == "long_context":
        key = lambda c: (-c.context, -c.quality, -c.seqs)  # noqa: E731
    elif goal == "batch_throughput":
        key = lambda c: (-c.seqs, -min(c.context, 16384), -c.quality, -c.headroom)  # noqa: E731
    elif goal == "agent_tool_use":
        key = lambda c: (c.context < 32768, not c.prefix, -c.quality, abs(lg(c.context) - 16), -c.seqs)  # noqa: E731
    else:  # chat
        key = lambda c: (not 16384 <= c.context <= 65536, -c.quality, abs(lg(c.seqs) - 5), -c.context)  # noqa: E731
    return sorted(cands, key=key)


def build_state(*, goal: str, ref: str, engine: str, weights: int, native: int, free: int, arch: str | None,
                is_moe: bool | None) -> str:
    return "\n".join([
        f"Goal: {goal} — {GOALS[goal]}",
        f"Engine: {engine}; weights {weights / GIB:.1f} GiB; native context {native} tokens; "
        f"{'mixture-of-experts' if is_moe else 'dense'} model; memory free for it now {free / GIB:.1f} GiB",
        "Each option fits in the free memory; more context or concurrency needs more KV-cache memory, fp8 KV halves "
        "it at a small quality cost, prefix caching reuses repeated prompt prefixes.",
        quoted_block("MODEL NAME", ref, max_chars=300),
        quoted_block("ARCHITECTURE", arch or "unknown", max_chars=200),
    ])


def _empty(goal: str, **kw) -> dict:
    return {"goal": goal, "options": [], "recommended": None, "top": [], "source": "none", "confidence": None,
            "decision_id": None, "free_bytes": None, "reason": None, "disabled": False, **kw}


def _required_of(deployer, engine: str, ref: str):
    def fn(cfg: dict) -> int | None:
        try:
            return deployer.prepare(engine, ref, cfg).estimate.total
        except Exception:  # DeployError / ValueError: this combination cannot be estimated
            return None
    return fn


def _native(artifact) -> int:
    a = artifact.arch
    return int(a.max_context) if a is not None and a.max_context else 32768


async def advise(ctx, model_ref: str, engine: str, goal: str) -> dict:
    """``{goal, options: [{key, config, summary, required_bytes, ...}], recommended, top, source, confidence,
    decision_id, free_bytes, reason, disabled}``. Raises KeyError for an unknown model (the route's 404)."""
    decider = decider_of(ctx)
    if not rules_on(decider, FEATURE):
        return _empty(goal, disabled=True)
    artifact = ctx.inventory.get(model_ref)
    if artifact is None:
        raise KeyError(model_ref)
    budget = ctx.budget.current() or await ctx.budget.refresh()
    free = int(budget.free_for_new)
    native = _native(artifact)
    cands = await asyncio.to_thread(candidates, engine, native, free, _required_of(ctx.deployer, engine, model_ref))
    opts = goal_order(cut(pareto(cands)), goal)
    if not opts:
        return _empty(goal, free_bytes=free, reason="nothing_fits")
    out = _empty(goal, free_bytes=free, options=[c.as_dict() for c in opts])
    state = build_state(goal=goal, ref=model_ref, engine=engine, weights=artifact.weights_bytes, native=native,
                        free=free, arch=artifact.architecture, is_moe=artifact.arch.is_moe if artifact.arch else None)
    options = [Option(c.key, c.key, c.summary) for c in opts]
    d = await safe_decide(decider, feature=FEATURE, subject=f"model:{model_ref}"[:256], state=state, question=QUESTION,
                          options=options, threshold=threshold_of(decider, FEATURE, NEVER), timeout_s=TIMEOUT_S)
    out.update(confidence=d.confidence, decision_id=d.decision_id)
    if d.source == "model" and d.choice is not None:
        out.update(recommended=d.choice, source="model")
    if d.probabilities:
        out["top"] = [k for k, _ in sorted(d.probabilities.items(), key=lambda kv: -kv[1])][:TOP]
    else:
        out["top"] = [c.key for c in opts[:TOP]]
    return out


# ---------------- evaluation adapter ----------------
def eval_options(case: dict) -> tuple[int, list[Candidate]]:
    """The options DGXMM offers for a corpus case (config.json, weights, free memory), sized by the same estimator
    as the live path (without calibration)."""
    from dgxmm.config import Settings
    from dgxmm.inventory.archspec import parse_hf_config
    from dgxmm.memory.estimate import VllmSizing, estimate_vllm
    settings = Settings()
    spec = parse_hf_config(case["config"])

    def required(cfg: dict) -> int:
        return estimate_vllm(spec, int(case["weights_bytes"]),
                             VllmSizing(cfg["max_model_len"], cfg["max_num_seqs"], 8192, cfg["kv_cache_dtype"],
                                        cfg["kv_tokens"]), settings).total
    native = int(spec.max_context or 32768)
    return native, goal_order(cut(pareto(candidates("vllm", native, int(case["free_bytes"]), required))),
                              case["goal"])


def eval_case(case: dict):
    native, opts = eval_options(case)
    spec_moe = bool(case["config"].get("num_experts") or case["config"].get("num_local_experts"))
    state = build_state(goal=case["goal"], ref=case.get("ref") or case["name"], engine="vllm",
                        weights=int(case["weights_bytes"]), native=native,
                        free=int(case["free_bytes"]), arch=(case["config"].get("architectures") or [None])[0],
                        is_moe=spec_moe)
    return SimpleNamespace(rules=None, state=state, question=QUESTION,
                           options=[Option(c.key, c.key, c.summary) for c in opts])
