"""Advanced feature 5 — engine choice: vLLM or llama.cpp for a model in the deploy wizard?

The formats on disk decide first. A model's siblings are the complete, deployable artifacts with the same base model
name (``planner.plans.base_model_name``: ``Qwen/Qwen3-8B`` and ``…/Qwen3-8B-Q4_K_M.gguf``). Rules:
- GGUF only -> llama.cpp; safetensors only -> vLLM;
- the architecture is not in the pinned vLLM image's list but a GGUF is on disk -> llama.cpp (with the GGUF's ref);
- only one format fits the free memory (``need_bytes``) -> that one; neither -> llama.cpp.
With both formats on disk and both fitting, the decision model chooses ``vllm`` / ``llamacpp`` from the goal, the sizes and
quantisations, the free memory and this box's benchmarks of the model. Below the threshold (or with the decider
disabled) nothing is suggested; the wizard keeps its default engine.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from sqlalchemy import func, select

from dgxmm.config import GIB
from dgxmm.decider.common import arecord_rules, quoted_block, safe_decide, threshold_of
from dgxmm.decider.compat import rules_on
from dgxmm.decider.config import GOALS
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option
from dgxmm.planner.plans import base_model_name

log = logging.getLogger(__name__)
FEATURE = "engine"
ENGINES = {"vllm": "vLLM: safetensors weights, high-concurrency batching (paged KV cache, CUDA graphs), sleep mode",
           "llamacpp": "llama.cpp: GGUF quantised weights, small footprint, few parallel slots"}
OPTIONS = [Option(k, k, d) for k, d in ENGINES.items()]
FORMAT_ENGINE = {"safetensors": "vllm", "gguf": "llamacpp"}
QUESTION = "Which engine should serve this model for the goal?"


def need_bytes(fmt: str, weights: int) -> int:
    """Rough memory a deployment of these weights needs with a modest KV cache (shown to the model as a fact; the
    wizard's memory check is the real calculator): vLLM weights + 25 % + 4 GiB, llama.cpp weights + 10 % + 2 GiB."""
    return int(weights * 1.25 + 4 * GIB) if fmt == "safetensors" else int(weights * 1.1 + 2 * GIB)
TIMEOUT_S = 4.0
NEVER = 1.01


def siblings(artifact, artifacts) -> dict[str, object]:
    """format -> the artifact to deploy in that format (the chosen one for its own format; else the smallest
    complete llm sibling of the same base model)."""
    base = base_model_name(artifact.ref)
    out: dict[str, object] = {artifact.format: artifact}
    for a in sorted(artifacts, key=lambda a: a.weights_bytes or 0):
        if a.format in FORMAT_ENGINE and a.format not in out and a.complete and a.kind == "llm" \
                and base_model_name(a.ref) == base:
            out[a.format] = a
    return {f: a for f, a in out.items() if f in FORMAT_ENGINE}


def rule(formats: set[str], vllm_supported: bool | None, sizes: dict[str, int] | None = None,
         free: int | None = None) -> tuple[str, str] | None:
    """(engine, reason) when the rules settle it; None (ask the model) only when both formats are on disk, vLLM
    supports the architecture and — free memory known — both fit (``need_bytes``). Only one fitting -> that one;
    neither -> llama.cpp (the smaller footprint, the least to free)."""
    if formats == {"gguf"}:
        return "llamacpp", "gguf_only"
    if formats == {"safetensors"}:
        return "vllm", "safetensors_only"
    if formats != {"gguf", "safetensors"}:
        return None
    if vllm_supported is False:
        return "llamacpp", "vllm_unsupported"
    if free is not None and sizes:
        st = need_bytes("safetensors", sizes["safetensors"]) <= free
        gg = need_bytes("gguf", sizes["gguf"]) <= free
        if st and not gg:
            return "vllm", "only_vllm_fits"
        if gg and not st:
            return "llamacpp", "only_llamacpp_fits"
        if not st and not gg:
            return "llamacpp", "neither_fits"
    return None


def _bench(db, base: str) -> dict[str, str]:
    """engine -> 'best N tok/s total at concurrency C' from this box's benchmarks of the same base model."""
    if db is None:
        return {}
    from dgxmm.db import Benchmark
    try:
        with db.session() as s:
            rows = s.execute(select(Benchmark.engine, Benchmark.model_ref, Benchmark.concurrency,
                                    func.max(Benchmark.tps_total)).group_by(Benchmark.engine, Benchmark.model_ref,
                                                                            Benchmark.concurrency)).all()
    except Exception:
        log.warning("engine choice: benchmarks unavailable", exc_info=True)
        return {}
    best: dict[str, tuple[float, int]] = {}
    for eng, ref, conc, tps in rows:
        if base_model_name(ref) == base and tps is not None and tps > best.get(eng, (-1.0, 0))[0]:
            best[eng] = (float(tps), int(conc))
    return {eng: f"{tps:.0f} tok/s total at concurrency {conc}" for eng, (tps, conc) in best.items()}


def build_state(*, goal: str, ref: str, fmts: dict[str, dict], free: int | None, bench: dict[str, str],
                architecture: str | None) -> str:
    lines = [f"Goal: {goal} — {GOALS[goal]}",
             quoted_block("MODEL", ref, max_chars=300),
             quoted_block("ARCHITECTURE", architecture or "unknown", max_chars=200)]
    for f, d in fmts.items():
        need = need_bytes(f, int(d["weights_bytes"]))
        fits = "" if free is None else ("; fits in the free memory" if need <= free
                                        else "; does NOT fit in the free memory")
        lines.append(f"{FORMAT_ENGINE[f]} would load the {f} weights: {d['weights_bytes'] / GIB:.1f} GiB, "
                     f"quantisation {d.get('quant') or 'unknown'}; needs about {need / GIB:.1f} GiB{fits}")
    lines.append(f"Memory free for a new deployment: {free / GIB:.1f} GiB" if free is not None
                 else "Free memory: unknown")
    for eng in ENGINES:
        lines.append(f"Benchmark on this box ({eng}): {bench.get(eng, 'none yet')}")
    return "\n".join(lines)


def _empty(**kw) -> dict:
    return {"engine": None, "model_ref": None, "source": "none", "confidence": None, "reason": None,
            "decision_id": None, "alternatives": {}, "disabled": False, **kw}


async def suggest(ctx, model_ref: str, goal: str) -> dict:
    """``{engine, model_ref, source, confidence, reason, decision_id, alternatives: {engine: ref}, disabled}``.
    KeyError for a model that is not on disk."""
    decider = decider_of(ctx)
    if not rules_on(decider, FEATURE):
        return _empty(disabled=True)
    artifact = ctx.inventory.get(model_ref)
    if artifact is None:
        raise KeyError(model_ref)
    sib = siblings(artifact, list(ctx.inventory.list()))
    alts = {FORMAT_ENGINE[f]: a.ref for f, a in sib.items()}
    st = sib.get("safetensors")
    archs = ctx.archs.get() if getattr(ctx, "archs", None) is not None else None
    supported = None if (archs is None or st is None or not st.architecture) else st.architecture in archs
    subject = f"model:{model_ref}"[:256]
    budget = None
    if getattr(ctx, "budget", None) is not None:
        budget = ctx.budget.current() or await ctx.budget.refresh()
    free = int(budget.free_for_new) if budget is not None else None
    hit = rule(set(sib), supported, {f: int(a.weights_bytes or 0) for f, a in sib.items()}, free)
    if hit is not None:
        eng, why = hit
        d = await arecord_rules(decider, feature=FEATURE, subject=subject, choice=eng, reason=why)
        return _empty(engine=eng, model_ref=alts[eng], source="rules", confidence=1.0, reason=why,
                      decision_id=d.decision_id, alternatives=alts)
    if not sib:
        return _empty(reason="no_engine_format")
    bench = await asyncio.to_thread(_bench, getattr(ctx, "db", None), base_model_name(model_ref))
    fmts = {f: {"weights_bytes": a.weights_bytes, "quant": a.quant} for f, a in sib.items()}
    state = build_state(goal=goal, ref=model_ref, fmts=fmts, free=budget.free_for_new if budget else None,
                        bench=bench, architecture=artifact.architecture)
    d = await safe_decide(decider, feature=FEATURE, subject=subject, state=state, question=QUESTION, options=OPTIONS,
                          threshold=threshold_of(decider, FEATURE, NEVER), timeout_s=TIMEOUT_S)
    if d.source == "model" and d.choice in alts:
        return _empty(engine=d.choice, model_ref=alts[d.choice], source="model", confidence=d.confidence,
                      reason="model", decision_id=d.decision_id, alternatives=alts)
    return _empty(confidence=d.confidence, decision_id=d.decision_id, alternatives=alts, reason="both_formats")


# ---------------- evaluation adapter ----------------
def eval_case(case: dict):
    """A corpus case ({goal, formats: {safetensors|gguf: {ref, weights_bytes, quant}}, vllm_supported, free_bytes,
    benchmarks}) -> the question DGXMM asks; the rules answer where the formats settle it."""
    fmts = case["formats"]
    hit = rule(set(fmts), case.get("vllm_supported"), {f: int(d["weights_bytes"]) for f, d in fmts.items()},
               case.get("free_bytes"))
    state = build_state(goal=case["goal"], ref=case.get("ref") or case["name"], fmts=fmts,
                        free=case.get("free_bytes"), bench=case.get("benchmarks") or {},
                        architecture=case.get("architecture"))
    return SimpleNamespace(rules=hit[0] if hit else None, state=state, question=QUESTION, options=OPTIONS)
