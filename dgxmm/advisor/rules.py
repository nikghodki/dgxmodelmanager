"""Tuning advice (spec §9.1). Each rule: AdviceInput -> Advice | None. Pure functions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from dgxmm.planner.plans import base_model_name

GIB = 1024**3


@dataclass
class AdviceInput:
    engine: str
    cfg: dict
    artifact: Any
    samples: list[dict]          # last ~15 min of live samples
    day: list[dict]              # last 24 h of minute rollups
    variants: list[Any]          # other local artifacts of the same base model
    drafts: list[Any]            # local draft models for this base model
    benchmarks: list[dict]       # benchmark rows for this model_ref
    estimate: dict
    reserved: int


@dataclass
class Advice:
    id: str
    title: str
    rationale: str
    expected_effect: str
    confidence: str = "rule"     # rule | estimated | measured
    patch: dict | None = None
    severity: str = "suggestion"  # warning | suggestion | info
    model_ref: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _max(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def _frac(rows: list[dict], pred: Callable[[dict], bool]) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


# ---- vLLM rules ----
def nvfp4_variant(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.artifact.quant not in ("bf16", "fp16"):
        return None
    v = next((v for v in i.variants if v.quant == "nvfp4"), None)
    if not v:
        return None
    return Advice("nvfp4_variant", f"Serve {v.ref} (NVFP4) instead",
                  "GB10 (Blackwell) has native FP4 tensor cores; an NVFP4 variant of this model is already on disk.",
                  "about 3.5× less weight memory and roughly 2× faster decode", model_ref=v.ref)


def fp8_kv(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.cfg.get("kv_cache_dtype") != "auto" or i.cfg.get("max_model_len", 0) <= 32768:
        return None
    frees = i.estimate.get("kv_cache", 0) / 2
    return Advice("fp8_kv", "Use an fp8 KV cache", "Long context makes the KV cache the biggest memory item.",
                  f"halves KV memory (frees about {frees / GIB:.1f} GiB) with negligible quality loss",
                  patch={"kv_cache_dtype": "fp8"})


def _spec_method(ref: str) -> tuple[str, int]:
    r = ref.lower()
    if "dflash" in r:
        return "dflash", 7
    if "eagle3" in r:
        return "eagle3", 3
    if "eagle" in r:
        return "eagle", 3
    return "draft_model", 3


def spec_decode(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.cfg.get("speculative"):
        return None
    mean_running = _mean(i.samples, "running")
    if mean_running is not None and mean_running > 2:
        return None
    if i.drafts:
        d = i.drafts[0]
        method, n = _spec_method(d.ref)
        spec = {"method": method, "model": d.ref, "num_speculative_tokens": n}
    elif any("mtp" in f.lower() for f in i.artifact.files) or any(k in (i.artifact.config or {}) for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers")):
        spec = {"method": "mtp", "num_speculative_tokens": 2}
    else:
        return None
    return Advice("spec_decode", f"Enable speculative decoding ({spec['method']})",
                  "Traffic is mostly one or two requests at a time, where decode is memory-bandwidth bound.",
                  "typically 1.5–2.5× faster single-stream decode", patch={"speculative": spec})


def spec_accept_low(i: AdviceInput) -> Advice | None:
    spec = i.cfg.get("speculative")
    rate = _mean(i.samples, "spec_accept_rate")
    if i.engine != "vllm" or not spec or rate is None or rate >= 0.5:
        return None
    n = int(spec.get("num_speculative_tokens", 1))
    patch = {"speculative": None} if n <= 1 else {"speculative": {**spec, "num_speculative_tokens": max(1, n // 2)}}
    return Advice("spec_accept_low", "Speculative decoding is rarely accepted",
                  f"Only {rate:.0%} of draft tokens are accepted, so drafting costs more than it saves.",
                  "fewer wasted draft tokens; higher throughput", patch=patch)


def raise_seqs(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or len(i.samples) < 12:
        return None
    queued = _frac(i.samples, lambda r: (r.get("waiting") or 0) > 0)
    kv = _mean(i.samples, "kv_usage")
    if queued <= 0.5 or kv is None or kv >= 0.7:
        return None
    new = min(int(i.cfg.get("max_num_seqs", 8)) * 2, 256)
    return Advice("raise_seqs", f"Allow {new} concurrent sequences",
                  f"Requests are queued {queued:.0%} of the time while the KV cache is only {kv:.0%} used.",
                  "shorter queue and higher total tok/s", patch={"max_num_seqs": new})


def kv_pressure(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm":
        return None
    peak = _max(i.samples, "kv_usage")
    pre = sum(r.get("preemptions") or 0 for r in i.samples)
    if peak is None or peak <= 0.95 or pre <= 0:
        return None
    if i.cfg.get("kv_cache_dtype") == "auto":
        patch = {"kv_cache_dtype": "fp8"}
    else:
        patch = {"max_num_seqs": max(1, int(i.cfg.get("max_num_seqs", 8)) // 2)}
    return Advice("kv_pressure", "KV cache is full and requests are being preempted",
                  f"Peak KV usage {peak:.0%} with {pre:.0f} preemptions recently.",
                  "stops preemption stalls", patch=patch, severity="warning")


def shrink_reservation(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or len(i.day) < 60:
        return None
    peak = _max(i.day, "kv_usage")
    kv = i.estimate.get("kv_cache", 0)
    if peak is None or peak >= 0.3 or kv < 4 * GIB:
        return None
    from dgxmm.memory.estimate import default_kv_tokens
    ctx_len = int(i.cfg["max_model_len"])
    pool = i.cfg.get("kv_tokens") or default_kv_tokens(ctx_len, int(i.cfg.get("max_num_seqs", 1)))
    new_pool = max(ctx_len, int(pool / 2))
    if new_pool >= pool:
        return None  # already at the minimum pool (one full-length request)
    return Advice("shrink_reservation", "Reserve a smaller KV pool",
                  f"The KV cache never went above {peak:.0%} in the last 24 hours.",
                  f"frees about {kv / 2 / GIB:.1f} GiB so another model can run alongside",
                  patch={"kv_tokens": new_pool})


def prefix_useless(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or not i.cfg.get("enable_prefix_caching") or len(i.day) < 60:
        return None
    hit = _mean(i.day, "prefix_hit_rate")
    if hit is None or hit >= 0.02:
        return None
    return Advice("prefix_useless", "Prefix caching is not helping this workload",
                  f"Prefix cache hit rate was {hit:.1%} over the last 24 hours.",
                  "no change needed; informational", severity="info")


def moe_batch(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or not getattr(i.artifact.arch, "is_moe", False) or int(i.cfg.get("max_num_seqs", 8)) >= 16:
        return None
    return Advice("moe_batch", "Batch more requests for this MoE model",
                  "Mixture-of-experts models activate few parameters per token, so larger batches are cheap.",
                  "higher total tok/s under load",
                  patch={"max_num_seqs": 16, "max_num_batched_tokens": max(16384, int(i.cfg.get("max_num_batched_tokens", 8192)))})


def try_llamacpp(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or len(i.samples) < 30:
        return None
    running = _mean(i.samples, "running")
    g = next((v for v in i.variants if v.source == "gguf"), None)
    if running is None or running > 1.2 or g is None:
        return None
    return Advice("try_llamacpp", "Benchmark the llama.cpp GGUF variant",
                  "This deployment almost always serves a single request; llama.cpp is often faster for one stream.",
                  f"possibly higher single-user tok/s with {g.ref.split('/')[-1]}", severity="info", model_ref=g.ref)


def batched_tokens_small(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or int(i.cfg.get("max_num_batched_tokens", 8192)) >= 8192:
        return None
    ttft = _mean(i.samples, "ttft_p50")
    if ttft is None or ttft <= 1.0:
        return None
    return Advice("batched_tokens_small", "Raise max batched tokens to 8192",
                  f"Median time-to-first-token is {ttft:.1f}s and prompts are prefilled in small chunks.",
                  "faster prefill / lower TTFT", patch={"max_num_batched_tokens": 8192})


def chunked_prefill_off(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.cfg.get("enable_chunked_prefill", True):
        return None
    return Advice("chunked_prefill_off", "Enable chunked prefill",
                  "Without it, one long prompt blocks decoding for every other request.",
                  "smoother latency with mixed prompt lengths", patch={"enable_chunked_prefill": True})


# ---- llama.cpp rules ----
def llamacpp_offload_all(i: AdviceInput) -> Advice | None:
    layers = getattr(i.artifact.arch, "num_layers", 0)
    if i.engine != "llamacpp" or int(i.cfg.get("n_gpu_layers", 999)) > layers:
        return None
    return Advice("llamacpp_offload_all", "Offload all layers to the GPU",
                  "GB10 memory is unified, so keeping layers on the CPU saves no memory and runs slower.",
                  "large decode speed-up", patch={"n_gpu_layers": 999})


def llamacpp_flash_attn(i: AdviceInput) -> Advice | None:
    if i.engine != "llamacpp" or i.cfg.get("flash_attn") != "off":
        return None
    return Advice("llamacpp_flash_attn", "Turn on flash attention", "Flash attention is faster and enables a quantized KV cache.",
                  "faster attention, less memory", patch={"flash_attn": "on"})


def llamacpp_quant_kv(i: AdviceInput) -> Advice | None:
    if i.engine != "llamacpp" or i.cfg.get("cache_type_k") != "f16" or int(i.cfg.get("ctx_size", 0)) <= 32768:
        return None
    return Advice("llamacpp_quant_kv", "Use a q8_0 KV cache", "Long context makes the f16 KV cache large.",
                  "about half the KV memory", patch={"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"})


RULES: list[Callable[[AdviceInput], Advice | None]] = [
    nvfp4_variant, fp8_kv, spec_decode, spec_accept_low, raise_seqs, kv_pressure, shrink_reservation, prefix_useless,
    moe_batch, try_llamacpp, batched_tokens_small, chunked_prefill_off, llamacpp_offload_all, llamacpp_flash_attn,
    llamacpp_quant_kv,
]
_SEVERITY = {"warning": 0, "suggestion": 1, "info": 2}


def _with_measurements(a: Advice, i: AdviceInput) -> Advice:
    if not a.patch or not i.benchmarks:
        return a
    from dgxmm.deployer.service import config_hash
    cur_h, new_h = config_hash(i.cfg), config_hash({**i.cfg, **a.patch})

    def tps(h):
        rows = [b for b in i.benchmarks if b["config_hash"] == h and b["concurrency"] == 1]
        return rows[0]["tps_per_req"] if rows else None

    cur, new = tps(cur_h), tps(new_h)
    if cur is not None and new is not None:
        a.expected_effect = f"measured on this box: {new:.0f} vs {cur:.0f} tok/s at concurrency 1"
        a.confidence = "measured"
    return a


def advise(i: AdviceInput) -> list[Advice]:
    out = []
    for rule in RULES:
        try:
            a = rule(i)
        except (KeyError, TypeError, ValueError):
            a = None
        if a:
            out.append(_with_measurements(a, i))
    return sorted(out, key=lambda a: _SEVERITY[a.severity])


class AdvisorService:
    def __init__(self, deployer, inventory, store, db):
        self.deployer, self.inventory, self.store, self.db = deployer, inventory, store, db

    def advise(self, dep_id: int) -> list[dict]:
        from sqlalchemy import select
        from dgxmm.db import Benchmark
        v = self.deployer.view(dep_id)
        artifact = self.inventory.get(v["model_ref"])
        if artifact is None:
            return []
        base = base_model_name(artifact.ref)
        others = [a for a in self.inventory.list() if a.ref != artifact.ref and a.complete]
        variants = [a for a in others if a.kind == "llm" and base_model_name(a.ref) == base]
        drafts = [a for a in others if a.kind == "draft" and base_model_name(a.ref).startswith(base)]
        with self.db.session() as s:
            rows = s.scalars(select(Benchmark).where(Benchmark.model_ref == v["model_ref"]).order_by(Benchmark.id.desc()))
            benches = [{"config_hash": b.config_hash, "concurrency": b.concurrency, "tps_per_req": b.tps_per_req,
                        "tps_total": b.tps_total} for b in rows]
        inp = AdviceInput(engine=v["engine"], cfg=v["config"], artifact=artifact, samples=self.store.series(dep_id)[-180:],
                          day=self.store.history(dep_id, 24), variants=variants, drafts=drafts, benchmarks=benches,
                          estimate=v["estimate"] or {}, reserved=v["reserved_bytes"])
        return [a.as_dict() for a in advise(inp)]
