"""Per-engine memory estimates. All values in bytes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from dgxmm.config import GIB, Settings
from dgxmm.inventory.archspec import ArchSpec
from dgxmm.memory.kv import (VLLM_MAX_CONCURRENT_BATCHES, kv_cache_bytes, kv_dtype_bytes, kv_shards, state_bytes,
                             vllm_dsv4_pool_bytes)

LOW_CONFIDENCE_MARGIN = 0.20
# vllm_gpu_util floors --gpu-memory-utilization to 1e-4 of device memory so vLLM never takes more than the
# reservation; every vLLM estimate (each rank: it is part of `runtime`) carries that much slack so the floored
# budget still covers the rest of the estimate. 16 MiB covers 1e-4 of devices up to 160 GiB (GB10: 121.7 GiB).
VLLM_UTIL_SLACK = 16 * 1024**2


@dataclass(frozen=True)
class MemoryEstimate:
    weights: int
    draft_weights: int
    kv_cache: int
    state: int
    activations: int
    runtime: int
    margin: int
    confidence: str  # low | estimated | calibrated | measured
    notes: tuple[str, ...] = field(default_factory=tuple)
    # how many ways tensor parallelism can split kv_cache (memory.kv.kv_shards; 1 = replicated on every rank,
    # 0 = no limit / an estimate stored before this field existed)
    kv_shards: int = 0

    @property
    def total(self) -> int:
        return (self.weights + self.draft_weights + self.kv_cache + self.state
                + self.activations + self.runtime + self.margin)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        d["total"] = self.total
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEstimate":
        return cls(**{k: (tuple(v) if k == "notes" else v) for k, v in d.items() if k != "total"})

    def with_total(self, total: int, confidence: str, note: str) -> "MemoryEstimate":
        """Adjust the margin so the total equals `total` (used by calibration)."""
        return replace(self, margin=self.margin + (total - self.total), confidence=confidence,
                       notes=self.notes + (note,))


def per_rank_bytes(engine: str, est: MemoryEstimate, free: dict[str, int]) -> dict[str, int]:
    """Memory each rank of a deployment needs, by node (``free``: node id -> its free_for_new, ranks in order).

    One node: the single-node total. Several: the shardable part (weights, draft weights, KV cache, recurrent state)
    and ``margin`` are split across the ranks; the absolute overheads (activations / compute buffers, runtime) are
    needed in full on every rank. vLLM shards the KV cache by KV head, so it splits at most ``est.kv_shards`` ways
    (a rank holds max(1, heads // tp) heads): MLA latents and other single-KV-head caches are replicated, i.e.
    needed in full on every rank (llama.cpp's layer split keeps each layer's KV with the layer: always split).
    ``margin`` is entirely proportional to the model: the low-confidence margin is
    LOW_CONFIDENCE_MARGIN x (KV + state + activations) and calibration (``with_total``) only scales the total
    (measured x 1.05, a mean ratio, +10% after an OOM), so it is split like the weights. (Absolute safety bytes live
    in the per-node Budget: settings.margin_bytes / os_reserve_bytes, already out of free_for_new.) vLLM (tensor
    parallel, tp = number of nodes): split evenly. llama.cpp (RPC layer split): split in proportion to each node's
    free memory (a node with none gets no share; all none: even). Rounded up (never under-estimates a rank)."""
    nodes = list(free)
    if len(nodes) <= 1:
        return {n: est.total for n in nodes}
    shard = est.weights + est.draft_weights + est.kv_cache + est.state + est.margin
    fixed = est.activations + est.runtime
    kv_split = min(len(nodes), est.kv_shards) if est.kv_shards else len(nodes)
    if engine != "llamacpp" and kv_split < len(nodes):
        shard -= est.kv_cache
        fixed += -(-est.kv_cache // kv_split)
    if engine == "llamacpp":
        weights = {n: max(free[n], 0) for n in nodes}
        if sum(weights.values()) <= 0:
            weights = {n: 1 for n in nodes}
    else:
        weights = {n: 1 for n in nodes}
    total_w = sum(weights.values())
    return {n: -(-shard * weights[n] // total_w) + fixed for n in nodes}


@dataclass(frozen=True)
class VllmSizing:
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    kv_tokens: int | None


def default_kv_tokens(max_model_len: int, max_num_seqs: int) -> int:
    """Default vLLM KV pool when kv_tokens is not set. vLLM pages all sequences into ONE shared pool,
    so reserving context x seqs is rarely needed; two full-length requests' worth is the default."""
    return min(max_model_len * max_num_seqs, 2 * max_model_len)


def _low_conf_margin(spec: ArchSpec, *parts: int) -> tuple[int, str, tuple[str, ...]]:
    if spec.confident:
        return 0, "estimated", ()
    why = []
    if spec.assumptions:
        why.append(f"config.json missing {', '.join(spec.assumptions)}")
    if spec.unmodeled:
        why.append(f"attention / KV-cache fields not modelled: {', '.join(spec.unmodeled)}")
    return int(LOW_CONFIDENCE_MARGIN * sum(parts)), "low", (f"{'; '.join(why)}; added 20% margin",)


def estimate_vllm(spec: ArchSpec, weights_bytes: int, cfg: VllmSizing, settings: Settings,
                  draft_spec: ArchSpec | None = None, draft_bytes: int = 0) -> MemoryEstimate:
    b = kv_dtype_bytes(cfg.kv_cache_dtype, spec.torch_dtype)
    tokens_total = cfg.kv_tokens or cfg.max_model_len * cfg.max_num_seqs
    # sliding windows also hold the tokens in flight: max_num_batched_tokens per concurrent (async) batch
    # (vllm/v1/kv_cache_interface.py SlidingWindowSpec.max_admission_blocks_per_request)
    kw = dict(tokens_total=tokens_total, seqs=cfg.max_num_seqs, context=cfg.max_model_len,
              bytes_per_elem=b, window_slack=VLLM_MAX_CONCURRENT_BATCHES * cfg.max_num_batched_tokens)

    def kv_of(s: ArchSpec) -> int:
        if any(g.kind == "csa" for g in s.layer_groups):  # vLLM's packed DeepSeek-V4 pool
            return vllm_dsv4_pool_bytes(s, tokens_total=tokens_total, seqs=cfg.max_num_seqs,
                                        context=cfg.max_model_len, kv_dtype=cfg.kv_cache_dtype,
                                        max_num_batched_tokens=cfg.max_num_batched_tokens)
        return kv_cache_bytes(s, **kw)
    kv = kv_of(spec)
    shards = [kv_shards(spec)]
    if draft_spec is not None:
        kv += kv_of(draft_spec)
        shards.append(kv_shards(draft_spec))
    st = state_bytes(spec, cfg.max_num_seqs)
    per_tok = (4 * spec.hidden_size + 2 * spec.intermediate_size) * 2
    act = cfg.max_num_batched_tokens * per_tok + cfg.max_num_seqs * spec.vocab_size * 4 * 2
    runtime = int((settings.vllm_runtime_fixed_gib + settings.vllm_cuda_graph_gib) * GIB) + VLLM_UTIL_SLACK
    margin, confidence, notes = _low_conf_margin(spec, kv, st, act)
    positive = [n for n in shards if n]
    return MemoryEstimate(weights=weights_bytes, draft_weights=draft_bytes, kv_cache=kv, state=st,
                          activations=act, runtime=runtime, margin=margin, confidence=confidence, notes=notes,
                          kv_shards=min(positive) if positive else 0)


def vllm_gpu_util(total_bytes: int, mem_total: int) -> float:
    """vLLM's --gpu-memory-utilization (a fraction of TOTAL device memory) for a reservation of ``total_bytes``:
    floored to 1e-4 so vLLM never takes more than DGXMM reserved (at most 1e-4 of the device less; the estimate
    carries VLLM_UTIL_SLACK for it). Integer arithmetic: no float rounding past the reservation."""
    return (max(total_bytes, 0) * 10_000 // mem_total) / 10_000


MIB = 1024**2


@dataclass(frozen=True)
class LlamaCppSizing:
    ctx_size: int
    parallel: int
    cache_type_k: str
    cache_type_v: str
    n_ubatch: int


def estimate_llamacpp(spec: ArchSpec, weights_bytes: int, cfg: LlamaCppSizing, settings: Settings) -> MemoryEstimate:
    b = (kv_dtype_bytes(cfg.cache_type_k, spec.torch_dtype) + kv_dtype_bytes(cfg.cache_type_v, spec.torch_dtype)) / 2
    kv = kv_cache_bytes(spec, tokens_total=cfg.ctx_size, seqs=cfg.parallel, context=max(cfg.ctx_size // cfg.parallel, 1),
                        bytes_per_elem=b, window_slack=cfg.n_ubatch)
    st = state_bytes(spec, cfg.parallel)
    compute = cfg.n_ubatch * spec.vocab_size * 4 + cfg.n_ubatch * spec.hidden_size * 4 * 8 + 256 * MIB
    runtime = int(settings.llamacpp_runtime_fixed_gib * GIB)
    margin, confidence, notes = _low_conf_margin(spec, kv, st, compute)
    return MemoryEstimate(weights=weights_bytes, draft_weights=0, kv_cache=kv, state=st, activations=compute,
                          runtime=runtime, margin=margin, confidence=confidence, notes=notes)
