"""Per-engine memory estimates. All values in bytes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from dgxmm.config import GIB, Settings
from dgxmm.inventory.archspec import ArchSpec
from dgxmm.memory.kv import kv_cache_bytes, kv_dtype_bytes, state_bytes

LOW_CONFIDENCE_MARGIN = 0.20


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
    return (int(LOW_CONFIDENCE_MARGIN * sum(parts)), "low",
            (f"config.json missing {', '.join(spec.assumptions)}; added 20% margin",))


def estimate_vllm(spec: ArchSpec, weights_bytes: int, cfg: VllmSizing, settings: Settings,
                  draft_spec: ArchSpec | None = None, draft_bytes: int = 0) -> MemoryEstimate:
    b = kv_dtype_bytes(cfg.kv_cache_dtype, spec.torch_dtype)
    tokens_total = cfg.kv_tokens or cfg.max_model_len * cfg.max_num_seqs
    kw = dict(tokens_total=tokens_total, seqs=cfg.max_num_seqs, context=cfg.max_model_len,
              bytes_per_elem=b, window_slack=cfg.max_num_batched_tokens)
    kv = kv_cache_bytes(spec, **kw)
    if draft_spec is not None:
        kv += kv_cache_bytes(draft_spec, **kw)
    st = state_bytes(spec, cfg.max_num_seqs)
    per_tok = (4 * spec.hidden_size + 2 * spec.intermediate_size) * 2
    act = cfg.max_num_batched_tokens * per_tok + cfg.max_num_seqs * spec.vocab_size * 4 * 2
    runtime = int((settings.vllm_runtime_fixed_gib + settings.vllm_cuda_graph_gib) * GIB)
    margin, confidence, notes = _low_conf_margin(spec, kv, st, act)
    return MemoryEstimate(weights=weights_bytes, draft_weights=draft_bytes, kv_cache=kv, state=st,
                          activations=act, runtime=runtime, margin=margin, confidence=confidence, notes=notes)


def vllm_gpu_util(total_bytes: int, mem_total: int) -> float:
    """vLLM's --gpu-memory-utilization is a fraction of TOTAL device memory."""
    return math.ceil(total_bytes / mem_total * 100) / 100


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
