"""KV-cache and recurrent-state sizing from an ArchSpec."""
from __future__ import annotations

import math

from dgxmm.inventory.archspec import ArchSpec

KV_DTYPE_BYTES: dict[str, float] = {
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "f16": 2, "f32": 4, "fp32": 4,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    # llama.cpp block formats: bytes per element including block scales
    "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32,
}


def kv_dtype_bytes(kv_dtype: str, model_dtype: str) -> float:
    if kv_dtype in ("auto", "", None):
        return 4.0 if model_dtype in ("float32", "fp32") else 2.0
    return KV_DTYPE_BYTES[kv_dtype]


def kv_cache_bytes(spec: ArchSpec, *, tokens_total: int, seqs: int, context: int,
                   bytes_per_elem: float, window_slack: int = 0) -> int:
    """KV bytes for a pool holding `tokens_total` tokens across `seqs` concurrent sequences.

    Sliding-window layers only keep min(context, window) tokens per sequence, plus `window_slack`
    tokens once for the prefill chunk in flight. Linear/mamba layers keep no per-token KV.
    """
    total = 0.0
    for g in spec.layer_groups:
        if g.kind == "full":
            total += g.count * 2 * g.kv_heads * g.head_dim * bytes_per_elem * tokens_total
        elif g.kind == "mla":
            total += g.count * g.mla_dim * bytes_per_elem * tokens_total
        elif g.kind == "sliding":
            w = min(context, g.window or context)
            tokens = min(tokens_total, w * seqs + window_slack)
            total += g.count * 2 * g.kv_heads * g.head_dim * bytes_per_elem * tokens
    return int(math.ceil(total))


def state_bytes(spec: ArchSpec, seqs: int, elem_bytes: int = 4) -> int:
    """Fixed per-sequence state of linear-attention / mamba layers (fp32, conservative)."""
    return sum(g.count * g.state_elems * seqs * elem_bytes for g in spec.layer_groups if g.kind in ("linear", "mamba"))
