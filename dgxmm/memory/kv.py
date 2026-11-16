"""KV-cache and recurrent-state sizing from an ArchSpec."""
from __future__ import annotations

import math
from collections import Counter

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
        elif g.kind == "csa":  # engine-neutral: window latents + compressed latents + indexer keys (no packing)
            w = min(context, g.window or context)
            total += g.count * g.head_dim * bytes_per_elem * min(tokens_total, w * seqs + window_slack)
            if g.compress_ratio > 1:
                total += g.count * (g.head_dim + g.index_dim) * bytes_per_elem * tokens_total / g.compress_ratio
    return int(math.ceil(total))


def kv_shards(spec: ArchSpec) -> int:
    """How many ways vLLM's tensor parallelism can split this model's KV cache: the KV heads of its attention
    groups (a rank holds max(1, heads // tp) of them). MLA and DeepSeek-V4 caches hold one latent per token
    (num_kv_heads=1 in vLLM's MLAAttentionSpec): replicated on every rank. 0: no per-token KV."""
    heads = [1 if g.kind in ("mla", "csa") else g.kv_heads for g in spec.layer_groups if g.kind in
             ("full", "sliding", "mla", "csa")]
    return max(min(heads), 1) if heads else 0


# ---- vLLM's packed KV pool for DeepSeek-V4 (vllm/v1/core/kv_cache_utils.py) ----
# Mirrors the vLLM image DGXMM deploys, vllm/vllm-openai:pinned-2026-08-27. Assumptions taken from that image:
# blocks of 256 tokens (compressed latent / indexer), 64 (SWA), 4 / 8 (compressor states); fp8 kv-cache stored as
# fp8_ds_mla (584 B per state, pages padded to 576 B; other dtypes: plain rows padded to 512 B); async scheduling,
# so 2 x max_num_batched_tokens in flight; no speculative (MTP) layers. After a vLLM upgrade re-run
# tools/vllm_dsv4_pool_sim.py in the new image and compare with tests/test_deepseek_v4_kv.py.
# The model registers several caches per layer with different page sizes (models/deepseek_v4/attention.py,
# compressor.py, v1/attention/backends/mla/sparse_swa.py). vLLM groups them (group_and_unify_kv_cache_specs,
# _get_kv_cache_groups_uniform_groups) and every group draws blocks from ONE shared pool whose block is as large as
# the largest group's pages (_get_kv_cache_bytes_per_block), so a small compressor-state page still costs a full
# pool block. A request needs the sum of its per-group block counts (_max_memory_usage_bytes_from_groups).
VLLM_DSV4_BLOCK = 256          # DeepseekSparseSWABackend.get_preferred_block_size
VLLM_DSV4_SWA_BLOCK = 64       # DeepseekV4SWACache.block_size
VLLM_MAX_CONCURRENT_BATCHES = 2  # async scheduling (default): VllmConfig.max_concurrent_batches, pp=1


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(x: int, a: int) -> int:
    return _cdiv(x, a) * a


def _approximate_gcd(values: list[int], lower_bound: int) -> int:
    """kv_cache_utils._approximate_gcd: the d >= lower_bound minimising the round-up padding (ties: larger d)."""
    best_d, best_pad = lower_bound, None
    for d in range(lower_bound, max(max(values), lower_bound) + 1):
        pad = sum((d - x % d) % d for x in values)
        if best_pad is None or pad <= best_pad:
            best_d, best_pad = d, pad
    return best_d


def _dsv4_caches(spec: ArchSpec, kv_dtype: str, model_dtype: str) -> list[tuple[int, int, int | None, int]]:
    """(page bytes, block size, sliding window or None = full, layer count) of every cache vLLM registers."""
    ds_mla = str(kv_dtype).startswith("fp8")  # fp8 -> fp8_ds_mla (attention.py _resolve_dsv4_kv_cache_dtype)
    align = 576 if ds_mla else 512
    out: list[tuple[int, int, int | None, int]] = []
    for g in spec.layer_groups:
        if g.kind != "csa":
            continue
        latent = 584 if ds_mla else int(g.head_dim * kv_dtype_bytes(kv_dtype, model_dtype))  # bytes per state
        out.append((_round_up(VLLM_DSV4_SWA_BLOCK * latent, align), VLLM_DSV4_SWA_BLOCK, g.window, g.count))
        r = g.compress_ratio
        if r > 1:
            out.append((_round_up(VLLM_DSV4_BLOCK // r * latent, align), VLLM_DSV4_BLOCK, None, g.count))
            coff = 1 + (r == 4)  # compressor state: fp32 kv + score, overlapping windows for ratio 4
            bs = 4 if r == 4 else 8
            out.append((_round_up(bs * 2 * coff * g.head_dim * 4, align), bs, coff * r, g.count))
        if r == 4 and g.index_dim:  # indexer keys: fp8 + one fp32 scale per 128 (132 B), and their compressor
            key = g.index_dim + g.index_dim // 128 * 4
            out.append((_round_up(VLLM_DSV4_BLOCK // r * key, align), VLLM_DSV4_BLOCK, None, g.count))
            out.append((_round_up(4 * 2 * 2 * g.index_dim * 4, align), 4, 8, g.count))
    return out


def vllm_dsv4_pool_bytes(spec: ArchSpec, *, tokens_total: int, seqs: int, context: int, kv_dtype: str,
                         max_num_batched_tokens: int) -> int:
    """Bytes of vLLM's shared DeepSeek-V4 KV pool for `tokens_total` tokens over `seqs` concurrent sequences of up
    to `context` tokens. One sequence at its full context reproduces vLLM's start-up check exactly (plus the null
    block the pool holds back). Several sequences: the full-attention groups hold their tokens (plus a partial
    block each); a windowed group's blocks are bounded by every sequence's window (+2 blocks for block alignment)
    plus the in-flight tokens once (SlidingWindowSpec.max_admission_blocks_per_request)."""
    caches = _dsv4_caches(spec, kv_dtype, spec.torch_dtype)
    if not any(window is None for _, _, window, _ in caches):
        raise ValueError("DeepSeek-V4 pool without a compressed layer: not the layout vLLM packs")
    in_flight = VLLM_MAX_CONCURRENT_BATCHES * max_num_batched_tokens
    full: Counter[int] = Counter()
    windowed: dict[tuple[int, int], Counter[int]] = {}
    # keyed by page size, as vLLM does (layers_per_size in _get_kv_cache_groups_uniform_groups): two caches with
    # equal pages in one (block, window) group become one size with both caches' layers (verified against vLLM:
    # tests/test_deepseek_v4_kv.py::test_caches_with_equal_page_sizes_match_vllm)
    for page, bs, window, count in caches:
        if window is None:
            full[page] += count
        else:
            windowed.setdefault((bs, window), Counter())[page] += count
    tuples = [full.most_common(1)[0][1]] + [c.most_common(1)[0][1] for c in windowed.values()]
    t = _approximate_gcd(tuples, lower_bound=tuples[0])
    bytes_per_block = sum(p * n for p, n in full.items())
    blocks = max(_cdiv(context, VLLM_DSV4_BLOCK),
                 _cdiv(tokens_total, VLLM_DSV4_BLOCK) + max(min(seqs, _cdiv(tokens_total, max(context, 1))) - 1, 0))
    for (bs, window), pages in windowed.items():
        layers = max(pages.values())
        n_sub = _cdiv(layers, t)
        tuple_bytes = sum(pages)  # one page of each size per layer tuple
        bytes_per_block = max(bytes_per_block, _cdiv(layers, n_sub) * tuple_bytes)
        one = _cdiv(min(window - 1 + in_flight, context), bs) + 1
        many = seqs * (_cdiv(min(window - 1, context), bs) + 2) + _cdiv(min(in_flight, tokens_total), bs) - 1
        blocks += n_sub * max(one, min(many, seqs * (_cdiv(context, bs) + 1)))
    return bytes_per_block * (blocks + 1)  # + the null block (get_kv_cache_configs: check_memory)


def state_bytes(spec: ArchSpec, seqs: int, elem_bytes: int = 4) -> int:
    """Fixed per-sequence state of linear-attention / mamba layers (fp32, conservative)."""
    return sum(g.count * g.state_elems * seqs * elem_bytes for g in spec.layer_groups if g.kind in ("linear", "mamba"))
