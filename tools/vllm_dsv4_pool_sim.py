#!/usr/bin/env python3
"""Ground truth for dgxmm.memory.kv.vllm_dsv4_pool_bytes: rebuild DeepSeek-V4's per-layer KV-cache specs with vLLM's
own classes and run vLLM's own grouping and sizing (vllm/v1/core/kv_cache_utils.py) on them. No GPU needed.

Run it inside the vLLM image DGXMM deploys (re-run after every vLLM image upgrade and compare with
tests/test_deepseek_v4_kv.py):

    docker run --rm --entrypoint python3 -v "$PWD/tools:/sim:ro" -v /path/to/config.json:/cfg.json:ro \\
        vllm/vllm-openai:pinned-2026-08-27 -I /sim/vllm_dsv4_pool_sim.py /cfg.json <max_model_len> \\
        [<max_num_batched_tokens> [<available_kv_gib>]]

It prints vLLM's KV-cache groups, the shared pool's bytes per block, the bytes one request at max_model_len needs
(vLLM's start-up check; the pool also holds back one null block) and, given the KV memory available, the maximum
model length vLLM would report. Last line: the same as JSON.

The spec constructors below copy what the model registers in vllm/vllm-openai:pinned-2026-08-27 (fp8 kv-cache ->
fp8_ds_mla layout; the backend's 256-token block; async scheduling, i.e. 2 x max_num_batched_tokens in flight). If a
vLLM upgrade changes models/deepseek_v4/attention.py, compressor.py or v1/attention/backends/mla/sparse_swa.py,
update both this file and dgxmm/memory/kv.py.
"""
import json
import sys
from types import SimpleNamespace

import torch
from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def specs_for(cfg: dict) -> dict:
    head = cfg["head_dim"]
    index_dim = cfg.get("index_head_dim") or 128
    index_key = index_dim + index_dim // 128 * 4  # attention.py: fp8 keys + one fp32 scale per 128
    out = {}
    for i, raw in enumerate(cfg["compress_ratios"][: cfg["num_hidden_layers"]]):
        r, p = max(1, raw), f"layers.{i}"
        out[p + ".swa"] = SlidingWindowMLASpec(  # sparse_swa.py DeepseekV4SWACache, every layer
            block_size=64, num_kv_heads=1, head_size=head, dtype=torch.uint8, sliding_window=cfg["sliding_window"],
            cache_dtype_str="fp8_ds_mla", state_content_bytes=584, alignment=576, model_version="deepseek_v4")
        if r > 1:  # attention.py compressed latent cache
            out[p + ".attn"] = MLAAttentionSpec(
                block_size=256, num_kv_heads=1, head_size=head, dtype=torch.uint8, tokens_per_state=r,
                cache_dtype_str="fp8_ds_mla", alignment=576, model_version="deepseek_v4", state_content_bytes=584)
            coff = 1 + (r == 4)  # compressor.py CompressorStateCache, state_dim = 2 * coff * head_dim
            out[p + ".comp_state"] = SlidingWindowMLASpec(
                block_size=4 if r == 4 else 8, num_kv_heads=1, head_size=2 * coff * head, dtype=torch.float32,
                sliding_window=coff * r, alignment=576)
        if r == 4:  # attention.py DeepseekV4IndexerCache + the indexer's compressor state
            out[p + ".idx"] = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=index_key, dtype=torch.uint8,
                                               tokens_per_state=4, alignment=576)
            out[p + ".idx_state"] = SlidingWindowMLASpec(block_size=4, num_kv_heads=1, head_size=2 * 2 * index_dim,
                                                         dtype=torch.float32, sliding_window=8, alignment=576)
    return out


def main() -> None:
    cfg = json.load(open(sys.argv[1]))
    max_len = int(sys.argv[2])
    batched = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    vc = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_len),
                         parallel_config=SimpleNamespace(decode_context_parallel_size=1),
                         scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
                         speculative_config=None, max_in_flight_tokens=2 * batched)
    groups = U.get_kv_cache_groups(vc, specs_for(cfg))
    bpb = U._pool_bytes_per_block(groups)
    for g in groups:
        s = g.kv_cache_spec
        print(f"{len(g.layer_names):3d} layers  block={s.block_size:4d}  "
              f"pages={sum(x.page_size_bytes for x in s.kv_cache_specs.values())}  blocks/req={s.max_memory_usage_pages(vc)}")
    need = U._max_memory_usage_bytes_from_groups(vc, groups)
    print(f"bytes_per_block {bpb}; one request at {max_len} needs {need} B ({need / 2**30:.3f} GiB)")
    result = {"bytes_per_block": bpb, "one_request": need, "max_model_len": max_len}
    if len(sys.argv) > 4:
        avail = int(float(sys.argv[4]) * 2**30)
        result["max_len_for_available"] = U._estimate_max_model_len_from_groups(vc, groups, avail - bpb)
        print(f"max model len for {sys.argv[4]} GiB of KV memory: {result['max_len_for_available']}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
