"""DeepSeek-V4 (compressed sparse attention) KV sizing, KV replication across tensor-parallel ranks, unmodelled
attention features and the --gpu-memory-utilization rounding.

Ground truth: the KV-cache specs vLLM (image pinned-2026-08-27) builds for DeepseekV4ForCausalLM, run through
vLLM's own grouping and sizing (v1/core/kv_cache_utils.py), and a real 2-node TP=2 start-up failure of
deepseek-ai/DeepSeek-V4-Flash-0731 (max_model_len 32768, kv_cache_dtype fp8, max_num_batched_tokens 8192):
  "To serve at least one request with the model's max seq len (32768), (6.36 GiB KV cache is needed ..."
"""
import json
from pathlib import Path

import pytest

from dgxmm.config import GIB
from dgxmm.inventory.archspec import LayerGroup, parse_hf_config
from dgxmm.memory.estimate import MemoryEstimate, VllmSizing, estimate_vllm, per_rank_bytes, vllm_gpu_util

FIX = Path(__file__).parent / "fixtures" / "configs"
DSV4 = json.loads((FIX / "deepseek-v4-flash.json").read_text())

# vLLM's packed DeepSeek-V4 pool for this config with fp8 (fp8_ds_mla): one pool block spans the largest group's
# pages, 21 x 37440 (C4 latent) + 21 x 8640 (C4 indexer) + 20 x 1728 (C128 latent) bytes
BYTES_PER_BLOCK = 1_002_240
# one request at max_model_len 32768, 2 x 8192 tokens in flight (async scheduling): 128 latent blocks,
# 2 x 259 SWA, 4099 C4-state and 2065 C128-state blocks (_max_memory_usage_bytes_from_groups)
ONE_REQUEST_32K = 6810 * BYTES_PER_BLOCK


def dsv4(**over):
    return parse_hf_config({**DSV4, **over})


def test_deepseek_v4_layer_groups():
    s = dsv4()
    assert s.confident, s.assumptions + s.unmodeled
    assert s.layer_groups == (
        LayerGroup("csa", 2, kv_heads=1, head_dim=512, window=128, compress_ratio=1),
        LayerGroup("csa", 21, kv_heads=1, head_dim=512, window=128, compress_ratio=4, index_dim=128),
        LayerGroup("csa", 20, kv_heads=1, head_dim=512, window=128, compress_ratio=128),
    )


def test_one_request_matches_vllm_check(settings):
    """vLLM needs ONE_REQUEST_32K (+ the null block it holds back) for a single max-length request: the failure
    message's 6.36 GiB."""
    est = estimate_vllm(dsv4(), 0, VllmSizing(32768, 1, 8192, "fp8", 32768), settings)
    assert est.kv_cache == ONE_REQUEST_32K + BYTES_PER_BLOCK
    assert f"{ONE_REQUEST_32K / GIB:.2f}" == "6.36"


def test_failed_deployment_estimate_is_not_30x_low(settings):
    """The deployment that failed: 4 sequences, the default pool. The old estimate was 0.36 GiB."""
    est = estimate_vllm(dsv4(), 166_886_535_336, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    assert ONE_REQUEST_32K + BYTES_PER_BLOCK <= est.kv_cache <= 1.10 * ONE_REQUEST_32K
    assert est.confidence == "estimated"


def test_bf16_kv_is_larger_than_fp8(settings):
    a = estimate_vllm(dsv4(), 0, VllmSizing(32768, 1, 8192, "auto", 32768), settings)
    assert a.kv_cache > ONE_REQUEST_32K


def test_context_grows_only_the_latent_blocks(settings):
    """Past window + in-flight tokens only the compressed latent / indexer group grows: 1 pool block per 256."""
    a = estimate_vllm(dsv4(), 0, VllmSizing(32768, 1, 8192, "fp8", 32768), settings)
    b = estimate_vllm(dsv4(), 0, VllmSizing(65536, 1, 8192, "fp8", 65536), settings)
    assert b.kv_cache - a.kv_cache == 128 * BYTES_PER_BLOCK


# ---------- KV replicated on every tensor-parallel rank ----------

def _two(est):
    return per_rank_bytes("vllm", est, {"local": 100 * GIB, "dgx-2": 100 * GIB})


def test_single_kv_head_cache_is_counted_on_every_rank(settings):
    est = estimate_vllm(dsv4(), 166_886_535_336, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    assert est.kv_shards == 1
    need = _two(est)
    for n in need:
        assert need[n] >= -(-est.weights // 2) + est.kv_cache + est.activations + est.runtime


def test_mla_latent_is_replicated_gqa_is_split(settings):
    mla = parse_hf_config({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3",
                           "num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128,
                           "num_key_value_heads": 128, "vocab_size": 129280, "max_position_embeddings": 163840,
                           "kv_lora_rank": 512, "qk_rope_head_dim": 64, "intermediate_size": 18432})
    e = estimate_vllm(mla, 10 * GIB, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    assert e.kv_shards == 1
    assert _two(e)["local"] >= 5 * GIB + e.kv_cache + e.activations + e.runtime
    gqa = parse_hf_config(json.loads((FIX / "qwen3-8b.json").read_text()))
    g = estimate_vllm(gqa, 10 * GIB, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    assert g.kv_shards == 8
    assert _two(g)["local"] == -(-(g.weights + g.kv_cache + g.state + g.margin) // 2) + g.activations + g.runtime


def test_kv_split_limited_by_kv_heads():
    est = MemoryEstimate(0, 0, 8 * GIB, 0, 0, 0, 0, "estimated", kv_shards=2)
    four = {f"n{i}": 100 * GIB for i in range(4)}
    assert set(per_rank_bytes("vllm", est, four).values()) == {4 * GIB}  # 2 heads over 4 ranks: halves, not quarters


def test_llamacpp_layer_split_still_splits_kv(settings):
    est = MemoryEstimate(0, 0, 8 * GIB, 0, 0, 0, 0, "estimated", kv_shards=1)
    assert set(per_rank_bytes("llamacpp", est, {"a": GIB, "b": GIB}).values()) == {4 * GIB}


def test_old_stored_estimate_without_kv_shards_still_loads():
    d = MemoryEstimate(1, 0, 2, 0, 0, 0, 0, "estimated").as_dict()
    d.pop("kv_shards")
    assert MemoryEstimate.from_dict(d).kv_shards == 0


# ---------- attention features the parser does not model ----------

@pytest.mark.parametrize("over", [
    {"model_type": "deepseek_v5", "architectures": ["DeepseekV5ForCausalLM"]},   # compress_ratios, unknown model
    {"compress_ratios": [0, 0] + [4, 64] * 20 + [4, 0, 0, 0]},                   # a ratio vLLM's code rejects
])
def test_compressed_attention_variants_are_low_confidence(settings, over):
    s = dsv4(**over)
    assert not s.confident and "compress_ratios" in s.unmodeled
    est = estimate_vllm(s, 10 * GIB, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    assert est.confidence == "low" and est.margin > 0
    assert any("compress_ratios" in n and "not modelled" in n for n in est.notes)


def test_indexer_fields_on_mla_model_are_low_confidence(settings):
    """DeepSeek-V3.2 style: MLA plus a lightning-indexer cache the parser does not size."""
    s = parse_hf_config({"architectures": ["DeepseekV32ForCausalLM"], "model_type": "deepseek_v32",
                         "num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128,
                         "vocab_size": 129280, "max_position_embeddings": 163840, "kv_lora_rank": 512,
                         "qk_rope_head_dim": 64, "index_head_dim": 128, "index_n_heads": 64, "index_topk": 2048})
    assert not s.confident and {"index_head_dim", "index_n_heads", "index_topk"} <= set(s.unmodeled)
    est = estimate_vllm(s, 10 * GIB, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    assert est.confidence == "low" and est.margin > 0


def test_ordinary_configs_stay_confident():
    for name in ("qwen3-8b", "gemma-4-31b-it", "qwen3.6-35b-a3b-nvfp4", "granite-4.2-30b"):
        s = parse_hf_config(json.loads((FIX / f"{name}.json").read_text()))
        assert s.confident and not s.unmodeled, name


# ---------- --gpu-memory-utilization never exceeds the reservation ----------

@pytest.mark.parametrize("reserved", [int(83.0 * GIB), 50 * GIB, 50 * GIB + 1, 12 * GIB, int(89.37 * GIB)])
def test_gpu_util_never_exceeds_reservation(reserved):
    total = int(121.69 * GIB)
    util = vllm_gpu_util(reserved, total)
    assert util * total <= reserved
    assert util * total >= reserved - total / 10_000  # floored to 1e-4 of the device: <= 12.5 MiB on a GB10


def test_reservation_leaves_vllm_its_estimate(settings):
    """The reservation carries the flooring slack, so the floored fraction still covers every other part."""
    from dgxmm.memory.estimate import VLLM_UTIL_SLACK
    total = int(121.69 * GIB)
    assert VLLM_UTIL_SLACK >= 128 * GIB / 10_000
    est = estimate_vllm(dsv4(), 166_886_535_336, VllmSizing(32768, 4, 8192, "fp8", 65536), settings)
    for need in _two(est).values():
        assert vllm_gpu_util(need, total) * total >= need - VLLM_UTIL_SLACK
