import json
from pathlib import Path

from dgxmm.config import GIB
from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.memory.estimate import MemoryEstimate, VllmSizing, estimate_vllm, vllm_gpu_util
from dgxmm.memory.kv import kv_cache_bytes, kv_dtype_bytes, state_bytes

FIX = Path(__file__).parent / "fixtures" / "configs"


def spec(name):
    return parse_hf_config(json.loads((FIX / f"{name}.json").read_text()))


def test_dense_kv_bytes():
    s = spec("qwen3-8b")
    assert kv_cache_bytes(s, tokens_total=1, seqs=1, context=1, bytes_per_elem=2) == 36 * 2 * 8 * 128 * 2


def test_vllm_estimate_qwen3_8b_exact(settings):
    s = spec("qwen3-8b")
    est = estimate_vllm(s, 16_384_000_000, VllmSizing(32768, 4, 8192, "auto", None), settings)
    assert est.kv_cache == 147456 * 32768 * 4 == 19_327_352_832
    assert est.activations == 8192 * (4 * 4096 + 2 * 12288) * 2 + 4 * 151936 * 4 * 2
    assert est.runtime == int(4.5 * GIB)
    assert est.margin == 0 and est.confidence == "estimated"
    assert est.total == 16_384_000_000 + 19_327_352_832 + est.activations + int(4.5 * GIB)
    fp8 = estimate_vllm(s, 16_384_000_000, VllmSizing(32768, 4, 8192, "fp8", None), settings)
    assert fp8.kv_cache == 19_327_352_832 // 2


def test_vllm_kv_tokens_pool_overrides_full_context_times_seqs(settings):
    s = spec("qwen3-8b")
    est = estimate_vllm(s, 0, VllmSizing(32768, 8, 8192, "auto", 65536), settings)
    assert est.kv_cache == 147456 * 65536


def test_hybrid_qwen36_kv_and_state(settings):
    s = spec("qwen3.6-35b-a3b-nvfp4")
    est = estimate_vllm(s, 22 * 10**9, VllmSizing(131072, 8, 8192, "fp8", None), settings)
    assert est.kv_cache == 10 * 2 * 2 * 256 * 1 * 131072 * 8 == 10_737_418_240
    assert est.state == 30 * 548864 * 8 * 4


def test_gemma_sliding_layers_cap_at_window(settings):
    s = spec("gemma-4-31b-it")
    kv = kv_cache_bytes(s, tokens_total=32768 * 4, seqs=4, context=32768, bytes_per_elem=2, window_slack=8192)
    full = 10 * 2 * 4 * 512 * 2 * 32768 * 4
    sliding = 50 * 2 * 16 * 256 * 2 * (1024 * 4 + 8192)
    assert kv == full + sliding == 20_803_747_840


def test_draft_model_adds_weights_and_kv(settings):
    s, d = spec("qwen3.8-27b-nvfp4"), spec("qwen3.8-27b-dflash2")
    base = estimate_vllm(s, 22 * 10**9, VllmSizing(65536, 4, 8192, "fp8", None), settings)
    with_draft = estimate_vllm(s, 22 * 10**9, VllmSizing(65536, 4, 8192, "fp8", None), settings,
                               draft_spec=d, draft_bytes=3 * 10**9)
    assert with_draft.draft_weights == 3 * 10**9
    assert with_draft.kv_cache > base.kv_cache


def test_low_confidence_adds_20_percent_margin(settings):
    s = parse_hf_config({"architectures": ["X"], "hidden_size": 2048})
    est = estimate_vllm(s, 10**9, VllmSizing(4096, 1, 2048, "auto", None), settings)
    assert est.confidence == "low"
    assert est.margin == int(0.2 * (est.kv_cache + est.state + est.activations))


def test_state_bytes_and_dtype_table():
    assert kv_dtype_bytes("auto", "bfloat16") == 2 and kv_dtype_bytes("fp8", "bfloat16") == 1
    assert kv_dtype_bytes("q8_0", "bfloat16") == 34 / 32
    s = spec("qwen3-8b")
    assert state_bytes(s, 8) == 0


def test_gpu_util_rounds_up():
    assert vllm_gpu_util(50 * GIB, 100 * GIB) == 0.50
    assert vllm_gpu_util(50 * GIB + 1, 100 * GIB) == 0.51


def test_estimate_dict_roundtrip(settings):
    est = estimate_vllm(spec("qwen3-0.6b"), 1_500_000_000, VllmSizing(8192, 2, 2048, "auto", None), settings)
    assert MemoryEstimate.from_dict(est.as_dict()) == est
    assert est.as_dict()["total"] == est.total
