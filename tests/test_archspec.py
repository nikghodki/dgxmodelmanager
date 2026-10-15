import json
from pathlib import Path

import pytest

from dgxmm.inventory.archspec import LayerGroup, detect_quant, parse_hf_config

FIX = Path(__file__).parent / "fixtures" / "configs"


def load(name):
    return parse_hf_config(json.loads((FIX / f"{name}.json").read_text()))


def groups(spec):
    return {g.kind: g for g in spec.layer_groups}


def test_dense_qwen3_8b():
    s = load("qwen3-8b")
    assert s.layer_groups == (LayerGroup("full", 36, kv_heads=8, head_dim=128),)
    assert s.quant == "bf16" and s.max_context == 40960 and s.confident and not s.is_moe


def test_granite_derives_head_dim_without_flagging():
    s = load("granite-4.2-30b")
    assert s.layer_groups == (LayerGroup("full", 64, kv_heads=8, head_dim=128),)
    assert s.confident


def test_hybrid_moe_qwen36():
    s = load("qwen3.6-35b-a3b-nvfp4")
    g = groups(s)
    assert g["full"] == LayerGroup("full", 10, kv_heads=2, head_dim=256)
    assert g["linear"].count == 30
    # conv (4-1)*(2*16*128 + 32*128) + recurrent 32*128*128
    assert g["linear"].state_elems == 3 * (2 * 2048 + 4096) + 32 * 128 * 128 == 548864
    assert s.is_moe and s.num_experts == 256 and s.experts_per_tok == 8
    assert s.intermediate_size == 512 * 8 + 512
    assert s.quant == "nvfp4"


def test_gemma4_sliding_and_global_layers():
    s = load("gemma-4-31b-it")
    g = groups(s)
    assert g["sliding"] == LayerGroup("sliding", 50, kv_heads=16, head_dim=256, window=1024)
    assert g["full"] == LayerGroup("full", 10, kv_heads=4, head_dim=512)


def test_qwen38_nvfp4_and_draft():
    s = load("qwen3.8-27b-nvfp4")
    g = groups(s)
    assert g["full"].count == 16 and g["full"].kv_heads == 4 and g["linear"].count == 48
    assert g["linear"].state_elems == 3 * (2 * 2048 + 6144) + 48 * 128 * 128
    assert s.quant == "nvfp4"
    d = load("qwen3.8-27b-dflash2")
    assert d.architecture == "DFlash2DraftModel"
    assert groups(d)["sliding"].window == 2048 and groups(d)["sliding"].count == 5


def test_mla_and_mamba_synthetic():
    mla = parse_hf_config({"architectures": ["DeepseekV3ForCausalLM"], "num_hidden_layers": 4, "hidden_size": 1024,
                           "num_attention_heads": 8, "vocab_size": 1000, "max_position_embeddings": 8192,
                           "kv_lora_rank": 512, "qk_rope_head_dim": 64, "intermediate_size": 2048})
    assert mla.layer_groups == (LayerGroup("mla", 4, mla_dim=576),)
    mamba = parse_hf_config({"architectures": ["GraniteMoeHybridForCausalLM"], "num_hidden_layers": 4, "hidden_size": 1024,
                             "num_attention_heads": 8, "num_key_value_heads": 2, "vocab_size": 1000,
                             "max_position_embeddings": 8192, "intermediate_size": 2048,
                             "layer_types": ["mamba", "mamba", "mamba", "attention"],
                             "mamba_n_heads": 16, "mamba_d_head": 64, "mamba_d_state": 128, "mamba_n_groups": 1,
                             "mamba_d_conv": 4})
    g = groups(mamba)
    assert g["full"].count == 1 and g["mamba"].count == 3
    assert g["mamba"].state_elems == 16 * 64 * 128 + 3 * (16 * 64 + 2 * 1 * 128)


@pytest.mark.parametrize("cfg", [
    {},
    {"architectures": ["LlavaForConditionalGeneration"], "text_config": {"hidden_size": 4096}},
    {"num_hidden_layers": 2, "layer_types": ["weird", "full_attention"]},
])
def test_sparse_configs_never_raise_and_flag_low_confidence(cfg):
    s = parse_hf_config(cfg)
    assert s.num_layers > 0 and s.layer_groups
    assert s.confident is False


def test_detect_quant_variants():
    assert detect_quant(None, "float16") == "fp16"
    assert detect_quant({"quant_method": "awq"}, "float16") == "awq"
    assert detect_quant({"quant_method": "modelopt", "quant_algo": "FP8"}, "bfloat16") == "fp8"
    assert detect_quant({"quant_method": "compressed-tensors",
                         "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}}, "bfloat16") == "fp8"
    assert detect_quant({"quant_method": "gptq", "bits": 4}, "float16") == "gptq"
