from types import SimpleNamespace

import pytest

from dgxmm.advisor.rules import AdviceInput, advise
from dgxmm.config import GIB
from dgxmm.deployer.service import config_hash

VCFG = {"max_model_len": 65536, "max_num_seqs": 8, "max_num_batched_tokens": 8192, "kv_cache_dtype": "auto",
        "enable_prefix_caching": True, "enable_chunked_prefill": True, "speculative": None, "kv_tokens": None}


def art(ref="Qwen/Qwen3-8B", quant="bf16", moe=False, files=(), layers=36, source="hf"):
    arch = SimpleNamespace(is_moe=moe, num_layers=layers, max_context=131072)
    return SimpleNamespace(ref=ref, quant=quant, arch=arch, files=list(files), source=source, kind="llm", config={})


def samples(n=200, **kw):
    base = {"running": 1.0, "waiting": 0.0, "kv_usage": 0.5, "ttft_p50": 0.2, "spec_accept_rate": None,
            "preemptions": 0.0, "prefix_hit_rate": 0.3}
    return [{**base, **kw} for _ in range(n)]


def ids(inp):
    return {a.id for a in advise(inp)}


def inp(**kw):
    d = dict(engine="vllm", cfg=dict(VCFG), artifact=art(), samples=samples(), day=samples(300), variants=[], drafts=[],
             benchmarks=[], estimate={"kv_cache": 20 * GIB, "weights": 16 * GIB}, reserved=40 * GIB)
    d.update(kw)
    return AdviceInput(**d)


def test_quant_kv_and_spec_rules():
    nv = art("nvidia/Qwen3-8B-NVFP4", "nvfp4")
    draft = art("org/Qwen3-8B-EAGLE3", "bf16")
    got = advise(inp(variants=[nv], drafts=[draft]))
    by = {a.id: a for a in got}
    assert by["nvfp4_variant"].model_ref == "nvidia/Qwen3-8B-NVFP4"
    assert by["fp8_kv"].patch == {"kv_cache_dtype": "fp8"}
    assert by["spec_decode"].patch["speculative"] == {"method": "eagle3", "model": "org/Qwen3-8B-EAGLE3", "num_speculative_tokens": 3}
    assert "spec_decode" in ids(inp(artifact=art(files=["model_mtp.safetensors"])))


def test_load_driven_rules():
    busy = samples(kv_usage=0.4, waiting=3.0, running=8.0)
    got_busy = {a.id: a for a in advise(inp(samples=busy))}
    assert got_busy["raise_seqs"].patch == {"max_num_seqs": 16} and "kv_pressure" not in got_busy
    hot = samples(kv_usage=0.99, preemptions=2.0, running=8.0)
    a = {x.id: x for x in advise(inp(samples=hot))}["kv_pressure"]
    assert a.severity == "warning" and a.patch == {"kv_cache_dtype": "fp8"}
    idle = samples(300, kv_usage=0.1, prefix_hit_rate=0.0)
    got = ids(inp(day=idle))
    assert {"shrink_reservation", "prefix_useless"} <= got


def test_spec_acceptance_low_and_moe_and_prefill():
    cfg = {**VCFG, "speculative": {"method": "eagle3", "model": "x/y", "num_speculative_tokens": 4}}
    a = {x.id: x for x in advise(inp(cfg=cfg, samples=samples(spec_accept_rate=0.3)))}["spec_accept_low"]
    assert a.patch == {"speculative": {"method": "eagle3", "model": "x/y", "num_speculative_tokens": 2}}
    assert "moe_batch" in ids(inp(artifact=art(moe=True)))
    assert "chunked_prefill_off" in ids(inp(cfg={**VCFG, "enable_chunked_prefill": False}))
    assert "batched_tokens_small" in ids(inp(cfg={**VCFG, "max_num_batched_tokens": 2048}, samples=samples(ttft_p50=2.0)))


def test_llamacpp_rules_and_engine_hint():
    lcfg = {"ctx_size": 65536, "parallel": 4, "n_gpu_layers": 20, "cache_type_k": "f16", "cache_type_v": "f16",
            "flash_attn": "off", "n_batch": 2048, "n_ubatch": 512}
    got = ids(inp(engine="llamacpp", cfg=lcfg, artifact=art(source="gguf", quant="gguf:Q4_K_M")))
    assert {"llamacpp_offload_all", "llamacpp_flash_attn", "llamacpp_quant_kv"} <= got
    gguf = art("/data/gguf/0/Qwen3-8B-Q4_K_M.gguf", "gguf:Q4_K_M", source="gguf")
    assert "try_llamacpp" in ids(inp(samples=samples(running=1.0), variants=[gguf]))


def test_measured_benchmarks_override_rule_estimate():
    patched = {**VCFG, "kv_cache_dtype": "fp8"}
    rows = [{"config_hash": config_hash(VCFG), "concurrency": 1, "tps_per_req": 20.0},
            {"config_hash": config_hash(patched), "concurrency": 1, "tps_per_req": 31.0}]
    a = {x.id: x for x in advise(inp(benchmarks=rows))}["fp8_kv"]
    assert a.confidence == "measured" and "31" in a.expected_effect and "20" in a.expected_effect


def test_measured_benchmarks_ignore_failed_zero_rows():
    """Bug B: the newest row for a config can be a failed (requests=0, tps=0) level; it must be skipped."""
    patched = {**VCFG, "kv_cache_dtype": "fp8"}
    rows = [{"config_hash": config_hash(patched), "concurrency": 1, "tps_per_req": 0.0, "requests": 0},  # newest, failed
            {"config_hash": config_hash(VCFG), "concurrency": 1, "tps_per_req": 0.0, "requests": 0},  # newest, failed
            {"config_hash": config_hash(VCFG), "concurrency": 1, "tps_per_req": 20.0, "requests": 8},
            {"config_hash": config_hash(patched), "concurrency": 1, "tps_per_req": 31.0, "requests": 8}]
    a = {x.id: x for x in advise(inp(benchmarks=rows))}["fp8_kv"]
    assert a.confidence == "measured" and "31" in a.expected_effect and "20" in a.expected_effect
    # only failed rows for the patched config -> no "measured" claim at all
    only_bad = [rows[0], rows[2]]
    b = {x.id: x for x in advise(inp(benchmarks=only_bad))}["fp8_kv"]
    assert b.confidence != "measured" and "measured on this box" not in b.expected_effect
