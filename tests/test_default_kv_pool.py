"""I6: default vLLM KV pool = min(max_model_len x max_num_seqs, 2 x max_model_len) when kv_tokens is unset.

Decision (recorded in the fix report): the pure estimator keeps kv_tokens=None meaning
max_model_len x max_num_seqs; the default pool is applied at the engine level (VllmEngine.sizing)."""
from types import SimpleNamespace

from dgxmm.advisor.rules import AdviceInput, shrink_reservation
from dgxmm.catalog.service import CatalogService
from dgxmm.config import GIB
from dgxmm.engines.vllm import VllmEngine
from dgxmm.memory.estimate import VllmSizing, default_kv_tokens, estimate_vllm
from tests.test_catalog import FakeHub, FakeLlmfit

SMALL = {"max_model_len": 4096, "max_num_seqs": 8, "max_num_batched_tokens": 2048}


def test_default_kv_tokens():
    assert default_kv_tokens(32768, 8) == 65536
    assert default_kv_tokens(32768, 1) == 32768


def test_engine_sizing_applies_default_pool_and_keeps_explicit(settings):
    e = VllmEngine(settings)
    assert e.sizing(e.validate_config(SMALL)).kv_tokens == 2 * 4096
    assert e.sizing(e.validate_config({**SMALL, "max_num_seqs": 1})).kv_tokens == 4096
    assert e.sizing(e.validate_config({**SMALL, "kv_tokens": 20000})).kv_tokens == 20000
    assert e.pool_tokens(e.validate_config(SMALL)) == 8192


async def test_preview_uses_default_pool_and_reports_it(stack):
    p = await stack.deployer.preview("vllm", "Qwen/Qwen3-Tiny", SMALL)
    a = stack.inv.get("Qwen/Qwen3-Tiny")
    pooled = estimate_vllm(a.arch, a.weights_bytes, VllmSizing(4096, 8, 2048, "auto", 8192), stack.settings)
    full = estimate_vllm(a.arch, a.weights_bytes, VllmSizing(4096, 8, 2048, "auto", None), stack.settings)
    assert p["estimate"]["kv_cache"] == pooled.kv_cache < full.kv_cache
    assert p["kv_pool_tokens"] == 8192


async def test_catalog_details_uses_default_pool(stack):
    cat = CatalogService(stack.settings, FakeHub(), FakeLlmfit(), stack.inv, stack.budget, stack.deployer,
                         SimpleNamespace(get=lambda: {"Qwen3ForCausalLM"}))
    d = await cat.details("Qwen/Qwen3-Tiny", ctx=4096)
    a = stack.inv.get("Qwen/Qwen3-Tiny")
    pooled = estimate_vllm(a.arch, d["estimate"]["weights"], VllmSizing(4096, 8, 8192, "auto", 8192), stack.settings)
    assert d["estimate"]["kv_cache"] == pooled.kv_cache


def test_shrink_advice_halves_the_default_pool():
    cfg = {"max_model_len": 65536, "max_num_seqs": 8, "kv_tokens": None}
    day = [{"kv_usage": 0.1}] * 100
    i = AdviceInput(engine="vllm", cfg=cfg, artifact=None, samples=[], day=day, variants=[], drafts=[], benchmarks=[],
                    estimate={"kv_cache": 20 * GIB}, reserved=40 * GIB)
    assert shrink_reservation(i).patch == {"kv_tokens": 65536}  # max(len, 2*len/2), not 8*len/2
