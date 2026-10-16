from pathlib import Path

import pytest
from pydantic import ValidationError

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.llamacpp import LlamaCppEngine
from dgxmm.inventory.gguf_scan import scan_gguf_roots
from dgxmm.memory.estimate import LlamaCppSizing, estimate_llamacpp
from tests.conftest import write_gguf

LLAMA_METRICS = """# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 120
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 80
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
"""


@pytest.fixture
def gguf_art(settings):
    root = Path(settings.gguf_roots[0].local)
    write_gguf(root / "qwen" / "Qwen3-8B-Q4_K_M.gguf", pad_bytes=5 * GIB)
    [a] = scan_gguf_roots([root])
    return a


def test_estimate_exact(settings, gguf_art):
    est = estimate_llamacpp(gguf_art.arch, gguf_art.weights_bytes, LlamaCppSizing(32768, 4, "f16", "f16", 512), settings)
    assert est.kv_cache == 147456 * 32768
    assert est.activations == 512 * 151936 * 4 + 512 * 4096 * 4 * 8 + 256 * 1024**2
    assert est.runtime == int(0.5 * GIB) and est.weights == gguf_art.weights_bytes
    q8 = estimate_llamacpp(gguf_art.arch, 0, LlamaCppSizing(32768, 4, "q8_0", "q8_0", 512), settings)
    assert q8.kv_cache == int(147456 * 32768 * 34 / 64 + 0.999999)


def test_config_validation(settings):
    e = LlamaCppEngine(settings)
    assert e.validate_config({}).ctx_size == 32768
    for bad in ({"cache_type_v": "q8_0", "flash_attn": "off"}, {"n_ubatch": 4096, "n_batch": 1024},
                {"extra_args": ["--model", "x"]}, {"alias": "a b"}):
        with pytest.raises((ValidationError, ValueError)):
            e.validate_config(bad)


def test_container_spec_maps_paths(settings, gguf_art):
    e = LlamaCppEngine(settings)
    cfg = e.validate_config({"ctx_size": 16384, "parallel": 2, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"})
    spec = e.container_spec(gguf_art, cfg, None, LaunchContext(3, "q", 8110, 10 * GIB, 120 * GIB), settings)
    c = spec.command
    assert c[c.index("-m") + 1] == "/models/0/qwen/Qwen3-8B-Q4_K_M.gguf"
    assert spec.mounts[0].host == settings.gguf_roots[0].host and spec.mounts[0].target == "/models/0" and spec.mounts[0].mode == "ro"
    assert c[c.index("-c") + 1] == "16384" and c[c.index("-np") + 1] == "2" and "--metrics" in c
    assert c[c.index("-a") + 1] == "qwen3-8b-q4_k_m" and spec.image == settings.images.llamacpp


def test_metrics_and_engine_flags(settings):
    e = LlamaCppEngine(settings)
    m = e.parse_metrics(LLAMA_METRICS)
    assert m.counters == {"prompt_tokens": 120.0, "generation_tokens": 80.0}
    assert m.gauges == {"running": 1.0, "waiting": 0.0}
    assert e.supports_sleep is False and e.kv_patch_options(e.validate_config({})) == [
        {"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "auto"}]


async def test_deploy_llamacpp_end_to_end(stack, settings):
    write_gguf(Path(settings.gguf_roots[0].local) / "Qwen3-8B-Q4_K_M.gguf", pad_bytes=1 * GIB)
    stack.inv.refresh()
    ref = next(a.ref for a in stack.inv.list() if a.source == "gguf")
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="g", engine="llamacpp", model_ref=ref,
                                                  config={"ctx_size": 8192, "parallel": 2}), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    assert stack.sdk.run_calls[-1]["image"] == settings.images.llamacpp
