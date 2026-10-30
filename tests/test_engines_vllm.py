import json

import httpx
import pytest
from pydantic import ValidationError

from dgxmm.config import GIB
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.prom import buckets, histogram_quantile, parse_samples
from dgxmm.engines.vllm import VllmEngine
from dgxmm.inventory.hf_scan import scan_hf_cache
from pathlib import Path
from tests.conftest import QWEN_TINY

VLLM_METRICS = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="q"} 2.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="q"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="q"} 0.25
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{engine="0",model_name="q"} 1000.0
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{engine="0",model_name="q"} 500.0
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{engine="0",model_name="q"} 200.0
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="q"} 50.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="q"} 3.0
vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="q"} 9.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="q"} 10.0
vllm:time_to_first_token_seconds_count{model_name="q"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="q"} 4.2
"""


@pytest.fixture
def tiny(settings, fake_hf_repo):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    [a] = scan_hf_cache(Path(settings.hf_cache.local) / "hub")
    return a


def test_argguard():
    assert validate_extra_args(["--enforce-eager", "--seed=42", "--max-logprobs", "5"], VLLM_EXTRA_FLAGS) == \
        ["--enforce-eager", "--seed", "42", "--max-logprobs", "5"]
    for bad in (["--model", "x"], ["--seed", "1;rm -rf /"], ["--gpu-memory-utilization", "0.9"],
                ["--enforce-eager=1"], ["--seed"], ["positional"]):
        with pytest.raises(ValueError):
            validate_extra_args(bad, VLLM_EXTRA_FLAGS)


def test_config_validation(settings):
    e = VllmEngine(settings)
    cfg = e.validate_config({"max_model_len": 8192, "kv_cache_dtype": "fp8"})
    assert cfg.max_num_seqs == 8 and cfg.enable_sleep_mode is True
    for bad in ({"gpu_memory_utilization": 0.9}, {"extra_args": ["--port", "1"]}, {"served_model_name": "a b"},
                {"max_model_len": 8192, "kv_tokens": 1024}):
        with pytest.raises((ValidationError, ValueError)):
            e.validate_config(bad)


def test_container_spec(settings, tiny):
    e = VllmEngine(settings)
    cfg = e.validate_config({"max_model_len": 4096, "max_num_seqs": 2, "kv_cache_dtype": "fp8",
                             "reasoning_parser": "qwen3", "speculative": {"method": "mtp", "num_speculative_tokens": 2},
                             "extra_args": ["--enforce-eager"]})
    spec = e.container_spec(tiny, cfg, None, LaunchContext(7, "q", 8105, 30 * GIB, 120 * GIB), settings)
    c = spec.command
    assert spec.name == "dgxmm-q" and spec.image == settings.images.vllm and spec.port == 8105
    assert c[c.index("--model") + 1] == "Qwen/Qwen3-Tiny" and c[c.index("--revision") + 1] == tiny.revision
    assert c[c.index("--gpu-memory-utilization") + 1] == "0.25"
    assert c[c.index("--served-model-name") + 1] == "qwen3-tiny"
    assert json.loads(c[c.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 2}
    assert "--enable-sleep-mode" in c and c[-1] == "--enforce-eager"
    assert spec.environment["VLLM_SERVER_DEV_MODE"] == "1" and spec.environment["HF_HUB_OFFLINE"] == "1"
    assert spec.labels == {"dgxmm.managed": "true", "dgxmm.deployment_id": "7", "dgxmm.engine": "vllm"}
    assert spec.mounts[0].host == settings.hf_cache.host and spec.mounts[0].target == "/root/.cache/huggingface"
    assert spec.publish_bind == "127.0.0.1" and spec.network == settings.docker_network == "dgxmm"



def test_container_spec_pins_draft_revision(settings, fake_hf_repo):
    # a pinned download (DGXMM's) has snapshots/<rev> and no refs/main: under HF_HUB_OFFLINE vLLM can resolve the
    # draft only through its revision, as it does the target's through --revision
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    fake_hf_repo("Org/Tiny-DFlash", rev="b" * 40, config=QWEN_TINY, files={"model.safetensors": GIB}, main=False)
    arts = {a.ref: a for a in scan_hf_cache(Path(settings.hf_cache.local) / "hub")}
    target, draft = arts["Qwen/Qwen3-Tiny"], arts["Org/Tiny-DFlash"]
    assert draft.revision == "b" * 40
    e = VllmEngine(settings)
    cfg = e.validate_config({"max_model_len": 4096,
                             "speculative": {"method": "dflash", "model": "Org/Tiny-DFlash", "num_speculative_tokens": 3}})
    c = e.container_spec(target, cfg, draft, LaunchContext(7, "q", 8105, 30 * GIB, 120 * GIB), settings).command
    assert json.loads(c[c.index("--speculative-config") + 1]) == {
        "method": "dflash", "num_speculative_tokens": 3, "model": "Org/Tiny-DFlash", "revision": "b" * 40}

def test_parse_metrics(settings):
    m = VllmEngine(settings).parse_metrics(VLLM_METRICS)
    assert m.gauges == {"running": 2.0, "waiting": 1.0, "kv_usage": 0.25}
    assert m.counters["generation_tokens"] == 500.0 and m.counters["prefix_hits"] == 50.0
    assert m.histograms["ttft"] == [(0.1, 3.0), (1.0, 9.0), (float("inf"), 10.0)]
    q = histogram_quantile(m.histograms["ttft"], 0.5)
    assert 0.1 < q < 1.0


def test_histogram_quantile_edges():
    assert histogram_quantile([], 0.5) is None
    assert histogram_quantile([(1.0, 0.0), (float("inf"), 0.0)], 0.5) is None
    assert histogram_quantile([(1.0, 10.0), (float("inf"), 10.0)], 0.5) == pytest.approx(0.5)


async def test_health_checks_health_then_completion(settings):
    seen = []

    def handler(req: httpx.Request):
        seen.append(req.url.path)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await VllmEngine(settings).health(http, "http://h:1", "q")
    assert seen == ["/health", "/v1/completions"]
