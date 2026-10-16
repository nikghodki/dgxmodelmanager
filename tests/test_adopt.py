import json

import pytest

from dgxmm.deployer.adopt import adopt, list_adoptable, parse_vllm_command
from dgxmm.deployer.service import DeployError

# The real command of dmm-chat-inference on dgx-host (2026-10-15)
DMM = ["--model", "unsloth/Qwen3.8-27B-NVFP4", "--served-model-name", "qwen3.8-27b", "--host", "0.0.0.0", "--port", "8000",
       "--max-model-len", "256000", "--gpu-memory-utilization", "0.50", "--trust-remote-code", "--enable-prefix-caching",
       "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
       "--enable-chunked-prefill", "--kv-cache-dtype", "fp8", "--max-num-batched-tokens", "16384", "--max-num-seqs", "8",
       "--speculative-config", json.dumps({"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7})]


def test_parse_real_command():
    ref, cfg, ignored = parse_vllm_command(DMM)
    assert ref == "unsloth/Qwen3.8-27B-NVFP4"
    assert cfg == {"served_model_name": "qwen3.8-27b", "max_model_len": 256000, "trust_remote_code": True,
                   "enable_prefix_caching": True, "enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder",
                   "reasoning_parser": "qwen3", "enable_chunked_prefill": True, "kv_cache_dtype": "fp8",
                   "max_num_batched_tokens": 16384, "max_num_seqs": 8, "enable_sleep_mode": False,
                   "speculative": {"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}}
    assert set(ignored) == {"--host", "--port", "--gpu-memory-utilization"}


def test_parse_variants():
    ref, cfg, _ = parse_vllm_command(["vllm", "serve", "nvidia/X", "--safetensors-load-strategy=prefetch", "--enable-sleep-mode"])
    assert ref == "nvidia/X" and cfg["safetensors_load_strategy"] == "prefetch" and cfg["enable_sleep_mode"] is True
    _, cfg, _ = parse_vllm_command(["--model", "a/b", "--enforce-eager", "--no-enable-prefix-caching"])
    assert cfg["extra_args"] == ["--enforce-eager"] and cfg["enable_prefix_caching"] is False
    with pytest.raises(DeployError) as e:
        parse_vllm_command(["--model", "a/b", "--tensor-parallel-size", "2", "--weird-flag"])
    assert e.value.code == "unsupported_flags" and "--weird-flag" in e.value.message


async def test_adopt_creates_stopped_row_and_leaves_container(stack):
    c = stack.sdk.add_unmanaged("vllm-qwen", image="vllm/vllm-openai:pinned-2026-08-27",
                                command=["--model", "Qwen/Qwen3-Tiny", "--max-model-len", "4096", "--enable-sleep-mode"],
                                status="created")
    stack.sdk.add_unmanaged("litellm-proxy", image="ghcr.io/berriai/litellm:main-latest")
    assert [a["name"] for a in list_adoptable(stack.deployer)] == ["vllm-qwen"]
    v = await adopt(stack.deployer, "vllm-qwen", "qwen-tiny", 1)
    assert v["state"] == "stopped" and v["adopted_from"] == "vllm-qwen" and v["config"]["max_model_len"] == 4096
    assert stack.sdk.container(c.id).status == "created"  # untouched
    assert list_adoptable(stack.deployer)[0]["adopted_as"] == "qwen-tiny"
    with pytest.raises(DeployError):
        await adopt(stack.deployer, "litellm-proxy", "x", 1)


def test_list_adoptable_marks_non_llm_and_unparseable(stack):
    """M3: only containers whose command parses and whose model is a downloaded LLM are adoptable."""
    stack.sdk.add_unmanaged("vllm-qwen", command=["--model", "Qwen/Qwen3-Tiny"])
    stack.sdk.add_unmanaged("whisper", command=["--model", "openai/whisper-large-v3", "--task", "transcription"])
    stack.sdk.add_unmanaged("whisper2", command=["--model", "openai/whisper-large-v3"])
    stack.sdk.add_unmanaged("weird", command=["--max-model-len", "32k"])
    rows = {r["name"]: r for r in list_adoptable(stack.deployer)}
    assert rows["vllm-qwen"]["adoptable"] is True and rows["vllm-qwen"]["reason"] is None
    assert rows["whisper"]["adoptable"] is False and "--task" in rows["whisper"]["reason"]
    assert rows["whisper2"]["adoptable"] is False and "not a downloaded LLM" in rows["whisper2"]["reason"]
    assert rows["weird"]["adoptable"] is False and "--model" in rows["weird"]["reason"]


def test_parse_human_readable_ints_and_bad_values():
    """M5: vLLM accepts 32k (decimal) / 32K (binary); garbage is a 422, not a 500."""
    _, cfg, _ = parse_vllm_command(["--model", "a/b", "--max-model-len", "32k", "--max-num-batched-tokens=16K"])
    assert cfg["max_model_len"] == 32000 and cfg["max_num_batched_tokens"] == 16384
    for bad in (["--model", "a/b", "--max-model-len", "lots"], ["--model", "a/b", "--max-num-seqs"],
                ["--model", "a/b", "--speculative-config", "{not json"],
                ["--model", "a/b", "--speculative-config", "[1, 2]"]):
        with pytest.raises(DeployError) as e:
            parse_vllm_command(bad)
        assert e.value.status == 422 and e.value.code == "invalid_value", bad


async def test_adopt_route_422_on_bad_args(api_stack):
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    api_stack.stack.sdk.add_unmanaged("vllm-bad", command=["--model", "Qwen/Qwen3-Tiny", "--max-model-len", "huge"])
    r = api_stack.post("/dgxmm/api/containers/vllm-bad/adopt", json={"name": "bad"}, headers=h)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_value"
