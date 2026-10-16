from pathlib import Path

from dgxmm.catalog.support import VllmArchRegistry, engine_support
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.vllm import VllmEngine
from dgxmm.inventory.hf_scan import scan_hf_cache
from tests.conftest import QWEN_TINY
from tests.fakes import FakeDockerClient


async def test_registry_extracts_once_and_caches(settings):
    sdk = FakeDockerClient()
    sdk.oneshot_output = "INFO noise\nARCHS=[\"LlamaForCausalLM\", \"Qwen3ForCausalLM\"]\n"
    reg = VllmArchRegistry(settings, DockerController(sdk))
    assert reg.get() is None
    assert await reg.ensure() == {"LlamaForCausalLM", "Qwen3ForCausalLM"}
    assert sdk.run_calls[-1]["entrypoint"][0] == "python3" and sdk.run_calls[-1]["image"] == settings.images.vllm
    again = VllmArchRegistry(settings, DockerController(FakeDockerClient()))
    assert again.get() == {"LlamaForCausalLM", "Qwen3ForCausalLM"}  # read from data_dir cache


async def test_registry_failure_is_none(settings):
    sdk = FakeDockerClient()
    sdk.oneshot_output = "Traceback: boom"
    assert await VllmArchRegistry(settings, DockerController(sdk)).ensure() is None


def test_engine_support_and_vllm_gate(settings, fake_hf_repo):
    assert engine_support("Qwen3ForCausalLM", False, None) == {"vllm": None, "llamacpp": False}
    assert engine_support("Qwen3ForCausalLM", False, {"Qwen3ForCausalLM"}) == {"vllm": True, "llamacpp": False}
    assert engine_support("gguf:qwen3", True, {"Qwen3ForCausalLM"}) == {"vllm": False, "llamacpp": True}
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    [a] = scan_hf_cache(Path(settings.hf_cache.local) / "hub")
    e = VllmEngine(settings)
    assert e.supports(a)
    e.arch_source = lambda: {"LlamaForCausalLM"}
    assert not e.supports(a)
    e.arch_source = lambda: None
    assert e.supports(a)
