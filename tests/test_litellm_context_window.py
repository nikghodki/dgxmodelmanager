"""Each deployment's context window reaches LiteLLM as model_info.max_input_tokens (coding tools such as Claude
Code size their compaction from /model/info): vLLM's max_model_len, a llama.cpp slot's share of -c (all of it with
--kv-unified); a changed window re-registers the entry once, adding before removing; a steady one is never rewritten."""
from dgxmm.engines.llamacpp import LlamaCppEngine
from dgxmm.engines.vllm import VllmEngine
from dgxmm.litellm import window_info
from tests.fakes import FakeLiteLLM
from tests.test_litellm_ondemand import SMALL, key, make, mine, ready, writes  # noqa: F401  (key: fixture)


def test_engine_context_windows(settings):
    v = VllmEngine(settings)
    assert v.context_window(v.validate_config({"max_model_len": 524288})) == 524288
    l = LlamaCppEngine(settings)
    assert l.context_window(l.validate_config({"ctx_size": 131072, "parallel": 4})) == 32768
    assert l.context_window(l.validate_config({"ctx_size": 131072, "parallel": 4, "extra_args": ["--kv-unified"]})) == 131072


def test_window_info_only_for_a_known_window():
    assert window_info({"context_window": 4096}) == {"max_input_tokens": 4096}
    assert window_info({"context_window": None}) == {} and window_info({}) == {}


async def test_view_and_both_modes_register_the_window(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    a = await ready(stack, "a", "on_demand", alias="chat")
    await ready(stack, "b", "always_on", alias="steady", config={**SMALL, "max_model_len": 2048})
    assert stack.deployer.view(a["id"])["context_window"] == 4096
    assert await sync.sync()
    by_name = {m["model_name"]: m for m in mine(fake)}
    assert by_name["chat"]["model_info"]["max_input_tokens"] == 4096
    assert by_name["steady"]["model_info"]["max_input_tokens"] == 2048
    before = writes(fake)
    for _ in range(3):
        assert await sync.sync()
    assert writes(fake) == before  # a steady window is never rewritten


async def test_changed_window_reregisters_once_adding_first(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    a = await ready(stack, "a", "on_demand", alias="chat")
    b = await ready(stack, "b", "always_on", alias="steady")
    assert await sync.sync()
    for dep_id in (a["id"], b["id"]):
        await stack.deployer.reconfigure(dep_id, {**SMALL, "max_model_len": 2048})
    n = len(fake.calls)
    assert await sync.sync()
    adds_and_deletes = [p for _, p in fake.calls[n:] if p in ("/model/new", "/model/delete")]
    assert adds_and_deletes.index("/model/delete") > adds_and_deletes.index("/model/new")
    assert {m["model_name"]: m["model_info"]["max_input_tokens"] for m in mine(fake)} == {"chat": 2048, "steady": 2048}
    before = writes(fake)
    assert await sync.sync() and writes(fake) == before


async def test_entry_registered_before_windows_gets_one_replacement(stack, key):
    """An always-on entry from an older DGXMM (no max_input_tokens) is replaced once."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    await ready(stack, "b", "always_on", alias="steady")
    assert await sync.sync()
    mine(fake)[0]["model_info"].pop("max_input_tokens")
    assert await sync.sync()
    assert mine(fake)[0]["model_info"]["max_input_tokens"] == 4096
    before = writes(fake)
    assert await sync.sync() and writes(fake) == before
