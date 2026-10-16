import httpx

from dgxmm.deployer.service import DeployRequest
from dgxmm.litellm import LiteLLMClient, LiteLLMSync
from tests.fakes import FakeLiteLLM  # noqa: F401  (re-exported for other tests)

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def make(stack, fake):
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return LiteLLMSync(LiteLLMClient.fixed("http://litellm:4000", "sk-test", http), stack.deployer, stack.db, stack.events)


async def ready(stack, name, alias=None):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=SMALL, alias=alias), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    return d


async def test_registers_once_under_alias_then_removes(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    mine = sorted(m["model_name"] for m in fake.models if m["model_info"].get("dgxmm_id"))
    assert mine == ["chat"]  # single endpoint: one registration per deployment, under its model name
    entry = next(m for m in fake.models if m["model_name"] == "chat")
    assert entry["litellm_params"] == {"model": "openai/qwen3-tiny", "api_base": f"http://engines.test:{d['port']}/v1", "api_key": "none"}
    assert entry["model_info"]["dgxmm_id"] == str(d["id"])
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"
    assert set(fake.auth) == {"Bearer sk-test"}
    assert sync.last["registered"] == 1 and sync.last["removed"] == 0
    await stack.deployer.stop(d["id"])
    assert await sync.sync()
    assert [m["model_name"] for m in fake.models] == ["hand-written"]  # never touches unmarked entries
    assert sync.last["removed"] == 1


async def test_without_alias_registers_served_name(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    await ready(stack, "a")
    assert await sync.sync()
    assert sorted(m["model_name"] for m in fake.models if m["model_info"].get("dgxmm_id")) == ["qwen3-tiny"]


async def test_migration_removes_second_name(stack):
    """Older DGXMM registered served_name AND alias; sync keeps only the canonical (alias) entry."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", alias="chat")
    base = f"http://engines.test:{d['port']}/v1"
    for name in ("qwen3-tiny", "chat"):
        fake.models.append({"model_name": name, "litellm_params": {"model": "openai/qwen3-tiny", "api_base": base, "api_key": "none"},
                            "model_info": {"dgxmm_id": str(d["id"]), "id": f"old-{name}"}})
    assert await sync.sync()
    mine = [(m["model_name"], m["model_info"]["id"]) for m in fake.models if m["model_info"].get("dgxmm_id")]
    assert mine == [("chat", "old-chat")]  # the alias entry is kept as-is, the served-name entry deleted
    assert sync.last == {**sync.last, "registered": 0, "removed": 1}


async def test_alias_change_reregisters_once_and_removes_old_name(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", alias="chat")
    await sync.sync()
    stack.deployer.set_alias(d["id"], "chat-v2")
    adds_before = sum(1 for c in fake.calls if c[1] == "/model/new")
    assert await sync.sync()
    assert sorted(m["model_name"] for m in fake.models if m["model_info"].get("dgxmm_id")) == ["chat-v2"]
    assert sum(1 for c in fake.calls if c[1] == "/model/new") == adds_before + 1
    assert await sync.sync()  # idempotent
    assert sum(1 for c in fake.calls if c[1] == "/model/new") == adds_before + 1


async def test_idempotent_and_dedupes(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    await ready(stack, "a")
    await sync.sync()
    fake.models.append({**fake.models[-1], "model_info": {**fake.models[-1]["model_info"], "id": "dup"}})
    await sync.sync()
    await sync.sync()
    assert sum(1 for m in fake.models if m["model_name"] == "qwen3-tiny") == 1


async def test_litellm_down_marks_pending_and_recovers(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a")
    fake.down = True
    assert not await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "pending"
    fake.down = False
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"


async def test_no_key_means_disabled(stack):
    sync = LiteLLMSync(None, stack.deployer, stack.db, stack.events)
    d = await ready(stack, "a")
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "disabled"
