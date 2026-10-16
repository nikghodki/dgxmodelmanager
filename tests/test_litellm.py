import json
import uuid

import httpx

from dgxmm.deployer.service import DeployRequest
from dgxmm.litellm import LiteLLMClient, LiteLLMSync

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


class FakeLiteLLM:
    def __init__(self):
        self.models = [{"model_name": "hand-written", "litellm_params": {"model": "openai/x"}, "model_info": {"id": "keep-me"}}]
        self.down = False
        self.auth = []

    def handler(self, req: httpx.Request):
        self.auth.append(req.headers.get("authorization"))
        if self.down:
            raise httpx.ConnectError("down")
        if req.url.path == "/model/info":
            return httpx.Response(200, json={"data": self.models})
        body = json.loads(req.content)
        if req.url.path == "/model/new":
            self.models.append({**body, "model_info": {**body["model_info"], "id": str(uuid.uuid4())}})
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/model/delete":
            self.models = [m for m in self.models if m["model_info"]["id"] != body["id"]]
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def make(stack, fake):
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return LiteLLMSync(LiteLLMClient("http://litellm:4000", "sk-test", http), stack.deployer, stack.db, stack.events)


async def ready(stack, name, alias=None):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=SMALL, alias=alias), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    return d


async def test_registers_served_name_and_alias_then_removes(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    mine = sorted(m["model_name"] for m in fake.models if m["model_info"].get("dgxmm_id"))
    assert mine == ["chat", "qwen3-tiny"]
    entry = next(m for m in fake.models if m["model_name"] == "chat")
    assert entry["litellm_params"] == {"model": "openai/qwen3-tiny", "api_base": f"http://engines.test:{d['port']}/v1", "api_key": "none"}
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"
    assert set(fake.auth) == {"Bearer sk-test"}
    await stack.deployer.stop(d["id"])
    assert await sync.sync()
    assert [m["model_name"] for m in fake.models] == ["hand-written"]  # never touches unmarked entries


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
