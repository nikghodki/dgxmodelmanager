"""Model names LiteLLM already serves from entries DGXMM did not create (static config.yaml entries, or DB entries
another tool added): deploy / start / wake / rename refuse them (409 model_name_taken), the sync loop never
registers next to them and flags existing registrations as conflicts (status + insight), and an unreadable model
list keeps the old behaviour (fail open)."""
import httpx
import pytest

from dgxmm.decider.insights import InsightService
from dgxmm.deployer.service import DeployError, DeployRequest
from dgxmm.litellm import LiteLLMClient, LiteLLMSync, api_base_host
from tests.fakes import FakeLiteLLM

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
FOREIGN_TEXT = "that DGXMM didn't create (static config or another tool)"


def static(name, api_base="http://old-engine:8000/v1", mid="static-1"):
    """A config.yaml entry as /model/info reports it (no dgxmm_id, db_model false)."""
    return {"model_name": name, "litellm_params": {"model": f"openai/{name}", "api_base": api_base},
            "model_info": {"id": mid, "db_model": False}}


def other_tool(name, api_base="http://elsewhere:9000/v1", mid="other-1"):
    """A DB entry some other tool added through /model/new (db_model true, but no dgxmm_id)."""
    return {"model_name": name, "litellm_params": {"model": f"openai/{name}", "api_base": api_base},
            "model_info": {"id": mid, "db_model": True}}


def make(stack, fake, insights=None):
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return LiteLLMSync(LiteLLMClient.fixed("http://litellm:4000", "sk-test", http), stack.deployer, stack.db,
                       stack.events, insights=insights)


def req(name, alias=None, **cfg):
    return DeployRequest(mode="always_on", name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config={**SMALL, **cfg}, alias=alias)


async def ready(stack, name, alias=None, **cfg):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req(name, alias, **cfg), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return d


def deployment_names(stack):
    return [v["name"] for v in stack.deployer.list_views()]


def assert_foreign(e: DeployError, name: str):
    assert e.code == "model_name_taken" and e.status == 409
    assert e.message == (f"LiteLLM already has a model named {name} {FOREIGN_TEXT}. Pick another name, or remove "
                         "that entry from LiteLLM first.")
    assert e.extra["holder"] is None


# ---------- deploy / start / wake / rename ----------

@pytest.mark.parametrize("entry", [static, other_tool], ids=["static", "other-tool"])
async def test_deploy_refuses_a_name_litellm_has_from_elsewhere(stack, entry):
    fake = FakeLiteLLM()
    fake.models.append(entry("chat"))
    make(stack, fake)
    with pytest.raises(DeployError) as ei:
        await stack.deployer.deploy(req("a", alias="chat"), 1)
    assert_foreign(ei.value, "chat")
    assert ei.value.extra["external"]["host"] in ("old-engine", "elsewhere")
    assert deployment_names(stack) == []                      # nothing was created
    with pytest.raises(DeployError) as ei:                    # the served name (no alias) is checked too
        await stack.deployer.deploy(req("b", served_model_name="chat"), 1)
    assert_foreign(ei.value, "chat")
    assert (await stack.deployer.deploy(req("c", alias="chat-2"), 1))["model_name"] == "chat-2"
    assert sum(1 for m in fake.models if m["model_name"] == "chat") == 1  # the foreign entry is untouched


async def test_dgxmm_own_entries_do_not_block(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    a = await ready(stack, "a")
    assert await sync.sync()
    await stack.deployer.stop(a["id"])          # its LiteLLM entry stays until the next sync
    assert any(m["model_name"] == "qwen3-tiny" and m["model_info"].get("dgxmm_id") for m in fake.models)
    b = await ready(stack, "b")                 # DGXMM's own (stale) entry is not a foreign holder
    assert b["model_name"] == "qwen3-tiny"


async def test_unreadable_model_list_fails_open(stack):
    fake = FakeLiteLLM()
    fake.models += [static(n, mid=n) for n in ("chat", "chat-b", "chat-c")]
    make(stack, fake)
    fake.down = True                                                      # unreachable
    assert (await stack.deployer.deploy(req("a", alias="chat"), 1))["model_name"] == "chat"
    fake.down = False
    fake.raw["/model/info"] = httpx.Response(500, json={"error": {"message": "boom"}})  # HTTP error
    assert (await stack.deployer.deploy(req("b", alias="chat-b"), 1))["model_name"] == "chat-b"
    fake.raw["/model/info"] = httpx.Response(200, json={"data": "nope"})  # unexpected shape
    assert (await stack.deployer.deploy(req("c", alias="chat-c"), 1))["model_name"] == "chat-c"


async def test_unreadable_model_list_fails_open_without_integration(stack):
    fake = FakeLiteLLM()
    fake.models.append(static("chat"))
    sync = make(stack, fake)
    sync.client = None                          # integration off: no check at all
    assert (await stack.deployer.deploy(req("a", alias="chat"), 1))["model_name"] == "chat"
    assert ("GET", "/model/info") not in fake.calls


async def test_rename_start_and_wake_refuse_foreign_names(stack):
    fake = FakeLiteLLM()
    make(stack, fake)
    a = await ready(stack, "a", alias="mine")
    fake.models.append(static("chat"))
    with pytest.raises(DeployError) as ei:
        await stack.deployer.rename(a["id"], "chat")
    assert_foreign(ei.value, "chat")
    assert stack.deployer.view(a["id"])["alias"] == "mine"
    assert (await stack.deployer.rename(a["id"], "mine"))["alias"] == "mine"  # unchanged name: no check needed

    await stack.deployer.stop(a["id"])
    fake.models.append(static("mine", mid="static-2"))
    with pytest.raises(DeployError) as ei:
        await stack.deployer.restart(a["id"])
    assert_foreign(ei.value, "mine")
    assert stack.deployer.view(a["id"])["state"] == "stopped"

    b = await ready(stack, "b", alias="sleepy", enable_sleep_mode=True)
    await stack.deployer.sleep(b["id"])
    fake.models.append(other_tool("sleepy"))
    with pytest.raises(DeployError) as ei:
        await stack.deployer.wake(b["id"])
    assert_foreign(ei.value, "sleepy")
    assert stack.deployer.view(b["id"])["state"] == "sleeping"


def test_api_base_host_never_carries_credentials():
    assert api_base_host(static("x", "http://user:s3cret@old-engine:8000/v1?key=abc")) == "old-engine"
    assert api_base_host(static("x", "os.environ/SOME_URL")) is None
    assert api_base_host({"model_name": "x", "litellm_params": {"model": "openai/gpt"}}) is None
    assert api_base_host({"model_name": "x"}) is None


# ---------- sync: conflicts on existing registrations ----------

async def test_sync_flags_conflict_raises_insight_then_clears(stack):
    fake = FakeLiteLLM()
    insights = InsightService(stack.db, stack.events)
    sync = make(stack, fake, insights)
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"
    assert insights.list(kind="litellm_conflict") == []

    # a static entry with the same name shows up (e.g. LiteLLM restarted with an old config.yaml)
    fake.models.append(static("chat", "http://user:pw@old-engine:8000/v1"))
    for _ in range(2):  # deduplicated: one insight however many syncs see it
        assert await sync.sync()
    view = stack.deployer.view(d["id"])
    assert view["litellm_status"] == "conflict"
    assert view["litellm_error"]["code"] == "litellm_name_conflict"
    assert view["litellm_error"]["message"].startswith(
        "LiteLLM has another entry named chat that DGXMM didn't create — see the LiteLLM page")
    assert "old-engine" not in str(view)  # deployment views (list, detail, SSE) are viewer-readable: no host
    assert sync.conflict_host(d["id"]) == "old-engine"  # admins get it through the LiteLLM tab
    assert sync.is_registered(d["id"], "chat") is True          # its own registration stays as it was
    assert [m["model_info"]["id"] for m in fake.models if m["model_name"] == "chat" and not
            m["model_info"].get("dgxmm_id")] == ["static-1"]    # the foreign entry is never deleted or changed
    live = insights.list(kind="litellm_conflict", status=["open"])
    assert len(live) == 1
    ins = live[0]
    assert ins["severity"] == "warning" and ins["subject"] == f"deployment:{d['id']}"
    assert ins["title"] == "LiteLLM model name chat is also used outside DGXMM"
    assert ins["body"].startswith("LiteLLM has another entry named chat that DGXMM didn't create — see the LiteLLM page")
    for leak in ("old-engine", "pw", "user", "8000"):  # health insights are viewer-visible: no host, ever
        assert leak not in ins["title"] and leak not in ins["body"], leak
    assert "remove" in ins["body"].lower() and "model name" in ins["body"]

    # the duplicate is removed from LiteLLM: status and insight clear by themselves
    fake.models = [m for m in fake.models if m["model_info"]["id"] != "static-1"]
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"
    assert stack.deployer.view(d["id"])["litellm_error"] is None
    assert sync.conflict_host(d["id"]) is None
    assert insights.list(kind="litellm_conflict", status=["open"]) == []
    assert [i["status"] for i in insights.list(kind="litellm_conflict")] == ["expired"]


async def test_no_host_in_any_published_event(stack):
    """SSE: deployment events carry the view, insight events only an id; neither may name the other host."""
    fake = FakeLiteLLM()
    insights = InsightService(stack.db, stack.events)
    sync = make(stack, fake, insights)
    q = stack.events.subscribe()
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    fake.models.append(static("chat", "http://old-engine:8000/v1"))
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "conflict"
    seen = []
    while not q.empty():
        seen.append(q.get_nowait())
    assert any(t == "insights" for t, _ in seen) and any(t == "deployment" for t, _ in seen)
    assert "old-engine" not in repr(seen)


async def test_sync_does_not_register_next_to_a_foreign_entry(stack):
    fake = FakeLiteLLM()
    insights = InsightService(stack.db, stack.events)
    sync = make(stack, fake, insights)
    d = await ready(stack, "a", alias="chat")    # deployed while the name was free
    fake.models.append(other_tool("chat"))       # then another tool took it before the first sync
    assert await sync.sync()
    assert sum(1 for m in fake.models if m["model_name"] == "chat") == 1   # no silent duplicate
    assert stack.deployer.view(d["id"])["litellm_status"] == "conflict"
    assert sync.is_registered(d["id"], "chat") is False
    assert len(insights.list(kind="litellm_conflict", status=["open"])) == 1
    fake.models = [m for m in fake.models if m["model_info"]["id"] != "other-1"]
    assert await sync.sync()
    assert sync.is_registered(d["id"], "chat") is True
    assert [m["model_info"].get("dgxmm_id") for m in fake.models if m["model_name"] == "chat"] == [str(d["id"])]
    assert insights.list(kind="litellm_conflict", status=["open"]) == []


async def test_failed_or_disabled_sync_keeps_or_clears_conflicts(stack):
    fake = FakeLiteLLM()
    insights = InsightService(stack.db, stack.events)
    sync = make(stack, fake, insights)
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    fake.models.append(static("chat"))
    assert await sync.sync()
    fake.down = True
    assert not await sync.sync()                                 # unreadable: nothing is decided
    assert len(insights.list(kind="litellm_conflict", status=["open"])) == 1
    fake.down = False
    sync.client, client = None, sync.client                      # integration switched off: nothing to conflict with
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "disabled"
    assert insights.list(kind="litellm_conflict", status=["open"]) == []
    sync.client = client


async def test_stopped_deployment_resolves_its_conflict(stack):
    fake = FakeLiteLLM()
    insights = InsightService(stack.db, stack.events)
    sync = make(stack, fake, insights)
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    fake.models.append(static("chat"))
    assert await sync.sync()
    await stack.deployer.stop(d["id"])
    assert await sync.sync()
    assert insights.list(kind="litellm_conflict", status=["open"]) == []
    assert stack.deployer.view(d["id"])["litellm_status"] == "none"
