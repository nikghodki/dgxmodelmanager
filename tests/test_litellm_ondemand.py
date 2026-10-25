"""LiteLLM registration of on-demand deployments (Task 4): registered through the DGXMM gateway in every state
except failed (and removed), with the on-demand key, long timeouts and no retries; always-on deployments keep their
direct, ready-only entry; a mode switch adds the new entry before removing the old one; the key is never logged,
returned or published."""
import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from dgxmm.config import Settings
from dgxmm.deployer.service import DeployRequest
from dgxmm.litellm import LiteLLMClient, LiteLLMSync, ondemand_params, params_fingerprint
from dgxmm.ondemand import HEALTH_HEADER
from tests.fakes import FakeLiteLLM

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
KEY = "sk-ondemand-" + "k" * 40
GATEWAY = "http://dgxmm:8090/dgxmm/ondemand/{}/v1"
HEALTH_INFO = {"health_check_params": {"extra_headers": {HEALTH_HEADER: "1"}}}


@pytest.fixture
def key(stack, tmp_path):
    p = tmp_path / "ondemand_key"
    p.write_text(KEY + "\n")
    stack.deployer.settings.ondemand_key_file = str(p)
    return p


def make(stack, fake):
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return LiteLLMSync(LiteLLMClient.fixed("http://litellm:4000", "sk-test", http), stack.deployer, stack.db,
                       stack.events)


async def ready(stack, name, mode, alias=None, config=None):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=config or SMALL, alias=alias, mode=mode), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    return d


def mine(fake):
    return [m for m in fake.models if m["model_info"].get("dgxmm_id")]


def writes(fake):
    return [p for _, p in fake.calls if p in ("/model/new", "/model/delete")]


def test_internal_url_default_follows_base_path():
    assert Settings().dgxmm_internal_url == "http://dgxmm:8090/dgxmm"  # docker network, never via nginx
    assert Settings(base_path="/models").dgxmm_internal_url == "http://dgxmm:8090/models"
    assert Settings(dgxmm_internal_url="http://mgr.internal:8090/dgxmm/").dgxmm_internal_url == \
        "http://mgr.internal:8090/dgxmm"
    with pytest.raises(ValidationError):
        Settings(dgxmm_internal_url="ftp://dgxmm/x")


async def test_ondemand_registered_while_stopped_with_gateway_base(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand", alias="chat")
    assert await sync.sync()
    [entry] = mine(fake)
    assert entry["model_name"] == "chat" and entry["model_info"]["dgxmm_id"] == str(d["id"])
    assert entry["litellm_params"]["api_base"] == GATEWAY.format(d["id"])
    await stack.deployer.stop(d["id"])
    assert stack.deployer.view(d["id"])["state"] == "stopped"
    before = writes(fake)
    assert await sync.sync()
    assert mine(fake) == [entry] and writes(fake) == before  # stays registered while asleep, nothing rewritten
    v = stack.deployer.view(d["id"])
    assert v["litellm_status"] == "synced" and v["litellm_error"] is None
    assert sync.is_registered(d["id"], "chat") is True
    # failed: deregistered (LiteLLM would only get errors), and no LiteLLM status/error is reported for it
    stack.deployer._transition(d["id"], "failed", message="boom", last_error="boom")
    assert await sync.sync()
    assert mine(fake) == []
    assert stack.deployer.view(d["id"])["litellm_status"] == "none"
    assert sync.error_for(d["id"], "failed") is None
    # removed: gone as well
    stack.deployer._transition(d["id"], "stopped", message="reset")
    assert await sync.sync() and len(mine(fake)) == 1
    await stack.deployer.remove(d["id"])
    assert await sync.sync() and mine(fake) == []


async def test_entry_params_timeout_and_no_retries(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand")
    assert await sync.sync()
    [entry] = mine(fake)
    assert stack.deployer.settings.ondemand_queue_timeout_s == 900
    assert ondemand_params(stack.deployer.settings) == {"timeout": 1500, "stream_timeout": 1500, "num_retries": 0}
    assert entry["litellm_params"] == {"model": "openai/qwen3-tiny", "api_base": GATEWAY.format(d["id"]),
                                       "api_key": KEY, "timeout": 1500, "stream_timeout": 1500, "num_retries": 0}
    assert entry["model_info"] == {**entry["model_info"], "dgxmm_id": str(d["id"])}


async def test_always_on_unchanged_only_ready(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "always_on", alias="chat")
    assert await sync.sync()
    [entry] = mine(fake)
    assert entry["litellm_params"] == {"model": "openai/qwen3-tiny", "api_base": f"http://engines.test:{d['port']}/v1",
                                       "api_key": "none"}
    await stack.deployer.stop(d["id"])
    assert await sync.sync() and mine(fake) == []
    assert stack.deployer.view(d["id"])["litellm_status"] == "none"


async def test_existing_always_on_registration_survives_upgrade(stack, key):
    """An entry an older DGXMM registered for an always-on deployment is left exactly as it is."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "always_on", alias="chat")
    old = {"model_name": "chat", "litellm_params": {"model": "openai/qwen3-tiny",
                                                    "api_base": f"http://engines.test:{d['port']}/v1", "api_key": "none"},
           "model_info": {"dgxmm_id": str(d["id"]), "id": "old-chat", "db_model": True}}
    fake.models.append(json.loads(json.dumps(old)))
    assert await sync.sync()
    assert writes(fake) == [] and mine(fake) == [old]
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"


@pytest.mark.parametrize("frm,to", [("always_on", "on_demand"), ("on_demand", "always_on")])
async def test_mode_switch_adds_before_removes(stack, key, frm, to):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", frm, alias="chat")
    assert await sync.sync()
    [old] = mine(fake)
    seen: list[list[str]] = []
    orig = fake.handler

    def handler(req):  # what LiteLLM serves under "chat" right before each write
        if req.url.path == "/model/delete":
            seen.append([m["litellm_params"]["api_base"] for m in mine(fake) if m["model_name"] == "chat"])
        return orig(req)
    fake.handler = handler
    sync.client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stack.deployer.set_mode(d["id"], to, None)
    n = len(fake.calls)
    assert await sync.sync()
    assert [p for _, p in fake.calls[n:] if p != "/model/info"] == ["/model/new", "/model/delete"]
    want = GATEWAY.format(d["id"]) if to == "on_demand" else f"http://engines.test:{d['port']}/v1"
    assert len(seen) == 1 and set(seen[0]) == {old["litellm_params"]["api_base"], want}  # never zero entries
    [new] = mine(fake)
    assert new["litellm_params"]["api_base"] == want and new["model_info"]["id"] != old["model_info"]["id"]
    assert sync.last["registered"] == 1 and sync.last["removed"] == 1


async def test_ondemand_without_key_is_not_registered(stack, tmp_path):
    stack.deployer.settings.ondemand_key_file = str(tmp_path / "missing")
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    a = await ready(stack, "a", "on_demand", alias="lazy")
    b = await ready(stack, "b", "always_on", alias="steady")
    assert await sync.sync()
    assert [m["model_name"] for m in mine(fake)] == ["steady"]  # never an entry without the key
    v = stack.deployer.view(a["id"])
    assert v["litellm_status"] == "error"
    assert v["litellm_error"]["code"] == "ondemand_not_configured" and "install.sh" in v["litellm_error"]["message"]
    assert stack.deployer.view(b["id"])["litellm_status"] == "synced"
    # a deployment switched to on-demand while the key is missing keeps its existing entry (no outage)
    stack.deployer.set_mode(b["id"], "on_demand", None)
    n = len(fake.calls)
    assert await sync.sync()
    assert [p for _, p in fake.calls[n:] if p != "/model/info"] == []
    assert [m["model_name"] for m in mine(fake)] == ["steady"]
    assert stack.deployer.view(b["id"])["litellm_error"]["code"] == "ondemand_not_configured"
    # once the key exists, both register through the gateway and the old direct entry goes
    (tmp_path / "missing").write_text(KEY)
    assert await sync.sync()
    assert sorted(m["litellm_params"]["api_base"] for m in mine(fake)) == sorted(
        [GATEWAY.format(a["id"]), GATEWAY.format(b["id"])])
    assert stack.deployer.view(a["id"])["litellm_status"] == "synced"


async def test_state_changes_that_matter_trigger_a_sync(stack, key):
    sync = make(stack, FakeLiteLLM())
    for frm, to in [("stopped", "failed"), ("failed", "stopped"), ("planned", "starting"), ("loading", "ready")]:
        sync._wake.clear()
        await sync.on_state(1, frm, to)
        assert sync._wake.is_set(), (frm, to)
    sync._wake.clear()
    sync.on_removed(1)
    assert sync._wake.is_set()


async def test_key_never_logged(stack, key, caplog):
    caplog.set_level(logging.DEBUG)
    published: list = []
    orig = stack.events.publish
    stack.events.publish = lambda topic, data: (published.append(json.dumps(data, default=str)), orig(topic, data))
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand")
    fake.raw["/model/new"] = httpx.Response(400, json={"error": {"message": f"bad litellm_params: api_key={KEY}"}})
    assert not await sync.sync()
    assert KEY not in json.dumps(sync.last) and "***" in sync.last["error"]
    del fake.raw["/model/new"]
    assert await sync.sync()
    view = json.dumps(stack.deployer.view(d["id"]), default=str)
    assert KEY not in caplog.text and KEY not in view and not any(KEY in p for p in published)
    assert KEY not in json.dumps(sync.registrations) and KEY not in json.dumps(sync.last)


def fp(k):
    import hashlib
    return hashlib.sha256(k.encode()).hexdigest()[:12]


async def test_entry_carries_key_fingerprint_and_unchanged_key_never_rewrites(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    await ready(stack, "a", "on_demand")
    await ready(stack, "b", "always_on", alias="steady")
    assert await sync.sync()
    od, ao = sorted(mine(fake), key=lambda m: m["model_name"] == "steady")
    assert od["model_info"]["dgxmm_key_fp"] == fp(KEY) and KEY not in json.dumps(od["model_info"])
    assert "dgxmm_key_fp" not in ao["model_info"]  # always-on entries carry no fingerprint
    before = writes(fake)
    for _ in range(3):
        assert await sync.sync()
    assert writes(fake) == before


async def test_key_rotation_reregisters_once(stack, key, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand")
    assert await sync.sync()
    [old] = mine(fake)
    new_key = "sk-rotated-" + "r" * 40
    key.write_text(new_key)
    n = len(fake.calls)
    assert await sync.sync()
    assert [p for _, p in fake.calls[n:] if p != "/model/info"] == ["/model/new", "/model/delete"]  # add, then remove
    [new] = mine(fake)
    assert new["model_info"]["id"] != old["model_info"]["id"]
    assert new["litellm_params"]["api_key"] == new_key and new["model_info"]["dgxmm_key_fp"] == fp(new_key)
    assert new["litellm_params"]["api_base"] == GATEWAY.format(d["id"])
    before = writes(fake)
    assert await sync.sync() and writes(fake) == before  # stable afterwards
    assert KEY not in caplog.text and new_key not in caplog.text


async def test_legacy_entry_without_fingerprint_replaced_once(stack, key):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand", alias="chat")
    fake.models.append({"model_name": "chat", "litellm_params": {"model": "openai/qwen3-tiny",
                                                                 "api_base": GATEWAY.format(d["id"]),
                                                                 **ondemand_params(stack.deployer.settings)},
                        "model_info": {"dgxmm_id": str(d["id"]), "id": "legacy"}})  # /model/info: no api_key shown
    assert await sync.sync()
    assert writes(fake) == ["/model/new", "/model/delete"]
    [new] = mine(fake)
    assert new["model_info"]["id"] != "legacy" and new["model_info"]["dgxmm_key_fp"] == fp(KEY)
    assert await sync.sync() and writes(fake) == ["/model/new", "/model/delete"]


@pytest.mark.parametrize("queue,timeout", [(60, 1200), (600, 1200), (900, 1500), (3600, 4200)])
def test_entry_timeout_covers_the_queue_limit(queue, timeout):
    """LiteLLM must not give up before the gateway's queue limit plus a start-up: max(1200, queue + 600) s."""
    assert ondemand_params(Settings(ondemand_queue_timeout_s=queue)) == {"timeout": timeout, "stream_timeout": timeout,
                                                                         "num_retries": 0}


async def test_queue_timeout_change_reregisters_once(stack, key):
    """A new ondemand_queue_timeout_s (config change + restart) re-registers the on-demand entries with the new
    timeouts, adding before removing; a change that leaves the timeouts as they are, and always-on entries, are not
    rewritten. The entry carries a fingerprint of its params (model_info.dgxmm_params_fp)."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    settings = stack.deployer.settings
    d = await ready(stack, "a", "on_demand")
    await ready(stack, "b", "always_on", alias="steady")
    assert await sync.sync()
    [old] = [m for m in mine(fake) if m["model_name"] != "steady"]
    assert old["model_info"]["dgxmm_params_fp"]
    settings.ondemand_queue_timeout_s = 3000
    n = len(fake.calls)
    assert await sync.sync()
    assert [p for _, p in fake.calls[n:] if p != "/model/info"] == ["/model/new", "/model/delete"]  # add, then remove
    [new] = [m for m in mine(fake) if m["model_name"] != "steady"]
    assert new["litellm_params"]["timeout"] == new["litellm_params"]["stream_timeout"] == 3600
    assert new["litellm_params"]["api_base"] == GATEWAY.format(d["id"]) and new["litellm_params"]["api_key"] == KEY
    assert new["model_info"]["dgxmm_params_fp"] != old["model_info"]["dgxmm_params_fp"]
    before = writes(fake)
    for _ in range(3):
        assert await sync.sync()
    assert writes(fake) == before  # stable
    settings.ondemand_queue_timeout_s = 300
    assert await sync.sync() and len(writes(fake)) == len(before) + 2  # 3600 -> 1200
    before = writes(fake)
    settings.ondemand_queue_timeout_s = 500  # still 1200: nothing to do
    assert await sync.sync() and writes(fake) == before
    assert "dgxmm_params_fp" not in next(m for m in mine(fake) if m["model_name"] == "steady")["model_info"]


async def test_entry_without_params_fingerprint_kept_when_its_params_match(stack, key):
    """An entry registered before params fingerprints, with the current key and timeouts, is not rewritten; one with
    other timeouts is replaced once."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand", alias="chat")
    entry = {"model_name": "chat", "litellm_params": {"model": "openai/qwen3-tiny", "api_base": GATEWAY.format(d["id"]),
                                                      **ondemand_params(stack.deployer.settings)},
             "model_info": {"dgxmm_id": str(d["id"]), "id": "earlier", "dgxmm_key_fp": fp(KEY), **HEALTH_INFO}}
    fake.models.append(json.loads(json.dumps(entry)))
    assert await sync.sync() and writes(fake) == []
    fake.models[-1]["litellm_params"].update(timeout=1200.0, stream_timeout=1200.0)
    assert await sync.sync() and writes(fake) == ["/model/new", "/model/delete"]
    [new] = mine(fake)
    assert new["model_info"]["id"] != "earlier" and new["litellm_params"]["timeout"] == 1500


async def test_ondemand_entry_health_checks_carry_the_gateway_header(stack, key):
    """LiteLLM's /health (and background) checks send a real completion; on-demand entries add HEALTH_HEADER to it
    (model_info.health_check_params, merged into the probe), so the gateway never wakes the model for one."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    await ready(stack, "a", "on_demand")
    await ready(stack, "b", "always_on", alias="steady")
    assert await sync.sync()
    od = next(m for m in mine(fake) if m["model_name"] != "steady")
    ao = next(m for m in mine(fake) if m["model_name"] == "steady")
    assert od["model_info"]["health_check_params"] == {"extra_headers": {HEALTH_HEADER: "1"}}
    assert "health_check_params" not in ao["model_info"]
    assert "extra_headers" not in od["litellm_params"]  # real requests never carry the header


async def test_entry_registered_without_health_params_replaced_once(stack, key):
    """An entry registered by the previous version (params fingerprint without the health-check info) is replaced
    once, adding before removing; the result is stable."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand", alias="chat")
    params = ondemand_params(stack.deployer.settings)
    fake.models.append({"model_name": "chat",
                        "litellm_params": {"model": "openai/qwen3-tiny", "api_base": GATEWAY.format(d["id"]),
                                           "api_key": KEY, **params},
                        "model_info": {"dgxmm_id": str(d["id"]), "id": "previous", "dgxmm_key_fp": fp(KEY),
                                       "dgxmm_params_fp": params_fingerprint(params)}})
    assert await sync.sync() and writes(fake) == ["/model/new", "/model/delete"]
    [new] = mine(fake)
    assert new["model_info"]["id"] != "previous" and new["model_info"]["health_check_params"]
    for _ in range(2):
        assert await sync.sync()
    assert writes(fake) == ["/model/new", "/model/delete"]

    fake.models[-1]["model_info"].pop("dgxmm_params_fp")  # fingerprint-less: compared field by field
    fake.models[-1]["model_info"].pop("health_check_params")
    assert await sync.sync() and len(writes(fake)) == 4


@pytest.mark.parametrize("frm", ["stopped", "sleeping"])
async def test_switch_to_always_on_keeps_one_entry_while_it_starts(stack, key, monkeypatch, frm):
    """On demand -> always on of a deployment that is not ready: it stays on demand (switch pending), registered
    through the gateway, for the whole start, and the direct entry replaces the gateway one only once it is ready.
    LiteLLM never has zero entries for the name (no "model not found" window), and after every sync exactly one."""
    from dgxmm.ondemand.controller import OnDemandController
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", "on_demand", alias="chat", config={**SMALL, "enable_sleep_mode": True})
    if frm == "stopped":
        await stack.deployer.stop(d["id"])
    else:
        await stack.deployer.sleep(d["id"])
    assert stack.deployer.view(d["id"])["state"] == frm
    assert await sync.sync()
    chat = lambda: [m for m in mine(fake) if m["model_name"] == "chat"]  # noqa: E731
    assert [m["litellm_params"]["api_base"] for m in chat()] == [GATEWAY.format(d["id"])]
    counts: list[int] = []
    orig = fake.handler

    def handler(req):  # entries under "chat" after every LiteLLM call
        r = orig(req)
        counts.append(len(chat()))
        return r
    sync.client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gate = asyncio.Event()
    if frm == "stopped":
        stack.http.all_healthy = False  # the engine takes a while to become healthy
    else:
        engine = stack.deployer.engines["vllm"]
        real_wake = engine.wake

        async def slow_wake(*a, **kw):
            await gate.wait()
            return await real_wake(*a, **kw)
        monkeypatch.setattr(engine, "wake", slow_wake)
    ctl = OnDemandController(SimpleNamespace(settings=stack.deployer.settings, deployer=stack.deployer, db=stack.db,
                                             litellm=sync, events=stack.events))
    switching = asyncio.create_task(ctl.set_mode(d["id"], "always_on", None, 1))
    for _ in range(20):
        await asyncio.sleep(0.01)
        v = stack.deployer.view(d["id"])
        assert v["state"] != "ready"
        assert (v["mode"], v["pending_mode"]) == ("on_demand", "always_on")
        assert v["litellm_status"] == "synced" and v["litellm_error"] is None
        assert await sync.sync()
        assert [m["litellm_params"]["api_base"] for m in chat()] == [GATEWAY.format(d["id"])]
    if frm == "stopped":
        await asyncio.wait_for(switching, 2)
        stack.http.all_healthy = True
    else:
        gate.set()
        await asyncio.wait_for(switching, 2)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    v = stack.deployer.view(d["id"])
    assert (v["mode"], v["pending_mode"]) == ("always_on", None)
    assert await sync.sync()
    assert [m["litellm_params"]["api_base"] for m in chat()] == [f"http://engines.test:{v['port']}/v1"]
    assert counts and min(counts) >= 1  # never without an entry; the add runs before the delete
    assert counts[-1] == 1
