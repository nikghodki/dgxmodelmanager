"""LiteLLMClient: status/version/DB-state caching, deadlines, response validation."""
import asyncio

import httpx
import pytest

from dgxmm.litellm import LiteLLMClient, LiteLLMError
from tests.fakes import FakeLiteLLM


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(fake, clock=None):
    c = LiteLLMClient.fixed("http://litellm:4000", fake.master_key, fake.client())
    if clock:
        c.clock = clock
    return c


def n(fake, path):
    return sum(1 for _, p in fake.calls if p == path)


async def test_status_cached_15s():
    fake, clock = FakeLiteLLM(), Clock()
    c = make(fake, clock)
    a = await c.status()
    assert a["reachable"] and a["version"] == "1.100.0"
    b = await c.status()
    assert b["checked_at"] == a["checked_at"] and n(fake, "/health/liveliness") == 1
    assert n(fake, "/health/readiness") == 1  # one readiness call per probe
    clock.t += 16
    await c.status()
    assert n(fake, "/health/liveliness") == 2
    await c.status(force=True)
    assert n(fake, "/health/liveliness") == 3
    assert all(a is None for a in fake.auth)  # status never sends the master key


async def test_status_single_flight():
    fake = FakeLiteLLM()
    c = make(fake)
    await asyncio.gather(*(c.status() for _ in range(5)))
    assert n(fake, "/health/liveliness") == 1 and n(fake, "/openapi.json") == 1


async def test_probe_overall_deadline():
    async def slow(req):
        await asyncio.sleep(10)
        return httpx.Response(200)
    c = LiteLLMClient.fixed("http://litellm:4000", "sk-x", httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    c.probe_deadline_s = 0.2
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    out = await c.status()
    assert loop.time() - t0 < 2
    assert out["reachable"] is False and "timed out" in out["error"]


async def test_cancelled_version_fetch_is_negative_cached():
    fake, clock = FakeLiteLLM(), Clock()

    async def handler(req):
        if req.url.path == "/openapi.json":
            fake.calls.append(("GET", "/openapi.json"))
            await asyncio.sleep(10)
        return fake.handler(req)
    c = LiteLLMClient.fixed("http://litellm:4000", "sk-test", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    c.clock, c.probe_deadline_s = clock, 0.2
    out = await c.status()
    assert out["reachable"] is True and out["version"] is None
    clock.t += 16  # status cache expired, negative version cache (30 s) not
    out = await c.status()
    assert out["reachable"] is True and n(fake, "/openapi.json") == 1
    for t in c._version_tasks.values():
        t.cancel()


async def test_not_a_litellm_endpoint():
    fake = FakeLiteLLM()
    fake.impostor = True
    out = await make(fake).probe(auth=True)
    assert out["reachable"] is False and "not a LiteLLM endpoint" in out["error"] and out["auth_ok"] is None
    assert all(a is None for a in fake.auth)


async def test_version_negative_cache():
    fake, clock = FakeLiteLLM(), Clock()
    fake.openapi_status = 500
    c = make(fake, clock)
    assert (await c.status())["version"] is None
    clock.t += 16
    assert (await c.status())["version"] is None
    assert n(fake, "/openapi.json") == 1  # negative result cached ~30 s
    fake.openapi_status = 200
    clock.t += 20
    assert (await c.status())["version"] == "1.100.0"
    assert n(fake, "/openapi.json") == 2


async def test_version_body_capped():
    fake = FakeLiteLLM()
    fake.openapi_pad = 5 * 1024 * 1024
    assert (await make(fake).status())["version"] is None


async def test_version_from_readiness_skips_openapi():
    fake = FakeLiteLLM()
    fake.readiness_version, fake.version = True, "1.55.0"
    assert (await make(fake).status())["version"] == "1.55.0"
    assert n(fake, "/openapi.json") == 0


@pytest.mark.parametrize("payload", [[1, 2], "x", None, 3])
async def test_non_dict_json_is_mapped(payload):
    fake = FakeLiteLLM()
    for path in ("/model/info", "/key/list", "/key/generate"):
        fake.raw[path] = httpx.Response(200, json=payload)
    c = make(fake)
    for coro in (c.list_models(), c.key_list(), c.key_generate({"key_alias": "a"})):
        with pytest.raises(LiteLLMError) as e:
            await coro
        assert e.value.status == 502 and e.value.code == "litellm_error"


async def test_db_state_cached_for_key_ops():
    fake, clock = FakeLiteLLM(), Clock()
    c = make(fake, clock)
    await c.key_list()
    await c.key_list()
    await c.key_generate({"key_alias": "a"})
    assert n(fake, "/health/readiness") == 1
    clock.t += 16
    await c.key_list()
    assert n(fake, "/health/readiness") == 2


async def test_key_list_cache():
    fake, clock = FakeLiteLLM(), Clock()
    c = make(fake, clock)
    assert await c.key_list(max_age=15) == []
    fake.add_key("x")
    assert await c.key_list(max_age=15) == []  # cached
    assert len(await c.key_list()) == 1  # fresh read refreshes the cache
    fake.add_key("y")
    c.invalidate_keys()
    assert len(await c.key_list(max_age=15)) == 2


async def test_version_fetch_outliving_the_probe_deadline_fills_the_status():
    """A cold, slow /openapi.json must not leave the page on "version unknown": the fetch finishes in the
    background and the (still cached) status then carries the version."""
    fake = FakeLiteLLM()

    async def handler(req):
        if req.url.path == "/openapi.json":
            await asyncio.sleep(0.4)
        return fake.handler(req)
    c = LiteLLMClient.fixed("http://litellm:4000", "sk-test", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    c.probe_deadline_s = 0.2
    out = await c.status()
    assert out["reachable"] is True and out["version"] is None
    await asyncio.sleep(0.5)
    out = await c.status()
    assert out["version"] == "1.100.0" and n(fake, "/openapi.json") == 1
