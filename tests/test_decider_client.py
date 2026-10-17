"""SystemOneClient: request shape, bearer auth, timeouts, LRU cache and circuit breaker."""
import asyncio
import json

import httpx
import pytest

from dgxmm.decider.client import CircuitBreaker, DeciderError, SystemOneClient
from dgxmm.decider.types import Option

OPTS = [Option("gpu_oom", "GPU out of memory", "memory ran out"), Option("port_in_use", "Port in use", ""),
        Option("other", "Other")]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def reply(choice="gpu_oom", probs=None):
    probs = probs or {"gpu_oom": 0.9, "port_in_use": 0.07, "other": 0.03}
    return {"model": "d3-mini", "answers": {"q": {"type": "choice", "choice": choice, "probabilities": probs,
                                                  "confidence": 0.5}}, "usage": {"input_tokens": 10}}


def make(handler, clock=None, key="k-secret", **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SystemOneClient(http, base_url=lambda: "http://decider:8100", api_key=lambda: key, revision="rev1",
                           clock=clock or Clock(), **kw)


async def test_choose_sends_typed_question_with_bearer_and_returns_top_probability():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, json=reply())

    c = make(handler)
    ans = await c.choose(feature="diagnosis", state="LOG (untrusted):\n<<<\nboom\n>>>", question="Why?", options=OPTS)
    assert ans.choice == "gpu_oom" and ans.confidence == pytest.approx(0.9)
    assert ans.probabilities["port_in_use"] == pytest.approx(0.07)
    req = seen[0]
    assert req.url.path == "/v1/systemone" and req.headers["authorization"] == "Bearer k-secret"
    body = json.loads(req.content)
    assert body["state"].startswith("LOG (untrusted)")
    q = body["questions"]["q"]
    assert q == {"type": "choice", "instructions": "Why?",
                 "criteria": {"gpu_oom": "memory ran out", "port_in_use": "Port in use", "other": "Other"}}


async def test_no_key_sends_no_authorization_header():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=reply())

    await make(handler, key=None).choose(feature="kind", state="s", question="q?", options=OPTS)
    assert "authorization" not in seen[0].headers


async def test_cache_hits_skip_http_and_key_includes_feature_input_and_revision():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json=reply())

    c = make(handler)
    a1 = await c.choose(feature="diagnosis", state="s", question="q?", options=OPTS)
    a2 = await c.choose(feature="diagnosis", state="s", question="q?", options=OPTS)
    assert len(calls) == 1 and a2.cached and not a1.cached
    await c.choose(feature="kind", state="s", question="q?", options=OPTS)        # other feature
    await c.choose(feature="diagnosis", state="s2", question="q?", options=OPTS)  # other input
    assert len(calls) == 3
    c.revision = "rev2"
    await c.choose(feature="diagnosis", state="s", question="q?", options=OPTS)
    assert len(calls) == 4


async def test_cache_is_bounded_lru():
    def handler(req):
        return httpx.Response(200, json=reply())

    c = make(handler, cache_size=2)
    for s in ("a", "b", "c"):
        await c.choose(feature="f", state=s, question="q", options=OPTS)
    assert len(c.cache) == 2
    assert c.input_hash("f", "a", "q", OPTS) not in c.cache


async def test_breaker_opens_after_three_failures_and_recovers_after_60s():
    clock = Clock()
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500, text="boom")

    c = make(handler, clock=clock)
    for i in range(3):
        with pytest.raises(DeciderError):
            await c.choose(feature="f", state=f"s{i}", question="q", options=OPTS)
    assert c.breaker.is_open()
    with pytest.raises(DeciderError) as e:
        await c.choose(feature="f", state="s9", question="q", options=OPTS)
    assert e.value.code == "circuit_open" and len(calls) == 3
    clock.t += 61
    assert not c.breaker.is_open()


async def test_success_resets_breaker_failures():
    clock = Clock()
    state = {"fail": True}

    def handler(req):
        return httpx.Response(500) if state["fail"] else httpx.Response(200, json=reply())

    c = make(handler, clock=clock)
    for i in range(2):
        with pytest.raises(DeciderError):
            await c.choose(feature="f", state=f"s{i}", question="q", options=OPTS)
    state["fail"] = False
    await c.choose(feature="f", state="ok", question="q", options=OPTS)
    assert c.breaker.failures == 0


async def test_timeout_raises_decider_error_and_counts_as_failure():
    async def slow(req):
        await asyncio.sleep(1)
        return httpx.Response(200, json=reply())

    c = make(slow)
    with pytest.raises(DeciderError) as e:
        await c.choose(feature="f", state="s", question="q", options=OPTS, timeout_s=0.05)
    assert e.value.code == "timeout" and c.breaker.failures == 1


async def test_malformed_reply_and_unknown_choice_are_errors():
    def bad(req):
        return httpx.Response(200, json={"answers": {"q": {"choice": "nope", "probabilities": {"nope": 1.0}}}})

    with pytest.raises(DeciderError) as e:
        await make(bad).choose(feature="f", state="s", question="q", options=OPTS)
    assert e.value.code == "bad_reply"

    def junk(req):
        return httpx.Response(200, text="not json")

    with pytest.raises(DeciderError):
        await make(junk).choose(feature="f", state="s", question="q", options=OPTS)


async def test_errors_never_contain_the_key():
    def unauthorized(req):
        return httpx.Response(401, json={"detail": "a valid bearer key is required"})

    with pytest.raises(DeciderError) as e:
        await make(unauthorized, key="sk-very-secret").choose(feature="f", state="s", question="q", options=OPTS)
    assert "sk-very-secret" not in str(e.value) and e.value.code == "unauthorized"


async def test_health_requires_ready_status_and_probe():
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ready", "authentication": True})
        assert req.headers["authorization"] == "Bearer k-secret"
        return httpx.Response(200, json={"answers": {"probe": {"choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}}})

    assert await make(handler).health() is True

    def down(req):
        return httpx.Response(503)

    assert await make(down).health() is False

    def listy(req):
        return httpx.Response(200, json=["ready"])

    assert await make(listy).health() is False


def test_breaker_unit():
    clock = Clock()
    b = CircuitBreaker(threshold=3, open_s=60, clock=clock)
    b.failure(); b.failure()
    assert not b.is_open()
    b.failure()
    assert b.is_open() and b.retry_in() == pytest.approx(60)
    clock.t += 60
    assert not b.is_open()
    b.failure()  # half-open: one more failure re-opens at once
    assert b.is_open()
