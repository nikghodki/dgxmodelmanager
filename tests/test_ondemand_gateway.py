"""On-demand models, part 3: the gateway (dgxmm/ondemand/gateway.py) LiteLLM calls for on-demand deployments —
bearer key, start-on-request through the controller, the queued notice, the 503s, byte-identical streaming and client
hang-ups (docs/superpowers/plans/2026-10-24-on-demand-models.md, Task 3). The engine is an httpx.MockTransport, the
controller a fake (or the real one over the controller tests' fake deployer); requests are driven straight through the
ASGI app so every message the client would see, and when, is visible."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from dgxmm.app import create_app
from dgxmm.config import Settings
from dgxmm.deployer.service import DeployError
from dgxmm.docker_ctl import DockerController
from dgxmm.ondemand.controller import OnDemandController, OnDemandTimeout, OnDemandUnavailable
from dgxmm.ondemand import HEALTH_HEADER
from dgxmm.ondemand.gateway import (AUTH_FAILURES_PER_MIN, MAX_BODY_BYTES, NOTICE, TRAFFIC, Gateway, PeerPolicy,
                                    peer_policy)
from dgxmm.proxy_headers import RAW_PEER_KEY, tcp_app
from tests.test_ondemand_controller import Clock, FakeDeployer as ControllerDeployer, FakeMetrics, until, use

KEY = "sk-test-gateway-key-0123456789"
SECRET = "do-not-log-this-prompt"
AUTH = {"authorization": f"Bearer {KEY}"}
BASE = "/dgxmm/ondemand/1/v1"
CHAT = {"model": "org/a", "messages": [{"role": "user", "content": SECRET}]}
# test addresses: the dgxmm docker network as the gateway sees it, its bridge gateway, and peers outside it
SUBNET = "198.51.100.0/24"
PEER = "198.51.100.5"
BRIDGE_GW = "198.51.100.1"
POLICY = PeerPolicy([ipaddress.ip_network(SUBNET)], frozenset({ipaddress.ip_address(BRIDGE_GW)}))
NOT_FOUND = {"error": {"message": "Not Found", "type": "invalid_request_error", "code": "not_found"}}
TIMEOUT_BODY = {"error": {"message": "model m-a is busy; try again later", "type": "server_error",
                          "code": "ondemand_queue_timeout"}}


# --- fakes ----------------------------------------------------------------------------------------

def _record(dep_id: int, name: str, state: str, mode: str = "on_demand", system: bool = False) -> dict:
    return {"id": dep_id, "name": name, "engine": "vllm", "state": state, "mode": mode, "system": system,
            "port": 8000 + dep_id, "served_name": f"org/{name}", "model_name": f"m-{name}",
            "created_at": "2026-10-24T10:00:00+00:00"}


class FakeDeployer:
    def __init__(self):
        self.deps: dict[int, dict] = {}

    def add(self, dep_id: int, name: str = "a", state: str = "stopped", **kw) -> None:
        self.deps[dep_id] = _record(dep_id, name, state, **kw)

    def view(self, dep_id: int) -> dict:
        if dep_id not in self.deps:
            raise DeployError("not_found", f"deployment {dep_id} not found", 404)
        return dict(self.deps[dep_id])

    def base_url(self, name: str, port: int, engine: str | None = None) -> str:
        return f"http://engine-{name}:{port}"


class FakeSlot:
    def __init__(self):
        self.released = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.release()

    def release(self):
        self.released = True


class FakeController:
    """acquire() runs ``script(dep_id, on_queued)`` (if set), then marks the deployment ready and hands out a slot."""

    def __init__(self, dep: FakeDeployer):
        self.dep, self.calls, self.slots, self.cancelled = dep, [], [], []
        self.script = None

    async def acquire(self, dep_id: int, *, on_queued=None):
        self.calls.append(dep_id)
        try:
            if self.script is not None:
                await self.script(dep_id, on_queued)
        except asyncio.CancelledError:
            self.cancelled.append(dep_id)
            raise
        self.dep.deps[dep_id]["state"] = "ready"
        slot = FakeSlot()
        self.slots.append(slot)
        return slot


class FakeEngine:
    """Answers every request with ``reply(request)`` (default: a small chat completion) and records the requests."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.reply = lambda req: httpx.Response(200, json=_completion("hello"))

    async def handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        resp = self.reply(request)
        if resp.is_stream_consumed:  # built from bytes (already read): served as a stream, like a real socket
            async def once(data=resp.content):
                yield data
            resp = httpx.Response(resp.status_code, headers=resp.headers, content=once())
        return resp


def _completion(text: str) -> dict:
    return {"id": "cmpl-1", "object": "chat.completion", "model": "org/a",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


class Call:
    """One HTTP request driven straight through the ASGI app; ``messages`` are what the client got, in order.
    ``hang_up()`` makes the client disconnect."""

    def __init__(self, app, method: str, path: str, body: bytes = b"", headers: dict | None = None,
                 chunks: list[bytes] | None = None, client: str = PEER, extra: dict | None = None):
        self.app, self.method, self.path, self.client, self.extra = app, method, path, client, extra or {}
        self.parts = chunks if chunks is not None else [body]
        hdrs = {k.lower(): v for k, v in (headers or {}).items()}
        if chunks is None:
            hdrs.setdefault("content-length", str(len(body)))
        self.headers = [(k.encode(), v.encode()) for k, v in hdrs.items()]
        self.messages: list[dict] = []
        self._gone = asyncio.Event()
        self.task: asyncio.Task | None = None

    def start(self) -> "Call":
        self.task = asyncio.create_task(self._run())
        return self

    async def _run(self) -> None:
        path, _, query = self.path.partition("?")
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": self.method, "scheme": "http", "path": unquote(path), "raw_path": path.encode(),
                 "query_string": query.encode(), "root_path": "", "headers": self.headers,
                 "client": (self.client, 40000), "server": ("testserver", 80), **self.extra}
        parts = list(self.parts)

        async def receive():
            if parts:
                body = parts.pop(0)
                return {"type": "http.request", "body": body, "more_body": bool(parts)}
            await self._gone.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            self.messages.append(message)

        await self.app(scope, receive, send)

    def hang_up(self) -> None:
        self._gone.set()

    async def done(self, timeout: float = 3.0) -> "Call":
        await asyncio.wait_for(self.task, timeout)
        return self

    @property
    def status(self) -> int | None:
        return next((m["status"] for m in self.messages if m["type"] == "http.response.start"), None)

    @property
    def response_headers(self) -> dict[str, str]:
        start = next(m for m in self.messages if m["type"] == "http.response.start")
        return {k.decode().lower(): v.decode() for k, v in start["headers"]}

    @property
    def chunks(self) -> list[bytes]:
        return [m["body"] for m in self.messages if m["type"] == "http.response.body" and m.get("body")]

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)

    @property
    def finished(self) -> bool:
        return any(m["type"] == "http.response.body" and not m.get("more_body") for m in self.messages)

    def json(self):
        return json.loads(self.body)


async def call(app, method: str, path: str, payload=None, headers: dict | None = AUTH, **kw) -> Call:
    body = json.dumps(payload).encode() if payload is not None else b""
    hdrs = dict(headers or {})
    if payload is not None:
        hdrs["content-type"] = "application/json"
    return await Call(app, method, path, body, hdrs, **kw).start().done()


def _key_file(settings, tmp_path, value: str | None = KEY):
    path = tmp_path / "dgxmm_ondemand_key"
    if value is not None:
        path.write_text(value + "\n")
    else:
        path.unlink(missing_ok=True)
    settings.ondemand_key_file = str(path)


@pytest.fixture
def gw(ctx, settings, tmp_path):
    _key_file(settings, tmp_path)
    dep = FakeDeployer()
    dep.add(1)
    ctl = FakeController(dep)
    engine = FakeEngine()
    ctx.deployer, ctx.ondemand = dep, ctl
    ctx.ondemand_peers = POLICY
    app = create_app(ctx)
    app.state.ondemand_gateway = Gateway(ctx, transport=httpx.MockTransport(engine.handle))
    return SimpleNamespace(app=app, dep=dep, ctl=ctl, engine=engine, ctx=ctx)


def _queued_until(gate: asyncio.Event):
    async def script(dep_id, on_queued):
        await on_queued()
        await gate.wait()
    return script


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _events(body: bytes) -> list:
    out = []
    for block in body.decode().split("\n\n"):
        if block.startswith("data: "):
            data = block[len("data: "):]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


async def _until_sent(c: Call, n: int) -> None:
    await until(lambda: len(c.chunks) >= n)


# --- auth, configuration, limits ------------------------------------------------------------------

async def test_rejects_missing_or_wrong_key(gw):
    for headers in ({}, {"authorization": "Bearer sk-wrong"}, {"authorization": KEY},
                    {"authorization": f"Basic {KEY}"}, {"authorization": f"Bearer {KEY}x"}):
        for method, path, payload in (("POST", f"{BASE}/chat/completions", CHAT), ("GET", f"{BASE}/models", None),
                                      ("GET", f"{BASE}/health", None)):
            c = await call(gw.app, method, path, payload, headers=headers)
            assert c.status == 401, (headers, path)
            assert c.json()["error"]["code"] == "invalid_api_key"
    assert gw.ctl.calls == [] and gw.engine.requests == []
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers={"authorization": f"bearer {KEY}"})
    assert c.status == 200  # no session cookie, no CSRF token: the gateway's own key is the only credential
    assert gw.ctl.calls == [1]


async def test_missing_key_fails_closed_and_logs_once(gw, settings, tmp_path, caplog):
    for value in (None, "   "):
        _key_file(settings, tmp_path, value)
        gw.app.state.ondemand_gateway = Gateway(gw.ctx, transport=httpx.MockTransport(gw.engine.handle))
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for headers in (AUTH, {}, {"authorization": "Bearer "}):
                c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=headers)
                assert c.status == 503
                assert c.json() == {"error": {"message": c.json()["error"]["message"], "type": "server_error",
                                              "code": "ondemand_not_configured"}}
        assert len([r for r in caplog.records if "gateway" in r.getMessage()]) == 1
    assert gw.ctl.calls == [] and gw.engine.requests == []
    _key_file(settings, tmp_path)  # the installer wrote the key: served without a restart
    gw.app.state.ondemand_gateway.clock = lambda: 1e9  # past the once-per-second recheck
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
    assert c.status == 200


async def test_key_rotation_without_restart(gw, settings, tmp_path, caplog, monkeypatch):
    """The key file is re-read when it changes (inode, mtime or size; checked at most once per second), so a rotated
    key is accepted and the old one refused without restarting DGXMM, the same way the LiteLLM sync re-registers
    the entries with the new key. A file that goes missing or empty fails closed."""
    import os
    now = [100.0]
    g = gw.app.state.ondemand_gateway
    g.clock = lambda: now[0]
    triggers = []
    gw.ctx.litellm = SimpleNamespace(trigger=lambda: triggers.append(1))
    stats, real_stat = [], os.stat
    assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)).status == 200
    new_key = "sk-rotated-gateway-key-9876543210"
    path = tmp_path / "dgxmm_ondemand_key"
    tmp = tmp_path / "dgxmm_ondemand_key.new"
    tmp.write_text(new_key + "\n")
    os.replace(tmp, path)  # a new inode (the mtime may well be the same second)
    new_auth = {"authorization": f"Bearer {new_key}"}
    # within the same second the cached key is still used: no file access per request
    with monkeypatch.context() as m:
        m.setattr(os, "stat", lambda p, *a, **kw: (stats.append(str(p)), real_stat(p, *a, **kw))[1])
        assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)).status == 200
    assert not [p for p in stats if "ondemand_key" in p]
    now[0] += 1.01
    with caplog.at_level(logging.DEBUG):
        c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
        assert c.status == 401 and c.json()["error"]["code"] == "invalid_api_key"
        assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=new_auth)).status == 200
    assert KEY not in caplog.text and new_key not in caplog.text
    assert triggers == [1]  # LiteLLM re-registers its entries with the new key straight away, not at the next 60 s
    # rewritten in place (same inode, new size): picked up too
    path.write_text(KEY + "-third\n")
    now[0] += 1.01
    assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=new_auth)).status == 401
    assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT,
                       headers={"authorization": f"Bearer {KEY}-third"})).status == 200
    # missing, then empty: fail closed for every key
    for value in (None, ""):
        _key_file(settings, tmp_path, value)
        now[0] += 1.01
        for headers in (AUTH, new_auth, {"authorization": f"Bearer {KEY}-third"}):
            c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=headers)
            assert c.status == 503 and c.json()["error"]["code"] == "ondemand_not_configured"
    _key_file(settings, tmp_path, new_key)
    now[0] += 1.01
    assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=new_auth)).status == 200


async def test_body_cap_16mib(gw):
    big = b"x" * (MAX_BODY_BYTES + 1)
    c = await Call(gw.app, "POST", f"{BASE}/chat/completions", big, AUTH).start().done()
    assert c.status == 413 and c.json()["error"]["code"] == "request_too_large"
    c = await Call(gw.app, "POST", f"{BASE}/chat/completions", headers=AUTH,
                   chunks=[b"x" * (8 * 1024 * 1024)] * 3).start().done()  # chunked, no Content-Length
    assert c.status == 413
    assert gw.ctl.calls == [] and gw.engine.requests == []
    ok = json.dumps({**CHAT, "pad": "y" * (MAX_BODY_BYTES - 200)}).encode()
    assert len(ok) <= MAX_BODY_BYTES
    c = await Call(gw.app, "POST", f"{BASE}/chat/completions", ok, AUTH).start().done()
    assert c.status == 200 and gw.engine.requests[0].content == ok


async def test_unknown_always_on_and_system_deployments(gw):
    c = await call(gw.app, "POST", "/dgxmm/ondemand/9/v1/chat/completions", CHAT)
    assert c.status == 404 and c.json()["error"]["code"] == "not_found"
    gw.dep.add(2, "sys", state="ready", mode="always_on", system=True)
    for path in ("chat/completions", "models", "health"):
        c = await call(gw.app, "POST" if path == "chat/completions" else "GET", f"/dgxmm/ondemand/2/v1/{path}",
                       CHAT if path == "chat/completions" else None)
        assert c.status == 503 and c.json()["error"]["code"] == "not_on_demand", path
    gw.dep.add(3, "b", state="stopped", mode="always_on")
    c = await call(gw.app, "POST", "/dgxmm/ondemand/3/v1/chat/completions", CHAT)
    assert c.status == 503 and c.json()["error"]["code"] == "not_on_demand"
    gw.dep.deps[3]["state"] = "ready"  # switched to always on while LiteLLM still points here: still answered
    c = await call(gw.app, "POST", "/dgxmm/ondemand/3/v1/chat/completions", CHAT)
    assert c.status == 200
    assert str(gw.engine.requests[-1].url) == "http://engine-b:8003/v1/chat/completions"
    assert gw.ctl.calls == [] and len(gw.engine.requests) == 1


# --- what wakes a model ---------------------------------------------------------------------------

async def test_models_answered_without_waking(gw):
    c = await call(gw.app, "GET", f"{BASE}/models")
    assert c.status == 200
    assert c.json() == {"object": "list", "data": [{"id": "org/a", "object": "model", "created": 1792836000,
                                                    "owned_by": "dgxmm"}]}
    gw.dep.deps[1]["state"] = "ready"
    assert (await call(gw.app, "GET", f"{BASE}/models")).json()["data"][0]["id"] == "org/a"
    assert gw.ctl.calls == [] and gw.engine.requests == []


async def test_get_health_never_wakes_returns_503_when_down(gw):
    c = await call(gw.app, "GET", f"{BASE}/models/org/a")  # the one passthrough: only while ready
    assert c.status == 503 and c.json()["error"]["code"] == "ondemand_not_ready"
    for method, path in (("GET", "health"), ("GET", "chat/completions"), ("POST", "tokenize"),
                         ("DELETE", "embeddings")):  # not on the allowlist: never proxied, never wakes
        c = await call(gw.app, method, f"{BASE}/{path}", {} if method == "POST" else None)
        assert c.status == 404 and c.json() == NOT_FOUND, path
    assert gw.ctl.calls == [] and gw.engine.requests == []
    gw.dep.deps[1]["state"] = "ready"
    gw.engine.reply = lambda req: httpx.Response(200, text="ok", headers={"x-engine": "1"})
    c = await call(gw.app, "GET", f"{BASE}/models/org/a?verbose=1")
    assert c.status == 200 and c.body == b"ok" and c.response_headers["x-engine"] == "1"
    assert str(gw.engine.requests[0].url) == "http://engine-a:8001/v1/models/org/a?verbose=1"
    c = await call(gw.app, "GET", f"{BASE}/health")
    assert c.status == 404
    assert gw.ctl.calls == [] and len(gw.engine.requests) == 1  # proxied, but no slot: never counts as traffic


async def test_engine_admin_and_dev_endpoints_never_reachable(gw):
    gw.dep.deps[1]["state"] = "ready"
    for method, path in (("POST", "sleep?level=1"), ("POST", "wake_up"), ("POST", "reset_prefix_cache"),
                         ("POST", "collective_rpc"), ("GET", "metrics"), ("GET", "health"), ("POST", "tokenize"),
                         ("POST", "detokenize"), ("GET", "models/org/a/extra/more"), ("POST", "models/org/a")):
        c = await call(gw.app, method, f"{BASE}/{path}", {} if method == "POST" else None)
        assert c.status == 404 and c.json() == NOT_FOUND, path
    for path in ("/dgxmm/ondemand/1/sleep", "/dgxmm/ondemand/1/health", "/dgxmm/ondemand/1/v2/models"):
        c = await call(gw.app, "POST", path, {})
        assert c.status in (404, 405), path
    assert gw.ctl.calls == [] and gw.engine.requests == []


async def test_traversal_and_encoded_paths_rejected(gw):
    gw.dep.deps[1]["state"] = "ready"
    for path in ("x/../chat/completions", "../sleep?level=1", "./chat/completions", "chat/./completions",
                 "models//org", "models/org/a/", "models/a%2Fb", "chat%2Fcompletions", "%2e%2e/sleep",
                 "%2E%2E/sleep", "models/a%5Cb", "models/a\\b", "models/a%23b", "models/a%3Fb", "models/a%00b",
                 "models/%2e"):
        for method in ("GET", "POST"):
            c = await call(gw.app, method, f"{BASE}/{path}", {} if method == "POST" else None)
            assert c.status == 400, (method, path)
            assert c.json()["error"]["type"] == "invalid_request_error"
            assert c.json()["error"]["code"] == "invalid_path"
    c = await call(gw.app, "POST", "/dgxmm/ondemand/1/v1/x/..%2Fchat/completions", CHAT)
    assert c.status == 400
    assert gw.ctl.calls == [] and gw.engine.requests == []


# --- streaming ------------------------------------------------------------------------------------

async def test_stream_passthrough_byte_identical(gw):
    gw.dep.deps[1]["state"] = "ready"
    parts = [b'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi \xc3\xa9"}}]}\n\n',
             b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n',
             b"data: [DONE]\n\n"]
    gate = asyncio.Event()

    async def body():
        yield parts[0]
        await gate.wait()  # the first chunk must reach the client while the engine is still generating
        yield parts[1]
        yield parts[2]
    gw.engine.reply = lambda req: httpx.Response(200, content=body(), headers={
        "content-type": "text/event-stream; charset=utf-8", "x-request-id": "r-1"})
    payload = {**CHAT, "stream": True}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    c = Call(gw.app, "POST", f"{BASE}/chat/completions?x=1", raw,
             {**AUTH, "content-type": "application/json", "x-trace": "t-9", "cookie": "s=1"}).start()
    await _until_sent(c, 1)
    assert c.chunks == [parts[0]] and not c.finished
    assert c.status == 200
    assert c.response_headers["content-type"] == "text/event-stream; charset=utf-8"
    assert c.response_headers["x-request-id"] == "r-1"
    assert gw.ctl.slots[0].released is False  # held while the body is still being sent
    gate.set()
    await c.done()
    assert c.chunks == parts and c.body == b"".join(parts)
    assert gw.ctl.slots[0].released is True
    req = gw.engine.requests[0]
    assert (req.method, str(req.url)) == ("POST", "http://engine-a:8001/v1/chat/completions?x=1")
    assert req.content == raw
    assert req.headers["x-trace"] == "t-9" and req.headers["content-type"] == "application/json"
    assert "authorization" not in req.headers and "cookie" not in req.headers  # the key never reaches the engine


async def test_stream_queued_notice_then_answer(gw):
    gate = asyncio.Event()
    gw.ctl.script = _queued_until(gate)
    answer = b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    gw.engine.reply = lambda req: httpx.Response(200, content=answer, headers={"content-type": "text/event-stream"})
    c = Call(gw.app, "POST", f"{BASE}/chat/completions", json.dumps({**CHAT, "stream": True}).encode(),
             {**AUTH, "content-type": "application/json"}).start()
    await _until_sent(c, 1)
    assert c.status == 200 and c.response_headers["content-type"].startswith("text/event-stream")
    assert gw.engine.requests == [] and not c.finished
    notice = NOTICE.format(model="m-a")
    assert notice == ("⏳ Your message is queued — waiting for memory to free up for m-a. It will start "
                      "automatically.\n\n")
    first = _events(c.chunks[0])[0]
    assert first["id"] == "ondemand-queued" and first["object"] == "chat.completion.chunk"
    assert first["model"] == "org/a"
    assert first["choices"] == [{"index": 0, "delta": {"role": "assistant", "content": notice},
                                 "finish_reason": None}]
    gate.set()
    await c.done()
    assert c.body == c.chunks[0] + answer  # the engine's stream follows unchanged

    gate = asyncio.Event()
    gw.ctl.script = _queued_until(gate)
    gw.dep.deps[1]["state"] = "stopped"
    gw.engine.reply = lambda req: httpx.Response(200, content=b"data: [DONE]\n\n",
                                                 headers={"content-type": "text/event-stream"})
    c = Call(gw.app, "POST", f"{BASE}/completions", json.dumps({"prompt": "x", "stream": True}).encode(),
             AUTH).start()
    await _until_sent(c, 1)
    first = _events(c.chunks[0])[0]
    assert first["object"] == "text_completion" and first["model"] == "org/a"
    assert first["choices"][0]["index"] == 0 and first["choices"][0]["text"] == notice
    gate.set()
    await c.done()
    assert c.body.endswith(b"data: [DONE]\n\n")


async def test_stream_start_without_queue_sends_no_notice(gw):
    gate = asyncio.Event()

    async def start_only(dep_id, on_queued):
        await gate.wait()  # waiting for a start that fits: not a queue
    gw.ctl.script = start_only
    gw.engine.reply = lambda req: httpx.Response(200, content=b"data: [DONE]\n\n",
                                                 headers={"content-type": "text/event-stream"})
    c = Call(gw.app, "POST", f"{BASE}/chat/completions", json.dumps({**CHAT, "stream": True}).encode(), AUTH).start()
    await asyncio.sleep(0.05)
    assert c.messages == []
    gate.set()
    await c.done()
    assert c.body == b"data: [DONE]\n\n"


async def test_stream_timeout_after_notice_sends_error_event_and_done(gw):
    async def queued_then_timeout(dep_id, on_queued):
        await on_queued()
        raise OnDemandTimeout("m-a")
    gw.ctl.script = queued_then_timeout
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", {**CHAT, "stream": True})
    assert c.status == 200 and c.finished
    events = _events(c.body)
    assert len(events) == 3 and events[0]["id"] == "ondemand-queued"
    assert events[1] == TIMEOUT_BODY and events[2] == "[DONE]"
    assert c.body.endswith(_sse(TIMEOUT_BODY) + b"data: [DONE]\n\n")
    assert gw.engine.requests == []


# --- non-streaming --------------------------------------------------------------------------------

async def test_nonstream_queued_notice_prepended(gw):
    notice = NOTICE.format(model="m-a")
    raw = json.dumps(_completion("hello")).encode()
    gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)  # started, never queued: untouched
    assert c.body == raw

    gate = asyncio.Event()
    gw.ctl.script = _queued_until(gate)
    pending = Call(gw.app, "POST", f"{BASE}/chat/completions", json.dumps(CHAT).encode(), AUTH).start()
    await asyncio.sleep(0.05)
    assert pending.messages == []  # held open: nothing is sent before the answer
    gate.set()
    c = await pending.done()
    assert c.status == 200
    out = c.json()
    assert out["choices"][0]["message"]["content"] == notice + "hello"
    assert {k: v for k, v in out.items() if k != "choices"} == {k: v for k, v in _completion("x").items()
                                                               if k != "choices"}
    assert int(c.response_headers["content-length"]) == len(c.body)

    done = asyncio.Event()
    done.set()
    gw.ctl.script = _queued_until(done)
    gw.engine.reply = lambda req: httpx.Response(200, json={"object": "text_completion",
                                                            "choices": [{"index": 0, "text": " world"}]})
    c = await call(gw.app, "POST", f"{BASE}/completions", {"prompt": "x"})
    assert c.json()["choices"][0]["text"] == notice + " world"


async def test_embeddings_never_modified(gw):
    done = asyncio.Event()
    done.set()
    gw.ctl.script = _queued_until(done)
    raw = b'{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}'
    gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    c = await call(gw.app, "POST", f"{BASE}/embeddings", {"model": "org/a", "input": "x", "stream": True})
    assert c.status == 200 and c.body == raw and c.chunks == [raw]
    assert gw.ctl.calls == [1] and gw.ctl.slots[0].released


# --- errors ---------------------------------------------------------------------------------------

async def test_timeout_503_with_retry_after_and_body(gw):
    async def timeout(dep_id, on_queued):
        raise OnDemandTimeout("m-a")
    gw.ctl.script = timeout
    for payload in (CHAT, {**CHAT, "stream": True}):  # no notice sent yet: a plain 503 either way
        c = await call(gw.app, "POST", f"{BASE}/chat/completions", payload)
        assert c.status == 503
        assert c.response_headers["retry-after"] == "60"
        assert c.response_headers["content-type"] == "application/json"
        assert c.json() == TIMEOUT_BODY
    assert gw.engine.requests == []


async def test_unavailable_maps_to_503(gw):
    for code, message in (("does_not_fit", "m-a needs 90.0 GiB but at most 60.0 GiB can be freed for on-demand "
                                           "models"),
                          ("start_failed", "a could not start: CUDA error"),
                          ("ondemand_error", "the on-demand controller could not serve this request; see the DGXMM "
                                             "log")):
        async def refuse(dep_id, on_queued, code=code, message=message):
            raise OnDemandUnavailable(code, message)
        gw.ctl.script = refuse
        c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
        assert c.status == 503
        assert c.json() == {"error": {"message": message, "type": "server_error", "code": code}}
        assert "retry-after" not in c.response_headers

    async def queued_then_refuse(dep_id, on_queued):
        await on_queued()
        raise OnDemandUnavailable("start_failed", "a could not start: CUDA error")
    gw.ctl.script = queued_then_refuse
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", {**CHAT, "stream": True})
    events = _events(c.body)
    assert events[1] == {"error": {"message": "a could not start: CUDA error", "type": "server_error",
                                   "code": "start_failed"}} and events[2] == "[DONE]"
    assert gw.engine.requests == []


async def test_upstream_error_passthrough(gw):
    gw.dep.deps[1]["state"] = "ready"
    for status in (400, 404, 500):
        raw = json.dumps({"object": "error", "message": f"bad {status}", "code": status}).encode()
        gw.engine.reply = lambda req, raw=raw, status=status: httpx.Response(
            status, content=raw, headers={"content-type": "application/json", "x-engine": "e"})
        for payload in (CHAT, {**CHAT, "stream": True}):
            c = await call(gw.app, "POST", f"{BASE}/chat/completions", payload)
            assert (c.status, c.body) == (status, raw)
            assert c.response_headers["content-type"] == "application/json"
            assert c.response_headers["x-engine"] == "e"
    assert all(s.released for s in gw.ctl.slots) and len(gw.ctl.slots) == 6

    gate = asyncio.Event()  # queued and streaming: the status is already sent, so the error becomes an SSE event
    gw.ctl.script = _queued_until(gate)
    gate.set()
    gw.engine.reply = lambda req: httpx.Response(400, json={"error": {"message": "too long", "type": "BadRequest",
                                                                      "code": 400}})
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", {**CHAT, "stream": True})
    events = _events(c.body)
    assert events[1] == {"error": {"message": "too long", "type": "BadRequest", "code": 400}}
    assert events[2] == "[DONE]"


async def test_upstream_unreachable_502(gw):
    gw.dep.deps[1]["state"] = "ready"

    def refuse(req):
        raise httpx.ConnectError("connection refused", request=req)
    gw.engine.reply = refuse
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
    assert c.status == 502 and c.json()["error"]["code"] == "ondemand_upstream_error"
    assert gw.ctl.slots[0].released


async def test_key_and_body_never_logged(gw, caplog):
    caplog.set_level(logging.DEBUG)
    gw.dep.deps[1]["state"] = "ready"
    await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
    await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers={"authorization": f"Bearer {KEY}-no"})

    def refuse(req):
        raise httpx.ReadTimeout("read timeout", request=req)
    gw.engine.reply = refuse
    await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)

    async def boom(dep_id, on_queued):
        raise OnDemandUnavailable("ondemand_error", "the on-demand controller could not serve this request")
    gw.ctl.script = boom
    await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
    text = caplog.text + " ".join(str(r.args) for r in caplog.records)
    assert KEY not in text and SECRET not in text


# --- hang-ups -------------------------------------------------------------------------------------

class GatewayDeployer(ControllerDeployer):
    """The controller tests' fake deployer plus what the gateway reads (port, served name, base_url)."""

    def view(self, dep_id: int) -> dict:
        out = super().view(dep_id)
        out.update(engine="vllm", port=8000 + dep_id, served_name=f"org/{out['name']}", created_at=None)
        return out

    def base_url(self, name: str, port: int, engine: str | None = None) -> str:
        return f"http://engine-{name}:{port}"


@pytest.fixture
def real(ctx, settings, tmp_path):
    """The real controller over a fake deployer: 100 GB, a (60 GB) recently used and b (60 GB) stopped."""
    _key_file(settings, tmp_path)
    clock = Clock()
    dep = GatewayDeployer()

    async def refresh():
        return SimpleNamespace(free_for_new=dep.free())
    ctx.deployer = dep
    ctx.metrics = FakeMetrics()
    ctx.budget = SimpleNamespace(refresh=refresh, current=lambda: SimpleNamespace(usable=dep.total))
    ctx.litellm = SimpleNamespace(trigger=lambda: None)
    ctl = OnDemandController(ctx, clock=clock, wall=lambda: 1_700_000_000.0 + clock.t, retry_s=0.005,
                             settle_poll_s=0.001)
    ctl.attach(dep)
    dep.ondemand_status = ctl.status
    ctx.ondemand = ctl
    ctx.ondemand_peers = POLICY
    engine = FakeEngine()
    app = create_app(ctx)
    app.state.ondemand_gateway = Gateway(ctx, transport=httpx.MockTransport(engine.handle))
    dep.add(1, "a", 60, state="ready")
    dep.add(2, "b", 60)
    return SimpleNamespace(app=app, dep=dep, ctl=ctl, clock=clock, engine=engine,
                           od=SimpleNamespace(ctl=ctl, dep=dep, clock=clock))


async def test_disconnect_while_queued_releases_entry(real):
    await use(real.od, 1)  # a was just used: b has to queue for its memory
    c = Call(real.app, "POST", "/dgxmm/ondemand/2/v1/chat/completions",
             json.dumps({**CHAT, "stream": True}).encode(), AUTH).start()
    await _until_sent(c, 1)  # the queued notice
    assert real.ctl.status(2)["queued"] == 1
    plain = Call(real.app, "POST", "/dgxmm/ondemand/2/v1/chat/completions", json.dumps(CHAT).encode(), AUTH).start()
    await until(lambda: real.ctl.status(2)["queued"] == 2)
    c.hang_up()
    plain.hang_up()
    await c.done()
    await plain.done()
    assert real.ctl.status(2)["queued"] == 0 and real.ctl.status(2)["inflight"] == 0
    real.clock.advance(10_000)  # a is long idle now, but nobody waits for b any more
    await asyncio.sleep(0.05)
    assert real.dep.restarts == [] and real.dep.stops == []
    assert real.engine.requests == []


async def test_disconnect_while_waiting_for_start_releases_waiter(gw):
    gate = asyncio.Event()

    async def starting(dep_id, on_queued):
        await gate.wait()
    gw.ctl.script = starting
    c = Call(gw.app, "POST", f"{BASE}/chat/completions", json.dumps(CHAT).encode(), AUTH).start()
    await until(lambda: gw.ctl.calls == [1])
    c.hang_up()
    await c.done()
    assert gw.ctl.cancelled == [1] and gw.ctl.slots == [] and gw.engine.requests == []


async def test_disconnect_while_streaming_cancels_upstream_and_releases_slot(gw):
    gw.dep.deps[1]["state"] = "ready"
    upstream = SimpleNamespace(closed=False)

    async def body():
        try:
            yield b'data: {"x":1}\n\n'
            await asyncio.Event().wait()  # generating forever
        finally:
            upstream.closed = True
    gw.engine.reply = lambda req: httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})
    c = Call(gw.app, "POST", f"{BASE}/chat/completions", json.dumps({**CHAT, "stream": True}).encode(), AUTH).start()
    await _until_sent(c, 1)
    assert gw.ctl.slots[0].released is False
    c.hang_up()
    await c.done()
    await until(lambda: upstream.closed)
    assert gw.ctl.slots[0].released is True


async def test_slot_released_when_the_grant_races_a_hang_up(gw):
    """acquire() hands out a slot in the same moment the client hangs up: the slot is never leaked."""
    for delay in (0, 1, 2, 3):
        c = Call(gw.app, "POST", f"{BASE}/chat/completions", json.dumps(CHAT).encode(), AUTH)

        async def grant_as_client_leaves(dep_id, on_queued, delay=delay):
            c.hang_up()
            for _ in range(delay):
                await asyncio.sleep(0)
        gw.ctl.script = grant_as_client_leaves
        await c.start().done()
    assert len(gw.ctl.slots) + len(gw.ctl.cancelled) == 4
    assert all(s.released for s in gw.ctl.slots)


# --- who may call the gateway (fix round 1) -------------------------------------------------------

async def test_peer_outside_docker_subnets_gets_404(gw, settings, tmp_path):
    for peer in ("203.0.113.20", "127.0.0.1", BRIDGE_GW, "192.0.2.5", "::1", "testclient"):
        for method, path, payload in (("POST", f"{BASE}/chat/completions", CHAT), ("GET", f"{BASE}/models", None),
                                      ("GET", "/dgxmm/ondemand/abc/v1/x", None), ("GET", f"{BASE}/../x", None)):
            c = await call(gw.app, method, path, payload, client=peer)
            assert c.status == 404 and c.json() == NOT_FOUND, (peer, path)
    _key_file(settings, tmp_path, None)  # before the key check: not even "not configured" is revealed
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, client="203.0.113.20")
    assert c.status == 404
    gw.ctx.ondemand_peers = None  # subnets not known (yet): nobody
    c = await call(gw.app, "GET", f"{BASE}/models")
    assert c.status == 404
    gw.ctx.ondemand_peers = POLICY
    gw.ctx.untrusted_proxy_networks = [ipaddress.ip_network("198.51.100.0/28")]  # the decider's network: never
    assert (await call(gw.app, "GET", f"{BASE}/models", client="198.51.100.7")).status == 404
    gw.ctx.untrusted_proxy_networks = []
    _key_file(settings, tmp_path)
    assert (await call(gw.app, "GET", f"{BASE}/models")).status == 200
    assert gw.ctl.calls == [] and gw.engine.requests == []


async def test_forwarded_requests_get_404(gw):
    for header in ({"x-forwarded-for": "203.0.113.9"}, {"forwarded": "for=203.0.113.9"},
                   {"x-real-ip": "203.0.113.9"}, {"X-Forwarded-For": PEER}):
        c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers={**AUTH, **header})
        assert c.status == 404 and c.json() == NOT_FOUND, header
    assert gw.ctl.calls == []


async def test_peer_check_uses_the_raw_peer(gw):
    # the TCP listener's proxy-headers handling may rewrite scope["client"]; the raw peer it recorded decides
    c = await call(gw.app, "GET", f"{BASE}/models", extra={RAW_PEER_KEY: "127.0.0.1"})
    assert c.status == 404
    c = await call(gw.app, "GET", f"{BASE}/models", client="127.0.0.1", extra={RAW_PEER_KEY: PEER})
    assert c.status == 200


@pytest.mark.parametrize("trusted", ["127.0.0.1,::1", "198.51.100.0/24,127.0.0.1", "*"])
def test_peer_check_works_whatever_trusted_proxies_says(gw, trusted):
    app = tcp_app(gw.app, Settings(trusted_proxies=trusted), gw.ctx)
    inside = TestClient(app, base_url="http://testserver", client=(PEER, 40000))
    assert inside.get(f"{BASE}/models", headers=AUTH).status_code == 200
    host = TestClient(app, base_url="http://testserver", client=("127.0.0.1", 40000))
    for headers in ({}, {"x-forwarded-for": PEER}, {"x-forwarded-for": PEER, "x-forwarded-proto": "https"}):
        r = host.get(f"{BASE}/models", headers={**AUTH, **headers})
        assert r.status_code == 404 and r.json() == NOT_FOUND
    lan = TestClient(app, base_url="http://testserver", client=("203.0.113.20", 40000))
    assert lan.get(f"{BASE}/models", headers={**AUTH, "x-forwarded-for": PEER}).status_code == 404
    gw_peer = TestClient(app, base_url="http://testserver", client=(BRIDGE_GW, 40000))
    assert gw_peer.get(f"{BASE}/models", headers=AUTH).status_code == 404
    assert host.get("/dgxmm/api/auth/me").status_code != 404  # the rest of the app is not affected


def test_peer_policy_auto_and_explicit():
    nets = [{"name": "dgxmm", "subnets": ["198.51.100.0/24", "fd00:89::/64"], "gateways": ["198.51.100.254"]},
            {"name": "dgxmm-decider", "subnets": ["192.0.2.128/25"], "gateways": ["192.0.2.129"]}]
    s = Settings()
    assert s.ondemand_allowed_cidrs == "auto"
    pol = peer_policy(s, nets)
    assert pol.allows("198.51.100.5") and pol.allows("::ffff:198.51.100.5") and pol.allows("fd00:89::5")
    assert not pol.allows("198.51.100.254")  # the network's gateway (docker-proxy, hairpin NAT)
    assert not pol.allows("198.51.100.1") and not pol.allows("fd00:89::1")  # first host: docker's default gateway
    assert not pol.allows("192.0.2.133")  # the decider's network is never allowed
    assert not pol.allows("203.0.113.20") and not pol.allows(None) and not pol.allows("testclient")
    s = Settings(ondemand_allowed_cidrs="203.0.113.0/25, 203.0.113.128/26")
    pol = peer_policy(s, nets)
    assert pol.allows("203.0.113.4") and pol.allows("203.0.113.129") and not pol.allows("198.51.100.5")
    assert Settings(ondemand_allowed_cidrs="").ondemand_allowed_cidrs == "auto"
    with pytest.raises(ValidationError):
        Settings(ondemand_allowed_cidrs="192.0.2.0/33")
    with pytest.raises(ValidationError):
        Settings(ondemand_allowed_cidrs="nope")


def test_peer_policy_explicit_still_excludes_gateways_and_decider(caplog):
    """An explicit ondemand_allowed_cidrs never re-admits a network gateway / first host (docker-proxy, hairpin NAT)
    or the decider's network: they are excluded anyway, with one warning per policy."""
    nets = [{"name": "dgxmm", "subnets": ["198.51.100.0/24"], "gateways": ["198.51.100.254"]},
            {"name": "dgxmm-decider", "subnets": ["192.0.2.128/25"], "gateways": ["192.0.2.129"]}]
    s = Settings(ondemand_allowed_cidrs="198.51.100.0/24,192.0.2.0/24")
    caplog.set_level(logging.WARNING)
    pol = peer_policy(s, nets)
    assert pol.allows("198.51.100.5") and pol.allows("192.0.2.5")
    assert not pol.allows("198.51.100.254") and not pol.allows("198.51.100.1")
    assert not pol.allows("192.0.2.129") and not pol.allows("192.0.2.133")  # the decider's network
    assert "ondemand_allowed_cidrs" in caplog.text and "excluded" in caplog.text
    caplog.clear()
    peer_policy(s, nets)  # refreshed every 30 s: the same policy does not warn again
    assert "ondemand_allowed_cidrs" not in caplog.text
    # a list that does not cover them: no warning, nothing changes
    pol = peer_policy(Settings(ondemand_allowed_cidrs="203.0.113.0/25"), nets)
    assert "ondemand_allowed_cidrs" not in caplog.text and pol.allows("203.0.113.4")



async def test_context_wiring_excludes_gateways_with_an_explicit_list(settings, monkeypatch, caplog):
    """The production wiring (context.build_context, refreshed every 30 s) reads DGXMM's attached networks whatever
    ondemand_allowed_cidrs says, so an explicit list that covers a network's gateway / first host or the decider's
    network still refuses them, with one warning. The key check is never reached for them."""
    from dgxmm import context
    from tests.fakes import FakeDockerClient
    nets = [{"name": "dgxmm", "subnets": ["198.51.100.0/24"], "gateways": ["198.51.100.254"]},
            {"name": "dgxmm-decider", "subnets": ["192.0.2.128/25"], "gateways": ["192.0.2.129"]}]
    docker = DockerController(FakeDockerClient())
    docker.own_networks = lambda: nets
    docker.network_subnets = lambda name: ["192.0.2.128/25"] if name == "dgxmm-decider" else []
    monkeypatch.setattr(DockerController, "from_env", classmethod(lambda cls, *a, **kw: docker))
    monkeypatch.setattr("dgxmm.ondemand.gateway._warned_overlaps", set())  # warned once per process
    settings.decider_network = "dgxmm-decider"
    settings.ondemand_allowed_cidrs = "198.51.100.0/24,192.0.2.0/24"
    caplog.set_level(logging.WARNING)
    ctx = context.build_context(settings)
    try:
        refresh = next(f for f in ctx.startup if f.__name__ == "refresh_untrusted_proxies")
        await refresh()
        app = create_app(ctx)
        for peer in ("198.51.100.254", "198.51.100.1", "192.0.2.129", "192.0.2.133"):
            c = await call(app, "POST", f"{BASE}/chat/completions", CHAT, client=peer)
            assert c.status == 404 and c.json() == NOT_FOUND, peer
        c = await call(app, "POST", f"{BASE}/chat/completions", CHAT, client="198.51.100.5")
        assert c.json() != NOT_FOUND  # an allowed peer gets past the guard (to the key check)
        assert ctx.ondemand_peers.allows("192.0.2.5") and ctx.ondemand_peers.allows("198.51.100.5")
        assert caplog.text.count("ondemand_allowed_cidrs covers") == 1
        await refresh()  # the 30 s refresh does not warn again
        assert caplog.text.count("ondemand_allowed_cidrs covers") == 1
    finally:
        await ctx.http.aclose()

def test_own_networks_from_docker():
    import docker as docker_sdk
    me = SimpleNamespace(attrs={"NetworkSettings": {"Networks": {
        "dgxmm": {"Gateway": "198.51.100.1", "IPAddress": "198.51.100.2", "IPPrefixLen": 24, "IPv6Gateway": ""},
        "orphan": {"Gateway": "192.0.2.1", "IPAddress": "192.0.2.3", "IPPrefixLen": 16}}}})
    ipam = {"dgxmm": SimpleNamespace(attrs={"IPAM": {"Config": [{"Subnet": "198.51.100.0/24", "Gateway": "198.51.100.1"},
                                                               {"Subnet": "fd00:89::/64"}]}})}

    def net_get(name):
        if name not in ipam:
            raise docker_sdk.errors.NotFound(name)
        return ipam[name]
    client = SimpleNamespace(containers=SimpleNamespace(get=lambda cid: me), networks=SimpleNamespace(get=net_get))
    d = DockerController(client)
    d.own_container_id = lambda: "self"
    assert d.own_networks() == [
        {"name": "dgxmm", "subnets": ["198.51.100.0/24", "fd00:89::/64"], "gateways": ["198.51.100.1"]},
        {"name": "orphan", "subnets": ["192.0.2.3/16"], "gateways": ["192.0.2.1"]}]
    d.own_container_id = lambda: None
    assert d.own_networks() == []  # on the host: no docker network, so nobody may call the gateway


# --- other fix-round-1 behaviour ------------------------------------------------------------------

async def test_removed_after_acquire_is_not_found(gw):
    async def removed(dep_id, on_queued):
        pass

    class Removing(FakeController):
        async def acquire(self, dep_id, *, on_queued=None):
            slot = await super().acquire(dep_id, on_queued=on_queued)
            del self.dep.deps[dep_id]  # removed between the grant and the proxying
            return slot
    gw.ctl = Removing(gw.dep)
    gw.ctx.ondemand = gw.ctl
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)
    assert c.status == 404 and c.json()["error"]["code"] == "not_found"
    gw.dep.add(1)
    gw.ctl.script = _queued_until(_set_event())
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", {**CHAT, "stream": True})
    events = _events(c.body)
    assert c.status == 200 and events[0]["id"] == "ondemand-queued"
    assert events[1]["error"]["code"] == "not_found" and events[2] == "[DONE]"
    assert all(s.released for s in gw.ctl.slots) and len(gw.ctl.slots) == 2
    assert gw.engine.requests == []


def _set_event() -> asyncio.Event:
    e = asyncio.Event()
    e.set()
    return e


async def test_failed_auth_logged_rate_limited_and_429(gw, caplog):
    caplog.set_level(logging.WARNING)
    codes = [(await call(gw.app, "GET", f"{BASE}/models", headers={"authorization": f"Bearer guess-{i}"})).status
             for i in range(AUTH_FAILURES_PER_MIN + 2)]
    assert codes == [401] * AUTH_FAILURES_PER_MIN + [429, 429]
    c = await call(gw.app, "GET", f"{BASE}/models")  # that peer is blocked for the window, right key or not
    assert c.status == 429 and c.json()["error"]["code"] == "rate_limited"
    assert c.response_headers["retry-after"] == "60"
    assert (await call(gw.app, "GET", f"{BASE}/models", client="198.51.100.6")).status == 200  # other peers unaffected
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "key" in r.getMessage()]
    assert len(warnings) == 1 and PEER in warnings[0].getMessage()
    assert "guess" not in caplog.text and KEY not in caplog.text


async def test_forwarding_and_connection_headers_stripped(gw):
    gw.dep.deps[1]["state"] = "ready"
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers={
        **AUTH, "connection": "keep-alive, X-Hop-Secret", "x-hop-secret": "1", "x-forwarded-proto": "https",
        "x-forwarded-host": "example.test", "x-forwarded-port": "443", "keep-alive": "timeout=5",
        "proxy-authorization": "Basic x", "te": "trailers", "x-keep": "yes"})
    assert c.status == 200
    sent = gw.engine.requests[0].headers
    for h in ("x-hop-secret", "x-forwarded-proto", "x-forwarded-host", "x-forwarded-port", "keep-alive",
              "proxy-authorization", "te", "authorization"):
        assert h not in sent, h
    assert sent["x-keep"] == "yes"


async def test_client_rotation_keeps_streams_and_tracks_close_tasks(gw):
    g = Gateway(gw.ctx, transport=httpx.MockTransport(gw.engine.handle))
    old = g.lease(1, "http://engine-a:8001")
    assert g.lease(1, "http://engine-a:8001") is old
    g.unlease(old)
    new = g.lease(1, "http://engine-a:8002")  # restarted on a new port while a stream runs on the old client
    assert new is not old
    await asyncio.sleep(0.01)
    assert not old.is_closed  # its stream is still running
    g.unlease(old)
    await until(lambda: old.is_closed)
    await until(lambda: not g._closing)  # close tasks are referenced until done, then dropped
    g.unlease(new)
    await g.aclose()
    assert new.is_closed


# --- every inference endpoint the engines serve under /v1 -----------------------------------------

RESPONSES_REQ = {"model": "org/a", "input": SECRET}


def _response(text: str) -> dict:
    """A non-streaming Responses API answer: a reasoning item, then the message with its output_text part."""
    return {"id": "resp_1", "object": "response", "model": "org/a", "status": "completed",
            "output": [{"id": "rs_1", "type": "reasoning", "summary": []},
                       {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []},
                                    {"type": "output_text", "text": "second", "annotations": []}]}]}


@pytest.mark.parametrize("endpoint", ["responses", "messages", "chat/completions/batch", "audio/transcriptions",
                                      "audio/translations", "rerank", "score"])
async def test_every_engine_inference_endpoint_is_traffic(gw, endpoint):
    """LiteLLM translates Anthropic-format clients to the Responses API (POST {api_base}/responses), and vLLM serves
    more inference endpoints under /v1: each one wakes the model, holds a slot and is forwarded unchanged."""
    raw = b'{"ok":true}'
    gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    c = await call(gw.app, "POST", f"{BASE}/{endpoint}?q=1", {"model": "org/a", "input": "x"})
    assert c.status == 200 and c.body == raw
    assert gw.ctl.calls == [1] and gw.ctl.slots[0].released
    req = gw.engine.requests[0]
    assert str(req.url) == f"http://engine-a:8001/v1/{endpoint}?q=1" and "authorization" not in req.headers


async def test_multipart_audio_forwarded_unchanged(gw):
    body = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\nRIFF\x00\x01\r\n'
            b"--b--\r\n")
    gw.engine.reply = lambda req: httpx.Response(200, json={"text": "hi"})
    c = await Call(gw.app, "POST", f"{BASE}/audio/transcriptions", body,
                   {**AUTH, "content-type": "multipart/form-data; boundary=b"}).start().done()
    assert c.status == 200 and c.json() == {"text": "hi"}
    assert gw.engine.requests[0].content == body
    assert gw.engine.requests[0].headers["content-type"] == "multipart/form-data; boundary=b"


async def test_responses_stream_never_gets_a_notice(gw):
    """A Responses (or Anthropic messages) stream has typed events that must start with response.created, so a
    queued request gets no notice: it waits, then gets the engine's stream exactly."""
    answer = (b'event: response.created\ndata: {"type":"response.created","sequence_number":0}\n\n'
              b'event: response.completed\ndata: {"type":"response.completed","sequence_number":1}\n\n')
    for endpoint in ("responses", "messages"):
        gate = asyncio.Event()
        gw.ctl.script = _queued_until(gate)
        gw.dep.deps[1]["state"] = "stopped"
        gw.engine.reply = lambda req: httpx.Response(200, content=answer,
                                                     headers={"content-type": "text/event-stream"})
        c = Call(gw.app, "POST", f"{BASE}/{endpoint}", json.dumps({**RESPONSES_REQ, "stream": True}).encode(),
                 {**AUTH, "content-type": "application/json"}).start()
        await asyncio.sleep(0.05)
        assert c.messages == [], endpoint  # queued: nothing sent yet
        gate.set()
        await c.done()
        assert c.status == 200 and c.body == answer, endpoint


async def test_responses_stream_timeout_is_a_plain_503(gw):
    async def queued_then_timeout(dep_id, on_queued):
        await on_queued()
        raise OnDemandTimeout("m-a")
    gw.ctl.script = queued_then_timeout
    c = await call(gw.app, "POST", f"{BASE}/responses", {**RESPONSES_REQ, "stream": True})
    assert c.status == 503 and c.json() == TIMEOUT_BODY and c.response_headers["retry-after"] == "60"


async def test_responses_nonstream_queued_notice_in_first_output_text(gw):
    notice = NOTICE.format(model="m-a")
    raw = json.dumps(_response("hello")).encode()
    gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    c = await call(gw.app, "POST", f"{BASE}/responses", RESPONSES_REQ)  # never queued: untouched
    assert c.body == raw

    done = asyncio.Event()
    done.set()
    gw.ctl.script = _queued_until(done)
    c = await call(gw.app, "POST", f"{BASE}/responses", RESPONSES_REQ)
    out = c.json()
    assert out["output"][1]["content"][0]["text"] == notice + "hello"
    assert out["output"][1]["content"][1]["text"] == "second"  # only the first output_text
    assert out["output"][0] == _response("x")["output"][0]
    assert int(c.response_headers["content-length"]) == len(c.body)

    tool_only = {"id": "resp_2", "object": "response", "output": [{"type": "function_call", "name": "f",
                                                                    "arguments": "{}", "call_id": "c1"}]}
    raw = json.dumps(tool_only).encode()
    gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    c = await call(gw.app, "POST", f"{BASE}/responses", RESPONSES_REQ)
    assert c.json() == tool_only  # no text to put it in: unchanged

    raw = b'{"id":"msg_1","type":"message","content":[{"type":"text","text":"hi"}]}'
    gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    c = await call(gw.app, "POST", f"{BASE}/messages", {"model": "org/a", "messages": []})
    assert c.body == raw  # Anthropic messages: never modified


async def test_non_inference_engine_posts_still_404(gw):
    gw.dep.deps[1]["state"] = "ready"
    for path in ("messages/count_tokens", "chat/completions/render", "completions/render", "responses/r1/cancel",
                 "load_lora_adapter", "unload_lora_adapter", "audio/speech"):
        c = await call(gw.app, "POST", f"{BASE}/{path}", {})
        assert c.status == 404 and c.json() == NOT_FOUND, path
    for method, path in (("GET", "responses"), ("GET", "responses/r1"), ("PUT", "responses"),
                         ("DELETE", "messages")):
        c = await call(gw.app, method, f"{BASE}/{path}", {} if method != "GET" else None)
        assert c.status in (404, 405), (method, path)
    assert gw.ctl.calls == [] and gw.engine.requests == []


# --- LiteLLM health checks (model_info.health_check_params adds HEALTH_HEADER) ---------------------
# Any LiteLLM client can set the header too (extra_headers in the request body), so a marked request is never
# forwarded, in any state: DGXMM answers it itself.

def _health_shape_ok(endpoint: str, out: dict) -> bool:
    """The shapes LiteLLM's health check (and its response parsing) accepts."""
    if endpoint == "chat/completions":
        return out["object"] == "chat.completion" and out["choices"][0]["message"]["role"] == "assistant"
    if endpoint == "completions":
        return out["object"] == "text_completion" and out["choices"][0]["text"] == ""
    if endpoint == "responses":
        return out["object"] == "response" and out["status"] == "completed" and \
            out["output"][0]["content"][0]["type"] == "output_text"
    if endpoint == "messages":
        return out["type"] == "message" and out["content"][0]["type"] == "text"
    if endpoint == "embeddings":
        return out["object"] == "list" and out["data"][0]["object"] == "embedding"
    return False


@pytest.mark.parametrize("state", ["ready", "stopped", "starting", "asleep", "queued"])
@pytest.mark.parametrize("endpoint", ["chat/completions", "completions", "responses", "messages", "embeddings"])
async def test_health_check_answered_locally_in_every_state(gw, state, endpoint):
    gw.dep.deps[1]["state"] = state
    c = await call(gw.app, "POST", f"{BASE}/{endpoint}", {**CHAT, "max_tokens": 16},
                   headers={**AUTH, HEALTH_HEADER: "1"})
    assert c.status == 200, c.body
    assert _health_shape_ok(endpoint, c.json())
    assert c.json()["model"] == "org/a"
    assert c.response_headers["x-dgxmm-ondemand-state"] == state
    assert gw.ctl.calls == [] and gw.engine.requests == []  # no wake, no slot, never forwarded


async def test_health_header_on_a_ready_model_never_reaches_the_engine(gw):
    """A client putting the header on a large streaming request gets the canned answer, not a generation."""
    gw.dep.deps[1]["state"] = "ready"
    big = {**CHAT, "stream": True, "max_tokens": 100000, "n": 8,
           "messages": [{"role": "user", "content": "x" * 1_000_000}]}
    for endpoint in TRAFFIC:
        c = await call(gw.app, "POST", f"{BASE}/{endpoint}", big, headers={**AUTH, HEALTH_HEADER: "1"})
        assert c.status in (200, 503), endpoint
        assert not c.response_headers["content-type"].startswith("text/event-stream"), endpoint
    assert gw.ctl.calls == [] and gw.engine.requests == [] and gw.ctl.slots == []
    gw.dep.deps[1]["mode"] = "always_on"  # switched while LiteLLM still points here: still never forwarded
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", big, headers={**AUTH, HEALTH_HEADER: "1"})
    assert c.status == 200 and gw.engine.requests == []


async def test_health_check_failed_or_unsupported_is_503(gw):
    hdr = {**AUTH, HEALTH_HEADER: "1"}
    c = await call(gw.app, "POST", f"{BASE}/rerank", {"query": "q", "documents": ["d"]}, headers=hdr)
    assert c.status == 503 and c.json()["error"]["code"] == "ondemand_health_unsupported"
    gw.dep.deps[1]["state"] = "failed"
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=hdr)
    assert c.status == 503 and c.json()["error"]["code"] == "ondemand_not_ready"
    assert c.response_headers["x-dgxmm-ondemand-state"] == "failed"
    assert gw.ctl.calls == [] and gw.engine.requests == []


async def test_health_header_not_set_is_normal_traffic_and_never_forwarded(gw):
    gw.dep.deps[1]["state"] = "ready"
    for value in ("0", "", "yes"):
        c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers={**AUTH, HEALTH_HEADER: value})
        assert c.status == 200 and c.json() == _completion("hello")
    assert len(gw.ctl.calls) == 3  # traffic: slot taken
    assert all(HEALTH_HEADER not in r.headers for r in gw.engine.requests)  # stripped before the engine


async def test_health_header_needs_the_key(gw):
    c = await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers={HEALTH_HEADER: "1"})
    assert c.status == 401


# --- final fix wave: structured requests, queue cap, key rotation ------------------------------------------

STRUCTURED = [{"response_format": {"type": "json_object"}},
              {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}},
              {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]},
              {"tool_choice": "none"},
              {"functions": [{"name": "f", "parameters": {"type": "object"}}]},
              {"guided_json": {"type": "object"}}, {"guided_regex": "a+"}, {"guided_choice": ["a", "b"]},
              {"structured_outputs": {"json": {"type": "object"}}}]


@pytest.mark.parametrize("extra", STRUCTURED, ids=lambda e: next(iter(e)))
async def test_structured_requests_never_get_the_queued_notice(gw, extra):
    """A request that asks for JSON, tools or guided output gets no queued notice (it would break the client's
    parsing): queued, it just waits, and the answer is the engine's, byte for byte. Chat and completions, streaming
    and not."""
    sse = b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"{}"}}]}\n\ndata: [DONE]\n\n'
    raw = json.dumps(_completion('{"a": 1}')).encode()
    for endpoint, payload in (("chat/completions", {**CHAT, **extra}), ("completions", {"prompt": "x", **extra})):
        for stream in (True, False):
            gate = asyncio.Event()
            gw.ctl.script = _queued_until(gate)
            gw.dep.deps[1]["state"] = "stopped"
            body = sse if stream else raw
            ctype = "text/event-stream" if stream else "application/json"
            gw.engine.reply = lambda req, b=body, t=ctype: httpx.Response(200, content=b, headers={"content-type": t})
            c = Call(gw.app, "POST", f"{BASE}/{endpoint}", json.dumps({**payload, "stream": stream}).encode(),
                     {**AUTH, "content-type": "application/json"}).start()
            await asyncio.sleep(0.05)
            assert c.messages == [], (endpoint, stream)  # held open: no notice chunk
            gate.set()
            await c.done()
            assert c.status == 200 and c.body == body, (endpoint, stream)


async def test_responses_with_tools_or_a_json_format_get_no_notice(gw):
    raw = json.dumps({"id": "r1", "object": "response", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}).encode()
    for extra in ({"tools": [{"type": "function", "name": "f"}]}, {"text": {"format": {"type": "json_object"}}}):
        done = asyncio.Event()
        done.set()
        gw.ctl.script = _queued_until(done)
        gw.engine.reply = lambda req: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
        c = await call(gw.app, "POST", f"{BASE}/responses", {"model": "org/a", "input": "x", **extra})
        assert c.body == raw, extra
    # a plain-text format still gets it
    gw.ctl.script = _queued_until(done)
    c = await call(gw.app, "POST", f"{BASE}/responses", {"model": "org/a", "input": "x",
                                                         "text": {"format": {"type": "text"}}})
    assert c.json()["output"][0]["content"][0]["text"].startswith("⏳")


async def test_queue_full_is_503_at_once(gw):
    async def full(dep_id, on_queued):
        raise OnDemandUnavailable("ondemand_queue_full", "64 requests already wait for on-demand models "
                                                         "(ondemand_max_queued); try again later")
    gw.ctl.script = full
    for payload in (CHAT, {**CHAT, "stream": True}):
        c = await call(gw.app, "POST", f"{BASE}/chat/completions", payload)
        assert c.status == 503 and c.json()["error"]["code"] == "ondemand_queue_full"
        assert c.response_headers["retry-after"] == "60"
    assert gw.engine.requests == []


async def test_key_rotation_clears_the_auth_limit_and_the_old_key_does_not_count(gw, settings, tmp_path):
    """Until LiteLLM re-syncs, it still sends the old key: those requests get 401 but never count towards the
    failed-attempt limit (so LiteLLM's address is not blocked with 429), and a rotation clears the limit."""
    import os
    now = [100.0]
    g = gw.app.state.ondemand_gateway
    g.clock = lambda: now[0]
    gw.ctx.litellm = SimpleNamespace(trigger=lambda: None)
    for i in range(AUTH_FAILURES_PER_MIN):  # blocked before the rotation
        await call(gw.app, "GET", f"{BASE}/models", headers={"authorization": f"Bearer guess-{i}"})
    assert (await call(gw.app, "GET", f"{BASE}/models")).status == 429
    new_key = "sk-rotated-gateway-key-9876543210"
    tmp = tmp_path / "dgxmm_ondemand_key.new"
    tmp.write_text(new_key + "\n")
    os.replace(tmp, tmp_path / "dgxmm_ondemand_key")
    now[0] += 1.01
    codes = [(await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT)).status
             for _ in range(AUTH_FAILURES_PER_MIN + 5)]
    assert set(codes) == {401}  # the old key: refused, never rate limited
    new_auth = {"authorization": f"Bearer {new_key}"}
    assert (await call(gw.app, "POST", f"{BASE}/chat/completions", CHAT, headers=new_auth)).status == 200
    codes = [(await call(gw.app, "GET", f"{BASE}/models", headers={"authorization": f"Bearer guess2-{i}"})).status
             for i in range(AUTH_FAILURES_PER_MIN + 1)]
    assert codes[-1] == 429  # other wrong keys still count
