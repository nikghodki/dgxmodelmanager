"""The on-demand gateway: LiteLLM's entry for an on-demand deployment points at ``<base_path>/ondemand/{id}/v1``.

- Only peers on DGXMM's own docker networks get an answer (``PeerGuard``, before anything else): the raw TCP peer, as
  uvicorn received it and before proxy headers are applied, must be in ``ctx.ondemand_peers`` (the attached networks'
  subnets without their gateway addresses, where host-port traffic shows up, and never the decider's network), and
  the request must carry no X-Forwarded-For / Forwarded / X-Real-IP (nginx and other proxies sit on the same
  network). Anything else is 404 ``not_found``, as if the route did not exist. Paths with ``.``/``..``/empty
  segments, ``\\``, ``?``, ``#``, control characters or an encoded ``/`` are 400 ``invalid_path``.
- Every request needs ``Authorization: Bearer <on-demand key>`` (``load_ondemand_key``, compared in constant time).
  No session cookie or CSRF token is involved: this router sits outside the UI's auth. Without a key file the gateway
  fails closed (503 ``ondemand_not_configured``, logged once). The file is checked at most once a second and re-read
  when it changed (inode, mtime or size), so a rotated key works without a restart and a removed or emptied file
  fails closed again. A peer with AUTH_FAILURES_PER_MIN failed attempts in a minute gets 429 for the rest of that
  minute (one WARNING per peer per minute, never any key material). A key change clears those counts, and for
  ROTATION_GRACE_S the previous key gets 401 without counting (LiteLLM sends it until its next sync).
- ``POST`` to an inference endpoint the engines serve under /v1 (TRAFFIC: chat/completions, completions, embeddings,
  responses, messages, chat/completions/batch, audio/transcriptions, audio/translations, rerank, score) is traffic: it
  asks the controller for a slot (starting, making room or queueing), then streams to the engine. The slot is held
  until the engine's answer has been sent in full. LiteLLM sends Anthropic-format clients' requests to an ``openai/``
  entry as Responses API calls (``POST {api_base}/responses``).
- A LiteLLM health check (HEALTH_HEADER, set through the entry's ``model_info.health_check_params``) is answered by
  the gateway itself and never reaches the engine, in any state: any LiteLLM client can set the same header
  (``extra_headers``), so forwarding it would be a generation without a slot. It never wakes a model or takes a slot.
  chat/completions, completions, responses, messages and embeddings get a minimal empty answer (200, with
  ``x-dgxmm-ondemand-state``); a failed deployment is 503 ``ondemand_not_ready``, other endpoints 503
  ``ondemand_health_unsupported``.
- ``GET models`` is answered from the deployment record and never wakes a model. ``GET models/{id}`` is proxied only
  when the deployment is ready (no slot, never wakes), otherwise 503 ``ondemand_not_ready``. Nothing else reaches the
  engine: its admin and dev endpoints (``/sleep``, ``/wake_up``, ``/collective_rpc`` ...) are 404.
- Streaming (``"stream": true``): when a chat/completions or completions request has to queue for memory, the
  response starts with a ``chat.completion.chunk`` (``text_completion`` for completions) carrying NOTICE, then the
  engine's SSE bytes follow unchanged. Responses and messages streams get no notice (their typed event streams must
  open with the engine's own ``response.created`` / ``message_start``): they wait like a non-streaming request. A
  request that never queued gets the engine's response exactly: status, headers and bytes.
  Non-streaming: the connection is held; if the request queued, NOTICE is prepended to the first choice's content
  (chat/completions, completions) or the first ``output_text`` part (responses). Nothing else is ever modified.
  A request that asks for structured output (``response_format``, ``tools``, ``tool_choice``, ``functions``,
  ``function_call``, ``structured_outputs``, any ``guided_*`` field, or a Responses ``text.format`` other than text)
  never gets a notice, streaming or not: its client parses the answer. It just waits.
- Errors are OpenAI-style ``{"error": {"message", "type", "code"}}``: the queue timeout is 503 + ``Retry-After: 60``,
  ``ondemand_queue_full`` too, any other OnDemandUnavailable 503 with its code and message. Once a notice was sent the status is already 200, so
  the error becomes a final ``data: {"error": ...}`` event and ``data: [DONE]``.
- A client hang-up at any point cancels the wait (the controller drops the queue entry) or the upstream request, and
  releases the slot. Request bodies and the key are never logged.
(docs/superpowers/plans/2026-10-24-on-demand-models.md, Task 3)."""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response

from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.config import load_ondemand_key
from dgxmm.deployer.service import DeployError
from dgxmm.ondemand import HEALTH_HEADER
from dgxmm.ondemand.controller import OnDemandError, OnDemandTimeout
from dgxmm.proxy_headers import RAW_PEER_KEY, Network, in_networks, parse_networks

log = logging.getLogger("dgxmm.ondemand.gateway")
router = APIRouter(prefix="/ondemand", include_in_schema=False)

MAX_BODY_BYTES = 16 * 1024 * 1024   # request bodies above this: 413 request_too_large, nothing read further
MAX_ERROR_BYTES = 1024 * 1024       # an engine error read to turn into an SSE error event (after a queued notice)
RETRY_AFTER_S = 60
KEY_RECHECK_S = 1.0                 # the key file is stat()ed at most this often; re-read when it changed
CONNECT_TIMEOUT_S = 10.0            # to the engine; reads have no timeout (generation can be long)
POOL_TIMEOUT_S = 60.0
POOL_LIMITS = httpx.Limits(max_connections=256, max_keepalive_connections=32)  # per deployment
NOTICE = "⏳ Your message is queued — waiting for memory to free up for {model}. It will start automatically.\n\n"
AUTH_FAILURES_PER_MIN = 20         # failed key checks per peer per minute before 429
ROTATION_GRACE_S = 300.0           # after a key change, the previous key is refused without counting as a failure
# request fields asking for structured output: no queued notice (the client parses the answer)
STRUCTURED_FIELDS = frozenset({"response_format", "tools", "tool_choice", "functions", "function_call",
                               "structured_outputs"})
# POST endpoints below /v1 that are traffic: the inference endpoints vLLM serves there (llama.cpp's are a subset).
# Not here, so 404: render/tokenize-style helpers, count_tokens, LoRA loading, stored responses (GET/cancel).
TRAFFIC = ("chat/completions", "completions", "embeddings", "responses", "messages", "chat/completions/batch",
           "audio/transcriptions", "audio/translations", "rerank", "score")
STREAM_NOTICE = ("chat/completions", "completions")      # a queued stream starts with the notice chunk
BODY_NOTICE = ("chat/completions", "completions", "responses")  # a queued answer gets the notice in its text
NEVER_STREAMS = ("embeddings", "rerank", "score", "chat/completions/batch")
OTHER_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
PASSTHROUGH_METHODS = ("GET", "HEAD")  # ... for models/{id} only (passthrough_allowed)
FORWARDED = frozenset({b"x-forwarded-for", b"forwarded", b"x-real-ip"})  # a proxied request: never served
ENCODED_REFUSED = (b"%2f", b"%5c", b"%23", b"%3f", b"%00")  # an encoded / \ # ? NUL in the raw path
PCHAR_SAFE = "-._~:@!$&'()*+,;="  # characters kept as they are when a path segment is quoted again
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                        "te", "trailer", "transfer-encoding", "upgrade"})
# never forwarded to the engine: the gateway key, the UI's cookies, forwarding headers, framing httpx sets itself;
# identity encoding is asked for so the bytes passed back are the bytes the engine wrote
DROP_REQUEST = HOP_BY_HOP | {"host", "content-length", "authorization", "cookie", "accept-encoding", HEALTH_HEADER,
                             "x-forwarded-for", "x-forwarded-proto", "x-forwarded-host", "x-forwarded-port",
                             "x-forwarded-prefix", "forwarded", "x-real-ip"}
DROP_RESPONSE = HOP_BY_HOP | {"server", "date"}  # uvicorn sets its own
SSE_HEADERS = [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"),
               (b"x-accel-buffering", b"no")]


def error_body(message: str, code: str, type_: str = "server_error") -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


NOT_FOUND = error_body("Not Found", "not_found", "invalid_request_error")


# ---------- who may call the gateway ----------
@dataclass(frozen=True)
class PeerPolicy:
    """Peers allowed to call the gateway: in ``allowed``, not one of ``excluded`` (network gateways) and not in
    ``blocked`` (the decider's network)."""
    allowed: list[Network] = field(default_factory=list)
    excluded: frozenset = frozenset()
    blocked: list[Network] = field(default_factory=list)

    def allows(self, host: str | None) -> bool:
        if not in_networks(host, self.allowed) or in_networks(host, self.blocked):
            return False
        ip = ipaddress.ip_address(host)
        if getattr(ip, "ipv4_mapped", None) is not None:
            ip = ip.ipv4_mapped
        return ip not in self.excluded


_warned_overlaps: set = set()  # explicit policies already warned about (refreshed every 30 s: warn once)


def peer_policy(settings, networks: list[dict]) -> PeerPolicy:
    """``ondemand_allowed_cidrs`` as given, or (``auto``) the subnets of ``networks`` (DockerController.own_networks)
    without the decider's network. Either way each network's gateway and first host address (docker's default
    gateway) are excluded, since traffic to a published host port arrives from there (docker-proxy, hairpin NAT),
    and the decider's network is never allowed. An explicit list that covers any of them gets one warning."""
    excluded: set = set()
    subnets: list[Network] = []
    blocked: list[Network] = []
    for n in networks:
        nets = parse_networks(n.get("subnets") or [])
        if settings.decider_network and n.get("name") == settings.decider_network:
            blocked += nets
        else:
            subnets += nets
        for net in nets:
            first = next(iter(net.hosts()), None)
            if first is not None:
                excluded.add(first)
        for g in n.get("gateways") or []:
            try:
                excluded.add(ipaddress.ip_address(g))
            except ValueError:
                pass
    if settings.ondemand_allowed_cidrs == "auto":
        return PeerPolicy(subnets, frozenset(excluded), blocked)
    allowed = parse_networks(settings.ondemand_allowed_cidrs.split(","))
    covered = sorted((str(ip) for ip in excluded if in_networks(str(ip), allowed)), key=str)
    overlaps = [str(b) for b in blocked if any(b.overlaps(a) for a in allowed if a.version == b.version)]
    if covered or overlaps:
        sig = (settings.ondemand_allowed_cidrs, tuple(covered), tuple(overlaps))
        if sig not in _warned_overlaps:
            _warned_overlaps.add(sig)
            log.warning("ondemand_allowed_cidrs covers %s; excluded anyway (network gateways forward host-port "
                        "traffic, and the decider's network may never call the gateway)",
                        ", ".join(covered + [f"the decider network {o}" for o in overlaps]))
    return PeerPolicy(allowed, frozenset(excluded), blocked)


def _peer(scope) -> str | None:
    """The raw TCP peer (dgxmm.proxy_headers records it before applying X-Forwarded-For)."""
    if RAW_PEER_KEY in scope:
        return scope[RAW_PEER_KEY]
    client = scope.get("client")
    return client[0] if client else None


def clean_path(scope, prefix: str) -> bool:
    """No dot, empty or encoded-slash segments and no \\ ? # or control characters below ``prefix``."""
    raw = (scope.get("raw_path") or scope["path"].encode()).lower()
    if any(t in raw for t in ENCODED_REFUSED):
        return False
    rest = scope["path"][len(prefix) + 1:]
    if any(c in rest for c in "\\?#") or any(ord(c) < 0x20 or ord(c) == 0x7f for c in rest):
        return False
    return all(seg not in ("", ".", "..") for seg in rest.split("/"))


class PeerGuard:
    """ASGI middleware for ``<base_path>/ondemand``: 404 unless the raw peer may call the gateway and the request
    was not forwarded by a proxy; then 400 for a path that is not clean. Runs before routing and the key check."""

    def __init__(self, app, prefix: str, ctx: Any):
        self.app, self.prefix, self.ctx = app, prefix, ctx

    def _allowed(self, scope) -> bool:
        if any(k.lower() in FORWARDED for k, _ in scope.get("headers") or []):
            return False
        peer, policy = _peer(scope), self.ctx.ondemand_peers
        if policy is None or not policy.allows(peer):
            return False
        return not in_networks(peer, getattr(self.ctx, "untrusted_proxy_networks", None) or [])

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not (path == self.prefix or path.startswith(self.prefix + "/")):
            return await self.app(scope, receive, send)
        if not self._allowed(scope):
            return await JSONResponse(NOT_FOUND, status_code=404)(scope, receive, send)
        if not clean_path(scope, self.prefix):
            return await _error(400, "invalid request path", "invalid_path", "invalid_request_error")(
                scope, receive, send)
        await self.app(scope, receive, send)


def passthrough_allowed(method: str, path: str) -> bool:
    """The engine paths (below /v1) proxied while ready: ``models/{id}`` (an id may hold one ``/``)."""
    segs = path.split("/")
    return method in PASSTHROUGH_METHODS and segs[0] == "models" and 2 <= len(segs) <= 3


def _error(status: int, message: str, code: str, type_: str = "server_error",
           headers: dict | None = None) -> JSONResponse:
    return JSONResponse(error_body(message, code, type_), status_code=status, headers=headers)


def _sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


DONE = b"data: [DONE]\n\n"


def _model(v: dict) -> str:
    """The name clients use (as in OnDemandTimeout)."""
    return v.get("model_name") or v["name"]


def _served(v: dict) -> str:
    """The name the engine serves (what LiteLLM sends as ``model``)."""
    return v.get("served_name") or _model(v)


class Gateway:
    """Per-app state: the key (re-read when its file changes) and one httpx client (connection pool) per
    deployment."""

    def __init__(self, ctx: Any, transport: httpx.AsyncBaseTransport | None = None):
        self.ctx, self._transport = ctx, transport
        self._clients: dict[int, tuple[str, httpx.AsyncClient]] = {}
        self._leases: dict[httpx.AsyncClient, int] = {}  # requests using a client right now
        self._retired: set[httpx.AsyncClient] = set()     # replaced (new base URL / removed): closed when unused
        self._closing: set[asyncio.Task] = set()
        self._key: str | None = None
        self._key_stat: tuple | None = None  # (inode, mtime_ns, size) of the file the key was read from
        self._key_checked: float | None = None
        self._key_logged = False
        self.clock = time.monotonic
        self.auth_failures = SlidingWindowLimiter(AUTH_FAILURES_PER_MIN, 60)
        self._auth_log = SlidingWindowLimiter(1, 60)  # one warning per peer per minute
        self._previous_key: str | None = None  # the key before the last rotation, refused without counting ...
        self._previous_until = 0.0             # ... until this clock()
        hooks = getattr(ctx.deployer, "removal_hooks", None)
        if hooks is not None:
            hooks.append(self._forget)

    def key(self) -> str | None:
        """The current key, or None (fail closed). At most once per KEY_RECHECK_S the file is stat()ed; a new inode,
        mtime or size (rotation: replaced or rewritten in place) re-reads it, so no restart is needed, and a new key
        starts a LiteLLM sync. A missing or empty file gives None until a key is there again (logged once per
        outage, never the key)."""
        now = self.clock()
        if self._key_checked is not None and now - self._key_checked < KEY_RECHECK_S:
            return self._key
        self._key_checked = now
        path = self.ctx.settings.ondemand_key_file
        try:
            st = os.stat(path)
            sig = (st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        if sig is not None and sig == self._key_stat and self._key is not None:
            return self._key
        old, self._key, self._key_stat = self._key, None, None
        try:
            self._key = load_ondemand_key(self.ctx.settings)
        except (RuntimeError, OSError) as e:
            if not self._key_logged:
                self._key_logged = True
                log.error("on-demand gateway refuses every request: %s", e)
            return None
        self._key_stat, self._key_logged = sig, False
        if old is not None and old != self._key:
            log.info("on-demand gateway: the key file changed; using the new key")
            # LiteLLM keeps sending the old key until its sync re-registered the entries: its address must not end
            # up rate limited for that
            self.auth_failures = SlidingWindowLimiter(AUTH_FAILURES_PER_MIN, 60)
            self._previous_key, self._previous_until = old, now + ROTATION_GRACE_S
            litellm = getattr(self.ctx, "litellm", None)
            if litellm is not None:
                litellm.trigger()  # re-register the LiteLLM entries with it now, not at the next periodic sync
        return self._key

    @staticmethod
    def authorized(request: Request, key: str) -> bool:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), key.encode())

    def stale_key(self, request: Request) -> bool:
        """The request carries the key from before the last rotation (within ROTATION_GRACE_S)."""
        if self._previous_key is None or self.clock() >= self._previous_until:
            return False
        return self.authorized(request, self._previous_key)

    def auth_failed(self, peer: str | None) -> None:
        self.auth_failures.hit(peer or "-")
        if self._auth_log.hit(peer or "-"):
            log.warning("on-demand gateway: refused a request without a valid key from %s", peer or "an unknown peer")

    def lease(self, dep_id: int, base: str) -> httpx.AsyncClient:
        """The deployment's client for ``base`` (a new one when its base URL changed), held until ``unlease``."""
        held = self._clients.get(dep_id)
        if held is None or held[0] != base:
            if held is not None:
                self._retire(held[1])
            held = (base, httpx.AsyncClient(base_url=base, transport=self._transport, limits=POOL_LIMITS,
                                            timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_S,
                                                                  pool=POOL_TIMEOUT_S)))
            self._clients[dep_id] = held
        self._leases[held[1]] = self._leases.get(held[1], 0) + 1
        return held[1]

    def unlease(self, c: httpx.AsyncClient) -> None:
        n = self._leases.get(c, 0) - 1
        if n > 0:
            self._leases[c] = n
            return
        self._leases.pop(c, None)
        if c in self._retired:
            self._retired.discard(c)
            self._close_later(c)

    def _retire(self, c: httpx.AsyncClient) -> None:
        """Close ``c`` once no request uses it: a stream still running on it is never cut."""
        if self._leases.get(c):
            self._retired.add(c)
        else:
            self._close_later(c)

    def _forget(self, dep_id: int) -> None:
        held = self._clients.pop(dep_id, None)
        if held is not None:
            self._retire(held[1])

    def _close_later(self, c: httpx.AsyncClient) -> None:
        try:
            t = asyncio.get_running_loop().create_task(c.aclose())
        except RuntimeError:  # no loop (sync removal in a test): the pool is dropped with the client
            return
        self._closing.add(t)
        t.add_done_callback(self._closing.discard)

    async def aclose(self) -> None:
        clients = [c for _, c in self._clients.values()] + list(self._retired)
        self._clients, self._retired = {}, set()
        for c in clients:
            await c.aclose()
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)


def _gateway(request: Request) -> Gateway:
    return request.app.state.ondemand_gateway


async def _read_body(request: Request) -> bytes | None:
    """The request body, or None when it is larger than MAX_BODY_BYTES (with or without a Content-Length)."""
    length = request.headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
        return None
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_BODY_BYTES:
            return None
    return bytes(buf)


async def _prepare(request: Request, dep_id: int) -> tuple[Gateway, dict, bytes] | Response:
    """Key, body cap and the deployment record: (gateway, view, body), or the error response."""
    gw = _gateway(request)
    key = gw.key()
    if key is None:
        return _error(503, "the on-demand gateway has no key configured (run ./install.sh)", "ondemand_not_configured")
    peer = _peer(request.scope)
    if gw.auth_failures.blocked(peer or "-"):
        return _error(429, "too many requests without a valid key; try again later", "rate_limited",
                      "invalid_request_error", headers={"Retry-After": str(RETRY_AFTER_S)})
    if not gw.authorized(request, key):
        if gw.stale_key(request):  # LiteLLM before its next sync: refused, but not a failed attempt
            return _error(401, "the on-demand gateway key was rotated; LiteLLM gets the new key at its next sync",
                          "invalid_api_key", "invalid_request_error")
        gw.auth_failed(peer)
        return _error(401, "missing or invalid on-demand gateway key", "invalid_api_key", "invalid_request_error")
    try:
        body = await _read_body(request)
    except ClientDisconnect:
        return Response(status_code=400)  # nobody is listening any more
    if body is None:
        return _error(413, f"request body larger than {MAX_BODY_BYTES} bytes", "request_too_large",
                      "invalid_request_error")
    try:
        v = gw.ctx.deployer.view(dep_id)
    except DeployError as e:
        if e.status == 404:
            return _error(404, e.message, "not_found", "invalid_request_error")
        raise
    if v.get("system"):
        return _error(503, f"{v['name']} is not an on-demand deployment", "not_on_demand")
    return gw, v, body


def _payload(body: bytes) -> Any:
    try:
        return json.loads(body)
    except ValueError:
        return None  # not JSON (multipart audio, garbage): passed on as it is, the engine answers it


def _wants_stream(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("stream") is True


def _structured(payload: Any) -> bool:
    """The request asks for JSON, tool calls or guided output: a queued notice in its answer would break its client's
    parsing. Any of STRUCTURED_FIELDS or ``guided_*`` set (not null), or a Responses ``text.format`` that is not
    plain text."""
    if not isinstance(payload, dict):
        return False
    if any(v is not None and (k in STRUCTURED_FIELDS or k.startswith("guided_")) for k, v in payload.items()):
        return True
    text = payload.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    return isinstance(fmt, dict) and fmt.get("type") not in (None, "text")


async def _traffic(request: Request, dep_id: int, endpoint: str) -> Response:
    prep = await _prepare(request, dep_id)
    if isinstance(prep, Response):
        return prep
    gw, v, body = prep
    payload = _payload(body)
    stream = endpoint not in NEVER_STREAMS and _wants_stream(payload)
    if request.headers.get(HEALTH_HEADER) == "1":
        return _health(v, endpoint)  # never forwarded, whatever the mode or state
    if v["mode"] != "on_demand":
        # switched to always on while LiteLLM still points here: answered while ready, never started from here
        if v["state"] == "ready":
            return _Exchange(gw, request, v, f"v1/{endpoint}", body, endpoint=endpoint)
        return _error(503, f"{v['name']} is not an on-demand deployment", "not_on_demand")
    return _Exchange(gw, request, v, f"v1/{endpoint}", body, endpoint=endpoint, stream=stream, wake=True,
                     notice=not _structured(payload))


def _health(v: dict, endpoint: str) -> Response:
    """A LiteLLM health check, answered here and never forwarded (the header is not a secret: any LiteLLM client can
    send it). It reports that the gateway and the deployment record are there, not that the engine is serving: a
    sleeping model is available on demand, so it is healthy. No slot, no wake, no in-flight change."""
    state, served, now = str(v["state"]), _served(v), int(time.time())
    headers = {"x-dgxmm-ondemand-state": state}
    if state == "failed":
        return _error(503, f"{_model(v)} failed; start it again from its page", "ondemand_not_ready", headers=headers)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if endpoint == "chat/completions":
        out = {"id": "ondemand-health", "object": "chat.completion", "created": now, "model": served,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
               "usage": usage}
    elif endpoint == "completions":
        out = {"id": "ondemand-health", "object": "text_completion", "created": now, "model": served,
               "choices": [{"index": 0, "text": "", "finish_reason": "stop"}], "usage": usage}
    elif endpoint == "responses":
        out = {"id": "resp_ondemand_health", "object": "response", "created_at": now, "model": served,
               "status": "completed",
               "output": [{"id": "msg_ondemand_health", "type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "", "annotations": []}]}],
               "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    elif endpoint == "messages":
        out = {"id": "msg_ondemand_health", "type": "message", "role": "assistant", "model": served,
               "content": [{"type": "text", "text": ""}], "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": 0, "output_tokens": 0}}
    elif endpoint == "embeddings":
        out = {"object": "list", "model": served, "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
               "usage": {"prompt_tokens": 0, "total_tokens": 0}}
    else:
        return _error(503, f"health checks of {endpoint} are not answered by the on-demand gateway",
                      "ondemand_health_unsupported", headers=headers)
    return JSONResponse(out, headers=headers)


def _traffic_route(endpoint: str):
    async def handler(request: Request, dep_id: int):
        return await _traffic(request, dep_id, endpoint)
    handler.__name__ = "traffic_" + endpoint.replace("/", "_")
    return handler


for _endpoint in TRAFFIC:
    router.add_api_route(f"/{{dep_id}}/v1/{_endpoint}", _traffic_route(_endpoint), methods=["POST"])


@router.get("/{dep_id}/v1/models")
async def models(request: Request, dep_id: int):
    """From the record: listing models (LiteLLM, health checks) never wakes one."""
    prep = await _prepare(request, dep_id)
    if isinstance(prep, Response):
        return prep
    _, v, _ = prep
    return JSONResponse({"object": "list", "data": [{"id": _served(v), "object": "model",
                                                     "created": _created(v.get("created_at")),
                                                     "owned_by": "dgxmm"}]})


def _created(iso: str | None) -> int:
    if not iso:
        return 0
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return 0
    return int((t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp())


@router.api_route("/{dep_id}/v1/{path:path}", methods=OTHER_METHODS)
async def other(request: Request, dep_id: int, path: str):
    """Not traffic: proxied only while the deployment is ready, without a slot; never wakes it."""
    prep = await _prepare(request, dep_id)
    if isinstance(prep, Response):
        return prep
    gw, v, body = prep
    if not passthrough_allowed(request.method, path):
        return JSONResponse(NOT_FOUND, status_code=404)
    if v["state"] != "ready":
        return _error(503, f"{_model(v)} is not running; only inference requests start it", "ondemand_not_ready")
    # rebuilt from the (clean_path-checked) segments, each quoted again: nothing can step out of /v1/models
    return _Exchange(gw, request, v, "v1/" + "/".join(quote(seg, safe=PCHAR_SAFE) for seg in path.split("/")), body)


@router.api_route("/{rest:path}", methods=OTHER_METHODS)
async def unknown(rest: str):
    return JSONResponse(NOT_FOUND, status_code=404)


class _Exchange(Response):
    """The ASGI response of one proxied request: waits for a slot (``wake``), then streams the engine's answer.
    A client hang-up at any point cancels whatever is in progress; the slot is released in every case."""

    def __init__(self, gw: Gateway, request: Request, v: dict, path: str, body: bytes, *,
                 endpoint: str | None = None, stream: bool = False, wake: bool = False, notice: bool = True):
        super().__init__()
        self.gw, self.v, self.path, self.content = gw, v, path, body
        self.method, self.query = request.method, request.url.query
        named = {t.strip().lower() for t in request.headers.get("connection", "").split(",") if t.strip()}
        self.req_headers = [(k, val) for k, val in request.headers.items()
                            if k.lower() not in DROP_REQUEST and k.lower() not in named]  # RFC 7230 6.1
        self.req_headers.append(("accept-encoding", "identity"))
        self.endpoint, self.stream, self.wake = endpoint, stream, wake
        self.notice = notice  # False: a structured-output request, never given the queued notice
        self.queued = False   # the request waited in the queue for memory
        self.started = False  # our own 200 + SSE headers (the queued notice) went out
        self._send = None

    async def __call__(self, scope, receive, send) -> None:
        self._send = send
        work = asyncio.ensure_future(self._run())
        gone = asyncio.ensure_future(_hung_up(receive))
        try:
            await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (work, gone):
                t.cancel()
            await asyncio.gather(work, gone, return_exceptions=True)
        if gone.done() and not gone.cancelled():
            log.debug("on-demand gateway: deployment %s: client hung up", self.v["id"])
            return
        if not work.cancelled() and (exc := work.exception()) is not None and not isinstance(exc, OSError):
            raise exc

    # ---------- flow ----------
    async def _run(self) -> None:
        slot = None
        if self.wake:
            slot = await self._acquire()
            if slot is None:
                return
        try:
            await self._forward()
        finally:
            if slot is not None:
                slot.release()

    async def _acquire(self):
        """The slot, or None after sending the error. Never leaks a slot granted while this request is cancelled."""
        queued = asyncio.Event()

        async def on_queued() -> None:
            queued.set()
        task = asyncio.ensure_future(self.gw.ctx.ondemand.acquire(self.v["id"], on_queued=on_queued))
        handed = False
        try:
            if self.stream and self.notice and self.endpoint in STREAM_NOTICE:
                flag = asyncio.ensure_future(queued.wait())
                try:
                    await asyncio.wait({task, flag}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    flag.cancel()
                if queued.is_set():
                    await self._notice()
            try:
                slot = await task
            except OnDemandError as e:
                await self._refuse(e)
                return None
            self.queued, handed = queued.is_set(), True
            return slot
        finally:
            if not handed:
                task.cancel()
                task.add_done_callback(_release_granted)

    async def _forward(self) -> None:
        try:
            v = self.gw.ctx.deployer.view(self.v["id"])  # fresh: the port of the (re)started engine
        except DeployError as e:
            if e.status != 404:
                raise
            await self._fail(404, error_body(e.message, "not_found", "invalid_request_error"))  # removed meanwhile
            return
        base = self.gw.ctx.deployer.base_url(v["name"], v["port"], v.get("engine"))
        client = self.gw.lease(v["id"], base)
        try:
            await self._exchange(client, v)
        finally:
            self.gw.unlease(client)

    async def _exchange(self, client: httpx.AsyncClient, v: dict) -> None:
        url = "/" + self.path + (f"?{self.query}" if self.query else "")
        request = client.build_request(self.method, url, headers=self.req_headers, content=self.content)
        try:
            up = await client.send(request, stream=True)
        except httpx.HTTPError as e:
            log.warning("on-demand gateway: deployment %s: %s %s failed: %s", v["id"], self.method, self.path,
                        type(e).__name__)
            await self._upstream_failed(v)
            return
        try:
            if self.started:
                await self._into_notice_stream(up)
            elif self.queued and self.notice and not self.stream and self.endpoint in BODY_NOTICE \
                    and 200 <= up.status_code < 300:
                await self._with_notice(up)
            else:
                await self._pass_through(up)
        finally:
            await up.aclose()

    # ---------- what the client gets ----------
    async def _start(self, status: int, headers: list[tuple[bytes, bytes]]) -> None:
        await self._send({"type": "http.response.start", "status": status, "headers": headers})

    async def _body(self, data: bytes, more: bool = True) -> None:
        await self._send({"type": "http.response.body", "body": data, "more_body": more})

    async def _respond(self, resp: Response) -> None:
        await self._start(resp.status_code, resp.raw_headers)
        await self._body(resp.body, more=False)

    async def _notice(self) -> None:
        notice = NOTICE.format(model=_model(self.v))
        if self.endpoint == "completions":
            chunk = {"id": "ondemand-queued", "object": "text_completion", "created": int(time.time()),
                     "model": _served(self.v), "choices": [{"index": 0, "text": notice, "finish_reason": None}]}
        else:
            chunk = {"id": "ondemand-queued", "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": _served(self.v), "choices": [{"index": 0, "delta": {"role": "assistant",
                                                                                  "content": notice},
                                                            "finish_reason": None}]}
        self.started = True
        await self._start(200, SSE_HEADERS)
        await self._body(_sse(chunk))

    async def _refuse(self, e: OnDemandError) -> None:
        body = error_body(e.message, e.code)
        if self.started:
            await self._body(_sse(body))
            await self._body(DONE, more=False)
            return
        busy = isinstance(e, OnDemandTimeout) or e.code == "ondemand_queue_full"
        headers = {"Retry-After": str(RETRY_AFTER_S)} if busy else None
        await self._respond(JSONResponse(body, status_code=503, headers=headers))

    async def _upstream_failed(self, v: dict) -> None:
        await self._fail(502, error_body(f"could not reach the engine of {_model(v)}", "ondemand_upstream_error"))

    async def _fail(self, status: int, body: dict) -> None:
        """An error response, or a final SSE error event + [DONE] when the queued notice already went out."""
        if self.started:
            await self._body(_sse(body))
            await self._body(DONE, more=False)
        else:
            await self._respond(JSONResponse(body, status_code=status))

    async def _pass_through(self, up: httpx.Response) -> None:
        """Status, headers and bytes exactly as the engine sent them, each chunk as it arrives."""
        await self._start(up.status_code, [(k, val) for k, val in up.headers.raw
                                           if k.decode("latin-1").lower() not in DROP_RESPONSE])
        try:
            async for chunk in up.aiter_raw():
                if chunk:
                    await self._body(chunk)
        except httpx.HTTPError as e:  # the engine went away mid-answer: end what was sent
            log.warning("on-demand gateway: deployment %s: answer cut off: %s", self.v["id"], type(e).__name__)
        await self._body(b"", more=False)

    async def _into_notice_stream(self, up: httpx.Response) -> None:
        """After the queued notice (200 + SSE already sent): the engine's SSE bytes, or its error as an event."""
        if up.status_code >= 400:
            raw = b""
            async for chunk in up.aiter_raw():
                raw += chunk
                if len(raw) > MAX_ERROR_BYTES:
                    break
            await self._body(_sse(_as_error(raw[:MAX_ERROR_BYTES], up.status_code)))
            await self._body(DONE, more=False)
            return
        try:
            async for chunk in up.aiter_raw():
                if chunk:
                    await self._body(chunk)
        except httpx.HTTPError as e:
            log.warning("on-demand gateway: deployment %s: answer cut off: %s", self.v["id"], type(e).__name__)
        await self._body(b"", more=False)

    async def _with_notice(self, up: httpx.Response) -> None:
        """Non-streaming answer to a request that queued: NOTICE before the first choice's text."""
        raw = await up.aread()
        try:
            out = json.loads(raw)
            notice = NOTICE.format(model=_model(self.v))
            if self.endpoint == "responses":
                part = _first_output_text(out)
                if part is not None:
                    part["text"] = notice + part["text"]
                    raw = json.dumps(out, ensure_ascii=False).encode()
            else:
                choice = out["choices"][0]
                if self.endpoint == "chat/completions" and isinstance(choice["message"].get("content"), str):
                    choice["message"]["content"] = notice + choice["message"]["content"]
                elif self.endpoint == "completions" and isinstance(choice.get("text"), str):
                    choice["text"] = notice + choice["text"]
                raw = json.dumps(out, ensure_ascii=False).encode()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            pass  # not the shape we know: unchanged
        headers = [(k, val) for k, val in up.headers.raw
                   if k.decode("latin-1").lower() not in DROP_RESPONSE | {"content-length", "content-encoding"}]
        headers.append((b"content-length", str(len(raw)).encode()))
        await self._start(up.status_code, headers)
        await self._body(raw, more=False)


def _first_output_text(out: Any) -> dict | None:
    """The first ``output_text`` part of the first message item in a Responses API answer, if it has one."""
    for item in out.get("output") or [] if isinstance(out, dict) else []:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    return part
    return None


def _as_error(raw: bytes, status: int) -> dict:
    """An engine error body as an OpenAI-style ``{"error": ...}`` object."""
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
        return {"error": obj["error"]}
    message = obj.get("message") if isinstance(obj, dict) and isinstance(obj.get("message"), str) else \
        raw.decode("utf-8", "replace") or f"engine answered HTTP {status}"
    return error_body(message, str(status))


def _release_granted(task: asyncio.Task) -> None:
    """A slot granted to a request that was cancelled meanwhile goes straight back."""
    if not task.cancelled() and task.exception() is None:
        task.result().release()


async def _hung_up(receive) -> None:
    """Returns when the client disconnects (the body was read before the response started)."""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
