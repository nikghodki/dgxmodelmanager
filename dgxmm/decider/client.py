"""Async client for the decider's System One API (``POST /v1/systemone``).

Typed questions in, a probability per option out. The bearer key comes from a callable at request time and is
never put into errors or logs. A small LRU cache (feature, input hash, model revision) and a circuit breaker
(``threshold`` consecutive failures -> open for ``open_s``) keep the callers' request paths cheap and fail-open.
"""
from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import httpx

from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
SYSTEM_ONE_PATH = "/v1/systemone"
HEALTH_PATH = "/health"
QID = "q"
# Answers are untrusted (docs/security.md): a reply larger than this, or with more options than were offered, is
# not read; probabilities must be finite numbers in [0, 1] whose sum is about 1, and the choice their argmax.
MAX_REPLY_BYTES = 64 * 1024
MAX_OPTIONS = 128
SUM_TOLERANCE = 0.02      # |sum - 1| up to this: renormalised; beyond: rejected
ARGMAX_TOLERANCE = 1e-6


class _ReplyTooLarge(ValueError):
    pass


def _no_constants(name: str):
    raise ValueError(f"non-finite number {name} in the reply")


async def capped_request(http: httpx.AsyncClient, method: str, url: str, *, timeout_s: float,
                         **kw) -> tuple[int, bytes]:
    """(status, body) of a request to the decider, reading at most MAX_REPLY_BYTES (decoded) of the body: every
    reply of the decider is untrusted (health, probe, warm-up and answers alike). Raises _ReplyTooLarge."""
    async with http.stream(method, url, timeout=timeout_s, **kw) as r:
        declared = r.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_REPLY_BYTES:
            raise _ReplyTooLarge()
        buf = bytearray()
        async for chunk in r.aiter_bytes():  # decoded: a compressed reply is capped by its decoded size
            buf += chunk
            if len(buf) > MAX_REPLY_BYTES:
                raise _ReplyTooLarge()
        return r.status_code, bytes(buf)


def json_object(content: bytes) -> dict:
    """The reply as a JSON object, or {} (too large, not JSON, NaN / Infinity, not an object)."""
    if len(content) > MAX_REPLY_BYTES:
        return {}
    try:
        body = json.loads(content, parse_constant=_no_constants)
    except (ValueError, RecursionError):
        return {}
    return body if isinstance(body, dict) else {}


def parse_answer(content: bytes, options: list[Option]) -> tuple[str, dict[str, float], float]:
    """(choice, probabilities, confidence) of a /v1/systemone reply, or ValueError. Rejects NaN / Infinity,
    probabilities outside [0, 1], options that were not offered, a sum far from 1 and a choice that is not the most
    probable option; confidence is the chosen option's probability."""
    if len(content) > MAX_REPLY_BYTES:
        raise ValueError("reply too large")
    ans = json.loads(content, parse_constant=_no_constants)["answers"][QID]
    raw = ans.get("probabilities") or {}
    choice = ans.get("choice")
    allowed = {o.key for o in options}
    if not isinstance(raw, dict) or not raw or len(raw) > len(allowed) or choice not in allowed:
        raise ValueError("choice outside the options")
    probs: dict[str, float] = {}
    for k, v in raw.items():
        if str(k) not in allowed or isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("probability for an option that was not offered, or not a number")
        f = float(v)
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError("probability outside [0, 1]")
        probs[str(k)] = f
    total = sum(probs.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"probabilities sum to {total:.3f}")
    probs = {k: v / total for k, v in probs.items()}
    if choice not in probs or probs[choice] < max(probs.values()) - ARGMAX_TOLERANCE:
        raise ValueError("the choice is not the most probable option")
    return choice, probs, probs[choice]


class DeciderError(Exception):
    """A failed decider call. ``code``: circuit_open | timeout | unreachable | unauthorized | http_error | bad_reply
    | unavailable | busy (shed by the client-side limiter). Messages never contain the API key or the request
    state."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class Answer:
    choice: str
    probabilities: dict[str, float]
    confidence: float  # top probability
    latency_ms: float
    cached: bool = False


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures for ``open_s`` seconds. After that a single trial call is let
    through (half-open): success closes it, another failure re-opens it at once."""

    def __init__(self, threshold: int = 3, open_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.open_s, self.clock = threshold, open_s, clock
        self.failures = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        return self.opened_at is not None and self.clock() - self.opened_at < self.open_s

    def retry_in(self) -> float:
        return max(self.open_s - (self.clock() - self.opened_at), 0.0) if self.is_open() else 0.0

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock()


@dataclass
class _LRU:
    size: int
    items: OrderedDict = field(default_factory=OrderedDict)

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key, value) -> None:
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.size:
            self.items.popitem(last=False)

    def __contains__(self, key) -> bool:
        return key in self.items

    def __len__(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items.clear()


PRIORITY_INTERACTIVE = 0  # request path (served first)
PRIORITY_BACKGROUND = 1   # background features (diagnosis, kind, workload, ...)
PRIORITY_LANE = 2         # the insight engine's jobs (dgxmm/decider/insight_engine.py): lowest, and capped


class Limiter:
    """At most ``max_in_flight`` requests at the decider; the rest wait, lower ``priority`` first (FIFO within a
    priority), for at most the given timeout, then are shed. d3_server answers one request at a time and finishes
    requests whose client already gave up, so an unbounded burst would make every later caller wait for work nobody
    needs any more.

    Background lane: requests at ``PRIORITY_LANE`` or lower priority hold at most ``lane_slots`` (= max_in_flight - 1,
    at least 1) slots, so with two or more slots an interactive caller never waits behind background jobs; with one
    slot it waits for at most the request in flight and then goes ahead of every queued lane request."""

    def __init__(self, max_in_flight: int = 2):
        self.max_in_flight = max(int(max_in_flight), 1)
        self.lane_slots = max(self.max_in_flight - 1, 1)
        self.in_flight = 0
        self.lane_in_flight = 0
        self.shed = 0
        self.service_s = 0.0  # moving average of the server's time per request (successful ones)
        self._waiters: list[list] = []  # heap of [priority, seq, future]
        self._seq = itertools.count()

    def stats(self) -> dict:
        queued = sum(1 for w in self._waiters if not w[2].done())
        return {"max_in_flight": self.max_in_flight, "in_flight": self.in_flight, "queued": queued,
                "shed": self.shed, "service_ms": round(self.service_s * 1000, 1)}

    def observe(self, seconds: float) -> None:
        self.service_s = seconds if self.service_s == 0 else 0.8 * self.service_s + 0.2 * seconds

    def wait_budget(self, deadline_s: float, priority: int, queue_timeout_s: float) -> float:
        """How long a caller may wait for a slot: never so long that the measured service time no longer fits in
        its deadline; interactive callers at most ``queue_timeout_s``, background ones up to half their deadline."""
        cap = queue_timeout_s if priority <= 0 else max(queue_timeout_s, deadline_s / 2)
        return min(cap, deadline_s - self.service_s)

    def _allowed(self, priority: int) -> bool:
        """A slot may go to this priority now (the lane cap)."""
        return priority < PRIORITY_LANE or self.lane_in_flight < self.lane_slots

    def _take(self, priority: int) -> None:
        if priority >= PRIORITY_LANE:
            self.lane_in_flight += 1

    def _eligible_waiter(self) -> bool:
        return any(not w[2].done() and self._allowed(w[0]) for w in self._waiters)

    async def acquire(self, priority: int, timeout_s: float) -> bool:
        """True = holds a slot (call release(priority)); False = shed."""
        if self.in_flight < self.max_in_flight and self._allowed(priority) and not self._eligible_waiter():
            self.in_flight += 1
            self._take(priority)
            return True
        if timeout_s <= 0:
            self.shed += 1
            return False
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, [priority, next(self._seq), fut])
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout_s)
            return True
        except asyncio.TimeoutError:
            if fut.done() and not fut.cancelled():
                self.release(priority)  # granted at the very moment the wait timed out: hand the slot on
            else:
                fut.cancel()
            self.shed += 1
            return False
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release(priority)
            else:
                fut.cancel()
            raise

    def release(self, priority: int = PRIORITY_INTERACTIVE) -> None:
        """Give back the slot taken at ``priority``: it passes to the first waiter (by priority, then arrival) that
        the lane cap admits; waiters held back by the cap keep waiting."""
        if priority >= PRIORITY_LANE:
            self.lane_in_flight = max(self.lane_in_flight - 1, 0)
        self._waiters = [w for w in self._waiters if not w[2].done()]
        heapq.heapify(self._waiters)
        for w in sorted(self._waiters):
            if self._allowed(w[0]):
                self._waiters.remove(w)
                heapq.heapify(self._waiters)
                self._take(w[0])
                w[2].set_result(True)  # the slot passes to this waiter; in_flight is unchanged
                return
        self.in_flight = max(self.in_flight - 1, 0)


def criteria(options: list[Option]) -> dict[str, str]:
    """Option key -> its meaning as the model sees it (description, else label)."""
    return {o.key: (o.description or o.label or o.key) for o in options}


class SystemOneClient:
    def __init__(self, http: httpx.AsyncClient, *, base_url: Callable[[], str | None], api_key: Callable[[], str | None],
                 revision: str, cache_size: int = 256, breaker: CircuitBreaker | None = None,
                 clock: Callable[[], float] = time.monotonic, max_in_flight: int = 2, queue_timeout_s: float = 0.5):
        self.http, self.base_url, self.api_key, self.revision = http, base_url, api_key, revision
        self.cache = _LRU(cache_size)
        self.breaker = breaker or CircuitBreaker(clock=clock)
        self.limiter = Limiter(max_in_flight)
        self.queue_timeout_s = queue_timeout_s

    @staticmethod
    def input_hash(feature: str, state: str, question: str, options: list[Option]) -> str:
        blob = json.dumps({"f": feature, "s": state, "q": question, "o": criteria(options)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _headers(self) -> dict[str, str]:
        key = self.api_key()
        return {"authorization": f"Bearer {key}"} if key else {}

    async def choose(self, *, feature: str, state: str, question: str, options: list[Option],
                     timeout_s: float = 2.0, priority: int = 0) -> Answer:
        """Ask one choice question. Raises DeciderError on any failure (callers fail open). ``timeout_s`` covers the
        queue wait and the request; ``priority`` 0 = interactive (served first), 1 = background, 2 = the insight engine's lane (capped)."""
        if not options:
            raise DeciderError("bad_request", "a choice question needs options")
        if len(options) > MAX_OPTIONS:
            raise DeciderError("bad_request", f"at most {MAX_OPTIONS} options per question")
        key = (feature, self.input_hash(feature, state, question, options), self.revision)
        hit = self.cache.get(key)
        if hit is not None:
            return Answer(hit.choice, dict(hit.probabilities), hit.confidence, 0.0, cached=True)
        if self.breaker.is_open():
            raise DeciderError("circuit_open", f"decider paused after repeated failures; retry in "
                                               f"{self.breaker.retry_in():.0f}s")
        base = self.base_url()
        if not base:
            raise DeciderError("unavailable", "no decider deployment is running")
        body = {"state": state, "questions": {QID: {"type": "choice", "instructions": question,
                                                     "criteria": criteria(options)}}}
        t_start = time.perf_counter()
        if not await self.limiter.acquire(priority, self.limiter.wait_budget(timeout_s, priority,
                                                                             self.queue_timeout_s)):
            raise DeciderError("busy", "decider busy (client queue full); answered by rules")
        try:
            remaining = timeout_s - (time.perf_counter() - t_start)
            if remaining <= 0.01:
                raise DeciderError("busy", "decider busy (deadline reached while queued); answered by rules")
            r, latency = await self._post(base, body, remaining, penalize=remaining >= timeout_s / 2)
        finally:
            self.limiter.release(priority)
        return self._read(r, options, latency, key)

    async def _post(self, base: str, body: dict, timeout_s: float, penalize: bool) -> tuple[httpx.Response, float]:
        """POST with a hard deadline. A timeout counts against the breaker only if the request had at least half of
        its caller's deadline (``penalize``): running out of time after a long queue wait is load, not ill health."""
        t0 = time.perf_counter()
        try:
            r = await asyncio.wait_for(self._post_capped(base + SYSTEM_ONE_PATH, body, timeout_s), timeout_s)
        except _ReplyTooLarge:
            self.breaker.failure()
            raise DeciderError("bad_reply", f"decider reply larger than {MAX_REPLY_BYTES} bytes") from None
        except (asyncio.TimeoutError, httpx.TimeoutException):
            if penalize:
                self.breaker.failure()
            raise DeciderError("timeout", f"decider did not answer within {timeout_s:g}s") from None
        except (httpx.HTTPError, OSError) as e:
            self.breaker.failure()
            raise DeciderError("unreachable", f"decider unreachable ({type(e).__name__})") from None
        return r, (time.perf_counter() - t0) * 1000

    async def _post_capped(self, url: str, body: dict, timeout_s: float) -> httpx.Response:
        """POST and read at most MAX_REPLY_BYTES of the reply body (an untrusted server could send any amount)."""
        status, content = await capped_request(self.http, "POST", url, json=body, headers=self._headers(),
                                               timeout_s=timeout_s)
        return httpx.Response(status, content=content)

    def _read(self, r: httpx.Response, options: list[Option], latency: float, key) -> Answer:
        if r.status_code == 200:
            self.limiter.observe(latency / 1000)
        if r.status_code == 401:
            self.breaker.failure()
            raise DeciderError("unauthorized", "decider rejected the API key")
        if r.status_code == 422:  # our question is malformed: a bug, not an unhealthy server
            raise DeciderError("bad_request", "decider refused the question (422)")
        if r.status_code != 200:
            self.breaker.failure()
            raise DeciderError("http_error", f"decider answered HTTP {r.status_code}")
        try:
            choice, probs, confidence = parse_answer(r.content, options)
        except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError, RecursionError):
            # ArithmeticError: e.g. OverflowError converting a 400-digit integer "probability" to float
            self.breaker.failure()
            raise DeciderError("bad_reply", "decider reply could not be read or is not a valid answer") from None
        self.breaker.success()
        out = Answer(choice, probs, confidence, latency)
        self.cache.put(key, out)
        return out

    async def health(self, timeout_s: float = 5.0) -> bool:
        """GET /health (status ready), then a one-question /v1/systemone probe with the key."""
        base = self.base_url()
        if not base:
            return False
        try:
            status, content = await capped_request(self.http, "GET", base + HEALTH_PATH, timeout_s=timeout_s)
            if status != 200 or json_object(content).get("status") != "ready":
                return False
            return await probe(self.http, base, self.api_key(), timeout_s)
        except (httpx.HTTPError, OSError, ValueError):
            return False


PROBE = {"state": "health check", "questions": {"probe": {"type": "noul", "instructions": "Is this a health check?"}}}


async def probe(http: httpx.AsyncClient, base: str, key: str | None, timeout_s: float = 30.0) -> bool:
    """The authenticated one-question probe answered (reply capped and parsed like every decider reply)."""
    headers = {"authorization": f"Bearer {key}"} if key else {}
    try:
        status, content = await capped_request(http, "POST", base + SYSTEM_ONE_PATH, json=PROBE, headers=headers,
                                               timeout_s=timeout_s)
    except _ReplyTooLarge:
        return False
    answers = json_object(content).get("answers") if status == 200 else None
    return isinstance(answers, dict) and isinstance(answers.get("probe"), dict)
