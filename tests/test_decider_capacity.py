"""Decider back-pressure: d3_server runs one request at a time, and a request a client gave up on still runs to the
end on the server. The client therefore keeps at most ``max_in_flight`` requests at the server; others wait in a
short client-side queue (interactive features ahead of background ones) and are shed with DeciderError("busy")
when the queue wait would exceed ``queue_timeout_s`` or their own deadline, so bursts fall back to rules quickly
instead of piling up behind each other."""
import asyncio
import time

import httpx
import pytest

from dgxmm.decider.client import DeciderError, Limiter, SystemOneClient
from dgxmm.decider.types import Option
from tests.test_decider_client import OPTS, Clock, reply


class SlowServer:
    """Serial like d3_server: one request at a time, ``work_s`` each; records the peak concurrency it saw."""

    def __init__(self, work_s=0.2):
        self.work_s, self.active, self.peak, self.served = work_s, 0, 0, 0
        self.turn = asyncio.Lock()

    async def handler(self, req: httpx.Request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            async with self.turn:
                await asyncio.sleep(self.work_s)
                self.served += 1
            return httpx.Response(200, json=reply())
        finally:
            self.active -= 1


def make(server, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return SystemOneClient(http, base_url=lambda: "http://decider:8100", api_key=lambda: "k", revision="rev1",
                           clock=Clock(), **kw)


def ask(c, i, timeout_s=2.0, priority=0):
    return c.choose(feature="plan", state=f"state {i}", question="q", options=OPTS, timeout_s=timeout_s,
                    priority=priority)


async def test_at_most_max_in_flight_requests_reach_the_server():
    srv = SlowServer(0.05)
    c = make(srv, max_in_flight=2, queue_timeout_s=5)
    out = await asyncio.gather(*(ask(c, i, timeout_s=5) for i in range(8)), return_exceptions=True)
    assert all(not isinstance(o, Exception) for o in out)
    assert srv.peak <= 2 and srv.served == 8


async def test_burst_is_shed_quickly_instead_of_timing_out():
    srv = SlowServer(0.3)
    c = make(srv, max_in_flight=1, queue_timeout_s=0.1)
    t0 = time.perf_counter()
    res = await asyncio.gather(*(ask(c, i, timeout_s=2.0) for i in range(6)), return_exceptions=True)
    ok = [r for r in res if not isinstance(r, Exception)]
    busy = [r for r in res if isinstance(r, DeciderError) and r.code == "busy"]
    assert len(ok) == 1 and len(busy) == 5
    assert time.perf_counter() - t0 < 1.0  # nobody waited for the 2 s deadline
    assert srv.served == 1  # shed requests never reached the server
    assert not c.breaker.is_open() and c.breaker.failures == 0  # back-pressure is not ill health
    assert c.limiter.shed == 5


async def test_queue_wait_counts_against_the_callers_deadline():
    srv = SlowServer(0.4)
    c = make(srv, max_in_flight=1, queue_timeout_s=5)
    first = asyncio.ensure_future(ask(c, 0, timeout_s=2.0))
    await asyncio.sleep(0.01)
    t0 = time.perf_counter()
    with pytest.raises(DeciderError) as e:
        await ask(c, 1, timeout_s=0.2)  # cannot even start before its own deadline
    assert e.value.code == "busy" and time.perf_counter() - t0 < 0.35
    await first
    c = make(srv, max_in_flight=1, queue_timeout_s=5)  # a fresh client: no service time learned yet
    t0 = time.perf_counter()
    blocker = asyncio.ensure_future(ask(c, 2, timeout_s=2.0))
    await asyncio.sleep(0.01)
    with pytest.raises(DeciderError) as e:
        await ask(c, 3, timeout_s=0.6)  # gets the slot after ~0.4 s, then only ~0.2 s are left for the request
    assert e.value.code == "timeout" and time.perf_counter() - t0 < 0.8
    await blocker


async def test_interactive_requests_go_ahead_of_background_ones():
    srv = SlowServer(0.1)
    c = make(srv, max_in_flight=1, queue_timeout_s=5)
    order = []

    async def run(i, prio):
        await ask(c, i, timeout_s=5, priority=prio)
        order.append(i)

    first = asyncio.ensure_future(run(0, 1))
    await asyncio.sleep(0.01)
    bg = asyncio.ensure_future(run(1, 1))   # background, queued first
    await asyncio.sleep(0.01)
    fg = asyncio.ensure_future(run(2, 0))   # interactive, queued later
    await asyncio.gather(first, bg, fg)
    assert order == [0, 2, 1]


async def test_cache_hits_bypass_the_limiter():
    srv = SlowServer(0.2)
    c = make(srv, max_in_flight=1, queue_timeout_s=0.05)
    await ask(c, 0)
    blocker = asyncio.ensure_future(ask(c, 1))
    await asyncio.sleep(0.01)
    hit = await ask(c, 0)  # same input: answered from the cache although the server slot is taken
    assert hit.cached
    await blocker


def test_limiter_unit_stats():
    lim = Limiter(max_in_flight=3)
    assert lim.stats() == {"max_in_flight": 3, "in_flight": 0, "queued": 0, "shed": 0, "service_ms": 0.0}
    lim.observe(0.2)
    assert lim.wait_budget(2.0, 0, 1.0) == 1.0 and lim.wait_budget(2.0, 1, 1.0) == 1.0
    assert lim.wait_budget(10.0, 1, 1.0) == 5.0 and lim.wait_budget(0.5, 0, 1.0) == pytest.approx(0.3)


async def test_decider_reports_load_and_treats_busy_as_abstain(settings, db):
    from dgxmm.decider.service import Decider
    from dgxmm.settings_store import SettingsStore
    store = SettingsStore(db, settings)
    srv = SlowServer(0.3)
    settings.decider_max_in_flight, settings.decider_queue_timeout_s = 1, 0.05
    d = Decider(db=db, settings=settings, store=store, deployer=lambda: None,
                http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(srv.handler)))
    d.set_enabled(True)
    d._base_url = lambda: "http://decider:8100"
    res = await asyncio.gather(*(d.decide(feature="plan", subject=f"s{i}", state=f"st{i}", question="q",
                                          options=[Option("gpu_oom", "a"), Option("port_in_use", "b"),
                                                   Option("other", "c")]) for i in range(4)))
    assert sum(r.source == "model" for r in res) == 1
    assert all(r.source == "none" and r.choice is None for r in res if r.source != "model")
    load = d.load()
    assert load["shed"] == 3 and load["max_in_flight"] == 1
    assert d.healthy() is True  # shedding never opens the breaker
    rows = d.recent()
    assert len(rows) == 1  # shed requests are counted, not logged as decisions (bursts would flood the log)


async def test_queue_budget_leaves_room_for_the_measured_service_time():
    srv = SlowServer(0.3)
    c = make(srv, max_in_flight=1, queue_timeout_s=5)
    for i in range(3):
        await ask(c, i)  # learns ~300 ms per request
    assert 0.25 < c.limiter.service_s < 0.5
    blocker = asyncio.ensure_future(ask(c, 10))
    await asyncio.sleep(0.01)
    t0 = time.perf_counter()
    with pytest.raises(DeciderError) as e:
        await ask(c, 11, timeout_s=0.4)  # 0.4 s deadline - 0.3 s service: may queue ~0.1 s at most
    assert e.value.code == "busy" and time.perf_counter() - t0 < 0.2
    await blocker


async def test_background_requests_may_queue_longer_than_interactive_ones():
    srv = SlowServer(0.3)
    c = make(srv, max_in_flight=1, queue_timeout_s=0.1)
    await ask(c, 0)
    blocker = asyncio.ensure_future(ask(c, 1))
    await asyncio.sleep(0.01)
    fg = asyncio.ensure_future(ask(c, 2, timeout_s=2.0, priority=0))
    bg = asyncio.ensure_future(ask(c, 3, timeout_s=10.0, priority=1))
    res = await asyncio.gather(blocker, fg, bg, return_exceptions=True)
    assert isinstance(res[1], DeciderError) and res[1].code == "busy"  # interactive: shed after 0.1 s
    assert not isinstance(res[2], Exception)  # background: waited for the slot (up to half its deadline)
