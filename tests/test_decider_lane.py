"""The decider client's background lane (insight-engine jobs): lowest priority, and never every slot, so interactive
questions are not starved by background load (spec "Background engine")."""
import asyncio
import time

import pytest

from dgxmm.decider.client import PRIORITY_BACKGROUND, PRIORITY_INTERACTIVE, PRIORITY_LANE, Limiter
from tests.test_decider_capacity import SlowServer, ask, make


async def test_lane_never_holds_every_slot():
    lim = Limiter(max_in_flight=2)
    assert await lim.acquire(PRIORITY_LANE, 1.0)
    assert not await lim.acquire(PRIORITY_LANE, 0.05)  # the second slot is kept for others
    assert await lim.acquire(PRIORITY_INTERACTIVE, 0.0)  # ... and an interactive call takes it without waiting
    lim.release(PRIORITY_INTERACTIVE)
    lim.release(PRIORITY_LANE)
    assert lim.in_flight == 0 and lim.lane_in_flight == 0


async def test_single_slot_lane_can_use_it_but_interactive_goes_first():
    lim = Limiter(max_in_flight=1)
    assert await lim.acquire(PRIORITY_LANE, 1.0)
    order = []

    async def want(prio, tag):
        assert await lim.acquire(prio, 2.0)
        order.append(tag)
        await asyncio.sleep(0)
        lim.release(prio)

    lane = asyncio.ensure_future(want(PRIORITY_LANE, "lane"))
    await asyncio.sleep(0.01)
    bg = asyncio.ensure_future(want(PRIORITY_BACKGROUND, "background"))
    await asyncio.sleep(0.01)
    fg = asyncio.ensure_future(want(PRIORITY_INTERACTIVE, "interactive"))
    await asyncio.sleep(0.01)
    lim.release(PRIORITY_LANE)
    await asyncio.gather(lane, bg, fg)
    assert order == ["interactive", "background", "lane"]
    assert lim.in_flight == 0 and lim.lane_in_flight == 0


async def test_capped_lane_waiter_does_not_block_a_free_slot():
    lim = Limiter(max_in_flight=2)
    assert await lim.acquire(PRIORITY_LANE, 1.0)
    waiting = asyncio.ensure_future(lim.acquire(PRIORITY_LANE, 0.5))  # capped: waits
    await asyncio.sleep(0.01)
    t0 = time.perf_counter()
    assert await lim.acquire(PRIORITY_INTERACTIVE, 0.0)  # the free slot is not reserved for the capped waiter
    assert time.perf_counter() - t0 < 0.3  # the capped waiter waits up to 0.5 s
    lim.release(PRIORITY_INTERACTIVE)  # the lane is still at its cap: the waiter keeps waiting
    await asyncio.sleep(0.01)
    assert not waiting.done()
    lim.release(PRIORITY_LANE)  # now it may run
    assert await waiting and lim.lane_in_flight == 1
    lim.release(PRIORITY_LANE)
    assert lim.in_flight == 0 and lim.lane_in_flight == 0


async def test_interactive_requests_are_not_starved_under_background_load():
    srv = SlowServer(0.15)
    c = make(srv, max_in_flight=2, queue_timeout_s=0.5)
    flood = [asyncio.ensure_future(ask(c, 100 + i, timeout_s=10, priority=PRIORITY_LANE)) for i in range(12)]
    await asyncio.sleep(0.05)
    lat = []
    for i in range(4):
        t0 = time.perf_counter()
        await ask(c, i, timeout_s=2.0)  # never shed, never waits behind the queued lane calls
        lat.append(time.perf_counter() - t0)
    # One lane request may be running on the (serial) server: at most one service time plus our own.
    assert max(lat) < 0.15 * 2 + 0.1
    assert c.limiter.lane_in_flight <= 1
    for f in flood:
        f.cancel()
    await asyncio.gather(*flood, return_exceptions=True)


async def test_decider_uses_the_lane_inside_a_job(settings, db):
    import httpx

    from dgxmm.decider import service
    from dgxmm.decider.service import Decider, call_priority
    from dgxmm.decider.types import Option
    from dgxmm.settings_store import SettingsStore
    assert call_priority("diagnosis") == PRIORITY_BACKGROUND and call_priority("parser_tool") == PRIORITY_INTERACTIVE
    with service.job_context(service.JobBudget("job", 5)):
        assert call_priority("parser_tool") == PRIORITY_LANE
    seen = []

    class Spy(Limiter):
        async def acquire(self, priority, timeout_s):
            seen.append(priority)
            return await super().acquire(priority, timeout_s)

    d = Decider(db=db, settings=settings, store=SettingsStore(db, settings), deployer=lambda: None,
                http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(SlowServer(0.0).handler)))
    d.set_enabled(True)
    d._base_url = lambda: "http://decider:8100"
    d.client().limiter = Spy(2)
    opts = [Option("gpu_oom", "a"), Option("port_in_use", "b"), Option("other", "c")]
    await d.decide(feature="health", subject="s", state="a", question="q", options=opts)
    with service.job_context(service.JobBudget("job", 5)):
        await d.decide(feature="health", subject="s", state="b", question="q", options=opts)
    assert seen == [PRIORITY_BACKGROUND, PRIORITY_LANE]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_lane_cap(n):
    assert Limiter(n).lane_slots == max(n - 1, 1)


# ---------------- fix round 1: the job budget lives in the context and Decider.decide enforces it ----------------
def _opts():
    from dgxmm.decider.types import Option
    return [Option("gpu_oom", "a"), Option("port_in_use", "b"), Option("other", "c")]


def _decider(settings, db, srv):
    import httpx

    from dgxmm.decider.service import Decider
    from dgxmm.settings_store import SettingsStore
    d = Decider(db=db, settings=settings, store=SettingsStore(db, settings), deployer=lambda: None,
                http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(srv.handler)))
    d.set_enabled(True)
    d._base_url = lambda: "http://decider:8100"
    return d


async def test_every_decide_inside_a_job_counts_against_its_budget(settings, db):
    from dgxmm.decider.service import JobBudget, job_context
    srv = SlowServer(0.0)
    d = _decider(settings, db, srv)
    budget = JobBudget("health_check", 2)
    with job_context(budget):
        out = [await d.decide(feature="health", subject="s", state=f"x{i}", question="q", options=_opts())
               for i in range(5)]
    assert [o.source for o in out[:2]] == ["model", "model"] and all(o.source == "none" for o in out[2:])
    assert (budget.used, budget.refused) == (2, 3) and srv.served == 2
    assert len(d.recent()) == 2  # refusals are counted, not logged
    after = await d.decide(feature="health", subject="s", state="later", question="q", options=_opts())
    assert after.source == "model"  # outside the job: no budget


async def test_budget_is_not_used_when_the_decider_would_not_be_asked(settings, db):
    from dgxmm.decider.service import JobBudget, job_context
    d = _decider(settings, db, SlowServer(0.0))
    d.set_enabled(False)
    budget = JobBudget("j", 1)
    with job_context(budget):
        await d.decide(feature="health", subject="s", state="a", question="q", options=_opts())
        d.set_feature("health", enabled=False)
        d.set_enabled(True)
        await d.decide(feature="health", subject="s", state="b", question="q", options=_opts())
    assert (budget.used, budget.refused) == (0, 0)


async def test_fake_decider_honours_the_job_budget():
    from dgxmm.decider.service import JobBudget, job_context
    from tests.fakes import FakeDecider
    fake = FakeDecider({"health": ("gpu_oom", 0.95)})
    budget = JobBudget("j", 1)
    with job_context(budget):
        a = await fake.decide(feature="health", subject="s", state="a", question="q", options=_opts())
        b = await fake.decide(feature="health", subject="s", state="b", question="q", options=_opts())
    assert (a.source, b.source) == ("model", "none") and budget.refused == 1
