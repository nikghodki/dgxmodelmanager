"""On-demand models, part 2: the controller (dgxmm/ondemand/controller.py) — start on request, make room, FIFO queue,
idle reaper (docs/superpowers/plans/2026-10-24-on-demand-models.md, Task 2). A fake deployer stands in for the real
one (states, fit check, stop with precheck); the clock is injected so the 5-minute / 15-minute rules run instantly."""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from dgxmm.db import AuditLog, DeploymentEvent
from dgxmm.deployer.service import DeployError, InsufficientMemory
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.ondemand.controller import OnDemandController, OnDemandTimeout, OnDemandUnavailable, Slot
from dgxmm.planner.fit import FitResult

GB = 1024**3
ACTIVE = ("planned", "queued", "starting", "loading", "ready", "unknown", "stopping")


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeDeployer:
    """The parts of dgxmm.deployer.service.Deployer the controller uses."""

    def __init__(self, total: int = 100 * GB):
        self.total = total
        self.deps: dict[int, dict] = {}
        self.restarts: list[int] = []
        self.wakes: list[int] = []
        self.stops: list[tuple[int, str, str | None]] = []
        self.notes: list[tuple[int, str, str | None]] = []
        self.state_hooks: list = []
        self.removal_hooks: list = []
        self.ondemand_status = None
        self.wake_failures: dict = {}
        self.restart_error: Exception | None = None
        self.restart_calls = 0
        self.restart_gate: asyncio.Event | None = None  # set: restart() blocks until it is set
        self.wake_background_failures = 0  # the next N wake-ups fail in the background for lack of memory
        self.lag: dict[int, int] = {}  # dep_id -> free() calls for which a stopped deployment still holds memory
        self.before_stop = None  # fn(dep_id) run just before the precheck (something changes meanwhile)
        self.wake_gate: asyncio.Event | None = None  # set: wake() blocks until it is set

    def add(self, dep_id: int, name: str, gb: int, state: str = "stopped", mode: str = "on_demand",
            system: bool = False, idle_timeout_s: int | None = None) -> None:
        self.deps[dep_id] = {"id": dep_id, "name": name, "model_name": f"m-{name}", "state": state, "mode": mode,
                             "system": system, "idle_timeout_s": idle_timeout_s, "reserved_bytes": gb * GB,
                             "actual_bytes": None, "last_error": None, "pending_mode": None}

    def free(self) -> int:
        used = 0
        for i, d in self.deps.items():
            if d["state"] in ACTIVE:
                used += d["reserved_bytes"]
            elif d["state"] == "sleeping":
                used += d["actual_bytes"] or 0
            elif self.lag.get(i):
                self.lag[i] -= 1
                used += d["reserved_bytes"]
        return self.total - used

    def view(self, dep_id: int) -> dict:
        d = self.deps.get(dep_id)
        if d is None:
            raise DeployError("not_found", f"deployment {dep_id} not found", 404)
        out = dict(d)
        out["ondemand"] = self.ondemand_status(dep_id, d["state"]) if self.ondemand_status and \
            d["mode"] == "on_demand" else None
        return out

    def list_views(self) -> list[dict]:
        return [self.view(i) for i in sorted(self.deps)]

    def set_state(self, dep_id: int, to: str, last_error: str | None = None) -> None:
        d = self.deps[dep_id]
        frm, d["state"] = d["state"], to
        if last_error is not None:
            d["last_error"] = last_error
        if to in ("ready", "failed", "stopped") and d.get("pending_mode"):  # as Deployer._transition
            d["mode"], d["pending_mode"] = d["pending_mode"], None
        for hook in self.state_hooks:
            asyncio.get_running_loop().create_task(hook(dep_id, frm, to))

    def _no_room(self, need: int) -> InsufficientMemory:
        free = self.free()
        return InsufficientMemory(FitResult(False, need, free, free - need,
                                            MemoryEstimate(need, 0, 0, 0, 0, 0, 0, "low"), None))

    async def restart(self, dep_id: int, user_id: int | None = None, **_) -> dict:
        self.restart_calls += 1
        await asyncio.sleep(0)
        if self.restart_gate is not None:
            await self.restart_gate.wait()
        d = self.deps[dep_id]
        if d["state"] not in ("stopped", "failed"):
            raise DeployError("not_stopped", f"deployment is {d['state']}; stop it first", 409)
        if self.restart_error is not None:
            raise self.restart_error
        if d["reserved_bytes"] > self.free():
            raise self._no_room(d["reserved_bytes"])
        self.restarts.append(dep_id)
        self.set_state(dep_id, "starting")
        return self.view(dep_id)

    async def wake(self, dep_id: int, user_id: int | None = None) -> dict:
        await asyncio.sleep(0)
        self.wakes.append(dep_id)
        if self.wake_gate is not None:
            await self.wake_gate.wait()
        d = self.deps[dep_id]
        if self.wake_background_failures:  # the slot was busy: the wake-up went on in the background, then failed
            self.wake_background_failures -= 1
            self.wake_failures[dep_id] = ("insufficient_memory", "needs 50.0 GiB but only 10.0 GiB is free")
            return self.view(dep_id)
        need = d["reserved_bytes"] - (d["actual_bytes"] or 0)
        if need > self.free():
            raise self._no_room(need)
        self.set_state(dep_id, "ready")
        return self.view(dep_id)

    async def stop(self, dep_id: int, user_id: int | None = None, reason: str = "user", precheck=None,
                   message: str | None = None, require_states=None) -> dict:
        d = self.deps[dep_id]
        if self.before_stop is not None:
            self.before_stop(dep_id)
        if require_states is not None and d["state"] not in require_states:
            raise DeployError("precheck", f"deployment is {d['state']}", 409)
        if precheck is not None and (d["state"] == "ready" or require_states is not None) and \
                (why := await precheck()):
            raise DeployError("precheck", why, 409)
        await asyncio.sleep(0)
        self.stops.append((dep_id, reason, message))
        self.set_state(dep_id, "stopped")
        return self.view(dep_id)

    def set_mode(self, dep_id: int, mode: str, idle_timeout_s: int | None, pending_mode: str | None = None) -> dict:
        d = self.deps[dep_id]
        d["mode"], d["idle_timeout_s"], d["pending_mode"] = mode, idle_timeout_s, pending_mode
        return self.view(dep_id)

    def note(self, dep_id: int, message: str, actor: str | None = None) -> None:
        self.notes.append((dep_id, message, actor))

    def remove(self, dep_id: int) -> None:
        del self.deps[dep_id]
        for hook in self.removal_hooks:
            hook(dep_id)


class FakeMetrics:
    def __init__(self):
        self.rings: dict[int, list[dict]] = {}

    def series(self, key):
        return self.rings.get(key, [])


@pytest.fixture
def od(settings, db):
    clock = Clock()
    dep = FakeDeployer()
    triggers = []
    refreshes = []

    async def refresh():
        refreshes.append(1)
        return SimpleNamespace(free_for_new=dep.free())
    ctx = SimpleNamespace(settings=settings, db=db, deployer=dep, metrics=FakeMetrics(),
                          budget=SimpleNamespace(refresh=refresh, current=lambda: SimpleNamespace(usable=dep.total)),
                          litellm=SimpleNamespace(trigger=lambda: triggers.append(1)))
    ctl = OnDemandController(ctx, clock=clock, wall=lambda: 1_700_000_000.0 + clock.t, retry_s=0.005,
                             settle_poll_s=0.001)
    ctl.attach(dep)
    dep.ondemand_status = ctl.status
    return SimpleNamespace(ctl=ctl, dep=dep, clock=clock, ctx=ctx, triggers=triggers)


async def until(cond, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.002)


async def use(od, dep_id: int) -> None:
    """One request that finishes right away (stamps the deployment's last request at the current clock)."""
    async with await od.ctl.acquire(dep_id):
        pass


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


# --- slots and starts -----------------------------------------------------------------------------

async def test_ready_deployment_returns_slot_immediately(od):
    od.dep.add(1, "a", 10, state="ready")
    slot = await od.ctl.acquire(1)
    assert isinstance(slot, Slot)
    assert od.ctl.status(1)["inflight"] == 1 and od.ctl.status(1)["phase"] == "serving"
    od.clock.advance(5)
    async with slot:
        pass
    st = od.ctl.status(1)
    assert st == {"inflight": 0, "queued": 0, "last_request_at": 1_700_000_000.0 + od.clock.t, "phase": "idle"}
    assert od.dep.restarts == [] and od.dep.stops == []
    async with slot:  # releasing twice never drives the counter negative
        pass
    assert od.ctl.status(1)["inflight"] == 0


async def test_stopped_fits_starts_and_waits(od, db):
    od.dep.add(1, "a", 10)
    queued = []

    async def on_queued():
        queued.append(1)
    task = asyncio.create_task(od.ctl.acquire(1, on_queued=on_queued))
    await until(lambda: od.dep.restarts == [1])
    assert od.ctl.status(1)["phase"] == "starting_for_request"
    assert not task.done()
    od.dep.set_state(1, "loading")
    await settle()
    assert not task.done()
    od.dep.set_state(1, "ready")
    slot = await asyncio.wait_for(task, 2)
    assert od.ctl.status(1)["inflight"] == 1
    slot.release()
    assert queued == []  # a start that fits is not a queue
    assert od.dep.notes == [(1, "Started for a request", "ondemand")]
    with db.session() as s:
        rows = s.scalars(select(AuditLog).where(AuditLog.action == "ondemand_start")).all()
        assert [(r.target, r.detail["actor"], r.detail["event"]) for r in rows] == \
            [("a", "ondemand", "Started for a request")]


async def test_concurrent_first_requests_start_once(od):
    od.dep.add(1, "a", 10)
    tasks = [asyncio.create_task(od.ctl.acquire(1)) for _ in range(10)]
    await until(lambda: od.dep.restarts == [1])
    await settle()
    od.dep.set_state(1, "ready")
    slots = await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert od.dep.restarts == [1]
    assert od.ctl.status(1)["inflight"] == 10
    for s in slots:
        s.release()
    assert od.ctl.status(1)["inflight"] == 0 and od.ctl.status(1)["phase"] == "idle"


async def test_sleeping_deployment_is_woken(od):
    od.dep.add(1, "a", 10, state="sleeping")
    slot = await asyncio.wait_for(od.ctl.acquire(1), 2)
    slot.release()
    assert od.dep.wakes == [1] and od.dep.restarts == []


async def test_not_on_demand_refused(od):
    od.dep.add(1, "a", 10, mode="always_on")
    od.dep.add(2, "sys", 10, system=True)
    for i in (1, 2):
        with pytest.raises(OnDemandUnavailable) as e:
            await od.ctl.acquire(i)
        assert e.value.code == "not_on_demand"
    with pytest.raises(OnDemandUnavailable) as e:
        await od.ctl.acquire(99)
    assert e.value.code == "not_found"
    assert od.dep.restarts == []


# --- make room and queue --------------------------------------------------------------------------

async def test_make_room_stops_idle_lru_first(od, db):
    od.dep.add(1, "a", 40, state="ready")
    od.dep.add(2, "b", 40, state="ready")
    od.dep.add(3, "c", 40)
    await use(od, 1)          # a used at t=1000
    od.clock.advance(100)
    await use(od, 2)          # b used at t=1100
    od.clock.advance(1000)    # both idle well past 5 min; c needs 40 GiB, 20 are free: one has to go
    queued = []

    async def on_queued():
        queued.append(1)
    task = asyncio.create_task(od.ctl.acquire(3, on_queued=on_queued))
    await until(lambda: od.dep.restarts == [3])
    assert od.dep.stops == [(1, "ondemand", "Put to sleep to make room for c")]  # the older one only
    assert od.dep.deps[2]["state"] == "ready"
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(task, 2)).release()
    assert queued == []  # room was made without waiting
    with db.session() as s:
        rows = s.scalars(select(AuditLog).where(AuditLog.action == "ondemand_sleep")).all()
        assert [(r.target, r.detail["actor"], r.detail["event"]) for r in rows] == \
            [("a", "ondemand", "Put to sleep to make room for c")]


async def test_busy_holder_queues_then_serves_after_idle(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    await use(od, 1)                 # a last used at t=1000
    od.clock.advance(10)
    queued = []

    async def on_queued():
        queued.append(1)
    task = asyncio.create_task(od.ctl.acquire(2, on_queued=on_queued))
    await until(lambda: queued == [1])
    await settle()
    assert od.dep.restarts == [] and od.dep.stops == []
    assert od.ctl.status(2) == {"inflight": 0, "queued": 1, "last_request_at": None, "phase": "queued"}
    od.clock.advance(289)            # t=1299: a used 299 s ago: still within its idle timeout
    await settle()
    assert od.dep.stops == [] and not task.done()
    od.clock.advance(1)              # t=1300: 300 s after a's last request
    await until(lambda: od.dep.restarts == [2])
    assert od.dep.stops == [(1, "ondemand", "Put to sleep to make room for b")]
    od.dep.set_state(2, "ready")
    (await asyncio.wait_for(task, 2)).release()
    assert queued == [1]             # awaited once, however many times it was retried
    assert od.ctl.status(2)["queued"] == 0


async def test_queue_timeout_raises_after_900s(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    holder = await od.ctl.acquire(1)  # in flight the whole time: never put to sleep
    task = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: od.ctl.status(2)["queued"] == 1)
    od.clock.advance(899)
    await settle()
    assert not task.done()
    od.clock.advance(1)
    with pytest.raises(OnDemandTimeout) as e:
        await asyncio.wait_for(task, 2)
    assert e.value.model == "m-b" and e.value.code == "ondemand_queue_timeout"
    assert e.value.message == "model m-b is busy; try again later"
    assert od.ctl.status(2) == {"inflight": 0, "queued": 0, "last_request_at": None, "phase": "sleeping"}
    assert (2, "Queued request timed out", "ondemand") in od.dep.notes
    assert od.dep.stops == [] and od.dep.restarts == []
    holder.release()


async def test_cancelled_waiter_releases_queue_entry(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    await use(od, 1)
    task = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: od.ctl.status(2)["queued"] == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert od.ctl.status(2)["queued"] == 0
    od.clock.advance(10_000)
    await settle()
    assert od.dep.restarts == [] and od.dep.stops == []  # nobody waits for b: nothing started or stopped for it


async def test_never_stops_inflight_or_engine_busy(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    assert await od.ctl.reap_once() == []  # first sight: idle timer starts now
    slot = await od.ctl.acquire(1)
    od.clock.advance(10_000)
    assert await od.ctl.reap_once() == []  # in flight through the gateway
    task = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: od.ctl.status(2)["queued"] == 1)
    await settle()
    assert od.dep.stops == []              # not for making room either
    slot.release()
    task.cancel()
    od.clock.advance(10_000)
    wall = 1_700_000_000.0 + od.clock.t
    od.ctx.metrics.rings[1] = [{"ts": wall - 5, "running": 1, "waiting": 0}]
    assert await od.ctl.reap_once() == []  # the engine itself reports a running request (direct call)
    od.ctx.metrics.rings[1] = [{"ts": wall - 5, "running": 0, "waiting": 2}]
    assert await od.ctl.reap_once() == []
    od.ctx.metrics.rings[1] = [{"ts": wall - 120, "running": 1, "waiting": 0}]  # stale: the gateway decides
    assert await od.ctl.reap_once() == [1]
    assert od.dep.stops == [(1, "ondemand", "Put to sleep after 5 min idle")]


async def test_never_stops_always_on_or_system(od):
    od.dep.add(1, "always", 30, state="ready", mode="always_on")
    od.dep.add(2, "sys", 30, state="ready", system=True)
    od.dep.add(3, "c", 40)
    od.dep.add(4, "busy", 40, state="ready")
    holder = await od.ctl.acquire(4)  # c could fit once busy goes idle: it is queued, not refused
    await od.ctl.reap_once()
    od.clock.advance(100_000)
    assert await od.ctl.reap_once() == []
    queued = []

    async def on_queued():
        queued.append(1)
    task = asyncio.create_task(od.ctl.acquire(3, on_queued=on_queued))
    await until(lambda: queued == [1])
    od.clock.advance(100_000)
    await settle()
    assert od.dep.stops == [] and od.dep.restarts == []
    task.cancel()
    holder.release()


async def test_reaper_stops_after_idle_timeout_and_respects_override(od, db):
    od.dep.add(1, "a", 10, state="ready")
    od.dep.add(2, "b", 10, state="ready", idle_timeout_s=600)
    assert await od.ctl.reap_once() == []
    od.clock.advance(299)
    assert await od.ctl.reap_once() == []
    od.clock.advance(1)
    assert await od.ctl.reap_once() == [1]
    od.clock.advance(299)
    assert await od.ctl.reap_once() == []
    od.clock.advance(1)
    assert await od.ctl.reap_once() == [2]
    assert od.dep.stops == [(1, "ondemand", "Put to sleep after 5 min idle"),
                            (2, "ondemand", "Put to sleep after 10 min idle")]
    assert od.ctl.status(1)["phase"] == "sleeping"
    with db.session() as s:
        rows = s.scalars(select(AuditLog).where(AuditLog.action == "ondemand_sleep")).all()
        assert [r.detail["event"] for r in rows] == ["Put to sleep after 5 min idle", "Put to sleep after 10 min idle"]


async def test_reaper_after_restart_waits_full_timeout(od):
    od.clock.advance(1_000_000)  # a DGXMM that has been up a long time restarts: counters are fresh
    od.dep.add(1, "a", 10, state="ready")
    assert od.ctl.status(1) == {"inflight": 0, "queued": 0, "last_request_at": None, "phase": "idle"}
    assert await od.ctl.reap_once() == []
    od.clock.advance(299)
    assert await od.ctl.reap_once() == []
    od.clock.advance(1)
    assert await od.ctl.reap_once() == [1]


async def test_reaper_timer_restarts_on_each_request_and_on_ready(od):
    od.dep.add(1, "a", 10)
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.restarts == [1])
    od.clock.advance(1000)            # a long start-up
    od.dep.set_state(1, "ready")
    slot = await asyncio.wait_for(task, 2)
    od.clock.advance(1000)            # a long generation
    slot.release()
    od.clock.advance(299)
    assert await od.ctl.reap_once() == []
    await use(od, 1)
    od.clock.advance(299)
    assert await od.ctl.reap_once() == []
    od.clock.advance(1)
    assert await od.ctl.reap_once() == [1]


async def test_start_failure_fails_waiters_once(od):
    od.dep.add(1, "a", 10)
    tasks = [asyncio.create_task(od.ctl.acquire(1)) for _ in range(5)]
    await until(lambda: od.dep.restarts == [1])
    await settle()
    od.dep.set_state(1, "failed", last_error="container exited with code 1")
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
    assert all(isinstance(r, OnDemandUnavailable) for r in results)
    assert {r.code for r in results} == {"start_failed"}
    assert all("container exited with code 1" in r.message for r in results)
    with pytest.raises(OnDemandUnavailable) as e:  # straight after: the same reason, no new start
        await od.ctl.acquire(1)
    assert "container exited with code 1" in e.value.message
    assert od.dep.restarts == [1]
    od.clock.advance(61)  # a later request may try again
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.restarts == [1, 1])
    od.dep.set_state(1, "ready")
    (await asyncio.wait_for(task, 2)).release()


async def test_start_refused_by_the_deployer_fails_waiters_once(od):
    od.dep.add(1, "a", 10)
    od.dep.restart_error = DeployError("docker_error", "could not start container: boom", 502)
    results = await asyncio.wait_for(asyncio.gather(*[od.ctl.acquire(1) for _ in range(3)],
                                                    return_exceptions=True), 2)
    assert [r.code for r in results] == ["docker_error"] * 3
    assert all("boom" in r.message for r in results)


async def test_fifo_order_across_models(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    od.dep.add(3, "c", 60)
    await use(od, 1)
    tb = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: od.ctl.status(2)["queued"] == 1)
    tc = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.ctl.status(3)["queued"] == 1)
    await settle()
    assert od.dep.restarts == []
    od.clock.advance(300)  # a goes idle: room for b (the head), not c
    await until(lambda: od.dep.restarts == [2])
    od.dep.set_state(2, "ready")
    b = await asyncio.wait_for(tb, 2)
    await settle()
    assert od.dep.restarts == [2] and not tc.done()
    b.release()
    od.clock.advance(300)  # b idle: c next
    await until(lambda: od.dep.restarts == [2, 3])
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(tc, 2)).release()
    assert [s[0] for s in od.dep.stops] == [1, 2]


async def test_set_mode_always_on_starts_or_refuses(od):
    od.dep.add(1, "a", 30)
    od.dep.add(2, "big", 80)
    view = await od.ctl.set_mode(1, "always_on", None, 7)
    # started; it stays on demand (registered through the gateway) until it is ready, then becomes always on
    assert view["mode"] == "on_demand" and view["pending_mode"] == "always_on" and od.dep.restarts == [1]
    assert od.triggers  # LiteLLM re-registers it
    with pytest.raises(InsufficientMemory):
        await od.ctl.set_mode(2, "always_on", 600, 7)
    d2 = od.dep.deps[2]
    assert (d2["mode"], d2["idle_timeout_s"], d2["pending_mode"]) == ("on_demand", None, None)  # refused: unchanged
    od.dep.set_state(1, "ready")
    await settle()
    assert od.dep.view(1)["mode"] == "always_on" and od.dep.view(1)["pending_mode"] is None
    view = await od.ctl.set_mode(1, "on_demand", 120, 7)  # back to on demand: the idle timer starts now
    assert view["mode"] == "on_demand" and od.dep.restarts == [1]
    od.clock.advance(119)
    assert await od.ctl.reap_once() == []
    od.clock.advance(1)
    assert await od.ctl.reap_once() == [1]


async def test_waiter_survives_a_switch_to_always_on(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    holder = await od.ctl.acquire(1)
    task = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: od.ctl.status(2)["queued"] == 1)
    holder.release()
    od.dep.deps[2]["mode"] = "always_on"  # switched while it waits: the controller no longer starts it ...
    od.clock.advance(400)
    await settle()
    assert od.dep.restarts == []
    od.dep.set_state(2, "ready")         # ... but a start by someone else serves the request
    (await asyncio.wait_for(task, 2)).release()


async def test_removed_deployment_releases_waiters(od):
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    od.dep.add(3, "c", 10)
    holder = await od.ctl.acquire(1)
    starting = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.dep.restarts == [3])
    queued = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: od.ctl.status(2)["queued"] == 1)
    od.dep.deps[3]["state"] = "stopped"  # stopped by the user and removed before any hook ran
    od.dep.remove(2)
    od.dep.remove(3)
    with pytest.raises(OnDemandUnavailable) as e:
        await asyncio.wait_for(queued, 2)
    assert e.value.code == "not_found"
    with pytest.raises(OnDemandUnavailable) as e:  # it was starting for this request: an administrator's action
        await asyncio.wait_for(starting, 2)
    assert e.value.code == "ondemand_stopped" and "removed by an administrator" in e.value.message
    assert od.ctl.status(2) == {"inflight": 0, "queued": 0, "last_request_at": None, "phase": "failed"}
    holder.release()


async def test_failed_or_removed_deployment_is_not_shown_sleeping(od):
    """A failed on-demand deployment is deregistered from LiteLLM, so requests through LiteLLM no longer wake it: its
    phase is ``failed`` (the UI shows the plain Failed badge), never ``sleeping``. The same for a removed one."""
    od.dep.add(1, "a", 10)
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.restarts == [1])
    od.dep.set_state(1, "failed", last_error="container exited with code 1")
    with pytest.raises(OnDemandUnavailable):
        await asyncio.wait_for(task, 2)
    assert od.ctl.status(1)["phase"] == "failed"
    assert od.ctl.status(1, "failed")["phase"] == "failed"  # what the deployment view passes
    assert od.ctl.status(1, "stopped")["phase"] == "sleeping"  # an admin fixed and stopped it: asleep again
    od.dep.set_state(1, "stopped")
    await settle()
    assert od.ctl.status(1)["phase"] == "sleeping"
    od.dep.add(2, "b", 10, state="stopped")
    assert od.ctl.status(2, "stopped")["phase"] == "sleeping"
    od.dep.remove(2)
    assert od.ctl.status(2)["phase"] == "failed" and od.ctl.status(2, "stopped")["phase"] == "failed"


async def test_run_loop_reaps_and_stops_on_cancel(od, monkeypatch):
    from dgxmm.ondemand import controller as mod
    monkeypatch.setattr(mod, "REAP_INTERVAL_S", 0.005)
    calls = []

    async def reap():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("one bad pass never ends the loop")
        return []
    od.ctl.reap_once = reap
    task = asyncio.create_task(od.ctl.run())
    await until(lambda: len(calls) >= 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- wiring with the real deployer ----------------------------------------------------------------

async def test_context_wires_the_controller(stack, settings, db):
    from dgxmm.context import AppContext
    from dgxmm.events import EventBroker
    ctx = AppContext(settings=settings, db=db, events=EventBroker(), deployer=stack.deployer, budget=stack.budget)
    assert ctx.ondemand.on_state in stack.deployer.state_hooks
    assert ctx.ondemand.on_removed in stack.deployer.removal_hooks
    assert stack.deployer.ondemand_status == ctx.ondemand.status


async def test_real_deployer_note_and_stop_message(stack, db):
    from dgxmm.deployer.service import DeployRequest
    dep = await stack.deployer.deploy(DeployRequest(name="tiny", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                    config={"max_model_len": 4096}), 1)
    stack.deployer.note(dep["id"], "Started for a request", "ondemand")
    state = stack.deployer.view(dep["id"])["state"]
    assert state not in ("ready", "sleeping")
    called = []

    async def precheck():
        called.append(1)
    with pytest.raises(DeployError) as e:  # left ready/sleeping before the lock: refused, never stopped
        await stack.deployer.stop(dep["id"], reason="ondemand", precheck=precheck, require_states=("ready", "sleeping"))
    assert e.value.code == "precheck" and called == []
    assert stack.deployer.view(dep["id"])["state"] == state
    await stack.deployer.stop(dep["id"], reason="ondemand", message="Put to sleep after 5 min idle")
    with db.session() as s:
        rows = s.scalars(select(DeploymentEvent).where(DeploymentEvent.deployment_id == dep["id"])
                         .order_by(DeploymentEvent.id)).all()
        got = [(r.message, r.actor) for r in rows]
    assert ("Started for a request", "ondemand") in got
    assert ("Put to sleep after 5 min idle", "ondemand") in got
    stack.deployer.note(999, "nothing", "ondemand")  # a removed deployment: ignored


# --- fix round 1 (review of 822fcff) -------------------------------------------------------------

async def test_queued_waiter_does_not_spin(od):
    """A queued request re-tries on a real change or every retry_s, never in a tight loop (also with a second
    request for the same model and one for another model waiting behind it)."""
    od.ctl.retry_s = 0.1
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 50)
    od.dep.add(3, "c", 50)
    holder = await od.ctl.acquire(1)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    tasks = [asyncio.create_task(od.ctl.acquire(i)) for i in (2, 2, 3)]
    await asyncio.sleep(0.5)
    elapsed = loop.time() - t0
    assert od.ctl.status(2)["queued"] == 2 and od.ctl.status(3)["queued"] == 1
    # b (the head) and c (may skip ahead into free memory): each at most once per retry interval
    assert od.dep.restart_calls <= 2 * (math.ceil(elapsed / 0.1) + 2), od.dep.restart_calls
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    holder.release()


async def test_impossible_request_fails_fast(od):
    od.dep.add(1, "always", 60, state="ready", mode="always_on")
    od.dep.add(2, "huge", 70)
    od.dep.add(3, "od", 10, state="ready")  # stopping every on-demand model frees at most 30 + 10 GiB
    queued = []

    async def on_queued():
        queued.append(1)
    with pytest.raises(OnDemandUnavailable) as e:
        await asyncio.wait_for(od.ctl.acquire(2, on_queued=on_queued), 2)
    assert e.value.code == "does_not_fit"
    assert e.value.message == "m-huge needs 70.0 GiB but at most 40.0 GiB can be freed for on-demand models"
    assert queued == [] and od.dep.stops == [] and od.ctl.status(2)["queued"] == 0


async def test_skip_ahead_into_free_memory_without_making_room(od):
    od.dep.add(1, "a", 50, state="ready")   # busy holder
    od.dep.add(2, "e", 20, state="ready")   # idle long enough, but too small to make room for b alone
    od.dep.add(3, "b", 60)                  # head: needs 30 more, only e's 20 could be freed now
    od.dep.add(4, "c", 10)                  # fits in the 30 free: starts at once, nothing stopped
    od.dep.add(5, "d", 25)                  # does not fit in what is free then (20); stopping e would do,
    holder = await od.ctl.acquire(1)        # but only the head makes room
    await od.ctl.reap_once()
    od.clock.advance(1000)                  # e is idle-eligible from now on
    tb = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.ctl.status(3)["queued"] == 1)
    await settle()
    assert od.dep.restarts == [] and od.dep.stops == []
    queued_c = []

    async def on_queued_c():
        queued_c.append(1)
    tc = asyncio.create_task(od.ctl.acquire(4, on_queued=on_queued_c))
    await until(lambda: od.dep.restarts == [4])
    od.dep.set_state(4, "ready")
    (await asyncio.wait_for(tc, 2)).release()
    assert queued_c == []
    td = asyncio.create_task(od.ctl.acquire(5))
    await until(lambda: od.ctl.status(5)["queued"] == 1)
    await settle()
    assert od.dep.stops == [] and od.dep.restarts == [4]
    assert not tb.done() and not td.done()
    for t in (tb, td):
        t.cancel()
    await asyncio.gather(tb, td, return_exceptions=True)
    holder.release()


async def test_deadline_bounds_a_slow_attempt(od):
    od.dep.add(1, "a", 10)
    od.dep.restart_gate = asyncio.Event()  # the deployer call hangs (e.g. behind a long docker stop)
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.restart_calls == 1)
    od.clock.advance(900)
    with pytest.raises(OnDemandTimeout):
        await asyncio.wait_for(task, 2)
    assert (1, "Queued request timed out", "ondemand") not in od.dep.notes  # it never waited in the queue
    od.dep.restart_gate.set()
    await settle()


async def test_make_room_waits_for_freed_memory(od):
    od.dep.add(1, "a", 40, state="ready")
    od.dep.add(2, "b", 40, state="ready")
    od.dep.add(3, "c", 40)
    await use(od, 1)
    od.clock.advance(100)
    await use(od, 2)
    od.clock.advance(1000)
    od.dep.lag[1] = 3  # a's memory shows up as free only some budget refreshes after its container is gone
    task = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.dep.restarts == [3])
    assert [s[0] for s in od.dep.stops] == [1]  # b was not stopped as well
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(task, 2)).release()


async def test_victim_rechecked_under_the_deployer_lock(od, db):
    od.dep.add(1, "a", 40, state="ready")
    od.dep.add(2, "c", 80)
    await use(od, 1)
    od.clock.advance(1000)

    def changed(dep_id):  # the user restarts it with new settings meanwhile: no longer ready
        od.dep.deps[dep_id]["state"] = "unknown"
    od.dep.before_stop = changed
    assert await od.ctl.reap_once() == []
    assert od.dep.stops == []
    with db.session() as s:
        assert s.scalars(select(AuditLog).where(AuditLog.action == "ondemand_sleep")).all() == []


async def test_sleeping_on_demand_deployments_count_as_idle(od):
    od.dep.add(1, "a", 40, state="sleeping")
    od.dep.deps[1]["actual_bytes"] = 30 * GB  # weights kept in memory while asleep
    od.dep.add(2, "b", 40, state="sleeping")
    od.dep.deps[2]["actual_bytes"] = 30 * GB
    od.dep.add(3, "c", 60)
    assert await od.ctl.reap_once() == []
    od.clock.advance(300)
    od.dep.add(4, "d", 1, state="ready")  # just seen: not idle yet
    task = asyncio.create_task(od.ctl.acquire(3))  # 40 free, needs 60: one sleeping model makes room
    await until(lambda: od.dep.restarts == [3])
    assert od.dep.stops == [(1, "ondemand", "Put to sleep to make room for c")]
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(task, 2)).release()
    assert await od.ctl.reap_once() == [2]  # and the idle reaper stops a sleeping one too
    assert od.dep.stops[-1] == (2, "ondemand", "Put to sleep after 5 min idle")


async def test_background_wake_without_memory_queues(od):
    od.dep.add(1, "a", 40, state="sleeping")
    od.dep.wake_background_failures = 1
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: len(od.dep.wakes) == 2)  # not a start failure: tried again
    slot = await asyncio.wait_for(task, 2)
    slot.release()


async def test_start_failure_backoff_doubles_and_resets(od):
    od.dep.add(1, "a", 10)

    async def fail_once():
        task = asyncio.create_task(od.ctl.acquire(1))
        n = len(od.dep.restarts)
        await until(lambda: len(od.dep.restarts) == n + 1)
        od.dep.set_state(1, "failed", last_error="boom")
        with pytest.raises(OnDemandUnavailable):
            await asyncio.wait_for(task, 2)

    async def refused() -> bool:
        n = len(od.dep.restarts)
        try:
            (await asyncio.wait_for(od.ctl.acquire(1), 0.2)).release()
        except OnDemandUnavailable as e:
            assert e.code == "start_failed"
            return len(od.dep.restarts) == n
        except asyncio.TimeoutError:
            pass
        return False
    for delay in (60, 120, 240, 480, 900, 900):
        await fail_once()
        od.clock.advance(delay - 1)
        assert await refused(), delay
        od.clock.advance(1)
    await fail_once()
    await od.ctl.set_mode(1, "on_demand", None, 7)  # a mode switch resets the backoff
    od.clock.advance(1)
    await fail_once()
    od.clock.advance(59)
    assert await refused()
    od.dep.deps[1]["config"] = {"max_model_len": 2048}  # reconfigured: a new config gets a new chance
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.deps[1]["state"] == "starting")
    od.dep.set_state(1, "ready")  # and a successful start resets it
    (await asyncio.wait_for(task, 2)).release()
    od.dep.deps[1]["state"] = "stopped"
    await fail_once()
    od.clock.advance(60)
    task = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.deps[1]["state"] == "starting")
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_unexpected_errors_are_mapped(od, caplog):
    od.dep.add(1, "a", 10)
    od.dep.restart_error = RuntimeError("connection to /var/run/secret-socket refused")
    with pytest.raises(OnDemandUnavailable) as e:
        await asyncio.wait_for(od.ctl.acquire(1), 2)
    assert e.value.code == "ondemand_error"
    assert "secret-socket" not in e.value.message


# --- fix round 2: freed memory that has not shown up yet (it counts as external meanwhile) ---------

async def test_no_false_does_not_fit_right_after_the_reaper_stopped_a_model(od):
    od.dep.add(1, "always", 20, state="ready", mode="always_on")
    od.dep.add(2, "x", 60, state="ready")
    od.dep.add(3, "b", 70)
    await od.ctl.reap_once()
    od.clock.advance(300)
    od.dep.lag[2] = 10  # x's memory comes back a few budget refreshes after its container is gone
    assert await od.ctl.reap_once() == [2]
    task = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.dep.restarts == [3])
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(task, 2)).release()


async def test_no_false_does_not_fit_after_stopping_a_victim(od):
    od.dep.add(1, "always", 20, state="ready", mode="always_on")
    od.dep.add(2, "x", 60, state="ready")
    od.dep.add(3, "b", 70)
    await od.ctl.reap_once()
    od.clock.advance(300)
    od.dep.lag[2] = 40  # longer than the settle polls
    task = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.dep.restarts == [3], timeout=5)
    assert [s[0] for s in od.dep.stops] == [2]
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(task, 2)).release()


async def test_skip_ahead_never_takes_memory_freed_for_the_head(od):
    od.dep.add(1, "holder", 30, state="ready")
    od.dep.add(2, "x", 50, state="ready")
    od.dep.add(3, "big", 60)
    od.dep.add(4, "small", 40)
    holder = await od.ctl.acquire(1)
    await od.ctl.reap_once()
    od.clock.advance(300)
    od.dep.lag[2] = 30
    tbig = asyncio.create_task(od.ctl.acquire(3))
    await until(lambda: od.ctl.status(3)["queued"] == 1)
    tsmall = asyncio.create_task(od.ctl.acquire(4))
    await until(lambda: od.dep.restarts == [3], timeout=5)
    od.dep.set_state(3, "ready")
    (await asyncio.wait_for(tbig, 2)).release()
    await settle()
    assert od.dep.restarts == [3] and not tsmall.done()  # 10 GiB left: small keeps waiting
    tsmall.cancel()
    await asyncio.gather(tsmall, return_exceptions=True)
    holder.release()


async def test_impossible_request_right_after_a_stop_queues_then_fails(od):
    od.dep.add(1, "always", 60, state="ready", mode="always_on")
    od.dep.add(2, "x", 10, state="ready")
    od.dep.add(3, "huge", 70)
    await od.ctl.reap_once()
    od.clock.advance(300)
    assert await od.ctl.reap_once() == [2]
    queued = []

    async def on_queued():
        queued.append(1)
    task = asyncio.create_task(od.ctl.acquire(3, on_queued=on_queued))
    await until(lambda: queued == [1])  # within the settle window of a stop: no verdict yet
    assert not task.done()
    od.clock.advance(1)
    with pytest.raises(OnDemandUnavailable) as e:
        await asyncio.wait_for(task, 2)
    assert e.value.code == "does_not_fit"


# --- final fix wave: switch to always on, administrator stops, queue cap, live counters ----------------------

async def test_switch_to_always_on_of_a_stopped_deployment_waits_as_on_demand(od):
    """On demand -> always on starts a stopped deployment and keeps it on demand until it is ready: requests that
    arrive during the start wait for that start (as on demand, through the gateway), start nothing themselves, and
    are answered once it is ready. It is never put to sleep meanwhile."""
    od.dep.add(1, "a", 30)
    view = await od.ctl.set_mode(1, "always_on", 600, 7)
    assert (view["mode"], view["pending_mode"], view["state"]) == ("on_demand", "always_on", "starting")
    assert view["idle_timeout_s"] == 600
    od.clock.advance(10_000)
    assert await od.ctl.reap_once() == []
    task = asyncio.create_task(od.ctl.acquire(1))
    await settle()
    assert not task.done() and od.dep.restarts == [1]
    od.dep.set_state(1, "ready")
    slot = await asyncio.wait_for(task, 2)
    assert od.dep.view(1)["mode"] == "always_on" and od.dep.view(1)["pending_mode"] is None
    slot.release()
    od.clock.advance(10_000)
    assert await od.ctl.reap_once() == []  # always on now


async def test_switch_to_always_on_of_a_sleeping_deployment_waits_as_on_demand(od):
    od.dep.add(1, "a", 30, state="sleeping")
    od.dep.wake_gate = asyncio.Event()
    switching = asyncio.create_task(od.ctl.set_mode(1, "always_on", None, 7))
    await until(lambda: od.dep.wakes == [1])
    v = od.dep.view(1)
    assert (v["mode"], v["pending_mode"]) == ("on_demand", "always_on")
    od.clock.advance(10_000)
    assert await od.ctl.reap_once() == []  # a pending switch is never put to sleep
    task = asyncio.create_task(od.ctl.acquire(1))
    await settle()
    assert not task.done() and od.dep.wakes == [1] and od.dep.restarts == []  # waits for the switch's wake-up
    od.dep.wake_gate.set()
    view = await asyncio.wait_for(switching, 2)
    assert view["mode"] == "always_on" and view["pending_mode"] is None
    (await asyncio.wait_for(task, 2)).release()


async def test_switch_to_always_on_started_meanwhile_keeps_the_switch(od):
    """A request started the deployment just before the switch's own start: the switch still completes when ready."""
    od.dep.add(1, "a", 30)
    od.dep.restart_gate = asyncio.Event()
    req = asyncio.create_task(od.ctl.acquire(1))
    await until(lambda: od.dep.restart_calls == 1)
    switching = asyncio.create_task(od.ctl.set_mode(1, "always_on", None, 7))
    await until(lambda: od.dep.restart_calls == 2)
    od.dep.restart_gate.set()
    view = await asyncio.wait_for(switching, 2)  # its own restart found it starting: not an error
    assert view["pending_mode"] == "always_on"
    od.dep.set_state(1, "ready")
    (await asyncio.wait_for(req, 2)).release()
    assert od.dep.view(1)["mode"] == "always_on"


@pytest.mark.parametrize("how", ["stop", "remove"])
async def test_administrator_stop_while_starting_for_a_request(od, how):
    """A Stop (or removal) of a deployment that is starting for requests: every waiter gets ondemand_stopped, and it
    is not a start failure (no backoff: the next request starts it again at once)."""
    od.dep.add(1, "a", 10)
    tasks = [asyncio.create_task(od.ctl.acquire(1)) for _ in range(3)]
    await until(lambda: od.dep.restarts == [1])
    await settle()
    if how == "stop":
        od.dep.set_state(1, "stopping")
        od.dep.set_state(1, "stopped")
    else:
        od.dep.deps[1]["state"] = "stopped"
        od.dep.remove(1)
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
    assert all(isinstance(r, OnDemandUnavailable) for r in results), results
    assert {r.code for r in results} == {"ondemand_stopped"}
    word = "stopped" if how == "stop" else "removed"
    assert all(f"{word} by an administrator" in r.message for r in results)
    if how == "stop":
        assert od.ctl._failures == {}
        task = asyncio.create_task(od.ctl.acquire(1))
        await until(lambda: od.dep.restarts == [1, 1])
        od.dep.set_state(1, "ready")
        (await asyncio.wait_for(task, 2)).release()


async def test_queue_cap_refuses_at_once(od):
    od.ctx.settings.ondemand_max_queued = 3
    od.dep.add(1, "a", 60, state="ready")
    od.dep.add(2, "b", 60)
    od.dep.add(3, "c", 10)
    await use(od, 1)
    waiting = [asyncio.create_task(od.ctl.acquire(2)) for _ in range(3)]
    await until(lambda: od.ctl.status(2)["queued"] == 3)
    for dep_id in (2, 3):  # full: any other request that would wait is refused straight away
        with pytest.raises(OnDemandUnavailable) as e:
            await asyncio.wait_for(od.ctl.acquire(dep_id), 1)
        assert e.value.code == "ondemand_queue_full" and "3" in e.value.message
    assert od.dep.restarts == []
    (await asyncio.wait_for(od.ctl.acquire(1), 1)).release()  # a ready model is answered whatever the queue holds
    waiting[0].cancel()
    await asyncio.gather(waiting[0], return_exceptions=True)
    late = asyncio.create_task(od.ctl.acquire(2))  # room again
    await until(lambda: od.ctl.status(2)["queued"] == 3)
    for t in waiting[1:] + [late]:
        t.cancel()
    await asyncio.gather(*waiting[1:], late, return_exceptions=True)
    assert od.ctl.status(2)["queued"] == 0


def test_max_queued_setting():
    from pydantic import ValidationError
    from dgxmm.config import Settings
    assert Settings().ondemand_max_queued == 64
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            Settings(ondemand_max_queued=bad)


async def test_counter_changes_publish_a_throttled_deployment_event(od):
    """Slot and queue counters publish a "deployment" event (the badges' data) at most once a second per deployment:
    the first change at once, later ones folded into one trailing event with the latest counts."""
    published = []
    od.ctx.events = SimpleNamespace(publish=lambda topic, data: published.append((topic, data["id"], data["ondemand"])))
    od.dep.add(1, "a", 10, state="ready")
    slots = [await od.ctl.acquire(1) for _ in range(5)]
    for s in slots:
        s.release()
    assert len(published) == 1 and published[0][:2] == ("deployment", 1) and published[0][2]["inflight"] == 1
    await until(lambda: len(published) == 2, timeout=3)
    assert published[1][2]["inflight"] == 0 and published[1][2]["phase"] == "idle"
    await asyncio.sleep(1.2)
    assert len(published) == 2  # nothing more without a change
    od.dep.add(2, "b", 95)  # does not fit next to a (busy): queued
    holder = await od.ctl.acquire(1)
    await asyncio.sleep(1.1)
    n = len(published)
    task = asyncio.create_task(od.ctl.acquire(2))
    await until(lambda: any(p[1] == 2 and p[2]["queued"] == 1 for p in published[n:]))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await until(lambda: published[-1][1] == 2 and published[-1][2]["queued"] == 0, timeout=3)
    holder.release()
