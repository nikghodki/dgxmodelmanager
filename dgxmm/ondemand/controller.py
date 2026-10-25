"""Per-deployment on-demand controller (ctx.ondemand): start on request, make room, FIFO queue, idle reaper.

A request for an on-demand deployment asks ``acquire`` for a slot. A ready deployment answers at once. A stopped one is
started through the deployer (its fit check and start-up gate decide), a vLLM-sleeping one is woken. When it does not
fit, on-demand deployments idle for their timeout (ready or sleeping) are stopped to make room, least recently used
first. When the memory is held by deployments that are busy or used recently, the request waits in a FIFO queue shared
by all on-demand deployments, at most ``ondemand_queue_timeout_s``:

- only the head of the queue makes room; a later entry may start meanwhile if it fits in the memory free right now
  (it never stops anything);
- a request that would not fit even with every on-demand deployment stopped is refused at once (``does_not_fit``),
  judged on the usable budget minus what always-on / system deployments hold, never on the momentary free memory
  (memory of a model that was just stopped shows up as external for a while);
- while the head waits for memory it stopped victims for, that memory is reserved for it: later entries only start
  into what is free beyond the reservation;
- a waiting request looks again when something changes (a deployment changes state or leaves, a waiter leaves) or
  after ``retry_s`` / the next idle timeout to expire, never in a loop.

Requests for a deployment already starting wait for that start. ``reap_once`` (every 15 s from ``run``) stops on-demand
deployments idle for their timeout. Never stopped: a deployment with a slot in use, one whose engine reports
running/waiting requests in a fresh metrics sample, always-on and system deployments; the checks run again under the
deployer's lock (``stop(precheck=..., require_states=...)``) and a deployment being stopped hands out no slot. A
failed start is not retried for a while: 60 s, doubling up to 900 s, reset by a successful start, a mode switch or a
new configuration. A Stop or removal of a deployment while it starts for requests fails their waiters with
``ondemand_stopped`` and is not a start failure. At most ``ondemand_max_queued`` requests wait at a time (queued or
waiting for a start); beyond that a request is refused at once (``ondemand_queue_full``). A change of a deployment's
slot or queue counters publishes a "deployment" event, at most once a second per deployment (the UI's badges).
Counters live in memory only: after a DGXMM restart every deployment's idle timer starts from the moment the
controller first sees it (docs/superpowers/specs/2026-10-24-on-demand-models-design.md).

Switching an on-demand deployment that is not ready (stopped, failed or asleep) to always on starts it and records
the switch as pending (``Deployment.pending_mode``): it stays on demand, registered through the gateway, until the
deployer moves it to ready (or failed / stopped) and applies the new mode, so LiteLLM never lacks an entry for the
name while it starts and requests meanwhile wait for that start."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from dgxmm.audit import audit
from dgxmm.deployer.service import DeployError, InsufficientMemory
from dgxmm.memory.budget import COMMITTED_STATES

log = logging.getLogger(__name__)
ACTOR = "ondemand"            # events and audit rows of the controller
REAP_INTERVAL_S = 15.0        # how often run() looks for idle deployments
RETRY_S = 15.0                # a queued request re-checks at least this often (and on every real change)
ENGINE_FRESH_S = 60.0         # an engine metrics sample this recent showing activity counts as busy
FAIL_BACKOFF_S = 60.0         # after a failed start: no new start for this long, doubling per failure ...
FAIL_BACKOFF_MAX_S = 900.0    # ... up to this
TIMEOUT_NOTE_S = 60.0         # at most one "Queued request timed out" event per deployment in this window
SETTLE_POLLS = 20             # after stopping victims: budget refreshes (settle_poll_s apart) waiting for the memory
START_FROM = frozenset({"stopped", "failed", "sleeping"})
DOWN_STATES = frozenset({"stopped", "failed"})
SWITCH_STARTS = DOWN_STATES | {"sleeping"}  # on demand -> always on starts (or wakes) the deployment from these
PUBLISH_MIN_S = 1.0           # counter changes publish a deployment event at most this often per deployment
IDLE_STATES = ("ready", "sleeping")  # an on-demand deployment in these states may be put to sleep
SLEEPING_PHASE_STATES = frozenset({"stopped", "sleeping", "stopping"})  # failed: phase "failed" (deregistered)
GIB = float(1024**3)


class OnDemandError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class OnDemandUnavailable(OnDemandError):
    """Not an on-demand deployment, removed, does not fit, or its start failed (``message`` says why)."""


class OnDemandTimeout(OnDemandError):
    """The request waited ``ondemand_queue_timeout_s`` without its model becoming ready."""

    def __init__(self, model: str):
        super().__init__("ondemand_queue_timeout", f"model {model} is busy; try again later")
        self.model = model


class Slot:
    """One request in flight. Leaving the context (or ``release()``, idempotent) frees it and stamps the last
    request time."""

    def __init__(self, ctl: "OnDemandController", dep_id: int):
        self._ctl, self.dep_id, self._open = ctl, dep_id, True

    async def __aenter__(self) -> "Slot":
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self._open:
            self._open = False
            self._ctl._release(self.dep_id)


@dataclass
class _Entry:
    """A deployment's place in the FIFO queue and how many requests wait on it."""
    dep_id: int
    waiters: int = 0


@dataclass
class _Start:
    """A start (or wake-up) made for waiting requests; ``error`` is set when it failed."""
    dep_id: int
    kind: str = "start"
    error: OnDemandUnavailable | None = None
    name: str = ""


@dataclass
class _Failure:
    count: int
    until: float          # clock(): no new start before
    error: OnDemandUnavailable
    signature: str        # model + config at the failure: a new configuration gets a new chance


def _model(v: dict) -> str:
    return v.get("model_name") or v["name"]


def _stopped(name: str, how: str) -> OnDemandUnavailable:
    """A deployment starting for requests was stopped or removed by someone else (an administrator, a schedule or
    the memory watchdog): its waiters fail with this, and it is not a start failure (no backoff)."""
    who = "an administrator (or the memory watchdog)" if how == "stopped" else "an administrator"
    return OnDemandUnavailable("ondemand_stopped", f"{name} was {how} by {who} while it was starting for this request")


def _signature(v: dict) -> str:
    return json.dumps([v.get("model_ref"), v.get("config")], sort_keys=True, default=str)


def _frees(v: dict) -> int:
    """What stopping ``v`` returns to the budget (dgxmm/memory/budget.py: committed = max(reserved, actual), sleeping
    and stopping deployments count their measured memory)."""
    if v["state"] in COMMITTED_STATES or v["state"] == "planned":
        return max(v.get("reserved_bytes") or 0, v.get("actual_bytes") or 0)
    if v["state"] in ("sleeping", "stopping"):
        return v.get("actual_bytes") or 0
    return 0


class OnDemandController:
    def __init__(self, ctx: Any, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 retry_s: float = RETRY_S, settle_poll_s: float = 1.0, publish_s: float = PUBLISH_MIN_S):
        self.ctx, self.clock, self.wall = ctx, clock, wall
        self.retry_s, self.settle_poll_s, self.publish_s = retry_s, settle_poll_s, publish_s
        self._waiting = 0                          # requests waiting (queued or for a start): ondemand_max_queued
        self._switching: set[int] = set()          # set_mode is starting it for a switch to always on
        self._published: dict[int, float] = {}     # time.monotonic() of the last counter event per deployment
        self._publish_later: dict[int, asyncio.TimerHandle] = {}
        self._inflight: dict[int, int] = {}
        self._idle_since: dict[int, float] = {}    # clock(): last request / became ready / switched to on demand
        self._last_request: dict[int, float] = {}  # wall(): end of the last request (status)
        self._states: dict[int, str] = {}          # last state seen
        self._removed: set[int] = set()            # removed deployments (phase "failed" from then on)
        self._queue: dict[int, _Entry] = {}        # FIFO (insertion order) of deployments waiting for memory
        self._blocked: set[int] = set()            # queued deployments whose last make-room attempt found no room
        self._starts: dict[int, _Start] = {}
        self._attempts: dict[int, asyncio.Task] = {}
        self._failures: dict[int, _Failure] = {}
        self._stopping: set[int] = set()
        self._timeout_noted: dict[int, float] = {}
        self._room = asyncio.Lock()  # one start / make-room decision at a time
        self._changed_fut: asyncio.Future | None = None
        self._gen = 0  # bumped by every _notify(): did anything change since an attempt decided "no room"?
        self._no_room_gen: dict[int, int] = {}
        self._reserved: dict[int, int] = {}  # head dep_id -> bytes it stopped victims for (skip-ahead keeps off)
        self._last_stop: float | None = None  # clock() of the last on-demand stop (its memory may still lag)

    def attach(self, deployer: Any) -> None:
        """Follow the deployer's state changes and removals (AppContext calls this)."""
        hooks = getattr(deployer, "state_hooks", None)
        if hooks is not None and self.on_state not in hooks:
            hooks.append(self.on_state)
        removal = getattr(deployer, "removal_hooks", None)
        if removal is not None and self.on_removed not in removal:
            removal.append(self.on_removed)

    # ---------- requests ----------
    async def acquire(self, dep_id: int, *, on_queued: Callable[[], Awaitable[None]] | None = None) -> Slot:
        """A slot once the deployment is ready. ``on_queued`` is awaited once, the first time the request has to wait
        for memory (not for a start that fits). Raises OnDemandTimeout after ondemand_queue_timeout_s, and
        OnDemandUnavailable when the deployment is not on demand, is removed, can never fit or fails to start; any
        other error becomes OnDemandUnavailable("ondemand_error") (logged, never shown)."""
        try:
            return await self._acquire(dep_id, on_queued)
        except OnDemandError:
            raise
        except Exception as e:
            if not getattr(e, "_ondemand_logged", False):
                log.exception("on-demand request for deployment %s failed", dep_id)
            raise OnDemandUnavailable("ondemand_error", "the on-demand controller could not serve this request; "
                                                        "see the DGXMM log")

    async def _acquire(self, dep_id: int, on_queued: Callable[[], Awaitable[None]] | None) -> Slot:
        v = self._view(dep_id)
        self._observe(v)
        if v["mode"] != "on_demand" or v.get("system"):
            raise OnDemandUnavailable("not_on_demand", f"{v['name']} is not an on-demand deployment")
        if self._ready(v):
            return self._grant(dep_id)
        limit = self.ctx.settings.ondemand_max_queued
        if self._waiting >= limit:
            raise OnDemandUnavailable("ondemand_queue_full", f"{limit} requests already wait for on-demand models "
                                                             "(ondemand_max_queued); try again later")
        deadline = self.clock() + self.ctx.settings.ondemand_queue_timeout_s
        entry: _Entry | None = None
        start: _Start | None = None
        queued = False
        self._waiting += 1
        try:
            while True:
                changed = self._changed()  # taken before the checks: no change is missed
                try:
                    v = self._view(dep_id)
                except OnDemandUnavailable:
                    if start is not None and start.error is not None:
                        raise start.error  # removed while it was starting for this request
                    raise
                self._observe(v)
                if self._ready(v):
                    return self._grant(dep_id)
                if start is not None and start.error is not None:
                    raise start.error
                if self.clock() >= deadline:
                    if queued:
                        self._timed_out(v)
                    raise OnDemandTimeout(_model(v))
                st = self._starts.get(dep_id)
                if st is not None:  # starting for a request: wait for it, not in the queue
                    start, entry = st, self._leave(entry)
                    await self._wait(changed, deadline)
                    continue
                failed = self._recent_failure(v)
                if failed is not None:
                    raise failed
                if v["mode"] != "on_demand" or v["state"] not in START_FROM or dep_id in self._stopping \
                        or dep_id in self._switching:
                    # started by someone else (or by a switch to always on), being stopped, or switched to always
                    # on: wait, never start it here
                    self._leave(entry)
                    entry = None
                    await self._wait(changed, deadline)
                    continue
                entry = self._join(dep_id, entry)
                head = next(iter(self._queue))
                if head != dep_id and head not in self._blocked:
                    # the head decides first: wait for its attempt (or a change), then look again
                    await self._wait(changed, deadline, also=self._attempts.get(head))
                    continue
                task = self._attempt_task(dep_id, make_room=head == dep_id)
                await self._wait(None, deadline, also=task)
                if not task.done():
                    continue  # the deadline (or retry_s) passed while the deployer works: look again
                outcome = task.result()
                if outcome == "started":
                    continue
                changed = self._changed()  # a fresh one: what the attempt itself did must not wake us
                if outcome == "no_room":
                    if not queued:
                        queued = True
                        if on_queued is not None:
                            await on_queued()
                    if self._gen != self._no_room_gen.get(dep_id, self._gen):
                        continue  # something changed after the attempt decided
                    await self._wait(changed, deadline, cap=self._next_retry())
                else:  # "retry": its state changed under it; the hooks say when it settles
                    await self._wait(changed, deadline)
        finally:
            self._waiting -= 1
            self._leave(entry)

    def _grant(self, dep_id: int) -> Slot:
        self._inflight[dep_id] = self._inflight.get(dep_id, 0) + 1
        self._idle_since[dep_id] = self.clock()
        self._counts_changed(dep_id)
        return Slot(self, dep_id)

    def _release(self, dep_id: int) -> None:
        n = self._inflight.get(dep_id, 0) - 1
        if n > 0:
            self._inflight[dep_id] = n
        else:
            self._inflight.pop(dep_id, None)
        self._idle_since[dep_id] = self.clock()
        self._last_request[dep_id] = self.wall()
        self._counts_changed(dep_id)
        # no _notify(): a release frees no memory (it only restarts the idle timer, which _next_retry follows), and
        # waking the queue on every request would re-run the deployer's fit check per request under load

    def _ready(self, v: dict) -> bool:
        return v["state"] == "ready" and v["id"] not in self._stopping

    # ---------- the queue ----------
    def _join(self, dep_id: int, entry: _Entry | None) -> _Entry:
        if entry is not None and self._queue.get(dep_id) is entry:
            return entry
        self._leave(entry)
        e = self._queue.get(dep_id)
        if e is None:
            e = self._queue[dep_id] = _Entry(dep_id)
        e.waiters += 1
        self._counts_changed(dep_id)
        return e

    def _leave(self, entry: _Entry | None) -> None:
        if entry is None:
            return None
        entry.waiters -= 1
        self._counts_changed(entry.dep_id)
        if entry.waiters <= 0 and self._queue.get(entry.dep_id) is entry:
            del self._queue[entry.dep_id]
            self._blocked.discard(entry.dep_id)
            self._reserved.pop(entry.dep_id, None)  # gave up (timed out / hung up): its memory is free for others
            self._notify()  # the next in line may go
        return None

    def _attempt_task(self, dep_id: int, make_room: bool) -> asyncio.Task:
        """One start attempt per deployment at a time, shared by its waiters; it is not cancelled when a waiter
        leaves (a deployer call already under way completes)."""
        t = self._attempts.get(dep_id)
        if t is None or t.done():
            t = asyncio.get_running_loop().create_task(self._attempt(dep_id, make_room))
            t.add_done_callback(lambda done: self._attempt_done(dep_id, done))
            self._attempts[dep_id] = t
        return t

    def _attempt_done(self, dep_id: int, t: asyncio.Task) -> None:
        if self._attempts.get(dep_id) is t:
            del self._attempts[dep_id]
        if t.cancelled():
            return
        if t.exception() is not None:
            exc = t.exception()
            if not isinstance(exc, OnDemandError):
                log.error("on-demand start of deployment %s failed", dep_id, exc_info=exc)
                try:
                    exc._ondemand_logged = True  # its waiters map it without logging it again
                except AttributeError:
                    pass
            self._notify()  # its waiters fail; the next in line may go
        elif t.result() == "started":
            self._notify()
        # "no_room" / "retry": nothing changed for anyone else; no wake-up (that would be a busy loop)

    async def _attempt(self, dep_id: int, make_room: bool) -> str:
        """"started", "no_room" (wait in the queue) or "retry" (its state changed: look again). Only the head of the
        queue (``make_room``) stops idle deployments; others start only into memory free right now."""
        deployer = self.ctx.deployer
        async with self._room:
            tried: set[int] = set()
            while True:
                e = self._queue.get(dep_id)
                if e is None or e.waiters <= 0:
                    return "retry"  # nobody waits for it any more: start nothing
                v = self._view(dep_id)
                self._observe(v)
                if v["mode"] != "on_demand" or v["state"] not in START_FROM or dep_id in self._starts \
                        or dep_id in self._stopping or dep_id in self._switching:
                    return "retry"
                failed = self._recent_failure(v)
                if failed is not None:
                    raise failed
                kind = "wake" if v["state"] == "sleeping" else "start"
                gen = self._gen
                if not make_room and not await self._fits_beside_reservations(v):
                    self._no_room_gen[dep_id] = gen
                    return "no_room"
                try:
                    if kind == "wake":
                        getattr(deployer, "wake_failures", {}).pop(dep_id, None)
                        await deployer.wake(dep_id, None)
                    else:
                        await deployer.restart(dep_id, None)
                except InsufficientMemory as exc:
                    need, free = exc.fit.required, exc.fit.free_for_new
                    total = need + (v.get("actual_bytes") or 0 if kind == "wake" else 0)  # a wake keeps its own
                    most = self._capacity(exc.fit)
                    if most is not None and total > most and not tried and not self._settling():
                        self._reserved.pop(dep_id, None)
                        raise OnDemandUnavailable("does_not_fit", f"{_model(v)} needs {total / GIB:.1f} GiB but at "
                                                                  f"most {most / GIB:.1f} GiB can be freed for "
                                                                  "on-demand models")
                    victims = self._victims(dep_id, need - free, tried) if make_room else []
                    if not victims:
                        if make_room:
                            self._blocked.add(dep_id)
                        self._no_room_gen[dep_id] = gen
                        return "no_room"
                    stopped = 0
                    for victim in victims:
                        tried.add(victim["id"])
                        if await self._put_to_sleep(victim, f"Put to sleep to make room for {v['name']}",
                                                    reason="make_room", for_deployment=v["name"]):
                            stopped += _frees(victim)
                    if stopped:
                        self._reserved[dep_id] = need  # until it starts or gives up: later entries keep off
                        await self._settle(free + stopped, need)
                    continue  # the deployer's fit check decides again
                except DeployError as exc:
                    if exc.code in ("not_stopped", "not_sleeping"):
                        return "retry"  # started / stopped by someone else meanwhile
                    raise self._fail(v, OnDemandUnavailable(exc.code, f"{v['name']} could not start: {exc.message}"))
                self._queue.pop(dep_id, None)  # its waiters now wait for the start
                self._blocked.discard(dep_id)
                self._reserved.pop(dep_id, None)
                self._starts[dep_id] = _Start(dep_id, kind, name=v["name"])
                self._counts_changed(dep_id)  # queued -> starting for a request
                self._record(v, "Started for a request", "ondemand_start", kind=kind)
                try:
                    self._observe(self._view(dep_id))  # it may already be ready, or have failed
                except OnDemandUnavailable:
                    pass
                return "started"

    def _capacity(self, fit) -> int | None:
        """The most memory on-demand deployments could ever have: the usable budget minus what always-on and system
        deployments hold. Independent of ``external`` (where a just-stopped model's memory sits until the host returns
        it). None when no budget is known (then nothing is refused as too big)."""
        b = getattr(fit, "budget", None)
        if b is None:
            budget = getattr(self.ctx, "budget", None)
            b = budget.current() if budget is not None and hasattr(budget, "current") else None
        usable = getattr(b, "usable", None)
        if usable is None:
            return None
        held = sum(_frees(v) for v in self.ctx.deployer.list_views()
                   if v["mode"] != "on_demand" or v.get("system"))
        return usable - held

    def _settle_window(self) -> float:
        return SETTLE_POLLS * self.settle_poll_s

    def _settling(self) -> bool:
        """An on-demand stop happened so recently that its memory may not show as free yet."""
        return self._last_stop is not None and self.clock() - self._last_stop < self._settle_window()

    async def _fits_beside_reservations(self, v: dict) -> bool:
        """A skip-ahead start must leave the memory the head stopped victims for: free now minus the head's
        reservation must hold this deployment (its fit check then decides as usual)."""
        others = sum(n for d, n in self._reserved.items() if d != v["id"])
        if not others:
            return True
        budget = getattr(self.ctx, "budget", None)
        if budget is None:
            return False
        b = await budget.refresh()
        own = (v.get("reserved_bytes") or 0) - ((v.get("actual_bytes") or 0) if v["state"] == "sleeping" else 0)
        return b.free_for_new - others >= own

    async def _settle(self, expect_free: int, need: int) -> None:
        """After stopping victims: give the host a moment to show their memory as free (container teardown lags the
        budget), so that no further victim is stopped for memory that is about to come back."""
        budget = getattr(self.ctx, "budget", None)
        if budget is None:
            return
        for _ in range(SETTLE_POLLS):
            try:
                b = await budget.refresh()
            except Exception:
                log.warning("budget refresh failed while waiting for freed memory", exc_info=True)
                return
            if b.free_for_new >= min(expect_free, need):
                return
            await asyncio.sleep(self.settle_poll_s)

    # ---------- make room / reap ----------
    def _timeout(self, v: dict) -> int:
        return v.get("idle_timeout_s") or self.ctx.settings.ondemand_idle_timeout_s

    def _on_demand_views(self) -> list[dict]:
        out = []
        for v in self.ctx.deployer.list_views():
            self._observe(v)
            if v["mode"] == "on_demand" and not v.get("system"):
                out.append(v)
        return out

    def _engine_busy(self, dep_id: int) -> bool:
        """Requests that bypass the gateway (direct engine calls) still count: a fresh engine sample with running or
        waiting requests. No sample, or an old one, leaves it to the gateway's own counters."""
        store = getattr(self.ctx, "metrics", None)
        if store is None:
            return False
        try:
            ring = store.series(dep_id)
        except Exception:
            return False
        latest = ring[-1] if ring else None
        if not latest or self.wall() - (latest.get("ts") or 0) > ENGINE_FRESH_S:
            return False
        return (latest.get("running") or 0) + (latest.get("waiting") or 0) > 0

    def _idle_eligible(self, v: dict, now: float) -> bool:
        if v["mode"] != "on_demand" or v.get("system") or v["state"] not in IDLE_STATES:
            return False
        i = v["id"]
        if self._inflight.get(i) or i in self._stopping or i in self._starts or i in self._switching \
                or v.get("pending_mode"):
            return False  # in use, or being switched to always on
        if now - self._idle_since.setdefault(i, now) < self._timeout(v):
            return False
        return v["state"] == "sleeping" or not self._engine_busy(i)

    def _victims(self, for_dep: int, shortfall: int, tried: set[int]) -> list[dict]:
        """Idle on-demand deployments to stop so that ``shortfall`` bytes free, least recently used first; [] when
        even all of them would not be enough (then nothing is stopped)."""
        now = self.clock()
        cands = [v for v in self._on_demand_views()
                 if v["id"] != for_dep and v["id"] not in tried and self._idle_eligible(v, now) and _frees(v) > 0]
        cands.sort(key=lambda v: self._idle_since.get(v["id"], now))
        picked, freed = [], 0
        for v in cands:
            if freed >= shortfall:
                break
            picked.append(v)
            freed += _frees(v)
        return picked if freed >= shortfall else []

    def _next_retry(self) -> float:
        """Real seconds until a queued request looks again: the next idle timeout to expire, at most retry_s."""
        now, soonest = self.clock(), self.retry_s
        try:
            views = self._on_demand_views()
        except Exception:
            return soonest
        for v in views:
            if v["state"] in IDLE_STATES and not self._inflight.get(v["id"]):
                left = self._idle_since.get(v["id"], now) + self._timeout(v) - now
                if 0 < left < soonest:
                    soonest = left
        if self._settling():  # memory of a just-stopped model is coming back: look again soon
            soonest = min(soonest, max(self.settle_poll_s, 0.001))
        return max(soonest, min(self.retry_s, 0.05, self.settle_poll_s))

    async def _put_to_sleep(self, v: dict, message: str, **detail) -> bool:
        """Stop ``v`` if, under the deployer's lock, it is still ready or sleeping and still idle; True if stopped
        (then audited with actor ondemand)."""
        dep_id = v["id"]

        async def precheck():  # under the deployer's lock, no await: atomic with acquire() handing out slots
            try:
                cur = self._view(dep_id)
            except OnDemandUnavailable:
                return "removed"
            if not self._idle_eligible(cur, self.clock()):
                return "in use"
            self._stopping.add(dep_id)
            return None
        try:
            await self.ctx.deployer.stop(dep_id, None, reason=ACTOR, precheck=precheck, message=message,
                                         require_states=IDLE_STATES)
        except DeployError as e:
            if e.code != "precheck":
                log.warning("on-demand: could not stop deployment %s: %s", dep_id, e.message)
            return False
        finally:
            self._stopping.discard(dep_id)
            self._notify()
        self._last_stop = self.clock()
        self._record(v, message, "ondemand_sleep", note=False, **detail)
        return True

    async def reap_once(self) -> list[int]:
        """Stop the on-demand deployments idle for their timeout (ready or sleeping); their ids."""
        now = self.clock()
        stopped = []
        for v in self._on_demand_views():
            if not self._idle_eligible(v, now):
                continue
            if await self._put_to_sleep(v, f"Put to sleep after {self._timeout(v) / 60:g} min idle", reason="idle"):
                stopped.append(v["id"])
        return stopped

    async def run(self) -> None:
        while True:
            try:
                await self.reap_once()
            except Exception:
                log.exception("on-demand reaper failed")
            await asyncio.sleep(REAP_INTERVAL_S)

    # ---------- mode ----------
    async def set_mode(self, dep_id: int, mode: str, idle_timeout_s: int | None, user_id: int | None) -> dict:
        """Persist ``mode`` and ``idle_timeout_s`` (None = the global default). On demand -> always on of a
        deployment that is not ready starts (or wakes) it and leaves it on demand with ``pending_mode`` always_on
        until it is ready (the deployer applies it), so its gateway entry stays in LiteLLM and requests meanwhile
        wait for that start; the fit check refuses with InsufficientMemory and nothing changes. Always on -> on
        demand starts its idle timer now. Resets the start-failure backoff. LiteLLM re-registers it."""
        deployer = self.ctx.deployer
        before = deployer.view(dep_id)
        if mode == "always_on" and before["mode"] == "on_demand" and not before.get("system") \
                and before["state"] in SWITCH_STARTS:
            await self._switch_to_always_on(before, idle_timeout_s, user_id)
        else:
            deployer.set_mode(dep_id, mode, idle_timeout_s)  # clears a pending switch
        self._failures.pop(dep_id, None)
        if mode == "on_demand":
            self._idle_since[dep_id] = self.clock()
        litellm = getattr(self.ctx, "litellm", None)
        if litellm is not None:
            litellm.trigger()
        self._notify()
        return deployer.view(dep_id)

    async def _switch_to_always_on(self, v: dict, idle_timeout_s: int | None, user_id: int | None) -> None:
        deployer, dep_id = self.ctx.deployer, v["id"]
        deployer.set_mode(dep_id, "on_demand", idle_timeout_s, pending_mode="always_on")
        self._switching.add(dep_id)
        try:
            if v["state"] == "sleeping":
                await deployer.wake(dep_id, user_id)
            else:
                await deployer.restart(dep_id, user_id)
        except Exception as e:
            code = e.code if isinstance(e, DeployError) else None
            if code not in ("not_stopped", "not_sleeping"):
                try:
                    deployer.set_mode(dep_id, "on_demand", v.get("idle_timeout_s"))  # refused: nothing changes
                except DeployError:
                    pass
                raise
            # started or woken meanwhile (by a request): the switch completes when it is ready
        finally:
            self._switching.discard(dep_id)

    # ---------- live counters ----------
    def _counts_changed(self, dep_id: int) -> None:
        """Publish the deployment (its ``ondemand`` counters) now, or once ``publish_s`` after the last event."""
        if getattr(self.ctx, "events", None) is None or dep_id in self._publish_later:
            return  # nobody listens, or an event is already due (it carries the latest counts)
        wait = self._published.get(dep_id, -math.inf) + self.publish_s - time.monotonic()
        if wait <= 0:
            self._publish(dep_id)
            return
        try:
            self._publish_later[dep_id] = asyncio.get_running_loop().call_later(wait, self._publish, dep_id)
        except RuntimeError:  # no running loop: drop it (the next state event shows the counts)
            pass

    def _publish(self, dep_id: int) -> None:
        self._publish_later.pop(dep_id, None)
        self._published[dep_id] = time.monotonic()
        try:
            view = self.ctx.deployer.view(dep_id)
        except DeployError:
            return  # removed
        except Exception:
            log.exception("could not publish the on-demand counters of deployment %s", dep_id)
            return
        self.ctx.events.publish("deployment", view)

    # ---------- state ----------
    def status(self, dep_id: int, state: str | None = None) -> dict:
        """``state``: the deployment's current state when the caller has it (deployment views). A failed or removed
        deployment is phase ``failed``: LiteLLM no longer lists it, so it is not asleep waiting for a request."""
        state = "failed" if dep_id in self._removed else state or self._states.get(dep_id)
        e = self._queue.get(dep_id)
        queued = e.waiters if e is not None else 0
        inflight = self._inflight.get(dep_id, 0)
        if queued:
            phase = "queued"
        elif dep_id in self._starts:
            phase = "starting_for_request"
        elif inflight:
            phase = "serving"
        elif state == "failed":
            phase = "failed"
        elif state in SLEEPING_PHASE_STATES:
            phase = "sleeping"
        else:
            phase = "idle"
        return {"inflight": inflight, "queued": queued, "last_request_at": self._last_request.get(dep_id),
                "phase": phase}

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        """Deployer state hook: follow starts that end (ready or failed) and wake up waiting requests."""
        try:
            self._observe(self.ctx.deployer.view(dep_id))
        except DeployError:
            pass
        self._notify()

    def on_removed(self, dep_id: int) -> None:
        st = self._starts.get(dep_id)
        if st is not None and st.error is None:
            st.error = _stopped(st.name or f"deployment {dep_id}", "removed")
        later = self._publish_later.pop(dep_id, None)
        if later is not None:
            later.cancel()
        self._published.pop(dep_id, None)
        for d in (self._inflight, self._idle_since, self._last_request, self._states, self._queue, self._starts,
                  self._failures, self._timeout_noted, self._no_room_gen, self._reserved):
            d.pop(dep_id, None)
        self._stopping.discard(dep_id)
        self._blocked.discard(dep_id)
        self._removed.add(dep_id)
        self._notify()  # its waiters find it gone

    def _observe(self, v: dict) -> None:
        dep_id, state = v["id"], v["state"]
        self._removed.discard(dep_id)  # an id used again by a new deployment
        prev = self._states.get(dep_id)
        self._states[dep_id] = state
        now = self.clock()
        if state == "ready":
            if prev != "ready" or dep_id not in self._idle_since:
                self._idle_since[dep_id] = now  # up (again), or first seen since DGXMM started: a full idle timeout
            self._starts.pop(dep_id, None)
            self._failures.pop(dep_id, None)  # a successful start resets the backoff
            return
        st = self._starts.get(dep_id)
        if st is None:
            return
        if state in ("stopping", "stopped"):  # stopped by an administrator (or the watchdog): not a failure
            st.error = _stopped(v["name"], "stopped")
            del self._starts[dep_id]
            self._notify()
            return
        why = None
        if state in DOWN_STATES:
            why = v.get("last_error") or f"it is {state}"
        elif state == "sleeping" and st.kind == "wake":
            failure = getattr(self.ctx.deployer, "wake_failures", {}).get(dep_id)
            if failure and failure[0] == "insufficient_memory":
                del self._starts[dep_id]  # the background wake-up found no room: queue like any other start
                self._notify()
                return
            why = failure[1] if failure else None
        if why:
            st.error = self._fail(v, OnDemandUnavailable("start_failed", f"{v['name']} could not start: {why}"))
            del self._starts[dep_id]
            self._notify()

    def _fail(self, v: dict, err: OnDemandUnavailable) -> OnDemandUnavailable:
        """Record a failed start: no new start for 60 s, 120 s, 240 s ... (at most 900 s) unless reset."""
        sig = _signature(v)
        old = self._failures.get(v["id"])
        count = old.count + 1 if old is not None and old.signature == sig else 1
        delay = min(FAIL_BACKOFF_S * 2 ** (count - 1), FAIL_BACKOFF_MAX_S)
        self._failures[v["id"]] = _Failure(count, self.clock() + delay, err, sig)
        return err

    def _recent_failure(self, v: dict) -> OnDemandUnavailable | None:
        f = self._failures.get(v["id"])
        if f is None:
            return None
        if f.signature != _signature(v):  # reconfigured / switched since: a new chance
            del self._failures[v["id"]]
            return None
        return f.error if self.clock() < f.until else None

    def _view(self, dep_id: int) -> dict:
        try:
            return self.ctx.deployer.view(dep_id)
        except DeployError as e:
            if e.code == "not_found":
                raise OnDemandUnavailable("not_found", f"deployment {dep_id} was removed")
            raise

    def _timed_out(self, v: dict) -> None:
        now = self.clock()
        last = self._timeout_noted.get(v["id"])
        if last is not None and now - last < TIMEOUT_NOTE_S:
            return
        self._timeout_noted[v["id"]] = now
        self._record(v, "Queued request timed out", "ondemand_queue_timeout")

    def _record(self, v: dict, message: str, action: str, note: bool = True, **detail) -> None:
        """A deployment event (actor ondemand) and an audit row."""
        if note:
            fn = getattr(self.ctx.deployer, "note", None)
            if fn is not None:
                try:
                    fn(v["id"], message, ACTOR)
                except Exception:
                    log.exception("could not record the on-demand event of deployment %s", v["id"])
        db = getattr(self.ctx, "db", None)
        if db is not None:
            try:
                audit(db, None, action, v["name"], {"actor": ACTOR, "deployment_id": v["id"], "event": message,
                                                    **detail})
            except Exception:
                log.exception("could not audit the on-demand event of deployment %s", v["id"])

    # ---------- wake-ups ----------
    def _changed(self) -> asyncio.Future:
        if self._changed_fut is None or self._changed_fut.done():
            self._changed_fut = asyncio.get_running_loop().create_future()
        return self._changed_fut

    def _notify(self) -> None:
        self._gen += 1
        fut, self._changed_fut = self._changed_fut, None
        if fut is not None and not fut.done():
            fut.set_result(None)

    async def _wait(self, changed: asyncio.Future | None, deadline: float, cap: float | None = None,
                    also: asyncio.Future | None = None) -> None:
        """Until ``changed`` / ``also`` completes, ``cap`` (default retry_s) real seconds pass, or the deadline."""
        timeout = max(0.0, min(self.retry_s if cap is None else cap, deadline - self.clock()))
        aws = [f for f in (changed, also) if f is not None]
        if aws:
            await asyncio.wait(aws, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        else:
            await asyncio.sleep(timeout)
