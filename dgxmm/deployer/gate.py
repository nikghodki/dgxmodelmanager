"""Start-up gate: at most ``limit`` engines are starting or loading at the same time.

vLLM profiles the free GPU memory while it initialises and asserts that nothing else changed it meanwhile
("Error in memory profiling ... other processes sharing the same GPU release or allocate memory"), and its KV cache
size is derived from that profile ("No available memory for the cache blocks"). On GB10 unified memory every engine
start-up (and every wake-up, which re-allocates the KV cache) changes free memory for all the others, so two
start-ups at once can fail each other. The deployer therefore holds a slot from just before it creates an engine's
container until the deployment leaves starting/loading (ready, failed, stopped), and waiters queue FIFO.

``limit`` is read at call time (``Settings.max_concurrent_startups``); 0 or less means no limit.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable


class GateTimeout(Exception):
    """Waited longer than the queue timeout for a start-up slot."""


class GateWithdrawn(Exception):
    """The waiter was taken out of the queue by someone else (its deployment was stopped or removed)."""


POLL_S = 1.0  # waiters re-check at least this often (e.g. a raised max_concurrent_startups frees room)


@dataclass
class _Waiter:
    dep_id: int
    kind: str  # "launch" | "wake"
    since: float  # time.time() when the wait started


class StartupGate:
    def __init__(self, limit: Callable[[], int]):
        self._limit = limit
        self.holders: dict[int, str] = {}  # dep_id -> what it holds the slot for ("launch" | "wake" | "resumed")
        self.waiters: list[_Waiter] = []   # FIFO
        self.names: dict[int, str] = {}     # dep_id -> deployment name (for "waiting for <name>" reasons)
        self._changed = asyncio.Event()

    # ---------- state ----------
    def limit(self) -> int:
        try:
            n = int(self._limit())
        except (TypeError, ValueError):
            n = 1
        if n != getattr(self, "_last_limit", n):
            self._last_limit = n
            self._notify()  # the limit changed: waiters re-check now
        self._last_limit = n
        return n

    def limit_changed(self) -> None:
        """Call after changing max_concurrent_startups: wakes the waiters (they also re-check every POLL_S)."""
        self.limit()
        self._notify()

    def _has_room(self) -> bool:
        n = self.limit()
        return n <= 0 or len(self.holders) < n

    def holds(self, dep_id: int) -> bool:
        return dep_id in self.holders

    def position(self, dep_id: int) -> int | None:
        """1-based queue position, or None if not waiting."""
        for i, w in enumerate(self.waiters):
            if w.dep_id == dep_id:
                return i + 1
        return None

    def waiting(self, dep_id: int) -> _Waiter | None:
        return next((w for w in self.waiters if w.dep_id == dep_id), None)

    def blocking(self) -> list[str]:
        """Names of the deployments holding a slot (what the queue waits for)."""
        return [self.names.get(i, f"deployment {i}") for i in self.holders]

    # ---------- acquire / release ----------
    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def try_acquire(self, dep_id: int, kind: str = "launch", name: str | None = None) -> bool:
        """Take a slot now if one is free and nobody queued earlier is waiting for it."""
        if name:
            self.names[dep_id] = name
        if dep_id in self.holders:
            return True
        first = self.waiters[0].dep_id if self.waiters else None
        if self._has_room() and (first is None or first == dep_id):
            self.holders[dep_id] = kind
            self.waiters = [w for w in self.waiters if w.dep_id != dep_id]
            self._notify()
            return True
        return False

    def hold(self, dep_id: int, kind: str = "resumed", name: str | None = None) -> None:
        """Count a deployment that is already starting/loading (e.g. after a dgxmm restart), even over the limit."""
        if name:
            self.names[dep_id] = name
        self.holders[dep_id] = kind
        self._notify()

    def enqueue(self, dep_id: int, kind: str = "launch", name: str | None = None) -> None:
        if name:
            self.names[dep_id] = name
        if dep_id not in self.holders and self.waiting(dep_id) is None:
            self.waiters.append(_Waiter(dep_id, kind, time.time()))
            self._notify()

    async def acquire(self, dep_id: int, timeout_s: float, kind: str = "launch", name: str | None = None) -> None:
        """Wait (FIFO) for a slot. Raises GateTimeout after ``timeout_s``; cancellation leaves the queue."""
        if self.try_acquire(dep_id, kind, name):
            return
        self.enqueue(dep_id, kind, name)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout_s, 0)
        try:
            while not self.try_acquire(dep_id, kind):
                if self.waiting(dep_id) is None:  # withdrawn by someone else (deployment stopped / removed)
                    raise GateWithdrawn(f"deployment {dep_id} left the start-up queue")
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise GateTimeout(f"waited {timeout_s:g}s for a start-up slot")
                changed = self._changed
                try:
                    await asyncio.wait_for(changed.wait(), min(remaining, POLL_S))
                except asyncio.TimeoutError:
                    pass
        except BaseException:
            self.withdraw(dep_id)
            raise

    def withdraw(self, dep_id: int) -> None:
        """Leave the queue (no-op if not queued)."""
        before = len(self.waiters)
        self.waiters = [w for w in self.waiters if w.dep_id != dep_id]
        if len(self.waiters) != before:
            self._notify()

    def release(self, dep_id: int) -> None:
        if self.holders.pop(dep_id, None) is not None:
            self._notify()

    def forget(self, dep_id: int) -> None:
        """A deployment was removed: drop every trace of it."""
        self.withdraw(dep_id)
        self.release(dep_id)
        self.names.pop(dep_id, None)

    def info(self, dep_id: int) -> dict | None:
        """Why ``dep_id`` waits (None if it does not): position, what it waits for, since when."""
        w = self.waiting(dep_id)
        if w is None:
            return None
        pos = self.position(dep_id)
        names = self.blocking()
        n = self.limit()
        what = "wake up" if w.kind == "wake" else "start"
        rule = "one engine starts at a time" if n == 1 else f"at most {n} engines start at a time"
        ahead = f"; {pos - 1} queued ahead" if pos and pos > 1 else ""
        if names:
            reason = f"waiting for {', '.join(names)} to finish starting before it can {what} ({rule}{ahead})"
        else:
            reason = f"waiting for a start-up slot ({rule}{ahead})"
        return {"position": pos, "waiting_for": names, "reason": reason, "kind": w.kind,
                "since": w.since, "limit": n}
