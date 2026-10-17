"""Keeps the enabled decider running.

Every ``decider_probe_interval_s`` while the decider is enabled:
- ready: probe it (``GET /health`` + a one-question ``/v1/systemone``). A good probe closes an open circuit breaker
  at once (the breaker otherwise waits out its 60 s). ``decider_hang_probes`` failed probes in a row while the
  container still runs = hung: it is restarted (stop + start).
- failed (crashed, OOM-killed, vanished, failed to start): restarted.
- stopped (by an admin), starting/loading/queued, sleeping, unknown: left alone.

Restarts go through the normal deployer path (fit check, start-up gate) with exponential backoff
(``decider_restart_backoff_s`` doubling up to ``decider_restart_backoff_max_s``; reset after ``decider_stable_s``
of healthy operation), at most ``decider_max_restarts_per_hour`` attempts, failed ones included (a pruned image or a
docker error must not retry forever; then it gives up, with the reason in its status, until the decider is enabled
again: the page's Retry), and never while the memory watchdog reports pressure or the restart would not fit. Every
restart is audited.

When the memory watchdog stops the decider (runaway memory, or the first victim of auto-protect; dgxmm/watchdog.py),
that counts as a failure: the restart waits out the backoff (which doubles) and the memory pressure, and counts
toward the hourly limit, so a decider that keeps running away is not restarted in a loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable

from dgxmm.audit import audit

log = logging.getLogger(__name__)
HOUR = 3600.0


class DeciderSupervisor:
    def __init__(self, *, decider, deployer: Callable[[], Any], watchdog: Callable[[], Any], settings, db,
                 clock: Callable[[], float] = time.monotonic):
        self.decider, self._deployer, self._watchdog = decider, deployer, watchdog
        self.settings, self.db, self.clock = settings, db, clock
        self.reset()

    def reset(self) -> None:
        """Forget restart history and backoff (called when an admin enables the decider)."""
        self.restarts: deque[float] = deque()
        self.backoff_s = float(self.settings.decider_restart_backoff_s)
        self.next_attempt_at: float | None = None
        self.failed_probes = 0
        self.gave_up = False
        self.last_probe: dict | None = None
        self.last_action: str | None = None
        self.last_reason: str | None = None
        # The supervisor stopped the decider itself (hung) and could not start it again yet: (deployment id, reason).
        # A stopped decider with this set is ours to retry, not an admin's stop.
        self.pending_restart: tuple[int, str] | None = None

    def on_watchdog_stop(self, view: dict, reason: str) -> None:
        """The memory watchdog stopped the decider: restart it later, as after a failure (backoff, pressure)."""
        current = self.decider.deployment()
        if current is None or current["id"] != view["id"]:
            return
        now = self.clock()
        self.pending_restart = (view["id"], reason)
        self.failed_probes = 0
        self._schedule(now)
        self._did("watchdog_stopped", reason)

    # ---------- view ----------
    def status(self) -> dict:
        now = self.clock()
        self._trim(now)
        return {"last_action": self.last_action, "last_reason": self.last_reason, "probe": self.last_probe,
                "failed_probes": self.failed_probes, "restarts_last_hour": len(self.restarts),
                "max_restarts_per_hour": self.settings.decider_max_restarts_per_hour,
                "backoff_s": self.backoff_s, "gave_up": self.gave_up,
                "pending_restart": self.pending_restart[1] if self.pending_restart else None,
                "next_attempt_in_s": round(max(self.next_attempt_at - now, 0.0), 1)
                if self.next_attempt_at is not None else None}

    def _trim(self, now: float) -> None:
        while self.restarts and now - self.restarts[0] > HOUR:
            self.restarts.popleft()

    # ---------- loop ----------
    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("decider supervisor tick failed")
            await asyncio.sleep(self.settings.decider_probe_interval_s)

    async def tick(self) -> str:
        """One supervision step; returns what it did (idle | rotating | probe_ok | probe_failed | restarted |
        backoff | memory_pressure | gave_up | restart_failed | rolled_back; watchdog_stopped is set by
        on_watchdog_stop)."""
        if not self.decider.enabled:
            self.failed_probes, self.pending_restart = 0, None
            return self._did("idle")
        view = self.decider.deployment()
        deployer = self._deployer()
        if view is None or deployer is None:
            return self._did("idle")
        if view["id"] in getattr(deployer, "rotating", ()):
            # a key rotation stops a running decider and starts it again itself (with the new key): never race it,
            # spend no restart budget or backoff, and keep a pending restart for after the rotation
            self.failed_probes = 0
            return self._did("rotating", "the API key is being rotated; the rotation restarts the decider")
        state = view["state"]
        if self.pending_restart is not None and (self.pending_restart[0] != view["id"] or state != "stopped"):
            self.pending_restart = None  # someone else started (or removed) it
        switch_from = getattr(self.decider, "switch_from", None)
        if state == "ready" and switch_from and view.get("model_ref") == self.decider.configured_model:
            self.decider.set_switch_from(None)  # the new model is up: nothing to roll back any more
        if state == "failed" and switch_from and view.get("model_ref") != switch_from:
            return await self._rollback(view, switch_from)
        if state == "ready":
            ok, latency = await self._probe()
            now = self.clock()
            self.last_probe = {"ok": ok, "latency_ms": latency, "at": time.time()}
            if ok:
                self.failed_probes = 0
                client = self.decider.client()
                if client is not None and client.breaker.is_open() and latency is not None and latency < 1000:
                    client.breaker.success()  # answering promptly again (e.g. after a restart): stop rules-only mode
                if not self.restarts or now - self.restarts[-1] >= self.settings.decider_stable_s:
                    self.backoff_s = float(self.settings.decider_restart_backoff_s)  # stable again
                    self.next_attempt_at = None
                return self._did("probe_ok")
            self.failed_probes += 1
            if self.failed_probes < self.settings.decider_hang_probes:
                return self._did("probe_failed")
            return await self._restart(view, f"not answering health probes ({self.failed_probes} in a row)",
                                       hung=True)
        if state == "failed":
            return await self._restart(view, f"crashed: {view.get('last_error') or 'failed'}", hung=False)
        if state == "stopped" and self.pending_restart is not None:
            return await self._restart(view, self.pending_restart[1], hung=False)
        return self._did("idle")  # stopped by an admin, starting, queued, ...: not ours to touch

    async def _probe(self) -> tuple[bool, float | None]:
        client = self.decider.client()
        if client is None:
            return False, None
        t0 = time.perf_counter()
        try:
            ok = await asyncio.wait_for(client.health(timeout_s=self.settings.decider_probe_timeout_s),
                                        self.settings.decider_probe_timeout_s + 1)
        except Exception:
            ok = False
        return ok, round((time.perf_counter() - t0) * 1000, 1)

    def _did(self, action: str, reason: str | None = None) -> str:
        self.last_action = action
        if reason is not None:
            self.last_reason = reason
        return action

    def _pressure(self) -> str | None:
        wd = self._watchdog()
        level = getattr(wd, "level", "ok") if wd is not None else "ok"
        return None if level == "ok" else f"memory {level}: not restarting the decider"

    async def _restart(self, view: dict, reason: str, *, hung: bool) -> str:
        from dgxmm.deployer.service import DeployError, InsufficientMemory
        now = self.clock()
        self._trim(now)
        if self.gave_up:
            return self._did("gave_up")
        if self.next_attempt_at is not None and now < self.next_attempt_at:
            return self._did("backoff", reason)
        if (p := self._pressure()) is not None:
            return self._did("memory_pressure", f"{reason}; {p}")
        if len(self.restarts) >= self.settings.decider_max_restarts_per_hour:
            self.gave_up = True
            log.error("decider: %d restart attempts in the last hour; giving up until it is enabled again",
                      len(self.restarts))
            if self.last_action == "restart_failed" and self.last_reason:
                reason = self.last_reason  # "<why it was restarted>; restart failed: <docker's reason>"
            audit(self.db, None, "decider_auto_restart_gave_up", view["name"], {"restarts": len(self.restarts),
                                                                                 "reason": reason[:300]})
            return self._did("gave_up", reason)
        deployer = self._deployer()
        expected = view["state"]
        try:
            if hung:
                await deployer.stop(view["id"], None, reason="decider supervisor: not answering")
                self.pending_restart = (view["id"], reason)
                expected = "stopped"
            # an admin may have disabled, stopped, removed or started it meanwhile: their action wins
            now_view = self.decider.deployment()
            if not self.decider.enabled or now_view is None or now_view["id"] != view["id"] \
                    or now_view["state"] != expected:
                self.pending_restart = None
                return self._did("idle", f"{reason}; not restarted: the decider was changed meanwhile")
            await deployer.restart(view["id"], None)
        except InsufficientMemory as e:
            self._schedule(now)
            return self._did("memory_pressure", f"{reason}; not enough memory to restart ({e.message})")
        except DeployError as e:
            if e.code == "rotation_in_progress":
                # a rotation began after our stop: it restarts only a running decider, so keep the pending restart
                # (set by the hang path) and start it on the first tick after the rotation, with the new key
                return self._did("rotating", f"{reason}; restart waits for the API key rotation")
            self.restarts.append(now)  # a failed attempt counts toward the hourly give-up too
            self._schedule(now)
            log.warning("decider restart failed: %s", e.message)
            return self._did("restart_failed", f"{reason}; restart failed: {e.message}")
        self.pending_restart = None
        self.restarts.append(now)
        self.failed_probes = 0
        self._schedule(now)
        log.warning("decider restarted (%s)", reason)
        audit(self.db, None, "decider_auto_restart", view["name"], {"reason": reason[:300],
                                                                     "restarts_last_hour": len(self.restarts)})
        return self._did("restarted", reason)

    async def _rollback(self, view: dict, previous: str) -> str:
        """The decider failed right after an admin switched it to another model: go back to the previous one (same
        deployment) and make it the configured model again; the reason stays visible in the status."""
        from dgxmm.deployer.service import DeployError, InsufficientMemory
        new = view.get("model_ref")
        reason = f"{new} failed to start ({view.get('last_error') or 'failed'}); rolled back to {previous}"
        if (p := self._pressure()) is not None:
            return self._did("memory_pressure", f"{reason} pending; {p}")
        try:
            await self._deployer().restart(view["id"], None, model_ref=previous)
        except InsufficientMemory as e:
            return self._did("memory_pressure", f"{reason} pending; not enough memory ({e.message})")
        except DeployError as e:
            self.decider.set_switch_from(None)  # fall back to the normal crash handling next time
            return self._did("restart_failed", f"rolling back to {previous} failed: {e.message}")
        self.decider.set_model(previous)
        self.decider.set_switch_from(None)
        log.warning("decider: %s", reason)
        audit(self.db, None, "decider_model_rollback", view["name"], {"from": new, "to": previous,
                                                                       "error": (view.get("last_error") or "")[:300]})
        return self._did("rolled_back", reason)

    def _schedule(self, now: float) -> None:
        self.next_attempt_at = now + self.backoff_s
        self.backoff_s = min(self.backoff_s * 2, float(self.settings.decider_restart_backoff_max_s))
