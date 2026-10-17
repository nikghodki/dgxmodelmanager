"""Watch MemAvailable; alert and optionally stop the lowest-priority deployment (spec §4.5).

The decision model (a system deployment) is advisory and fails open, so it is sacrificed before any user
deployment: the watchdog stops it when its measured memory exceeds its reservation by more than
``decider_overuse_margin_gib`` (OVERUSE_SAMPLES budget samples in a row; GPU memory is not capped by its cgroup), and
it is the first victim when auto-protect needs to free memory. The decider supervisor is told and restarts it
later with backoff, never while memory is under pressure (dgxmm/decider/supervisor.py). Other system deployments
are never stopped."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from dgxmm.audit import audit
from dgxmm.db import utcnow

log = logging.getLogger(__name__)
GIB = 1024**3
OVERUSE_SAMPLES = 2
VICTIM_STATES = ("ready", "sleeping", "unknown")


def is_decider(v: dict) -> bool:
    return bool(v.get("system")) and v.get("engine") == "decision"


class Watchdog:
    def __init__(self, settings, deployer, events, db, clock: Callable[[], float] = time.monotonic):
        self.settings, self.deployer, self.events, self.db, self.clock = settings, deployer, events, db, clock
        self.level, self.message, self.since, self.action = "ok", "", None, None
        self._last_action = float("-inf")
        self._over: dict[int, int] = {}  # decider deployment id -> consecutive samples over its reservation
        # decider id -> its state when stopping it as a victim failed: skipped as a victim until its state changes,
        # so auto-protect falls back to the user-deployment order instead of retrying a stop that fails
        self._unstoppable: dict[int, str] = {}
        # called with (view, reason) after the watchdog stopped a system deployment (the decider supervisor)
        self.system_stop_listeners: list[Callable[[dict, str], Any]] = []

    def state(self) -> dict:
        return {"level": self.level, "message": self.message, "since": self.since, "action": self.action}

    async def check(self, budget) -> str:
        margin = self.settings.margin_bytes
        avail = budget.available
        if avail < margin:
            level = "critical"
            msg = f"only {avail / GIB:.1f} GiB available, below the {margin / GIB:.0f} GiB safety margin"
        elif avail < 1.5 * margin:
            level = "warning"
            msg = f"{avail / GIB:.1f} GiB available, approaching the {margin / GIB:.0f} GiB safety margin"
        else:
            level, msg = "ok", ""
        if level != self.level:
            self.level, self.message, self.since, self.action = level, msg, utcnow().isoformat(), None
            self.events.publish("alert", self.state())
            log.warning("memory watchdog: %s %s", level, msg) if level != "ok" else log.info("memory watchdog: ok")
        else:
            self.message = msg
        if await self._runaway_decider(budget):
            return level
        if level == "critical" and self.settings.auto_protect and self.clock() - self._last_action >= 60:
            victim = self._victim()
            if victim:
                self._last_action = self.clock()
                if is_decider(victim):
                    if not await self._stop_system(victim, "watchdog: memory critical (the decision model is "
                                                           "stopped before any user deployment)", {"available": avail}):
                        self._unstoppable[victim["id"]] = victim["state"]
                    return level
                try:
                    await self.deployer.stop(victim["id"], None, reason="watchdog: memory critical")
                    self.action = f"stopped {victim['name']} (priority {victim['priority']})"
                    audit(self.db, None, "watchdog_stop", victim["name"], {"available": avail})
                except Exception as e:
                    self.action = f"failed to stop {victim['name']}: {e}"
                    log.exception("watchdog stop failed")
                self.events.publish("alert", self.state())
        return level

    async def _runaway_decider(self, budget) -> bool:
        """Stop a decider whose measured memory stays over its reservation + margin. True if one was stopped."""
        extra = int(self.settings.decider_overuse_margin_gib * GIB)
        usage = {u.deployment_id: u for u in getattr(budget, "deployments", ()) or ()}
        seen: set[int] = set()
        for v in self.deployer.list_views():
            if not is_decider(v) or v["state"] not in VICTIM_STATES or v["id"] not in usage:
                continue
            u = usage[v["id"]]
            reserved = max(u.reserved, v.get("reserved_bytes") or 0)
            if u.actual <= reserved + extra:
                continue
            seen.add(v["id"])
            self._over[v["id"]] = self._over.get(v["id"], 0) + 1
            if self._over[v["id"]] < OVERUSE_SAMPLES:
                continue
            reason = (f"watchdog: the decision model used {u.actual / GIB:.1f} GiB, over its {reserved / GIB:.1f} GiB "
                      f"reservation + {extra / GIB:.1f} GiB margin")
            await self._stop_system(v, reason, {"actual": u.actual, "reserved": reserved})
            self._over.pop(v["id"], None)
            return True
        self._over = {k: n for k, n in self._over.items() if k in seen}
        return False

    async def _stop_system(self, v: dict, reason: str, detail: dict) -> bool:
        log.warning("%s; stopping %s", reason, v["name"])
        try:
            await self.deployer.stop(v["id"], None, reason=reason)
            self.action = f"stopped {v['name']} ({reason.removeprefix('watchdog: ')})"
            audit(self.db, None, "watchdog_stop", v["name"], {**detail, "reason": reason})
        except Exception as e:
            self.action = f"failed to stop {v['name']}: {e}"
            log.exception("watchdog stop failed")
            self.events.publish("alert", self.state())
            return False
        self.events.publish("alert", self.state())
        for fn in list(self.system_stop_listeners):
            try:
                fn(v, reason)
            except Exception:
                log.exception("watchdog stop listener failed")
        return True

    def _victim(self) -> dict | None:
        views = self.deployer.list_views()
        current = {v["id"]: v["state"] for v in views}
        self._unstoppable = {i: st for i, st in self._unstoppable.items() if current.get(i) == st}
        active = [v for v in views if v["state"] in VICTIM_STATES]
        deciders = [v for v in active if is_decider(v) and v["id"] not in self._unstoppable]
        if deciders:
            return deciders[0]
        cands = [v for v in active if not v.get("system")]
        cands.sort(key=lambda v: (v["priority"], -v["reserved_bytes"]))
        return cands[0] if cands else None
