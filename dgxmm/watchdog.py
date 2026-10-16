"""Watch MemAvailable; alert and optionally stop the lowest-priority deployment (spec §4.5)."""
from __future__ import annotations

import logging
import time
from typing import Callable

from dgxmm.audit import audit
from dgxmm.db import utcnow

log = logging.getLogger(__name__)
GIB = 1024**3


class Watchdog:
    def __init__(self, settings, deployer, events, db, clock: Callable[[], float] = time.monotonic):
        self.settings, self.deployer, self.events, self.db, self.clock = settings, deployer, events, db, clock
        self.level, self.message, self.since, self.action = "ok", "", None, None
        self._last_action = float("-inf")

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
        if level == "critical" and self.settings.auto_protect and self.clock() - self._last_action >= 60:
            victim = self._victim()
            if victim:
                self._last_action = self.clock()
                try:
                    await self.deployer.stop(victim["id"], None, reason="watchdog: memory critical")
                    self.action = f"stopped {victim['name']} (priority {victim['priority']})"
                    audit(self.db, None, "watchdog_stop", victim["name"], {"available": avail})
                except Exception as e:
                    self.action = f"failed to stop {victim['name']}: {e}"
                    log.exception("watchdog stop failed")
                self.events.publish("alert", self.state())
        return level

    def _victim(self) -> dict | None:
        cands = [v for v in self.deployer.list_views() if v["state"] in ("ready", "sleeping", "unknown")]
        cands.sort(key=lambda v: (v["priority"], -v["reserved_bytes"]))
        return cands[0] if cands else None
