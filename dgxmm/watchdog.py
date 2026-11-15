"""Watch MemAvailable; alert and optionally stop the lowest-priority deployment (spec §4.5).

The decision model (a system deployment) is advisory and fails open, so it is sacrificed before any user
deployment: the watchdog stops it when its measured memory exceeds its reservation by more than
``decider_overuse_margin_gib`` (OVERUSE_SAMPLES budget samples in a row; GPU memory is not capped by its cgroup), and
it is the first victim when auto-protect needs to free memory. The decider supervisor is told and restarts it
later with backoff, never while memory is under pressure (dgxmm/decider/supervisor.py). Other system deployments
are never stopped.

Per node (multi-node): every node's budget (BudgetService.all()) is checked on its own, with its own alert level and
once-per-minute auto-protect limit; a victim is always a deployment on the pressured node, stopped through that node's
controller (Deployer.stop). A remote node whose memory reading is stale is not evaluated (no action on stale data).
The decider runs on the local node, so the runaway check and ``level`` (read by the decider supervisor) are the local
node's; ``state()`` reports the worst node, plus every node under "nodes" when remote nodes are configured."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from dgxmm.audit import audit
from dgxmm.config import LOCAL_NODE_ID
from dgxmm.db import utcnow

log = logging.getLogger(__name__)
GIB = 1024**3
OVERUSE_SAMPLES = 2
VICTIM_STATES = ("ready", "sleeping", "unknown")
LEVELS = ("ok", "warning", "critical")


@dataclass
class NodeAlert:
    level: str = "ok"
    message: str = ""
    since: str | None = None
    action: str | None = None
    last_action: float = float("-inf")

    def state(self) -> dict:
        return {"level": self.level, "message": self.message, "since": self.since, "action": self.action}


def is_decider(v: dict) -> bool:
    return bool(v.get("system")) and v.get("engine") == "decision"


class Watchdog:
    def __init__(self, settings, deployer, events, db, clock: Callable[[], float] = time.monotonic):
        self.settings, self.deployer, self.events, self.db, self.clock = settings, deployer, events, db, clock
        self.alerts: dict[str, NodeAlert] = {LOCAL_NODE_ID: NodeAlert()}  # node -> its alert
        self._over: dict[int, int] = {}  # decider deployment id -> consecutive samples over its reservation
        # decider id -> its state when stopping it as a victim failed: skipped as a victim until its state changes,
        # so auto-protect falls back to the user-deployment order instead of retrying a stop that fails
        self._unstoppable: dict[int, str] = {}
        # called with (view, reason) after the watchdog stopped a system deployment (the decider supervisor)
        self.system_stop_listeners: list[Callable[[dict, str], Any]] = []

    # the local node's alert (the decider supervisor reads and tests set ``level``)
    level = property(lambda self: self.alerts[LOCAL_NODE_ID].level,
                     lambda self, v: setattr(self.alerts[LOCAL_NODE_ID], "level", v))
    message = property(lambda self: self.alerts[LOCAL_NODE_ID].message)
    since = property(lambda self: self.alerts[LOCAL_NODE_ID].since)
    action = property(lambda self: self.alerts[LOCAL_NODE_ID].action)

    def _multi_node(self) -> bool:
        has_remote = getattr(self.deployer, "_has_remote", None)
        return bool(has_remote and has_remote())

    def _label(self, node: str) -> str:
        try:
            return self.deployer.nodes.config(node).label or node
        except Exception:
            return node

    def state(self) -> dict:
        """The local node's alert with a single node (as before); else the worst node's (a remote one's message starts
        with its label), and every node's under "nodes"."""
        if not self._multi_node():
            return self.alerts[LOCAL_NODE_ID].state()
        worst = LOCAL_NODE_ID
        for node, a in self.alerts.items():
            if LEVELS.index(a.level) > LEVELS.index(self.alerts[worst].level):
                worst = node
        out = self.alerts[worst].state()
        if worst != LOCAL_NODE_ID:
            out["message"] = f"{self._label(worst)}: {out['message']}"
        out["nodes"] = {n: a.state() for n, a in self.alerts.items()}
        return out

    async def check_nodes(self, budget) -> str:
        """Budget listener: ``budget`` (the local node's, as refresh() returns it), then every remote node's budget
        whose memory reading is fresh. Returns the worst level."""
        levels = [await self.check(budget)]
        service = getattr(self.deployer, "budget", None)
        budgets = service.all() if service is not None and hasattr(service, "all") else {}
        for node, b in budgets.items():
            if node == LOCAL_NODE_ID or not self.deployer.node_configured(node):
                continue
            if self.deployer._node_stale(node):  # fail safe: never act on an old reading
                continue
            levels.append(await self.check(b, node))
        return max(levels, key=LEVELS.index)

    async def check(self, budget, node: str = LOCAL_NODE_ID) -> str:
        alert = self.alerts.setdefault(node, NodeAlert())
        where = "" if node == LOCAL_NODE_ID else f" on node {node}"
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
        if level != alert.level:
            alert.level, alert.message, alert.since, alert.action = level, msg, utcnow().isoformat(), None
            self.events.publish("alert", self.state())
            log.warning("memory watchdog%s: %s %s", where, level, msg) if level != "ok" else \
                log.info("memory watchdog%s: ok", where)
        else:
            alert.message = msg
        if node == LOCAL_NODE_ID and await self._runaway_decider(budget):
            return level
        if level == "critical" and self.settings.auto_protect and self.clock() - alert.last_action >= 60:
            victim = self._victim(node)
            if victim:
                alert.last_action = self.clock()
                detail = {"available": avail} if node == LOCAL_NODE_ID else {"available": avail, "node": node}
                if is_decider(victim):
                    if not await self._stop_system(victim, "watchdog: memory critical (the decision model is "
                                                           "stopped before any user deployment)", detail, alert):
                        self._unstoppable[victim["id"]] = victim["state"]
                    return level
                try:
                    await self.deployer.stop(victim["id"], None, reason="watchdog: memory critical")
                    alert.action = f"stopped {victim['name']} (priority {victim['priority']})"
                    audit(self.db, None, "watchdog_stop", victim["name"], detail)
                except Exception as e:
                    alert.action = f"failed to stop {victim['name']}: {e}"
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
            await self._stop_system(v, reason, {"actual": u.actual, "reserved": reserved}, self.alerts[LOCAL_NODE_ID])
            self._over.pop(v["id"], None)
            return True
        self._over = {k: n for k, n in self._over.items() if k in seen}
        return False

    async def _stop_system(self, v: dict, reason: str, detail: dict, alert: NodeAlert) -> bool:
        log.warning("%s; stopping %s", reason, v["name"])
        try:
            await self.deployer.stop(v["id"], None, reason=reason)
            alert.action = f"stopped {v['name']} ({reason.removeprefix('watchdog: ')})"
            audit(self.db, None, "watchdog_stop", v["name"], {**detail, "reason": reason})
        except Exception as e:
            alert.action = f"failed to stop {v['name']}: {e}"
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

    def _victim(self, node: str = LOCAL_NODE_ID) -> dict | None:
        """The deployment to stop on ``node``: only deployments that run there."""
        views = self.deployer.list_views()
        current = {v["id"]: v["state"] for v in views}
        self._unstoppable = {i: st for i, st in self._unstoppable.items() if current.get(i) == st}
        on = self.deployer.nodes_by_id() if self._multi_node() else {}
        active = [v for v in views if v["state"] in VICTIM_STATES and on.get(v["id"], LOCAL_NODE_ID) == node]
        deciders = [v for v in active if is_decider(v) and v["id"] not in self._unstoppable]
        if deciders:
            return deciders[0]
        cands = [v for v in active if not v.get("system")]
        cands.sort(key=lambda v: (v["priority"], -v["reserved_bytes"]))
        return cands[0] if cands else None
