"""Health probe of ready deployments (ctx.health): a periodic engine.health() after the start-up.

The start-up monitor (Deployer._monitor) probes an engine only until it is ready, and the liveness check
(Deployer.check_liveness, every budget refresh) only asks Docker whether the container still runs. An engine whose
process is alive but no longer answers (vLLM's EngineCore died while the API server and the container stayed up; an
engine left asleep by a sleep that failed halfway) would stay "ready" for ever. Every ``engine_health_interval_s``
(0 = off) ``check_once`` probes each ready deployment with its engine's health() without the start-up check's test
completion (``completion=False``: vLLM / llama.cpp GET /health, 5 s timeout; a cluster: its head). A completion
every interval would show as prompt / generation tokens in the metrics (decider/common.py BUSY_KEYS), so schedules,
plans and proposals would never see an idle deployment; and under load it would queue behind busy slots. vLLM's
/health fails once its EngineCore is dead. All probes run concurrently and outside the deployment's lock (a probe
must never block a stop, sleep or wake). ``engine_health_failures`` failed probes in a row
fail the deployment through the normal path: "stopping", its container removed (a cluster: every rank,
cluster.fail), then "failed" with last_error; the state hooks deregister it from LiteLLM and tell the on-demand
controller (an on-demand deployment starts again on its next request).

Never probed (and never failed by it): deployments that are not ready (queued, starting and loading belong to the
start-up monitor, sleeping / waking / stopping / stopped / failed answer nothing), system deployments (the decision
model has its own supervisor, dgxmm/decider/supervisor.py), a deployment whose lock is held (a stop, sleep, wake or
restart is under way) and one on a node whose memory reading is stale (an unreachable node is not a broken engine).
A probe whose deployment changed meanwhile (any new deployment event, e.g. it was stopped and started again) is
discarded, and the count starts again after any change; one slow or failed probe alone never fails a deployment.
The counts live in memory: after a DGXMM restart they start from zero."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import func, select

from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer import cluster

log = logging.getLogger(__name__)
PROBED_STATES = ("ready",)


class EngineHealth:
    def __init__(self, settings, deployer):
        self.settings, self.deployer = settings, deployer
        self._failures: dict[int, tuple[int, int]] = {}  # dep_id -> (its last event id, failed probes in a row)

    def _epoch(self, dep_id: int) -> int:
        """The id of the deployment's latest event: any transition (or note) changes it."""
        with self.deployer.db.session() as s:
            return s.scalar(select(func.max(DeploymentEvent.id)).where(DeploymentEvent.deployment_id == dep_id)) or 0

    def _candidates(self) -> list[Deployment]:
        d = self.deployer
        with d.db.session() as s:
            rows = [r for r in s.scalars(select(Deployment)) if r.state in PROBED_STATES and not r.system]
        out = []
        for r in rows:
            nodes = cluster.nodes_of(r) if cluster.is_cluster(r) else [d._node_of(r)]
            if any(not d.node_configured(n) or d._is_remote(n) and d._node_stale(n) for n in nodes):
                continue  # an unreachable or removed node: not the engine's fault (liveness / reconcile own it)
            if d._lock(r.id).locked():
                continue  # a stop, sleep, wake or restart is under way
            out.append(r)
        return out

    async def check_once(self) -> list[int]:
        """Probe every ready deployment once; the ids of those failed now."""
        rows = self._candidates()
        live = {r.id for r in rows}
        self._failures = {i: v for i, v in self._failures.items() if i in live}
        results = await asyncio.gather(*(self._probe(r) for r in rows))
        failed = []
        for r, (epoch, ok) in zip(rows, results):
            if ok is None:
                continue
            if await self._record(r.id, epoch, ok):
                failed.append(r.id)
        return failed

    async def _probe(self, dep: Deployment) -> tuple[int, bool | None]:
        """(epoch before the probe, healthy?) — None: no verdict (the probe itself could not be made)."""
        epoch = self._epoch(dep.id)
        try:
            return epoch, bool(await self.deployer._healthy(dep, completion=False))
        except Exception as e:  # e.g. a timeout outside httpx, or a node removed from the configuration
            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                return epoch, False
            log.warning("health probe of deployment %s could not run: %s", dep.id, e)
            return epoch, None

    async def _record(self, dep_id: int, epoch: int, ok: bool) -> bool:
        """Count one probe result; True if the deployment was failed now."""
        if self._epoch(dep_id) != epoch:  # it changed while the probe ran: the answer is about something else
            self._failures.pop(dep_id, None)
            return False
        if ok:
            self._failures.pop(dep_id, None)
            return False
        last, n = self._failures.get(dep_id, (epoch, 0))
        n = n + 1 if last == epoch else 1
        self._failures[dep_id] = (epoch, n)
        limit = self.settings.engine_health_failures
        log.warning("deployment %s failed its health probe (%d of %d in a row)", dep_id, n, limit)
        if n < limit:
            return False
        lock = self.deployer._lock(dep_id)
        if lock.locked():
            return False  # an operation started meanwhile; look again next time
        async with lock:
            dep = self.deployer._get(dep_id)
            if dep is None or dep.state not in PROBED_STATES or self._epoch(dep_id) != epoch:
                self._failures.pop(dep_id, None)
                return False
            self._failures.pop(dep_id, None)
            await self.deployer.fail_unhealthy(dep, f"engine not answering its health check ({n} probes in a row)")
        return True

    async def run(self) -> None:
        interval = self.settings.engine_health_interval_s
        if interval <= 0:
            log.info("engine health probe disabled (engine_health_interval_s = 0)")
            return
        while True:
            await asyncio.sleep(interval)
            try:
                await self.check_once()
            except Exception:
                log.exception("engine health probe failed")
