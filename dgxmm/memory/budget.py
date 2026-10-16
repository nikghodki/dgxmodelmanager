"""Host memory budget (spec §4.1)."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Callable

from dgxmm.config import Settings
from dgxmm.memory.hostmem import HostMemory, HostSample

COMMITTED_STATES = frozenset({"starting", "loading", "ready", "unknown"})
MEASURED_STATES = COMMITTED_STATES | {"sleeping", "stopping"}
CREDITED_STATES = frozenset({"ready"})  # reservation fully allocated: unattributed gap is ours

UsageRow = tuple[int, str, str, int, str | None]


@dataclass(frozen=True)
class DeploymentUsage:
    deployment_id: int
    name: str
    state: str
    reserved: int
    actual: int
    container_id: str | None = None


@dataclass(frozen=True)
class Budget:
    total: int
    available: int
    margin: int
    os_reserve: int
    usable: int
    committed: int
    managed_actual: int
    external: int
    free_for_new: int
    deployments: tuple[DeploymentUsage, ...]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["deployments"] = [asdict(u) for u in self.deployments]
        return d


def compute_budget(mem: HostMemory, usages: list[DeploymentUsage], settings: Settings) -> Budget:
    usable = mem.total - settings.margin_bytes - settings.os_reserve_bytes
    committed = 0
    for u in usages:
        if u.state in COMMITTED_STATES:
            committed += max(u.reserved, u.actual)
        elif u.state in ("sleeping", "stopping"):
            committed += u.actual
    managed_actual = sum(u.actual for u in usages if u.state in MEASURED_STATES)
    # GB10 unified memory: CUDA allocations are neither charged to the container cgroup nor reported
    # per process by NVML, so a ready model's reservation is in `used` but only partly in `actual`.
    # Credit that unattributed gap for READY deployments only (their allocation is complete); never for
    # starting/loading (still growing) so the gate stays conservative.
    ready_gap = sum(max(u.reserved - u.actual, 0) for u in usages if u.state in CREDITED_STATES)
    external = max(mem.used - managed_actual - ready_gap, 0)
    return Budget(total=mem.total, available=mem.available, margin=settings.margin_bytes,
                  os_reserve=settings.os_reserve_bytes, usable=usable, committed=committed,
                  managed_actual=managed_actual, external=external,
                  free_for_new=usable - committed - external, deployments=tuple(usages))


class BudgetService:
    def __init__(self, settings: Settings, probe, usage_provider: Callable[[], list[UsageRow]]):
        self.settings, self.probe, self.usage_provider = settings, probe, usage_provider
        self._current: Budget | None = None
        self.last_sample: HostSample | None = None

    def current(self) -> Budget | None:
        return self._current

    async def refresh(self) -> Budget:
        rows = self.usage_provider()
        ids = {cid for (_, _, _, _, cid) in rows if cid}
        sample = await asyncio.to_thread(self.probe.sample, ids)
        usages = [DeploymentUsage(i, n, st, res, sample.managed.get(cid, 0) if cid else 0, cid)
                  for (i, n, st, res, cid) in rows]
        self.last_sample = sample
        self._current = compute_budget(sample.mem, usages, self.settings)
        return self._current
