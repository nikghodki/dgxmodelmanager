"""Host memory budget (spec §4.1)."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Callable

from dgxmm.config import Settings
from dgxmm.memory.hostmem import HostMemory, HostSample

COMMITTED_STATES = frozenset({"queued", "starting", "loading", "ready", "unknown"})  # queued: reservation held
MEASURED_STATES = COMMITTED_STATES | {"sleeping", "stopping"}
CREDITED_STATES = frozenset({"ready"})  # reservation fully allocated: unattributed gap is ours

UsageRow = tuple[int, str, str, int, str | None, str]  # (id, name, state, reserved, container_id, node)


@dataclass(frozen=True)
class DeploymentUsage:
    deployment_id: int
    name: str
    state: str
    reserved: int
    actual: int
    container_id: str | None = None
    node: str = "local"


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
    # GB10 unified memory: CUDA allocations are not charged to the container cgroup. `actual` includes them
    # when NVML reports per-process GPU memory (driver 580+, DGXMM in the host PID namespace); otherwise a
    # ready model's reservation is in `used` but only partly in `actual`. Credit that unattributed gap for
    # READY deployments only (their allocation is complete); never for starting/loading (still growing) so
    # the gate stays conservative. A measured `actual` at or above the reservation leaves no gap.
    ready_gap = sum(max(u.reserved - u.actual, 0) for u in usages if u.state in CREDITED_STATES)
    external = max(mem.used - managed_actual - ready_gap, 0)
    return Budget(total=mem.total, available=mem.available, margin=settings.margin_bytes,
                  os_reserve=settings.os_reserve_bytes, usable=usable, committed=committed,
                  managed_actual=managed_actual, external=external,
                  free_for_new=usable - committed - external, deployments=tuple(usages))


class BudgetService:
    def __init__(self, settings: Settings, probe, usage_provider: Callable[[], list[UsageRow]], nodes=None,
                 node_probe=None):
        """``node_probe`` (NodeProbe): remote nodes' memory comes from its last sample (taken by the remote probe
        loop in a worker thread), so a refresh never waits on a remote node. Without it, ``probe.sample(node_id)``."""
        self.settings, self.probe, self.usage_provider = settings, probe, usage_provider
        self.nodes = nodes
        self.node_probe = node_probe
        self._current: dict[str, Budget] = {}
        self.last_sample: HostSample | None = None

    def current(self, node_id: str = "local") -> Budget | None:
        return self._current.get(node_id)

    def all(self) -> dict[str, Budget]:
        """Return all per-node budgets."""
        return dict(self._current)

    async def refresh(self) -> Budget:
        """Refresh budgets for all nodes (local and remote)."""
        rows = self.usage_provider()
        
        # Group usage rows by node
        by_node: dict[str, list[UsageRow]] = {}
        for row in rows:
            if len(row) == 6:
                # New format: (id, name, state, reserved, cid, node)
                node = row[5]
            else:
                # Old format: (id, name, state, reserved, cid) - default to local
                node = "local"
            if node not in by_node:
                by_node[node] = []
            by_node[node].append(row)
        
        # Always refresh local node, even if no rows
        local_rows = by_node.get("local", [])
        ids = {cid for row in local_rows if len(row) >= 5 and row[4] for cid in [row[4]]}
        sample = await asyncio.to_thread(self.probe.sample, ids)
        usages = []
        for row in local_rows:
            if len(row) == 6:
                i, n, st, res, cid, node = row
            else:
                i, n, st, res, cid = row
                node = "local"
            actual = sample.managed.get(cid, 0) if cid else 0
            usages.append(DeploymentUsage(i, n, st, res, actual, cid, node))
        self.last_sample = sample
        self._current["local"] = compute_budget(sample.mem, usages, self.settings)
        
        # Refresh remote nodes if node probe is available
        if self.nodes:
            for node_id in self.nodes.ids():
                if node_id != "local" and self.nodes.is_remote(node_id):
                    mem = self.node_probe.last(node_id) if self.node_probe is not None \
                        else self.probe.sample(node_id)
                    if mem:
                        node_rows = by_node.get(node_id, [])
                        usages = []
                        for row in node_rows:
                            if len(row) == 6:
                                i, n, st, res, cid, node = row
                            else:
                                # Should not happen for remote nodes, but handle it
                                i, n, st, res, cid = row
                                node = node_id
                            # For remote nodes, we don't have container-level memory yet
                            actual = 0
                            usages.append(DeploymentUsage(i, n, st, res, actual, cid, node))
                        self._current[node_id] = compute_budget(mem, usages, self.settings)
        
        return self._current["local"]
