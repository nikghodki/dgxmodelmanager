"""Placement options: which placements (one node, or the nodes together as a cluster) fit a deployment, and which
one to suggest ("it is a big model: deploy it on the cluster").

Rule: the fitting single node with the most free memory; else the cluster if it fits; else nothing. Reasons:
``fits``; ``too_big_for_one_node`` (a single node that does not fit while a cluster option exists, i.e. a placement
that needs less per node); ``insufficient_memory`` (a cluster that does not fit, or a single node when no cluster
option exists); ``node_stale`` (a remote node without a recent memory reading, or without a budget at all: never
fits, never recommended)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dgxmm.config import LOCAL_NODE_ID, Settings
from dgxmm.deployer.service import CLUSTER_ENGINES, Placement, cluster_ready
from dgxmm.memory.budget import Budget
from dgxmm.memory.estimate import MemoryEstimate, per_rank_bytes

FITS = "fits"
TOO_BIG_FOR_ONE_NODE = "too_big_for_one_node"
INSUFFICIENT_MEMORY = "insufficient_memory"
NODE_STALE = "node_stale"


def _free(budgets: dict[str, Budget], node_id: str) -> int:
    b = budgets.get(node_id)
    return b.free_for_new if b is not None else 0


def rank_bytes(engine: str, estimate: MemoryEstimate, placement: Placement,
               budgets: dict[str, Budget]) -> dict[str, int]:
    """Memory each node of ``placement`` needs (see memory.estimate.per_rank_bytes)."""
    return per_rank_bytes(engine, estimate, {n: _free(budgets, n) for n in placement.nodes})


@dataclass
class PlacementOption:
    placement: Placement
    label: str
    per_node: dict[str, int]
    free: dict[str, int]
    fits: bool
    recommended: bool
    reason: str

    def as_dict(self) -> dict:
        return {"placement": self.placement.model_dump(), "label": self.label, "per_node": dict(self.per_node),
                "free": dict(self.free), "fits": self.fits, "recommended": self.recommended, "reason": self.reason}


def cluster_candidate(settings: Settings, engine: str) -> Placement | None:
    """One cluster over every cluster-ready node (deployer.service.cluster_ready), the local node as head; None unless the
    engine supports clusters, the local node is cluster-ready and at least one other node is."""
    if engine not in CLUSTER_ENGINES:
        return None
    ready = [n.id for n in settings.all_nodes() if cluster_ready(n, engine)]
    if LOCAL_NODE_ID not in ready or len(ready) < 2:
        return None
    return Placement(kind="cluster", nodes=[LOCAL_NODE_ID] + [n for n in ready if n != LOCAL_NODE_ID])


def build_options(engine: str, estimate: MemoryEstimate, settings: Settings, budgets: dict[str, Budget],
                  stale: Callable[[str], bool] = lambda n: False,
                  allow_cluster: bool = True) -> list[PlacementOption]:
    """Every configured node as a single option (config order), plus the cluster option (``cluster_candidate``)
    when ``allow_cluster``; exactly one (or no) option is recommended."""
    placements = [Placement(kind="single", nodes=[n.id]) for n in settings.all_nodes()]
    cluster = cluster_candidate(settings, engine) if allow_cluster else None
    if cluster is not None:
        placements.append(cluster)
    opts = []
    for pl in placements:
        need = rank_bytes(engine, estimate, pl, budgets)
        free = {n: _free(budgets, n) for n in pl.nodes}
        if any(budgets.get(n) is None or stale(n) for n in pl.nodes):
            fits, reason = False, NODE_STALE
        elif all(need[n] <= free[n] for n in pl.nodes):
            fits, reason = True, FITS
        else:
            fits = False
            reason = TOO_BIG_FOR_ONE_NODE if pl.kind == "single" and cluster is not None else INSUFFICIENT_MEMORY
        opts.append(PlacementOption(placement=pl, label=pl.label(settings), per_node=need, free=free, fits=fits,
                                    recommended=False, reason=reason))
    singles = [o for o in opts if o.fits and o.placement.kind == "single"]
    if singles:
        max(singles, key=lambda o: o.free[o.placement.head]).recommended = True  # ties: the first in config order
    else:
        clusters = [o for o in opts if o.fits and o.placement.kind == "cluster"]
        if clusters:
            clusters[0].recommended = True
    return opts
