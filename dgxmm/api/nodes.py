"""Nodes API: the configured nodes with their memory budgets (GET), and copying a snapshot from this node's HF cache
to a remote node (dgxmm/nodes/weights_sync.py)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.config import LOCAL_NODE_ID
from dgxmm.context import AppContext
from dgxmm.nodes.weights_sync import CopyError

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


def _reachable(ctx: AppContext, node_id: str, stale: bool) -> bool:
    """The last memory probe of a remote node succeeded (NodeProbe.reachable; no I/O here). Without that record
    (a probe that keeps none): a node with a recent reading is reachable. The local node always is."""
    if node_id == LOCAL_NODE_ID:
        return True
    probe = ctx.node_probe if ctx.node_probe is not None else getattr(ctx.budget, "probe", None)
    known = getattr(probe, "reachable", None)
    seen = known(node_id) if callable(known) else None
    return (not stale) if seen is None else bool(seen and not stale)


def _budget_out(b) -> dict | None:
    """A node budget as GET /api/host serialises it, minus the container ids of its usage rows: a cluster has one
    row per rank, and rank container ids never reach a view (R15)."""
    if b is None:
        return None
    out = b.as_dict()
    out["deployments"] = [{k: v for k, v in u.items() if k != "container_id"} for u in out["deployments"]]
    return out


@router.get("")
async def list_nodes(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    """Every configured node (config order; a single-node install: the implicit local node) with its label, whether it
    is remote, reachable and stale (no recent memory reading), and its memory budget as GET /api/host serialises it
    (None for a remote node never sampled). Any signed-in user. No Docker or SSH call: the budgets of the last
    refresh (one refresh first when there is none yet, as GET /api/host does)."""
    if ctx.budget.current() is None:
        await ctx.budget.refresh()
    out = []
    for n in ctx.settings.all_nodes():
        remote = n.id != LOCAL_NODE_ID
        stale = bool(remote and ctx.deployer is not None and ctx.deployer.node_stale(n.id))
        b = ctx.budget.current(n.id)
        out.append({"id": n.id, "label": n.label, "remote": remote, "reachable": _reachable(ctx, n.id, stale),
                    "stale": stale, "budget": _budget_out(b)})
    return out


class CopyWeightsIn(BaseModel):
    ref: str = Field(max_length=200)
    revision: str = Field(max_length=64)


@router.post("/{node_id}/copy-weights", status_code=202)
async def copy_weights(node_id: str, body: CopyWeightsIn, p: Principal = Depends(require_admin),
                       ctx: AppContext = Depends(get_ctx)):
    """Admin only (CSRF via require_admin), audited. The job is a Download row with kind "copy" (202: it runs in
    the background; the Downloads page shows and cancels it). The deploy refusal weights_missing carries this
    call as its action."""
    if ctx.weights_sync is None:
        if node_id == LOCAL_NODE_ID:
            raise api_error(400, "local_node", "weights are copied from the local node to a remote node")
        raise api_error(400, "unknown_node", f"node {node_id!r} is not configured")
    try:
        job = await ctx.weights_sync.copy_weights(node_id, body.ref, body.revision, p.user.id)
    except CopyError as e:
        raise api_error(e.status, e.code, e.message)
    audit(ctx.db, p.user.id, "copy_weights", body.ref, {"node": node_id, "revision": body.revision,
                                                        "download_id": job.id})
    return ctx.downloads.view(job.id)
