"""Nodes API: copy a snapshot from this node's HF cache to a remote node (dgxmm/nodes/weights_sync.py)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, get_ctx, require_admin
from dgxmm.config import LOCAL_NODE_ID
from dgxmm.context import AppContext
from dgxmm.nodes.weights_sync import CopyError

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


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
