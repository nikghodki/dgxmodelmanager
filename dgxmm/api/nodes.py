"""Node management API (Task 6: weight copy)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, get_ctx, require_admin
from dgxmm.config import LOCAL_NODE_ID
from dgxmm.context import AppContext
from dgxmm.nodes.weights_sync import copy_weights

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


class CopyWeightsIn(BaseModel):
    ref: str
    revision: str


@router.post("/{node_id}/copy-weights", status_code=202)
async def copy_weights_to_node(
    node_id: str,
    body: CopyWeightsIn,
    p: Principal = Depends(require_admin),
    ctx: AppContext = Depends(get_ctx)
):
    """Copy model weights to a remote node."""
    # Validate node
    if node_id == LOCAL_NODE_ID:
        raise api_error(400, "invalid_node", "cannot copy to local node")
    
    if not ctx.nodes or node_id not in ctx.nodes.ids():
        raise api_error(400, "unknown_node", f"node {node_id!r} is not configured")
    
    try:
        view = await copy_weights(
            ctx.db,
            ctx.settings,
            ctx.nodes,
            node_id,
            body.ref,
            body.revision,
            p.user.id,
            ctx.events
        )
    except ValueError as e:
        raise api_error(400, "invalid_request", str(e))
    
    audit(ctx.db, p.user.id, "copy_weights", body.ref, {"node": node_id, "revision": body.revision})
    return view
