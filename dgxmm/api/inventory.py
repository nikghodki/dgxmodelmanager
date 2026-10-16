import asyncio

from fastapi import APIRouter, Depends

from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


def _with_support(ctx: AppContext, a) -> dict:
    d = a.as_dict()
    d["engines"] = [name for name, e in ctx.engines.items() if e.supports(a)]
    return d


@router.get("")
def list_inventory(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return [_with_support(ctx, a) for a in ctx.inventory.list()]


@router.post("/refresh")
async def refresh(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    items = await asyncio.to_thread(ctx.inventory.refresh)
    audit(ctx.db, p.user.id, "inventory_refreshed", "", {"count": len(items)})
    return [_with_support(ctx, a) for a in ctx.inventory.list()]
