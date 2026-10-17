import asyncio

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, Field

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.inventory.weights import DeleteError, WeightsDeleter

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


class DeleteIn(BaseModel):
    ref: str = Field(min_length=1, max_length=1024)


class CleanupIn(BaseModel):
    refs: list[str] | None = Field(default=None, max_length=500)


def _with_support(ctx: AppContext, a) -> dict:
    d = a.as_dict()
    d["engines"] = [name for name, e in ctx.engines.items() if e.supports(a)]
    return d


def _weights(ctx: AppContext) -> WeightsDeleter:
    if ctx.weights is None:
        ctx.weights = WeightsDeleter(ctx.settings, ctx.db, ctx.inventory, docker=ctx.docker, deployer=ctx.deployer,
                                     downloads=ctx.downloads)
    return ctx.weights


def _err(e: DeleteError):
    return api_error(e.status, e.code, e.message, **e.extra)


@router.get("")
def list_inventory(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return [_with_support(ctx, a) for a in ctx.inventory.list()]


@router.post("/refresh")
async def refresh(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    items = await asyncio.to_thread(ctx.inventory.refresh)
    audit(ctx.db, p.user.id, "inventory_refreshed", "", {"count": len(items), **p.audit_extra()})
    return [_with_support(ctx, a) for a in ctx.inventory.list()]


@router.get("/delete-preview")
async def delete_preview(ref: str = Query(min_length=1, max_length=1024), p: Principal = Depends(require_admin),
                         ctx: AppContext = Depends(get_ctx)):
    try:
        return await asyncio.to_thread(_weights(ctx).preview, ref)
    except DeleteError as e:
        raise _err(e)


@router.delete("")
async def delete_model(ref: str | None = Query(None, max_length=1024), body: DeleteIn | None = Body(None),
                       p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    ref = ref or (body.ref if body else None)
    if not ref:
        raise api_error(422, "missing_ref", "pass ?ref=<model ref> or {\"ref\": ...}")
    try:
        out = await asyncio.to_thread(_weights(ctx).delete, ref, p.user.id)  # audits success and refusals
    except DeleteError as e:
        if e.code == "delete_failed":
            ctx.events.publish("inventory", {"reason": "delete", "ref": ref})
        raise _err(e)
    ctx.events.publish("inventory", {"reason": "delete", "ref": ref})
    return out


@router.get("/cleanup-incomplete/preview")
async def cleanup_preview(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await asyncio.to_thread(_weights(ctx).cleanup_preview)


@router.post("/cleanup-incomplete")
async def cleanup_incomplete(body: CleanupIn = Body(default_factory=CleanupIn), p: Principal = Depends(require_admin),
                             ctx: AppContext = Depends(get_ctx)):
    out = await asyncio.to_thread(_weights(ctx).cleanup, body.refs, p.user.id)  # audits each deletion
    audit(ctx.db, p.user.id, "incomplete_cleanup", "", {
        "deleted": [d["ref"] for d in out["deleted"]], "bytes": out["freed_bytes"],
        "skipped": [{"ref": s["ref"], "code": s["code"]} for s in out["skipped"]],
        "failed": [f["ref"] for f in out["failed"]]})
    ctx.events.publish("inventory", {"reason": "cleanup", "count": len(out["deleted"])})
    return out
