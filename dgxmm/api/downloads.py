import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.downloads import DownloadError

router = APIRouter(prefix="/api/downloads", tags=["downloads"])


class DownloadIn(BaseModel):
    repo_id: str
    revision: str | None = None
    include: list[str] | None = None


@router.get("")
def list_downloads(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return ctx.downloads.list()


@router.post("", status_code=202)
async def start(body: DownloadIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await asyncio.to_thread(ctx.downloads.start, body.repo_id, body.revision, body.include, p.user.id)
    except DownloadError as e:
        raise api_error(e.status, e.code, e.message)
    await ctx.downloads.launch(view["id"])
    audit(ctx.db, p.user.id, "download", body.repo_id, {"files": view["files"], "bytes": view["bytes_total"]})
    return view


@router.post("/{dl_id}/cancel")
async def cancel(dl_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    # async: cancel() publishes on the EventBroker (asyncio queues), which must run on the event loop
    try:
        view = ctx.downloads.cancel(dl_id)
    except DownloadError as e:
        raise api_error(e.status, e.code, e.message)
    audit(ctx.db, p.user.id, "download_cancelled", str(dl_id))
    return view
