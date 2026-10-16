import shutil

from fastapi import APIRouter, Depends

from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/host", tags=["host"])


def host_payload(ctx: AppContext, budget) -> dict:
    sample = ctx.budget.last_sample
    out = {"budget": budget.as_dict(),
           "external_top": [{"pid": p.pid, "name": p.name, "rss": p.rss, "gpu": p.gpu}
                            for p in sample.external_top] if sample else [],
           "gpu": sample.gpu if sample else {},
           "alert": ctx.watchdog.state() if ctx.watchdog else {"level": "ok"}}
    try:
        du = shutil.disk_usage(ctx.settings.hf_cache.local)
        out["disk"] = {"total": du.total, "free": du.free}
    except OSError:
        out["disk"] = None
    return out


@router.get("")
async def host(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    b = ctx.budget.current() or await ctx.budget.refresh()
    return host_payload(ctx, b)
