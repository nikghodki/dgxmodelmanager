import asyncio
import shutil
import time
from typing import Literal

from fastapi import APIRouter, Depends, Query

from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/host", tags=["host"])


def _deployment_names(ctx: AppContext) -> dict[int, str]:
    from sqlalchemy import select

    from dgxmm.db import Deployment
    with ctx.db.session() as s:
        return {i: n for i, n in s.execute(select(Deployment.id, Deployment.name))}


def _label(label: dict | None, names: dict[int, str]) -> dict | None:
    if not label:
        return None
    return {"deployment_id": label["deployment_id"], "name": label.get("name") or names.get(label["deployment_id"])}


def litellm_ready(ctx: AppContext) -> bool:
    """The LiteLLM integration is on and its last sync succeeded (as already known, no LiteLLM call): on-demand
    models can then be started by requests through LiteLLM."""
    sync = getattr(ctx, "litellm", None)
    if sync is None:
        return False
    try:
        active = sync.active()
    except Exception:
        return False
    last = getattr(sync, "last", None) or {}
    return bool(active and last.get("ok") is True and last.get("status") == "synced")


def host_payload(ctx: AppContext, budget) -> dict:
    sample = ctx.budget.last_sample
    top = sample.external_top if sample else []
    names = _deployment_names(ctx) if any(p.dgxmm_label for p in top) else {}
    out = {"budget": budget.as_dict(),
           # `command` is redacted by the probe (dgxmm.memory.redact); viewers see the same data.
           "external_top": [{"pid": p.pid, "name": p.name, "rss": p.rss, "gpu": p.gpu, "command": p.command,
                             "user": p.user, "user_scope": p.user_scope, "container": p.container,
                             "dgxmm_label": _label(p.dgxmm_label, names)}
                            for p in top],
           "gpu": sample.gpu if sample else {},
           "alert": ctx.watchdog.state() if ctx.watchdog else {"level": "ok"},
           # global on-demand defaults (a deployment's idle_timeout_s null = this); shown in the wizard and Mode card
           "ondemand": {"idle_timeout_s": ctx.settings.ondemand_idle_timeout_s,
                        "queue_timeout_s": ctx.settings.ondemand_queue_timeout_s,
                        # the deploy wizard defaults to on demand only when LiteLLM can start the model
                        "litellm_ready": litellm_ready(ctx)}}
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


def _managed_map(ctx: AppContext) -> dict[str, dict]:
    rows = ctx.deployer.usage_rows() if ctx.deployer is not None else []
    return {cid: {"deployment_id": dep_id, "name": name} for (dep_id, name, _state, _res, cid) in rows if cid}


def processes_payload(procs: list, scope: str, limit: int) -> dict:
    if scope == "external":
        procs = [p for p in procs if p.managed is None]
    elif scope == "managed":
        procs = [p for p in procs if p.managed is not None]
    groups: dict[str, dict] = {}
    host = {"rss": 0, "gpu": 0, "process_count": 0}
    for p in procs:
        if p.container is None:
            g = host
        else:
            g = groups.setdefault(p.container["id"], {**p.container, "rss": 0, "gpu": 0, "process_count": 0,
                                                      "managed": p.managed, "dgxmm_label": p.dgxmm_label})
        g["rss"] += p.rss
        g["gpu"] += p.gpu
        g["process_count"] += 1
    containers = sorted(groups.values(), key=lambda g: g["rss"] + g["gpu"], reverse=True)
    return {"scope": scope, "total": len(procs), "processes": [p.as_dict() for p in procs[:limit]],
            "containers": containers, "host": host}


@router.get("/processes")
async def processes(scope: Literal["all", "external", "managed"] = "all", limit: int = Query(50, ge=1, le=1000),
                    p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    """What is running on the host, biggest first. Readable by viewers; commands are redacted for everyone and
    /proc/<pid>/environ is never read."""
    t0 = time.perf_counter()

    def _collect():
        return ctx.probe.processes(_managed_map(ctx), _deployment_names(ctx))
    procs = await asyncio.to_thread(_collect)
    out = processes_payload(procs, scope, limit)
    out["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
