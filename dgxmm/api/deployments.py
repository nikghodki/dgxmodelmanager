import asyncio

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


class PreviewIn(BaseModel):
    engine: str
    model_ref: str
    config: dict = {}
    replacing: int | None = None


class DeployIn(DeployRequest):
    actions: list[dict] = []


def deploy_error(e: DeployError):
    extra = {"fit": e.fit.as_dict()} if isinstance(e, InsufficientMemory) else {}
    return api_error(e.status, e.code, e.message, **extra)


@router.post("/preview")
async def preview(body: PreviewIn, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        return await ctx.deployer.preview(body.engine, body.model_ref, body.config, replacing=body.replacing)
    except DeployError as e:
        raise deploy_error(e)


@router.post("", status_code=201)
async def create(body: DeployIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    req = DeployRequest(**body.model_dump(exclude={"actions"}))
    applied = []
    try:
        if body.actions:
            applied = await ctx.deployer.apply_actions(body.actions, p.user.id)
        view = await ctx.deployer.deploy(req, p.user.id)
    except InsufficientMemory as e:
        await ctx.deployer.undo_actions(applied)
        try:
            plans = (await ctx.deployer.preview(req.engine, req.model_ref, req.config))["plans"]
        except DeployError:
            plans = []
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict(), plans=plans)
    except DeployError as e:
        await ctx.deployer.undo_actions(applied)
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "deploy", req.name, {"model": req.model_ref, "engine": req.engine, "actions": body.actions})
    return view


@router.get("")
def list_deployments(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return ctx.deployer.list_views()


@router.get("/{dep_id}")
def get_deployment(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        return ctx.deployer.view(dep_id, with_events=True)
    except DeployError as e:
        raise deploy_error(e)


@router.post("/{dep_id}/stop")
async def stop(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await ctx.deployer.stop(dep_id, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "stop", view["name"])
    return view


@router.post("/{dep_id}/start")
async def start(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await ctx.deployer.restart(dep_id, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "start", view["name"])
    return view


@router.delete("/{dep_id}", status_code=204)
async def remove(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        name = ctx.deployer.view(dep_id)["name"]
        await ctx.deployer.remove(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "remove", name)
    return Response(status_code=204)


@router.get("/{dep_id}/logs")
async def logs(dep_id: int, tail: int = Query(200, ge=1, le=5000), p: Principal = Depends(require_admin),
               ctx: AppContext = Depends(get_ctx)):
    try:
        view = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    return {"logs": await asyncio.to_thread(ctx.docker.logs, view["container_id"], tail)}


class SwitchIn(BaseModel):
    request: DeployRequest
    actions: list[dict] = []


class ReconfigureIn(BaseModel):
    config: dict
    model_ref: str | None = None
    actions: list[dict] = []


async def _run(ctx, p, action: str, target: str, coro, detail: dict | None = None):
    try:
        view = await coro
    except InsufficientMemory as e:
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict())
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, action, target, detail)
    return view


@router.post("/{dep_id}/sleep")
async def sleep(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "sleep", str(dep_id), ctx.deployer.sleep(dep_id, p.user.id))


@router.post("/{dep_id}/wake")
async def wake(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "wake", str(dep_id), ctx.deployer.wake(dep_id, p.user.id))


@router.post("/{dep_id}/switch")
async def switch(dep_id: int, body: SwitchIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "switch", f"{dep_id}->{body.request.name}",
                      ctx.deployer.switch(dep_id, body.request, body.actions, p.user.id),
                      {"model": body.request.model_ref, "actions": body.actions})


@router.post("/{dep_id}/reconfigure")
async def reconfigure(dep_id: int, body: ReconfigureIn, p: Principal = Depends(require_admin),
                      ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "reconfigure", str(dep_id),
                      ctx.deployer.reconfigure(dep_id, body.config, body.model_ref, body.actions, p.user.id),
                      {"config": body.config, "model_ref": body.model_ref, "actions": body.actions})
