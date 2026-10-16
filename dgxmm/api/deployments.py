import asyncio

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from dgxmm.api.errors import api_error
from dgxmm.api.integrations import key_allows, key_out
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory
from dgxmm.litellm import LiteLLMError
from dgxmm.planner.plans import PlanAction

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


class PreviewIn(BaseModel):
    engine: str
    model_ref: str
    config: dict = {}
    replacing: int | None = None


class DeployIn(DeployRequest):
    actions: list[PlanAction] = []


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


class DeploymentPatch(BaseModel):
    alias: str | None = None  # LiteLLM model name; null or "" = use the served name


@router.patch("/{dep_id}")
async def patch_deployment(dep_id: int, body: DeploymentPatch, p: Principal = Depends(require_admin),
                           ctx: AppContext = Depends(get_ctx)):
    if "alias" not in body.model_fields_set:
        raise api_error(422, "nothing_to_update", "send {\"alias\": ...}")
    try:
        before = ctx.deployer.view(dep_id)
        view = ctx.deployer.set_alias(dep_id, body.alias)
    except DeployError as e:
        raise deploy_error(e)
    if before["alias"] != view["alias"]:
        audit(ctx.db, p.user.id, "deployment_alias", view["name"], {"from": before["alias"], "to": view["alias"]})
        if ctx.litellm is not None:
            ctx.litellm.trigger()  # re-register under the new name, drop the old one
    return view


@router.get("/{dep_id}/litellm")
async def deployment_litellm(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        view = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    cfg = ctx.settings_store.litellm()
    model_name = view["model_name"]
    out = {"deployment_id": dep_id, "model_name": model_name, "endpoint": cfg.endpoint, "registered": None,
           "status": view["litellm_status"], "enabled": cfg.enabled, "error": None, "keys": [],
           "keys_visible": p.user.role == "admin", "keys_error": None}
    client = ctx.litellm.client if ctx.litellm is not None else None
    if client is None or not cfg.enabled:
        err = ("litellm_disabled", "the LiteLLM integration is disabled") if not cfg.enabled else \
            ("litellm_unavailable", "the LiteLLM integration is not initialised")
        out["error"] = {"code": err[0], "message": err[1]}
        return out
    try:
        out["registered"] = await ctx.litellm.is_registered(dep_id, model_name) if model_name else False
    except LiteLLMError as e:
        out["error"] = {"code": e.code, "message": e.message}
    if out["keys_visible"]:
        try:
            out["keys"] = [key_out(k) for k in await client.key_list() if model_name and key_allows(k, model_name)]
        except LiteLLMError as e:
            out["keys_error"] = {"code": e.code, "message": e.message}
    return out


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
    actions: list[PlanAction] = []


class ReconfigureIn(BaseModel):
    config: dict
    model_ref: str | None = None
    actions: list[PlanAction] = []


async def _run(ctx, p, action: str, target: str, coro, detail: dict | None = None):
    try:
        view = await coro
    except InsufficientMemory as e:
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict())
    except DeployError as e:
        if action in ("switch", "reconfigure"):
            audit(ctx.db, p.user.id, f"{action}_failed", target, {**(detail or {}), "code": e.code})
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
    try:
        view = await ctx.deployer.switch(dep_id, body.request, body.actions, p.user.id)
    except InsufficientMemory as e:
        try:
            plans = (await ctx.deployer.preview(body.request.engine, body.request.model_ref, body.request.config, replacing=dep_id))["plans"]
        except DeployError:
            plans = []
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict(), plans=plans)
    except DeployError as e:
        audit(ctx.db, p.user.id, "switch_failed", f"{dep_id}->{body.request.name}",
              {"model": body.request.model_ref, "actions": body.actions, "code": e.code})
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "switch", f"{dep_id}->{body.request.name}", {"model": body.request.model_ref, "actions": body.actions})
    return view


@router.post("/{dep_id}/reconfigure")
async def reconfigure(dep_id: int, body: ReconfigureIn, p: Principal = Depends(require_admin),
                      ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "reconfigure", str(dep_id),
                      ctx.deployer.reconfigure(dep_id, body.config, body.model_ref, body.actions, p.user.id),
                      {"config": body.config, "model_ref": body.model_ref, "actions": body.actions})
