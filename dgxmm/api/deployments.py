import asyncio

from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel, Field, StrictBool

from dgxmm.api.errors import api_error
from dgxmm.api.integrations import key_allows, key_out
from dgxmm.api.ratelimit import allow
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory
from dgxmm.litellm import LiteLLMError
from dgxmm.memory.redact import redact_log
from dgxmm.planner.plans import PlanAction

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


class PreviewIn(BaseModel):
    engine: str
    model_ref: str
    config: dict = {}
    replacing: int | None = None
    priority: int | None = Field(None, ge=0, le=100)  # the request's priority (plan safety); None = replaced's / 0


class DeployIn(DeployRequest):
    actions: list[PlanAction] = []


def deploy_error(e: DeployError):
    extra = {"fit": e.fit.as_dict()} if isinstance(e, InsufficientMemory) else dict(getattr(e, "extra", {}) or {})
    return api_error(e.status, e.code, e.message, **extra)


def plan_consult(ctx: AppContext, p: Principal):
    """May this request ask the decision model to rank plans? (rate limit `decider_plan`; over it: rules only)"""
    return lambda: allow(ctx, "decider_plan", p.user.id)


@router.post("/preview")
async def preview(body: PreviewIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    """Admin-only like every deployment change (the deploy wizard and the switch / reconfigure dialogs, its only
    callers, are admin-only): it may ask the decision model and writes `decisions` rows."""
    try:
        return await ctx.deployer.preview(body.engine, body.model_ref, body.config, replacing=body.replacing,
                                          priority=body.priority, consult=plan_consult(ctx, p))
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
            pv = await ctx.deployer.preview(req.engine, req.model_ref, req.config, priority=req.priority,
                                            consult=plan_consult(ctx, p))
        except DeployError:
            pv = {}
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict(), plans=pv.get("plans", []),
                        notes=pv.get("notes", []))
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
    rewrite_keys: StrictBool = False  # also replace the old model name in keys scoped to it


def _renamed(models: list[str], old: str, new: str) -> list[str]:
    out: list[str] = []
    for m in models:
        m = new if m == old else m
        if m not in out:  # a key that already listed the new name must not get it twice
            out.append(m)
    return out


async def _rename_in_keys(ctx: AppContext, old: str, new: str, rewrite: bool) -> tuple[list[dict], int, dict | None]:
    """Keys whose model list names ``old`` explicitly (unrestricted keys are unaffected by a rename), each with
    ``rewritten``; with ``rewrite`` their model lists are updated. A failure stops the loop but keeps the keys
    already rewritten in the result, so the response and the audit row say exactly what changed in LiteLLM."""
    cfg = ctx.settings_store.litellm()
    if ctx.litellm is None or ctx.litellm.client is None or not cfg.active:
        return [], 0, {"code": "litellm_disabled" if cfg.master_key else "litellm_not_configured",
                       "message": "LiteLLM keys are not available"}
    client = ctx.litellm.client
    try:
        affected = [k for k in await client.key_list() if old in (k.get("models") or [])]
    except LiteLLMError as e:
        return [], 0, {"code": e.code, "message": e.message}
    out = [{"token": k.get("token"), "key_alias": k.get("key_alias"), "models": list(k.get("models") or []),
            "rewritten": False} for k in affected]
    error = None
    if rewrite:
        for item in out:
            try:
                await client.key_update(item["token"], {"models": _renamed(item["models"], old, new)})
            except LiteLLMError as e:
                error = {"code": e.code, "message": e.message}
                break
            item["rewritten"] = True
    return out, sum(i["rewritten"] for i in out), error


@router.patch("/{dep_id}")
async def patch_deployment(dep_id: int, body: DeploymentPatch, p: Principal = Depends(require_admin),
                           ctx: AppContext = Depends(get_ctx)):
    if "alias" not in body.model_fields_set:
        raise api_error(422, "nothing_to_update", "send {\"alias\": ...}")
    try:
        before = ctx.deployer.view(dep_id)
        view = await ctx.deployer.rename(dep_id, body.alias)
    except DeployError as e:
        raise deploy_error(e)
    affected, rewritten, keys_error = [], 0, None
    if before["model_name"] != view["model_name"] and before["model_name"] and view["model_name"]:
        affected, rewritten, keys_error = await _rename_in_keys(ctx, before["model_name"], view["model_name"],
                                                                body.rewrite_keys)
    if before["alias"] != view["alias"]:
        audit(ctx.db, p.user.id, "deployment_alias", view["name"],
              {"from": before["alias"], "to": view["alias"], "keys_rewritten": rewritten,
               "rewritten_keys": [k["key_alias"] or (k["token"] or "")[:12] for k in affected if k["rewritten"]]})
        if ctx.litellm is not None:
            ctx.litellm.trigger()  # re-register under the new name, drop the old one
    return {**view, "affected_keys": affected, "keys_rewritten": rewritten, "keys_error": keys_error}


KEY_CACHE_S = 15.0  # per-deployment key listing may be this stale (API key writes invalidate it)


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
    sync = ctx.litellm
    if sync is None or sync.client is None or not cfg.enabled:
        err = ("litellm_disabled", "the LiteLLM integration is disabled") if not cfg.enabled else \
            ("litellm_unavailable", "the LiteLLM integration is not initialised")
        out["error"] = {"code": err[0], "message": err[1]}
        return out
    # Registration comes from the sync loop's last result: no LiteLLM call on this (viewer-readable) path.
    out["registered"] = sync.is_registered(dep_id, model_name) if model_name else False
    if dep_id in sync.conflicts:
        out["error"] = {"code": "model_name_conflict", "message": sync.conflicts[dep_id]}
    elif sync.last["ok"] is False:
        out["error"] = {"code": sync.last["code"] or "litellm_error", "message": sync.last["error"]}
    elif not cfg.master_key:
        out["error"] = {"code": "litellm_not_configured", "message": cfg.master_key_hint or "no master key is set"}
    if out["keys_visible"]:
        try:
            keys = await sync.client.key_list(max_age=KEY_CACHE_S)
            out["keys"] = [key_out(k) for k in keys if model_name and key_allows(k, model_name)]
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
        if ctx.deployer.view(dep_id)["system"] and not ctx.decider.enabled:
            # the decision model loads ~10 GiB that no feature would use: it starts only by enabling it
            raise DeployError("decider_disabled", "the decision model is disabled; enable it on the Decision model "
                                                  "page to start it", 409)
        view = await ctx.deployer.restart(dep_id, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "start", view["name"])
    return view


@router.delete("/{dep_id}", status_code=204)
async def remove(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    """Removing the System decider also disables it (Decider.on_deployment_removed clears its deployment id)."""
    was_enabled = None
    try:
        view = ctx.deployer.view(dep_id)
        if view["system"] and view["state"] in ("stopped", "failed"):
            was_enabled = ctx.decider.enabled
            ctx.decider.set_enabled(False)  # first: the supervisor must not restart it while it is being removed
        await ctx.deployer.remove(dep_id)
    except DeployError as e:
        if was_enabled:
            ctx.decider.set_enabled(True)
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "remove", view["name"])
    if view["system"]:
        audit(ctx.db, p.user.id, "decider_disable", view["name"],
              {"deployment_id": dep_id, "removed": True, **p.audit_extra()})
    return Response(status_code=204)


@router.get("/{dep_id}/logs")
async def logs(dep_id: int, tail: int = Query(200, ge=1, le=5000), p: Principal = Depends(require_admin),
               ctx: AppContext = Depends(get_ctx)):
    try:
        view = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    def read() -> str:  # engine output can echo tokens, auth headers, URL credentials: mask them line by line
        return redact_log(ctx.docker.logs(view["container_id"], tail))
    return {"logs": await asyncio.to_thread(read)}


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
    view = await _run(ctx, p, "wake", str(dep_id), ctx.deployer.wake(dep_id, p.user.id))
    if view.get("state") == "sleeping" and view.get("startup_queue"):  # waiting for its turn: continues async
        return JSONResponse(view, status_code=202)
    return view


@router.post("/{dep_id}/switch")
async def switch(dep_id: int, body: SwitchIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await ctx.deployer.switch(dep_id, body.request, body.actions, p.user.id)
    except InsufficientMemory as e:
        try:
            pv = await ctx.deployer.preview(body.request.engine, body.request.model_ref, body.request.config,
                                            replacing=dep_id)
        except DeployError:
            pv = {}
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict(), plans=pv.get("plans", []),
                        notes=pv.get("notes", []))
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
