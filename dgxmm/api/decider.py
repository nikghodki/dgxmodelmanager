"""/api/decider — the local decision model: status, settings, enable/disable (normal fit/plan flow), image build,
decision log and feedback. The decider's API key is never returned, audited or logged."""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

from dgxmm.api.deployments import deploy_error
from dgxmm.api.errors import api_error
from dgxmm.api.ratelimit import RATE_LIMITS, limited  # noqa: F401  (RATE_LIMITS re-exported for tests/docs)
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider.common import DECIDER_RUNNING, decider_view as _existing
from dgxmm.decider.rotation import RotationInProgress, in_progress, rotate_decider_key
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory
from dgxmm.engines.decision import DECIDER_MODELS, DEFAULT_MODEL
from dgxmm.planner.plans import PlanAction

router = APIRouter(prefix="/api/decider", tags=["decider"])
DECIDER_ENGINE = "decision"
DECIDER_NAME = "decider"
DECIDER_PRIORITY = 100
RUNNING_STATES = DECIDER_RUNNING
MAX_REQUEST_BYTES = 64 * 1024  # every /api/decider body (enforced in app.BodyLimit before parsing)
FeatureName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z_]{0,31}$")]


class FeatureIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class SettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[FeatureName, FeatureIn] = Field(default_factory=dict, max_length=16)
    model: str | None = Field(None, max_length=128)  # one of DECIDER_MODELS; applied by the next enable


class ModelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(max_length=128)


class EnableIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[PlanAction] = Field(default_factory=list, max_length=16)


class DisableIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    remove: StrictBool = False


class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: StrictBool


async def image_status(ctx: AppContext) -> dict:
    """Image status; the docker call runs in a worker thread (never blocks the event loop)."""
    img = ctx.decider_image
    present = await asyncio.to_thread(img.present)
    return {"image": img.tag, "present": present, "building": img.building, "error": img.error,
            "last_build": img.last_build}


def models_out(ctx: AppContext) -> list[dict]:
    engine = (ctx.engines or {}).get(DECIDER_ENGINE)
    supported = ctx.decider_image.supported_models()
    out = []
    for repo, m in DECIDER_MODELS.items():
        art = ctx.inventory.get(repo) if ctx.inventory is not None else None
        out.append({"repo": repo, "revision": m.revision, "served_name": m.served_name, "params": m.params,
                    "reservation_bytes": m.reservation_bytes, "default": repo == DEFAULT_MODEL,
                    "downloaded": bool(art is not None and art.complete and engine is not None
                                       and engine.supports(art)),
                    "image_supports": None if supported is None else f"{repo}@{m.revision}" in supported})
    return out


async def status_out(ctx: AppContext) -> dict:
    from dgxmm.decider import upgrade
    d = ctx.decider
    view = d.deployment()
    img = await image_status(ctx)
    models = await asyncio.to_thread(models_out, ctx)
    running = view is not None and view["state"] in RUNNING_STATES
    client = d.client()
    breaker = client.breaker if client is not None else None
    up = await asyncio.to_thread(upgrade.check, ctx)
    return {"upgrade": up, "enabled": d.enabled, "healthy": d.healthy(), "deployment_id": view["id"] if view else None,
            "state": view["state"] if view else None, "model": d.model, "revision": d.revision,
            "configured_model": d.configured_model,
            "restart_required": bool(running and view["model_ref"] != d.configured_model),
            "models": models, **d.stats(), "features": d.features(),
            "breaker": {"open": breaker.is_open(), "failures": breaker.failures,
                        "retry_in_s": round(breaker.retry_in(), 1)} if breaker is not None else None,
            "load": d.load() if hasattr(d, "load") else None,
            "supervisor": d.supervisor.status() if getattr(d, "supervisor", None) is not None else None,
            "image": {"present": img["present"], "building": img["building"], "error": img["error"]}}


@router.get("/status")
async def status(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return await status_out(ctx)


@router.put("/settings")
async def put_settings(body: SettingsIn, p: Principal = Depends(limited("decider_admin")),
                       ctx: AppContext = Depends(get_ctx)):
    known = set(ctx.decider.features())
    unknown = sorted(set(body.features) - known)
    if unknown:
        raise api_error(422, "unknown_feature", f"unknown feature(s): {', '.join(unknown)}", known=sorted(known))
    if body.model is not None and body.model not in DECIDER_MODELS:
        raise api_error(422, "unknown_model", f"{body.model} is not a supported decision model",
                        known=sorted(DECIDER_MODELS))
    for name, f in body.features.items():
        ctx.decider.set_feature(name, enabled=f.enabled, threshold=f.threshold)
    if body.model is not None:
        ctx.decider.set_model(body.model)
    audit(ctx.db, p.user.id, "decider_settings", "decider",
          {"features": {k: v.model_dump(exclude_none=True) for k, v in body.features.items()},
           **({"model": body.model} if body.model is not None else {})})
    return await status_out(ctx)


@router.post("/enable", status_code=202)
async def enable(body: EnableIn, p: Principal = Depends(limited("decider_admin")), ctx: AppContext = Depends(get_ctx)):
    if in_progress(ctx):
        raise api_error(409, "rotation_in_progress", "the decider's API key is being rotated; it restarts by itself")
    if await asyncio.to_thread(ctx.decider_image.present) is False:
        audit(ctx.db, p.user.id, "decider_enable_failed", "decider", {"code": "image_missing"})
        raise api_error(409, "image_missing", f"the decider image {ctx.decider_image.tag} is not built yet",
                        build_url=f"{ctx.settings.base_path}/api/decider/image/build", image=ctx.decider_image.tag)
    existing = _existing(ctx)
    model = ctx.decider.configured_model
    if await asyncio.to_thread(ctx.decider_image.supports, model) is False:
        audit(ctx.db, p.user.id, "decider_enable_failed", "decider", {"code": "image_outdated", "model": model})
        raise api_error(409, "image_outdated", f"the decider image {ctx.decider_image.tag} was built before {model} "
                                               "was supported; rebuild it",
                        build_url=f"{ctx.settings.base_path}/api/decider/image/build", image=ctx.decider_image.tag)
    applied: list = []
    switched_from: str | None = None
    try:
        if existing is not None and existing["state"] in RUNNING_STATES and existing["model_ref"] == model:
            view = existing
        else:
            if body.actions:
                applied = await ctx.deployer.apply_actions(body.actions, p.user.id)
            if existing is not None:
                if existing["state"] in RUNNING_STATES:  # another model is running: replace it (same deployment)
                    switched_from = existing["model_ref"]
                    await ctx.deployer.stop(existing["id"], p.user.id, reason=f"decider model change to {model}")
                view = await ctx.deployer.restart(existing["id"], p.user.id, model_ref=model)
            else:
                view = await ctx.deployer.deploy(DeployRequest(name=DECIDER_NAME, engine=DECIDER_ENGINE,
                                                               model_ref=model, config={},
                                                               priority=DECIDER_PRIORITY), p.user.id)
    except (InsufficientMemory, DeployError) as e:
        await ctx.deployer.undo_actions(applied)
        audit(ctx.db, p.user.id, "decider_enable_failed", "decider", {"code": e.code})
        if switched_from is not None:  # the new model could not start: bring the previous one back
            try:
                await ctx.deployer.restart(existing["id"], p.user.id, model_ref=switched_from)
            except DeployError:
                pass
        if not isinstance(e, InsufficientMemory):
            raise deploy_error(e)
        try:
            plans = (await ctx.deployer.preview(DECIDER_ENGINE, model, existing["config"] if existing else {},
                                                replacing=existing["id"] if switched_from else None))["plans"]
        except DeployError:
            plans = []
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict(), plans=plans)
    ctx.decider.set_deployment_id(view["id"])
    ctx.decider.set_enabled(True)
    if hasattr(ctx.decider, "set_switch_from"):
        ctx.decider.set_switch_from(switched_from)  # the supervisor rolls back if the new model fails to start
    if getattr(ctx.decider, "supervisor", None) is not None:
        ctx.decider.supervisor.reset()  # an admin (re-)enabled it: fresh restart budget and backoff
    audit(ctx.db, p.user.id, "decider_enable", view["name"],
          {"deployment_id": view["id"], "model": model, "actions": [a.model_dump() for a in body.actions],
           **({"switched_from": switched_from} if switched_from else {}), **p.audit_extra()})
    return {"deployment_id": view["id"], "state": view["state"]}


@router.post("/disable")
async def disable(body: DisableIn = DisableIn(), p: Principal = Depends(limited("decider_admin")),
                  ctx: AppContext = Depends(get_ctx)):
    ctx.decider.set_enabled(False)
    existing = _existing(ctx)
    state = None
    try:
        if existing is not None:
            state = existing["state"]
            if state != "stopped":
                state = (await ctx.deployer.stop(existing["id"], p.user.id, reason="decider disabled"))["state"]
            if body.remove:
                await ctx.deployer.remove(existing["id"])
                ctx.decider.set_deployment_id(None)
                state = None
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "decider_disable", existing["name"] if existing else "decider",
          {"deployment_id": existing["id"] if existing else None, "removed": body.remove, **p.audit_extra()})
    return {"enabled": False, "deployment_id": None if body.remove or existing is None else existing["id"],
            "state": state}


@router.post("/models/download", status_code=202)
async def download_model(body: ModelIn, p: Principal = Depends(limited("decider_admin")),
                         ctx: AppContext = Depends(get_ctx)):
    """Download a supported decision model at its pinned revision with the verified snapshot downloader
    (dgxmm/decider/snapshot.py via DownloadService.start_pinned): resumable, every file hashed before it is linked.
    Progress is a normal download (Downloads page, SSE ``download`` events); a complete snapshot is only verified."""
    m = DECIDER_MODELS.get(body.model)
    if m is None:
        raise api_error(422, "unknown_model", f"{body.model} is not a supported decision model",
                        known=sorted(DECIDER_MODELS))
    if ctx.downloads is None:
        raise api_error(503, "downloads_unavailable", "downloads are not available")
    from dgxmm.downloads import DownloadError
    try:
        view = await asyncio.to_thread(ctx.downloads.start_pinned, m.repo, m.revision, p.user.id)
    except DownloadError as e:
        raise api_error(e.status, e.code, e.message)
    await ctx.downloads.launch(view["id"])
    audit(ctx.db, p.user.id, "decider_model_download", m.repo, {"revision": m.revision, **p.audit_extra()})
    return view


@router.get("/image")
async def image(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return await image_status(ctx)


@router.post("/image/build", status_code=202)
async def build_image(p: Principal = Depends(limited("decider_admin")), ctx: AppContext = Depends(get_ctx)):
    if ctx.docker is None:
        raise api_error(503, "docker_unavailable", "docker is not available")
    if not ctx.decider_image.start_build():
        raise api_error(409, "build_in_progress", "the decider image is already being built")
    audit(ctx.db, p.user.id, "decider_image_build", ctx.decider_image.tag, p.audit_extra())
    return {"building": True, "image": ctx.decider_image.tag}


@router.get("/decisions")
async def decisions(feature: str | None = Query(None, max_length=32), limit: int = Query(100, ge=1, le=1000),
                    p: Principal = Depends(limited("decider_read")), ctx: AppContext = Depends(get_ctx)):
    return ctx.decider.recent(feature=feature, limit=limit)


@router.post("/decisions/{decision_id}/feedback")
async def feedback(decision_id: int, body: FeedbackIn, p: Principal = Depends(limited("decider_feedback")),
                   ctx: AppContext = Depends(get_ctx)):
    row = ctx.decider.feedback(decision_id, body.accepted, p.user.id)
    if row is None:
        raise api_error(404, "not_found", f"decision {decision_id} not found")
    audit(ctx.db, p.user.id, "decider_feedback", str(decision_id), {"accepted": body.accepted,
                                                                     "feature": row["feature"]})
    return row


@router.post("/rotate-key")
async def rotate_key(p: Principal = Depends(limited("decider_admin")), ctx: AppContext = Depends(get_ctx)):
    """New bearer key; a running decider is stopped and restarted with it (features fall back to rules meanwhile).
    The key itself is never returned."""
    try:
        return await rotate_decider_key(ctx, p.user.id)
    except RotationInProgress:
        raise api_error(409, "rotation_in_progress", "a key rotation is already running")
