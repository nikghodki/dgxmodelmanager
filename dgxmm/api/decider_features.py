"""HTTP endpoints of the decision-model features: diagnosis, workload, parser suggestions (contract.md)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from dgxmm.api.deployments import deploy_error
from dgxmm.api.errors import api_error
from dgxmm.api.ratelimit import limited
from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider.features import ensure_features
from dgxmm.deployer.service import DeployError

router = APIRouter(prefix="/api", tags=["decider-features"])


class SuggestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ref: str = Field(min_length=1, max_length=1024)
    engine: str = Field(pattern=r"^[a-z]{1,16}$")


@router.get("/deployments/{dep_id}/diagnosis")
def diagnosis(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    ensure_features(ctx)
    return {"diagnosis": ctx.diagnosis.latest(dep_id)}


@router.get("/deployments/{dep_id}/workload")
async def workload(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    ensure_features(ctx)
    return await ctx.workload.get(dep_id)


@router.post("/decider/suggest-config")
async def suggest_config(body: SuggestIn, p: Principal = Depends(limited("decider_suggest")),
                         ctx: AppContext = Depends(get_ctx)):
    from dgxmm.decider.config_suggest import suggest_for_artifact
    if body.engine not in (ctx.engines or {}):
        raise api_error(422, "unknown_engine", f"engine {body.engine!r} is not available")
    artifact = ctx.inventory.get(body.model_ref) if ctx.inventory is not None else None
    if artifact is None:
        raise api_error(404, "model_not_found", f"{body.model_ref} is not downloaded")
    return await suggest_for_artifact(ctx, artifact, body.engine)
