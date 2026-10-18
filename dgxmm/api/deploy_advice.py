"""HTTP endpoints of the deploy-time decision-model features (spec 2026-10-17 features 2, 3, 4, 5 and 11):
config advice, the pre-download compatibility check, draft pairing, engine choice and remote-code risk.

Every route is role-checked and rate-limited from the feature registry (``limited(feature_bucket(...))``); POST
bodies are capped by app.BodyLimit (``/api/decider``). Answers carry ``source: rules|model|none`` and never fail
because of the decision model."""
import re

from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from dgxmm.api.errors import api_error
from dgxmm.api.ratelimit import limited
from dgxmm.auth.deps import Principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider import registry
from dgxmm.inventory.artifact import REPO_RE

router = APIRouter(prefix="/api", tags=["deploy-advice"])


class ConfigAdviceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ref: str = Field(min_length=1, max_length=1024)
    engine: str = Field(pattern=r"^[a-z]{1,16}$")
    goal: Literal["chat", "agent_tool_use", "batch_throughput", "long_context"]


class DraftIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_ref: str = Field(min_length=1, max_length=1024)
    method: Literal["draft_model", "eagle", "eagle3", "dflash", "mtp", "ngram"]


class EngineIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ref: str = Field(min_length=1, max_length=1024)
    goal: Literal["chat", "agent_tool_use", "batch_throughput", "long_context"]


class RemoteCodeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ref: str = Field(min_length=1, max_length=1024)


def _engine_ok(ctx: AppContext, engine: str) -> None:
    if engine not in (ctx.engines or {}):
        raise api_error(422, "unknown_engine", f"engine {engine!r} is not available")


def _artifact(ctx: AppContext, ref: str):
    a = ctx.inventory.get(ref) if ctx.inventory is not None else None
    if a is None:
        raise api_error(404, "model_not_found", f"{ref} is not downloaded")
    return a


def _repo_ok(repo: str) -> bool:
    return bool(re.fullmatch(REPO_RE, repo)) and ".." not in repo and len(repo) <= 256


@router.get("/catalog/compat")
async def compat(repo: str = Query(..., max_length=256),
                 p: Principal = Depends(limited(registry.feature_bucket("compat"), admin=False)),
                 ctx: AppContext = Depends(get_ctx)):
    from dgxmm.decider.compat import check
    if not _repo_ok(repo):
        raise api_error(422, "invalid_repo", "repo id must look like org/name")
    # viewers' checks go out without the admin's HF token (they cannot probe gated-repo access with it)
    return await check(ctx, repo, anonymous=p.user.role != "admin")


@router.post("/decider/config-advice")
async def config_advice(body: ConfigAdviceIn, p: Principal = Depends(limited(registry.feature_bucket("config"))),
                        ctx: AppContext = Depends(get_ctx)):
    from dgxmm.decider.config import advise
    _engine_ok(ctx, body.engine)
    _artifact(ctx, body.model_ref)
    try:
        return await advise(ctx, body.model_ref, body.engine, body.goal)
    except KeyError:
        raise api_error(404, "model_not_found", f"{body.model_ref} is not downloaded")


@router.post("/decider/draft-suggest")
async def draft_suggest(body: DraftIn, p: Principal = Depends(limited(registry.feature_bucket("draft"))),
                        ctx: AppContext = Depends(get_ctx)):
    from dgxmm.decider.draft import suggest
    _artifact(ctx, body.target_ref)
    try:
        return await suggest(ctx, body.target_ref, body.method)
    except KeyError:
        raise api_error(404, "model_not_found", f"{body.target_ref} is not downloaded")


@router.post("/decider/engine-suggest")
async def engine_suggest(body: EngineIn, p: Principal = Depends(limited(registry.feature_bucket("engine"))),
                         ctx: AppContext = Depends(get_ctx)):
    from dgxmm.decider.engine import suggest
    _artifact(ctx, body.model_ref)
    try:
        return await suggest(ctx, body.model_ref, body.goal)
    except KeyError:
        raise api_error(404, "model_not_found", f"{body.model_ref} is not downloaded")


@router.post("/decider/remote-code-check")
async def remote_code_check(body: RemoteCodeIn,
                            p: Principal = Depends(limited(registry.feature_bucket("remote_code"))),
                            ctx: AppContext = Depends(get_ctx)):
    from dgxmm.decider.remote_code import check
    _artifact(ctx, body.model_ref)
    try:
        return await check(ctx, body.model_ref)
    except KeyError:
        raise api_error(404, "model_not_found", f"{body.model_ref} is not downloaded")
