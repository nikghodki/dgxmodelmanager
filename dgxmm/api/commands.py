"""Natural-language ⌘K commands (decision-model feature 6, dgxmm/decider/command.py).

- ``POST /api/decider/command {text}`` (viewer; at most 300 characters; rate-limited; body cap of /api/decider):
  interpretations only — the palette opens the normal confirmation or dialog for the one the user picks.
- ``GET /api/decider/free-plans?gib=`` (admin): the plans that would free that much memory (sleep / stop sets, ranked
  and annotated by the plan safety filter; the request priority is the maximum — the user asked for it — so only
  traffic, dependencies, start-ups and system deployments make a plan unsafe) for the "free N GiB" plan chooser.
- ``POST /api/decider/free-memory {gib, plan_id, acknowledge_unsafe}`` (admin, audited): applies the chosen plan
  through ``Deployer.apply_actions`` (as a deploy's plan: undone if any action fails); an unsafe plan needs
  ``acknowledge_unsafe`` = the ``unsafe_fingerprint`` of the reason the admin was shown (the plan id and the reason
  with its numbers masked, so "2 min ago" -> "3 min ago" still matches but a new reason does not).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dgxmm.api.errors import api_error
from dgxmm.api.ratelimit import limited
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider import registry
from dgxmm.decider.command import MAX_GIB, MAX_TEXT, CommandService, Result

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/decider", tags=["decider-command"])
BUCKET = registry.feature_bucket("command")
DEADLINE_S = 8.0  # the whole interpretation; beyond it: rules only
GIB = 1024**3
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def unsafe_fingerprint(plan_id: str, reason: str | None) -> str | None:
    """What an acknowledgement of an unsafe plan is bound to: the plan and its reason, numbers masked."""
    if not reason:
        return None
    return hashlib.sha256(f"{plan_id}|{_NUM.sub('#', reason)}".encode()).hexdigest()[:16]


class CommandIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_TEXT)


@router.post("/command")
async def command(body: CommandIn, p: Principal = Depends(limited(BUCKET, admin=False)),
                  ctx: AppContext = Depends(get_ctx)):
    admin = p.user.role == "admin"
    svc = CommandService(ctx)
    targets = await asyncio.to_thread(svc.snapshot, admin=admin)
    box: dict = {}
    try:
        async with asyncio.timeout(DEADLINE_S):
            res = await svc.interpret(body.text, admin=admin, targets=targets, fallback=box)
    except TimeoutError:
        # the rules' result was computed before the model was asked: answer with it, never start over
        log.warning("command interpretation exceeded %ss; answered by rules", DEADLINE_S)
        res = box.get("rules") or Result(message="Couldn't interpret that in time. Try again, or use the palette's "
                                                 "search.")
    return res.out()


async def _plans(ctx: AppContext, gib: float) -> dict:
    from dgxmm.decider.plans import unsafe_reason
    from dgxmm.planner.plans import eviction_plans
    want = int(gib * GIB)
    budget = await ctx.budget.refresh()
    shortfall = max(want - budget.free_for_new, 0)
    plans: list[dict] = []
    if shortfall > 0:
        facts = await ctx.deployer.plan_facts() or {}
        active = await asyncio.to_thread(ctx.deployer.active_deps, None, facts)
        for opt in eviction_plans(active, shortfall, want):
            d = opt.as_dict()
            # the user asked to free memory: priorities do not block (request priority = the maximum)
            d["unsafe_reason"] = unsafe_reason(d, facts, raw_cfg={}, defaults={}, context_key="", seqs_key="",
                                               request_priority=100) if facts else "traffic unknown"
            d["unsafe_fingerprint"] = unsafe_fingerprint(d["id"], d["unsafe_reason"])
            plans.append(d)
    return {"bytes": want, "free_for_new": budget.free_for_new, "shortfall": shortfall, "plans": plans}


@router.get("/free-plans")
async def free_plans(gib: float = Query(gt=0, le=MAX_GIB), p: Principal = Depends(limited(BUCKET)),
                     ctx: AppContext = Depends(get_ctx)):
    return await _plans(ctx, gib)


class FreeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gib: float = Field(gt=0, le=MAX_GIB)
    plan_id: str = Field(min_length=1, max_length=256)
    acknowledge_unsafe: StrictStr | None = Field(None, max_length=64)  # the unsafe_fingerprint the admin saw


@router.post("/free-memory")
async def free_memory(body: FreeIn, p: Principal = Depends(limited(BUCKET)), ctx: AppContext = Depends(get_ctx)):
    """Apply one of the free-memory plans, recomputed now, through the same path as a deploy's plan
    (``Deployer.apply_actions``: system deployments refused, everything undone if one action fails). A plan the
    safety filter flags needs ``acknowledge_unsafe`` equal to its current ``unsafe_fingerprint``."""
    from dgxmm.api.deployments import deploy_error
    from dgxmm.deployer.service import DeployError
    from dgxmm.planner.plans import PlanAction
    cur = await _plans(ctx, body.gib)
    plan = next((x for x in cur["plans"] if x["id"] == body.plan_id), None)
    if plan is None:
        raise api_error(409, "plan_gone", "that plan is no longer offered; choose again", plans=cur["plans"])
    if plan["unsafe_reason"] and body.acknowledge_unsafe != plan["unsafe_fingerprint"]:
        raise api_error(409, "plan_unsafe", f"this plan is not safe: {plan['unsafe_reason']}", plan=plan)
    actions = [PlanAction.model_validate(a) for a in plan["actions"]]
    try:
        applied = await ctx.deployer.apply_actions(actions, p.user.id)
    except DeployError as e:
        await asyncio.to_thread(audit, ctx.db, p.user.id, "free_memory_failed", plan["id"],
                                {"code": e.code, **p.audit_extra()})
        raise deploy_error(e)
    await asyncio.to_thread(audit, ctx.db, p.user.id, "free_memory", plan["id"],
                            {"actions": [a.model_dump() for a in actions], "unsafe": plan["unsafe_reason"],
                             **p.audit_extra()})
    return {"applied": [{"type": t, "deployment_id": i} for t, i in applied], "plan": plan}
