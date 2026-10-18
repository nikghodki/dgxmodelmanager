"""Sleep/wake schedules of a deployment (decision-model feature 7, dgxmm/decider/schedule.py).

``GET /api/deployments/{id}/schedules`` (viewer) and, admin only (CSRF, audited): ``POST`` the same path (one schedule
or a sleep/wake pair, created atomically), ``PUT /api/schedules/{id}`` (edit, pause, resume) and ``DELETE``. A
proposal insight's "Review schedule" dialog creates its pair through POST; nothing here runs a deployment action —
the schedule runner does, at the schedule's time, through the deployer's normal paths."""
from __future__ import annotations

import asyncio
import re
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from dgxmm.api.deployments import deploy_error
from dgxmm.api.errors import api_error
from dgxmm.api.ratelimit import limited
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider import registry
from dgxmm.decider.schedule import (ScheduleError, ScheduleIn, ScheduleService, allowed_actions, box_zone,
                                    describe_days, zone_name)
from dgxmm.deployer.service import DeployError

router = APIRouter(prefix="/api", tags=["schedules"])
MAX_REQUEST_BYTES = 4 * 1024  # every schedules body (app.BodyLimit, before parsing)
BODY_PATHS = (re.compile(r"/api/deployments/\d+/schedules/?$"), "/api/schedules")
BUCKET = registry.feature_bucket("schedule")
Action = Literal["sleep", "wake", "stop", "start"]
HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"


class ScheduleBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Action
    days: StrictInt = Field(ge=1, le=127)  # bit 0 = Monday .. bit 6 = Sunday
    at_local: str = Field(pattern=HHMM)
    timezone: str = Field(min_length=1, max_length=64)
    enabled: StrictBool = True


class CreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schedules: list[ScheduleBody] = Field(min_length=1, max_length=2)


class PauseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused_all: StrictBool


class PatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Action | None = None
    days: StrictInt | None = Field(None, ge=1, le=127)
    at_local: str | None = Field(None, pattern=HHMM)
    timezone: str | None = Field(None, min_length=1, max_length=64)
    enabled: StrictBool | None = None


def service(ctx: AppContext) -> ScheduleService:
    if ctx.schedules is None:
        ctx.schedules = ScheduleService(ctx)
    return ctx.schedules


def _view(ctx: AppContext, dep_id: int) -> dict:
    try:
        return ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)


def _error(e: ScheduleError):
    return api_error(e.status, e.code, e.message)


def _detail(s: dict) -> dict:
    return {"schedule_id": s["id"], "deployment_id": s["deployment_id"], "action": s["action"],
            "days": describe_days(s["days"]), "at_local": s["at_local"], "timezone": s["timezone"],
            "enabled": s["enabled"]}


@router.get("/deployments/{dep_id}/schedules")
async def list_schedules(dep_id: int, p: Principal = Depends(limited(BUCKET, admin=False)),
                         ctx: AppContext = Depends(get_ctx)):
    view = await asyncio.to_thread(_view, ctx, dep_id)
    rows = await asyncio.to_thread(service(ctx).list, dep_id)
    if p.user.role != "admin":  # who created a schedule (a user id) is admin information
        rows = [{k: v for k, v in r.items() if k != "created_by"} for r in rows]
    return {"deployment_id": dep_id, "schedules": rows, "allowed_actions": list(allowed_actions(view)),
            "timezone": zone_name(box_zone(ctx)), "paused_all": await asyncio.to_thread(service(ctx).paused_all)}


# Declared before /schedules/{sid}: "settings" is not a schedule id.
@router.get("/schedules/settings")
async def schedule_settings(p: Principal = Depends(limited(BUCKET, admin=False)), ctx: AppContext = Depends(get_ctx)):
    return {"paused_all": await asyncio.to_thread(service(ctx).paused_all), "timezone": zone_name(box_zone(ctx))}


@router.put("/schedules/settings")
async def put_schedule_settings(body: PauseIn, p: Principal = Depends(limited(BUCKET)),
                                ctx: AppContext = Depends(get_ctx)):
    """Pause (or resume) every schedule at once: while paused, the runner does nothing (audited)."""
    svc = service(ctx)
    await asyncio.to_thread(svc.set_paused_all, body.paused_all)
    await asyncio.to_thread(audit, ctx.db, p.user.id, "schedules_pause_all" if body.paused_all else
                            "schedules_resume_all", "", p.audit_extra())
    return {"paused_all": body.paused_all, "timezone": zone_name(box_zone(ctx))}


@router.post("/deployments/{dep_id}/schedules", status_code=201)
async def create_schedules(dep_id: int, body: CreateIn, p: Principal = Depends(limited(BUCKET)),
                           ctx: AppContext = Depends(get_ctx)):
    view = await asyncio.to_thread(_view, ctx, dep_id)
    items = [ScheduleIn(**b.model_dump()) for b in body.schedules]
    try:
        rows = await asyncio.to_thread(service(ctx).create, view, items, p.user.id)
    except ScheduleError as e:
        raise _error(e)
    for s in rows:
        await asyncio.to_thread(audit, ctx.db, p.user.id, "schedule_create", view["name"],
                                {**_detail(s), **p.audit_extra()})
    return {"schedules": rows}


async def _schedule_view(ctx: AppContext, sid: int) -> tuple:
    row = await asyncio.to_thread(service(ctx).get, sid)
    if row is None:
        raise api_error(404, "not_found", f"schedule {sid} not found")
    return row, await asyncio.to_thread(_view, ctx, row.deployment_id)


@router.put("/schedules/{sid}")
async def update_schedule(sid: int, body: PatchIn, p: Principal = Depends(limited(BUCKET)),
                          ctx: AppContext = Depends(get_ctx)):
    _row, view = await _schedule_view(ctx, sid)
    patch = body.model_dump(exclude_unset=True, exclude_none=True)
    if not patch:
        raise api_error(422, "nothing_to_update", "send at least one of action, days, at_local, timezone, enabled")
    try:
        out = await asyncio.to_thread(service(ctx).update, sid, view, patch)
    except ScheduleError as e:
        raise _error(e)
    await asyncio.to_thread(audit, ctx.db, p.user.id, "schedule_update", view["name"],
                            {**_detail(out), "changed": sorted(patch), **p.audit_extra()})
    return out


@router.delete("/schedules/{sid}", status_code=204)
async def delete_schedule(sid: int, p: Principal = Depends(limited(BUCKET)), ctx: AppContext = Depends(get_ctx)):
    row, view = await _schedule_view(ctx, sid)
    gone = await asyncio.to_thread(service(ctx).delete, sid)
    if gone is None:
        raise api_error(404, "not_found", f"schedule {sid} not found")
    await asyncio.to_thread(audit, ctx.db, p.user.id, "schedule_delete", view["name"],
                            {"schedule_id": sid, "deployment_id": row.deployment_id, "action": row.action,
                             "at_local": row.at_local, **p.audit_extra()})
    return Response(status_code=204)
