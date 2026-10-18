"""/api/insights — the insights feed (dgxmm/decider/insights.py). Viewers read (features with ``viewer_visible``,
without actions); admins dismiss, snooze and mark applied (CSRF, audited). Applying an insight's action goes through
the existing endpoints and their confirmations; nothing here executes it."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from dgxmm.api.errors import api_error
from dgxmm.api.ratelimit import limited
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider.insights import MAX_SNOOZE_HOURS, STATUSES, InsightClosed, insight_out

router = APIRouter(prefix="/api/insights", tags=["insights"])
MAX_REQUEST_BYTES = 4 * 1024  # every /api/insights body (enforced in app.BodyLimit before parsing)


class SnoozeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: StrictInt = Field(ge=1, le=MAX_SNOOZE_HOURS)


def _statuses(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    out = [s.strip() for s in raw.split(",") if s.strip()]
    bad = sorted(set(out) - set(STATUSES))
    if bad:
        raise api_error(422, "unknown_status", f"unknown status(es): {', '.join(bad)}", known=list(STATUSES))
    return out


@router.get("")
async def list_insights(status: str | None = Query(None, max_length=64),
                        subject: str | None = Query(None, max_length=256), kind: str | None = Query(None, max_length=32),
                        limit: int = Query(200, ge=1, le=500),
                        p: Principal = Depends(limited("insights_read", admin=False)),
                        ctx: AppContext = Depends(get_ctx)):
    ctx.insights.bind_loop(asyncio.get_running_loop())  # SSE from the worker thread goes through this loop
    return await asyncio.to_thread(ctx.insights.list, status=_statuses(status), subject=subject, kind=kind,
                                   limit=limit, admin=p.user.role == "admin")


async def _act(ctx: AppContext, p: Principal, insight_id: int, verb: str, fn, **detail) -> dict:
    """Database work in a worker thread (never on the event loop)."""
    ctx.insights.bind_loop(asyncio.get_running_loop())
    try:
        row = await asyncio.to_thread(fn)
    except InsightClosed as e:
        raise api_error(409, "insight_closed", f"insight {insight_id} is {e.args[0]}")
    if row is None:
        raise api_error(404, "not_found", f"insight {insight_id} not found")
    await asyncio.to_thread(audit, ctx.db, p.user.id, f"insight_{verb}", str(insight_id),
                            {"kind": row.kind, "feature": row.feature, "subject": row.subject, **detail,
                             **p.audit_extra()})
    return insight_out(row, admin=True)


@router.post("/{insight_id}/dismiss")
async def dismiss(insight_id: int, p: Principal = Depends(limited("insights_write")),
                  ctx: AppContext = Depends(get_ctx)):
    return await _act(ctx, p, insight_id, "dismiss", lambda: ctx.insights.dismiss(insight_id))


@router.post("/{insight_id}/snooze")
async def snooze(insight_id: int, body: SnoozeIn, p: Principal = Depends(limited("insights_write")),
                 ctx: AppContext = Depends(get_ctx)):
    return await _act(ctx, p, insight_id, "snooze", lambda: ctx.insights.snooze(insight_id, body.hours), hours=body.hours)


@router.post("/{insight_id}/applied")
async def applied(insight_id: int, p: Principal = Depends(limited("insights_write")),
                  ctx: AppContext = Depends(get_ctx)):
    return await _act(ctx, p, insight_id, "applied", lambda: ctx.insights.mark_applied(insight_id))
