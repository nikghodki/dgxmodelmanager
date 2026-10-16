from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from dgxmm.auth.deps import Principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.db import AuditLog, User

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("")
def list_audit(limit: int = Query(200, ge=1, le=2000), p: Principal = Depends(require_admin),
               ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        names = {u.id: u.username for u in s.scalars(select(User))}
        rows = s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit))
        return [{"ts": a.ts.isoformat(), "user": names.get(a.user_id, "system"), "action": a.action,
                 "target": a.target, "detail": a.detail} for a in rows]
