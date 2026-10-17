"""Per-user (and optionally global) sliding-window rate limits for API routes, as FastAPI dependencies.

The limiter runs after authentication and the role check (a viewer hitting an admin route gets 403, never 429, and
does not use up the admin's budget). Over the limit -> 429 ``rate_limited`` with a Retry-After header."""
from __future__ import annotations

from fastapi import Depends, HTTPException

from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.context import AppContext

# bucket -> (requests per user per window, window seconds, requests per window across all users or None)
RATE_LIMITS: dict[str, tuple[int, float, int | None]] = {
    "decider_admin": (10, 60.0, None),      # settings, enable, disable, image build, key rotation (shared)
    "decider_suggest": (30, 60.0, 120),     # each call may consult the decision model twice
    "decider_feedback": (60, 60.0, None),
    "decider_read": (120, 60.0, None),      # the admin decision log
    "decider_plan": (30, 60.0, 120),        # plan ranking by the decision model in POST /api/deployments/preview
}


def _limiter(ctx: AppContext, name: str, limit: int, window: float) -> SlidingWindowLimiter:
    lim = ctx.rate_limiters.get(name)
    if lim is None:
        lim = ctx.rate_limiters[name] = SlidingWindowLimiter(limit, window)
    return lim


def _too_many(window: float) -> HTTPException:
    return HTTPException(429, detail={"code": "rate_limited", "message": f"too many requests; retry in {window:.0f} s",
                                      "retry_after_s": int(window)}, headers={"Retry-After": str(int(window))})


def allow(ctx: AppContext, bucket: str, user_id) -> bool:
    """Take one request from `bucket` for this user (and the all-users budget); False when over the limit. For calls
    that fall back instead of failing (e.g. plan ranking: over the limit, the preview is answered without the
    decision model)."""
    per_user, window, total = RATE_LIMITS[bucket]
    if not _limiter(ctx, f"{bucket}:user", per_user, window).hit(str(user_id)):
        return False
    return total is None or _limiter(ctx, f"{bucket}:all", total, window).hit("all")


def limited(bucket: str, *, admin: bool = True):
    per_user, window, total = RATE_LIMITS[bucket]
    base = require_admin if admin else current_principal

    def dep(p: Principal = Depends(base), ctx: AppContext = Depends(get_ctx)) -> Principal:
        if not _limiter(ctx, f"{bucket}:user", per_user, window).hit(str(p.user.id)):
            raise _too_many(window)
        if total is not None and not _limiter(ctx, f"{bucket}:all", total, window).hit("all"):
            raise _too_many(window)
        return p

    return dep
