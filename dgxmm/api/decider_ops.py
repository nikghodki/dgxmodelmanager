"""HTTP endpoints of the operations features (dgxmm/decider/operations.py): LiteLLM routing advice. Advisory only:
nothing here talks to LiteLLM."""
from fastapi import APIRouter, Depends

from dgxmm.api.ratelimit import limited
from dgxmm.auth.deps import Principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.decider import registry

router = APIRouter(prefix="/api/decider", tags=["decider-operations"])


@router.get("/routing-advice")
async def routing_advice(p: Principal = Depends(limited(registry.feature_bucket("routing"))),
                         ctx: AppContext = Depends(get_ctx)):
    """Per traffic class: the suggested deployment, its LiteLLM model name, source/confidence and the candidates;
    plus the copyable LiteLLM snippet. Before the first job run, a rules-only answer (no decider call)."""
    if ctx.routing is None:
        from dgxmm.decider.routing import RoutingService
        ctx.routing = RoutingService(ctx)
    enabled = ctx.decider.rules_enabled("routing") if hasattr(ctx.decider, "rules_enabled") else True
    if not enabled:
        return {"enabled": False, "classes": [], "snippet": "", "computed_at": None, "source": "none"}
    return {"enabled": True, **await ctx.routing.current()}
