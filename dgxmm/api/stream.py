import asyncio
import json
from typing import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from dgxmm import lifecycle
from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api", tags=["stream"])

POLL_S = 1.0  # how often an idle stream checks for a disconnect or a server shutdown
PING_S = 15.0  # keep-alive comment after this long without events

# SSE topics by role (docs/security.md, "SSE topics"). Every topic a viewer receives carries what the same viewer
# can already read over REST (deployment views, budget, metrics, inventory, downloads, benchmarks, alerts,
# diagnosis notices). Admin-only topics carry what only admins may read.
ADMIN_ONLY_TOPICS = frozenset({
    "decider",  # decision-model image build progress: docker build output and the image's package list (SBOM)
})
VIEWER_TOPICS = frozenset({
    "alert", "benchmark", "budget", "deployment", "deployment_removed", "diagnosis", "download", "inventory", "metrics",
    "insights",  # {"id": <insight id>} only: clients re-fetch GET /api/insights, which filters by role
})


def visible(topic: str, is_admin: bool) -> bool:
    return is_admin or topic not in ADMIN_ONLY_TOPICS


async def sse_events(q: asyncio.Queue, is_admin: bool, disconnected: Callable[[], Awaitable[bool]],
                     ping_s: float = PING_S, poll_s: float = POLL_S) -> AsyncIterator[str]:
    yield "event: hello\ndata: {}\n\n"
    poll = min(poll_s, ping_s)
    quiet = 0.0
    while True:
        # The server is stopping (SIGTERM): end the stream so shutdown need not wait (dgxmm/lifecycle.py).
        if lifecycle.stopping() or await disconnected():
            break
        try:
            topic, data = await asyncio.wait_for(q.get(), timeout=poll)
        except asyncio.TimeoutError:
            quiet += poll
            if quiet >= ping_s:
                quiet = 0.0
                yield ": ping\n\n"
            continue
        quiet = 0.0
        if visible(topic, is_admin):
            yield f"event: {topic}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/stream")
async def stream(request: Request, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    q = ctx.events.subscribe()
    is_admin = p.user.role == "admin"

    async def gen():
        try:
            async for chunk in sse_events(q, is_admin, request.is_disconnected):
                yield chunk
        finally:
            ctx.events.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
