"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from dgxmm.api import auth as auth_api, users as users_api, host as host_api, inventory as inventory_api
from dgxmm.api import engines as engines_api, deployments as deployments_api, stream as stream_api
from dgxmm.api import audit as audit_api, containers as containers_api
from dgxmm.api import catalog as catalog_api, downloads as downloads_api, planner as planner_api
from dgxmm.api import metrics as metrics_api, integrations as integrations_api
from dgxmm.api import decider as decider_api
from dgxmm.api import decider_features as decider_features_api
from dgxmm.context import AppContext

log = logging.getLogger("dgxmm")
BACKGROUND_STOP_S = 3.0  # how long shutdown waits for cancelled background loops

ROUTERS = [auth_api, users_api, host_api, inventory_api, engines_api, deployments_api, stream_api, audit_api, containers_api, catalog_api, downloads_api, planner_api, metrics_api, integrations_api, decider_api,
           decider_features_api]


class BodyLimit:
    """ASGI middleware: request bodies under the given path prefixes may not exceed their byte limit (413
    ``request_too_large``), whether sent with a Content-Length or chunked. The body is read (up to the limit) before
    the route runs, so nothing is parsed or validated for an oversized request."""

    def __init__(self, app, limits: list[tuple[str, int]]):
        self.app, self.limits = app, limits

    @staticmethod
    async def _refuse(scope, receive, send, limit: int) -> None:
        resp = JSONResponse({"detail": {"code": "request_too_large",
                                        "message": f"request body larger than {limit} bytes"}}, status_code=413)
        await resp(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = next((n for prefix, n in self.limits if scope["path"].startswith(prefix)), None)
        if limit is None:
            return await self.app(scope, receive, send)
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > limit):
            return await self._refuse(scope, receive, send, limit)
        body, more = b"", True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
            if len(body) > limit:
                return await self._refuse(scope, receive, send, limit)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for fn in ctx.startup:
            await fn()
        tasks = [asyncio.create_task(fn(), name=getattr(fn, "__name__", "bg")) for fn in ctx.background]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            if tasks:  # bounded: a task that swallows cancellation must not hold up the shutdown
                _, stuck = await asyncio.wait(tasks, timeout=BACKGROUND_STOP_S)
                for t in stuck:
                    log.warning("background task %s did not stop within %ss", t.get_name(), BACKGROUND_STOP_S)
                for t in tasks:
                    if t.done() and not t.cancelled() and t.exception() is not None:
                        log.warning("background task %s ended with %r", t.get_name(), t.exception())
            if ctx.downloads is not None and hasattr(ctx.downloads, "close"):
                ctx.downloads.close()  # decision-model snapshot downloads stop before their next chunk
            if ctx.deployer is not None:
                await ctx.deployer.aclose()

    app = FastAPI(title="DGX Model Manager", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.ctx = ctx
    bp = ctx.settings.base_path
    app.add_middleware(BodyLimit, limits=[(f"{bp}/api/decider", decider_api.MAX_REQUEST_BYTES)])
    for module in ROUTERS:
        app.include_router(module.router, prefix=bp)
    if bp:
        _mount_root_redirect(app, bp)
    _mount_static(app, Path(ctx.settings.static_dir), bp)
    return app


def _mount_root_redirect(app: FastAPI, bp: str) -> None:
    """Send `/` and `<bp>` to `<bp>/` so http://host:8090/ opens the UI (relative Location: proxy-safe)."""

    def to_ui():
        return RedirectResponse(bp + "/", status_code=307)

    app.add_api_route("/", to_ui, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route(bp, to_ui, methods=["GET", "HEAD"], include_in_schema=False)


def _mount_static(app: FastAPI, static: Path, bp: str) -> None:
    """Serve the built SPA; unknown non-API paths fall back to index.html."""

    @app.get(bp + "/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            from dgxmm.api.errors import api_error
            raise api_error(404, "not_found", "no such API route")
        candidate = (static / path).resolve()
        if path and candidate.is_file() and static.resolve() in candidate.parents:
            return FileResponse(candidate)
        index = static / "index.html"
        if index.is_file():
            return FileResponse(index)
        from dgxmm.api.errors import api_error
        raise api_error(404, "ui_not_built", "frontend has not been built")
