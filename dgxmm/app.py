"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from dgxmm.api import auth as auth_api, users as users_api, host as host_api, inventory as inventory_api
from dgxmm.api import engines as engines_api, deployments as deployments_api, stream as stream_api
from dgxmm.api import audit as audit_api, containers as containers_api
from dgxmm.api import catalog as catalog_api, downloads as downloads_api, planner as planner_api
from dgxmm.api import metrics as metrics_api
from dgxmm.context import AppContext

log = logging.getLogger("dgxmm")

ROUTERS = [auth_api, users_api, host_api, inventory_api, engines_api, deployments_api, stream_api, audit_api, containers_api, catalog_api, downloads_api, planner_api, metrics_api]


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
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if ctx.deployer is not None:
                await ctx.deployer.aclose()

    app = FastAPI(title="DGX Model Manager", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.ctx = ctx
    bp = ctx.settings.base_path
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
