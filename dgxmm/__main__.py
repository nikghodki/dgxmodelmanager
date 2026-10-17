import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn

from dgxmm import lifecycle
from dgxmm.app import create_app
from dgxmm.config import Settings, load_settings
from dgxmm.context import build_context

# Open SSE streams (every browser tab) never end on their own: without a grace period uvicorn waits for them
# forever and `docker stop` ends in SIGKILL. 5 s stays inside docker's default 10 s stop timeout.
SHUTDOWN_GRACE_S = 5


def server_kwargs(settings: Settings) -> dict:
    """Build uvicorn keyword arguments from Settings."""
    return {
        "host": "0.0.0.0",
        "port": 8090,
        # X-Forwarded-* from `trusted_proxies` is applied by dgxmm.proxy_headers around the TCP app (never from the
        # decider's network); uvicorn's own handling is off so it cannot apply them first
        "proxy_headers": False,
        "log_level": "info",
        "timeout_graceful_shutdown": SHUTDOWN_GRACE_S,
    }


def run_server(app, admin_socket: Path | str | None = None, tcp_app=None, **kwargs) -> int:
    """uvicorn.run without asyncio.run's teardown, which waits (up to 5 min) for the default executor's worker
    threads: one blocked in a long call (a oneshot container, a download, an image build) kept the process alive
    after "Application shutdown complete". Returns the exit code; the caller exits with os._exit so the interpreter
    does not join those threads either.

    With ``admin_socket`` the TCP listener is paired with the admin unix socket (dgxmm/admin_listener.py
    ``DualServer``): bound after the app's startup, stopped and removed in the TCP server's shutdown. ``tcp_app``
    (default ``app``) is what the TCP listener serves: the app behind the proxy-headers guard."""
    if admin_socket is not None:
        from dgxmm.admin_listener import DualServer
        dual = DualServer(app, kwargs, Path(admin_socket), main_app=tcp_app)
        main_server, serve = dual.main, dual.serve
    else:
        main_server = uvicorn.Server(uvicorn.Config(tcp_app or app, **kwargs))
        serve = main_server.serve
    uvicorn_exit = main_server.handle_exit

    def handle_exit(sig, frame) -> None:  # uvicorn installs self.handle_exit as the SIGTERM / SIGINT handler
        lifecycle.request_stop()  # open SSE streams end now instead of being cancelled after the grace period
        uvicorn_exit(sig, frame)

    main_server.handle_exit = handle_exit
    factory = main_server.config.get_loop_factory() or asyncio.new_event_loop
    loop = factory()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(serve())
    finally:
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=2))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()  # shuts the default executor down without waiting for its threads
    return 0 if main_server.started else 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    # TCP listener for the UI/API, plus the admin unix socket used by `python -m dgxmm.admin decider ...`.
    from dgxmm.proxy_headers import tcp_app
    ctx = build_context(settings)
    app = create_app(ctx)
    code = run_server(app, admin_socket=settings.admin_socket_path, tcp_app=tcp_app(app, settings, ctx),
                      **server_kwargs(settings))
    logging.getLogger("dgxmm").info("exiting (%d)", code)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
