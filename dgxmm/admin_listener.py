"""The admin listener: a unix socket inside the container (``<data_dir>/run/admin.sock``, mode 0600 in a 0700
directory owned by DGXMM's user) served by a second uvicorn server on the same app. It is the only place CLI tokens
(dgxmm/auth/cli_tokens.py) are accepted, so they cannot be presented over the network at all: the TCP listener runs
uvicorn's proxy-headers middleware, where a trusted peer (for example any container on the docker network, when
``trusted_proxies`` covers it) could otherwise claim ``X-Forwarded-For: 127.0.0.1``. This listener has no
proxy-headers handling and no lifespan.

``DualServer`` composes the two: the TCP server owns the app's lifespan and the process signals; the admin socket is
bound only after the app's startup completed, and is stopped and removed in the TCP server's shutdown (before the
app's own shutdown), which also runs on SIGTERM/SIGINT. If ``data/run`` is unusable (a symlink, not a directory,
owned by someone else, a non-socket at the socket path, or any OS error) only the admin socket is disabled, with an
error in the log: the UI and API keep working.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import stat
from pathlib import Path

log = logging.getLogger(__name__)
SCOPE_KEY = "dgxmm.admin_listener"
ADMIN_SHUTDOWN_GRACE_S = 5


class AdminListener:
    """ASGI wrapper marking every request as received on the admin socket."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = {**scope, SCOPE_KEY: True}
        await self.app(scope, receive, send)


def on_admin_listener(scope) -> bool:
    return scope.get(SCOPE_KEY) is True


class AdminSocketRefused(Exception):
    pass


def bind_admin_socket(path: Path) -> socket.socket:
    """Bind the unix socket owner-only. The directory must be a real directory (never a symlink) owned by this user;
    it is created (0700) or tightened. Only a stale *socket* is replaced; anything else at the path is refused."""
    path = Path(path)
    parent = path.parent
    try:
        st = os.lstat(parent)
    except FileNotFoundError:
        parent.mkdir(mode=0o700, parents=True)
        st = os.lstat(parent)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise AdminSocketRefused(f"{parent} is not a real directory")
    if st.st_uid != os.geteuid():
        raise AdminSocketRefused(f"{parent} belongs to uid {st.st_uid}, not {os.geteuid()}")
    os.chmod(parent, 0o700)
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISSOCK(existing.st_mode):
            raise AdminSocketRefused(f"{path} exists and is not a socket")
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        sock.bind(str(path))
    except OSError:
        sock.close()
        raise
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    sock.listen(64)
    return sock


def _remove_socket(path: Path) -> None:
    with contextlib.suppress(OSError):
        if stat.S_ISSOCK(os.lstat(path).st_mode):
            Path(path).unlink()


def _main_server_class():
    import uvicorn

    class MainServer(uvicorn.Server):
        """The TCP server; stops the admin socket first when it shuts down (graceful exit, SIGTERM or SIGINT)."""

        def __init__(self, config, before_shutdown):
            super().__init__(config)
            self._before_shutdown = before_shutdown

        async def shutdown(self, sockets=None):
            await self._before_shutdown()
            await super().shutdown(sockets=sockets)

    return MainServer


class DualServer:
    def __init__(self, app, main_config_kwargs: dict, socket_path: Path, main_app=None):
        """`main_app`: what the TCP listener serves (default `app`); the admin socket always serves `app` itself."""
        import uvicorn
        self.socket_path = Path(socket_path)
        self.main = _main_server_class()(uvicorn.Config(main_app or app, **main_config_kwargs), self.stop_admin)
        # log_config=None: keep the logging the TCP server's config set up (a second Config would reconfigure
        # uvicorn's loggers and silence the main server's INFO lines). access_log stays on: "uvicorn.access" is one
        # global logger and access_log=False strips its handlers, which silenced the TCP server's request log too.
        # The same grace period as the TCP listener (5 s by default): a CLI request still running on the socket at
        # shutdown is cut off then, instead of holding the process until docker's SIGKILL.
        grace = main_config_kwargs.get("timeout_graceful_shutdown") or ADMIN_SHUTDOWN_GRACE_S
        self.admin = uvicorn.Server(uvicorn.Config(AdminListener(app), lifespan="off", proxy_headers=False,
                                                   log_config=None,
                                                   timeout_graceful_shutdown=grace))
        self.admin.capture_signals = contextlib.nullcontext  # the TCP server handles the process signals
        self.admin_task: asyncio.Task | None = None

    def start_admin(self) -> None:
        try:
            sock = bind_admin_socket(self.socket_path)
        except (OSError, AdminSocketRefused) as e:
            log.error("admin socket %s disabled: %s. The UI and API keep working, but `python -m dgxmm.admin "
                      "decider status|enable|disable` cannot reach DGXMM until this is fixed and DGXMM restarted",
                      self.socket_path, e)
            return
        self.admin_task = asyncio.create_task(self.admin.serve(sockets=[sock]), name="admin-socket")

    async def stop_admin(self) -> None:
        task, self.admin_task = self.admin_task, None
        if task is not None:
            self.admin.should_exit = True
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            _remove_socket(self.socket_path)

    async def serve(self) -> None:
        main_task = asyncio.create_task(self.main.serve(), name="main-server")
        while not self.main.started and not main_task.done():  # app startup (lifespan) first, then the socket
            await asyncio.sleep(0.05)
        if self.main.started and not main_task.done():
            self.start_admin()
        try:
            await main_task
        finally:
            await self.stop_admin()

