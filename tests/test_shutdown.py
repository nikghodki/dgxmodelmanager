"""Shutdown must be prompt: `docker stop` gives 10 s, then SIGKILLs (exit 137).

Found live: an open SSE stream (any browser tab) kept uvicorn in "Waiting for connections to close" forever, and a
blocking worker thread (e.g. the catalog's oneshot container, a download) kept asyncio.run's teardown waiting for
the default executor (up to 5 min) after "Application shutdown complete"."""
import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import httpx
import pytest

from dgxmm.__main__ import SHUTDOWN_GRACE_S, server_kwargs


def test_server_cuts_open_connections_after_a_grace_period(settings):
    assert server_kwargs(settings)["timeout_graceful_shutdown"] == SHUTDOWN_GRACE_S <= 5


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


CHILD = textwrap.dedent("""
    import asyncio, os, signal, sys, time
    from pathlib import Path
    from dgxmm.__main__ import SHUTDOWN_GRACE_S, run_server
    from dgxmm.app import create_app
    from dgxmm.auth.passwords import hash_password
    from dgxmm.config import PathMapping, Settings
    from dgxmm.context import AppContext
    from dgxmm.db import Database, User
    from dgxmm.events import EventBroker

    port, tmp, mode = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    # In a container DGXMM is PID 1: the kernel ignores a SIGTERM it has no handler for, so uvicorn's re-raised
    # SIGTERM after shutdown does nothing there. Emulate that (uvicorn restores this disposition before re-raising).
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (tmp / "data").mkdir()
    s = Settings(data_dir=str(tmp / "data"), static_dir=str(tmp), hf_cache=PathMapping(host="/x", local=str(tmp)),
                 gguf_roots=[])
    db = Database(f"sqlite:///{s.db_path}")
    with db.session() as ss:
        ss.add(User(username="root", pw_hash=hash_password("root-password-123"), role="admin"))
    ctx = AppContext(settings=s, db=db, events=EventBroker())

    async def blocked_worker():  # like the catalog's oneshot container or a download: a thread busy for minutes
        await asyncio.to_thread(time.sleep, 120)
    ctx.background.append(blocked_worker)
    kw = dict(host="127.0.0.1", port=port, log_level="warning")
    if mode in ("admin", "events"):  # as in production: the TCP listener plus the admin unix socket
        kw["admin_socket"] = tmp / "data" / "run" / "admin.sock"
    if mode == "events":  # as in production: budget / metrics events every few seconds
        async def ticker():
            while True:
                ctx.events.publish("budget", {"t": time.time()})
                await asyncio.sleep(0.3)
        ctx.background.append(ticker)
    os._exit(run_server(create_app(ctx), timeout_graceful_shutdown=SHUTDOWN_GRACE_S, **kw))
""")


def start(tmp_path, mode, stderr=None):
    port = free_port()
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    work = tmp_path / mode
    work.mkdir()
    p = subprocess.Popen([sys.executable, str(script), str(port), str(work), mode],
                         env=dict(os.environ, PYTHONPATH=os.getcwd()), stderr=stderr)
    base = f"http://127.0.0.1:{port}/dgxmm/api"
    for _ in range(150):
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(f"{base}/auth/me", timeout=1).status_code == 401:
                break
        time.sleep(0.1)
    client = httpx.Client(base_url=base, timeout=None)
    assert client.post("/auth/login", json={"username": "root", "password": "root-password-123"}).status_code == 200
    resp = client.send(client.build_request("GET", "/stream"), stream=True)
    lines = resp.iter_lines()
    assert next(lines).startswith("event: hello")  # an open SSE stream, like a browser tab
    p.stream, p.lines = resp, lines  # type: ignore[attr-defined]
    return p


def stop(p, wait_s):
    t0 = time.monotonic()
    p.send_signal(signal.SIGTERM)
    try:
        return p.wait(wait_s), time.monotonic() - t0
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        return None, time.monotonic() - t0


@pytest.mark.timeout(90)
def test_pid1_exits_promptly_with_an_open_stream_and_a_blocked_worker_thread(tmp_path):
    code, took = stop(start(tmp_path, "new"), 15)
    assert code == 0 and took < SHUTDOWN_GRACE_S + 3, (code, took)


@pytest.mark.timeout(90)
def test_pid1_with_the_admin_socket_exits_promptly_and_removes_the_socket(tmp_path):
    p = start(tmp_path, "admin")
    sock = tmp_path / "admin" / "data" / "run" / "admin.sock"
    for _ in range(100):  # bound once the app's startup completed
        if sock.exists():
            break
        time.sleep(0.05)
    assert sock.is_socket()
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), timeout=5) as c:
        assert c.get("http://admin/dgxmm/api/auth/me").status_code == 401  # served on the socket (not signed in)
    code, took = stop(p, 15)
    assert code == 0 and took < SHUTDOWN_GRACE_S + 3, (code, took)
    assert not sock.exists()


async def test_lifespan_does_not_wait_forever_for_a_background_task_that_ignores_cancellation(ctx, monkeypatch):
    import dgxmm.app as app_mod
    monkeypatch.setattr(app_mod, "BACKGROUND_STOP_S", 0.5)

    async def stubborn():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(1.5)  # swallows the cancellation and keeps running for a while

    ctx.background.append(stubborn)
    app = app_mod.create_app(ctx)
    t0 = time.monotonic()
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.05)
    assert time.monotonic() - t0 < 2


@pytest.mark.timeout(90)
def test_sigterm_ends_open_streams_instead_of_cancelling_them(tmp_path):
    """Found live (docker restart dgxmm with a browser tab open): uvicorn waited the whole grace period for the SSE
    stream, then cancelled it and logged "Exception in ASGI application ... CancelledError" tracebacks. On SIGTERM
    the stream ends by itself, so shutdown is quick and the log stays clean."""
    err = tmp_path / "stderr.log"
    with open(err, "w") as fh:
        p = start(tmp_path, "events", stderr=fh)
        code, took = stop(p, 15)
    log = err.read_text()
    assert code == 0, log
    assert took < 3, (took, log)
    assert "Traceback" not in log and "CancelledError" not in log, log
