"""The combined server (dgxmm/admin_listener.py): TCP listener + admin unix socket on one app. The socket starts
only after the app's startup completed, is removed on shutdown (also on SIGTERM), and a bad ``data/run`` disables
only the admin socket, never the app."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from dgxmm import admin_listener
from dgxmm.admin_listener import DualServer

ROOT = Path(__file__).resolve().parent.parent


def make_app(events: list, sock: Path):
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await asyncio.sleep(0.3)  # slow startup: the socket must not exist yet
                    events.append(("startup", sock.exists()))
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    events.append(("shutdown", sock.exists()))
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        flag = "admin" if admin_listener.on_admin_listener(scope) else "tcp"
        await send({"type": "http.response.body", "body": flag.encode()})
    return app


def kwargs():
    return {"host": "127.0.0.1", "port": 0, "log_level": "warning", "proxy_headers": True}


async def wait_for(cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        await asyncio.sleep(0.02)


async def test_admin_socket_starts_after_startup_serves_marked_requests_and_is_removed(tmp_path):
    sock, events = tmp_path / "run" / "admin.sock", []
    ds = DualServer(make_app(events, sock), kwargs(), sock)
    task = asyncio.create_task(ds.serve())
    await wait_for(lambda: sock.exists())
    assert events == [("startup", False)]
    assert oct(sock.stat().st_mode & 0o777) == "0o600" and oct(sock.parent.stat().st_mode & 0o777) == "0o700"
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(sock)), base_url="http://x") as c:
        assert (await c.get("/")).text == "admin"
    port = ds.main.servers[0].sockets[0].getsockname()[1]
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
        assert (await c.get("/")).text == "tcp"
    ds.main.should_exit = True
    await asyncio.wait_for(task, 10)
    assert not sock.exists()
    assert events[-1] == ("shutdown", False)  # the admin socket is gone before the app shuts down


@pytest.mark.parametrize("setup", ["dangling_symlink", "regular_file", "foreign_owner"])
async def test_a_bad_run_directory_disables_only_the_admin_socket(tmp_path, monkeypatch, caplog, setup):
    run = tmp_path / "run"
    sock = run / "admin.sock"
    if setup == "dangling_symlink":
        run.symlink_to(tmp_path / "nowhere")
    elif setup == "regular_file":
        run.mkdir(mode=0o700)
        sock.write_text("not a socket")
    else:
        run.mkdir(mode=0o700)
        real = os.geteuid()
        monkeypatch.setattr(admin_listener.os, "geteuid", lambda: real + 4242)
    events: list = []
    ds = DualServer(make_app(events, sock), kwargs(), sock)
    task = asyncio.create_task(ds.serve())
    await wait_for(lambda: ds.main.started)
    port = ds.main.servers[0].sockets[0].getsockname()[1]
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
        assert (await c.get("/")).text == "tcp"  # the app keeps working
    assert any("admin socket" in r.getMessage() for r in caplog.records)
    assert ds.admin_task is None
    ds.main.should_exit = True
    await asyncio.wait_for(task, 10)
    if setup == "regular_file":
        assert sock.read_text() == "not a socket"  # never deletes something that is not its socket
    assert not (tmp_path / "nowhere").exists()


def test_sigterm_removes_the_admin_socket(tmp_path):
    sock = tmp_path / "run" / "admin.sock"
    script = textwrap.dedent(f"""
        import asyncio, sys
        sys.path.insert(0, {str(ROOT)!r})
        from dgxmm.admin_listener import DualServer
        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    m = await receive()
                    await send({{"type": m["type"] + ".complete"}})
                    if m["type"] == "lifespan.shutdown":
                        return
        asyncio.run(DualServer(app, {{"host": "127.0.0.1", "port": 0, "log_level": "warning"}}, {str(sock)!r}).serve())
    """)
    p = subprocess.Popen([sys.executable, "-c", script])
    try:
        t0 = time.monotonic()
        while not sock.exists():
            assert p.poll() is None and time.monotonic() - t0 < 20
            time.sleep(0.05)
        p.send_signal(signal.SIGTERM)
        p.wait(20)
    finally:
        if p.poll() is None:
            p.kill()
    assert not sock.exists()


def test_the_admin_server_does_not_silence_the_main_servers_logs(tmp_path):
    import logging
    DualServer(lambda *a: None, {"host": "127.0.0.1", "port": 0, "log_level": "info"}, tmp_path / "s.sock")
    assert logging.getLogger("uvicorn.error").getEffectiveLevel() == logging.INFO


def test_the_admin_server_keeps_the_main_servers_access_log():
    """uvicorn's access log is one global logger ("uvicorn.access"): a second Config with access_log=False removes
    its handlers and stops its propagation, which silenced every request line of the TCP server."""
    import logging

    import uvicorn
    uvicorn.Config(lambda *a: None, host="127.0.0.1", port=0, log_level="info")  # the main server's logging setup
    access = logging.getLogger("uvicorn.access")
    assert access.hasHandlers()
    DualServer(lambda *a: None, {"host": "127.0.0.1", "port": 0, "log_level": "info"}, Path("/nonexistent/s.sock"))
    assert access.hasHandlers()


def test_sigterm_with_an_admin_request_in_flight_exits_promptly(tmp_path):
    """A CLI request still running on the admin socket (e.g. a slow enable) must not hold the shutdown: the admin
    server has the same grace period as the TCP listener."""
    sock = tmp_path / "run" / "admin.sock"
    script = textwrap.dedent(f"""
        import asyncio, signal, sys
        sys.path.insert(0, {str(ROOT)!r})
        from dgxmm.__main__ import SHUTDOWN_GRACE_S, run_server
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # PID 1 in a container (uvicorn restores this, then re-raises)
        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    m = await receive()
                    await send({{"type": m["type"] + ".complete"}})
                    if m["type"] == "lifespan.shutdown":
                        return
            await asyncio.sleep(120)  # a request that takes minutes
        import os
        os._exit(run_server(app, admin_socket={str(sock)!r}, host="127.0.0.1", port=0, log_level="warning",
                            timeout_graceful_shutdown=SHUTDOWN_GRACE_S))
    """)
    p = subprocess.Popen([sys.executable, "-c", script])
    client = None
    try:
        t0 = time.monotonic()
        while not sock.exists():
            assert p.poll() is None and time.monotonic() - t0 < 20
            time.sleep(0.05)
        import socket as so
        client = so.socket(so.AF_UNIX, so.SOCK_STREAM)
        client.connect(str(sock))
        client.sendall(b"GET /slow HTTP/1.1\r\nHost: admin\r\n\r\n")  # in flight: the handler sleeps
        time.sleep(0.5)
        t1 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        code = p.wait(30)
        took = time.monotonic() - t1
    finally:
        if client is not None:
            client.close()
        if p.poll() is None:
            p.kill()
    assert code == 0 and took < 9, (code, took)
    assert not sock.exists()
