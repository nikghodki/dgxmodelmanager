"""Short-lived, scoped admin tokens for the local admin CLI (``docker exec dgxmm python -m dgxmm.admin decider ...``).

They let the installer drive the running app's API without a password (the CLI already has the database), are
stored hashed, expire, are accepted only on the admin unix-socket listener (never on the TCP listener, whatever
X-Forwarded-For says), only on the decider routes, and never need a CSRF header."""
from __future__ import annotations

import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from dgxmm.auth import cli_tokens
from dgxmm.db import CliToken, User, utcnow
from tests.test_api_decider import B, api  # noqa: F401  (fixture)

HDR = cli_tokens.HEADER


def local(api):  # noqa: F811
    """A client of the admin listener (what the unix socket serves)."""
    from dgxmm.admin_listener import AdminListener
    return TestClient(AdminListener(api.app), base_url="http://testserver", client=("127.0.0.1", 40000))


def admin_id(db, name="root"):
    with db.session() as s:
        return s.scalar(select(User.id).where(User.username == name))


def test_token_works_on_decider_routes_from_loopback_without_csrf(api):  # noqa: F811
    tok = cli_tokens.issue(api.ctx.db, admin_id(api.ctx.db), ttl_s=60)
    c = local(api)
    assert c.get(f"{B}/status", headers={HDR: tok}).status_code == 200
    r = c.post(f"{B}/enable", headers={HDR: tok}, json={})
    assert r.status_code == 202, r.text
    assert c.post("/dgxmm/api/inventory/refresh", headers={HDR: tok}).status_code in (200, 202)
    assert c.post(f"{B}/disable", headers={HDR: tok}, json={"remove": True}).status_code == 200


def test_token_is_scoped_hashed_and_revocable(api):  # noqa: F811
    db = api.ctx.db
    tok = cli_tokens.issue(db, admin_id(db), ttl_s=60)
    with db.session() as s:
        stored = [r.id for r in s.scalars(select(CliToken))]
    assert stored and tok not in stored and all(len(i) == 64 for i in stored)
    c = local(api)
    for method, path in (("GET", "/dgxmm/api/users"), ("POST", "/dgxmm/api/deployments"),
                         ("PUT", f"{B}/settings"), ("GET", f"{B}/decisions"), ("GET", "/dgxmm/api/auth/me")):
        r = c.request(method, path, headers={HDR: tok}, json={})
        assert r.status_code == 403 and r.json()["detail"]["code"] == "cli_scope", (method, path)
    cli_tokens.revoke(db, tok)
    r = c.get(f"{B}/status", headers={HDR: tok})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "invalid_cli_token"


def test_expired_and_unknown_tokens_are_rejected(api):  # noqa: F811
    db = api.ctx.db
    tok = cli_tokens.issue(db, admin_id(db), ttl_s=60)
    assert local(api).get(f"{B}/status", headers={HDR: "dgxmmcli_" + "x" * 43}).status_code == 401
    with db.session() as s:
        for row in s.scalars(select(CliToken)):
            row.expires_at = utcnow() - timedelta(seconds=1)
    assert local(api).get(f"{B}/status", headers={HDR: tok}).status_code == 401
    with db.session() as s:
        assert s.scalars(select(CliToken)).all() == []  # expired rows are swept


def test_tcp_listener_never_accepts_cli_tokens_even_from_loopback(api):  # noqa: F811
    tok = cli_tokens.issue(api.ctx.db, admin_id(api.ctx.db), ttl_s=60)
    for client in (api, TestClient(api.app, base_url="http://testserver", client=("127.0.0.1", 40000))):
        r = client.get(f"{B}/status", headers={HDR: tok})
        assert r.status_code == 403 and r.json()["detail"]["code"] == "cli_not_admin_listener"


def test_spoofed_forwarded_for_through_proxy_headers_cannot_reach_the_cli_path(api):  # noqa: F811
    """An engine container on the docker network (trusted as a proxy) sends X-Forwarded-For: 127.0.0.1."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    tok = cli_tokens.issue(api.ctx.db, admin_id(api.ctx.db), ttl_s=60)
    c = TestClient(ProxyHeadersMiddleware(api.app, trusted_hosts="*"), base_url="http://testserver",
                   client=("172.18.0.5", 40000))
    for hdr in ({"X-Forwarded-For": "127.0.0.1"}, {"X-Forwarded-For": "::1", "X-Forwarded-Proto": "http"}):
        r = c.post(f"{B}/enable", headers={HDR: tok, **hdr}, json={})
        assert r.status_code == 403 and r.json()["detail"]["code"] == "cli_not_admin_listener"
    assert api.get(f"{B}/status").json()["enabled"] is False


@pytest.mark.parametrize("hdr", [{"X-Forwarded-For": "127.0.0.1"}, {"Forwarded": "for=127.0.0.1"},
                                 {"X-Real-IP": "127.0.0.1"}])
def test_forwarded_requests_are_refused_even_on_the_admin_listener(api, hdr):  # noqa: F811
    tok = cli_tokens.issue(api.ctx.db, admin_id(api.ctx.db), ttl_s=60)
    r = local(api).get(f"{B}/status", headers={HDR: tok, **hdr})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "cli_forwarded"


def test_issue_and_revoke_are_audited_without_the_token(api):  # noqa: F811
    from dgxmm.db import AuditLog
    db = api.ctx.db
    tok = cli_tokens.issue(db, admin_id(db), ttl_s=60)
    cli_tokens.revoke(db, tok)
    with db.session() as s:
        rows = {a.action: a for a in s.scalars(select(AuditLog))}
    assert {"cli_token_issued", "cli_token_revoked"} <= set(rows)
    assert rows["cli_token_issued"].detail["scope"] == "decider" and rows["cli_token_issued"].detail["ttl_s"] == 60
    assert all(tok not in str(a.detail) and tok not in a.target for a in rows.values())


def test_cli_actions_are_audited_with_via_cli(api):  # noqa: F811
    from dgxmm.db import AuditLog
    tok = cli_tokens.issue(api.ctx.db, admin_id(api.ctx.db), ttl_s=60)
    c = local(api)
    assert c.post("/dgxmm/api/inventory/refresh", headers={HDR: tok}).status_code == 200
    assert c.post(f"{B}/enable", headers={HDR: tok}, json={}).status_code == 202
    assert c.post(f"{B}/disable", headers={HDR: tok}, json={}).status_code == 200
    with api.ctx.db.session() as s:
        rows = {a.action: a.detail for a in s.scalars(select(AuditLog))}
    for action in ("inventory_refreshed", "decider_enable", "decider_disable"):
        assert rows[action].get("via") == "cli", action
    # a browser session's actions are not marked
    api.post(f"{B}/disable", headers=api.h, json={})
    with api.ctx.db.session() as s:
        last = s.scalars(select(AuditLog).where(AuditLog.action == "decider_disable").order_by(AuditLog.id.desc())).first()
    assert "via" not in last.detail


def test_only_admins_get_tokens_and_a_demoted_user_loses_them(api):  # noqa: F811
    db = api.ctx.db
    with pytest.raises(PermissionError):
        cli_tokens.issue(db, admin_id(db, "vic"), ttl_s=60)
    tok = cli_tokens.issue(db, admin_id(db), ttl_s=60)
    with db.session() as s:
        s.scalar(select(User).where(User.username == "root")).role = "viewer"
    r = local(api).post(f"{B}/enable", headers={HDR: tok}, json={})
    assert r.status_code == 403


def test_fresh_install_admin_with_pending_password_change_can_use_the_cli(api):  # noqa: F811
    db = api.ctx.db
    with db.session() as s:
        s.scalar(select(User).where(User.username == "root")).must_change_pw = True
    tok = cli_tokens.issue(db, admin_id(db), ttl_s=60)
    assert local(api).get(f"{B}/status", headers={HDR: tok}).status_code == 200


def test_ttl_is_bounded(api):  # noqa: F811
    db = api.ctx.db
    with pytest.raises(ValueError):
        cli_tokens.issue(db, admin_id(db), ttl_s=cli_tokens.MAX_TTL_S + 1)
    t0 = time.time()
    cli_tokens.issue(db, admin_id(db), ttl_s=5)
    with db.session() as s:
        row = s.scalars(select(CliToken)).one()
        from dgxmm.db import as_aware
        assert as_aware(row.expires_at).timestamp() - t0 <= 6


def test_real_unix_socket_listener(tmp_path, settings, db, make_user):
    """uvicorn on the admin unix socket (mode 0600 in a 0700 directory); the CLI's default client reaches it."""
    import os
    import stat as st
    import threading

    import uvicorn

    from dgxmm.admin_listener import AdminListener, bind_admin_socket
    from dgxmm.app import create_app
    from dgxmm.auth.ratelimit import SlidingWindowLimiter
    from dgxmm.context import AppContext
    from dgxmm.decider import cli
    from dgxmm.events import EventBroker
    make_user("root", "root-password-123", "admin")
    ctx = AppContext(settings=settings, db=db, events=EventBroker(), login_limiter=SlidingWindowLimiter(5, 60))
    path = settings.admin_socket_path
    sock = bind_admin_socket(path)
    assert st.S_IMODE(os.stat(path).st_mode) == 0o600 and st.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    server = uvicorn.Server(uvicorn.Config(AdminListener(create_app(ctx)), lifespan="off", log_level="warning"))
    t = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            __import__("time").sleep(0.05)
        lines = []
        rc = cli.run(["status"], db=db, settings=settings, out=lines.append)
        assert rc == 0 and "disabled" in "\n".join(lines)
    finally:
        server.should_exit = True
        t.join(5)
