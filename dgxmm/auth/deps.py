import secrets
from dataclasses import dataclass

from fastapi import Depends, Request

from dgxmm.api.errors import api_error
from dgxmm.auth.sessions import resolve_session
from dgxmm.context import AppContext
from dgxmm.db import User, UserSession

COOKIE = "dgxmm_session"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
PW_CHANGE_ALLOWED = {"/api/auth/me", "/api/auth/logout", "/api/auth/change-password"}


@dataclass
class Principal:
    user: User
    session: UserSession
    via: str = "session"  # "cli": an admin-CLI token on the admin socket

    def audit_extra(self) -> dict:
        """Extra audit detail marking actions taken through the admin CLI."""
        return {"via": "cli"} if self.via == "cli" else {}


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


FORWARDING_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")


def _cli_principal(request: Request, ctx: AppContext, token: str) -> Principal:
    """A local admin-CLI token (dgxmm/auth/cli_tokens.py): admin unix socket only, no forwarding headers, scoped
    routes, admin user, no CSRF (the token is not a cookie)."""
    from dgxmm.admin_listener import on_admin_listener
    from dgxmm.auth import cli_tokens
    # Never trust peer addresses here: on the TCP listener they come from proxy headers, which a trusted proxy
    # peer (any container on the docker network, when trusted_proxies covers it) controls.
    if not on_admin_listener(request.scope):
        raise api_error(403, "cli_not_admin_listener", "CLI tokens are only accepted on the admin socket")
    if any(h in request.headers for h in FORWARDING_HEADERS):
        raise api_error(403, "cli_forwarded", "CLI token requests must not be forwarded")
    found = cli_tokens.resolve(ctx.db, token)
    if not found:
        raise api_error(401, "invalid_cli_token", "unknown or expired CLI token")
    row, user = found
    path = request.url.path.removeprefix(ctx.settings.base_path)
    if not cli_tokens.allowed(row.scope, request.method, path):
        raise api_error(403, "cli_scope", "this CLI token is not valid for this route")
    if user.role != "admin":
        raise api_error(403, "admin_required", "this action needs the admin role")
    return Principal(user, UserSession(id="cli", user_id=user.id, csrf_token="", ip="cli", expires_at=row.expires_at),
                     via="cli")


def current_principal(request: Request, ctx: AppContext = Depends(get_ctx)) -> Principal:
    cli_token = request.headers.get("x-dgxmm-cli-token")
    if cli_token is not None:
        return _cli_principal(request, ctx, cli_token)
    found = resolve_session(ctx.db, request.cookies.get(COOKIE), ctx.settings.session_ttl_s)
    if not found:
        raise api_error(401, "not_authenticated", "log in first")
    sess, user = found
    if request.method not in SAFE_METHODS:
        sent = request.headers.get("x-csrf-token", "")
        if not secrets.compare_digest(sent, sess.csrf_token):
            raise api_error(403, "csrf_failed", "missing or invalid X-CSRF-Token header")
    path = request.url.path.removeprefix(ctx.settings.base_path)
    if user.must_change_pw and path not in PW_CHANGE_ALLOWED:
        raise api_error(403, "password_change_required", "change your password before continuing")
    return Principal(user, sess)


def require_admin(p: Principal = Depends(current_principal)) -> Principal:
    if p.user.role != "admin":
        raise api_error(403, "admin_required", "this action needs the admin role")
    return p
