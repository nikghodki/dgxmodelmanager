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


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def current_principal(request: Request, ctx: AppContext = Depends(get_ctx)) -> Principal:
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
