from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel
from sqlalchemy import select

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import COOKIE, Principal, current_principal, get_ctx
from dgxmm.auth.passwords import DUMMY_HASH, hash_password, validate_new_password, verify_password
from dgxmm.auth.sessions import create_session, delete_session, delete_user_sessions
from dgxmm.context import AppContext
from dgxmm.db import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str


def user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "role": u.role, "must_change_pw": u.must_change_pw}


def _secure(request: Request, ctx: AppContext) -> bool:
    mode = ctx.settings.secure_cookies
    if mode != "auto":
        return mode == "always"
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response, ctx: AppContext = Depends(get_ctx)):
    ip = request.client.host if request.client else "unknown"
    username = body.username.strip().lower()
    ip_ok = ctx.login_limiter.hit(f"ip:{ip}")
    user_ok = ctx.login_limiter.hit(f"user:{username}")
    if not (ip_ok and user_ok):
        audit(ctx.db, None, "login_rate_limited", username, {"ip": ip})
        raise api_error(429, "too_many_attempts", "too many login attempts; wait a minute")
    with ctx.db.session() as s:
        user = s.scalar(select(User).where(User.username == username))
    if user is None:
        verify_password(DUMMY_HASH, body.password)
        raise api_error(401, "invalid_credentials", "wrong username or password")
    if not verify_password(user.pw_hash, body.password):
        audit(ctx.db, user.id, "login_failed", username, {"ip": ip})
        raise api_error(401, "invalid_credentials", "wrong username or password")
    token, csrf = create_session(ctx.db, user.id, ip, ctx.settings.session_ttl_s)
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=_secure(request, ctx),
                        max_age=ctx.settings.session_ttl_s, path=ctx.settings.base_path or "/")
    audit(ctx.db, user.id, "login", username, {"ip": ip})
    return {"user": user_out(user), "csrf_token": csrf}


@router.post("/logout")
def logout(request: Request, response: Response, p: Principal = Depends(current_principal),
           ctx: AppContext = Depends(get_ctx)):
    delete_session(ctx.db, request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path=ctx.settings.base_path or "/")
    return {"ok": True}


@router.get("/me")
def me(p: Principal = Depends(current_principal)):
    return {"user": user_out(p.user), "csrf_token": p.session.csrf_token}


@router.post("/change-password")
def change_password(body: ChangePasswordIn, request: Request, p: Principal = Depends(current_principal),
                    ctx: AppContext = Depends(get_ctx)):
    if not verify_password(p.user.pw_hash, body.current_password):
        raise api_error(401, "invalid_credentials", "current password is wrong")
    try:
        validate_new_password(body.new_password)
    except ValueError as e:
        raise api_error(422, "weak_password", str(e))
    with ctx.db.session() as s:
        u = s.get(User, p.user.id)
        u.pw_hash = hash_password(body.new_password)
        u.must_change_pw = False
    delete_user_sessions(ctx.db, p.user.id, keep_token=request.cookies.get(COOKIE))
    audit(ctx.db, p.user.id, "password_changed", p.user.username)
    return {"ok": True}
