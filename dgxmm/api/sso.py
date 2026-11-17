"""Single sign-on with OpenID Connect (dgxmm/auth/oidc.py; docs/user-guide/single-sign-on.md).

``GET /api/auth/config`` (public) tells the login page what to offer. ``GET /api/auth/oidc/login`` starts a sign-in
and ``GET <auth.oidc.redirect_path>`` is the callback; both answer with redirects only, so every failure lands on
the login page as ``?sso_error=<code>`` (never with IdP text in it). A successful callback creates an ordinary DGXMM
session (dgxmm/api/auth.py: start_session): same cookie, flags, TTL and CSRF token as a password sign-in."""
from __future__ import annotations

import logging
import re
import secrets

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dgxmm.api.auth import secure_cookie, start_session
from dgxmm.audit import audit
from dgxmm.auth.deps import get_ctx
from dgxmm.auth.oidc import (FLOW_TTL_S, FlowStore, OidcClient, OidcError, digest, has_claim, map_role, pkce_pair,
                             safe_return_to)
from dgxmm.auth.sessions import delete_user_sessions
from dgxmm.context import AppContext
from dgxmm.db import SsoIdentity, User, utcnow

log = logging.getLogger("dgxmm.sso")
router = APIRouter(prefix="/api/auth", tags=["auth"])
BINDING_COOKIE = "dgxmm_oidc"
_BINDING = re.compile(r"^[A-Za-z0-9_-]{43}$")
SSO_PW_HASH = "!sso"  # never a valid argon2 hash: SSO users have no DGXMM password
USERNAME_MAX = 32


def oidc_client(ctx: AppContext) -> OidcClient:
    if ctx.oidc is None:
        ctx.oidc = OidcClient(ctx.settings.auth.oidc)
    return ctx.oidc


def oidc_flows(ctx: AppContext) -> FlowStore:
    if ctx.oidc_flows is None:
        ctx.oidc_flows = FlowStore()
    return ctx.oidc_flows


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _to_login(ctx: AppContext, code: str) -> RedirectResponse:
    resp = RedirectResponse(f"{ctx.settings.base_path}/login?sso_error={code}", status_code=303)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _redirect_uri(request: Request, ctx: AppContext) -> str:
    cfg = ctx.settings.auth.oidc
    # without public_url: the scheme as the trusted proxy reported it (X-Forwarded-Proto) and the Host header. A
    # forged Host only yields a redirect URI the IdP has not registered, so the sign-in fails there.
    origin = cfg.public_url or f"{request.url.scheme}://{request.url.netloc}"
    return origin + ctx.settings.base_path + cfg.redirect_path


def _binding_path(ctx: AppContext) -> str:
    """The cookie path that covers both the sign-in start and the callback (their longest common prefix)."""
    bp = ctx.settings.base_path
    start = (bp + "/api/auth/oidc/login").split("/")
    cb = (bp + ctx.settings.auth.oidc.redirect_path).split("/")
    common = []
    for a, b in zip(start, cb):
        if a != b:
            break
        common.append(a)
    return "/".join(common) or "/"


@router.get("/config")
def auth_config(ctx: AppContext = Depends(get_ctx)):
    """What the login page offers (no secrets): the password form, and the SSO button with its label."""
    cfg = ctx.settings.auth.oidc
    oidc = {"label": cfg.button_label, "auto_redirect": cfg.auto_redirect,
            "login_path": "/api/auth/oidc/login"} if cfg.enabled else None
    return {"local_login": ctx.settings.auth.local_login, "oidc": oidc}


@router.get("/oidc/login", include_in_schema=False)
def oidc_login(request: Request, return_to: str | None = None, ctx: AppContext = Depends(get_ctx)):
    cfg = ctx.settings.auth.oidc
    if not cfg.enabled:
        return _to_login(ctx, "sso_disabled")
    ip = _ip(request)
    if not ctx.sso_limiter.hit(f"ip:{ip}"):
        audit(ctx.db, None, "sso_rate_limited", "", {"ip": ip})
        return _to_login(ctx, "rate_limited")
    client = oidc_client(ctx)
    binding = request.cookies.get(BINDING_COOKIE) or ""
    if not _BINDING.fullmatch(binding):  # one per browser, so parallel tabs do not invalidate each other
        binding = secrets.token_urlsafe(32)
    verifier, challenge = pkce_pair()
    nonce = secrets.token_urlsafe(32)
    redirect_uri = _redirect_uri(request, ctx)
    try:
        client.metadata()  # fails before a flow is stored when the IdP is down
        state = oidc_flows(ctx).add(nonce=nonce, verifier=verifier, redirect_uri=redirect_uri,
                                    return_to=safe_return_to(return_to, ctx.settings.base_path), binding=binding)
        url = client.authorization_url(redirect_uri=redirect_uri, state=state, nonce=nonce, challenge=challenge)
    except OidcError as e:
        log.warning("OIDC sign-in could not start: %s", e)
        return _to_login(ctx, e.code)
    resp = RedirectResponse(url, status_code=302)
    # SameSite=Lax: it must come back on the IdP's top-level redirect to the callback (Strict would not)
    resp.set_cookie(BINDING_COOKIE, binding, max_age=FLOW_TTL_S, httponly=True, samesite="lax",
                    secure=secure_cookie(request, ctx), path=_binding_path(ctx))
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


def oidc_callback(request: Request, ctx: AppContext = Depends(get_ctx)):
    """The redirect URI (mounted at base_path + auth.oidc.redirect_path by dgxmm/app.py)."""
    resp = _callback(request, ctx)
    # the binding cookie is left to expire (FLOW_TTL_S): another tab's sign-in may still need it
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


def _failed(ctx: AppContext, ip: str, code: str, why: str = "") -> RedirectResponse:
    audit(ctx.db, None, "sso_login_failed", "", {"ip": ip, "reason": code})
    if why:
        log.warning("OIDC sign-in failed (%s): %s", code, why)
    return _to_login(ctx, code)


def _callback(request: Request, ctx: AppContext) -> RedirectResponse:
    cfg = ctx.settings.auth.oidc
    if not cfg.enabled:
        return _to_login(ctx, "sso_disabled")
    ip = _ip(request)
    if not ctx.sso_limiter.hit(f"ip:{ip}"):
        audit(ctx.db, None, "sso_rate_limited", "", {"ip": ip})
        return _to_login(ctx, "rate_limited")
    q = request.query_params
    flow = oidc_flows(ctx).pop(q.get("state", ""))
    if flow is None:
        return _failed(ctx, ip, "invalid_state", "unknown, expired or already used state")
    binding = request.cookies.get(BINDING_COOKIE) or ""
    if not secrets.compare_digest(digest(binding), flow.binding):
        return _failed(ctx, ip, "invalid_state", "the sign-in was started in another browser")
    if q.get("error"):
        code = "idp_denied" if q.get("error") == "access_denied" else "idp_error"
        return _failed(ctx, ip, code, "the identity provider returned an error")
    auth_code = q.get("code", "")
    if not auth_code or len(auth_code) > 4096:
        return _failed(ctx, ip, "invalid_state", "no authorization code")
    client = oidc_client(ctx)
    try:
        client.check_iss_param(q.get("iss"))
        tokens = client.exchange(code=auth_code, redirect_uri=flow.redirect_uri, verifier=flow.verifier)
        claims = client.validate_id_token(tokens["id_token"], nonce=flow.nonce)
        if not has_claim(claims, cfg.role_mapping.claim):
            # some IdPs put groups only into userinfo; the ID token's own claims win
            claims = {**client.userinfo(tokens.get("access_token"), claims["sub"]), **claims}
    except OidcError as e:
        return _failed(ctx, ip, e.code, str(e))
    email = claims.get("email") if isinstance(claims.get("email"), str) else ""
    email = email[:320]
    who = email or claims["sub"]
    role = map_role(claims, cfg)
    if role is None:
        # also end the sessions of an existing SSO user who lost the group
        with ctx.db.session() as s:
            ident = s.scalar(select(SsoIdentity).where(SsoIdentity.issuer == cfg.issuer,
                                                       SsoIdentity.subject == claims["sub"]))
            user_id = ident.user_id if ident else None
        if user_id is not None:
            delete_user_sessions(ctx.db, user_id)
        audit(ctx.db, user_id, "sso_login_denied", who, {"ip": ip, "reason": "no_role",
                                                         "claim": cfg.role_mapping.claim})
        log.warning("OIDC sign-in refused: no %s value maps to a DGXMM role", cfg.role_mapping.claim)
        return _to_login(ctx, "denied")
    try:
        user, created, old_role = _provision(ctx, claims, role, email)
    except OidcError as e:
        audit(ctx.db, None, "sso_login_denied", who, {"ip": ip, "reason": e.code})
        return _to_login(ctx, e.code)
    resp = RedirectResponse(ctx.settings.base_path + flow.return_to, status_code=303)
    start_session(ctx, request, resp, user.id, ip)
    if created:
        audit(ctx.db, user.id, "sso_user_provisioned", user.username, {"role": role, "email": email})
    elif old_role != role:
        audit(ctx.db, user.id, "sso_role_changed", user.username, {"from": old_role, "to": role})
    audit(ctx.db, user.id, "sso_login", user.username, {"ip": ip, "role": role, "email": email})
    return resp


def _provision(ctx: AppContext, claims: dict, role: str, email: str) -> tuple[User, bool, str | None]:
    """The DGXMM user of this identity (issuer + sub), with the role from this sign-in; created when allowed.
    Returns (user, created, previous role). Never links to an existing account by username or email."""
    cfg = ctx.settings.auth.oidc
    sub = claims["sub"]
    for attempt in range(2):  # a concurrent first sign-in of the same identity wins the unique constraint
        try:
            with ctx.db.session() as s:
                ident = s.scalar(select(SsoIdentity).where(SsoIdentity.issuer == cfg.issuer,
                                                           SsoIdentity.subject == sub))
                if ident is not None:
                    user = s.get(User, ident.user_id)
                    old = user.role
                    user.role = role
                    ident.email, ident.last_login_at = email, utcnow()
                    s.flush()
                    return user, False, old
                if not cfg.auto_provision:
                    raise OidcError("not_provisioned", "this identity has no DGXMM user and auto_provision is off")
                user = User(username=_unique_username(s, claims), pw_hash=SSO_PW_HASH, role=role,
                            must_change_pw=False, auth_source="sso")
                s.add(user)
                s.flush()
                s.add(SsoIdentity(user_id=user.id, issuer=cfg.issuer, subject=sub, email=email,
                                  last_login_at=utcnow()))
                s.flush()
                return user, True, None
        except IntegrityError:
            if attempt:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _username_base(claims: dict) -> str:
    for key in ("preferred_username", "email", "name"):
        v = claims.get(key)
        if isinstance(v, str) and v.strip():
            v = v.strip().lower()
            if key == "email":
                v = v.split("@", 1)[0]
            v = re.sub(r"[^a-z0-9_.-]+", "-", v).strip("-._")[:USERNAME_MAX - 3]
            if len(v) >= 3:
                return v
    return "sso-user"


def _unique_username(s, claims: dict) -> str:
    base = _username_base(claims)
    taken = set(s.scalars(select(User.username).where(User.username.like(f"{base}%"))))
    if base not in taken:
        return base
    for i in range(2, 1000):
        cand = f"{base[:USERNAME_MAX - len(str(i)) - 1]}-{i}"
        if cand not in taken:
            return cand
    return f"sso-{secrets.token_hex(6)}"
