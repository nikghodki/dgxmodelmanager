"""Single sign-on (OpenID Connect): dgxmm/auth/oidc.py and dgxmm/api/sso.py against a fake IdP (httpx.MockTransport)."""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from dgxmm.auth.oidc import FlowStore, OidcClient, OidcError, map_role, pkce_pair, safe_return_to
from dgxmm.config import AuthSettings, OidcRoleMapping, OidcSettings, Settings
from dgxmm.db import AuditLog, SsoIdentity, User, UserSession, utcnow

ISSUER = "https://idp.example.com"
CLIENT_ID = "dgxmm"
SECRET = "client-secret-value"
CALLBACK = "/dgxmm/api/auth/oidc/callback"


def _jwk(key, kid: str, alg: str = "RS256") -> dict:
    if isinstance(key, rsa.RSAPrivateKey):
        d = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    else:
        d = jwt.algorithms.ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
    d.update(kid=kid, use="sig", alg=alg)
    return d


class FakeIdP:
    """Discovery, JWKS, token (checks PKCE S256, the client secret and the redirect URI) and userinfo."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "k1"
        self.codes: dict[str, dict] = {}
        self.down = False
        self.meta_extra: dict = {}
        self.jwks_hits = 0
        self.token_forms: list[dict] = []
        self.userinfo_claims: dict | None = None
        self.mint_overrides: dict = {}
        self.sign_with = None  # (key, alg, kid) to sign with something else

    def meta(self) -> dict:
        return {"issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize", "token_endpoint": f"{ISSUER}/token",
                "jwks_uri": f"{ISSUER}/jwks", "userinfo_endpoint": f"{ISSUER}/userinfo",
                "code_challenge_methods_supported": ["S256"], "id_token_signing_alg_values_supported": ["RS256"],
                "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
                "authorization_response_iss_parameter_supported": True, **self.meta_extra}

    def mint(self, nonce: str, claims: dict) -> str:
        now = int(time.time())
        body = {"iss": ISSUER, "aud": CLIENT_ID, "sub": "user-123", "iat": now, "exp": now + 300, "nonce": nonce,
                "email": "alice@example.com", "preferred_username": "alice", "groups": ["admins"], **claims,
                **self.mint_overrides}
        body = {k: v for k, v in body.items() if v is not None}
        key, alg, kid = self.sign_with or (self.key, "RS256", self.kid)
        return jwt.encode(body, key, algorithm=alg, headers={"kid": kid} if kid else None)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/.well-known/openid-configuration":
            return httpx.Response(200, json=self.meta())
        if path == "/jwks":
            self.jwks_hits += 1
            return httpx.Response(200, json={"keys": [_jwk(self.key, self.kid)]})
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_forms.append(form)
            auth = request.headers.get("authorization", "")
            if auth != "Basic " + base64.b64encode(f"{CLIENT_ID}:{SECRET}".encode()).decode():
                return httpx.Response(401, json={"error": "invalid_client"})
            grant = self.codes.pop(form.get("code", ""), None)
            if grant is None or form.get("redirect_uri") != grant["redirect_uri"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=")
            if challenge.decode() != grant["challenge"] or grant["method"] != "S256":
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "at-" + secrets.token_hex(8), "token_type": "Bearer",
                                             "id_token": self.mint(grant["nonce"], grant["claims"])})
        if path == "/userinfo":
            return httpx.Response(200, json=self.userinfo_claims or {"sub": "user-123"})
        return httpx.Response(404)


def oidc_settings(**kw) -> OidcSettings:
    base = dict(enabled=True, issuer=ISSUER, client_id=CLIENT_ID, client_secret_file="/nonexistent",
                button_label="Sign in with Example ID",
                role_mapping=OidcRoleMapping(admin=["admins"], viewer=["family"]))
    base.update(kw)
    return OidcSettings(**base)


@pytest.fixture
def idp():
    return FakeIdP()


@pytest.fixture
def sso(settings: Settings, ctx, idp):
    """A TestClient with OIDC on, talking to the fake IdP. ``sso.configure(**kw)`` changes the OIDC settings."""
    from dgxmm.app import create_app

    def configure(**kw):
        settings.auth = AuthSettings(oidc=oidc_settings(**kw))
        ctx.oidc = OidcClient(settings.auth.oidc, transport=httpx.MockTransport(idp.handler),
                              read_secret=lambda _p: SECRET)

    configure()
    with TestClient(create_app(ctx), base_url="https://dgx.example.com") as c:
        c.configure = configure
        yield c


def start(client, return_to: str | None = "/deployments") -> dict:
    url = "/dgxmm/api/auth/oidc/login" + (f"?return_to={return_to}" if return_to is not None else "")
    r = client.get(url, follow_redirects=False)
    assert r.status_code == 302, r.text
    loc = urlsplit(r.headers["location"])
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == f"{ISSUER}/authorize"
    return {k: v[0] for k, v in parse_qs(loc.query).items()}


def grant(idp: FakeIdP, q: dict, **claims) -> str:
    code = secrets.token_urlsafe(16)
    idp.codes[code] = {"nonce": q["nonce"], "challenge": q["code_challenge"], "method": q["code_challenge_method"],
                       "redirect_uri": q["redirect_uri"], "claims": claims}
    return code


def callback(client, q: dict, code: str, **extra):
    params = {"code": code, "state": q["state"], "iss": ISSUER, **extra}
    params = {k: v for k, v in params.items() if v is not None}
    return client.get(CALLBACK, params=params, follow_redirects=False)


def sign_in(client, idp, return_to="/deployments", **claims):
    q = start(client, return_to)
    return callback(client, q, grant(idp, q, **claims))


def error_of(r) -> str | None:
    assert r.status_code == 303, r.text
    loc = r.headers["location"]
    return parse_qs(urlsplit(loc).query).get("sso_error", [None])[0]


def actions(db) -> list[str]:
    with db.session() as s:
        return [a.action for a in s.scalars(select(AuditLog).order_by(AuditLog.id))]


# --- happy path ------------------------------------------------------------------------------------------------------

def test_config_endpoint(client, sso):
    r = sso.get("/dgxmm/api/auth/config").json()
    assert r == {"local_login": True, "oidc": {"label": "Sign in with Example ID", "auto_redirect": False,
                                               "login_path": "/dgxmm/api/auth/oidc/login".removeprefix("/dgxmm")}}


def test_config_endpoint_without_oidc(client):
    assert client.get("/dgxmm/api/auth/config").json() == {"local_login": True, "oidc": None}


def test_sign_in_provisions_an_admin_and_creates_a_normal_session(sso, idp, db, caplog):
    caplog.set_level(logging.DEBUG)
    q = start(sso)
    assert q["response_type"] == "code" and q["client_id"] == CLIENT_ID
    assert q["redirect_uri"] == "https://dgx.example.com" + CALLBACK
    assert q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 43
    assert q["scope"] == "openid email profile groups"
    code = grant(idp, q)
    r = callback(sso, q, code)
    assert r.status_code == 303 and r.headers["location"] == "/dgxmm/deployments"
    cookie = next(h for h in r.headers.get_list("set-cookie") if h.startswith("dgxmm_session="))
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Secure" in cookie and "Path=/dgxmm" in cookie
    me = sso.get("/dgxmm/api/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["user"]["username"] == "alice" and body["user"]["role"] == "admin"
    assert body["user"]["auth_source"] == "sso" and body["user"]["must_change_pw"] is False
    assert body["sso"] == {"email": "alice@example.com"}
    # the CSRF token works like a password session's
    csrf = body["csrf_token"]
    assert sso.post("/dgxmm/api/users", json={"username": "bob", "password": "x" * 12, "role": "viewer"}).status_code == 403
    assert sso.post("/dgxmm/api/users", headers={"X-CSRF-Token": csrf},
                    json={"username": "bob", "password": "x" * 12, "role": "viewer"}).status_code == 201
    with db.session() as s:
        ident = s.scalar(select(SsoIdentity))
        assert (ident.issuer, ident.subject, ident.email) == (ISSUER, "user-123", "alice@example.com")
    assert actions(db)[:2] == ["sso_user_provisioned", "sso_login"]
    # PKCE: the verifier sent with the code hashes to the challenge (the fake IdP checks it), secret via Basic
    form = idp.token_forms[-1]
    assert "client_secret" not in form and form["grant_type"] == "authorization_code"
    # nothing secret in DGXMM's logs or the audit log (the test client's own request log is not DGXMM's)
    with db.session() as s:
        details = " ".join(str(a.detail) + a.target for a in s.scalars(select(AuditLog)))
    logs = " ".join(r.getMessage() for r in caplog.records if not r.name.startswith("httpx"))
    for secret in (code, q["state"], q["nonce"], form["code_verifier"], SECRET):
        assert secret not in logs and secret not in details


def test_access_log_drops_the_callback_query(sso):
    from dgxmm.auth.oidc import RedactQuery
    f = next(f for f in logging.getLogger("uvicorn.access").filters if isinstance(f, RedactQuery))
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', None, None)
    rec.args = ("192.0.2.1:1234", "GET", CALLBACK + "?code=secret-code&state=s", "1.1", 303)
    f.filter(rec)
    assert "secret-code" not in rec.getMessage() and CALLBACK + "?<redacted>" in rec.getMessage()
    rec.args = ("192.0.2.1:1234", "GET", "/dgxmm/api/models?q=1", "1.1", 200)
    f.filter(rec)
    assert "/dgxmm/api/models?q=1" in rec.getMessage()


def test_second_sign_in_reuses_the_user(sso, idp, db):
    assert error_of(sign_in(sso, idp)) is None
    assert error_of(sign_in(sso, idp, email="alice@new.example.com")) is None
    with db.session() as s:
        assert s.query(User).filter(User.auth_source == "sso").count() == 1
        assert s.scalar(select(SsoIdentity)).email == "alice@new.example.com"


def test_viewer_group_and_role_follow_the_idp_at_every_sign_in(sso, idp, db):
    assert error_of(sign_in(sso, idp, groups=["family"])) is None
    assert sso.get("/dgxmm/api/auth/me").json()["user"]["role"] == "viewer"
    assert error_of(sign_in(sso, idp, groups=["family", "admins"])) is None
    assert sso.get("/dgxmm/api/auth/me").json()["user"]["role"] == "admin"
    assert "sso_role_changed" in actions(db)


def test_user_in_no_mapped_group_is_denied_and_not_created(sso, idp, db):
    assert error_of(sign_in(sso, idp, groups=["other"])) == "denied"
    assert sso.get("/dgxmm/api/auth/me").status_code == 401
    with db.session() as s:
        assert s.query(User).count() == 0
    assert actions(db) == ["sso_login_denied"]


def test_removed_group_ends_access_at_the_next_sign_in(sso, idp, db):
    assert error_of(sign_in(sso, idp)) is None
    other = TestClient(sso.app, base_url="https://dgx.example.com")
    assert error_of(sign_in(other, idp)) is None          # a second session of the same user
    assert error_of(sign_in(sso, idp, groups=[])) == "denied"
    assert other.get("/dgxmm/api/auth/me").status_code == 401  # its sessions were ended


def test_missing_groups_claim_is_denied(sso, idp):
    assert error_of(sign_in(sso, idp, groups=None)) == "denied"


def test_groups_from_userinfo_when_the_id_token_has_none(sso, idp):
    idp.userinfo_claims = {"sub": "user-123", "groups": ["family"]}
    assert error_of(sign_in(sso, idp, groups=None)) is None
    assert sso.get("/dgxmm/api/auth/me").json()["user"]["role"] == "viewer"


def test_userinfo_for_another_subject_is_refused(sso, idp):
    idp.userinfo_claims = {"sub": "someone-else", "groups": ["admins"]}
    assert error_of(sign_in(sso, idp, groups=None)) == "invalid_token"


def test_auto_provision_off_refuses_unknown_identities(sso, idp, db):
    sso.configure(auto_provision=False)
    assert error_of(sign_in(sso, idp)) == "not_provisioned"
    sso.configure()
    assert error_of(sign_in(sso, idp)) is None
    sso.configure(auto_provision=False)
    assert error_of(sign_in(sso, idp)) is None  # existing SSO users still sign in


def test_never_links_to_an_existing_local_account(sso, idp, make_user, db):
    make_user("alice", "local-password-123", "viewer")
    assert error_of(sign_in(sso, idp)) is None
    me = sso.get("/dgxmm/api/auth/me").json()["user"]
    assert me["username"] == "alice-2" and me["auth_source"] == "sso" and me["role"] == "admin"


@pytest.mark.parametrize("claims, expected", [
    ({"preferred_username": "Nik.Example"}, "nik.example"),
    ({"preferred_username": None, "email": "carol+x@example.com"}, "carol-x"),
    ({"preferred_username": "a", "email": None, "name": "Dee Dee"}, "dee-dee"),
    ({"preferred_username": None, "email": None, "name": None}, "sso-user"),
])
def test_usernames_are_derived_and_sanitised(sso, idp, claims, expected):
    assert error_of(sign_in(sso, idp, **claims)) is None
    assert sso.get("/dgxmm/api/auth/me").json()["user"]["username"] == expected


# --- flow integrity --------------------------------------------------------------------------------------------------

def test_state_cannot_be_replayed(sso, idp):
    q = start(sso)
    code = grant(idp, q)
    assert error_of(callback(sso, q, code)) is None
    assert error_of(callback(sso, q, grant(idp, q))) == "invalid_state"


def test_unknown_state(sso, idp):
    q = start(sso)
    assert error_of(callback(sso, {**q, "state": "forged"}, grant(idp, q))) == "invalid_state"


def test_callback_from_another_browser_is_refused(sso, idp):
    """Login CSRF: an attacker's own callback URL opened in the victim's browser lacks the binding cookie."""
    q = start(sso)
    victim = TestClient(sso.app, base_url="https://dgx.example.com")
    assert error_of(callback(victim, q, grant(idp, q))) == "invalid_state"
    assert victim.get("/dgxmm/api/auth/me").status_code == 401


def test_binding_cookie_is_lax_short_lived_and_scoped_to_the_sign_in_routes(sso, idp):
    r = sso.get("/dgxmm/api/auth/oidc/login", follow_redirects=False)
    cookie = next(h for h in r.headers.get_list("set-cookie") if h.startswith("dgxmm_oidc="))
    assert "SameSite=lax" in cookie and "HttpOnly" in cookie and "Path=/dgxmm/api/auth/oidc;" in cookie
    assert "Secure" in cookie
    assert r.headers["cache-control"] == "no-store"
    assert "Max-Age=600" in cookie


def test_parallel_sign_ins_in_one_browser_both_work(sso, idp):
    q1, q2 = start(sso), start(sso)
    assert error_of(callback(sso, q2, grant(idp, q2))) is None
    assert error_of(callback(sso, q1, grant(idp, q1))) is None


def test_iss_parameter_must_match(sso, idp):
    q = start(sso)
    assert error_of(callback(sso, q, grant(idp, q), iss="https://evil.example.com")) == "invalid_state"


def test_iss_parameter_required_when_the_idp_announces_it(sso, idp):
    q = start(sso)
    assert error_of(callback(sso, q, grant(idp, q), iss=None)) == "invalid_state"
    idp.meta_extra = {"authorization_response_iss_parameter_supported": False}
    sso.configure()
    q = start(sso)
    assert error_of(callback(sso, q, grant(idp, q), iss=None)) is None


def test_idp_error_response(sso, idp):
    q = start(sso)
    r = sso.get(CALLBACK, params={"error": "access_denied", "state": q["state"], "iss": ISSUER},
                follow_redirects=False)
    assert error_of(r) == "idp_denied"


def test_wrong_code_is_refused_by_the_idp(sso, idp):
    q = start(sso)
    grant(idp, q)
    assert error_of(callback(sso, q, "not-the-code")) == "idp_error"


# --- ID token validation ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {"exp": int(time.time()) - 3600},                        # expired
    {"aud": "another-client"},                               # wrong audience
    {"iss": "https://evil.example.com"},                     # wrong issuer
    {"nonce": "not-the-nonce"},                              # wrong nonce
    {"nonce": None},                                         # no nonce
    {"iat": int(time.time()) - 3600, "exp": int(time.time()) + 300},  # issued too long ago
    {"iat": int(time.time()) + 3600, "exp": int(time.time()) + 7200},  # issued in the future
    {"aud": [CLIENT_ID, "other"]},                           # several audiences, no azp
    {"azp": "other"},                                        # authorized party is someone else
    {"sub": None},                                           # no subject
])
def test_invalid_id_tokens_are_refused(sso, idp, db, overrides):
    idp.mint_overrides = overrides
    assert error_of(sign_in(sso, idp)) == "invalid_token"
    with db.session() as s:
        assert s.query(User).count() == 0
    assert "sso_login_failed" in actions(db)


def test_several_audiences_with_our_azp_are_accepted(sso, idp):
    idp.mint_overrides = {"aud": [CLIENT_ID, "other"], "azp": CLIENT_ID}
    assert error_of(sign_in(sso, idp)) is None


def test_token_signed_by_another_key_is_refused(sso, idp):
    idp.sign_with = (rsa.generate_private_key(public_exponent=65537, key_size=2048), "RS256", idp.kid)
    assert error_of(sign_in(sso, idp)) == "invalid_token"


@pytest.mark.parametrize("alg", ["none", "HS256"])
def test_unsigned_and_hmac_tokens_are_refused(sso, idp, alg):
    # HS256 "signed" with the public key's bytes is the classic algorithm-confusion attack
    key = None if alg == "none" else _jwk(idp.key, idp.kid)["n"].encode()
    idp.sign_with = (key, alg, idp.kid)
    assert error_of(sign_in(sso, idp)) == "invalid_token"


def test_key_of_the_wrong_type_for_the_algorithm_is_refused(sso, idp):
    ec_key = ec.generate_private_key(ec.SECP256R1())
    idp.sign_with = (ec_key, "ES256", idp.kid)  # the kid names the RSA key
    assert error_of(sign_in(sso, idp)) == "invalid_token"


def test_signing_key_rotation_refetches_the_jwks(sso, idp):
    assert error_of(sign_in(sso, idp)) is None
    assert idp.jwks_hits == 1
    assert error_of(sign_in(sso, idp)) is None
    assert idp.jwks_hits == 1  # cached
    idp.key, idp.kid = rsa.generate_private_key(public_exponent=65537, key_size=2048), "k2"
    sso.app.state.ctx.oidc._keys_fetched_at -= 120  # the refetch throttle has passed
    assert error_of(sign_in(sso, idp)) is None
    assert idp.jwks_hits == 2


def test_unknown_kid_does_not_hammer_the_idp(sso, idp):
    assert error_of(sign_in(sso, idp)) is None
    idp.sign_with = (idp.key, "RS256", "unknown-kid")
    assert error_of(sign_in(sso, idp)) == "invalid_token"
    assert error_of(sign_in(sso, idp)) == "invalid_token"
    assert idp.jwks_hits == 1  # within JWKS_REFRESH_MIN_S of the last fetch: no refetch


# --- the IdP as a trust boundary -------------------------------------------------------------------------------------

def test_idp_down_shows_a_readable_error(sso, idp, db):
    idp.down = True
    r = sso.get("/dgxmm/api/auth/oidc/login", follow_redirects=False)
    assert error_of(r) == "idp_unavailable"
    assert len(sso.app.state.ctx.oidc_flows or []) == 0


def test_idp_down_during_the_callback(sso, idp):
    q = start(sso)
    code = grant(idp, q)
    idp.down = True
    assert error_of(callback(sso, q, code)) == "idp_unavailable"


def test_discovery_must_name_the_configured_issuer(sso, idp):
    idp.meta_extra = {"issuer": "https://other.example.com"}
    assert error_of(sso.get("/dgxmm/api/auth/oidc/login", follow_redirects=False)) == "idp_error"


@pytest.mark.parametrize("extra", [{"token_endpoint": "http://idp.example.com/token"},
                                   {"jwks_uri": None}, {"code_challenge_methods_supported": ["plain"]}])
def test_discovery_endpoints_must_be_https_and_support_s256(sso, idp, extra):
    idp.meta_extra = extra
    assert error_of(sso.get("/dgxmm/api/auth/oidc/login", follow_redirects=False)) == "idp_error"


def test_disabled_oidc_routes_redirect_to_login(client):
    assert error_of(client.get("/dgxmm/api/auth/oidc/login", follow_redirects=False)) == "sso_disabled"
    assert error_of(client.get(CALLBACK, follow_redirects=False)) == "sso_disabled"


def test_sign_in_starts_are_rate_limited_per_ip(sso, idp, db):
    codes = [sso.get("/dgxmm/api/auth/oidc/login", follow_redirects=False).status_code for _ in range(21)]
    assert codes[:20] == [302] * 20
    assert error_of(sso.get("/dgxmm/api/auth/oidc/login", follow_redirects=False)) == "rate_limited"
    assert "sso_rate_limited" in actions(db)


def test_public_url_sets_the_redirect_uri(sso, idp):
    sso.configure(public_url="https://models.example.com")
    assert start(sso)["redirect_uri"] == "https://models.example.com" + CALLBACK


def test_custom_redirect_path(settings, ctx, idp):
    from dgxmm.app import create_app
    settings.auth = AuthSettings(oidc=oidc_settings(redirect_path="/sso/callback"))
    ctx.oidc = OidcClient(settings.auth.oidc, transport=httpx.MockTransport(idp.handler), read_secret=lambda _p: SECRET)
    with TestClient(create_app(ctx), base_url="https://dgx.example.com") as c:
        q = start(c)
        assert q["redirect_uri"] == "https://dgx.example.com/dgxmm/sso/callback"
        r = c.get("/dgxmm/sso/callback", params={"code": grant(idp, q), "state": q["state"], "iss": ISSUER},
                  follow_redirects=False)
        assert error_of(r) is None


def test_public_client_sends_no_secret(settings, ctx, idp):
    from dgxmm.app import create_app
    settings.auth = AuthSettings(oidc=oidc_settings())
    ctx.oidc = OidcClient(settings.auth.oidc, transport=httpx.MockTransport(idp.handler), read_secret=lambda _p: None)
    with TestClient(create_app(ctx), base_url="https://dgx.example.com") as c:
        q = start(c)
        assert error_of(callback(c, q, grant(idp, q))) == "idp_error"  # the fake IdP insists on its secret
    form = idp.token_forms[-1]
    assert form["client_id"] == CLIENT_ID and "client_secret" not in form


# --- return_to -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("/deployments", "/deployments"),
    ("/deployments/3?tab=logs", "/deployments/3?tab=logs"),
    ("/dgxmm/models", "/models"),
    ("/dgxmm", "/"),
    (None, "/"), ("", "/"),
    ("//evil.example.com", "/"),
    ("https://evil.example.com/", "/"),
    ("/\\evil.example.com", "/"),
    ("/%2f%2fevil.example.com", "/"),
    ("/%5cevil", "/"),
    ("/a/../../x", "/"),
    ("/./x", "/"),
    ("/x%2e%2e/", "/"),
    ("/login", "/"),
    ("/api/users", "/"),
    ("/ondemand/v1/models", "/"),
    ("/x\r\nSet-Cookie: a=b", "/"),
    ("javascript:alert(1)", "/"),
    ("/" + "a" * 600, "/"),
])
def test_return_to_is_a_same_origin_ui_path(value, expected):
    assert safe_return_to(value, "/dgxmm") == expected


def test_callback_redirects_to_the_sanitised_return_to(sso, idp):
    r = sign_in(sso, idp, return_to="//evil.example.com")
    assert r.headers["location"] == "/dgxmm/"


# --- SSO users and the rest of auth ----------------------------------------------------------------------------------

def test_sso_users_have_no_password(sso, idp, client):
    assert error_of(sign_in(sso, idp)) is None
    me = sso.get("/dgxmm/api/auth/me").json()
    r = sso.post("/dgxmm/api/auth/change-password", headers={"X-CSRF-Token": me["csrf_token"]},
                 json={"current_password": "!sso", "new_password": "a-long-new-password"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "sso_managed"
    for pw in ("!sso", "", "anything-at-all"):
        assert client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": pw}).status_code in (401, 429)


def test_admins_cannot_change_role_or_password_of_sso_users_but_can_delete_them(sso, idp, make_user, db):
    assert error_of(sign_in(sso, idp, groups=["family"])) is None
    make_user("root", "root-password-123", "admin")
    admin = TestClient(sso.app, base_url="https://dgx.example.com")
    csrf = admin.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"}).json()["csrf_token"]
    h = {"X-CSRF-Token": csrf}
    users = admin.get("/dgxmm/api/users").json()
    alice = next(u for u in users if u["username"] == "alice")
    assert alice["auth_source"] == "sso" and alice["email"] == "alice@example.com"
    assert next(u for u in users if u["username"] == "root")["auth_source"] == "local"
    for body in ({"role": "admin"}, {"password": "a-long-new-password"}):
        r = admin.patch(f"/dgxmm/api/users/{alice['id']}", headers=h, json=body)
        assert r.status_code == 409 and r.json()["detail"]["code"] == "sso_managed"
    assert admin.delete(f"/dgxmm/api/users/{alice['id']}", headers=h).status_code == 204
    assert sso.get("/dgxmm/api/auth/me").status_code == 401
    with db.session() as s:
        assert s.scalar(select(SsoIdentity)) is None  # cascade


def test_sso_session_ends_after_reauth_after_s(sso, idp, db):
    assert error_of(sign_in(sso, idp)) is None
    with db.session() as s:
        s.execute(update(UserSession).values(created_at=utcnow() - timedelta(hours=13)))
    assert sso.get("/dgxmm/api/auth/me").status_code == 401
    with db.session() as s:
        assert s.query(UserSession).count() == 0


def test_sso_sessions_end_when_oidc_is_turned_off(sso, idp, settings):
    assert error_of(sign_in(sso, idp)) is None
    settings.auth = AuthSettings()
    assert sso.get("/dgxmm/api/auth/me").status_code == 401


def test_local_login_can_be_turned_off_only_with_oidc_on(sso, idp, client, make_user, settings):
    make_user("root", "root-password-123", "admin")
    sso.configure(local_login=False)
    assert sso.get("/dgxmm/api/auth/config").json()["local_login"] is False
    r = sso.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "local_login_disabled"
    settings.auth = AuthSettings(oidc=OidcSettings(enabled=False, local_login=False))
    assert settings.auth.local_login is True
    assert sso.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"}).status_code == 200


def test_local_admin_still_signs_in_with_oidc_on(sso, make_user):
    make_user("admin", "break-glass-pass-1", "admin")
    r = sso.post("/dgxmm/api/auth/login", json={"username": "admin", "password": "break-glass-pass-1"})
    assert r.status_code == 200 and r.json()["user"]["auth_source"] == "local"


# --- units -----------------------------------------------------------------------------------------------------------

def test_pkce_pair_is_s256():
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert pkce_pair()[0] != verifier


def test_flow_store_is_single_use_expiring_and_bounded():
    now = [0.0]
    store = FlowStore(ttl_s=10, max_pending=2, clock=lambda: now[0])
    kw = dict(nonce="n", verifier="v", redirect_uri="r", return_to="/", binding="b")
    s1 = store.add(**kw)
    store.add(**kw)
    with pytest.raises(OidcError) as e:
        store.add(**kw)
    assert e.value.code == "busy"
    assert store.pop(s1) is not None and store.pop(s1) is None
    s3 = store.add(**kw)
    now[0] = 11
    assert store.pop(s3) is None and len(store) == 0


def test_map_role():
    cfg = oidc_settings(role_mapping=OidcRoleMapping(claim="realm_access.roles", admin=["ops"], viewer=["dev"]))
    assert map_role({"realm_access": {"roles": ["dev", "ops"]}}, cfg) == "admin"
    assert map_role({"realm_access": {"roles": ["dev"]}}, cfg) == "viewer"
    assert map_role({"realm_access": {"roles": "dev"}}, cfg) == "viewer"
    assert map_role({"realm_access": {"roles": [{"x": 1}]}}, cfg) is None
    assert map_role({"groups": ["ops"]}, cfg) is None


@pytest.mark.parametrize("kw, message", [
    ({"issuer": "http://idp.example.com"}, "https"),
    ({"issuer": ""}, "https"),
    ({"client_id": ""}, "client_id"),
    ({"role_mapping": OidcRoleMapping()}, "role_mapping"),
    ({"scopes": ["email"]}, "openid"),
    ({"redirect_path": "callback"}, "redirect_path"),
    ({"redirect_path": "/a/../b"}, "redirect_path"),
    ({"public_url": "https://x.example.com/dgxmm"}, "origin"),
])
def test_config_validation(kw, message):
    with pytest.raises(ValueError, match=message):
        oidc_settings(**kw)


def test_disabled_oidc_needs_no_settings():
    assert OidcSettings().enabled is False


def test_config_from_yaml(tmp_path):
    from dgxmm.config import load_settings
    p = tmp_path / "c.yaml"
    p.write_text(f"""auth:
  oidc:
    enabled: true
    issuer: {ISSUER}
    client_id: dgxmm
    button_label: Sign in with Example ID
    auto_redirect: true
    role_mapping:
      admin: [admins]
""")
    s = load_settings(str(p))
    assert s.auth.oidc.enabled and s.auth.oidc.auto_redirect and s.auth.oidc.role_mapping.claim == "groups"
    assert s.auth.oidc.scopes == ["openid", "email", "profile", "groups"]
