"""OpenID Connect sign-in: Authorization Code flow with PKCE (S256), for an external identity provider (IdP).

The flow (dgxmm/api/sso.py has the routes):

1. ``GET <base>/api/auth/oidc/login`` creates a pending flow: a random ``state`` (the key), ``nonce`` and PKCE
   ``code_verifier``, the redirect URI and the sanitised ``return_to``. Flows live in this process's memory only
   (``FlowStore``: at most ``MAX_PENDING``, ``FLOW_TTL_S``, single use) and are bound to the browser that started
   them by a random cookie whose SHA-256 the flow keeps, so a callback URL carrying someone else's code cannot sign
   a victim in (login CSRF). The browser goes to the IdP's authorization endpoint.
2. The callback pops the flow by ``state`` (a replayed or unknown state fails), checks the binding cookie and the
   RFC 9207 ``iss`` parameter, exchanges the code (with the verifier and, for a confidential client, the secret)
   and validates the ID token completely: signature against the IdP's JWKS (cached; an unknown ``kid`` refetches
   it, at most every ``JWKS_REFRESH_MIN_S``, so key rotation works), ``alg`` from an asymmetric allowlist (never
   ``none`` or HMAC), ``iss``, ``aud`` (and ``azp`` when there are several audiences), ``exp``, ``iat`` (not in the
   future, not older than ``MAX_TOKEN_AGE_S``) and ``nonce``.
3. The role comes from the configured claim at every sign-in (``map_role``); no matching value means no access.

Nothing here logs or audits tokens, codes, the state, the nonce or the verifier. The IdP is a trust boundary
(docs/security.md): its discovery document must name the configured issuer exactly, its endpoints must be https,
and responses are size-capped.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit

import httpx
import jwt

from dgxmm.config import OidcSettings

log = logging.getLogger("dgxmm.oidc")

ALLOWED_ALGS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"})
KEY_TYPE = {"RS": "RSA", "PS": "RSA", "ES": "EC", "Ed": "OKP"}  # alg prefix -> the JWK kty it needs
FLOW_TTL_S = 600            # a sign-in at the IdP must finish within this
MAX_PENDING = 512           # pending flows held in memory at once
METADATA_TTL_S = 3600       # discovery document cache
JWKS_TTL_S = 3600           # signing keys cache
JWKS_REFRESH_MIN_S = 60     # an unknown kid refetches the keys at most this often
LEEWAY_S = 60               # clock skew allowed on exp / iat
MAX_TOKEN_AGE_S = 600       # an ID token issued longer ago than this is refused (with the leeway)
MAX_RESPONSE_BYTES = 1 << 20


class OidcError(Exception):
    """A sign-in failure with a stable code (shown on the login page) and a log-safe message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def pkce_pair() -> tuple[str, str]:
    """(code_verifier, S256 code_challenge) per RFC 7636: 43 characters of 256-bit randomness."""
    verifier = _b64url(secrets.token_bytes(32))
    return verifier, _b64url(hashlib.sha256(verifier.encode()).digest())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# --- return_to -------------------------------------------------------------------------------------------------------

_RETURN_TO = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*(\?[A-Za-z0-9._~!$&'()*+,;=:@%/?-]*)?$")
_ENCODED_SEPARATORS = re.compile(r"%(2f|5c|2e|00|0d|0a|09)", re.I)


def safe_return_to(value: str | None, base_path: str = "") -> str:
    """The UI path to land on after sign-in: an absolute path inside the UI (``/deployments/3``), else ``/``.

    Only same-origin relative paths survive: no scheme, host, ``//``, backslash, dot segments, encoded separators or
    control characters, and nothing outside the UI (``/api/...``, the login page itself). A value that already
    starts with the base path is accepted without it. The caller prefixes the base path, so the redirect can only
    ever go to ``<base_path>/...`` on this origin."""
    if not value or len(value) > 512:
        return "/"
    if base_path and (value == base_path or value.startswith(base_path + "/")):
        value = value[len(base_path):] or "/"
    if not _RETURN_TO.fullmatch(value) or "//" in value or _ENCODED_SEPARATORS.search(value):
        return "/"
    path = value.split("?", 1)[0]
    segments = path.split("/")
    if any(seg in (".", "..") for seg in segments):
        return "/"
    if path == "/login" or path.startswith("/api/") or path == "/api" or path.startswith("/ondemand"):
        return "/"
    return value


# --- pending flows ---------------------------------------------------------------------------------------------------

@dataclass
class Flow:
    nonce: str
    verifier: str
    redirect_uri: str
    return_to: str
    binding: str  # sha256 of the browser-binding cookie
    expires: float


class FlowStore:
    """Pending sign-ins, keyed by sha256(state). Single use (``pop``), expiring, bounded."""

    def __init__(self, ttl_s: float = FLOW_TTL_S, max_pending: int = MAX_PENDING,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_s, self.max_pending, self.clock = ttl_s, max_pending, clock
        self._flows: OrderedDict[str, Flow] = OrderedDict()
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        for key in [k for k, f in self._flows.items() if f.expires <= now]:
            del self._flows[key]

    def add(self, *, nonce: str, verifier: str, redirect_uri: str, return_to: str, binding: str) -> str:
        state = secrets.token_urlsafe(32)
        now = self.clock()
        with self._lock:
            self._sweep(now)
            if len(self._flows) >= self.max_pending:
                raise OidcError("busy", "too many sign-ins in progress; try again in a few minutes")
            self._flows[digest(state)] = Flow(nonce, verifier, redirect_uri, return_to, digest(binding),
                                              now + self.ttl_s)
        return state

    def pop(self, state: str) -> Flow | None:
        if not state or len(state) > 256:
            return None
        now = self.clock()
        with self._lock:
            flow = self._flows.pop(digest(state), None)
            self._sweep(now)
        if flow is None or flow.expires <= now:
            return None
        return flow

    def __len__(self) -> int:
        return len(self._flows)


# --- logs ------------------------------------------------------------------------------------------------------------

class RedactQuery(logging.Filter):
    """Drops the query string of the given paths from uvicorn's access log lines: the callback's carries the
    authorization code and the state."""

    def __init__(self, paths: set[str]):
        super().__init__()
        self.paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str) and "?" in args[2]:
            path = args[2].split("?", 1)[0]
            if path in self.paths:
                record.args = (*args[:2], path + "?<redacted>", *args[3:])
        return True


def redact_access_log(path: str, logger_name: str = "uvicorn.access") -> None:
    """Install (once per process) the access-log filter for ``path``."""
    logger = logging.getLogger(logger_name)
    existing = next((f for f in logger.filters if isinstance(f, RedactQuery)), None)
    if existing is None:
        logger.addFilter(RedactQuery({path}))
    else:
        existing.paths.add(path)


# --- role mapping ----------------------------------------------------------------------------------------------------

def claim_values(claims: dict, path: str) -> list[str]:
    """The values of a claim (``groups``, or a dotted path such as ``realm_access.roles``) as strings. A list keeps
    its string items; a single string is one value."""
    cur: Any = claims
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return []
        cur = cur[part]
    if isinstance(cur, str):
        return [cur]
    if isinstance(cur, list):
        return [v for v in cur if isinstance(v, str)]
    return []


def has_claim(claims: dict, path: str) -> bool:
    cur: Any = claims
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def map_role(claims: dict, cfg: OidcSettings) -> str | None:
    """``admin`` / ``viewer`` from the configured claim, or None (no access). Admin wins when both match."""
    values = set(claim_values(claims, cfg.role_mapping.claim))
    if values & set(cfg.role_mapping.admin):
        return "admin"
    if values & set(cfg.role_mapping.viewer):
        return "viewer"
    return None


# --- the IdP ---------------------------------------------------------------------------------------------------------

class OidcClient:
    """Discovery, JWKS, authorization URL, code exchange, ID token validation and userinfo for one IdP."""

    def __init__(self, cfg: OidcSettings, *, transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.time, read_secret: Callable[[str], str | None] | None = None):
        self.cfg = cfg
        self.clock = clock
        self._transport = transport
        self._read_secret = read_secret or _read_secret_file
        self._lock = threading.Lock()
        self._meta: dict | None = None
        self._meta_at = 0.0
        self._keys: dict[str, tuple[jwt.PyJWK, dict]] = {}  # kid -> (key, its JWK)
        self._keys_at = 0.0
        self._keys_fetched_at = -1e18

    # HTTP ----------------------------------------------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        verify: Any = self.cfg.ca_bundle or True
        return httpx.Client(timeout=self.cfg.timeout_s, verify=verify, transport=self._transport,
                            follow_redirects=False, headers={"Accept": "application/json"})

    def _json(self, method: str, url: str, **kw) -> dict:
        try:
            with self._client() as c:
                r = c.request(method, url, **kw)
        except httpx.HTTPError as e:
            # the exception text names the URL (never a code or token: those travel in POST bodies / headers)
            log.warning("OIDC: %s %s failed: %s", method, _redact_url(url), type(e).__name__)
            raise OidcError("idp_unavailable", "the identity provider could not be reached") from None
        except (OSError, ValueError) as e:  # e.g. an unreadable ca_bundle
            log.warning("OIDC: %s %s failed: %s", method, _redact_url(url), e)
            raise OidcError("idp_unavailable", "the identity provider could not be reached") from None
        if len(r.content) > MAX_RESPONSE_BYTES:
            raise OidcError("idp_error", "the identity provider's response is too large")
        try:
            data = r.json()
        except ValueError:
            data = None
        if r.status_code >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            log.warning("OIDC: %s %s answered %s%s", method, _redact_url(url), r.status_code,
                        f" ({_safe_word(err)})" if err else "")
            raise OidcError("idp_error", "the identity provider refused the request")
        if not isinstance(data, dict):
            raise OidcError("idp_error", "the identity provider sent an invalid response")
        return data

    # discovery -----------------------------------------------------------------------------------------------------

    def metadata(self) -> dict:
        with self._lock:
            if self._meta is not None and self.clock() - self._meta_at < METADATA_TTL_S:
                return self._meta
        url = self.cfg.issuer.rstrip("/") + "/.well-known/openid-configuration"
        meta = self._json("GET", url)
        if meta.get("issuer") != self.cfg.issuer:
            log.warning("OIDC: the discovery document names another issuer than auth.oidc.issuer")
            raise OidcError("idp_error", "the identity provider's issuer does not match the configuration")
        for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not _is_https(meta.get(key)):
                raise OidcError("idp_error", f"the identity provider's {key} is missing or not https")
        if meta.get("userinfo_endpoint") is not None and not _is_https(meta["userinfo_endpoint"]):
            meta = {k: v for k, v in meta.items() if k != "userinfo_endpoint"}
        methods = meta.get("code_challenge_methods_supported")
        if isinstance(methods, list) and "S256" not in methods:
            raise OidcError("idp_error", "the identity provider does not support PKCE S256")
        with self._lock:
            self._meta, self._meta_at = meta, self.clock()
        return meta

    # signing keys --------------------------------------------------------------------------------------------------

    def _fetch_keys(self) -> None:
        data = self._json("GET", self.metadata()["jwks_uri"])
        keys: dict[str, tuple[jwt.PyJWK, dict]] = {}
        for i, raw in enumerate(data.get("keys") or []):
            if not isinstance(raw, dict) or raw.get("use", "sig") != "sig" or raw.get("kty") == "oct":
                continue
            if raw.get("alg") is not None and raw["alg"] not in ALLOWED_ALGS:
                continue
            try:
                key = jwt.PyJWK(raw)
            except Exception:  # unsupported key types are skipped, not fatal
                continue
            keys[str(raw.get("kid", f"#{i}"))] = (key, raw)
        with self._lock:
            self._keys, self._keys_at, self._keys_fetched_at = keys, self.clock(), self.clock()

    def signing_key(self, kid: str | None) -> tuple[jwt.PyJWK, dict]:
        with self._lock:
            fresh = self.clock() - self._keys_at < JWKS_TTL_S
            key = self._pick(kid) if fresh else None
            may_refetch = self.clock() - self._keys_fetched_at >= JWKS_REFRESH_MIN_S
        if key is not None:
            return key
        if fresh and not may_refetch:
            raise OidcError("invalid_token", "the ID token is signed with an unknown key")
        self._fetch_keys()
        with self._lock:
            key = self._pick(kid)
        if key is None:
            raise OidcError("invalid_token", "the ID token is signed with an unknown key")
        return key

    def _pick(self, kid: str | None) -> tuple[jwt.PyJWK, dict] | None:
        if kid is not None:
            return self._keys.get(str(kid))
        return next(iter(self._keys.values())) if len(self._keys) == 1 else None

    # the flow ------------------------------------------------------------------------------------------------------

    def authorization_url(self, *, redirect_uri: str, state: str, nonce: str, challenge: str) -> str:
        endpoint = self.metadata()["authorization_endpoint"]
        query = urlencode({"response_type": "code", "client_id": self.cfg.client_id, "redirect_uri": redirect_uri,
                           "scope": " ".join(self.cfg.scopes), "state": state, "nonce": nonce,
                           "code_challenge": challenge, "code_challenge_method": "S256"})
        return endpoint + ("&" if urlsplit(endpoint).query else "?") + query

    def check_iss_param(self, iss: str | None) -> None:
        """RFC 9207: the authorization response's ``iss`` must be our issuer (mix-up defence); required when the IdP
        says it sends one."""
        meta = self.metadata()
        if iss is None:
            if meta.get("authorization_response_iss_parameter_supported") is True:
                raise OidcError("invalid_state", "the sign-in response did not name its issuer")
            return
        if iss != self.cfg.issuer:
            raise OidcError("invalid_state", "the sign-in response came from another issuer")

    def exchange(self, *, code: str, redirect_uri: str, verifier: str) -> dict:
        meta = self.metadata()
        form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
                "code_verifier": verifier}
        auth = None
        secret = self._read_secret(self.cfg.client_secret_file)
        methods = meta.get("token_endpoint_auth_methods_supported") or ["client_secret_basic"]
        if secret and "client_secret_basic" in methods:
            auth = httpx.BasicAuth(_form_quote(self.cfg.client_id), _form_quote(secret))
        elif secret:
            form.update(client_id=self.cfg.client_id, client_secret=secret)
        else:
            form["client_id"] = self.cfg.client_id
        tokens = self._json("POST", meta["token_endpoint"], data=form, auth=auth)
        if not isinstance(tokens.get("id_token"), str):
            raise OidcError("invalid_token", "the identity provider returned no ID token")
        token_type = tokens.get("token_type")
        if token_type is not None and str(token_type).lower() != "bearer":
            raise OidcError("invalid_token", "the identity provider returned an unexpected token type")
        return tokens

    def validate_id_token(self, id_token: str, *, nonce: str) -> dict:
        try:
            header = jwt.get_unverified_header(id_token)
        except jwt.InvalidTokenError:
            raise OidcError("invalid_token", "the ID token is malformed") from None
        alg = header.get("alg")
        if alg not in ALLOWED_ALGS:
            raise OidcError("invalid_token", "the ID token uses a signature algorithm DGXMM does not accept")
        key, jwk = self.signing_key(header.get("kid"))
        if jwk.get("alg") not in (None, alg) or jwk.get("kty") != KEY_TYPE[alg[:2]]:
            raise OidcError("invalid_token", "the ID token's algorithm does not match its key")
        now = time.time()  # PyJWT checks exp / iat against the wall clock too
        try:
            claims = jwt.decode(id_token, key.key, algorithms=[alg], audience=self.cfg.client_id,
                                issuer=self.cfg.issuer, leeway=LEEWAY_S,
                                options={"require": ["iss", "sub", "aud", "exp", "iat"], "verify_iat": True})
        except jwt.ExpiredSignatureError:
            raise OidcError("invalid_token", "the ID token has expired") from None
        except jwt.InvalidAudienceError:
            raise OidcError("invalid_token", "the ID token is for another client") from None
        except jwt.InvalidIssuerError:
            raise OidcError("invalid_token", "the ID token is from another issuer") from None
        except jwt.ImmatureSignatureError:
            raise OidcError("invalid_token", "the ID token is not valid yet") from None
        except jwt.MissingRequiredClaimError as e:
            raise OidcError("invalid_token", f"the ID token lacks the {e.claim} claim") from None
        except Exception:  # bad signature, a key of the wrong type, malformed claims
            raise OidcError("invalid_token", "the ID token's signature or claims are invalid") from None
        if not isinstance(claims.get("iat"), (int, float)) or claims["iat"] < now - MAX_TOKEN_AGE_S - LEEWAY_S:
            raise OidcError("invalid_token", "the ID token is too old")
        aud = claims["aud"]
        if isinstance(aud, list) and len(aud) > 1 and claims.get("azp") != self.cfg.client_id:
            raise OidcError("invalid_token", "the ID token's authorized party is another client")
        if claims.get("azp") is not None and claims["azp"] != self.cfg.client_id:
            raise OidcError("invalid_token", "the ID token's authorized party is another client")
        got = claims.get("nonce")
        if not isinstance(got, str) or not secrets.compare_digest(got.encode(), nonce.encode()):
            raise OidcError("invalid_token", "the ID token's nonce does not match this sign-in")
        sub = claims.get("sub")
        if not isinstance(sub, str) or not 0 < len(sub) <= 255:
            raise OidcError("invalid_token", "the ID token's subject is invalid")
        return claims

    def userinfo(self, access_token: str | None, sub: str) -> dict:
        """Claims from the userinfo endpoint (for IdPs that keep groups out of the ID token); {} when there is none.
        Its ``sub`` must be the ID token's."""
        endpoint = self.metadata().get("userinfo_endpoint")
        if not endpoint or not access_token:
            return {}
        info = self._json("GET", endpoint, headers={"Authorization": f"Bearer {access_token}"})
        if info.get("sub") != sub:
            raise OidcError("invalid_token", "the userinfo response is for another user")
        return info


def _is_https(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.hostname) and not parts.username


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _safe_word(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "", str(v))[:64]


def _form_quote(v: str) -> str:
    # RFC 6749 2.3.1: client_secret_basic credentials are form-urlencoded before the base64
    from urllib.parse import quote_plus
    return quote_plus(v)


def _read_secret_file(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
    except OSError:
        return None
    return value or None
