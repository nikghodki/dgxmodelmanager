"""Short-lived, scoped admin tokens for the local admin CLI.

``docker exec dgxmm python -m dgxmm.admin decider enable`` must drive the *running* app (only it owns the deployer),
without a password and without printing a secret. Whoever can run that command already has the database and the
Docker socket, so the CLI issues itself a token straight into the database and sends it in the ``X-DGXMM-CLI-Token``
header to ``http://127.0.0.1:8090``. A token:

- is random (256 bit), stored only as its SHA-256, and deleted by the CLI when it finishes;
- expires (at most ``MAX_TTL_S``), and expired rows are swept on every lookup;
- is accepted only on the admin unix socket (dgxmm/admin_listener.py), never on the TCP listener, and never with
  forwarding headers;
- is accepted only on the routes of its scope (``SCOPES``), so it cannot manage users or read audit data;
- belongs to an admin user (checked at issue and at every use), who is the actor in the audit log.

No CSRF header is needed: the token is not a cookie, so a browser cannot send it cross-site. The pending
first-login password change does not block it (the CLI is how a fresh install enables the decider).
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta

from sqlalchemy import delete

from dgxmm.audit import audit
from dgxmm.db import CliToken, Database, User, as_aware, utcnow

HEADER = "X-DGXMM-CLI-Token"
PREFIX = "dgxmmcli_"
MAX_TTL_S = 3 * 3600
# scope -> allowed (method, path without base path)
SCOPES: dict[str, frozenset[tuple[str, str]]] = {
    "decider": frozenset({
        ("GET", "/api/decider/status"), ("POST", "/api/decider/enable"), ("POST", "/api/decider/disable"),
        ("GET", "/api/decider/image"), ("POST", "/api/decider/image/build"), ("POST", "/api/inventory/refresh"),
    }),
}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue(db: Database, user_id: int, ttl_s: int, scope: str = "decider") -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}")
    if not 0 < ttl_s <= MAX_TTL_S:
        raise ValueError(f"ttl_s must be 1..{MAX_TTL_S}")
    token = PREFIX + secrets.token_urlsafe(32)
    with db.session() as s:
        user = s.get(User, user_id)
        if user is None or user.role != "admin":
            raise PermissionError("CLI tokens are only issued to admin users")
        expires = utcnow() + timedelta(seconds=ttl_s)
        s.add(CliToken(id=_digest(token), user_id=user_id, scope=scope, expires_at=expires))
    # The token itself is never audited: only a short prefix of its hash, to pair issue and revoke.
    audit(db, user_id, "cli_token_issued", scope, {"scope": scope, "ttl_s": ttl_s, "token_ref": _digest(token)[:12],
                                                   "expires_at": expires.isoformat()})
    return token


def revoke(db: Database, token: str) -> None:
    with db.session() as s:
        row = s.get(CliToken, _digest(token))
        user_id = row.user_id if row else None
        s.execute(delete(CliToken).where(CliToken.id == _digest(token)))
    audit(db, user_id, "cli_token_revoked", "", {"token_ref": _digest(token)[:12]})


def resolve(db: Database, token: str) -> tuple[CliToken, User] | None:
    """The token row and its user, or None (unknown, expired or user gone). Sweeps expired rows."""
    now = utcnow()
    with db.session() as s:
        s.execute(delete(CliToken).where(CliToken.expires_at < now))
        row = s.get(CliToken, _digest(token)) if token.startswith(PREFIX) else None
        if row is None or as_aware(row.expires_at) < now:
            return None
        user = s.get(User, row.user_id)
        if user is None:
            return None
        return row, user


def allowed(scope: str, method: str, path: str) -> bool:
    return (method.upper(), path) in SCOPES.get(scope, frozenset())
