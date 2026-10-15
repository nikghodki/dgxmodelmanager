import hashlib
import secrets
from datetime import timedelta

from sqlalchemy import delete

from dgxmm.db import Database, User, UserSession, as_aware, utcnow


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db: Database, user_id: int, ip: str, ttl_s: int) -> tuple[str, str]:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    with db.session() as s:
        s.add(UserSession(id=_digest(token), user_id=user_id, csrf_token=csrf, ip=ip,
                          expires_at=utcnow() + timedelta(seconds=ttl_s)))
    return token, csrf


def resolve_session(db: Database, token: str | None, ttl_s: int) -> tuple[UserSession, User] | None:
    if not token:
        return None
    with db.session() as s:
        sess = s.get(UserSession, _digest(token))
        if sess is None:
            return None
        user = s.get(User, sess.user_id)
        if user is None or as_aware(sess.expires_at) < utcnow():
            s.delete(sess)
            return None
        sess.expires_at = utcnow() + timedelta(seconds=ttl_s)  # sliding expiry
        return sess, user


def delete_session(db: Database, token: str | None) -> None:
    if token:
        with db.session() as s:
            s.execute(delete(UserSession).where(UserSession.id == _digest(token)))


def delete_user_sessions(db: Database, user_id: int, keep_token: str | None = None) -> None:
    with db.session() as s:
        q = delete(UserSession).where(UserSession.user_id == user_id)
        if keep_token:
            q = q.where(UserSession.id != _digest(keep_token))
        s.execute(q)
