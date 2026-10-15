import logging
import secrets

from sqlalchemy import func, select

from dgxmm.auth.passwords import hash_password
from dgxmm.db import Database, User

log = logging.getLogger("dgxmm.bootstrap")


def bootstrap_admin(db: Database) -> str | None:
    """Create `admin` with a one-time random password when no users exist."""
    with db.session() as s:
        if s.scalar(select(func.count()).select_from(User)):
            return None
        password = secrets.token_urlsafe(18)
        s.add(User(username="admin", pw_hash=hash_password(password), role="admin", must_change_pw=True))
    log.warning("DGXMM first run: created user 'admin' with password %s (change it at first login)", password)
    return password
