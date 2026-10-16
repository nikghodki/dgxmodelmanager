"""Admin command line, run inside the dgxmm container.

    docker exec dgxmm python -m dgxmm.admin reset-password <username>
    docker exec dgxmm python -m dgxmm.admin list-users

reset-password sets a new random password, forces a change at the next login, signs the user out
everywhere and prints the new password once. It works while DGXMM is running (SQLite WAL).
"""
from __future__ import annotations

import argparse
import secrets
import sys

from sqlalchemy import select

from dgxmm.audit import audit
from dgxmm.auth.passwords import hash_password
from dgxmm.auth.sessions import delete_user_sessions
from dgxmm.config import load_settings
from dgxmm.db import Database, User


def _db(config: str | None) -> Database:
    settings = load_settings(config)
    if not settings.db_path.is_file():
        raise SystemExit(f"error: no DGXMM database at {settings.db_path} (has DGXMM started once?)")
    return Database(f"sqlite:///{settings.db_path}")


def reset_password(db: Database, username: str) -> str:
    """Give `username` a new random password; returns it. Raises LookupError if there is no such user."""
    name = username.strip().lower()
    password = secrets.token_urlsafe(18)
    with db.session() as s:
        user = s.scalar(select(User).where(User.username == name))
        if user is None:
            raise LookupError(name)
        user.pw_hash = hash_password(password)
        user.must_change_pw = True
        user_id = user.id
    delete_user_sessions(db, user_id)
    audit(db, None, "password_reset_cli", name)
    return password


def list_users(db: Database) -> list[tuple[str, str]]:
    with db.session() as s:
        return [(u.username, u.role) for u in s.scalars(select(User).order_by(User.username))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dgxmm.admin", description="DGXMM admin commands")
    parser.add_argument("--config", help="config file (default: $DGXMM_CONFIG or config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("reset-password", help="set a new random password and sign the user out")
    rp.add_argument("username")
    sub.add_parser("list-users", help="list user names and roles")
    args = parser.parse_args(argv)

    db = _db(args.config)
    if args.cmd == "list-users":
        for name, role in list_users(db):
            print(f"{name}\t{role}")
        return 0
    try:
        password = reset_password(db, args.username)
    except LookupError:
        known = ", ".join(n for n, _ in list_users(db)) or "none"
        print(f"error: no user named {args.username!r} (users: {known})", file=sys.stderr)
        return 1
    print(f"New password for {args.username.strip().lower()}: {password}")
    print("It is shown only once. The user must change it at the next login; existing sessions were signed out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
