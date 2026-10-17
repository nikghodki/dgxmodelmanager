"""Admin command line, run inside the dgxmm container.

    docker exec dgxmm python -m dgxmm.admin reset-password <username>
    docker exec dgxmm python -m dgxmm.admin list-users
    docker exec dgxmm python -m dgxmm.admin reset-calibration <repo> [--engine vllm|llamacpp]
    docker exec dgxmm python -m dgxmm.admin decider status|build|download|enable|disable [...]
    docker exec dgxmm python -m dgxmm.admin rotate-decider-key [--wait SECONDS]

decider manages the decision model (see dgxmm/decider/cli.py; ./install.sh --with-decider uses it).

reset-calibration forgets learned memory calibration (measurements and OOM records) for one model.

rotate-decider-key gives the decision model a new bearer key: the running DGXMM picks the request up (within a
few seconds), stops a running decider, writes the new key and starts the decider again with it. The key is never
printed. Exit 0 = rotated, 1 = failed, 2 = DGXMM did not pick the request up in time (it stays pending),
3 = still in progress (re-running waits for the same rotation; it never starts a second one).

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


def reset_calibration(db: Database, repo: str, engine: str | None = None) -> int:
    """Delete calibration rows for `repo` (optionally one engine); returns how many were deleted."""
    from dgxmm.memory.calibration import Calibration
    deleted = Calibration(db).reset(repo, engine=engine)
    audit(db, None, "calibration_reset_cli", repo, {"engine": engine, "deleted": deleted})
    return deleted


def rotate_decider_key(settings, wait_s: float) -> int:
    from dgxmm.decider import rotation
    nonce = rotation.request_rotation(settings)
    print("Rotation requested; waiting for DGXMM to rotate the key and restart the decider ...")
    res = rotation.wait_result(settings, nonce, wait_s)
    if res is None:
        print(f"DGXMM did not pick the request up within {wait_s:.0f} s (is it running?). It stays pending and runs "
              "when DGXMM next starts.", file=sys.stderr)
        return 2
    if res.get("status") != "done":
        print(f"The rotation is still in progress after {wait_s:.0f} s. Run the command again to keep waiting "
              "(it does not start a second rotation).", file=sys.stderr)
        return 3
    if not res.get("ok"):
        print(f"Key rotation failed: {res.get('error') or 'unknown error'} (rotated: {res.get('rotated')})",
              file=sys.stderr)
        return 1
    if res.get("restarted"):
        print(f"Key rotated; the decider (deployment {res.get('deployment_id')}) is restarting with it "
              f"(state {res.get('state')}).")
    else:
        print("Key rotated. No decider was running; it will use the new key when it starts.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dgxmm.admin", description="DGXMM admin commands")
    parser.add_argument("--config", help="config file (default: $DGXMM_CONFIG or config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("reset-password", help="set a new random password and sign the user out")
    rp.add_argument("username")
    sub.add_parser("list-users", help="list user names and roles")
    rc = sub.add_parser("reset-calibration", help="forget learned memory calibration for a model")
    rc.add_argument("repo")
    rc.add_argument("--engine", help="only this engine (vllm or llamacpp)")
    dc = sub.add_parser("decider", help="decision model: status, build, download, enable, disable", add_help=False)
    dc.add_argument("args", nargs=argparse.REMAINDER)
    rk = sub.add_parser("rotate-decider-key", help="new bearer key for the decision model (restarts it)")
    rk.add_argument("--wait", type=float, default=90.0,
                    help="seconds to wait for the rotation to finish (default 90; the restart itself takes minutes)")
    args = parser.parse_args(argv)

    if args.cmd == "decider":
        from dgxmm.decider import cli
        # build/download work before DGXMM ever started: the database is opened only when a command needs it
        return cli.run(args.args, db=lambda: _db(args.config), settings=load_settings(args.config))
    db = _db(args.config)
    if args.cmd == "rotate-decider-key":
        return rotate_decider_key(load_settings(args.config), args.wait)
    if args.cmd == "reset-calibration":
        n = reset_calibration(db, args.repo, engine=args.engine)
        print(f"deleted {n} calibration row(s) for {args.repo}" + (f" ({args.engine})" if args.engine else ""))
        return 0
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
