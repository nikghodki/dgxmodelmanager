import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from sqlalchemy import func, select

from dgxmm.admin import list_users, main, reset_password
from dgxmm.auth.passwords import verify_password
from dgxmm.auth.sessions import create_session, resolve_session
from dgxmm.db import AuditLog, User, UserSession


def _user(db, name):
    with db.session() as s:
        return s.scalar(select(User).where(User.username == name))


def test_reset_password_sets_random_pw_forces_change_and_revokes_sessions(db, make_user):
    make_user("alice", "correct-horse-battery", "admin")
    make_user("bob", "another-long-pass", "viewer")
    alice, bob = _user(db, "alice"), _user(db, "bob")
    t1, _ = create_session(db, alice.id, "1.2.3.4", 3600)
    t2, _ = create_session(db, alice.id, "1.2.3.5", 3600)
    tb, _ = create_session(db, bob.id, "1.2.3.6", 3600)

    pw = reset_password(db, "Alice ")
    assert len(pw) >= 16
    after = _user(db, "alice")
    assert verify_password(after.pw_hash, pw)
    assert not verify_password(after.pw_hash, "correct-horse-battery")
    assert after.must_change_pw is True
    assert resolve_session(db, t1, 3600) is None and resolve_session(db, t2, 3600) is None
    assert resolve_session(db, tb, 3600) is not None  # other users keep their sessions
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(UserSession).where(UserSession.user_id == alice.id)) == 0
        row = s.scalar(select(AuditLog).where(AuditLog.action == "password_reset_cli"))
        assert row.target == "alice" and pw not in str(row.detail)
    assert reset_password(db, "alice") != pw  # random each time


def test_reset_password_unknown_user(db, make_user):
    make_user("alice", "correct-horse-battery", "admin")
    with pytest.raises(LookupError):
        reset_password(db, "mallory")


def test_list_users(db, make_user):
    make_user("bob", "another-long-pass", "viewer")
    make_user("alice", "correct-horse-battery", "admin")
    assert list_users(db) == [("alice", "admin"), ("bob", "viewer")]


def _config(tmp_path, settings) -> str:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": settings.data_dir}))
    return str(cfg)


def test_main_prints_password_once(tmp_path, settings, db, make_user, capsys):
    make_user("admin", "correct-horse-battery", "admin")
    assert main(["--config", _config(tmp_path, settings), "reset-password", "admin"]) == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("New password for admin: "))
    pw = line.split(": ", 1)[1]
    assert out.count(pw) == 1
    assert verify_password(_user(db, "admin").pw_hash, pw)


def test_main_unknown_user_exits_1(tmp_path, settings, db, make_user, capsys):
    make_user("admin", "correct-horse-battery", "admin")
    assert main(["--config", _config(tmp_path, settings), "reset-password", "nobody"]) == 1
    err = capsys.readouterr().err
    assert "no user named 'nobody'" in err and "admin" in err


def test_main_missing_database(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(tmp_path / "nothing")}))
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "reset-password", "admin"])
    assert "no DGXMM database" in str(e.value)
    assert not (tmp_path / "nothing").exists()


def test_module_entry_point(tmp_path, settings, db, make_user):
    """`python -m dgxmm.admin` works the way the docs say (DGXMM_CONFIG selects the config)."""
    make_user("admin", "correct-horse-battery", "admin")
    env = {**os.environ, "DGXMM_CONFIG": _config(tmp_path, settings)}
    r = subprocess.run([sys.executable, "-m", "dgxmm.admin", "reset-password", "admin"], env=env,
                       capture_output=True, text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr
    pw = r.stdout.splitlines()[0].split(": ", 1)[1]
    assert verify_password(_user(db, "admin").pw_hash, pw)
    r = subprocess.run([sys.executable, "-m", "dgxmm.admin", "list-users"], env=env,
                       capture_output=True, text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and r.stdout.strip() == "admin\tadmin"
