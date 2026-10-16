"""Tests for /api/users endpoints."""
from sqlalchemy import select

from dgxmm.auth.passwords import hash_password
from dgxmm.db import User


def admin_headers(c):
    with c.ctx.db.session() as s:
        s.add(User(username="admin", pw_hash=hash_password("admin-pw-123"), role="admin"))
        s.add(User(username="viewer", pw_hash=hash_password("viewer-pw-123"), role="viewer"))
    r = c.post("/dgxmm/api/auth/login", json={"username": "admin", "password": "admin-pw-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def viewer_headers(c):
    with c.ctx.db.session() as s:
        if not s.scalar(select(User).where(User.username == "viewer")):
            s.add(User(username="viewer", pw_hash=hash_password("viewer-pw-123"), role="viewer"))
    r = c.post("/dgxmm/api/auth/login", json={"username": "viewer", "password": "viewer-pw-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def test_list_users_requires_admin(api_stack):
    admin_headers(api_stack)  # create users
    vh = viewer_headers(api_stack)
    r = api_stack.get("/dgxmm/api/users", headers=vh)
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "admin_required"


def test_list_users_works_for_admin(api_stack):
    ah = admin_headers(api_stack)
    r = api_stack.get("/dgxmm/api/users", headers=ah)
    assert r.status_code == 200
    users = r.json()
    assert len(users) == 2
    assert {u["username"] for u in users} == {"admin", "viewer"}
