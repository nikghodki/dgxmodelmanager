from dgxmm.auth.bootstrap import bootstrap_admin
from dgxmm.auth.passwords import hash_password
from dgxmm.db import User


def test_bootstrap_creates_admin_once(db):
    pw = bootstrap_admin(db)
    assert pw and len(pw) >= 16
    assert bootstrap_admin(db) is None


def test_login_me_logout(client, make_user):
    make_user("alice", "correct-horse-battery", "admin")
    r = client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": "wrong-password-x"})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "invalid_credentials"
    r = client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": "correct-horse-battery"})
    assert r.status_code == 200
    csrf = r.json()["csrf_token"]
    me = client.get("/dgxmm/api/auth/me").json()
    assert me["user"]["username"] == "alice" and me["csrf_token"] == csrf
    assert client.post("/dgxmm/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
    assert client.get("/dgxmm/api/auth/me").status_code == 401


def test_csrf_required_for_mutations(client, make_user, login):
    make_user("alice", "correct-horse-battery", "admin")
    login("alice", "correct-horse-battery")
    r = client.post("/dgxmm/api/users", json={"username": "bob", "password": "another-long-pass", "role": "viewer"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf_failed"


def test_must_change_password_gate(client, db, login):
    pw = bootstrap_admin(db)
    csrf = login("admin", pw)
    r = client.get("/dgxmm/api/users")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "password_change_required"
    r = client.post("/dgxmm/api/auth/change-password", headers={"X-CSRF-Token": csrf},
                    json={"current_password": pw, "new_password": "short"})
    assert r.status_code == 422
    r = client.post("/dgxmm/api/auth/change-password", headers={"X-CSRF-Token": csrf},
                    json={"current_password": pw, "new_password": "a-much-longer-password"})
    assert r.status_code == 200
    assert client.get("/dgxmm/api/users").status_code == 200


def test_rate_limit_login(client, make_user):
    make_user("alice", "correct-horse-battery", "admin")
    codes = [client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": "nope-nope-nope"}).status_code
             for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_viewer_cannot_manage_users(client, make_user, login):
    make_user("vic", "viewer-password-1", "viewer")
    csrf = login("vic", "viewer-password-1")
    r = client.post("/dgxmm/api/users", headers={"X-CSRF-Token": csrf},
                    json={"username": "bob", "password": "another-long-pass", "role": "viewer"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required"


def test_user_crud_and_guards(client, make_user, login):
    make_user("alice", "correct-horse-battery", "admin")
    csrf = login("alice", "correct-horse-battery")
    h = {"X-CSRF-Token": csrf}
    bad = client.post("/dgxmm/api/users", headers=h, json={"username": "../x", "password": "another-long-pass", "role": "viewer"})
    assert bad.status_code == 422
    r = client.post("/dgxmm/api/users", headers=h, json={"username": "bob", "password": "another-long-pass", "role": "viewer"})
    assert r.status_code == 201
    bob_id = r.json()["id"]
    assert client.patch(f"/dgxmm/api/users/{bob_id}", headers=h, json={"role": "admin"}).json()["role"] == "admin"
    users = client.get("/dgxmm/api/users").json()
    alice_id = next(u["id"] for u in users if u["username"] == "alice")
    r = client.delete(f"/dgxmm/api/users/{alice_id}", headers=h)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "cannot_delete_self"
    assert client.delete(f"/dgxmm/api/users/{bob_id}", headers=h).status_code == 204
