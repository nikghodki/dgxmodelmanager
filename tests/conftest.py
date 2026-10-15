from pathlib import Path

import pytest

from dgxmm.config import PathMapping, Settings
from dgxmm.db import Database


@pytest.fixture
def settings(tmp_path) -> Settings:
    for d in ("data", "static", "proc", "cgroup", "hf/hub", "gguf"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=str(tmp_path / "data"),
        static_dir=str(tmp_path / "static"),
        host_proc=str(tmp_path / "proc"),
        host_cgroup=str(tmp_path / "cgroup"),
        hf_cache=PathMapping(host="/host/hf", local=str(tmp_path / "hf")),
        gguf_roots=[PathMapping(host="/host/gguf", local=str(tmp_path / "gguf"))],
        litellm_key_file=str(tmp_path / "litellm_key"),
        hf_token_file=str(tmp_path / "hf_token"),
        engine_base_host="engines.test",
        monitor_poll_s=0.01,
    )


@pytest.fixture
def db(settings) -> Database:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return Database(f"sqlite:///{settings.db_path}")


from fastapi.testclient import TestClient

from dgxmm.auth.passwords import hash_password
from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.context import AppContext
from dgxmm.db import User
from dgxmm.events import EventBroker


@pytest.fixture
def ctx(settings, db) -> AppContext:
    return AppContext(settings=settings, db=db, events=EventBroker(),
                      login_limiter=SlidingWindowLimiter(5, 60))


@pytest.fixture
def client(ctx):
    from dgxmm.app import create_app
    with TestClient(create_app(ctx), base_url="http://testserver") as c:
        yield c


@pytest.fixture
def make_user(db):
    def _make(username, password, role, must_change=False):
        with db.session() as s:
            u = User(username=username, pw_hash=hash_password(password), role=role, must_change_pw=must_change)
            s.add(u)
            s.flush()
            return u.id
    return _make


@pytest.fixture
def login(client):
    def _login(username, password) -> str:
        r = client.post("/dgxmm/api/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return r.json()["csrf_token"]
    return _login


@pytest.fixture
def admin_csrf(make_user, login) -> str:
    make_user("root", "root-password-123", "admin")
    return login("root", "root-password-123")
