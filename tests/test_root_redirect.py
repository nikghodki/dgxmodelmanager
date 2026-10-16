from fastapi.testclient import TestClient

from dgxmm.app import create_app
from dgxmm.context import AppContext
from dgxmm.events import EventBroker
from dgxmm.auth.ratelimit import SlidingWindowLimiter


def test_root_redirects_to_base_path(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/dgxmm/"


def test_base_path_without_slash_redirects(client):
    r = client.get("/dgxmm", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/dgxmm/"


def test_no_redirect_when_base_path_empty(settings, db):
    settings = settings.model_copy(update={"base_path": ""})
    (__import__("pathlib").Path(settings.static_dir) / "index.html").write_text("<html>ui</html>")
    ctx = AppContext(settings=settings, db=db, events=EventBroker(), login_limiter=SlidingWindowLimiter(5, 60))
    with TestClient(create_app(ctx), base_url="http://testserver") as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert "ui" in r.text
