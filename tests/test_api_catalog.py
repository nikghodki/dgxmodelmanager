from types import SimpleNamespace

from dgxmm.catalog.service import CatalogService
from dgxmm.downloads import DownloadService
from tests.test_api_deployments import admin
from tests.test_catalog import FakeHub, FakeLlmfit
from tests.test_downloads import FILES, Hub, Usage, fake_downloader
from dgxmm.config import GIB


def wire(c):
    st = c.stack
    c.ctx.catalog = CatalogService(st.settings, FakeHub(), FakeLlmfit(), st.inv, st.budget, st.deployer,
                                   SimpleNamespace(get=lambda: None))
    c.ctx.downloads = DownloadService(st.settings, st.db, st.inv, Hub(FILES), st.events, downloader=fake_downloader(),
                                      disk_usage=lambda p: Usage(0, 0, 1000 * GIB))


def test_catalog_routes(api_stack):
    admin(api_stack)
    wire(api_stack)
    r = api_stack.get("/dgxmm/api/catalog/search", params={"q": "qwen"}).json()
    assert r["results"][0]["repo_id"] == "Qwen/Qwen3-Tiny"
    d = api_stack.get("/dgxmm/api/catalog/details", params={"repo": "Qwen/Qwen3-Tiny"}).json()
    assert d["fit"]["badge"] == "fits_now"
    assert api_stack.get("/dgxmm/api/catalog/details", params={"repo": "../x"}).status_code == 422


def test_download_routes(api_stack):
    h = admin(api_stack)
    wire(api_stack)
    r = api_stack.post("/dgxmm/api/downloads", headers=h, json={"repo_id": "org/new-model"})
    assert r.status_code == 202, r.text
    assert api_stack.get("/dgxmm/api/downloads").json()[0]["repo_id"] == "org/new-model"
    bad = api_stack.post("/dgxmm/api/downloads", headers=h, json={"repo_id": "../../etc"})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_repo"


def test_planner_and_favourites(api_stack):
    h = admin(api_stack)
    assert api_stack.put("/dgxmm/api/planner/favourites/Qwen/Qwen3-Tiny", headers=h).status_code == 204
    v = api_stack.get("/dgxmm/api/planner").json()
    assert next(x for x in v["candidates"] if x["ref"] == "Qwen/Qwen3-Tiny")["favourite"] is True
    assert api_stack.delete("/dgxmm/api/planner/favourites/Qwen/Qwen3-Tiny", headers=h).status_code == 204


def test_favourite_gguf_path_url_encoded(api_stack):
    """M9: the UI sends encodeURIComponent(ref); a GGUF absolute path must round-trip intact."""
    from dgxmm.db import Favourite
    h = admin(api_stack)
    ref = "/data/gguf/Qwen3-8B-Q4_K_M.gguf"
    from urllib.parse import quote
    enc = quote(ref, safe="")
    assert api_stack.put(f"/dgxmm/api/planner/favourites/{enc}", headers=h).status_code == 204
    with api_stack.ctx.db.session() as s:
        assert {f.ref for f in s.query(Favourite)} == {ref}
    # double-encoded (some proxies re-escape %): still the same ref
    assert api_stack.put(f"/dgxmm/api/planner/favourites/{quote(enc, safe='')}", headers=h).status_code == 204
    with api_stack.ctx.db.session() as s:
        assert {f.ref for f in s.query(Favourite)} == {ref}
    assert api_stack.delete(f"/dgxmm/api/planner/favourites/{enc}", headers=h).status_code == 204
    with api_stack.ctx.db.session() as s:
        assert s.query(Favourite).count() == 0


class _RecordingHub(FakeHub):
    def __init__(self):
        self.calls = []

    def search(self, q, limit=30, gguf=False, anonymous=False):
        self.calls.append(("search", anonymous))
        return super().search(q, limit, gguf)

    def details(self, repo_id, anonymous=False):
        self.calls.append(("details", anonymous))
        if anonymous and repo_id == "org/private":
            raise LookupError("org/private not found")
        return super().details(repo_id)


def test_catalog_lookups_are_anonymous_for_viewers_and_private_repos_are_not_served(api_stack):
    """Review follow-up (L1): /api/catalog/details and /search for a viewer use no HF token; a private repo that
    only the admin's token can read is not described to a viewer."""
    admin(api_stack)
    wire(api_stack)
    hub = _RecordingHub()
    api_stack.ctx.catalog.hub = hub
    assert api_stack.get("/dgxmm/api/catalog/details", params={"repo": "org/private"}).status_code == 200
    assert hub.calls and all(anon is False for _, anon in hub.calls)
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert r.status_code == 200
    hub.calls.clear()
    assert api_stack.get("/dgxmm/api/catalog/details", params={"repo": "Qwen/Qwen3-Tiny"}).status_code == 200
    assert api_stack.get("/dgxmm/api/catalog/search", params={"q": "qwen"}).status_code == 200
    assert {kind for kind, _ in hub.calls} == {"details", "search"} and all(anon for _, anon in hub.calls)
    r = api_stack.get("/dgxmm/api/catalog/details", params={"repo": "org/private"})
    assert r.status_code == 502 and "Qwen" not in r.text
