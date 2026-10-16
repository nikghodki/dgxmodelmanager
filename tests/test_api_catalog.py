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
