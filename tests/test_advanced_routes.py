"""Integration (spec 2026-10-17): every route the advanced decision-model features add is in one access matrix
(anon 401 / viewer 403 or 200 / admin allowed / CSRF on state changes), and each is rate-limited."""
import pytest

from tests.test_api_decider import api, viewer  # noqa: F401  (fixture)
from tests.test_decider_api_security import _tiny, admin_login, anon
from tests.test_decider_compat import QWEN, Ctx, FakeHub, repo

A = "/dgxmm/api"
# API modules of the advanced features: every route they define must be in routes() below
MODULES = {"dgxmm.api.insights", "dgxmm.api.deploy_advice", "dgxmm.api.decider_ops", "dgxmm.api.schedules",
           "dgxmm.api.commands"}


def routes(dep_id: int = 1):
    """(method, path, json body, viewer allowed)."""
    sched = {"schedules": [{"action": "stop", "days": 1, "at_local": "03:00", "timezone": "UTC"}]}
    return [
        ("GET", f"{A}/insights", None, True),
        ("POST", f"{A}/insights/1/dismiss", None, False),
        ("POST", f"{A}/insights/1/snooze", {"hours": 2}, False),
        ("POST", f"{A}/insights/1/applied", None, False),
        ("GET", f"{A}/catalog/compat?repo=org/m", None, True),
        ("POST", f"{A}/decider/config-advice", {"model_ref": "Qwen/Qwen3-Tiny", "engine": "vllm", "goal": "chat"},
         False),
        ("POST", f"{A}/decider/draft-suggest", {"target_ref": "Qwen/Qwen3-Tiny", "method": "draft_model"}, False),
        ("POST", f"{A}/decider/engine-suggest", {"model_ref": "Qwen/Qwen3-Tiny", "goal": "chat"}, False),
        ("POST", f"{A}/decider/remote-code-check", {"model_ref": "Qwen/Qwen3-Tiny"}, False),
        ("GET", f"{A}/decider/routing-advice", None, False),
        ("GET", f"{A}/deployments/{dep_id}/schedules", None, True),
        ("POST", f"{A}/deployments/{dep_id}/schedules", sched, False),
        ("PUT", f"{A}/schedules/999", {"enabled": False}, False),
        ("DELETE", f"{A}/schedules/999", None, False),
        ("GET", f"{A}/schedules/settings", None, True),
        ("PUT", f"{A}/schedules/settings", {"paused_all": False}, False),
        ("POST", f"{A}/decider/command", {"text": "open the insights page"}, True),
        ("GET", f"{A}/decider/free-plans?gib=1", None, False),
        ("POST", f"{A}/decider/free-memory", {"gib": 1, "plan_id": "nope"}, False),
    ]


def _template(method: str, path: str) -> tuple[str, str]:
    path = path.split("?")[0].replace("/insights/1/", "/insights/{insight_id}/")
    return method, path.replace("/deployments/1/", "/deployments/{dep_id}/").replace("/schedules/999", "/schedules/{sid}")


def _module_routes():
    """The APIRoutes of MODULES, each of which the app serves (dgxmm.app.ROUTERS) under the base path."""
    import importlib

    from fastapi.routing import APIRoute

    from dgxmm.app import ROUTERS
    out = []
    for name in sorted(MODULES):
        mod = importlib.import_module(name)
        assert mod in ROUTERS, name
        out += [r for r in mod.router.routes if isinstance(r, APIRoute)]
    return out


def test_every_advanced_route_is_in_the_matrix():
    defined = {(m, "/dgxmm" + r.path) for r in _module_routes() for m in r.methods}
    assert defined == {_template(m, p) for m, p, _, _ in routes()}


def _limited(dependant) -> bool:
    return any(d.call is not None and getattr(d.call, "__qualname__", "") == "limited.<locals>.dep" or _limited(d)
               for d in dependant.dependencies)


def test_every_advanced_route_is_rate_limited():
    found = _module_routes()
    assert found and all(_limited(r.dependant) for r in found), [r.path for r in found if not _limited(r.dependant)]


def test_access_matrix_of_the_advanced_routes(api):
    api.ctx.hub = FakeHub({"org/m": repo(QWEN)})
    api.ctx.archs = Ctx(None, None).archs
    dep_id = _tiny(api)
    for method, path, body, viewer_ok in routes(dep_id):
        r = api.request(method, path, headers=anon(api), json=body)
        assert r.status_code == 401, (method, path, "anon", r.status_code)
        r = api.request(method, path, headers=viewer(api), json=body)
        if viewer_ok:
            assert r.status_code == 200, (method, path, "viewer", r.status_code, r.text)
        else:
            assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required", (method, path, r.text)
        r = api.request(method, path, headers=admin_login(api), json=body)
        assert r.status_code < 500 and r.status_code not in (401, 403, 429), (method, path, r.status_code, r.text)
        if method != "GET":  # a state-changing request without the CSRF header is refused even for an admin
            r = api.request(method, path, json=body)
            assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf_failed", (method, path, r.text)


@pytest.mark.parametrize("bucket", ["insights_read", "insights_write", "decider_config", "decider_compat",
                                    "decider_draft", "decider_engine", "decider_remote_code", "decider_routing",
                                    "decider_schedule", "decider_command"])
def test_buckets_of_the_advanced_routes_resolve(bucket):
    from dgxmm.api.ratelimit import bucket_limits
    per_user, window, _ = bucket_limits(bucket)
    assert per_user > 0 and window > 0
