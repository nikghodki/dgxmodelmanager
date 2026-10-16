"""GET /api/host/processes (spec §5): viewer-readable, redacted for everyone, scope/limit, container summaries."""
from tests.test_api_deployments import admin
from tests.test_processes import CID, OTHER, SECRET, build


def wire(api_stack, tmp_path):
    probe = build(api_stack.ctx.settings, tmp_path)
    api_stack.ctx.probe = probe
    rows = lambda: [(7, "qwen", "ready", 0, CID), (8, "stopped-one", "starting", 0, None)]  # noqa: E731
    api_stack.ctx.deployer.usage_rows = rows
    api_stack.ctx.budget.usage_provider = rows
    return probe


def viewer(api_stack):
    admin(api_stack)
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert r.status_code == 200


def test_requires_login(api_stack):
    assert api_stack.get("/dgxmm/api/host/processes").status_code == 401


def test_viewer_gets_redacted_processes_and_container_groups(api_stack, tmp_path):
    wire(api_stack, tmp_path)
    viewer(api_stack)
    r = api_stack.get("/dgxmm/api/host/processes")
    assert r.status_code == 200
    body = r.json()
    assert SECRET not in r.text and "environ" not in r.text
    assert [p["pid"] for p in body["processes"]] == [100, 300, 200, 101, 400]
    assert body["total"] == 5 and body["scope"] == "all"
    p100 = body["processes"][0]
    assert p100["managed"] == {"deployment_id": 7, "name": "qwen"} and p100["command"].endswith("--api-key ***")
    assert p100["container"]["name"] == "dgxmm-qwen" and (p100["user"], p100["user_scope"]) == ("root", "container")
    groups = {g["name"]: g for g in body["containers"]}
    assert groups["dgxmm-qwen"]["process_count"] == 2 and groups["dgxmm-qwen"]["managed"] == {"deployment_id": 7, "name": "qwen"}
    assert groups["dgxmm-qwen"]["gpu"] == 40 * 2**30
    assert groups["dgxmm-qwen"]["rss"] == (2_000_000 + 500_000) * 1024
    assert groups["litellm-proxy"]["image"] == "ghcr.io/berriai/litellm:main-latest" and groups["litellm-proxy"]["managed"] is None
    assert [g["name"] for g in body["containers"]][0] == "dgxmm-qwen"  # sorted by rss+gpu desc
    assert body["host"] == {"rss": 3_000_000 * 1024, "gpu": 2**30, "process_count": 1}
    assert isinstance(body["duration_ms"], (int, float))


def test_scope_and_limit(api_stack, tmp_path):
    wire(api_stack, tmp_path)
    viewer(api_stack)
    ext = api_stack.get("/dgxmm/api/host/processes?scope=external").json()
    assert [p["pid"] for p in ext["processes"]] == [300, 200, 400] and ext["total"] == 3
    assert {g["name"] for g in ext["containers"]} == {"litellm-proxy", None}
    man = api_stack.get("/dgxmm/api/host/processes?scope=managed").json()
    assert [p["pid"] for p in man["processes"]] == [100, 101] and man["host"]["process_count"] == 0
    lim = api_stack.get("/dgxmm/api/host/processes?limit=2").json()
    assert [p["pid"] for p in lim["processes"]] == [100, 300] and lim["total"] == 5
    assert sum(g["process_count"] for g in lim["containers"]) + lim["host"]["process_count"] == 5  # groups use all
    assert api_stack.get("/dgxmm/api/host/processes?scope=bogus").status_code == 422
    assert api_stack.get("/dgxmm/api/host/processes?limit=0").status_code == 422
    assert api_stack.get("/dgxmm/api/host/processes?limit=100000").status_code == 422


def test_host_external_top_is_enriched_and_backward_compatible(api_stack, tmp_path):
    wire(api_stack, tmp_path)
    viewer(api_stack)
    api_stack.ctx.budget.probe = api_stack.ctx.probe
    body = api_stack.get("/dgxmm/api/host").json()
    top = body["external_top"]
    assert [p["pid"] for p in top] == [300, 200, 400]
    for p in top:
        assert {"pid", "name", "rss", "gpu", "command", "user", "user_scope", "container"} <= set(p)
    assert top[1]["container"]["name"] == "litellm-proxy" and top[0]["container"] is None
    assert SECRET not in str(body)


def test_label_managed_container_gets_the_deployment_name(api_stack, tmp_path):
    from dgxmm.db import Deployment
    wire(api_stack, tmp_path)
    api_stack.ctx.deployer.usage_rows = lambda: []  # deployment 7 not active, container still labelled
    viewer(api_stack)
    with api_stack.ctx.db.session() as s:
        s.add(Deployment(id=7, name="qwen-old", engine="vllm", model_ref="Qwen/Qwen3-8B", state="stopped"))
    body = api_stack.get("/dgxmm/api/host/processes?scope=managed").json()
    assert [p["managed"] for p in body["processes"]] == [{"deployment_id": 7, "name": "qwen-old"}] * 2


def test_many_processes(api_stack, tmp_path):
    from pathlib import Path
    from tests.test_processes import proc
    wire(api_stack, tmp_path)
    root = Path(api_stack.ctx.settings.host_proc)
    for pid in range(1000, 1600):
        proc(root, pid, ["python", "-m", "worker", f"--id={pid}", "--api-key", SECRET], 1000 + pid,
             f"/system.slice/docker-{OTHER}.scope")
    viewer(api_stack)
    body = api_stack.get("/dgxmm/api/host/processes?limit=50").json()
    # timing is measured on the real box (lane report), not asserted here: wall-clock bounds are flaky in CI
    assert body["total"] == 605 and len(body["processes"]) == 50 and SECRET not in str(body)
