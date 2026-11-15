"""GET /api/nodes and where each deployment runs (placement, placement_label, rank_summary): Task 15."""


def test_get_nodes_lists_both_with_budget(client_viewer_two_nodes):
    r = client_viewer_two_nodes.get("/api/nodes").json()
    assert [n["id"] for n in r] == ["local", "dgx-2"] and "free_for_new" in r[1]["budget"]


def test_get_nodes_shape_for_a_viewer(client_viewer_two_nodes):
    r = client_viewer_two_nodes.get("/api/nodes")
    assert r.status_code == 200, r.text
    local, remote = r.json()
    assert set(local) == {"id", "label", "remote", "reachable", "stale", "budget"}
    assert (local["label"], local["remote"], local["reachable"], local["stale"]) == ("dgx-1", False, True, False)
    assert (remote["label"], remote["remote"], remote["reachable"], remote["stale"]) == ("dgx-2", True, True, False)
    # the per-node Budget dict, as GET /api/host serialises it
    assert set(local["budget"]) >= {"total", "free_for_new", "committed", "external", "deployments"}


def test_get_nodes_reports_a_stale_node(client_viewer_two_nodes, deployer_two_nodes):
    deployer_two_nodes.mark_stale("dgx-2")
    remote = client_viewer_two_nodes.get("/api/nodes").json()[1]
    assert remote["stale"] is True and remote["reachable"] is False


def test_get_nodes_needs_sign_in(api_two_nodes):
    assert api_two_nodes.get("/dgxmm/api/nodes").status_code == 401


async def test_deployment_response_has_placement_label(client_admin_two_nodes, ready_cluster_dep):
    d = client_admin_two_nodes.get(f"/api/deployments/{ready_cluster_dep.id}").json()
    assert d["placement_label"] == "dgx-1 + dgx-2 (cluster)"
    assert d["placement"] == {"kind": "cluster", "nodes": ["local", "dgx-2"]}


async def test_cluster_view_has_rank_summary_without_container_ids_or_ports(client_viewer_two_nodes,
                                                                            ready_cluster_dep):
    d = client_viewer_two_nodes.get(f"/api/deployments/{ready_cluster_dep.id}").json()
    assert [{k: r[k] for k in ("rank", "node", "node_label", "role")} for r in d["rank_summary"]] == [
        {"rank": 0, "node": "local", "node_label": "dgx-1", "role": "head"},
        {"rank": 1, "node": "dgx-2", "node_label": "dgx-2", "role": "worker"}]
    for r in d["rank_summary"]:
        assert set(r) == {"rank", "node", "node_label", "role", "status"}
    raw = str(d["rank_summary"])
    for rank in ready_cluster_dep.ranks:
        assert rank["container_id"] not in raw
    assert "ranks" not in d  # the raw rank entries stay internal


async def test_rank_status_comes_from_the_last_liveness_check_without_docker_calls(deployer_two_nodes,
                                                                                   ready_cluster_dep):
    d = deployer_two_nodes
    await d.check_liveness()
    assert [r["status"] for r in d.view(ready_cluster_dep.id)["rank_summary"]] == ["running", "running"]

    def no_docker(*a, **kw):
        raise AssertionError("a view must make no Docker call")
    d._docker_for = no_docker
    d.docker.get = no_docker
    assert [r["status"] for r in d.view(ready_cluster_dep.id)["rank_summary"]] == ["running", "running"]
    assert [r["status"] for r in next(v for v in d.list_views() if v["id"] == ready_cluster_dep.id)
            ["rank_summary"]] == ["running", "running"]


async def test_rank_status_without_a_liveness_check_is_the_deployment_state(deployer_two_nodes, starting_cluster_dep):
    v = deployer_two_nodes.view(starting_cluster_dep.id)
    assert [r["status"] for r in v["rank_summary"]] == ["starting", "starting"]


async def test_single_deployment_has_no_rank_summary(client_viewer_two_nodes, ready_remote_dep):
    d = client_viewer_two_nodes.get(f"/api/deployments/{ready_remote_dep.id}").json()
    assert d["rank_summary"] == [] and d["placement_label"] == "dgx-2"


async def test_cluster_sleep_label_in_the_view(client_viewer_two_nodes, ready_cluster_dep):
    d = client_viewer_two_nodes.get(f"/api/deployments/{ready_cluster_dep.id}").json()
    assert d["sleep_label"] == "sleep (stops all nodes)" and d["sleep_capable"] is False


def test_get_nodes_single_node_install(api_stack):
    from tests.test_api_deployments import admin
    admin(api_stack)
    r = api_stack.get("/dgxmm/api/nodes")
    assert r.status_code == 200, r.text
    (only,) = r.json()
    assert (only["id"], only["remote"], only["reachable"], only["stale"]) == ("local", False, True, False)
    assert "free_for_new" in only["budget"]


def test_get_nodes_unreachable_before_stale(client_viewer_two_nodes, api_two_nodes):
    """A probe whose last attempt failed: unreachable at once, though its reading is not stale yet."""
    class Probe:
        def reachable(self, node_id):
            return node_id == "local"
    api_two_nodes.ctx.node_probe = Probe()
    remote = client_viewer_two_nodes.get("/api/nodes").json()[1]
    assert remote["reachable"] is False and remote["stale"] is False


async def test_get_nodes_never_shows_rank_container_ids(client_viewer_two_nodes, deployer_two_nodes, ready_cluster_dep):
    await deployer_two_nodes.budget.refresh()  # usage rows of both ranks, with their container ids, in each budget
    assert any(u.container_id for u in deployer_two_nodes.budget.current("dgx-2").deployments)
    r = client_viewer_two_nodes.get("/api/nodes").json()
    raw = str(r)
    for rank in ready_cluster_dep.ranks:
        assert rank["container_id"] and rank["container_id"] not in raw
    assert all("container_id" not in u for n in r for u in n["budget"]["deployments"])
    assert {u["node"] for n in r for u in n["budget"]["deployments"]} == {"local", "dgx-2"}  # the share on both
