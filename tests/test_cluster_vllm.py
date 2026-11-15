"""vLLM cluster deployments: one model across two nodes with tensor parallelism (Task 12)."""
import asyncio

import pytest

from dgxmm.config import GIB
from dgxmm.db import Deployment
from dgxmm.deployer.reconcile import reconcile
from dgxmm.deployer.service import DeployError, DeployRequest, Placement
from tests.nodes_fixtures import CLUSTER, DEEPSEEK, SMALL_VLLM


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1]


# ---------- engine: rank specs ----------

def test_cluster_specs_rank_flags(vllm_engine, cluster_launch):
    head, worker = vllm_engine.cluster_specs(**cluster_launch)
    assert (head.node, head.rank, worker.node, worker.rank) == ("local", 0, "dgx-2", 1)
    assert "--headless" not in head.spec.command and "--headless" in worker.spec.command
    for r in (head, worker):
        c = r.spec.command
        assert c[c.index("--master-addr") + 1] == "192.0.2.1" and c[c.index("--nnodes") + 1] == "2"
        assert r.spec.network_mode == "host" and r.spec.environment["NCCL_IB_HCA"] == "rocep1s0f0,roceP2p1s0f0"
    assert worker.spec.environment["VLLM_HOST_IP"] == "192.0.2.2"


def test_cluster_specs_host_networking_and_api(vllm_engine, cluster_launch):
    head, worker = vllm_engine.cluster_specs(**cluster_launch)
    assert _flag(head.spec.command, "--host") == "192.0.2.1" and _flag(head.spec.command, "--port") == "8123"
    assert "--host" not in worker.spec.command
    for r in (head, worker):
        s, c = r.spec, r.spec.command
        assert (_flag(c, "--tensor-parallel-size"), _flag(c, "--node-rank"), _flag(c, "--master-port")) == \
            ("2", str(r.rank), "29500")
        assert s.devices == ["/dev/infiniband"] and s.cap_add == ["IPC_LOCK"]
        assert s.ulimits == {"memlock": (-1, -1)} and s.shm_size == "16g" and s.publish_bind is None
        assert s.environment["NCCL_SOCKET_IFNAME"] == s.environment["GLOO_SOCKET_IFNAME"] == "enp1s0f0np0"
        assert s.name == f"dgxmm-ds-r{r.rank}"
        assert s.labels["dgxmm.rank"] == str(r.rank) and s.labels["dgxmm.deployment_id"] == "7"
        assert s.labels["dgxmm.managed"] == "true"
    assert head.spec.environment["VLLM_HOST_IP"] == "192.0.2.1"
    # the engine key is the head's (it serves the API); headless workers have none
    assert head.spec.environment["VLLM_API_KEY"] == "k-test" and "VLLM_API_KEY" not in worker.spec.environment
    # per-rank utilisation from that node's reservation and memory total
    assert _flag(head.spec.command, "--gpu-memory-utilization") == "0.10"
    assert _flag(worker.spec.command, "--gpu-memory-utilization") == "0.20"
    # the worker mounts the node's own HF cache read-only
    assert [(m.host, m.mode) for m in worker.spec.mounts] == [("/home/dgxuser/.cache/huggingface", "ro")]


def test_single_container_spec_unchanged_by_cluster_support(vllm_engine, cluster_launch):
    from dgxmm.engines.base import LaunchContext
    launch = LaunchContext(7, "ds", 8123, 20 * GIB, 121 * GIB, "local", api_key="k-test")
    spec = vllm_engine.container_spec(cluster_launch["artifact"], cluster_launch["cfg"], None, launch,
                                      cluster_launch["settings"])
    assert spec.network_mode is None and spec.name == "dgxmm-ds"
    assert spec.command[:8] == ["--model", DEEPSEEK, "--revision", "a" * 40, "--served-model-name",
                                "deepseek-tiny", "--host", "0.0.0.0"]
    assert "--nnodes" not in spec.command and "--headless" not in spec.command


# ---------- deploy / monitor / stop ----------

async def test_cluster_deploy_creates_one_container_per_node(deployer_two_nodes, deepseek_cluster_req):
    await deployer_two_nodes.deploy(deepseek_cluster_req, user_id=1)
    assert [len(deployer_two_nodes.fake(n).containers.list()) for n in ("local", "dgx-2")] == [1, 1]
    await deployer_two_nodes.aclose()


async def test_cluster_deploy_starts_workers_first_and_records_ranks(deployer_two_nodes, deepseek_cluster_req):
    d = deployer_two_nodes
    v = await d.deploy(deepseek_cluster_req, user_id=1)
    row = d._get(v["id"])
    head, worker = row.ranks
    assert (head["node"], head["rank"], worker["node"], worker["rank"]) == ("local", 0, "dgx-2", 1)
    assert int(worker["container_id"], 16) < int(head["container_id"], 16)  # the worker was created first
    assert row.container_id == head["container_id"] and row.state in ("starting", "loading")
    assert head["reserved"] + worker["reserved"] == row.reserved_bytes
    assert d.fake("local").containers.get(head["container_id"]).name == "dgxmm-ds-r0"
    assert d.fake("dgx-2").containers.get(worker["container_id"]).name == "dgxmm-ds-r1"
    assert v["placement"] == CLUSTER
    await d.aclose()


async def test_cluster_health_on_head_and_endpoints(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    v = d.view(ready_cluster_dep.id)
    assert d.base_url_for(v) == f"http://192.0.2.1:{ready_cluster_dep.port}"
    assert d.base_url_for(d._get(ready_cluster_dep.id)) == f"http://192.0.2.1:{ready_cluster_dep.port}"
    assert v["endpoint"] == v["host_endpoint"] == f"http://192.0.2.1:{ready_cluster_dep.port}/v1"
    hosts = {(h, p) for h, p, path in d.engines_http.hosts if path == "/health"}
    assert hosts == {("192.0.2.1", ready_cluster_dep.port)}
    assert v["sleep_capable"] is False


async def test_worker_exit_fails_whole_deployment_and_stops_head(deployer_two_nodes, ready_cluster_dep):
    deployer_two_nodes.fake("dgx-2").exit_container(ready_cluster_dep.ranks[1]["container_id"], code=1)
    await deployer_two_nodes.monitor_tick(ready_cluster_dep.id)
    assert deployer_two_nodes.state(ready_cluster_dep.id) == "failed"
    assert deployer_two_nodes.fake("local").containers.list() == []
    assert deployer_two_nodes.fake("dgx-2").containers.list() == []
    assert "rank 1" in deployer_two_nodes.row(ready_cluster_dep.id).last_error


async def test_head_exit_during_startup_fails_and_stops_worker(deployer_two_nodes, deepseek_cluster_req):
    from dgxmm.deployer.service import MonitorState
    d = deployer_two_nodes
    v = await d.deploy(deepseek_cluster_req, user_id=1)
    d._monitors.pop(v["id"]).cancel()
    head = d._get(v["id"]).ranks[0]
    d.fake("local").exit_container(head["container_id"], code=3)
    assert await d._monitor_tick(v["id"], MonitorState(start=d.clock())) is True
    assert d.state(v["id"]) == "failed" and d._get(v["id"]).exit_code == 3
    assert d.fake("dgx-2").containers.list() == [] and d.fake("local").containers.list() == []
    await d.aclose()


async def test_cluster_reservation_kept_until_all_ranks_gone(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    d.cluster_retry_s = 0.01
    deployer_two_nodes.fake("local").fail_remove = True
    await deployer_two_nodes.stop(ready_cluster_dep.id)
    assert d.state(ready_cluster_dep.id) == "stopping"
    await d.budget.refresh()
    assert d.budget.current("local").committed >= ready_cluster_dep.ranks_reserved("local")
    assert d.budget.current("dgx-2").committed == 0  # the worker is confirmed gone
    # removal is retried with backoff; once it succeeds the deployment is stopped and the memory released
    d.fake("local").fail_remove = False
    for _ in range(200):
        if d.state(ready_cluster_dep.id) == "stopped":
            break
        await asyncio.sleep(0.01)
    assert d.state(ready_cluster_dep.id) == "stopped"
    await d.budget.refresh()
    assert d.budget.current("local").committed == 0
    assert d.fake("local").containers.list() == []


async def test_cluster_stop_order_head_then_workers(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    order = []
    for node in ("local", "dgx-2"):
        fake = d.fake(node)
        for c in fake.containers.list():
            orig = c.stop
            c.stop = (lambda o, n: (lambda timeout=10: (order.append(n), o(timeout=timeout))))(orig, node)
    v = await d.stop(ready_cluster_dep.id)
    assert v["state"] == "stopped" and order == ["local", "dgx-2"]
    assert d.fake("local").containers.list() == [] == d.fake("dgx-2").containers.list()


async def test_cluster_restart_after_stop(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    await d.stop(ready_cluster_dep.id)
    v = await d.restart(ready_cluster_dep.id)
    assert v["state"] in ("starting", "loading", "ready")
    assert [len(d.fake(n).containers.list()) for n in ("local", "dgx-2")] == [1, 1]
    assert await d.wait_ready(ready_cluster_dep.id, 5) == "ready"


async def test_cluster_remove_removes_every_rank(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    await d.stop(ready_cluster_dep.id)
    d.fake("dgx-2").add_managed(ready_cluster_dep.id, name="dgxmm-ds-r1", rank=1, status="created")
    await d.remove(ready_cluster_dep.id)
    assert d._get(ready_cluster_dep.id) is None
    assert d.fake("dgx-2").containers.list() == []


# ---------- ruling R12: ports, budgets, weights, refusals ----------

async def test_cluster_ports_reserved_on_every_node(deployer_two_nodes, ready_cluster_dep, qwen_req):
    d = deployer_two_nodes
    head = d._get(ready_cluster_dep.id).ranks[0]
    lo, hi = d.settings.cluster_port_range
    assert (lo, hi) == (29500, 29599) and lo <= head["master_port"] <= hi
    for node in ("local", "dgx-2"):
        assert {ready_cluster_dep.port, head["master_port"]} <= d._db_ports(None, node)
    other = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert other["port"] != ready_cluster_dep.port
    local = await d.deploy(qwen_req.model_copy(update={"name": "q-local", "alias": "q-local"}), user_id=1)
    assert local["port"] != ready_cluster_dep.port


async def test_second_cluster_gets_other_master_port(deployer_two_nodes, ready_cluster_dep, deepseek_cluster_req):
    d = deployer_two_nodes
    v = await d.deploy(deepseek_cluster_req.model_copy(update={"name": "ds2", "alias": "ds2"}), user_id=1)
    a, b = d._get(ready_cluster_dep.id).ranks[0], d._get(v["id"]).ranks[0]
    assert a["master_port"] != b["master_port"] and v["port"] != ready_cluster_dep.port


async def test_cluster_usage_rows_one_per_node(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    rows = [r for r in d.usage_rows() if r[0] == ready_cluster_dep.id]
    assert sorted((r[5], r[3]) for r in rows) == sorted(
        (r["node"], r["reserved"]) for r in ready_cluster_dep.ranks)
    await d.budget.refresh()
    for node in ("local", "dgx-2"):
        u = [u for u in d.budget.current(node).deployments if u.deployment_id == ready_cluster_dep.id]
        assert len(u) == 1 and u[0].reserved == ready_cluster_dep.ranks_reserved(node)


async def test_cluster_does_not_fit_when_one_node_is_short(deployer_two_nodes, deepseek_cluster_req):
    from dgxmm.deployer.service import InsufficientMemory
    d = deployer_two_nodes
    d.set_free("dgx-2", 2 * GIB)
    with pytest.raises(InsufficientMemory) as e:
        await d.deploy(deepseek_cluster_req, user_id=1)
    assert e.value.extra.get("node") == "dgx-2"
    assert d.count_rows() == 0 and d.fake("local").containers.list() == []


async def test_cluster_weights_checked_on_every_node(deployer_two_nodes, deepseek_cluster_req):
    d = deployer_two_nodes
    d.fake("dgx-2").weights_on_node.discard(DEEPSEEK)
    with pytest.raises(DeployError) as e:
        await d.deploy(deepseek_cluster_req, user_id=1)
    assert e.value.code == "weights_missing" and e.value.extra["action"]["node"] == "dgx-2"
    assert d.count_rows() == 0


async def test_cluster_refuses_sleep_switch_mode_reconfigure(deployer_two_nodes, ready_cluster_dep, qwen_req,
                                                              deepseek_cluster_req):
    d = deployer_two_nodes
    dep_id = ready_cluster_dep.id
    calls = [d.sleep(dep_id), d.switch(dep_id, qwen_req.model_copy(update={"name": "new"}), [], 1),
             d.reconfigure(dep_id, dict(SMALL_VLLM, max_num_seqs=4))]
    for c in calls:
        with pytest.raises(DeployError) as e:
            await c
        assert e.value.code == "cluster_unsupported"
    with pytest.raises(DeployError) as e:
        d.set_mode(dep_id, "on_demand", None)
    assert e.value.code == "cluster_unsupported"
    with pytest.raises(DeployError) as e:
        await d.deploy(deepseek_cluster_req.model_copy(update={"name": "od", "alias": "od", "mode": "on_demand"}), 1)
    assert e.value.code == "cluster_unsupported"
    assert d.state(dep_id) == "ready"


async def test_llamacpp_cluster_still_not_implemented(deployer_two_nodes):
    req = DeployRequest(name="lc", engine="llamacpp", model_ref="Qwen/Qwen3-Tiny", placement=CLUSTER)
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(req, None)
    assert e.value.code == "cluster_not_implemented"


# ---------- reconcile / LiteLLM / watchdog ----------

async def test_reconcile_adopts_cluster_ranks(deployer_two_nodes, starting_cluster_dep):
    for rank in starting_cluster_dep.ranks:
        deployer_two_nodes.fake(rank["node"]).add_managed(deployment_id=starting_cluster_dep.id, rank=rank["rank"])
    await reconcile(deployer_two_nodes)
    assert deployer_two_nodes.count_rows() == 1 and all(r["container_id"] for r in deployer_two_nodes.row(starting_cluster_dep.id).ranks)
    assert deployer_two_nodes.has_monitor(starting_cluster_dep.id)
    await deployer_two_nodes.aclose()


async def test_reconcile_fails_cluster_with_missing_rank(deployer_two_nodes, starting_cluster_dep):
    d = deployer_two_nodes
    d.fake("local").add_managed(deployment_id=starting_cluster_dep.id, rank=0)
    await reconcile(d)
    assert d.state(starting_cluster_dep.id) == "failed" and d.count_rows() == 1
    assert d.fake("local").containers.list() == []


def test_cluster_litellm_entry_points_to_head(litellm_sync_two_nodes, ready_cluster_dep):
    want = litellm_sync_two_nodes.desired()
    api_base = next(v[0] for k, v in want.items() if k[0] == str(ready_cluster_dep.id))
    assert api_base.startswith("http://192.0.2.1:")


async def test_watchdog_victim_on_worker_node(deployer_two_nodes, ready_cluster_dep):
    from dgxmm.watchdog import Watchdog
    d = deployer_two_nodes
    w = Watchdog(d.settings, d, d.events, d.db)
    assert w._victim("dgx-2")["id"] == ready_cluster_dep.id


# ---------- launch failures and the start-up queue ----------

async def test_worker_start_failure_leaves_no_container(deployer_two_nodes, deepseek_cluster_req):
    d = deployer_two_nodes
    d.fake("dgx-2").fail_run = True
    with pytest.raises(DeployError) as e:
        await d.deploy(deepseek_cluster_req, user_id=1)
    assert e.value.code == "docker_error"
    assert d.fake("local").containers.list() == [] == d.fake("dgx-2").containers.list()
    row = d._get(d.list_views()[0]["id"])
    assert row.state == "failed" and d.gate.holders.get(row.id) is None


async def test_head_start_failure_removes_started_worker(deployer_two_nodes, deepseek_cluster_req):
    d = deployer_two_nodes
    d.fake("local").fail_run = True
    with pytest.raises(DeployError):
        await d.deploy(deepseek_cluster_req, user_id=1)
    assert d.fake("dgx-2").containers.list() == [] and d.fake("local").containers.list() == []
    v = d.list_views()[0]
    assert v["state"] == "failed" and "rank 0" in v["last_error"]
    await d.budget.refresh()
    assert d.budget.current("dgx-2").committed == 0


async def test_queued_cluster_launches_when_slot_frees(deployer_two_nodes, deepseek_cluster_req, qwen_req):
    d = deployer_two_nodes
    first = await d.deploy(qwen_req, user_id=1)  # holds the only start-up slot
    v = await d.deploy(deepseek_cluster_req, user_id=1)
    assert v["state"] == "queued"
    await d.budget.refresh()
    assert any(u.deployment_id == v["id"] for u in d.budget.current("dgx-2").deployments)  # reserved while queued
    d.engines_http.healthy.add(first["port"])
    assert await d.wait_ready(first["id"], 5) == "ready"
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    assert [len(d.fake(n).containers.list()) for n in ("local", "dgx-2")] == [2, 1]
    await d.aclose()
