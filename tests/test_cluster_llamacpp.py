"""llama.cpp cluster deployments: one GGUF across several nodes over the ggml RPC backend (Task 13). Workers run
ggml-rpc-server on their cluster IP; the head runs llama-server with --rpc and --tensor-split and serves the API."""
from pathlib import Path

import pytest

from dgxmm.config import GIB, Images
from dgxmm.deployer.service import DeployError, DeployRequest, Placement
from tests.nodes_fixtures import CLUSTER

SMALL_LLAMA = {"ctx_size": 4096, "parallel": 2}
CACHE_TARGET = "/root/.cache/llama.cpp"


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _hf_gguf(settings, repo="org/g", rev="abc", name="m-Q4_K_M.gguf"):
    from tests.conftest import write_gguf
    p = Path(settings.hf_cache.local) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / name
    write_gguf(p, layers=4, hidden=256, heads=4, kv=2, head_dim=64, ctx=4096, vocab=1000, pad_bytes=4 * GIB)
    return p


def _artifact(settings, path):
    from dgxmm.inventory.service import InventoryService
    inv = InventoryService(settings)
    inv.refresh()
    return inv.get(str(path))


@pytest.fixture
def llama_engine(two_node_settings):
    from dgxmm.engines.llamacpp import LlamaCppEngine
    return LlamaCppEngine(two_node_settings)


def _launch(engine, registry, nodes, rank_memory, rpc_port=50052, path=None):
    from dgxmm.engines.base import LaunchContext
    settings = registry.settings
    art = _artifact(settings, path or _hf_gguf(settings))
    launch = LaunchContext(7, "lc", 8123, sum(r for r, _ in rank_memory.values()), 121 * GIB, nodes[0],
                           api_key="k-test", rank_memory=rank_memory)
    return dict(artifact=art, cfg=engine.validate_config(dict(SMALL_LLAMA)), draft=None, launch=launch,
                placement=Placement(kind="cluster", nodes=nodes), nodes=registry, rpc_port=rpc_port,
                settings=settings)


@pytest.fixture
def cluster_launch(llama_engine, two_node_registry):
    """Keyword arguments of LlamaCppEngine.cluster_specs (overrides the vLLM fixture of the same name): head local,
    worker dgx-2, RPC port 50052."""
    return _launch(llama_engine, two_node_registry, ["local", "dgx-2"],
                   {"local": (12 * GIB, 121 * GIB), "dgx-2": (8 * GIB, 120 * GIB)})


@pytest.fixture
def cluster_launch_free_60_40(llama_engine, two_node_registry):
    """The reservations memory.estimate.per_rank_bytes gives when the worker (dgx-2) has 60 GiB free and the head
    (local) 40 GiB. llama.cpp puts the RPC devices in front of its local GPU, so --tensor-split lists the workers
    first (in --rpc order), then the head: 60,40."""
    from dgxmm.memory.estimate import per_rank_bytes
    kw = _launch(llama_engine, two_node_registry, ["local", "dgx-2"], {"local": (1, 1), "dgx-2": (1, 1)})
    est = llama_engine.estimate(kw["artifact"], kw["cfg"], None, kw["settings"])
    need = per_rank_bytes("llamacpp", est, {"local": 40 * GIB, "dgx-2": 60 * GIB})
    assert need["local"] != need["dgx-2"]
    from dataclasses import replace
    kw["launch"] = replace(kw["launch"], rank_memory={n: (need[n], 121 * GIB) for n in need})
    return kw


# ---------- engine: rank specs (brief) ----------

def test_llamacpp_cluster_specs_bind_rpc_to_cluster_ip(llama_engine, cluster_launch):
    head, worker = llama_engine.cluster_specs(**cluster_launch)
    assert worker.spec.command[:5] == ["ggml-rpc-server", "--host", "192.0.2.2", "-p", "50052"]
    assert head.spec.command[head.spec.command.index("--rpc") + 1] == "192.0.2.2:50052"


def test_tensor_split_from_budgets(llama_engine, cluster_launch_free_60_40):
    head, _ = llama_engine.cluster_specs(**cluster_launch_free_60_40)
    assert head.spec.command[head.spec.command.index("--tensor-split") + 1] == "60,40"


# ---------- engine: rank specs (rulings) ----------

def test_worker_rpc_server_binds_cluster_ip_never_all_interfaces(llama_engine, cluster_launch):
    """RPC has no authentication: the worker listens on its cluster IP only, with the local cache (-c)."""
    _, worker = llama_engine.cluster_specs(**cluster_launch)
    assert (worker.node, worker.rank) == ("dgx-2", 1)
    assert worker.spec.command == ["ggml-rpc-server", "--host", "192.0.2.2", "-p", "50052", "-c"]
    assert "0.0.0.0" not in worker.spec.command
    s = worker.spec
    assert s.network_mode == "host" and s.port is None and s.publish_bind is None
    assert s.devices == ["/dev/infiniband"] and s.cap_add == ["IPC_LOCK"] and s.ulimits == {"memlock": (-1, -1)}
    assert s.image == cluster_launch["settings"].images.llamacpp_rpc and s.gpus
    assert s.environment == {"LLAMA_CACHE": CACHE_TARGET}  # no engine key on a worker: it serves no API
    assert s.name == "dgxmm-lc-r1" and s.labels["dgxmm.rank"] == "1" and s.labels["dgxmm.engine"] == "llamacpp"
    assert s.labels["dgxmm.managed"] == "true" and s.labels["dgxmm.deployment_id"] == "7"


def test_rpc_cache_mount_is_the_only_writable_mount(llama_engine, cluster_launch):
    """The worker's -c cache lives under its node's HF cache dir, mounted read-write for that directory only; the
    model (HF cache) stays read-only; the worker does not mount the model at all."""
    head, worker = llama_engine.cluster_specs(**cluster_launch)
    assert [(m.host, m.target, m.mode) for m in worker.spec.mounts] == [
        ("/home/dgxuser/.cache/huggingface/llama.cpp-rpc-cache", CACHE_TARGET, "rw")]
    assert [(m.host, m.target, m.mode) for m in head.spec.mounts] == [
        (cluster_launch["settings"].hf_cache.host, "/root/.cache/huggingface", "ro")]


def test_head_serves_api_on_cluster_ip_with_single_node_flags(llama_engine, cluster_launch):
    head, _ = llama_engine.cluster_specs(**cluster_launch)
    s, c = head.spec, head.spec.command
    assert (head.node, head.rank) == ("local", 0) and c[0] == "llama-server"
    assert _flag(c, "--host") == "192.0.2.1" and _flag(c, "--port") == "8123" and "0.0.0.0" not in c
    assert _flag(c, "-m") == "/root/.cache/huggingface/hub/models--org--g/snapshots/abc/m-Q4_K_M.gguf"
    for flag, value in (("-c", "4096"), ("-np", "2"), ("-ngl", "999"), ("-ctk", "f16"), ("-ctv", "f16"),
                        ("-fa", "auto"), ("-b", "2048"), ("-ub", "512"), ("-a", "m-q4_k_m"), ("-lm", "none")):
        assert _flag(c, flag) == value, flag
    assert "--metrics" in c and "--slots" in c
    # the key in the environment, never argv
    assert s.environment == {"LLAMA_API_KEY": "k-test"} and "--api-key" not in c
    assert s.network_mode == "host" and s.port == 8123 and s.publish_bind is None
    assert s.devices == ["/dev/infiniband"] and s.cap_add == ["IPC_LOCK"] and s.ulimits == {"memlock": (-1, -1)}
    assert s.image == cluster_launch["settings"].images.llamacpp_rpc and s.name == "dgxmm-lc-r0"
    assert s.labels["dgxmm.rank"] == "0"


def test_extra_args_cannot_override_the_cluster_flags(llama_engine, cluster_launch):
    cluster_launch["cfg"] = llama_engine.validate_config({**SMALL_LLAMA, "extra_args": ["--jinja", "--no-mmap"]})
    head, _ = llama_engine.cluster_specs(**cluster_launch)
    c = head.spec.command
    assert c.index("--jinja") < c.index("--rpc") and c.index("--no-mmap") < c.index("--tensor-split")


def test_remote_head_and_local_worker(llama_engine, two_node_registry):
    kw = _launch(llama_engine, two_node_registry, ["dgx-2", "local"],
                 {"dgx-2": (12 * GIB, 120 * GIB), "local": (8 * GIB, 121 * GIB)})
    head, worker = llama_engine.cluster_specs(**kw)
    assert (head.node, worker.node) == ("dgx-2", "local")
    assert _flag(head.spec.command, "--host") == "192.0.2.2"
    assert _flag(head.spec.command, "--rpc") == "192.0.2.1:50052"
    assert worker.spec.command[:3] == ["ggml-rpc-server", "--host", "192.0.2.1"]
    assert [(m.host, m.mode) for m in head.spec.mounts] == [("/home/dgxuser/.cache/huggingface", "ro")]
    assert [(m.host, m.mode) for m in worker.spec.mounts] == [
        (kw["settings"].hf_cache.host + "/llama.cpp-rpc-cache", "rw")]


def test_remote_head_refuses_gguf_outside_hf_cache(llama_engine, two_node_registry):
    from dgxmm.engines.base import SpecError
    s = two_node_registry.settings
    from tests.conftest import write_gguf
    p = write_gguf(Path(s.gguf_roots[0].local) / "g" / "m-Q4_K_M.gguf", layers=4, hidden=256, heads=4, kv=2,
                   head_dim=64, ctx=4096, vocab=1000)
    kw = _launch(llama_engine, two_node_registry, ["dgx-2", "local"],
                 {"dgx-2": (GIB, 120 * GIB), "local": (GIB, 121 * GIB)}, path=p)
    with pytest.raises(SpecError) as e:
        llama_engine.cluster_specs(**kw)
    assert e.value.code == "not_on_node"


def test_local_head_serves_gguf_from_gguf_roots(llama_engine, two_node_registry):
    """Only the head reads the model file: a local head serves a GGUF from the local GGUF roots (read-only)."""
    s = two_node_registry.settings
    from tests.conftest import write_gguf
    p = write_gguf(Path(s.gguf_roots[0].local) / "g" / "m-Q4_K_M.gguf", layers=4, hidden=256, heads=4, kv=2,
                   head_dim=64, ctx=4096, vocab=1000)
    kw = _launch(llama_engine, two_node_registry, ["local", "dgx-2"],
                 {"local": (GIB, 121 * GIB), "dgx-2": (GIB, 120 * GIB)}, path=p)
    head, _ = llama_engine.cluster_specs(**kw)
    assert [(m.host, m.target, m.mode) for m in head.spec.mounts] == [("/host/gguf", "/models/0", "ro")]
    assert _flag(head.spec.command, "-m") == "/models/0/g/m-Q4_K_M.gguf"


def test_three_nodes_rpc_list_and_split_order(llama_engine, settings_factory, isolated_home):
    from dgxmm.nodes.registry import NodeRegistry
    from tests.fakes import FakeDockerClient

    def node(i):
        return {"id": f"dgx-{i}", "label": f"dgx-{i}", "docker_url": f"ssh://dgxuser@192.0.2.{i}",
                "cluster_ip": f"192.0.2.{i}", "ssh_user": "dgxuser",
                "known_hosts": f"192.0.2.{i} ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestingPurposesOnly{i}",
                "hf_cache_host": "/home/dgxuser/.cache/huggingface",
                "nccl": {"ib_hca": "rocep1s0f0", "socket_ifname": "enp1s0f0np0"}}
    local = {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock", "cluster_ip": "192.0.2.1",
             "nccl": {"ib_hca": "rocep1s0f0", "socket_ifname": "enp1s0f0np0"}}
    s = settings_factory(nodes=[local, node(2), node(3)])
    fakes = {n.id: FakeDockerClient() for n in s.nodes}
    reg = NodeRegistry(s, client_factory=lambda n: fakes[n.id])
    from dgxmm.engines.llamacpp import LlamaCppEngine
    engine = LlamaCppEngine(s)
    kw = _launch(engine, reg, ["local", "dgx-2", "dgx-3"], {"local": (1, 1), "dgx-2": (1, 1), "dgx-3": (1, 1)},
                 rpc_port=29501)
    est = engine.estimate(kw["artifact"], kw["cfg"], None, s)
    fixed = est.activations + est.runtime
    from dataclasses import replace
    kw["launch"] = replace(kw["launch"], rank_memory={"local": (fixed + 20 * GIB, 121 * GIB),
                                                      "dgx-2": (fixed + 50 * GIB, 121 * GIB),
                                                      "dgx-3": (fixed + 30 * GIB, 121 * GIB)})
    specs = engine.cluster_specs(**kw)
    assert [(r.node, r.rank) for r in specs] == [("local", 0), ("dgx-2", 1), ("dgx-3", 2)]
    head = specs[0].spec.command
    assert _flag(head, "--rpc") == "192.0.2.2:29501,192.0.2.3:29501"
    assert _flag(head, "--tensor-split") == "50,30,20"  # the workers in --rpc order, then the head's own GPU
    assert [r.spec.command[:5] for r in specs[1:]] == [["ggml-rpc-server", "--host", f"192.0.2.{i}", "-p", "29501"]
                                                       for i in (2, 3)]


def test_refuses_node_without_cluster_ip(three_node_registry):
    from dgxmm.engines.base import SpecError
    from dgxmm.engines.llamacpp import LlamaCppEngine
    engine = LlamaCppEngine(three_node_registry.settings)
    kw = _launch(engine, three_node_registry, ["local", "dgx-2"], {"local": (GIB, GIB), "dgx-2": (GIB, GIB)})
    with pytest.raises(SpecError) as e:
        engine.cluster_specs(**kw)
    assert e.value.code == "cluster_node_not_ready"


def test_single_node_spec_and_image_unchanged(llama_engine, cluster_launch):
    from dgxmm.engines.base import LaunchContext
    s = cluster_launch["settings"]
    spec = llama_engine.container_spec(cluster_launch["artifact"], cluster_launch["cfg"], None,
                                       LaunchContext(7, "lc", 8123, GIB, 121 * GIB, "local", api_key="k"), s)
    assert spec.image == s.images.llamacpp and spec.command[:4] == ["-m", spec.command[1], "--host", "0.0.0.0"]
    assert "--rpc" not in spec.command and "-lm" not in spec.command and spec.network_mode is None
    assert Images().llamacpp == ("ghcr.io/ggml-org/llama.cpp@sha256:"
                                 "e4cfff28678327b0ae7eda129e700e56a0fa55aa23db2f016256ea155c15c2b5")
    assert Images().llamacpp_rpc == "dgxmm-llamacpp-rpc:61dbf49"


# ---------- deployer: the Task 12 lifecycle with llama.cpp ranks ----------

@pytest.fixture
def llama_cluster_req(deployer_two_nodes):
    """A llama.cpp cluster request (head local, worker dgx-2) for a GGUF in both nodes' HF caches."""
    d = deployer_two_nodes
    p = _hf_gguf(d.settings)
    d.inventory.refresh()
    d.fake("dgx-2").weights_on_node.add("org/g")
    return DeployRequest(name="lc", engine="llamacpp", model_ref=d.inventory.get(str(p)).ref,
                         config=dict(SMALL_LLAMA), placement=CLUSTER)


@pytest.fixture
async def ready_llama_cluster(deployer_two_nodes, llama_cluster_req):
    d = deployer_two_nodes
    v = await d.deploy(llama_cluster_req, user_id=1)
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    yield d._get(v["id"])
    await d.aclose()


def _rpc_port(row):
    [w] = [r for r in row.ranks if r["rank"] == 1]
    return w["rpc_port"]


async def test_llamacpp_cluster_deploys_and_serves_from_head(deployer_two_nodes, ready_llama_cluster):
    d, row = deployer_two_nodes, ready_llama_cluster
    lo, hi = d.settings.cluster_port_range
    rpc = _rpc_port(row)
    assert lo <= rpc <= hi and "master_port" not in row.ranks[0]
    worker = d.fake("dgx-2").run_calls[-1]
    head = d.fake("local").run_calls[-1]
    assert worker["command"] == ["ggml-rpc-server", "--host", "192.0.2.2", "-p", str(rpc), "-c"]
    assert head["command"][0] == "llama-server" and _flag(head["command"], "--rpc") == f"192.0.2.2:{rpc}"
    assert "--tensor-split" in head["command"]
    assert worker["image"] == head["image"] == d.settings.images.llamacpp_rpc
    assert worker["network_mode"] == head["network_mode"] == "host"
    # health on the head only, at its cluster IP
    assert ("192.0.2.1", row.port, "/health") in d.engines_http.hosts
    assert sum(r["reserved"] for r in row.ranks) == row.reserved_bytes


async def test_rpc_port_counted_on_every_node(deployer_two_nodes, ready_llama_cluster, llama_cluster_req, qwen_req):
    d, row = deployer_two_nodes, ready_llama_cluster
    rpc = _rpc_port(row)
    for node in ("local", "dgx-2"):
        assert {row.port, rpc} <= d._db_ports(None, node)
    v = await d.deploy(llama_cluster_req.model_copy(update={"name": "lc2", "alias": "lc2"}), user_id=1)
    assert _rpc_port(d._get(v["id"])) != rpc and v["port"] != row.port
    other = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert other["port"] not in (row.port, rpc)


async def test_worker_exit_fails_the_whole_llamacpp_cluster(deployer_two_nodes, ready_llama_cluster):
    d, row = deployer_two_nodes, ready_llama_cluster
    d.fake("dgx-2").exit_container(row.ranks[1]["container_id"], code=1)
    await d.monitor_tick(row.id)
    assert d.state(row.id) == "failed" and "rank 1" in d.row(row.id).last_error
    assert d.fake("local").containers.list() == [] == d.fake("dgx-2").containers.list()


async def test_restart_keeps_rpc_port(deployer_two_nodes, ready_llama_cluster):
    d, row = deployer_two_nodes, ready_llama_cluster
    await d.stop(row.id)
    v = await d.restart(row.id)
    assert v["state"] in ("starting", "loading", "ready")
    new = d._get(row.id)
    assert _rpc_port(new) == _rpc_port(row) and new.port == row.port
    assert [len(d.fake(n).containers.list()) for n in ("local", "dgx-2")] == [1, 1]


async def test_gguf_on_the_head_only_is_admitted(deployer_two_nodes, llama_cluster_req):
    """Ruling R13: only the head reads the GGUF; the worker (dgx-2) is not even checked."""
    d = deployer_two_nodes
    d.fake("dgx-2").weights_on_node.discard("org/g")
    v = await d.deploy(llama_cluster_req, user_id=1)
    assert d.fake("dgx-2").exec_calls == []  # no presence check on the worker
    assert d.fake("dgx-2").run_calls[-1]["command"][0] == "ggml-rpc-server"
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    await d.aclose()


async def test_gguf_missing_on_a_remote_head_is_refused(deployer_two_nodes, llama_cluster_req):
    """Ruling R13: the head (here dgx-2) must hold the GGUF; weights_missing names the head and copies to it."""
    d = deployer_two_nodes
    d.fake("dgx-2").weights_on_node.discard("org/g")
    req = llama_cluster_req.model_copy(update={"placement": Placement(kind="cluster", nodes=["dgx-2", "local"])})
    with pytest.raises(DeployError) as e:
        await d.deploy(req, user_id=1)
    assert (e.value.code, e.value.status) == ("weights_missing", 409)
    assert e.value.extra["action"] == {"type": "copy_weights", "node": "dgx-2", "ref": "org/g", "revision": "abc"}
    assert d.fake("dgx-2").run_calls == [] == d.fake("local").run_calls and d.list_views() == []
    d.fake("dgx-2").weights_on_node.add("org/g")  # copied to the head: admitted
    v = await d.deploy(req, user_id=1)
    assert d._get(v["id"]).node == "dgx-2"
    assert d.fake("dgx-2").run_calls[-1]["command"][0] == "llama-server"
    await d.aclose()


def _no_nccl_deployer(d):
    for n in d.settings.nodes:
        n.nccl = None
    return d


def test_llamacpp_cluster_needs_cluster_ip_but_not_nccl(deployer_two_nodes):
    """Ruling R14: nccl is a vLLM requirement; llama.cpp clusters need cluster_ip on every node."""
    d = _no_nccl_deployer(deployer_two_nodes)
    p = Placement(kind="cluster", nodes=["local", "dgx-2"])
    d.validate_placement(p, "llamacpp")
    with pytest.raises(DeployError) as e:
        d.validate_placement(p, "vllm")
    assert e.value.code == "cluster_node_not_ready" and "nccl" in e.value.message
    d.settings.node("dgx-2").cluster_ip = None
    with pytest.raises(DeployError) as e:
        d.validate_placement(p, "llamacpp")
    assert e.value.code == "cluster_node_not_ready" and "cluster_ip" in e.value.message


def test_cluster_candidate_without_nccl_only_for_llamacpp(two_node_settings):
    from dgxmm.planner.placement import cluster_candidate
    for n in two_node_settings.nodes:
        n.nccl = None
    assert cluster_candidate(two_node_settings, "vllm") is None
    assert cluster_candidate(two_node_settings, "llamacpp").nodes == ["local", "dgx-2"]


async def test_llamacpp_cluster_deploys_without_nccl(deployer_two_nodes, llama_cluster_req):
    d = _no_nccl_deployer(deployer_two_nodes)
    v = await d.deploy(llama_cluster_req, user_id=1)
    assert [len(d.fake(n).containers.list()) for n in ("local", "dgx-2")] == [1, 1]
    assert d._get(v["id"]).placement["kind"] == "cluster"
    await d.aclose()


async def test_llamacpp_cluster_sleep_stops_every_rank_and_mode_works(deployer_two_nodes, ready_llama_cluster):
    """llama.cpp has no sleep: a llama.cpp cluster's "sleep" stops every rank; on-demand mode is allowed."""
    d, row = deployer_two_nodes, ready_llama_cluster
    assert d.set_mode(row.id, "on_demand", None)["mode"] == "on_demand"
    v = d.view(row.id)
    assert v["sleep_capable"] is False and v["sleep_label"] == "sleep (stops all nodes)"
    assert (await d.sleep(row.id))["state"] == "stopped"
    assert [d.fake(n).containers.list() for n in ("local", "dgx-2")] == [[], []]
