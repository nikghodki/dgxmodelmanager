"""Multi-node test fixtures."""
from pathlib import Path

import pytest

from dgxmm.config import GIB, PathMapping, Settings


@pytest.fixture
def settings_factory(tmp_path):
    """Factory fixture that creates Settings instances with optional overrides."""
    def _make(**overrides):
        for d in ("data", "static", "proc", "cgroup", "hf/hub", "gguf"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        defaults = {
            "data_dir": str(tmp_path / "data"),
            "static_dir": str(tmp_path / "static"),
            "host_proc": str(tmp_path / "proc"),
            "host_cgroup": str(tmp_path / "cgroup"),
            "hf_cache": PathMapping(host="/host/hf", local=str(tmp_path / "hf")),
            "gguf_roots": [PathMapping(host="/host/gguf", local=str(tmp_path / "gguf"))],
            "litellm_key_file": str(tmp_path / "litellm_key"),
            "hf_token_file": str(tmp_path / "hf_token"),
            "engine_base_host": "engines.test",
            "monitor_poll_s": 0.01,
        }
        defaults.update(overrides)
        return Settings(**defaults)
    return _make


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """NodeRegistry writes $HOME/.ssh/config for remote nodes: never the real one in tests."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def two_node_settings(settings_factory, isolated_home):
    """Settings with two nodes: local (dgx-1) and remote (dgx-2)."""
    return settings_factory(
        nodes=[
            {
                "id": "local",
                "label": "dgx-1",
                "docker_url": "unix:///var/run/docker.sock"
            },
            {
                "id": "dgx-2",
                "label": "dgx-2",
                "docker_url": "ssh://dgxuser@192.0.2.2",
                "cluster_ip": "192.0.2.2",
                "ssh_user": "dgxuser",
                "known_hosts": "192.0.2.2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestingPurposesOnly",
                "hf_cache_host": "/home/dgxuser/.cache/huggingface",
                "nccl": {
                    "ib_hca": "rocep1s0f0,roceP2p1s0f0",
                    "socket_ifname": "enp1s0f0np0"
                }
            }
        ]
    )


@pytest.fixture
def two_node_registry(two_node_settings):
    """NodeRegistry with fake Docker clients for testing."""
    from dgxmm.nodes.registry import NodeRegistry
    from tests.fakes import FakeDockerClient
    
    # Create a fake client for each node upfront
    fake_clients = {
        "local": FakeDockerClient(),
        "dgx-2": FakeDockerClient(),
    }
    
    def client_factory(node):
        return fake_clients[node.id]
    
    registry = NodeRegistry(two_node_settings, client_factory=client_factory)
    
    # Add a helper method to get the fake client for a node
    registry.fake = lambda node_id: fake_clients[node_id]
    
    return registry


@pytest.fixture
def budget_service_two_nodes(two_node_registry):
    """BudgetService with two nodes and deployments on each."""
    from dgxmm.memory.budget import BudgetService
    from dgxmm.memory.hostmem import HostMemory, HostSample
    from dgxmm.nodes.probe import NodeProbe
    
    # Set up memory samples for remote node
    two_node_registry.fake("dgx-2").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    
    # Create usage rows: one local, one remote
    def usage_provider():
        return [
            (1, "local-model", "ready", 10 * GIB, "local-cid", "local"),
            (2, "remote-model", "ready", 20 * GIB, "remote-cid", "dgx-2"),
        ]
    
    # Create a fake probe that returns HostSample for local
    class FakeLocalProbe:
        def sample(self, ids):
            return HostSample(
                mem=HostMemory(total=121 * GIB, available=100 * GIB, free=0, cached=0, swap_total=0, swap_free=0),
                managed={cid: 0 for cid in ids},
                external_top=[],
                gpu={},
                managed_gpu={}
            )
    
    # Create NodeProbe for remote nodes
    node_probe = NodeProbe(two_node_registry, two_node_registry.settings)
    
    # Monkey-patch NodeProbe to also handle local node
    orig_sample = node_probe.sample
    local_probe = FakeLocalProbe()
    
    def combined_sample(node_id_or_ids):
        if isinstance(node_id_or_ids, str):
            # Called with node_id
            if node_id_or_ids == "local":
                return None
            return orig_sample(node_id_or_ids)
        else:
            # Called with container ids (local probe)
            return local_probe.sample(node_id_or_ids)
    
    node_probe.sample = combined_sample
    
    budget = BudgetService(two_node_registry.settings, node_probe, usage_provider, nodes=two_node_registry)
    
    return budget


@pytest.fixture
def three_node_settings(settings_factory, isolated_home):
    """Settings with three nodes: local (dgx-1), dgx-2, dgx-3."""
    return settings_factory(
        nodes=[
            {
                "id": "local",
                "label": "dgx-1",
                "docker_url": "unix:///var/run/docker.sock"
            },
            {
                "id": "dgx-2",
                "label": "dgx-2",
                "docker_url": "ssh://dgxuser@192.0.2.2",
                "cluster_ip": "192.0.2.2",
                "ssh_user": "dgxuser",
                "known_hosts": "192.0.2.2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestingPurposesOnly",
                "hf_cache_host": "/home/dgxuser/.cache/huggingface",
                "nccl": {
                    "ib_hca": "rocep1s0f0,roceP2p1s0f0",
                    "socket_ifname": "enp1s0f0np0"
                }
            },
            {
                "id": "dgx-3",
                "label": "dgx-3",
                "docker_url": "ssh://dgxuser@192.0.2.3",
                "cluster_ip": "192.0.2.3",
                "ssh_user": "dgxuser",
                "known_hosts": "192.0.2.3 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestingPurposesOnly2",
                "hf_cache_host": "/home/dgxuser/.cache/huggingface",
                "nccl": {
                    "ib_hca": "rocep1s0f0,roceP2p1s0f0",
                    "socket_ifname": "enp1s0f0np0"
                }
            }
        ]
    )


@pytest.fixture
def three_node_registry(three_node_settings):
    """NodeRegistry with fake Docker clients for testing (3 nodes)."""
    from dgxmm.nodes.registry import NodeRegistry
    from tests.fakes import FakeDockerClient
    
    # Create a fake client for each node upfront
    fake_clients = {
        "local": FakeDockerClient(),
        "dgx-2": FakeDockerClient(),
        "dgx-3": FakeDockerClient(),
    }
    
    def client_factory(node):
        return fake_clients[node.id]
    
    registry = NodeRegistry(three_node_settings, client_factory=client_factory)
    
    # Add a helper method to get the fake client for a node
    registry.fake = lambda node_id: fake_clients[node_id]
    
    return registry


@pytest.fixture
def v40_db_with_one_deployment(tmp_path):
    """Database at version 40 (before node column) with one deployment."""
    import sqlite3
    from dgxmm.db import Database

    db_path = tmp_path / "v40.sqlite3"

    # Create fresh database (applies all migrations including 41)
    db = Database(f"sqlite:///{db_path}")
    db.engine.dispose()

    # Roll back migration 41: remove record and rebuild table without node
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DELETE FROM schema_version WHERE version = 41")

    # Get columns except node
    cols = [c[1] for c in conn.execute('PRAGMA table_info("deployments")') if c[1] != "node"]
    col_list = ", ".join(f'"{c}"' for c in cols)

    # Safe f-string: col_list built from PRAGMA table_info column names
    # Create new table without node, copy data, swap
    conn.execute(f"""
        CREATE TABLE deployments_new AS SELECT {col_list} FROM deployments
    """)
    conn.execute("DROP TABLE deployments")
    conn.execute("ALTER TABLE deployments_new RENAME TO deployments")

    # Insert test deployment
    conn.execute("""
        INSERT INTO deployments 
        (id, name, engine, model_ref, config, state, priority, reserved_bytes, 
         estimate, litellm_status, port, created_at, updated_at)
        VALUES (1, 'test-model', 'vllm', 'Qwen/Qwen3-0.6B', '{}', 'stopped', 
                0, 0, '{}', 'none', 8100, '2026-01-01 00:00:00', '2026-01-01 00:00:00')
    """)

    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    conn.close()

    return db_path


class FakeNodesProbe:
    """Budget probe for two-node tests: the local host sample (sample(container ids)) like FakeProbe, and per remote
    node a memory sample (sample(node_id)) and staleness (stale(node_id)) like NodeProbe."""

    def __init__(self, total_gib: float = 121, available_gib: float = 100, registry=None):
        from tests.fakes import FakeProbe
        self.registry = registry  # exec(): the node's FakeDockerClient answers, like NodeProbe's probe container
        self.local = FakeProbe(total_gib, available_gib)
        self.remote_total = int(total_gib * GIB)
        self.remote_available: dict[str, int] = {}
        self.stale_nodes: set[str] = set()
        self.default_available = int(available_gib * GIB)

    def meminfo(self):
        return self.local.meminfo()

    def sample(self, arg):
        from dgxmm.memory.hostmem import HostMemory
        if not isinstance(arg, str):
            return self.local.sample(arg)
        if arg == "local":
            return None
        avail = self.remote_available.get(arg, self.default_available)
        return HostMemory(total=self.remote_total, available=avail, free=0, cached=0, swap_total=0, swap_free=0)

    def stale(self, node_id: str) -> bool:
        return node_id in self.stale_nodes

    def exec(self, node_id: str, argv: list[str]) -> str:
        """NodeProbe.exec on the node's fake client (no probe container is listed on the node)."""
        client = self.registry.fake(node_id)
        client._maybe_raise()  # the node's transport is down
        code, out = client.exec(argv)
        if code != 0:
            raise RuntimeError(f"exec failed with code {code}")
        return out.decode()

    def processes(self, managed=None, names=None):
        return []


@pytest.fixture
def deployer_two_nodes(two_node_registry, tmp_path, fake_hf_repo):
    """Deployer with two nodes (local and remote dgx-2), real engines, a tiny Qwen model downloaded and a budget
    per node. Helpers on the returned deployer: fake(node), set_free(node, bytes), mark_stale(node),
    monitor_tick(dep_id), state(dep_id), engines_http."""
    from dgxmm.db import Database
    from dgxmm.deployer.service import Deployer
    from dgxmm.engines.registry import build_engines
    from dgxmm.events import EventBroker
    from dgxmm.inventory.service import InventoryService
    from dgxmm.memory.budget import BudgetService
    from tests.conftest import QWEN_TINY
    from tests.fakes import FakeEngines

    settings = two_node_registry.settings
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    db = Database(f"sqlite:///{tmp_path / 'deployer.sqlite3'}")
    inventory = InventoryService(settings)
    inventory.refresh()
    engines_http = FakeEngines()
    probe = FakeNodesProbe(registry=two_node_registry)
    two_node_registry.fake("dgx-2").weights_on_node = {"Qwen/Qwen3-Tiny"}  # dgx-2 has the model's snapshot
    deployer = Deployer(
        settings=settings,
        db=db,
        docker=two_node_registry.docker("local"),
        inventory=inventory,
        engines=build_engines(settings),
        http=engines_http.client(),
        events=EventBroker(),
        nodes=two_node_registry
    )
    deployer.budget = BudgetService(settings, probe, deployer.usage_rows, nodes=two_node_registry)
    deployer.engines_http = engines_http
    deployer.probe = probe
    deployer.fake = two_node_registry.fake

    def set_free(node_id: str, available: int) -> None:
        if node_id == "local":
            probe.local.available = available
        else:
            probe.remote_available[node_id] = available
    deployer.set_free = set_free
    deployer.mark_stale = probe.stale_nodes.add

    ticks: dict = {}

    async def monitor_tick(dep_id: int) -> bool:
        """One liveness check plus one monitor iteration (with the monitor's error accounting kept per deployment
        across ticks). True if the monitor would stop."""
        from dgxmm.deployer.service import MonitorState
        st = ticks.setdefault(dep_id, MonitorState(start=deployer.clock()))
        await deployer.check_liveness()
        return await deployer._monitor_tick(dep_id, st)
    deployer.monitor_tick = monitor_tick
    deployer.monitor_states = ticks
    deployer.state = lambda dep_id: deployer._get(dep_id).state
    return deployer


SMALL_VLLM = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


@pytest.fixture
def qwen_req():
    from dgxmm.deployer.service import DeployRequest
    return DeployRequest(name="qwen", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=dict(SMALL_VLLM))


@pytest.fixture
async def ready_remote_dep(deployer_two_nodes, qwen_req):
    """A vLLM deployment on dgx-2 that reached ready."""
    from types import SimpleNamespace
    d = deployer_two_nodes
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    return SimpleNamespace(id=v["id"], port=v["port"], view=d.view(v["id"]))


def insert_deployment(deployer, *, node: str, port: int, name: str = None):
    """Helper to insert a deployment directly into the database."""
    from dgxmm.db import Deployment
    from sqlalchemy import select
    
    if name is None:
        # Generate unique name
        with deployer.db.session() as s:
            count = s.scalar(select(Deployment.id).order_by(Deployment.id.desc()))
            name = f"model-{(count or 0) + 1}"
    
    with deployer.db.session() as s:
        d = Deployment(
            name=name,
            engine="vllm",
            model_ref="Qwen/Qwen3-0.6B",
            config={},
            state="ready",
            port=port,
            node=node
        )
        s.add(d)
        s.commit()


def row(db_path, query: str):
    """Execute a SELECT query and return the first row as a tuple."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    result = conn.execute(query).fetchone()
    conn.close()
    return result
