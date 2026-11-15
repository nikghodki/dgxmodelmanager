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
def two_node_settings(settings_factory):
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
def three_node_settings(settings_factory):
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
