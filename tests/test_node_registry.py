"""NodeRegistry tests."""
import pytest

from dgxmm.config import LOCAL_NODE_ID
from dgxmm.nodes.registry import NodeRegistry, UnknownNode
from tests.fakes import FakeDockerClient


def test_registry_builds_local_controller_for_unix_url(settings):
    """Registry returns a DockerController for the local node."""
    reg = NodeRegistry(settings, client_factory=lambda n: FakeDockerClient())
    assert reg.ids() == ["local"] and not reg.is_remote("local")
    from dgxmm.docker_ctl import DockerController
    assert isinstance(reg.docker("local"), DockerController)


def test_registry_writes_ssh_config_with_pinned_known_hosts(two_node_settings, tmp_path, monkeypatch):
    """Registry writes SSH config with pinned known_hosts per remote node."""
    monkeypatch.setenv("HOME", str(tmp_path))
    NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    cfg = (tmp_path / ".ssh" / "config").read_text()
    assert "Host 192.0.2.2" in cfg and "StrictHostKeyChecking yes" in cfg
    assert f"IdentityFile {two_node_settings.node_ssh_key_file}" in cfg
    kh = (tmp_path / ".ssh" / "known_hosts_dgx-2").read_text()
    assert kh.strip() == two_node_settings.node("dgx-2").known_hosts
    assert oct((tmp_path / ".ssh" / "config").stat().st_mode & 0o777) == "0o600"


def test_unknown_node_raises(settings):
    """Registry raises UnknownNode for unknown node IDs."""
    with pytest.raises(UnknownNode):
        NodeRegistry(settings, client_factory=lambda n: FakeDockerClient()).docker("dgx-9")


def test_registry_returns_correct_node_ids(two_node_settings):
    """Registry returns all configured node IDs."""
    reg = NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    assert sorted(reg.ids()) == ["dgx-2", "local"]


def test_registry_identifies_remote_nodes(two_node_settings):
    """Registry correctly identifies remote vs local nodes."""
    reg = NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    assert not reg.is_remote("local")
    assert reg.is_remote("dgx-2")


def test_registry_returns_node_config(two_node_settings):
    """Registry returns the correct NodeConfig for a node ID."""
    reg = NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    node = reg.config("dgx-2")
    assert node.id == "dgx-2"
    assert node.cluster_ip == "192.0.2.2"
    assert node.docker_url == "ssh://dgxuser@192.0.2.2"


def test_registry_caches_docker_controllers(two_node_settings):
    """Registry caches DockerController instances."""
    reg = NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    docker1 = reg.docker("local")
    docker2 = reg.docker("local")
    assert docker1 is docker2


def test_registry_local_id_property(two_node_settings):
    """Registry exposes the local node ID."""
    reg = NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    assert reg.local_id == LOCAL_NODE_ID
