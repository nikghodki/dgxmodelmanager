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
    assert "BatchMode yes" in cfg
    assert f"IdentityFile {two_node_settings.node_ssh_key_path}" in cfg
    kh_path = tmp_path / ".ssh" / "known_hosts_dgx-2"
    kh = kh_path.read_text()
    assert kh.strip() == two_node_settings.node("dgx-2").known_hosts
    assert oct((tmp_path / ".ssh" / "config").stat().st_mode & 0o777) == "0o600"
    assert oct(kh_path.stat().st_mode & 0o777) == "0o600"


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





def test_registry_raises_on_missing_cluster_ip(settings_factory, tmp_path, monkeypatch):
    """Registry raises if remote node lacks cluster_ip."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Note: Settings validator would catch this, but test defense in depth
    s = settings_factory(nodes=[
        {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"}
    ])
    # Manually create a broken node config bypassing Settings validator
    from dgxmm.config import NodeConfig
    from dgxmm.nodes.registry import NodeRegistry
    broken_node = NodeConfig(
        id="dgx-2", label="dgx-2", docker_url="ssh://dgxuser@192.0.2.2",
        cluster_ip=None, ssh_user="dgxuser", known_hosts="192.0.2.2 ssh-ed25519 AAAA"
    )
    s.nodes = [s.all_nodes()[0], broken_node]
    
    with pytest.raises(ValueError, match="remote node dgx-2 requires cluster_ip"):
        NodeRegistry(s, client_factory=lambda n: FakeDockerClient())


def test_registry_raises_on_missing_known_hosts(settings_factory, tmp_path, monkeypatch):
    """Registry raises if remote node lacks known_hosts."""
    monkeypatch.setenv("HOME", str(tmp_path))
    s = settings_factory(nodes=[
        {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"}
    ])
    from dgxmm.config import NodeConfig
    from dgxmm.nodes.registry import NodeRegistry
    broken_node = NodeConfig(
        id="dgx-2", label="dgx-2", docker_url="ssh://dgxuser@192.0.2.2",
        cluster_ip="192.0.2.2", ssh_user="dgxuser", known_hosts=None
    )
    s.nodes = [s.all_nodes()[0], broken_node]
    
    with pytest.raises(ValueError, match="remote node dgx-2 requires known_hosts"):
        NodeRegistry(s, client_factory=lambda n: FakeDockerClient())


def test_registry_derives_ssh_user_from_docker_url(settings_factory, tmp_path, monkeypatch):
    """Registry derives ssh_user from docker_url if not explicitly set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Create settings with ssh_user not set (will be derived from URL)
    s = settings_factory(nodes=[
        {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"},
        {"id": "dgx-2", "label": "dgx-2", "docker_url": "ssh://deriveduser@192.0.2.2",
         "cluster_ip": "192.0.2.2", "known_hosts": "192.0.2.2 ssh-ed25519 AAAA"}
    ])
    
    from dgxmm.nodes.registry import NodeRegistry
    NodeRegistry(s, client_factory=lambda n: FakeDockerClient())
    cfg = (tmp_path / ".ssh" / "config").read_text()
    assert "User deriveduser" in cfg


def test_registry_raises_if_ssh_user_not_derivable(settings_factory, tmp_path, monkeypatch):
    """Registry raises if ssh_user cannot be derived from docker_url."""
    monkeypatch.setenv("HOME", str(tmp_path))
    s = settings_factory(nodes=[
        {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"}
    ])
    from dgxmm.config import NodeConfig
    from dgxmm.nodes.registry import NodeRegistry
    # docker_url without user part
    broken_node = NodeConfig(
        id="dgx-2", label="dgx-2", docker_url="ssh://192.0.2.2",
        cluster_ip="192.0.2.2", ssh_user=None, known_hosts="192.0.2.2 ssh-ed25519 AAAA"
    )
    s.nodes = [s.all_nodes()[0], broken_node]
    
    with pytest.raises(ValueError, match="remote node dgx-2 requires ssh_user"):
        NodeRegistry(s, client_factory=lambda n: FakeDockerClient())

def test_registry_local_id_property(two_node_settings):
    """Registry exposes the local node ID."""
    reg = NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    assert reg.local_id == LOCAL_NODE_ID
