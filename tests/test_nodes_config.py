"""Node configuration tests."""
import pytest
from pydantic import ValidationError


def test_no_nodes_means_one_implicit_local_node(settings):
    nodes = settings.all_nodes()
    assert [n.id for n in nodes] == ["local"]
    assert nodes[0].docker_url == "unix:///var/run/docker.sock"
    assert nodes[0].hf_cache_host == settings.hf_cache.host


def test_two_nodes_parse(settings_factory):
    s = settings_factory(nodes=[
        {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"},
        {"id": "dgx-2", "label": "dgx-2", "docker_url": "ssh://dgxuser@192.0.2.2",
         "cluster_ip": "192.0.2.2", "ssh_user": "dgxuser",
         "known_hosts": "192.0.2.2 ssh-ed25519 AAAA...", "hf_cache_host": "/home/dgxuser/.cache/huggingface",
         "nccl": {"ib_hca": "rocep1s0f0,roceP2p1s0f0", "socket_ifname": "enp1s0f0np0"}}])
    assert s.node("dgx-2").cluster_ip == "192.0.2.2"
    assert s.node("dgx-2").publish_bind == "192.0.2.2"   # defaults to cluster_ip for remote nodes


def test_remote_node_requires_known_hosts_and_cluster_ip(settings_factory):
    with pytest.raises(ValidationError, match="known_hosts"):
        settings_factory(nodes=[{"id": "local", "label": "a", "docker_url": "unix:///var/run/docker.sock"},
                                {"id": "dgx-2", "label": "b", "docker_url": "ssh://u@198.51.100.2", "cluster_ip": "198.51.100.2"}])


def test_duplicate_or_missing_local_rejected(settings_factory):
    with pytest.raises(ValidationError, match="exactly one local node"):
        settings_factory(nodes=[{"id": "dgx-2", "label": "b", "docker_url": "ssh://u@198.51.100.2",
                                 "cluster_ip": "198.51.100.2", "known_hosts": "198.51.100.2 ssh-ed25519 AAAA"}])


def _remote(**extra):
    return [{"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"},
            {"id": "dgx-2", "label": "dgx-2", "docker_url": "ssh://dgxuser@192.0.2.2", "cluster_ip": "192.0.2.2",
             "known_hosts": "192.0.2.2 ssh-ed25519 AAAA", **extra}]


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "127.0.0.1", "::1", "224.0.0.1", "ff02::1", "255.255.255.255",
                                  "dgx-host"])
def test_remote_publish_bind_must_be_a_concrete_unicast_ip(settings_factory, bind):
    with pytest.raises(ValidationError, match="publish_bind"):
        settings_factory(nodes=_remote(publish_bind=bind))


@pytest.mark.parametrize("cluster_ip", ["0.0.0.0", "127.0.0.1"])
def test_remote_cluster_ip_default_bind_is_checked_too(settings_factory, cluster_ip):
    nodes = _remote()
    nodes[1]["cluster_ip"] = cluster_ip
    with pytest.raises(ValidationError, match="publish_bind"):
        settings_factory(nodes=nodes)


def test_remote_publish_bind_accepts_cluster_addresses(settings_factory):
    s = settings_factory(nodes=_remote(publish_bind="192.0.2.12"))
    assert s.node("dgx-2").publish_bind == "192.0.2.12" and s.node("dgx-2").engine_address() == "192.0.2.12"
    s = settings_factory(nodes=_remote())
    assert s.node("dgx-2").engine_address() == "192.0.2.2"


@pytest.mark.parametrize("node_id", ["DGX-2", "dgx_2", "-dgx2", "dgx 2", "a" * 33, ""])
def test_remote_node_id_must_match_the_node_id_pattern(settings_factory, node_id):
    """A remote id outside NODE_ID_PATTERN is refused at load (it would fail later, as a 500 on placement)."""
    nodes = _remote()
    nodes[1]["id"] = node_id
    with pytest.raises(ValidationError, match="node id"):
        settings_factory(nodes=nodes)


@pytest.mark.parametrize("node_id", ["dgx-2", "2", "a" * 32, "spark-b0"])
def test_remote_node_id_accepts_valid_ids(settings_factory, node_id):
    nodes = _remote()
    nodes[1]["id"] = node_id
    assert settings_factory(nodes=nodes).node(node_id).id == node_id
