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
