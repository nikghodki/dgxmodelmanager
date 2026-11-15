"""The remote probe container as docker-py creates it (final review C3, T3, M14, M15)."""
import docker.errors
import pytest
import requests

from dgxmm.nodes.probe import PROBE_NAME, NodeProbe
from dgxmm.nodes.registry import NodeUnreachable
from tests.fakes import FakeContainer, FakeDockerClient


def test_fake_rejects_keywords_docker_py_rejects():
    """The fake validates containers.run/create keywords like docker-py (_create_container_args)."""
    fake = FakeDockerClient()
    with pytest.raises(TypeError, match="read_only_rootfs"):
        fake.containers.run("busybox", ["true"], detach=True, read_only_rootfs=True)
    with pytest.raises(TypeError, match="bogus"):
        fake.containers.create("busybox", ["true"], bogus=1)
    fake.containers.run("busybox", ["true"], detach=True, read_only=True, cap_drop=["ALL"], network_mode="none")


def _probe(two_node_registry):
    return NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: 0.0)


def test_ensure_creates_a_read_only_labelled_probe(two_node_registry):
    fake = two_node_registry.fake("dgx-2")
    _probe(two_node_registry).ensure("dgx-2")
    probes = [c for c in fake._containers.values() if c.name == PROBE_NAME]
    assert len(probes) == 1 and probes[0].status == "running"
    kw = fake.containers.last_run_kwargs
    assert kw["read_only"] is True and kw["network_mode"] == "none" and kw["cap_drop"] == ["ALL"]
    assert kw["security_opt"] == ["no-new-privileges"]
    assert kw["labels"] == {"dgxmm.managed": "true", "dgxmm.role": "probe"}
    mounts = {(m["Source"], m["Target"], m["ReadOnly"], m["Type"]) for m in kw["mounts"]}
    assert mounts == {("/proc", "/host/proc", True, "bind"),
                      ("/home/dgxuser/.cache/huggingface", "/hf", True, "bind")}


def test_ensure_replaces_a_probe_whose_start_fails(two_node_registry):
    """T3: a stopped probe that cannot start is removed and created again (no swallowed error, no duplicate)."""
    fake = two_node_registry.fake("dgx-2")
    old = FakeContainer(fake, "old-probe", PROBE_NAME, two_node_registry.settings.images.probe, ["sleep", "1"],
                        {"dgxmm.managed": "true", "dgxmm.role": "probe"}, {}, [], [], [], status="exited")
    fake._containers[old.id] = old
    fake.fail_container_start.add("old-probe")
    _probe(two_node_registry).ensure("dgx-2")
    probes = [c for c in fake._containers.values() if c.name == PROBE_NAME]
    assert len(probes) == 1 and probes[0].id != "old-probe" and probes[0].status == "running"


def test_ensure_never_touches_an_unmanaged_container_with_the_probe_name(two_node_registry):
    """M14: the managed-label guard applies to the probe too."""
    fake = two_node_registry.fake("dgx-2")
    other = fake.add_unmanaged(PROBE_NAME, image="busybox", status="exited")
    with pytest.raises(PermissionError):
        _probe(two_node_registry).ensure("dgx-2")
    assert other.status == "exited" and fake.run_calls == []


def test_ensure_maps_transport_errors_and_keeps_daemon_errors(two_node_registry):
    """M14: through RemoteDocker (NodeUnreachable); a daemon error on get is raised, never read as "no probe"."""
    fake = two_node_registry.fake("dgx-2")
    fake.raise_on_all = requests.exceptions.ConnectionError("ssh: connect to host 192.0.2.2: timed out")
    with pytest.raises(NodeUnreachable):
        _probe(two_node_registry).ensure("dgx-2")
    fake.raise_on_all = None
    fake.always_fail_get = True
    with pytest.raises(docker.errors.APIError):
        _probe(two_node_registry).ensure("dgx-2")
    assert fake.run_calls == []


def test_exec_goes_through_the_guarded_mapped_controller(two_node_registry):
    fake = two_node_registry.fake("dgx-2")
    fake.exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    p = _probe(two_node_registry)
    assert p.sample("dgx-2").total == 127000000 * 1024
    fake.raise_on_all = requests.exceptions.ConnectionError("broken pipe")
    with pytest.raises(NodeUnreachable):
        p.exec("dgx-2", ["true"])


def test_stale_reason_names_the_last_probe_failure(two_node_registry):
    """M15: why a node is stale (e.g. the probe could not be created: its hf_cache_host is missing)."""
    fake = two_node_registry.fake("dgx-2")
    fake.run_error = docker.errors.APIError(
        "invalid mount config for type \"bind\": bind source path does not exist: /home/dgxuser/.cache/huggingface")
    p = _probe(two_node_registry)
    with pytest.raises(docker.errors.APIError):
        p.sample("dgx-2")
    assert "bind source path does not exist" in p.reason("dgx-2")
    assert p.reason("local") is None
    fake.run_error = None
    fake.exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    p.sample("dgx-2")
    assert p.reason("dgx-2") is None
    fake.exec_fail = True
    p.sample("dgx-2")
    assert "exec" in p.reason("dgx-2")
