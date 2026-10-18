import pytest

from dgxmm.config import Mount
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.base import ContainerSpec, managed_labels
from tests.fakes import FakeDockerClient


def spec(name="dgxmm-a", labels=None):
    return ContainerSpec(name=name, image="img", command=["--x"], environment={"A": "1"},
                         mounts=[Mount(host="/h", target="/t", mode="ro")], labels=labels or managed_labels(1, "vllm"),
                         network="gateway", port=8100, publish_bind="127.0.0.1")


def test_run_passes_expected_sdk_kwargs():
    sdk = FakeDockerClient()
    d = DockerController(sdk)
    cid = d.run(spec())
    call = sdk.run_calls[0]
    assert call["volumes"] == {"/h": {"bind": "/t", "mode": "ro"}}
    assert call["ports"] == {"8100/tcp": ("127.0.0.1", 8100)}
    assert call["ipc_mode"] == "host" and call["restart_policy"] == {"Name": "no"}
    assert call["device_requests"][0]["Capabilities"] == [["gpu"]]
    info = d.get(cid)
    assert info.managed and info.host_ports == {8100} and info.networks == ["gateway"]


def test_guard_refuses_unmanaged_containers():
    sdk = FakeDockerClient()
    other = sdk.add_unmanaged("litellm-proxy")
    d = DockerController(sdk)
    with pytest.raises(PermissionError):
        d.stop_and_remove(other.id)
    with pytest.raises(PermissionError):
        d.run(spec(labels={"foo": "bar"}))
    with pytest.raises(PermissionError):
        d.remove_by_name("litellm-proxy")
    assert sdk.container(other.id).stopped is False


def test_stop_and_remove_managed_and_missing():
    sdk = FakeDockerClient()
    d = DockerController(sdk)
    cid = d.run(spec())
    d.stop_and_remove(cid)
    assert d.get(cid) is None
    d.stop_and_remove(cid)       # already gone: no error
    d.stop_and_remove(None)


def test_list_managed_and_oneshot():
    sdk = FakeDockerClient()
    sdk.add_unmanaged("tts-service")
    d = DockerController(sdk)
    d.run(spec())
    assert [c.name for c in d.list_managed()] == ["dgxmm-a"]
    assert len(d.list_all()) == 2
    sdk.oneshot_output = "ARCHS=[]\n"
    assert d.run_oneshot("img", ["python3", "-c"], ["print(1)"]) == "ARCHS=[]\n"
    assert sdk.run_calls[-1]["remove"] is True
