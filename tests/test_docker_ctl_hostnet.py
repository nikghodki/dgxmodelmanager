"""Host networking and RDMA container spec tests."""
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.base import ContainerSpec, SpecError, MANAGED_LABEL
from tests.fakes import FakeDockerClient
import pytest


def spec(**kw) -> ContainerSpec:
    """Build a minimal spec with MANAGED_LABEL and common defaults."""
    defaults = dict(
        name="test", image="test:latest", command=["serve"], environment={},
        mounts=[], labels={MANAGED_LABEL: "true"}, network="testnet", port=8000,
        publish_bind=None
    )
    defaults.update(kw)
    return ContainerSpec(**defaults)


def test_existing_spec_kwargs_unchanged(fake_docker):
    """Baseline: an existing spec (no new fields) produces the exact same kwargs as before."""
    ctl = DockerController(fake_docker)
    ctl.run(spec(
        name="vllm-model",
        image="vllm/vllm:latest",
        command=["python", "-m", "vllm.entrypoints.openai.api_server"],
        environment={"MODEL": "meta-llama/Llama-3.1-8B"},
        network="dgxmm-bridge",
        port=8000,
        publish_bind="198.51.100.1",
        gpus=True,
        ipc_host=True,
    ))
    kw = fake_docker.containers.last_run_kwargs
    assert kw["network"] == "dgxmm-bridge"
    assert kw["ports"] == {"8000/tcp": ("198.51.100.1", 8000)}
    assert kw["ipc_mode"] == "host"
    assert "device_requests" in kw
    assert "network_mode" not in kw
    assert "devices" not in kw
    assert "cap_add" not in kw
    assert "ulimits" not in kw
    assert "shm_size" not in kw


def test_run_maps_host_network_devices_caps_ulimits():
    """Map network_mode, devices, cap_add, ulimits, shm_size to docker-py kwargs."""
    fake_docker = FakeDockerClient()
    ctl = DockerController(fake_docker)
    ctl.run(spec(
        network_mode="host",
        devices=["/dev/infiniband"],
        cap_add=["IPC_LOCK"],
        ulimits={"memlock": (-1, -1)},
        shm_size="16g"
    ))
    kw = fake_docker.containers.last_run_kwargs
    assert kw["network_mode"] == "host"
    assert kw["cap_add"] == ["IPC_LOCK"]
    assert kw["shm_size"] == "16g"
    assert kw["devices"] == ["/dev/infiniband:/dev/infiniband:rwm"]
    assert [(u.name, u.soft, u.hard) for u in kw["ulimits"]] == [("memlock", -1, -1)]


def test_host_network_spec_publishes_no_ports():
    """With network_mode=host, do not pass network, ports, or attach to any network."""
    fake_docker = FakeDockerClient()
    ctl = DockerController(fake_docker)
    ctl.run(spec(network_mode="host", publish_bind="198.51.100.1"))
    kw = fake_docker.containers.last_run_kwargs
    assert kw["network_mode"] == "host"
    assert "ports" not in kw
    assert "network" not in kw


def test_host_network_with_internal_network_raises():
    """network_mode=host + internal_network=True is invalid."""
    fake_docker = FakeDockerClient()
    ctl = DockerController(fake_docker)
    with pytest.raises(SpecError) as exc:
        ctl.run(spec(network_mode="host", internal_network=True))
    assert exc.value.code == "host_network_conflict"


def test_multiple_devices_and_ulimits():
    """Multiple devices and ulimits are all mapped."""
    fake_docker = FakeDockerClient()
    ctl = DockerController(fake_docker)
    ctl.run(spec(
        devices=["/dev/infiniband", "/dev/nvidia0"],
        ulimits={"memlock": (-1, -1), "nofile": (65536, 65536)}
    ))
    kw = fake_docker.containers.last_run_kwargs
    assert set(kw["devices"]) == {
        "/dev/infiniband:/dev/infiniband:rwm",
        "/dev/nvidia0:/dev/nvidia0:rwm"
    }
    ulimits_by_name = {u.name: (u.soft, u.hard) for u in kw["ulimits"]}
    assert ulimits_by_name == {
        "memlock": (-1, -1),
        "nofile": (65536, 65536)
    }
