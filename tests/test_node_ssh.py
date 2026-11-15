"""Remote Docker over SSH as it runs in the DGXMM container (final review C1, C2, M11, M12).

docker-py's ssh:// transport (use_ssh_client=True) runs plain ``ssh -l <user> -- <host> docker system dial-stdio``:
no -F, and OpenSSH reads ~/.ssh/config from the passwd home, never $HOME. The image therefore puts DGXMM's ssh
wrapper (deploy/ssh/dgxmm-ssh) first on PATH; it adds ``-F <the config NodeRegistry writes>``."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dgxmm.config import Settings
from dgxmm.nodes.registry import NodeRegistry, ssh_config_path
from tests.fakes import FakeDockerClient

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "deploy" / "ssh" / "dgxmm-ssh"


def _remote(**over):
    node = {"id": "dgx-2", "label": "dgx-2", "docker_url": "ssh://dgxuser@192.0.2.2", "cluster_ip": "192.0.2.2",
            "ssh_user": "dgxuser", "known_hosts": "192.0.2.2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTests",
            "hf_cache_host": "/home/dgxuser/.cache/huggingface"}
    node.update(over)
    return [{"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock", "cluster_ip": "192.0.2.1"},
            node]


def test_ssh_config_lives_in_the_data_dir_never_under_home(settings_factory, tmp_path, monkeypatch):
    home = tmp_path / "a-home-that-does-not-exist"
    monkeypatch.setenv("HOME", str(home))
    s = settings_factory(nodes=_remote())
    NodeRegistry(s, client_factory=lambda n: FakeDockerClient())
    assert ssh_config_path(s) == Path(s.data_dir) / "ssh" / "config"
    assert ssh_config_path(s).is_file()
    assert not home.exists()  # nothing written to (or created under) any home directory


def test_first_start_creates_the_ssh_dir_and_its_missing_parents_0700(settings_factory, tmp_path):
    """C2: a fresh install: nothing exists yet under the data dir (the first multi-node start must not crash)."""
    data = tmp_path / "fresh" / "data"
    s = settings_factory(nodes=_remote(), data_dir=str(data))
    NodeRegistry(s, client_factory=lambda n: FakeDockerClient())
    for d in (data, data / "ssh"):
        assert d.is_dir() and oct(d.stat().st_mode & 0o777) == "0o700"
    assert oct(ssh_config_path(s).stat().st_mode & 0o777) == "0o600"


def test_dgxmm_ssh_config_env_overrides_the_path(settings_factory, tmp_path, monkeypatch):
    """The image sets DGXMM_SSH_CONFIG for the wrapper and the registry alike: one value."""
    target = tmp_path / "elsewhere" / "ssh" / "config"
    monkeypatch.setenv("DGXMM_SSH_CONFIG", str(target))
    s = settings_factory(nodes=_remote())
    NodeRegistry(s, client_factory=lambda n: FakeDockerClient())
    assert ssh_config_path(s) == target and target.is_file()


def test_generated_config_bounds_connects_and_dead_links(settings_factory):
    """M12: ConnectTimeout bounds a connect to a node that is down; ServerAliveCountMax a link that died."""
    s = settings_factory(nodes=_remote())
    NodeRegistry(s, client_factory=lambda n: FakeDockerClient())
    cfg = ssh_config_path(s).read_text()
    assert "  ConnectTimeout 10" in cfg and "  ServerAliveInterval 15" in cfg and "  ServerAliveCountMax 3" in cfg


def test_docker_url_host_must_be_the_cluster_ip(settings_factory):
    """M11: the ssh Host block is keyed by cluster_ip, so docker_url must name that address."""
    with pytest.raises(ValueError, match="docker_url host"):
        settings_factory(nodes=_remote(docker_url="ssh://dgxuser@192.0.2.9"))
    with pytest.raises(ValueError, match="docker_url host"):
        settings_factory(nodes=_remote(docker_url="ssh://dgxuser@dgx-host"))
    settings_factory(nodes=_remote(docker_url="ssh://dgxuser@192.0.2.2"))  # fine


def test_ssh_transport_of_docker_py_is_installed():
    """docker-py's ssh:// support needs paramiko at import time (docker.transport.SSHHTTPAdapter), even with
    use_ssh_client=True: without it DockerClient(base_url="ssh://...") raises."""
    import docker.transport
    assert hasattr(docker.transport, "SSHHTTPAdapter")
    deps = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'"docker\[ssh\]>=', deps)


def test_image_puts_the_wrapper_first_on_path_with_the_registry_config():
    df = (ROOT / "Dockerfile").read_text()
    assert "COPY deploy/ssh/dgxmm-ssh /usr/local/bin/ssh" in df
    m = re.search(r"DGXMM_SSH_CONFIG=(\S+)", df)
    assert m and m.group(1) == str(Path(Settings.model_fields["data_dir"].default) / "ssh" / "config")
    assert "libnss-wrapper" in df  # ssh as a uid with no passwd entry in the image
    assert WRAPPER.is_file() and os.access(WRAPPER, os.X_OK)
    compose = (ROOT / "docker-compose.yaml").read_text()
    assert "HOME: /app/data/home" not in compose  # nothing depends on HOME any more


def _docker_py_ssh_argv(monkeypatch, url_host: str) -> list[str]:
    """The exact argv docker-py's ssh:// transport spawns (use_ssh_client=True)."""
    import docker.transport.sshconn as sshconn
    seen = {}

    class _Popen:
        def __init__(self, args, **kw):
            seen["args"] = list(args)
    monkeypatch.setattr(sshconn.subprocess, "Popen", _Popen)
    sshconn.SSHSocket(url_host).connect()
    return seen["args"]


@pytest.mark.skipif(not Path("/usr/bin/ssh").exists(), reason="needs the OpenSSH client")
def test_docker_py_ssh_argv_resolves_through_the_registry_config(settings_factory, monkeypatch):
    """C1: run docker-py's own argv through the wrapper with -G: ssh prints the options it would use."""
    s = settings_factory(nodes=_remote())
    NodeRegistry(s, client_factory=lambda n: FakeDockerClient())
    argv = _docker_py_ssh_argv(monkeypatch, "dgxuser@192.0.2.2")
    assert argv[0] == "ssh" and argv[-1] == "docker system dial-stdio"
    monkeypatch.undo()
    env = {**os.environ, "DGXMM_SSH_CONFIG": str(ssh_config_path(s))}
    out = subprocess.run([str(WRAPPER), "-G", *argv[1:]], env=env, capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    opts = {}
    for line in out.stdout.splitlines():
        k, _, v = line.partition(" ")
        opts.setdefault(k, v)
    kh = Path(s.data_dir) / "ssh" / "known_hosts_dgx-2"
    assert opts["user"] == "dgxuser" and opts["hostname"] == "192.0.2.2"
    assert opts["userknownhostsfile"] == str(kh)
    assert opts["identityfile"] == str(s.node_ssh_key_path)
    assert opts["stricthostkeychecking"] == "true" and opts["batchmode"] == "yes"
    assert opts["connecttimeout"] == "10" and opts["serveralivecountmax"] == "3"


@pytest.mark.skipif(not Path("/usr/bin/ssh").exists(), reason="needs the OpenSSH client")
def test_wrapper_fails_closed_without_the_config(tmp_path):
    """No config written (no remote node): ssh refuses instead of falling back to a home ~/.ssh/config."""
    env = {**os.environ, "DGXMM_SSH_CONFIG": str(tmp_path / "missing")}
    out = subprocess.run([str(WRAPPER), "-G", "192.0.2.2"], env=env, capture_output=True, text=True, timeout=10)
    assert out.returncode != 0
