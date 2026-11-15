"""Multi-node test fixtures."""
from pathlib import Path

import pytest

from dgxmm.config import PathMapping, Settings


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
