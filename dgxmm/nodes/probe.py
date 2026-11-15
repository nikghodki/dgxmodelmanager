"""Remote node memory probing via exec."""
from __future__ import annotations

import time
from typing import Callable

from dgxmm.memory.hostmem import HostMemory, parse_meminfo
from dgxmm.nodes.registry import NodeRegistry, RemoteDocker

PROBE_NAME = "dgxmm-node-probe"


class NodeProbe:
    """Memory probe for remote nodes via exec in probe container."""

    def __init__(self, registry: NodeRegistry, settings, clock: Callable[[], float] = time.monotonic):
        self.registry = registry
        self.settings = settings
        self.clock = clock
        self._samples: dict[str, HostMemory] = {}
        self._sample_times: dict[str, float] = {}
        self._reachable: dict[str, bool] = {}  # node -> whether its last sample attempt succeeded
        self._errors: dict[str, str] = {}  # node -> why its last sample attempt failed (GET /api/nodes)

    def _docker(self, node_id: str) -> RemoteDocker:
        """The node's controller (managed-label guard), with transport failures raised as NodeUnreachable."""
        return RemoteDocker(lambda: self.registry.docker(node_id), node_id)

    def ensure(self, node_id: str) -> None:
        """Ensure the probe container exists and is running on the given (remote) node (DockerController.ensure_probe)."""
        if self.registry.is_remote(node_id):
            node_config = self.registry.config(node_id)
            hf_cache_host = node_config.hf_cache_host or "/var/lib/huggingface"
            self._docker(node_id).ensure_probe(PROBE_NAME, self.settings.images.probe, hf_cache_host)

    def sample(self, node_id: str) -> HostMemory | None:
        """Sample memory from a node. For local, return None. For remote, exec cat /host/proc/meminfo."""
        if node_id == "local" or not self.registry.is_remote(node_id):
            return None

        try:
            self.ensure(node_id)
        except Exception as e:
            self._failed(node_id, f"probe container: {e}")
            raise

        try:
            output = self.exec(node_id, ["cat", "/host/proc/meminfo"])
            mem = parse_meminfo(output)
            self._samples[node_id] = mem
            self._sample_times[node_id] = self.clock()
            self._reachable[node_id] = True
            self._errors.pop(node_id, None)
            return mem
        except Exception as e:
            # Return last sample if exec fails (do not update timestamp)
            self._failed(node_id, f"meminfo exec: {e}")
            return self._samples.get(node_id)

    def _failed(self, node_id: str, why: str) -> None:
        self._reachable[node_id] = False
        self._errors[node_id] = why[:500]

    def reason(self, node_id: str) -> str | None:
        """Why the last sample attempt on a remote node failed (None: it succeeded, none yet, or the local node)."""
        return self._errors.get(node_id)

    def reachable(self, node_id: str) -> bool | None:
        """Whether the last sample attempt on a remote node succeeded (None: none yet; True for the local node). No
        I/O: GET /api/nodes reads it."""
        if node_id == "local" or not self.registry.is_remote(node_id):
            return True
        return self._reachable.get(node_id)

    def last(self, node_id: str) -> HostMemory | None:
        """The last successful sample of a node (no I/O: the remote probe loop takes the samples)."""
        return self._samples.get(node_id)

    def stale(self, node_id: str) -> bool:
        """Check if a node's sample is stale (>3x remote_probe_interval_s)."""
        if node_id == "local" or not self.registry.is_remote(node_id):
            return False

        last_time = self._sample_times.get(node_id)
        if last_time is None:
            return True

        age = self.clock() - last_time
        return age > 3 * self.settings.remote_probe_interval_s

    def exec(self, node_id: str, argv: list[str]) -> str:
        """Execute a command in the probe container and return stdout."""
        exit_code, output = self._docker(node_id).exec_managed(PROBE_NAME, argv)
        if exit_code != 0:
            raise RuntimeError(f"exec failed with code {exit_code}")

        return output.decode() if isinstance(output, bytes) else output
