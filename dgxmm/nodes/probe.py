"""Remote node memory probing via exec."""
from __future__ import annotations

import time
from typing import Callable

from dgxmm.memory.hostmem import HostMemory, parse_meminfo
from dgxmm.nodes.registry import NodeRegistry


class NodeProbe:
    """Memory probe for remote nodes via exec in probe container."""

    def __init__(self, registry: NodeRegistry, settings, clock: Callable[[], float] = time.monotonic):
        self.registry = registry
        self.settings = settings
        self.clock = clock
        self._samples: dict[str, HostMemory] = {}
        self._sample_times: dict[str, float] = {}

    def ensure(self, node_id: str) -> None:
        """Ensure probe container exists on the given node."""
        if self.registry.is_remote(node_id):
            controller = self.registry.docker(node_id)
            docker = controller.client
            node_config = self.registry.config(node_id)
            
            # Check if probe container already exists
            try:
                docker.containers.get("dgxmm-node-probe")
                return
            except Exception:
                pass
            
            # Pull image if missing
            try:
                docker.images.get(self.settings.images.probe)
            except Exception:
                docker.images.pull(self.settings.images.probe)
            
            # Create probe container
            hf_cache_host = node_config.hf_cache_host or "/var/lib/huggingface"
            docker.containers.run(
                self.settings.images.probe,
                name="dgxmm-node-probe",
                command=["sleep", "2147483647"],
                detach=True,
                labels={"dgxmm.managed": "true"},
                network_mode="none",
                read_only_rootfs=True,
                cap_drop=["ALL"],
                mounts=[
                    {"Type": "bind", "Source": "/proc", "Target": "/host/proc", "ReadOnly": True},
                    {"Type": "bind", "Source": hf_cache_host, "Target": "/hf", "ReadOnly": True},
                ],
            )

    def sample(self, node_id: str) -> HostMemory | None:
        """Sample memory from a node. For local, return None. For remote, exec cat /host/proc/meminfo."""
        if node_id == "local" or not self.registry.is_remote(node_id):
            return None
        
        self.ensure(node_id)
        
        try:
            output = self.exec(node_id, ["cat", "/host/proc/meminfo"])
            mem = parse_meminfo(output)
            self._samples[node_id] = mem
            self._sample_times[node_id] = self.clock()
            return mem
        except Exception:
            # Return last sample if exec fails
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
        controller = self.registry.docker(node_id)
        docker = controller.client
        container = docker.containers.get("dgxmm-node-probe")
        
        exit_code, output = container.exec_run(argv)
        if exit_code != 0:
            raise RuntimeError(f"exec failed with code {exit_code}")
        
        return output.decode() if isinstance(output, bytes) else output
