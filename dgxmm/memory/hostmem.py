"""Read host memory, per-container usage and top external consumers."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dgxmm.config import Settings

log = logging.getLogger(__name__)
_DOCKER_SCOPE = re.compile(r"docker-([0-9a-f]{64})\.scope")


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    free: int
    cached: int
    swap_total: int
    swap_free: int

    @property
    def used(self) -> int:
        return self.total - self.available


@dataclass(frozen=True)
class ProcMem:
    pid: int
    name: str
    rss: int
    gpu: int  # no command line: it can carry secrets and /api/host is visible to viewers


@dataclass
class HostSample:
    mem: HostMemory
    managed: dict[str, int]
    external_top: list[ProcMem]
    gpu: dict = field(default_factory=dict)


class GpuSource(Protocol):
    def process_gpu_bytes(self) -> dict[int, int]: ...
    def device_stats(self) -> dict: ...


def parse_meminfo(text: str) -> HostMemory:
    vals: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            vals[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return HostMemory(total=vals.get("MemTotal", 0), available=vals.get("MemAvailable", 0),
                      free=vals.get("MemFree", 0), cached=vals.get("Cached", 0),
                      swap_total=vals.get("SwapTotal", 0), swap_free=vals.get("SwapFree", 0))


class NvmlGpu:
    """NVML access; on GB10 device memory is 'Not Supported' but per-process usage works."""

    def __init__(self):
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # no driver / no device in tests
            log.warning("NVML unavailable: %s", e)

    def process_gpu_bytes(self) -> dict[int, int]:
        if self._handle is None:
            return {}
        out: dict[int, int] = {}
        for fn in (self._nvml.nvmlDeviceGetComputeRunningProcesses, self._nvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                for p in fn(self._handle):
                    out[p.pid] = out.get(p.pid, 0) + int(p.usedGpuMemory or 0)
            except Exception:
                continue
        return out

    def device_stats(self) -> dict:
        if self._handle is None:
            return {}
        n, h, stats = self._nvml, self._handle, {}
        for key, fn in (("util", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu),
                        ("temp_c", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                        ("power_w", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000.0)):
            try:
                stats[key] = fn()
            except Exception:
                stats[key] = None
        return stats


class HostProbe:
    def __init__(self, settings: Settings, gpu: GpuSource | None):
        self.proc = Path(settings.host_proc)
        self.cgroup = Path(settings.host_cgroup)
        self.gpu = gpu

    def meminfo(self) -> HostMemory:
        return parse_meminfo((self.proc / "meminfo").read_text())

    def _cgroup_anon(self, cid: str) -> int:
        stat = self.cgroup / "system.slice" / f"docker-{cid}.scope" / "memory.stat"
        try:
            vals = dict(line.split() for line in stat.read_text().splitlines() if " " in line)
        except OSError:
            return 0
        kernel = int(vals.get("kernel", 0)) or sum(int(vals.get(k, 0)) for k in ("kernel_stack", "slab_unreclaimable", "sock"))
        return int(vals.get("anon", 0)) + kernel

    def sample(self, managed_ids: set[str], top_n: int = 8) -> HostSample:
        mem = self.meminfo()
        gpu_by_pid = self.gpu.process_gpu_bytes() if self.gpu else {}
        managed = {cid: self._cgroup_anon(cid) for cid in managed_ids}
        external: list[ProcMem] = []
        for d in self.proc.iterdir():
            if not d.name.isdigit():
                continue
            pid = int(d.name)
            try:
                cg = (d / "cgroup").read_text()
                status = (d / "status").read_text()
            except OSError:
                continue
            m = _DOCKER_SCOPE.search(cg)
            if m and m.group(1) in managed:
                managed[m.group(1)] += gpu_by_pid.get(pid, 0)
                continue
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            rss_kb = int(fields.get("VmRSS", "0 kB").split()[0])
            gpu = gpu_by_pid.get(pid, 0)
            if rss_kb == 0 and gpu == 0:
                continue
            external.append(ProcMem(pid=pid, name=fields.get("Name", "?").strip(), rss=rss_kb * 1024, gpu=gpu))
        external.sort(key=lambda p: p.rss + p.gpu, reverse=True)
        return HostSample(mem=mem, managed=managed, external_top=external[:top_n],
                          gpu=self.gpu.device_stats() if self.gpu else {})
