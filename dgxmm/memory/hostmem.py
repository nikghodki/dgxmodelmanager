"""Read host memory, per-container usage and top external consumers."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

from dgxmm.config import Settings
from dgxmm.memory.redact import format_command

log = logging.getLogger(__name__)
_DOCKER_SCOPE = re.compile(r"(?:docker-|/docker/)([0-9a-f]{64})")
CMDLINE_READ_CAP = 64 * 1024  # bytes read from /proc/<pid>/cmdline; the output is capped at 4 KiB anyway
# Never read /proc/<pid>/environ: it holds secrets and nothing in the UI needs it.

# Container id → {"id": short id, "name", "image", "deployment_id": int | None} (read-only Docker lookup), or None.
ContainerLookup = Callable[[str], "dict | None"]


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
    gpu: int
    command: str = ""  # REDACTED (dgxmm.memory.redact): /api/host is visible to viewers
    user: str = ""
    container: dict | None = None  # {id, name, image}; None = host process


@dataclass(frozen=True)
class ProcInfo:
    """One row of GET /api/host/processes. `command` is redacted; the raw argv never leaves the probe."""
    pid: int
    ppid: int
    uid: int | None
    user: str
    name: str
    exe: str
    command: str
    command_truncated: bool
    container: dict | None
    managed: dict | None  # {deployment_id, name} when inside a DGXMM engine container
    rss: int
    gpu: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Proc:
    pid: int
    ppid: int
    uid: int | None
    name: str
    rss: int
    gpu: int
    cid: str | None


class UserNames:
    """uid → name from a (read-only mounted) host /etc/passwd, re-read when it changes; numeric fallback."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._mtime: float | None = None
        self._names: dict[int, str] = {}

    def _load(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime, self._names = None, {}
            return
        if mtime == self._mtime:
            return
        names: dict[int, str] = {}
        try:
            for line in self.path.read_text(errors="replace").splitlines():
                parts = line.split(":")
                if len(parts) >= 3 and parts[2].isdigit():
                    names.setdefault(int(parts[2]), parts[0])
        except OSError:
            return
        self._mtime, self._names = mtime, names

    def name(self, uid: int | None) -> str:
        if uid is None:
            return "?"
        self._load()
        return self._names.get(uid, str(uid))


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
    def __init__(self, settings: Settings, gpu: GpuSource | None, containers: ContainerLookup | None = None):
        self.proc = Path(settings.host_proc)
        self.cgroup = Path(settings.host_cgroup)
        self.gpu = gpu
        self.containers = containers
        self.users = UserNames(settings.host_passwd)

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

    def _scan(self, gpu_by_pid: dict[int, int]) -> Iterator[_Proc]:
        """Live user-space processes using memory. Skips kernel threads, zombies and zero-RSS/zero-GPU processes."""
        for d in self.proc.iterdir():
            if not d.name.isdigit():
                continue
            pid = int(d.name)
            try:
                status = (d / "status").read_text()
            except OSError:
                continue  # exited mid-scan
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            ppid = _int(fields.get("PPid"))
            if pid == 2 or ppid == 2 or fields.get("State", "").strip().startswith(("Z", "X")):
                continue  # kthreadd and its children, zombies, dead
            rss = _int(fields.get("VmRSS", "0 kB").split()[0] if fields.get("VmRSS") else "0") * 1024
            gpu = gpu_by_pid.get(pid, 0)
            if rss == 0 and gpu == 0:
                continue
            try:
                cg = (d / "cgroup").read_text()
            except OSError:
                continue
            m = _DOCKER_SCOPE.search(cg)
            uid_field = fields.get("Uid", "").split()
            yield _Proc(pid=pid, ppid=ppid, uid=_int(uid_field[0]) if uid_field else None,
                        name=fields.get("Name", "?").strip(), rss=rss, gpu=gpu, cid=m.group(1) if m else None)

    def _argv(self, pid: int) -> list[str]:
        try:
            with open(self.proc / str(pid) / "cmdline", "rb") as f:
                raw = f.read(CMDLINE_READ_CAP)
        except OSError:
            return []
        parts = raw.split(b"\0")
        if parts and parts[-1] == b"":
            parts.pop()
        return [a.decode("utf-8", "replace") for a in parts]

    def _exe(self, pid: int, fallback: str) -> str:
        try:
            return os.path.basename(os.readlink(self.proc / str(pid) / "exe")) or fallback
        except OSError:
            return fallback  # other users' processes: not readable without ptrace rights

    def _container(self, cid: str | None) -> dict | None:
        if cid is None:
            return None
        found = None
        if self.containers is not None:
            try:
                found = self.containers(cid)
            except Exception:  # Docker unavailable: still show the process, just unnamed
                log.debug("container lookup failed", exc_info=True)
        return found or {"id": cid[:12], "name": None, "image": None, "deployment_id": None}

    def processes(self, managed: dict[str, dict] | None = None) -> list[ProcInfo]:
        """Every live process using memory, biggest first. `managed` maps container id → {deployment_id, name}."""
        managed = managed or {}
        gpu_by_pid = self.gpu.process_gpu_bytes() if self.gpu else {}
        out: list[ProcInfo] = []
        for p in self._scan(gpu_by_pid):
            c = self._container(p.cid)
            mg = None
            if p.cid is not None:
                mg = managed.get(p.cid) or next((v for k, v in managed.items() if k and p.cid.startswith(k)), None)
                if mg is None and c.get("deployment_id") is not None:
                    mg = {"deployment_id": c["deployment_id"], "name": c.get("name")}
            command, truncated = format_command(self._argv(p.pid))
            out.append(ProcInfo(pid=p.pid, ppid=p.ppid, uid=p.uid, user=self.users.name(p.uid), name=p.name,
                                exe=self._exe(p.pid, p.name), command=command, command_truncated=truncated,
                                container=_public(c), managed=dict(mg) if mg else None, rss=p.rss, gpu=p.gpu))
        out.sort(key=lambda x: x.rss + x.gpu, reverse=True)
        return out

    def sample(self, managed_ids: set[str], top_n: int = 8) -> HostSample:
        mem = self.meminfo()
        gpu_by_pid = self.gpu.process_gpu_bytes() if self.gpu else {}
        managed = {cid: self._cgroup_anon(cid) for cid in managed_ids}
        external: list[_Proc] = []
        for p in self._scan(gpu_by_pid):
            if p.cid in managed:
                managed[p.cid] += p.gpu
                continue
            external.append(p)
        external.sort(key=lambda p: p.rss + p.gpu, reverse=True)
        top = [ProcMem(pid=p.pid, name=p.name, rss=p.rss, gpu=p.gpu, command=format_command(self._argv(p.pid))[0],
                       user=self.users.name(p.uid), container=_public(self._container(p.cid)))
               for p in external[:top_n]]
        return HostSample(mem=mem, managed=managed, external_top=top,
                          gpu=self.gpu.device_stats() if self.gpu else {})


def _int(v: str | None) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def _public(c: dict | None) -> dict | None:
    return None if c is None else {"id": c.get("id"), "name": c.get("name"), "image": c.get("image")}
