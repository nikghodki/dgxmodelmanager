from pathlib import Path

from dgxmm.memory.hostmem import HostProbe, parse_meminfo

MEMINFO = """MemTotal:       127600524 kB
MemFree:        50703908 kB
MemAvailable:   97402460 kB
Buffers:          159568 kB
Cached:         47960852 kB
SwapTotal:      15999996 kB
SwapFree:       10485756 kB
"""
CID = "a" * 64


class FakeGpu:
    def __init__(self, procs):
        self.procs = procs

    def process_gpu_bytes(self):
        return self.procs

    def device_stats(self):
        return {"util": 50}


def make_proc(root: Path, pid: int, name: str, rss_kb: int, cgroup: str):
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "status").write_text(f"Name:\t{name}\nVmRSS:\t{rss_kb} kB\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "cmdline").write_bytes(name.encode() + b"\0--flag\0")


def test_parse_meminfo():
    m = parse_meminfo(MEMINFO)
    assert m.total == 127600524 * 1024 and m.available == 97402460 * 1024
    assert m.used == (127600524 - 97402460) * 1024


def test_sample_attributes_gpu_and_anon_to_managed_container(settings):
    proc, cg = Path(settings.host_proc), Path(settings.host_cgroup)
    (proc / "meminfo").write_text(MEMINFO)
    make_proc(proc, 100, "vllm", 2_000_000, f"/system.slice/docker-{CID}.scope")
    make_proc(proc, 200, "python", 1_000_000, "/user.slice/user-1000.slice/session-3.scope")
    make_proc(proc, 300, "kworker", 0, "/")
    scope = cg / "system.slice" / f"docker-{CID}.scope"
    scope.mkdir(parents=True)
    (scope / "memory.stat").write_text("anon 1000000\nfile 999999999\nkernel 50000\n")
    probe = HostProbe(settings, FakeGpu({100: 30 * 2**30, 200: 20 * 2**30}))
    s = probe.sample({CID})
    assert s.managed[CID] == 1_000_000 + 50_000 + 30 * 2**30
    assert s.managed_gpu[CID] == 30 * 2**30  # the NVML part, so callers can separate host memory
    assert [p.pid for p in s.external_top] == [200]
    assert s.external_top[0].gpu == 20 * 2**30 and s.external_top[0].rss == 1_000_000 * 1024
    assert s.gpu == {"util": 50}


def test_sample_tolerates_vanishing_processes_and_missing_cgroup(settings):
    proc = Path(settings.host_proc)
    (proc / "meminfo").write_text(MEMINFO)
    (proc / "999").mkdir()  # no status/cgroup files: process exited mid-scan
    s = HostProbe(settings, None).sample({CID})
    assert s.managed == {CID: 0}
    assert s.external_top == []
