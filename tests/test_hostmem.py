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


# ---- per-process GPU memory of managed deployments (NVML) ----
CID2 = "b" * 64
OTHER = "c" * 64  # a container DGXMM does not manage


def _box(settings, anon=6 * 2**30):
    proc, cg = Path(settings.host_proc), Path(settings.host_cgroup)
    (proc / "meminfo").write_text(MEMINFO)
    for cid in (CID, CID2):
        scope = cg / "system.slice" / f"docker-{cid}.scope"
        scope.mkdir(parents=True)
        (scope / "memory.stat").write_text(f"anon {anon}\nkernel 0\n")
    # the API server and its engine core of one deployment, a second deployment, a foreign container, the desktop
    make_proc(proc, 100, "vllm", 500_000, f"/system.slice/docker-{CID}.scope")
    make_proc(proc, 101, "VLLM::EngineCore", 900_000, f"/system.slice/docker-{CID}.scope")
    make_proc(proc, 200, "llama-server", 100_000, f"/system.slice/docker-{CID2}.scope")
    make_proc(proc, 300, "python", 700_000, f"/system.slice/docker-{OTHER}.scope")
    make_proc(proc, 400, "Xorg", 300_000, "/user.slice/user-1000.slice/session-3.scope")
    return proc


def test_sample_sums_gpu_of_every_pid_in_a_managed_container(settings):
    _box(settings)
    gpu = {100: 1 * 2**30, 101: 47 * 2**30, 200: 3 * 2**30, 300: 2 * 2**30, 400: 2**28}
    s = HostProbe(settings, FakeGpu(gpu)).sample({CID, CID2})
    assert s.managed_gpu == {CID: 48 * 2**30, CID2: 3 * 2**30}
    assert s.managed[CID] == 6 * 2**30 + 48 * 2**30  # cgroup anon + every PID's NVML bytes, RSS not added
    assert s.managed[CID2] == 6 * 2**30 + 3 * 2**30
    # managed PIDs are never external; external processes keep rss + gpu
    assert {p.pid for p in s.external_top} == {300, 400}
    other = next(p for p in s.external_top if p.pid == 300)
    assert other.gpu == 2 * 2**30 and other.rss == 700_000 * 1024
    assert [p.pid for p in s.external_top] == [300, 400]  # sorted by rss + gpu


def test_sample_without_nvml_figures_is_cgroup_only(settings):
    """Older drivers / NVML unavailable / NVML blind to other PID namespaces: exactly the cgroup figure."""
    _box(settings)
    for gpu in (None, FakeGpu({}), FakeGpu({100: 0, 101: 0, 300: 0})):
        s = HostProbe(settings, gpu).sample({CID, CID2})
        assert s.managed == {CID: 6 * 2**30, CID2: 6 * 2**30}
        assert s.managed_gpu == {CID: 0, CID2: 0}
        assert {p.pid for p in s.external_top} == {300, 400}


def test_nvml_pid_not_in_host_proc_is_ignored(settings):
    """NVML PIDs are matched against the host /proc; a PID that is not there (exited, or a namespace-local PID)
    is neither managed nor external."""
    _box(settings)
    s = HostProbe(settings, FakeGpu({101: 40 * 2**30, 9999: 30 * 2**30})).sample({CID})
    assert s.managed[CID] == 6 * 2**30 + 40 * 2**30
    assert 9999 not in {p.pid for p in s.external_top}
    assert sum(p.gpu for p in s.external_top) == 0


def test_pid_namespace_nesting_is_detected():
    from dgxmm.memory.hostmem import nested_pid_namespace
    assert nested_pid_namespace("Name:\tpython\nNSpid:\t58394\n") is False  # host PID namespace (pid: host)
    assert nested_pid_namespace("Name:\tpython\nNSpid:\t70404\t7\n") is True  # a container's own namespace
    assert nested_pid_namespace("Name:\tpython\n") is False  # old kernel: unknown, do not warn


def test_probe_warns_when_nvml_cannot_see_other_pid_namespaces(settings, caplog, tmp_path):
    status = tmp_path / "status"
    status.write_text("NSpid:\t70404\t7\n")
    with caplog.at_level("WARNING"):
        HostProbe(settings, FakeGpu({}), self_status=str(status))
    assert "PID namespace" in caplog.text
    caplog.clear()
    status.write_text("NSpid:\t70404\n")
    with caplog.at_level("WARNING"):
        HostProbe(settings, FakeGpu({}), self_status=str(status))
        HostProbe(settings, None, self_status=str(tmp_path / "missing"))
    assert "PID namespace" not in caplog.text
