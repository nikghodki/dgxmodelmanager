"""Processes view (spec §5): per-process fields, exclusions, container resolution, managed flag, redaction."""
import builtins
import io
import os
from pathlib import Path

from dgxmm.memory.hostmem import HostProbe
from tests.test_hostmem import MEMINFO, FakeGpu

CID = "a" * 64      # managed engine container
OTHER = "b" * 64    # unmanaged container (litellm-proxy)
GONE = "c" * 64     # container docker no longer knows about
SECRET = "s3cr3t-master-key-value"
HOST_CG = "/user.slice/user-1000.slice/session-3.scope"


def proc(root: Path, pid: int, argv: list[str] | None, rss_kb: int | None, cgroup: str = HOST_CG, *, name: str | None = None,
         uid: int = 1000, ppid: int = 1, state: str = "S (sleeping)", environ: str | None = None):
    d = root / str(pid)
    d.mkdir(parents=True)
    lines = [f"Name:\t{name or (Path(argv[0]).name[:15] if argv else 'kthread')}", f"State:\t{state}", f"PPid:\t{ppid}",
             f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}"]
    if rss_kb is not None:
        lines.append(f"VmRSS:\t{rss_kb} kB")
    (d / "status").write_text("\n".join(lines) + "\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in (argv or [])))
    (d / "environ").write_bytes(f"LITELLM_MASTER_KEY={environ or SECRET}\0".encode())
    return d


def scope(cid):
    return f"/system.slice/docker-{cid}.scope"


CONTAINERS = {
    CID: {"id": CID[:12], "name": "dgxmm-qwen", "image": "vllm/vllm-openai:v0.11", "deployment_id": 7},
    OTHER: {"id": OTHER[:12], "name": "litellm-proxy", "image": "ghcr.io/berriai/litellm:main-latest", "deployment_id": None},
}


LOOKUPS: list[tuple[str, bool]] = []


def lookup(cid, refresh=True):
    LOOKUPS.append((cid, refresh))
    return CONTAINERS.get(cid)


def build(settings, tmp_path, passwd=True):
    p = Path(settings.host_proc)
    (p / "meminfo").write_text(MEMINFO)
    if passwd:
        (tmp_path / "passwd").write_text("root:x:0:0:root:/root:/bin/bash\ndgxuser:x:1000:1000::/home/n:/bin/bash\n")
    settings = settings.model_copy(update={"host_passwd": str(tmp_path / "passwd")})
    proc(p, 2, None, None, "/", name="kthreadd", ppid=0, uid=0)
    proc(p, 50, None, None, "/", name="kworker/0:1", ppid=2, uid=0)
    proc(p, 60, ["defunct"], None, HOST_CG, state="Z (zombie)")
    proc(p, 70, ["sleep", "100"], 0)
    proc(p, 100, ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/m", "--api-key", SECRET],
         2_000_000, scope(CID), uid=0)
    proc(p, 101, ["python3", "-c", "from multiprocessing.spawn import spawn_main"], 500_000, scope(CID), uid=0, ppid=100)
    proc(p, 200, ["/bin/sh", "-c", f"DATABASE_URL=postgres://llm:{SECRET}@db:5432/litellm litellm --port 4000"],
         1_000_000, scope(OTHER), uid=0)
    proc(p, 300, ["python", "train.py", f"--hf-token={SECRET}"], 3_000_000, uid=1000)
    proc(p, 400, ["/usr/bin/weird"], 10, scope(GONE), uid=4242)
    gpu = FakeGpu({100: 40 * 2**30, 300: 1 * 2**30})
    return HostProbe(settings, gpu, containers=lookup)


def by_pid(items):
    return {i.pid: i for i in items}


def test_processes_fields_exclusions_and_sort(settings, tmp_path):
    probe = build(settings, tmp_path)
    ps = probe.processes({CID: {"deployment_id": 7, "name": "qwen"}})
    assert [p.pid for p in ps] == [100, 300, 200, 101, 400]  # rss+gpu desc; kthreads, zombie, zero-RSS dropped
    p = by_pid(ps)
    assert p[100].command == "python3 -m vllm.entrypoints.openai.api_server --model /m --api-key ***"
    assert p[100].container == {"id": CID[:12], "name": "dgxmm-qwen", "image": "vllm/vllm-openai:v0.11"}
    assert p[100].managed == {"deployment_id": 7, "name": "qwen"} and p[101].managed == p[100].managed
    # host accounts only for host processes; container uids are not host accounts
    assert (p[300].user, p[300].user_scope) == ("dgxuser", "host")
    assert (p[100].user, p[100].user_scope) == ("root", "container")
    assert (p[400].user, p[400].user_scope) == ("uid 4242", "container")
    assert p[100].gpu == 40 * 2**30 and p[100].rss == 2_000_000 * 1024 and p[101].ppid == 100
    assert p[200].container["name"] == "litellm-proxy" and p[200].managed is None
    assert "postgres://***@db:5432/litellm" in p[200].command
    assert p[300].container is None and p[300].command == "python train.py --hf-token=***"
    assert p[400].container == {"id": GONE[:12], "name": None, "image": None}
    assert p[100].name == "python3" and p[100].exe
    for item in ps:
        assert SECRET not in repr(item.as_dict())
        assert set(item.as_dict()) == {"pid", "ppid", "uid", "user", "user_scope", "name", "exe", "command",
                                       "command_truncated", "container", "managed", "rss", "gpu"}


def test_user_falls_back_to_numeric_uid_without_passwd(settings, tmp_path):
    probe = build(settings, tmp_path, passwd=False)
    p = by_pid(probe.processes({}))
    assert p[300].user == "1000" and p[100].user == "root"


def test_managed_flag_from_container_label_uses_the_deployment_name(settings, tmp_path):
    probe = build(settings, tmp_path)
    p = by_pid(probe.processes({}, {7: "qwen-old"}))
    assert p[100].managed == {"deployment_id": 7, "name": "qwen-old"}
    assert p[200].managed is None
    assert by_pid(probe.processes({}))[100].managed == {"deployment_id": 7, "name": None}  # never the container name


def test_sample_and_processes_share_one_managed_definition(settings, tmp_path):
    probe = build(settings, tmp_path)
    # label-managed container not in the active deployment rows: managed in /processes AND not "external" in sample
    s = probe.sample(set())
    managed_pids = {p.pid for p in probe.processes({}) if p.managed}
    assert managed_pids == {100, 101}
    assert not managed_pids & {p.pid for p in s.external_top}


def test_sample_uses_cache_only_container_lookups(settings, tmp_path):
    probe = build(settings, tmp_path)
    LOOKUPS.clear()
    probe.sample({CID})
    assert LOOKUPS and all(refresh is False for _, refresh in LOOKUPS)
    LOOKUPS.clear()
    probe.processes({})
    assert any(refresh is True for _, refresh in LOOKUPS)


def test_passwd_is_statted_once_per_scan(settings, tmp_path, monkeypatch):
    probe = build(settings, tmp_path)
    calls = []
    real = probe.users.refresh
    monkeypatch.setattr(probe.users, "refresh", lambda: (calls.append(1), real())[1])
    probe.processes({})
    assert len(calls) == 1


def test_sample_external_top_is_enriched_and_redacted(settings, tmp_path):
    probe = build(settings, tmp_path)
    s = probe.sample({CID})
    assert [p.pid for p in s.external_top] == [300, 200, 400]
    top = by_pid(s.external_top)
    assert top[300].command == "python train.py --hf-token=***" and top[300].user == "dgxuser"
    assert top[300].user_scope == "host" and (top[200].user, top[200].user_scope) == ("root", "container")
    assert top[300].container is None
    assert top[200].container == {"id": OTHER[:12], "name": "litellm-proxy", "image": "ghcr.io/berriai/litellm:main-latest"}
    assert SECRET not in repr(s.external_top)
    assert s.managed[CID] == 40 * 2**30  # no memory.stat in the fake cgroup: GPU only


def test_probe_never_reads_environ(settings, tmp_path, monkeypatch):
    probe = build(settings, tmp_path)
    opened: list[str] = []
    real_open, real_io_open = builtins.open, io.open

    def spy(file, *a, **kw):
        opened.append(os.fspath(file) if not isinstance(file, int) else str(file))
        return real_open(file, *a, **kw)

    def spy_io(file, *a, **kw):
        opened.append(os.fspath(file) if not isinstance(file, int) else str(file))
        return real_io_open(file, *a, **kw)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy_io)
    probe.processes({})
    probe.sample({CID})
    assert opened and any(f.endswith("cmdline") for f in opened)
    assert not [f for f in opened if os.path.basename(f) == "environ"]


def test_vanishing_process_mid_scan_is_skipped(settings, tmp_path):
    probe = build(settings, tmp_path)
    (Path(settings.host_proc) / "999").mkdir()  # exited between listdir and read
    os.remove(Path(settings.host_proc) / "300" / "cmdline")  # exited between status and cmdline
    p = by_pid(probe.processes({}))
    assert 999 not in p and 100 in p
    assert p.get(300) is None or p[300].command == ""


def test_long_command_is_truncated(settings, tmp_path):
    probe = build(settings, tmp_path)
    proc(Path(settings.host_proc), 500, ["python", "-c", "x" * 100_000], 9_000_000)
    p = by_pid(probe.processes({}))[500]
    assert p.command_truncated and len(p.command.encode()) <= 4096 + 3
