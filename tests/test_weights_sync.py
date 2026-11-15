"""Task 6: the weights presence check of remote deploys and the copy of a snapshot to a node (tar | ssh tar)."""
import asyncio
import os
import threading
from pathlib import Path

import pytest
from sqlalchemy import select

from dgxmm.config import GIB, NodeConfig
from dgxmm.deployer.service import DeployError
from dgxmm.nodes.weights_sync import (WEIGHTS_CHECK_SCRIPT, CopyError, WeightsSync, copy_argv, weights_present)

NODE = NodeConfig(id="dgx-2", label="dgx-2", docker_url="ssh://dgxuser@192.0.2.2", cluster_ip="192.0.2.2",
                  ssh_user="dgxuser", hf_cache_host="/home/dgxuser/.cache/huggingface")
REV = "a" * 40
TINY = "models--Qwen--Qwen3-Tiny"


# ---------- presence check before a remote deploy ----------

async def test_deploy_refuses_when_weights_missing_on_node(deployer_two_nodes, qwen_req):
    deployer_two_nodes.fake("dgx-2").weights_on_node = set()          # `test -f` → absent
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert e.value.code == "weights_missing" and e.value.extra["action"]["type"] == "copy_weights"
    assert e.value.status == 409
    assert e.value.extra["action"] == {"type": "copy_weights", "node": "dgx-2", "ref": "Qwen/Qwen3-Tiny",
                                       "revision": REV}
    assert deployer_two_nodes.list_views() == [] and deployer_two_nodes.fake("dgx-2").run_calls == []


async def test_deploy_checks_the_snapshot_files_off_the_loop_and_before_mem_lock(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    seen = []
    d.fake("dgx-2").exec_hooks.append(
        lambda cmd: seen.append((threading.current_thread() is threading.main_thread(), d.mem_lock.locked())))
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert d._get(v["id"]).node == "dgx-2"
    assert seen == [(False, False)]  # one check, in a worker thread, mem_lock free
    snap = f"/hf/hub/{TINY}/snapshots/{REV}"
    assert d.fake("dgx-2").exec_calls == [["sh", "-c", WEIGHTS_CHECK_SCRIPT, "sh",
                                           f"{snap}/config.json", f"{snap}/model.safetensors"]]
    assert d.fake("local").exec_calls == []


async def test_local_deploy_does_not_check_weights(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.fake("dgx-2").weights_on_node = set()
    await d.deploy(qwen_req, user_id=1)
    assert d.fake("dgx-2").exec_calls == [] and d.fake("local").exec_calls == []


async def test_stale_node_is_refused_before_the_weights_check(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.mark_stale("dgx-2")
    with pytest.raises(DeployError) as e:
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert (e.value.code, e.value.status) == ("node_stale", 503)
    assert d.fake("dgx-2").exec_calls == []


async def test_failing_weights_check_is_503_not_weights_missing(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.fake("dgx-2").weights_on_node = None
    d.fake("dgx-2").exec_fail = True  # the probe exec itself fails (not: the files are absent)
    with pytest.raises(DeployError) as e:
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert (e.value.code, e.value.status) == ("weights_check_failed", 503)
    assert d.list_views() == []


async def test_restart_on_remote_refuses_when_weights_are_gone(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    await d.stop(v["id"])
    d.fake("dgx-2").weights_on_node = set()
    with pytest.raises(DeployError) as e:
        await d.restart(v["id"], user_id=1)
    assert (e.value.code, e.value.status) == ("weights_missing", 409)
    assert d.state(v["id"]) == "stopped"


def test_weights_present_execs_in_the_probe_and_reads_its_answer(two_node_registry):
    from dgxmm.inventory.artifact import ModelArtifact
    from dgxmm.nodes.probe import NodeProbe
    art = ModelArtifact(ref="Qwen/Qwen3-Tiny", source="hf", local_path="/x", revision=REV, format="safetensors",
                        weights_bytes=1, size_bytes=1, complete=True, kind="llm", arch=None, architecture=None,
                        quant=None, files=["model.safetensors", "sub/tokenizer.json"])
    fake = two_node_registry.fake("dgx-2")
    fake.add_unmanaged("dgxmm-node-probe", image="probe", status="running")
    probe = NodeProbe(two_node_registry, two_node_registry.settings)
    fake.exec_output = ""  # the script prints nothing when a file is missing
    assert weights_present(probe, "dgx-2", art) is False
    fake.exec_output = "present\n"
    assert weights_present(probe, "dgx-2", art) is True
    snap = f"/hf/hub/{TINY}/snapshots/{REV}"
    assert fake.exec_calls[-1] == ["sh", "-c", WEIGHTS_CHECK_SCRIPT, "sh", f"{snap}/config.json",
                                   f"{snap}/model.safetensors", f"{snap}/sub/tokenizer.json"]


def test_weights_check_script_reports_present_only_when_every_file_is_there(tmp_path):
    import subprocess
    (tmp_path / "a").write_text("x")
    run = lambda *paths: subprocess.run(["sh", "-c", WEIGHTS_CHECK_SCRIPT, "sh", *paths],
                                        capture_output=True, text=True, check=True).stdout
    assert run(str(tmp_path / "a")) == "present\n"
    assert run(str(tmp_path / "a"), str(tmp_path / "missing")) == ""
    assert run(str(tmp_path / "a"), "$(touch " + str(tmp_path / "pwned") + ")") == ""  # arguments, never code
    assert not (tmp_path / "pwned").exists()


# ---------- copy_argv ----------

def test_copy_argv_is_two_argv_lists_with_pinned_paths(two_node_settings):
    tar, ssh = copy_argv("/hf-local", two_node_settings.node("dgx-2"), "models--unsloth--Qwen3.8-27B-NVFP4", "57926bac")
    assert tar[:3] == ["tar", "-C", "/hf-local/hub"] and "models--unsloth--Qwen3.8-27B-NVFP4" in tar
    assert ssh[0] == "ssh" and "192.0.2.2" in ssh
    assert ssh[-5:] == ["tar", "-C", "/home/dgxuser/.cache/huggingface/hub", "-xf", "-"]


def test_copy_rejects_path_traversal_ref():
    with pytest.raises(ValueError):
        copy_argv("/hf-local", NODE, "models--../../etc", "main")


@pytest.mark.parametrize("repo_dir", ["/etc/passwd", "no-prefix--a--b", "models--a", "models--a--b/../c",
                                      "models--a--b/x", "models--a--b c", "models---a--b"])
def test_copy_rejects_bad_repo_dirs(repo_dir):
    with pytest.raises(ValueError):
        copy_argv("/hf-local", NODE, repo_dir, "main")


@pytest.mark.parametrize("revision", ["", "..", "../main", "a/b", "-rf", "a b", "x" * 65])
def test_copy_rejects_bad_revisions(revision):
    with pytest.raises(ValueError):
        copy_argv("/hf-local", NODE, TINY, revision)


def test_copy_needs_cluster_ip_and_absolute_hf_cache():
    for bad in (NODE.model_copy(update={"cluster_ip": None}), NODE.model_copy(update={"hf_cache_host": None}),
                NODE.model_copy(update={"hf_cache_host": "~/.cache/huggingface"}),
                NODE.model_copy(update={"hf_cache_host": "/home/../etc"})):
        with pytest.raises(ValueError):
            copy_argv("/hf-local", bad, TINY, REV)


def test_copy_argv_lists_only_the_snapshot_its_blobs_and_refs(fake_hf_repo, settings):
    fake_hf_repo("Qwen/Qwen3-Tiny", config={"a": 1}, files={"model.safetensors": 10, "sub/t.json": 5})
    fake_hf_repo("Qwen/Qwen3-Tiny", rev="b" * 40, files={"other.safetensors": 7})  # another revision: not copied
    tar, ssh = copy_argv(settings.hf_cache.local, NODE, TINY, REV)
    blobs = sorted(f"{TINY}/blobs/{REV[:8]}-{n}" for n in ("config.json", "model.safetensors", "sub_t.json"))
    assert tar == ["tar", "-C", f"{settings.hf_cache.local}/hub", "-cf", "-", "--no-recursion", TINY, "--recursion",
                   *blobs, f"{TINY}/refs", f"{TINY}/snapshots/{REV}"]  # the links last: a cut copy has no dangling one
    from dgxmm.nodes.registry import ssh_config_path
    assert ssh == ["ssh", "-F", str(ssh_config_path()), "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes",
                   "-o", "ConnectTimeout=10", "-l", "dgxuser", "--", "192.0.2.2",
                   "tar", "-C", "/home/dgxuser/.cache/huggingface/hub", "-xf", "-"]
    assert not any(".." in a.split("/") for a in tar + ssh)


def test_copy_argv_refuses_a_snapshot_link_outside_its_blobs(fake_hf_repo, settings):
    snap = fake_hf_repo("Qwen/Qwen3-Tiny", files={"model.safetensors": 10})
    os.symlink("../../../models--other--repo/blobs/x", snap / "evil.safetensors")
    with pytest.raises(ValueError):
        copy_argv(settings.hf_cache.local, NODE, TINY, REV)


def test_copy_argv_round_trips_through_real_tar(fake_hf_repo, settings, tmp_path):
    """The tar side really archives only that snapshot, its blobs and refs (local tar, no ssh)."""
    import subprocess
    fake_hf_repo("Qwen/Qwen3-Tiny", config={"a": 1}, files={"model.safetensors": 10})
    fake_hf_repo("Qwen/Qwen3-Tiny", rev="b" * 40, files={"other.safetensors": 7})
    tar, _ssh = copy_argv(settings.hf_cache.local, NODE, TINY, REV)
    out = tmp_path / "remote-hub"
    out.mkdir()
    data = subprocess.run(tar, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-C", str(out), "-xf", "-"], input=data, check=True)
    got = sorted(str(p.relative_to(out)) for p in out.rglob("*") if not p.is_dir())
    assert got == sorted([f"{TINY}/blobs/{REV[:8]}-config.json", f"{TINY}/blobs/{REV[:8]}-model.safetensors",
                          f"{TINY}/refs/main", f"{TINY}/snapshots/{REV}/config.json",
                          f"{TINY}/snapshots/{REV}/model.safetensors"])
    assert (out / TINY / "snapshots" / REV / "model.safetensors").is_symlink()
    assert (out / TINY / "snapshots" / REV / "model.safetensors").stat().st_size == 10


# ---------- the copy job ----------

class FakeProc:
    def __init__(self, argv, kw, rc=0, stderr=b"", hang=False):
        self.argv, self.kw, self._rc, self._stderr = list(argv), kw, rc, stderr
        self.returncode = None
        self.killed = False
        self._exit = asyncio.Event()
        if not hang:
            self._exit.set()

    async def communicate(self, input=None):
        await self._exit.wait()
        self.returncode = self._rc
        return None, self._stderr

    async def wait(self):
        await self._exit.wait()
        self.returncode = self._rc
        return self.returncode

    def kill(self):
        self.killed = True
        self._rc = -9
        self._exit.set()


class FakeSpawn:
    """Stands in for asyncio.create_subprocess_exec: records argv, answers by program (tar / ssh)."""

    def __init__(self, **plans):
        self.plans = plans
        self.procs: list[FakeProc] = []

    async def __call__(self, *argv, **kw):
        p = FakeProc(argv, kw, **self.plans.get(argv[0], {}))
        self.procs.append(p)
        return p


@pytest.fixture
def sync(two_node_registry, fake_hf_repo, tmp_path):
    from dgxmm.db import Database
    from dgxmm.downloads import DownloadService
    from dgxmm.events import EventBroker
    fake_hf_repo("Qwen/Qwen3-Tiny", config={"a": 1}, files={"model.safetensors": 2 * GIB})
    settings = two_node_registry.settings
    db = Database(f"sqlite:///{tmp_path / 'sync.sqlite3'}")
    events = EventBroker()
    downloads = DownloadService(settings, db, None, None, events)

    def make(**plans):
        s = WeightsSync(settings, db, two_node_registry, events, downloads, spawn=FakeSpawn(**plans))
        return s
    make.downloads = downloads
    make.db = db
    return make


async def finished(downloads, dl_id, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        v = downloads.view(dl_id)
        if v["state"] not in ("queued", "running"):
            return v
        await asyncio.sleep(0.01)
    raise AssertionError(f"copy {dl_id} still {downloads.view(dl_id)['state']}")


async def test_copy_spawns_tar_piped_into_ssh_with_exact_argv(sync):
    s = sync()
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV, user_id=1)
    v = await finished(sync.downloads, job.id)
    assert v["state"] == "done" and v["error"] is None
    from dgxmm.nodes.registry import ssh_config_path
    tar_argv, ssh_argv = copy_argv(s.settings.hf_cache.local, s.nodes.config("dgx-2"), TINY, REV,
                                   ssh_config_path(s.settings))
    assert ssh_argv[2] == str(Path(s.settings.data_dir) / "ssh" / "config")
    tar, ssh = s.spawn.procs
    assert tar.argv == tar_argv and ssh.argv == ssh_argv
    # one pipe: tar writes it, ssh reads it; stderr captured; never a shell
    assert isinstance(tar.kw["stdout"], int) and isinstance(ssh.kw["stdin"], int)
    assert tar.kw["stderr"] == ssh.kw["stderr"] == asyncio.subprocess.PIPE
    assert tar.kw["stdin"] == ssh.kw["stdout"] == asyncio.subprocess.DEVNULL
    assert "shell" not in tar.kw and "shell" not in ssh.kw
    assert v["bytes_total"] == v["bytes_done"] > 2 * GIB


async def test_copy_appears_in_downloads_with_kind_copy(sync):
    s = sync()
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV, user_id=1)
    await finished(sync.downloads, job.id)
    [v] = sync.downloads.list()
    assert (v["id"], v["kind"], v["node"], v["repo_id"], v["revision"]) == (job.id, "copy", "dgx-2",
                                                                            "Qwen/Qwen3-Tiny", REV)


async def test_downloads_keep_kind_download(sync):
    from dgxmm.db import Download
    with sync.db.session() as s:
        s.add(Download(repo_id="a/b", revision="main", files=[], state="done"))
    [v] = sync.downloads.list()
    assert v["kind"] == "download" and v["node"] is None


async def test_failing_ssh_marks_job_failed_with_its_stderr(sync):
    s = sync(ssh={"rc": 1, "stderr": b"line 1\nPermission denied (publickey).\n"})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    v = await finished(sync.downloads, job.id)
    assert v["state"] == "failed"
    assert "ssh exited 1" in v["error"] and "Permission denied (publickey)." in v["error"]


async def test_failing_tar_fails_the_job_and_kills_ssh(sync):
    s = sync(tar={"rc": 2, "stderr": b"tar: models--Qwen--Qwen3-Tiny/blobs/x: Cannot open\n"}, ssh={"hang": True})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    v = await finished(sync.downloads, job.id)
    tar, ssh = s.spawn.procs
    assert v["state"] == "failed" and "tar exited 2" in v["error"] and "Cannot open" in v["error"]
    assert ssh.killed and not tar.killed


async def test_stderr_tail_is_bounded(sync):
    noise = b"".join(b"noise %d\n" % i for i in range(5000))
    s = sync(ssh={"rc": 255, "stderr": noise + b"the last line\n"})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    v = await finished(sync.downloads, job.id)
    assert v["error"].endswith("the last line") and len(v["error"]) < 1200


async def test_cancel_kills_both_processes(sync):
    s = sync(tar={"hang": True}, ssh={"hang": True})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    for _ in range(200):
        if len(s.spawn.procs) == 2 and sync.downloads.view(job.id)["state"] == "running":
            break
        await asyncio.sleep(0.01)
    sync.downloads.cancel(job.id)  # the Downloads page's cancel
    v = await finished(sync.downloads, job.id)
    assert v["state"] == "cancelled"
    assert [p.killed for p in s.spawn.procs] == [True, True]


async def test_spawn_failure_of_ssh_kills_tar_and_fails(sync):
    s = sync(tar={"hang": True})
    fake = s.spawn

    async def spawn(*argv, **kw):
        if argv[0] == "ssh":
            raise FileNotFoundError("No such file or directory: 'ssh'")
        return await fake(*argv, **kw)
    s.spawn = spawn
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    v = await finished(sync.downloads, job.id)
    assert v["state"] == "failed" and "ssh" in v["error"]
    [tar] = fake.procs
    assert tar.killed


@pytest.mark.parametrize("node,ref,rev,code,status", [
    ("local", "Qwen/Qwen3-Tiny", REV, "local_node", 400),
    ("dgx-9", "Qwen/Qwen3-Tiny", REV, "unknown_node", 400),
    ("dgx-2", "not-a-valid-ref", REV, "invalid_ref", 400),
    ("dgx-2", "../../etc/passwd", REV, "invalid_ref", 400),
    ("dgx-2", "Qwen/Qwen3-Tiny", "../x", "invalid_revision", 400),
    ("dgx-2", "Qwen/Qwen3-Tiny", "b" * 40, "not_local", 404),
    ("dgx-2", "org/absent", REV, "not_local", 404),
])
async def test_copy_refusals(sync, node, ref, rev, code, status):
    s = sync()
    with pytest.raises(CopyError) as e:
        await s.copy_weights(node, ref, rev)
    assert (e.value.code, e.value.status) == (code, status)
    assert sync.downloads.list() == [] and s.spawn.procs == []


async def test_one_copy_of_a_snapshot_to_a_node_at_a_time(sync):
    s = sync(tar={"hang": True}, ssh={"hang": True})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    with pytest.raises(CopyError) as e:
        await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    assert (e.value.code, e.value.status) == ("already_copying", 409)
    sync.downloads.cancel(job.id)
    await finished(sync.downloads, job.id)


# ---------- API ----------

@pytest.fixture
def nodes_api(two_node_registry, fake_hf_repo, tmp_path):
    from fastapi.testclient import TestClient
    from dgxmm.app import create_app
    from dgxmm.auth.passwords import hash_password
    from dgxmm.auth.ratelimit import SlidingWindowLimiter
    from dgxmm.context import AppContext
    from dgxmm.db import Database, User
    from dgxmm.downloads import DownloadService
    from dgxmm.events import EventBroker
    fake_hf_repo("Qwen/Qwen3-Tiny", config={"a": 1}, files={"model.safetensors": 10})
    settings = two_node_registry.settings
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    db = Database(f"sqlite:///{settings.db_path}")
    events = EventBroker()
    downloads = DownloadService(settings, db, None, None, events)
    spawn = FakeSpawn()
    ctx = AppContext(settings=settings, db=db, events=events, login_limiter=SlidingWindowLimiter(100, 60),
                     nodes=two_node_registry, downloads=downloads,
                     weights_sync=WeightsSync(settings, db, two_node_registry, events, downloads, spawn=spawn))
    with db.session() as s:
        s.add(User(username="root", pw_hash=hash_password("root-password-123"), role="admin"))
        s.add(User(username="vic", pw_hash=hash_password("viewer-password-1"), role="viewer"))
    with TestClient(create_app(ctx), base_url="http://testserver") as c:
        c.ctx, c.spawn = ctx, spawn
        yield c


def _login(c, user, pw):
    r = c.post("/dgxmm/api/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}


URL = "/dgxmm/api/nodes/{}/copy-weights"


def test_api_copy_weights_returns_job_and_is_audited(nodes_api):
    from dgxmm.db import AuditLog
    h = _login(nodes_api, "root", "root-password-123")
    r = nodes_api.post(URL.format("dgx-2"), json={"ref": "Qwen/Qwen3-Tiny", "revision": REV}, headers=h)
    assert r.status_code == 202, r.text
    v = r.json()
    assert (v["kind"], v["node"], v["repo_id"], v["revision"]) == ("copy", "dgx-2", "Qwen/Qwen3-Tiny", REV)
    listed = nodes_api.get("/dgxmm/api/downloads").json()
    assert [(d["id"], d["kind"]) for d in listed] == [(v["id"], "copy")]
    with nodes_api.ctx.db.session() as s:
        a = s.scalars(select(AuditLog).where(AuditLog.action == "copy_weights")).one()
        assert (a.target, a.detail) == ("Qwen/Qwen3-Tiny", {"node": "dgx-2", "revision": REV, "download_id": v["id"]})


def test_api_copy_weights_is_admin_only_and_needs_csrf(nodes_api):
    body = {"ref": "Qwen/Qwen3-Tiny", "revision": REV}
    _login(nodes_api, "root", "root-password-123")
    assert nodes_api.post(URL.format("dgx-2"), json=body).status_code == 403  # no X-CSRF-Token
    hv = _login(nodes_api, "vic", "viewer-password-1")
    r = nodes_api.post(URL.format("dgx-2"), json=body, headers=hv)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required"
    assert nodes_api.spawn.procs == [] and nodes_api.ctx.downloads.list() == []


@pytest.mark.parametrize("node,body,code", [
    ("dgx-9", {"ref": "Qwen/Qwen3-Tiny", "revision": REV}, "unknown_node"),
    ("local", {"ref": "Qwen/Qwen3-Tiny", "revision": REV}, "local_node"),
    ("dgx-2", {"ref": "../../etc", "revision": REV}, "invalid_ref"),
    ("dgx-2", {"ref": "Qwen/Qwen3-Tiny", "revision": "../../x"}, "invalid_revision"),
])
def test_api_copy_weights_refusals_are_400(nodes_api, node, body, code):
    h = _login(nodes_api, "root", "root-password-123")
    r = nodes_api.post(URL.format(node), json=body, headers=h)
    assert r.status_code == 400 and r.json()["detail"]["code"] == code, r.text
    assert nodes_api.spawn.procs == []


# ---------- fix round 1 ----------

def _hf_gguf(settings, repo="org/g", rev="abc", name="m-Q4_K_M.gguf"):
    from tests.conftest import write_gguf
    p = Path(settings.hf_cache.local) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / name
    write_gguf(p, pad_bytes=GIB)
    return p


async def test_remote_gguf_absent_on_node_is_refused(deployer_two_nodes):
    from dgxmm.deployer.service import DeployRequest
    d = deployer_two_nodes
    p = _hf_gguf(d.settings)
    d.inventory.refresh()
    req = DeployRequest(name="g", engine="llamacpp", model_ref=d.inventory.get(str(p)).ref, node="dgx-2")
    with pytest.raises(DeployError) as e:
        await d.deploy(req, user_id=1)
    assert (e.value.code, e.value.status) == ("weights_missing", 409)
    assert e.value.extra["action"] == {"type": "copy_weights", "node": "dgx-2", "ref": "org/g", "revision": "abc"}
    assert d.fake("dgx-2").exec_calls == [["sh", "-c", WEIGHTS_CHECK_SCRIPT, "sh",
                                           "/hf/hub/models--org--g/snapshots/abc/m-Q4_K_M.gguf"]]
    assert d.fake("dgx-2").run_calls == [] and d.list_views() == []
    d.fake("dgx-2").weights_on_node = {"org/g"}  # copied: the deploy goes ahead
    v = await d.deploy(req, user_id=1)
    assert d._get(v["id"]).node == "dgx-2" and len(d.fake("dgx-2").run_calls) == 1


async def test_transport_error_in_the_presence_check_is_node_unreachable(deployer_two_nodes, qwen_req):
    import requests
    d = deployer_two_nodes
    d.fake("dgx-2").raise_on_all = requests.exceptions.ConnectionError("ssh: connect to host 192.0.2.2: timed out")
    with pytest.raises(DeployError) as e:
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert (e.value.code, e.value.status) == ("node_unreachable", 503)
    assert d.fake("dgx-2").raised == 1  # the presence check, the first remote I/O, already mapped it
    assert d.fake("dgx-2").exec_calls == [] and d.list_views() == []


def test_copy_ssh_pins_the_config_nodes_registry_wrote(two_node_registry):
    from dgxmm.nodes.registry import ssh_config_path
    cfg = ssh_config_path(two_node_registry.settings)
    assert cfg == Path(two_node_registry.settings.data_dir) / "ssh" / "config" and cfg.exists()  # the registry's
    assert "StrictHostKeyChecking yes" in cfg.read_text() and "known_hosts_dgx-2" in cfg.read_text()
    _tar, ssh = copy_argv("/hf-local", two_node_registry.config("dgx-2"), TINY, REV, cfg)
    assert ssh[:7] == ["ssh", "-F", str(cfg), "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes"]


def test_migration_44_adds_kind_and_node_and_keeps_old_rows_null(tmp_path):
    import sqlite3
    from sqlalchemy import create_engine
    from dgxmm.db import Database
    from dgxmm.migrations import migrate
    path = tmp_path / "v43.sqlite3"
    Database(f"sqlite:///{path}").engine.dispose()
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("DELETE FROM schema_version WHERE version = 44")
    conn.execute("ALTER TABLE downloads DROP COLUMN kind")
    conn.execute("ALTER TABLE downloads DROP COLUMN node")
    conn.execute("INSERT INTO downloads (repo_id, revision, files, state, bytes_done, bytes_total, created_at) "
                 "VALUES ('a/b', 'main', '[]', 'done', 0, 0, '2026-01-01 00:00:00')")
    conn.close()
    engine = create_engine(f"sqlite:///{path}")
    assert migrate(engine) == [44]
    engine.dispose()
    conn = sqlite3.connect(str(path))
    cols = {r[1] for r in conn.execute('PRAGMA table_info("downloads")')}
    assert {"kind", "node"} <= cols
    assert conn.execute("SELECT kind, node FROM downloads").fetchall() == [(None, None)]
    conn.close()


async def test_finished_copy_leaves_no_cancel_flag(sync):
    s = sync(tar={"hang": True}, ssh={"hang": True})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    sync.downloads.cancel(job.id)
    await finished(sync.downloads, job.id)
    await asyncio.gather(job.task, return_exceptions=True)
    await asyncio.sleep(0)  # the done callbacks
    assert job.id not in sync.downloads._cancel and job.id not in sync.downloads._tasks
    assert job.id not in s.jobs


async def test_copy_in_flight_blocks_hub_download_with_a_copy_message(sync):
    from dgxmm.downloads import DownloadError
    s = sync(tar={"hang": True}, ssh={"hang": True})
    job = await s.copy_weights("dgx-2", "Qwen/Qwen3-Tiny", REV)
    with pytest.raises(DownloadError) as e:
        sync.downloads.start("Qwen/Qwen3-Tiny")
    assert e.value.status == 409 and "being copied to dgx-2" in e.value.message
    sync.downloads.cancel(job.id)
    await finished(sync.downloads, job.id)


def test_recover_names_interrupted_copies_as_copies(sync):
    from dgxmm.db import Download
    with sync.db.session() as s:
        s.add(Download(repo_id="a/b", revision="main", files=[], state="running", kind="copy", node="dgx-2"))
        s.add(Download(repo_id="c/d", revision="main", files=[], state="queued"))
    assert sync.downloads.recover() == 2
    errors = {v["repo_id"]: v["error"] for v in sync.downloads.list()}
    assert "copy" in errors["a/b"] and "dgx-2" in errors["a/b"] and "download" not in errors["a/b"]
    assert errors["c/d"] == "interrupted by dgxmm restart; start the download again"
