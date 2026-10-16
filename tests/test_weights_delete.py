"""Deleting downloaded weights and cleaning up incomplete ones (spec §1 + §4)."""
from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import pytest

from dgxmm.db import Deployment, Download
from dgxmm.docker_ctl import HELPER_LABEL, HELPER_MOUNT, DockerController
from dgxmm.engines.base import MANAGED_LABEL
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.inventory.service import InventoryService
from dgxmm.inventory.weights import DeleteError, WeightsDeleter
from tests.conftest import QWEN_TINY, write_gguf
from tests.fakes import FakeDockerClient


def hub(settings) -> Path:
    return Path(settings.hf_cache.local) / "hub"


def groot(settings) -> Path:
    return Path(settings.gguf_roots[0].local)


def add_dep(db, name, ref, state, config=None):
    with db.session() as s:
        s.add(Deployment(name=name, engine="vllm", model_ref=ref, state=state, config=config or {}))


def add_download(db, repo, state):
    with db.session() as s:
        s.add(Download(repo_id=repo, state=state, files=[]))


def local_for(settings, host: str) -> Path:
    """Inverse of the host mapping used by the helper (test-only)."""
    for m in [settings.hf_cache, *settings.gguf_roots]:
        if host == m.host or host.startswith(m.host.rstrip("/") + "/"):
            return Path(m.local + host[len(m.host):])
    raise AssertionError(f"unmapped host path {host}")


def simulate_rm(settings):
    """Make the fake helper container behave like `rm -rf` on its single mount."""
    def act(call):
        [host] = call["volumes"]
        base = local_for(settings, host)
        for arg in call["command"]:
            if arg.startswith(HELPER_MOUNT + "/"):
                p = base / arg[len(HELPER_MOUNT) + 1:]
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                elif os.path.lexists(p):
                    p.unlink()
    return act


@pytest.fixture
def make(settings, db):
    def _make(sdk=None, scanners=None):
        inv = InventoryService(settings, scanners=scanners)
        inv.refresh()
        docker = DockerController(sdk) if sdk is not None else None
        return WeightsDeleter(settings, db, inv, docker=docker), inv
    return _make


# ---------------------------------------------------------------- HF repos

def test_preview_hf_repo_lists_repo_and_locks_as_host_paths(settings, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 5000})
    locks = hub(settings) / ".locks" / "models--Qwen--Qwen3-Tiny"
    locks.mkdir(parents=True)
    (locks / "x.lock").write_text("")
    d, _ = make()
    p = d.preview("Qwen/Qwen3-Tiny")
    assert p["paths"] == ["/host/hf/hub/models--Qwen--Qwen3-Tiny", "/host/hf/hub/.locks/models--Qwen--Qwen3-Tiny"]
    assert p["bytes"] >= 5000 and p["name"] == "Qwen3-Tiny" and p["source"] == "hf"
    assert p["deployments_blocking"] == [] and p["deployments_affected"] == [] and p["download_active"] is False
    assert p["blocked"] is False


def test_delete_hf_repo_removes_all_revisions_locks_and_refreshes(settings, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", rev="a" * 40, config=QWEN_TINY, files={"model.safetensors": 3000})
    fake_hf_repo("Qwen/Qwen3-Tiny", rev="b" * 40, config=QWEN_TINY, files={"model.safetensors": 3000}, main=False)
    fake_hf_repo("org/keep", config=QWEN_TINY)
    (hub(settings) / ".locks" / "models--Qwen--Qwen3-Tiny").mkdir(parents=True)
    d, inv = make()
    out = d.delete("Qwen/Qwen3-Tiny")
    assert not (hub(settings) / "models--Qwen--Qwen3-Tiny").exists()
    assert not (hub(settings) / ".locks" / "models--Qwen--Qwen3-Tiny").exists()
    assert (hub(settings) / "models--org--keep").is_dir() and (hub(settings) / ".locks").is_dir()
    assert out["freed_bytes"] >= 6000 and out["ref"] == "Qwen/Qwen3-Tiny"
    assert inv.get("Qwen/Qwen3-Tiny") is None and inv.get("org/keep") is not None


def test_unknown_ref_is_404(make):
    d, _ = make()
    with pytest.raises(DeleteError) as e:
        d.preview("nope/nope")
    assert e.value.status == 404 and e.value.code == "not_found"


@pytest.mark.parametrize("state", ["ready", "starting", "loading", "sleeping", "unknown", "planned"])
def test_model_in_use_is_409_and_nothing_deleted(settings, db, fake_hf_repo, make, state):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    add_dep(db, "tiny", "Qwen/Qwen3-Tiny", state)
    d, _ = make()
    p = d.preview("Qwen/Qwen3-Tiny")
    assert p["blocked"] and [x["name"] for x in p["deployments_blocking"]] == ["tiny"]
    assert p["deployments_blocking"][0]["state"] == state and "id" in p["deployments_blocking"][0]
    with pytest.raises(DeleteError) as e:
        d.delete("Qwen/Qwen3-Tiny")
    assert e.value.status == 409 and e.value.code == "model_in_use"
    assert e.value.extra["deployments"][0]["name"] == "tiny"
    assert (hub(settings) / "models--Qwen--Qwen3-Tiny").is_dir()


def test_draft_model_of_running_deployment_is_in_use(settings, db, fake_hf_repo, make):
    fake_hf_repo("org/draft", config=QWEN_TINY)
    add_dep(db, "big", "org/other", "ready", config={"speculative": {"model": "org/draft", "method": "eagle"}})
    d, _ = make()
    with pytest.raises(DeleteError) as e:
        d.delete("org/draft")
    assert e.value.code == "model_in_use"


def test_stopped_and_failed_deployments_are_warnings_only(settings, db, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    add_dep(db, "a", "Qwen/Qwen3-Tiny", "stopped")
    add_dep(db, "b", "Qwen/Qwen3-Tiny", "failed")
    d, _ = make()
    p = d.preview("Qwen/Qwen3-Tiny")
    assert not p["blocked"] and sorted(x["name"] for x in p["deployments_affected"]) == ["a", "b"]
    out = d.delete("Qwen/Qwen3-Tiny")
    assert sorted(x["name"] for x in out["warnings"]) == ["a", "b"]
    assert not (hub(settings) / "models--Qwen--Qwen3-Tiny").exists()


@pytest.mark.parametrize("state,blocked", [("queued", True), ("running", True), ("done", False),
                                           ("failed", False), ("cancelled", False)])
def test_active_download_is_409(settings, db, fake_hf_repo, make, state, blocked):
    fake_hf_repo("org/partial", config=QWEN_TINY, incomplete=True)
    add_download(db, "org/partial", state)
    d, _ = make()
    assert d.preview("org/partial")["download_active"] is blocked
    if blocked:
        with pytest.raises(DeleteError) as e:
            d.delete("org/partial")
        assert e.value.status == 409 and e.value.code == "download_active"
        assert (hub(settings) / "models--org--partial").is_dir()
    else:
        d.delete("org/partial")
        assert not (hub(settings) / "models--org--partial").exists()


# ---------------------------------------------------------------- safety

def test_symlinked_repo_dir_escaping_the_cache_is_refused(settings, tmp_path, fake_hf_repo, make):
    outside = tmp_path / "outside"
    fake_hf_repo("evil/model", config=QWEN_TINY)
    shutil.move(str(hub(settings) / "models--evil--model"), str(outside))
    (hub(settings) / "models--evil--model").symlink_to(outside, target_is_directory=True)
    d, inv = make()
    assert inv.get("evil/model") is not None
    with pytest.raises(DeleteError) as e:
        d.delete("evil/model")
    assert e.value.code == "unsafe_path" and e.value.status == 400
    assert (outside / "snapshots").is_dir()


def test_gguf_symlink_escaping_the_roots_is_refused(settings, tmp_path, make):
    real = write_gguf(tmp_path / "elsewhere" / "Real-Q4_K_M.gguf", vocab=10)
    (groot(settings) / "Link-Q4_K_M.gguf").symlink_to(real)
    d, inv = make()
    ref = str(groot(settings) / "Link-Q4_K_M.gguf")
    assert inv.get(ref) is not None
    with pytest.raises(DeleteError) as e:
        d.preview(ref)
    assert e.value.code == "unsafe_path"
    with pytest.raises(DeleteError):
        d.delete(ref)
    assert real.exists() and (groot(settings) / "Link-Q4_K_M.gguf").is_symlink()


def _fake_art(ref, source="gguf", complete=True):
    return ModelArtifact(ref=ref, source=source, local_path=ref, revision=None, format="gguf", weights_bytes=1,
                         size_bytes=1, complete=complete, kind="llm", arch=None, architecture=None, quant=None)


@pytest.mark.parametrize("which", ["gguf_root", "hub", "hf_cache", "outside", "traversal"])
def test_never_deletes_a_root_or_anything_outside(settings, tmp_path, make, which):
    (tmp_path / "victim.gguf").write_bytes(b"x")
    ref = {"gguf_root": str(groot(settings)), "hub": str(hub(settings)), "hf_cache": settings.hf_cache.local,
           "outside": str(tmp_path / "victim.gguf"),
           "traversal": str(groot(settings) / ".." / "victim.gguf")}[which]
    d, _ = make(scanners=[lambda: [_fake_art(ref)]])
    with pytest.raises(DeleteError) as e:
        d.delete(ref)
    assert e.value.code == "unsafe_path"
    assert groot(settings).is_dir() and hub(settings).is_dir() and (tmp_path / "victim.gguf").exists()


@pytest.mark.parametrize("ref", ["org/..", "../x", "org/name/../../x"])
def test_invalid_hf_refs_are_refused(settings, make, ref):
    d, _ = make(scanners=[lambda: [_fake_art(ref, source="hf")]])
    with pytest.raises(DeleteError) as e:
        d.delete(ref)
    assert e.value.code in ("unsafe_path", "invalid_ref")
    assert hub(settings).is_dir()


# ---------------------------------------------------------------- GGUF

def test_split_gguf_deletes_all_shards_and_empty_parent(settings, make):
    root = groot(settings)
    write_gguf(root / "big" / "M-Q4_K_M-00001-of-00003.gguf", vocab=10)
    (root / "big" / "M-Q4_K_M-00002-of-00003.gguf").write_bytes(b"x" * 500)
    (root / "big" / "M-Q4_K_M-00003-of-00003.gguf").write_bytes(b"x" * 300)
    write_gguf(root / "keep" / "Other-Q8_0.gguf", vocab=10)
    d, inv = make()
    ref = str(root / "big" / "M-Q4_K_M-00001-of-00003.gguf")
    p = d.preview(ref)
    assert sorted(p["paths"]) == [f"/host/gguf/big/M-Q4_K_M-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    assert p["bytes"] > 800 and p["name"] == "M-Q4_K_M-00001-of-00003.gguf"
    out = d.delete(ref)
    assert not (root / "big").exists() and (root / "keep" / "Other-Q8_0.gguf").exists()
    assert out["freed_bytes"] == p["bytes"] and inv.get(ref) is None


def test_gguf_in_root_keeps_the_root_and_non_empty_parent(settings, make):
    root = groot(settings)
    write_gguf(root / "A-Q8_0.gguf", vocab=10)
    write_gguf(root / "dir" / "B-Q8_0.gguf", vocab=10)
    (root / "dir" / "README.md").write_text("hi")
    d, _ = make()
    d.delete(str(root / "A-Q8_0.gguf"))
    d.delete(str(root / "dir" / "B-Q8_0.gguf"))
    assert root.is_dir() and (root / "dir" / "README.md").exists()
    assert not (root / "A-Q8_0.gguf").exists() and not (root / "dir" / "B-Q8_0.gguf").exists()


def test_gguf_inside_hf_snapshot_deletes_link_and_blob(settings, db, fake_hf_repo, make):
    snap = fake_hf_repo("unsloth/Qwen3-GGUF", files={}, main=True)
    blob = write_gguf(snap.parent.parent / "blobs" / "gg", vocab=10, ftype=7)
    os.symlink(os.path.relpath(blob, snap), snap / "Qwen3-0.6B-Q8_0.gguf")
    d, _ = make()
    ref = str(snap / "Qwen3-0.6B-Q8_0.gguf")
    add_download(db, "unsloth/Qwen3-GGUF", "running")
    assert d.preview(ref)["download_active"] is True
    with db.session() as s:
        for row in s.query(Download).all():
            row.state = "done"
    size = blob.stat().st_size
    out = d.delete(ref)
    assert not os.path.lexists(snap / "Qwen3-0.6B-Q8_0.gguf") and not blob.exists()
    assert out["freed_bytes"] == size and snap.is_dir()


def test_gguf_in_use_by_llamacpp_deployment_is_409(settings, db, make):
    p = write_gguf(groot(settings) / "A-Q8_0.gguf", vocab=10)
    add_dep(db, "a", str(p), "ready")
    d, _ = make()
    with pytest.raises(DeleteError) as e:
        d.delete(str(p))
    assert e.value.code == "model_in_use" and p.exists()


# ---------------------------------------------------------------- helper container

def test_helper_container_args(settings, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    (hub(settings) / ".locks" / "models--Qwen--Qwen3-Tiny").mkdir(parents=True)
    sdk = FakeDockerClient()
    other = sdk.add_unmanaged("litellm-proxy")
    sdk.helper_action = simulate_rm(settings)
    d, _ = make(sdk)
    out = d.delete("Qwen/Qwen3-Tiny")
    assert out["method"] == "helper"
    runs = {r["volumes"] and next(iter(r["volumes"])): r for r in sdk.helper_runs}
    assert set(runs) == {"/host/hf/hub", "/host/hf/hub/.locks"}
    r = runs["/host/hf/hub"]
    assert r["volumes"] == {"/host/hf/hub": {"bind": HELPER_MOUNT, "mode": "rw"}}
    assert r["image"] == "dgxmm:latest" and r["user"] == "0:0" and r["network_mode"] == "none"
    assert r["auto_remove"] is True and r["detach"] is True
    assert r["labels"] == {HELPER_LABEL: "true"} and MANAGED_LABEL not in r["labels"]
    assert r["entrypoint"] == ["rm"] and r["command"] == ["-rf", "--", f"{HELPER_MOUNT}/models--Qwen--Qwen3-Tiny"]
    assert sdk.helper_waits == [settings.helper_timeout_s, settings.helper_timeout_s]
    assert not (hub(settings) / "models--Qwen--Qwen3-Tiny").exists()
    assert list(sdk._containers) == [other.id] and not sdk.container(other.id).stopped


def test_helper_for_files_uses_rm_f_and_groups_shards(settings, make):
    root = groot(settings)
    write_gguf(root / "big" / "M-00001-of-00002.gguf", vocab=10)
    (root / "big" / "M-00002-of-00002.gguf").write_bytes(b"x")
    sdk = FakeDockerClient()
    sdk.helper_action = simulate_rm(settings)
    d, _ = make(sdk)
    d.delete(str(root / "big" / "M-00001-of-00002.gguf"))
    [r] = sdk.helper_runs
    assert next(iter(r["volumes"])) == "/host/gguf/big"
    assert r["command"] == ["-f", "--", f"{HELPER_MOUNT}/M-00001-of-00002.gguf", f"{HELPER_MOUNT}/M-00002-of-00002.gguf"]


def test_helper_image_from_settings_or_own_container(settings, fake_hf_repo, make):
    fake_hf_repo("a/one", config=QWEN_TINY)
    fake_hf_repo("a/two", config=QWEN_TINY)
    sdk = FakeDockerClient()
    sdk.add_unmanaged(socket.gethostname(), image="dgxmm:build-42")
    sdk.helper_action = simulate_rm(settings)
    d, _ = make(sdk)
    d.delete("a/one")
    assert sdk.helper_runs[-1]["image"] == "dgxmm:build-42"
    settings.helper_image = "custom/helper:1"
    d.delete("a/two")
    assert sdk.helper_runs[-1]["image"] == "custom/helper:1"


def test_helper_failure_falls_back_to_in_process(settings, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    sdk = FakeDockerClient()
    sdk.fail_helper = True
    d, _ = make(sdk)
    out = d.delete("Qwen/Qwen3-Tiny")
    assert out["method"] == "in_process" and not (hub(settings) / "models--Qwen--Qwen3-Tiny").exists()


def test_helper_that_leaves_files_is_completed_in_process(settings, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    sdk = FakeDockerClient()   # helper "runs" but deletes nothing
    d, _ = make(sdk)
    out = d.delete("Qwen/Qwen3-Tiny")
    assert out["method"] == "helper+in_process" and not (hub(settings) / "models--Qwen--Qwen3-Tiny").exists()


def test_no_docker_deletes_in_process(settings, fake_hf_repo, make):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    d, _ = make()
    assert d.delete("Qwen/Qwen3-Tiny")["method"] == "in_process"


def test_permission_error_reports_chown_hint(settings, fake_hf_repo, make, monkeypatch):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    import dgxmm.inventory.weights as w

    def denied(path, *a, **kw):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(w.shutil, "rmtree", denied)
    d, _ = make()
    with pytest.raises(DeleteError) as e:
        d.delete("Qwen/Qwen3-Tiny")
    assert e.value.code == "delete_failed" and e.value.status == 500
    assert f"sudo chown -R {os.getuid()}:{os.getgid()} /host/hf/hub/models--Qwen--Qwen3-Tiny" in e.value.message


def test_run_helper_refuses_bad_mounts():
    d = DockerController(FakeDockerClient())
    for bad in ("/", "relative/dir", "", "/a/../"):
        with pytest.raises(ValueError):
            d.run_helper("img", bad, ["rm", "-rf", "--", f"{HELPER_MOUNT}/x"])


def test_run_helper_kills_on_timeout():
    sdk = FakeDockerClient()
    sdk.helper_wait_error = TimeoutError("read timed out")
    d = DockerController(sdk)
    assert d.run_helper("img", "/h/dir", ["rm", "-rf", "--", f"{HELPER_MOUNT}/x"], timeout=1) is None
    assert sdk.helper_killed == 1


# ---------------------------------------------------------------- incomplete cleanup

@pytest.fixture
def incomplete_set(settings, db, fake_hf_repo):
    fake_hf_repo("org/partial", config=QWEN_TINY, incomplete=True)
    fake_hf_repo("org/downloading", config=QWEN_TINY, incomplete=True)
    fake_hf_repo("org/inuse", config=QWEN_TINY, incomplete=True)
    fake_hf_repo("org/broken", files={}, broken=["config.json"])
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    add_download(db, "org/downloading", "running")
    add_dep(db, "x", "org/inuse", "starting")


def test_cleanup_preview_lists_incomplete_with_skip_reasons(incomplete_set, make):
    d, _ = make()
    p = d.cleanup_preview()
    items = {i["ref"]: i for i in p["items"]}
    assert set(items) == {"org/partial", "org/downloading", "org/inuse", "org/broken"}
    assert items["org/partial"]["eligible"] and items["org/broken"]["eligible"]
    assert not items["org/downloading"]["eligible"] and items["org/downloading"]["skip_reason"] == "download_active"
    assert not items["org/inuse"]["eligible"] and items["org/inuse"]["skip_reason"] == "model_in_use"
    assert items["org/partial"]["problems"] and items["org/partial"]["bytes"] > 0
    assert p["count"] == 2 and p["bytes"] == items["org/partial"]["bytes"] + items["org/broken"]["bytes"]


def test_cleanup_all_deletes_eligible_and_reports_skips(settings, incomplete_set, make):
    d, inv = make()
    out = d.cleanup()
    assert sorted(x["ref"] for x in out["deleted"]) == ["org/broken", "org/partial"]
    assert {x["ref"]: x["code"] for x in out["skipped"]} == {"org/downloading": "download_active",
                                                              "org/inuse": "model_in_use"}
    assert out["freed_bytes"] == sum(x["freed_bytes"] for x in out["deleted"])
    assert not (hub(settings) / "models--org--partial").exists()
    for keep in ("org--downloading", "org--inuse", "Qwen--Qwen3-Tiny"):
        assert (hub(settings) / f"models--{keep}").is_dir()
    assert inv.get("org/partial") is None and inv.get("Qwen/Qwen3-Tiny") is not None


def test_cleanup_selected_refs_never_deletes_complete_models(settings, incomplete_set, make):
    d, _ = make()
    out = d.cleanup(["Qwen/Qwen3-Tiny", "org/partial", "nope/nope"])
    assert [x["ref"] for x in out["deleted"]] == ["org/partial"]
    codes = {x["ref"]: x["code"] for x in out["skipped"]}
    assert codes == {"Qwen/Qwen3-Tiny": "not_incomplete", "nope/nope": "not_found"}
    assert (hub(settings) / "models--Qwen--Qwen3-Tiny").is_dir()
    assert (hub(settings) / "models--org--broken").is_dir()
