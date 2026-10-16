import asyncio
from collections import namedtuple
from pathlib import Path

import pytest

from dgxmm.catalog.hub import HubDetails, HubFile
from dgxmm.config import GIB
from dgxmm.downloads import DownloadError, DownloadService
from tests.conftest import QWEN_TINY

Usage = namedtuple("Usage", "total used free")


class Hub:
    def __init__(self, files):
        self.files = files

    def details(self, repo_id):
        return HubDetails(repo_id, self.files, QWEN_TINY, False, "abc")


def fake_downloader(fail_on=None, exc=None):
    calls = []
    cache_dirs = []

    def _dl(repo_id, filename, revision=None, cache_dir=None, token=None):
        calls.append(filename)
        cache_dirs.append(cache_dir)
        if filename == fail_on:
            raise exc
        # Match real hf_hub_download: cache_dir is the hub cache, write models--... directly into it
        repo_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
        blobs = repo_dir / "blobs"
        snap = repo_dir / "snapshots" / "abc"
        refs = repo_dir / "refs"
        blobs.mkdir(parents=True, exist_ok=True)
        snap.mkdir(parents=True, exist_ok=True)
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text("abc")
        blob = blobs / f"abc-{filename.replace('/', '_')}"
        if filename == "config.json":
            import json
            blob.write_text(json.dumps(QWEN_TINY))
        else:
            blob.write_bytes(b"x" * 100)
        link = snap / filename
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        return str(link)
    _dl.calls = calls
    _dl.cache_dirs = cache_dirs
    return _dl


FILES = [HubFile("config.json", 100), HubFile("model.safetensors", 3 * GIB), HubFile("original/consolidated.pth", 9 * GIB),
         HubFile("README.md", 10), HubFile("model.Q4.gguf", GIB)]


def svc(stack, downloader, free=1000 * GIB, files=FILES):
    return DownloadService(stack.settings, stack.db, stack.inv, Hub(files), stack.events, downloader=downloader,
                           disk_usage=lambda p: Usage(0, 0, free))


async def wait_state(s, dl_id, states=("done", "failed", "cancelled")):
    for _ in range(300):
        v = s.view(dl_id)
        if v["state"] in states:
            return v
        await asyncio.sleep(0.01)
    return s.view(dl_id)


async def test_downloads_default_files_and_refreshes_inventory(stack):
    dl = fake_downloader()
    s = svc(stack, dl)
    v = s.start("org/new-model", user_id=1)
    assert v["bytes_total"] == 100 + 3 * GIB + 10 and v["files"] == ["config.json", "model.safetensors", "README.md"]
    await s.launch(v["id"])
    done = await wait_state(s, v["id"])
    assert done["state"] == "done" and dl.calls == ["config.json", "model.safetensors", "README.md"]
    # Regression: downloader should be called with hub_dir (hf_cache.local/hub), not base dir
    expected_hub_dir = str(Path(stack.settings.hf_cache.local) / "hub")
    assert all(cd == expected_hub_dir for cd in dl.cache_dirs), f"Expected {expected_hub_dir}, got {dl.cache_dirs}"
    # Inventory refresh happens before state="done", so model should be visible immediately
    assert stack.inv.get("org/new-model") is not None


async def test_refuses_low_disk_bad_input_and_gguf_without_choice(stack):
    s = svc(stack, fake_downloader(), free=40 * GIB)
    with pytest.raises(DownloadError) as e:
        s.start("org/new-model")
    assert e.value.code == "insufficient_disk"
    s = svc(stack, fake_downloader())
    for bad in ("../etc", "org/x;rm"):
        with pytest.raises(DownloadError):
            s.start(bad)
    with pytest.raises(DownloadError):
        s.start("org/x", include=["../../*"])
    gg = svc(stack, fake_downloader(), files=[HubFile("a.Q4_K_M.gguf", GIB), HubFile("a.Q8_0.gguf", 2 * GIB)])
    with pytest.raises(DownloadError) as e:
        gg.start("org/gguf-only")
    assert e.value.code == "choose_file"
    v = gg.start("org/gguf-only", include=["a.Q4_K_M.gguf"])
    assert v["files"] == ["a.Q4_K_M.gguf"]


async def test_permission_error_has_actionable_message(stack):
    err = PermissionError(13, "Permission denied", "/data/hf/hub/models--org--m/blobs/x")
    s = svc(stack, fake_downloader(fail_on="model.safetensors", exc=err))
    v = s.start("org/m")
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "failed" and "owned by root" in out["error"] and "chown" in out["error"]
    import os
    assert f"chown -R {os.getuid()}:{os.getgid()} " in out["error"]


async def test_cancel_between_files(stack):
    dl = fake_downloader()
    s = svc(stack, dl)
    v = s.start("org/c")
    s.cancel(v["id"])
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "cancelled" and dl.calls == []


def _row(stack, state, repo="org/stuck"):
    from dgxmm.db import Download
    with stack.db.session() as s:
        d = Download(repo_id=repo, revision="abc", files=["config.json"], bytes_total=100, state=state)
        s.add(d)
        s.flush()
        return d.id


async def test_restart_marks_interrupted_downloads_failed(stack):
    """I3: queued/running rows left by a previous process are failed at startup."""
    q, r, done = _row(stack, "queued", "org/q"), _row(stack, "running", "org/r"), _row(stack, "done", "org/d")
    s = svc(stack, fake_downloader())
    s.recover()
    assert s.view(q)["state"] == "failed" and "interrupted by dgxmm restart" in s.view(q)["error"]
    assert s.view(r)["state"] == "failed" and s.view(done)["state"] == "done"
    s.start("org/r")  # the repo can be downloaded again


async def test_cancel_without_live_task_marks_cancelled(stack):
    r = _row(stack, "running", "org/r")
    s = svc(stack, fake_downloader())
    assert s.cancel(r)["state"] == "cancelled"


async def test_cancel_unknown_id_is_404(stack):
    s = svc(stack, fake_downloader())
    with pytest.raises(DownloadError) as e:
        s.cancel(424242)
    assert e.value.status == 404 and e.value.code == "not_found"


def test_cancel_route_is_async_and_404s(api_stack):
    """M7: the cancel route publishes events, so it must run on the event loop (async def)."""
    import inspect
    from dgxmm.api import downloads as api_downloads
    from tests.test_api_deployments import admin
    assert inspect.iscoroutinefunction(api_downloads.cancel)
    h = admin(api_stack)
    api_stack.ctx.downloads = svc(api_stack.stack, fake_downloader())
    r = api_stack.post("/dgxmm/api/downloads/424242/cancel", headers=h)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found"
