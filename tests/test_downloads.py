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

    def _dl(repo_id, filename, revision=None, cache_dir=None, token=None):
        calls.append(filename)
        if filename == fail_on:
            raise exc
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


async def test_cancel_between_files(stack):
    dl = fake_downloader()
    s = svc(stack, dl)
    v = s.start("org/c")
    s.cancel(v["id"])
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "cancelled" and dl.calls == []
