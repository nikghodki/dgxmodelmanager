"""Download cancel stops the file in flight, removes its partial file, and an unfinished download is never
reported complete (H6 D2)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dgxmm.catalog.hub import HubFile
from dgxmm.config import GIB
from dgxmm.downloads import SubprocessDownloader
from dgxmm.inventory.hf_scan import DOWNLOAD_MARKER
from tests.conftest import QWEN_TINY
from tests.test_downloads import FILES, fake_downloader, svc, wait_state

# A stand-in for hf_hub_download in the child: lays out the repo like huggingface_hub, then either writes the
# file at once or "streams" it slowly into blobs/<etag>.incomplete.
FAKE_CHILD = r"""
import json, os, sys, time
repo, filename, revision, cache_dir = sys.argv[1:5]
d = os.path.join(cache_dir, "models--" + repo.replace("/", "--"))
os.makedirs(os.path.join(d, "blobs"), exist_ok=True)
snap = os.path.join(d, "snapshots", revision)
os.makedirs(snap, exist_ok=True)
info = os.path.join(cache_dir, "child-" + filename.replace("/", "_") + ".json")
json.dump({"argv": sys.argv[1:], "token": os.environ.get("HF_TOKEN"), "pid": os.getpid()}, open(info, "w"))
if filename == "boom.json":
    print(json.dumps({"type": "RepositoryNotFoundError", "message": "404 nope", "filename": ""}), flush=True)
    sys.exit(1)
if filename == "denied.json":
    print(json.dumps({"type": "PermissionError", "message": "Permission denied",
                      "filename": os.path.join(d, "blobs", "x")}), flush=True)
    sys.exit(1)
blob = os.path.join(d, "blobs", "etag-" + filename)
if filename.endswith(".safetensors"):
    with open(blob + ".incomplete", "wb") as f:
        for _ in range(600):          # 60 s "stream"
            f.write(b"x" * 1024); f.flush(); time.sleep(0.1)
    os.replace(blob + ".incomplete", blob)
else:
    with open(blob, "w") as f:
        f.write(json.dumps(%s) if filename == "config.json" else "x")
os.symlink(os.path.relpath(blob, snap), os.path.join(snap, filename))
""" % json.dumps(QWEN_TINY)


def child_downloader():
    return SubprocessDownloader(python=sys.executable, code=FAKE_CHILD)


def hub(stack) -> Path:
    return Path(stack.settings.hf_cache.local) / "hub"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is dead for our purposes
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


async def test_cancel_stops_the_file_in_flight_and_removes_its_partial(stack):
    s = svc(stack, child_downloader(), files=[HubFile("config.json", 100), HubFile("model.safetensors", GIB)])
    s.token = "hf_secret_for_test"
    v = s.start("org/slow")
    await s.launch(v["id"])
    repo = hub(stack) / "models--org--slow"
    partial = repo / "blobs" / "etag-model.safetensors.incomplete"
    for _ in range(200):
        if partial.exists() and partial.stat().st_size > 0:
            break
        await asyncio.sleep(0.05)
    assert partial.exists(), "the slow file never started"
    info = json.loads((hub(stack) / "child-model.safetensors.json").read_text())
    t0 = time.monotonic()
    s.cancel(v["id"])
    out = await wait_state(s, v["id"])
    took = time.monotonic() - t0
    assert out["state"] == "cancelled" and took < 3, (out, took)
    assert not alive(info["pid"])
    assert not list((repo / "blobs").glob("*.incomplete"))
    art = stack.inv.get("org/slow")
    assert art is None or not art.complete


async def test_the_token_goes_to_the_child_by_environment_not_argv(stack):
    s = svc(stack, child_downloader(), files=[HubFile("config.json", 100), HubFile("tokenizer.json", 10),
                                               HubFile("x.safetensors", 0)])
    s.downloader.code = FAKE_CHILD.replace('if filename.endswith(".safetensors"):', "if False:")
    s.token = "hf_secret_for_test"
    v = s.start("org/tok")
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "done", out
    info = json.loads((hub(stack) / "child-config.json.json").read_text())
    assert info["token"] == "hf_secret_for_test" and "hf_secret_for_test" not in json.dumps(info["argv"])
    assert not (hub(stack) / "models--org--tok" / DOWNLOAD_MARKER).exists()
    assert stack.inv.get("org/tok").complete


async def test_child_failures_keep_their_type_and_permission_hint(stack):
    s = svc(stack, child_downloader(), files=[HubFile("boom.json", 1), HubFile("w.safetensors", 1)])
    v = s.start("org/boom", include=["boom.json"])
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "failed" and out["error"] == "RepositoryNotFoundError: 404 nope"
    s2 = svc(stack, child_downloader(), files=[HubFile("denied.json", 1)])
    v = s2.start("org/denied", include=["denied.json"])
    await s2.launch(v["id"])
    out = await wait_state(s2, v["id"])
    assert out["state"] == "failed" and "owned by root" in out["error"] and "chown" in out["error"]


async def test_a_download_cancelled_between_files_is_never_reported_complete(stack):
    """config.json + the weights are on disk (it looks deployable) but the tokenizer never came."""
    files = [HubFile("config.json", 100), HubFile("model.safetensors", 3 * GIB), HubFile("tokenizer.json", 10)]
    inner = fake_downloader()
    holder = {}

    def dl(**kw):
        out = inner(**kw)
        if kw["filename"] == "model.safetensors":
            holder["svc"].cancel(holder["id"])
        return out
    s = svc(stack, dl, files=files)
    holder["svc"] = s
    v = s.start("org/half")
    holder["id"] = v["id"]
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "cancelled" and inner.calls == ["config.json", "model.safetensors"]
    art = stack.inv.get("org/half")
    assert art is not None and not art.complete
    assert any("tokenizer.json" in p for p in art.problems)


async def test_a_failed_download_is_never_reported_complete(stack):
    files = [HubFile("config.json", 100), HubFile("model.safetensors", 3 * GIB), HubFile("tokenizer.json", 10)]
    s = svc(stack, fake_downloader(fail_on="tokenizer.json", exc=OSError("connection reset")), files=files)
    v = s.start("org/fail")
    await s.launch(v["id"])
    assert (await wait_state(s, v["id"]))["state"] == "failed"
    stack.inv.refresh()
    assert not stack.inv.get("org/fail").complete


async def test_finished_download_clears_the_marker_and_is_complete(stack):
    s = svc(stack, fake_downloader())
    v = s.start("org/full")
    await s.launch(v["id"])
    assert (await wait_state(s, v["id"]))["state"] == "done"
    assert not (hub(stack) / "models--org--full" / DOWNLOAD_MARKER).exists()
    assert stack.inv.get("org/full").complete


def test_marker_for_another_revision_or_with_every_file_present_does_not_mark_incomplete(settings, fake_hf_repo):
    from dgxmm.inventory.hf_scan import scan_hf_cache
    snap = fake_hf_repo("org/m", config=QWEN_TINY, files={"model.safetensors": 10})
    repo = snap.parent.parent
    (repo / DOWNLOAD_MARKER).write_text(json.dumps({"revision": snap.name, "files": ["config.json",
                                                                                     "model.safetensors"]}))
    [a] = scan_hf_cache(repo.parent)
    assert a.complete
    (repo / DOWNLOAD_MARKER).write_text(json.dumps({"revision": "b" * 40, "files": ["tokenizer.json"]}))
    [a] = scan_hf_cache(repo.parent)
    assert a.complete
    (repo / DOWNLOAD_MARKER).write_text(json.dumps({"revision": snap.name, "files": ["tokenizer.json"]}))
    [a] = scan_hf_cache(repo.parent)
    assert not a.complete and any("download not finished" in p for p in a.problems)
    (repo / DOWNLOAD_MARKER).write_text("{not json")
    [a] = scan_hf_cache(repo.parent)
    assert a.complete


async def test_cancelled_download_is_offered_by_incomplete_cleanup(stack, settings, db):
    from dgxmm.inventory.weights import WeightsDeleter
    files = [HubFile("config.json", 100), HubFile("model.safetensors", 3 * GIB), HubFile("tokenizer.json", 10)]
    inner = fake_downloader()
    holder = {}

    def dl(**kw):
        out = inner(**kw)
        if kw["filename"] == "model.safetensors":
            holder["svc"].cancel(holder["id"])
        return out
    s = svc(stack, dl, files=files)
    holder["svc"] = s
    v = s.start("org/half2")
    holder["id"] = v["id"]
    await s.launch(v["id"])
    await wait_state(s, v["id"])
    items = {i["ref"]: i for i in WeightsDeleter(settings, db, stack.inv).cleanup_preview()["items"]}
    assert items["org/half2"]["eligible"]


assert FILES  # the shared fixture list stays importable
