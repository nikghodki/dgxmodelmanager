"""The Decision model page's Download (and any download of a decision model) goes through the pinned, verified,
resumable snapshot downloader (dgxmm/decider/snapshot.py), with progress in the normal Downloads view."""
import asyncio
import hashlib
import json
import threading
from collections import namedtuple
from pathlib import Path

import httpx
import pytest

from dgxmm.config import GIB
from dgxmm.decider import snapshot
from dgxmm.downloads import DownloadService
from dgxmm.engines.decision import DECIDER_MODELS, hash_list_name
from dgxmm.inventory.hf_scan import DOWNLOAD_MARKER
from tests.test_api_deployments import admin
from tests.test_decider_snapshot import FakeServer, fetcher

NANO = "vllm-sr/d3-nano"
REV = DECIDER_MODELS[NANO].revision
Usage = namedtuple("Usage", "total used free")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


CODE = {"d3_server.py": b"print('serve')\n", "config.json": b'{"model_type": "d3"}'}
WEIGHTS = {"model.safetensors": b"w" * 5000, "LICENSE": b"licence text"}


def remote_files() -> dict[str, bytes]:
    manifest = json.dumps({"files_sha256": {n: sha(d) for n, d in {**CODE, **WEIGHTS}.items()}}).encode()
    return {**CODE, "MODEL_MANIFEST.json": manifest, **WEIGHTS, "assets/card.png": b"\x89PNG", ".gitattributes": b"*"}


class Hub:
    """Fake Hub for the snapshot path: listing + a fetch that writes the cache layout (verified by sha256)."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.calls, self.lists = files, [], 0
        self.block: threading.Event | None = None  # set: the fetch waits until cancelled

    def list_files(self, repo_id, revision):
        assert (repo_id, revision) == (NANO, REV)
        self.lists += 1
        return [(n, len(d)) for n, d in self.files.items()]

    def factory(self, echo, should_stop):
        def fetch(repo_id, filename, revision, cache_dir, force, want):
            assert (repo_id, revision) == (NANO, REV)
            self.calls.append(filename)
            if self.block is not None and filename == "model.safetensors":
                while not should_stop():
                    self.block.wait(0.01)
                raise snapshot.DownloadCancelled("cancelled")
            data = self.files[filename]
            if sha(data) != want:
                raise snapshot.HashMismatch(f"{filename}: downloaded SHA-256 differs; discarded")
            repo = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
            blob = repo / "blobs" / sha(data)
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
            link = repo / "snapshots" / revision / filename
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(blob)
            return str(link)
        return fetch


@pytest.fixture
def pinned(stack, tmp_path):
    files = remote_files()
    build = tmp_path / "build"
    build.mkdir()
    (build / hash_list_name(NANO, REV)).write_text(
        "".join(f"{sha(files[n])}  {n}\n" for n in [*CODE, "MODEL_MANIFEST.json"]))
    stack.settings.decider_build_dir = str(build)
    hub = Hub(files)

    class NoGenericHub:
        def details(self, repo_id):
            raise AssertionError("a decision model never goes through the generic hf_hub_download path")

    svc = DownloadService(stack.settings, stack.db, stack.inv, NoGenericHub(), stack.events,
                          downloader=lambda **kw: (_ for _ in ()).throw(AssertionError("generic downloader used")),
                          disk_usage=lambda p: Usage(0, 0, 1000 * GIB))
    svc.snapshot_fetch, svc.snapshot_list = hub.factory, hub.list_files
    return svc, hub, stack


async def wait_state(s, dl_id, states=("done", "failed", "cancelled")):
    for _ in range(500):
        v = s.view(dl_id)
        if v["state"] in states:
            return v
        await asyncio.sleep(0.01)
    raise AssertionError(f"download stuck in {s.view(dl_id)['state']}")


def snap_dir(stack) -> Path:
    return snapshot.snapshot_dir(stack.settings.hf_cache.local, NANO)


async def test_decision_model_download_uses_the_verified_snapshot_path(pinned):
    svc, hub, stack = pinned
    v = svc.start(NANO, None, ["*.safetensors"], 1)  # the generic entry point, too: pinned, verified, resumable
    assert v["revision"] == REV and v["state"] == "queued"
    await svc.launch(v["id"])
    v = await wait_state(svc, v["id"])
    assert v["state"] == "done", v
    assert "assets/card.png" not in hub.calls and ".gitattributes" not in hub.calls
    assert set(v["files"]) == {*CODE, "MODEL_MANIFEST.json", *WEIGHTS}
    want = sum(len(remote_files()[n]) for n in v["files"])
    assert v["bytes_total"] == want and v["bytes_done"] == want
    assert snapshot.verify(snap_dir(stack), snapshot.read_pins(Path(stack.settings.decider_build_dir)
                                                              / hash_list_name(NANO, REV)), lambda s: None) == []
    assert not (snap_dir(stack).parent.parent / DOWNLOAD_MARKER).exists()
    art = stack.inv.get(NANO)
    assert art is not None and art.complete


async def test_a_complete_snapshot_is_verified_without_downloading_again(pinned):
    svc, hub, stack = pinned
    v = svc.start(NANO, None, None, 1)
    await svc.launch(v["id"])
    assert (await wait_state(svc, v["id"]))["state"] == "done"
    hub.calls.clear()
    lists = hub.lists
    v = svc.start(NANO, None, None, 1)
    await svc.launch(v["id"])
    v = await wait_state(svc, v["id"])
    assert v["state"] == "done" and hub.calls == [] and hub.lists == lists  # verify-only: no network at all
    assert v["bytes_done"] == v["bytes_total"] > 0


async def test_wrong_bytes_from_the_hub_fail_and_are_never_linked(pinned):
    svc, hub, stack = pinned
    hub.files["model.safetensors"] = b"tampered" * 10
    v = svc.start(NANO, None, None, 1)
    await svc.launch(v["id"])
    v = await wait_state(svc, v["id"])
    assert v["state"] == "failed" and "model.safetensors" in v["error"] and "SHA-256" in v["error"]
    assert not (snap_dir(stack) / "model.safetensors").exists()
    assert (snap_dir(stack).parent.parent / DOWNLOAD_MARKER).exists()  # never offered as complete
    art = stack.inv.get(NANO)
    assert art is None or not art.complete


async def test_cancel_stops_the_file_in_flight(pinned):
    svc, hub, stack = pinned
    hub.block = threading.Event()
    v = svc.start(NANO, None, None, 1)
    await svc.launch(v["id"])
    for _ in range(300):
        if "model.safetensors" in hub.calls:
            break
        await asyncio.sleep(0.01)
    svc.cancel(v["id"])
    v = await wait_state(svc, v["id"])
    assert v["state"] == "cancelled"
    art = stack.inv.get(NANO)
    assert art is None or not art.complete


async def test_a_second_download_of_the_same_model_is_refused(pinned):
    from dgxmm.downloads import DownloadError
    svc, hub, stack = pinned
    hub.block = threading.Event()
    v = svc.start(NANO, None, None, 1)
    await svc.launch(v["id"])
    with pytest.raises(DownloadError) as e:
        svc.start(NANO, None, None, 1)
    assert e.value.code == "already_downloading"
    svc.cancel(v["id"])
    await wait_state(svc, v["id"])


def test_other_revisions_of_a_decision_model_are_refused(pinned):
    from dgxmm.downloads import DownloadError
    svc, _hub, _stack = pinned
    with pytest.raises(DownloadError) as e:
        svc.start(NANO, "main", None, 1)
    assert e.value.code == "pinned_revision" and e.value.status == 422


def test_decision_model_endpoint_starts_the_snapshot_download(api_stack, pinned, fake_hf_repo):
    svc, hub, stack = pinned
    api_stack.ctx.downloads = svc
    h = admin(api_stack)
    r = api_stack.post("/dgxmm/api/decider/models/download", headers=h, json={"model": NANO})
    assert r.status_code == 202, r.text
    assert r.json()["revision"] == REV
    for _ in range(500):
        if svc.view(r.json()["id"])["state"] in ("done", "failed"):
            break
        import time
        time.sleep(0.01)
    assert svc.view(r.json()["id"])["state"] == "done"
    assert "model.safetensors" in hub.calls


def test_http_fetcher_stops_mid_file_when_asked_and_keeps_the_partial(tmp_path):
    from dgxmm.engines.decision import DECIDER_REPO, DECIDER_REVISION
    data = b"x" * 700
    server = FakeServer({"big.bin": data})
    seen = {"n": 0}

    def should_stop():
        seen["n"] += 1
        return seen["n"] > 3
    f = fetcher(server, should_stop=should_stop)
    with pytest.raises(snapshot.DownloadCancelled):
        f(DECIDER_REPO, "big.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(data))
    blobs = tmp_path / "hub" / "models--vllm-sr--d3-mini" / "blobs"
    partial = list(blobs.glob("*.incomplete"))
    assert len(partial) == 1 and 0 < partial[0].stat().st_size < len(data)  # resumable
    assert not (tmp_path / "hub" / "models--vllm-sr--d3-mini" / "snapshots" / DECIDER_REVISION / "big.bin").exists()
    kept = partial[0].stat().st_size
    f = fetcher(server)  # the next run resumes from the partial file
    f(DECIDER_REPO, "big.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(data))
    assert server.ranges[-1] == ("big.bin", f"bytes={kept}-")
