"""Pre-downloading and verifying the pinned decider snapshot (dgxmm/decider/snapshot.py)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from dgxmm.decider import snapshot
from dgxmm.engines.decision import DECIDER_REPO, DECIDER_REVISION


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CODE = {"d3_server.py": b"print('serve')\n", "config.json": b'{"model_type": "d3"}'}
WEIGHTS = {"model-00001-of-00002.safetensors": b"w1" * 1000, "model-00002-of-00002.safetensors": b"w2" * 1000}
EXTRA = {"README.md": b"# card\n", "assets/banner.png": b"\x89PNG"}  # in the manifest, not in the pinned list


def manifest(weights=WEIGHTS) -> bytes:
    files = {n: sha(d) for n, d in {**CODE, **weights, **EXTRA}.items()}
    return json.dumps({"schema": "d3-package-manifest/1", "files_sha256": files}).encode()


@pytest.fixture
def remote():
    """What the Hub serves for the pinned revision."""
    return {**CODE, "MODEL_MANIFEST.json": manifest(), **WEIGHTS, **EXTRA}


@pytest.fixture
def pins(tmp_path, remote) -> Path:
    path = tmp_path / "pins.sha256"
    path.write_text("".join(f"{sha(remote[n])}  {n}\n" for n in [*CODE, "MODEL_MANIFEST.json"]))
    return path


class FakeHub:
    """hf_hub_download / list_repo_files stand-ins that write into a cache-like snapshot directory."""

    def __init__(self, hf_home: Path, files: dict[str, bytes]):
        self.hf_home, self.files, self.calls, self.fail_once = hf_home, files, [], set()

    def list_files(self, repo_id: str, revision: str) -> list[tuple[str, int]]:
        assert (repo_id, revision) == (DECIDER_REPO, DECIDER_REVISION)
        return [(n, len(d)) for n, d in self.files.items()]

    def fetch(self, repo_id: str, filename: str, revision: str, cache_dir: str, force: bool, sha256: str) -> str:
        assert (repo_id, revision) == (DECIDER_REPO, DECIDER_REVISION)
        assert cache_dir == str(self.hf_home / "hub")
        assert re.fullmatch(r"[0-9a-f]{64}", sha256)  # every fetched file has an expected digest
        self.calls.append((filename, force))
        if filename in self.fail_once:
            self.fail_once.discard(filename)
            raise OSError("connection reset")
        path = snapshot.snapshot_dir(self.hf_home) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return str(path)


@pytest.fixture
def hub(tmp_path, remote) -> FakeHub:
    return FakeHub(tmp_path / "hf", remote)


def ensure(hub, pins, **kw):
    lines: list[str] = []
    result = snapshot.ensure_snapshot(hub.hf_home, pins, fetch=hub.fetch, list_files=hub.list_files,
                                      echo=lines.append, **kw)
    return result, lines


def test_snapshot_dir_is_the_pinned_revision(tmp_path):
    assert snapshot.snapshot_dir(tmp_path) == (
        tmp_path / "hub" / "models--vllm-sr--d3-mini" / "snapshots" / DECIDER_REVISION)


def test_download_skips_assets_and_verifies_everything(hub, pins, remote):
    ok, lines = ensure(hub, pins)
    assert ok
    fetched = [f for f, _ in hub.calls]
    assert "assets/banner.png" not in fetched  # never loaded by the runtime
    assert set(fetched) == set(remote) - {"assets/banner.png"}
    assert fetched.index("MODEL_MANIFEST.json") < fetched.index("model-00001-of-00002.safetensors")
    text = "\n".join(lines)
    assert "verified" in text and DECIDER_REVISION[:12] in text


def test_complete_snapshot_needs_no_network(hub, pins):
    assert ensure(hub, pins)[0]
    hub.calls.clear()

    def no_network(*a, **k):
        raise AssertionError("must not touch the network")
    ok = snapshot.ensure_snapshot(hub.hf_home, pins, fetch=no_network, list_files=no_network, echo=lambda s: None)
    assert ok and hub.calls == []


def test_verify_only_reports_missing_and_changed_files(hub, pins):
    ok, lines = ensure(hub, pins, verify_only=True)
    assert not ok and hub.calls == [] and any("missing" in line for line in lines)
    ensure(hub, pins)
    (snapshot.snapshot_dir(hub.hf_home) / "model-00002-of-00002.safetensors").write_bytes(b"tampered")
    ok, lines = ensure(hub, pins, verify_only=True)
    assert not ok and any("model-00002-of-00002.safetensors" in line and "SHA-256" in line for line in lines)


def test_corrupt_file_is_downloaded_again_once(hub, pins):
    ensure(hub, pins)
    (snapshot.snapshot_dir(hub.hf_home) / "d3_server.py").write_bytes(b"import os; os.system('evil')")
    hub.calls.clear()
    ok, _ = ensure(hub, pins)
    assert ok and hub.calls == [("d3_server.py", True)]


def test_hub_serving_wrong_bytes_fails_without_retrying_forever(hub, pins, remote):
    hub.files = {**remote, "model-00001-of-00002.safetensors": b"not the pinned weights"}
    ok, lines = ensure(hub, pins)
    assert not ok
    assert hub.calls.count(("model-00001-of-00002.safetensors", True)) == 1
    assert any("model-00001-of-00002.safetensors" in line for line in lines)


def test_manifest_must_match_the_pinned_list_before_it_is_trusted(hub, pins, remote):
    evil = manifest(weights={k: b"evil" for k in WEIGHTS})
    hub.files = {**remote, "MODEL_MANIFEST.json": evil}
    ok, lines = ensure(hub, pins)
    assert not ok and any("MODEL_MANIFEST.json" in line for line in lines)
    assert not any(f.endswith(".safetensors") for f, _ in hub.calls)  # stops before the big files


def test_interrupted_download_resumes_on_the_next_run(hub, pins):
    hub.fail_once.add("model-00001-of-00002.safetensors")
    with pytest.raises(snapshot.DownloadFailed, match="connection reset"):
        ensure(hub, pins)
    hub.calls.clear()
    ok, _ = ensure(hub, pins)
    assert ok
    # files completed before the failure are not fetched again (hf_hub_download itself resumes partial blobs)
    assert ("d3_server.py", False) not in hub.calls and ("model-00001-of-00002.safetensors", False) in hub.calls


def test_pinned_list_parser_rejects_garbage(tmp_path):
    p = tmp_path / "bad.sha256"
    p.write_text("not-a-hash  file.py\n")
    with pytest.raises(ValueError):
        snapshot.read_pins(p)
    p.write_text(f"{'a' * 64}  ../escape.py\n")
    with pytest.raises(ValueError):
        snapshot.read_pins(p)


def test_real_pinned_list_is_readable():
    from dgxmm.decider.image import HASH_LIST, default_build_dir
    pins = snapshot.read_pins(default_build_dir() / HASH_LIST)
    assert {"d3_server.py", "MODEL_MANIFEST.json", "config.json"} <= set(pins)


# --- the resumable HTTP downloader (huggingface_hub 2.x restarts partial files from zero) -------------------

import httpx  # noqa: E402


class FakeServer:
    """/{repo}/resolve/{rev}/{file} with Range support, optional mid-stream drops and Range-ignoring."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.ranges, self.drop_after, self.ignore_range = files, [], {}, False
        self.wrong_range_start = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        prefix = f"/{DECIDER_REPO}/resolve/{DECIDER_REVISION}/"
        assert request.url.path.startswith(prefix)
        assert "authorization" not in request.headers  # the HF token is never sent
        name = request.url.path[len(prefix):]
        data = self.files[name]
        rng = request.headers.get("range")
        self.ranges.append((name, rng))
        start = int(rng.split("=")[1].rstrip("-")) if rng and not self.ignore_range else 0
        if start >= len(data) and rng:
            return httpx.Response(416)
        body = data[start:]
        drop = self.drop_after.pop(name, None)

        def stream():
            sent = 0
            for i in range(0, len(body), 7):
                if drop is not None and sent >= drop:
                    raise httpx.ReadError("connection reset by peer")
                chunk = body[i:i + 7]
                sent += len(chunk)
                yield chunk
        status = 206 if start else 200
        headers = {"content-length": str(len(body))}
        if start:
            headers["content-range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        if self.wrong_range_start and rng:
            body = data
            headers = {"content-length": str(len(data)), "content-range": f"bytes 0-{len(data) - 1}/{len(data)}"}
            status = 206
        return httpx.Response(status, content=stream(), headers=headers)


def fetcher(server: FakeServer, lines: list[str] | None = None, **kw):
    client = httpx.Client(transport=httpx.MockTransport(server.handler), base_url="https://hub.test")
    return snapshot.HttpFetcher(echo=(lines.append if lines is not None else (lambda s: None)), client=client,
                                endpoint="https://hub.test", retries=kw.pop("retries", 3), sleep=lambda s: None, **kw)


def test_http_fetch_writes_a_cache_layout_with_relative_links(tmp_path):
    data = b"x" * 1000
    server = FakeServer({"sub/dir/file.bin": data})
    path = fetcher(server)(DECIDER_REPO, "sub/dir/file.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["sub/dir/file.bin"]))
    link = snapshot.snapshot_dir(tmp_path) / "sub/dir/file.bin"
    assert Path(path) == link and link.is_symlink() and link.read_bytes() == data
    target = Path(__import__("os").readlink(link))
    assert not target.is_absolute() and target.name == sha(data)
    assert (tmp_path / "hub" / "models--vllm-sr--d3-mini" / "blobs" / sha(data)).read_bytes() == data


def test_http_fetch_resumes_within_a_run_after_a_dropped_connection(tmp_path):
    data = bytes(range(256)) * 40
    server = FakeServer({"w.safetensors": data})
    server.drop_after["w.safetensors"] = 3000
    fetcher(server)(DECIDER_REPO, "w.safetensors", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["w.safetensors"]))
    assert (snapshot.snapshot_dir(tmp_path) / "w.safetensors").read_bytes() == data
    assert server.ranges[0] == ("w.safetensors", None) and server.ranges[1][1].startswith("bytes=30")


def test_http_fetch_resumes_a_partial_file_from_an_earlier_run(tmp_path):
    data = b"0123456789" * 500
    server = FakeServer({"w.safetensors": data})
    server.drop_after["w.safetensors"] = 2000
    with pytest.raises(snapshot.DownloadFailed, match="connection reset"):
        fetcher(server, retries=0)(DECIDER_REPO, "w.safetensors", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["w.safetensors"]))
    blobs = tmp_path / "hub" / "models--vllm-sr--d3-mini" / "blobs"
    partial = [p for p in blobs.iterdir() if p.name.endswith(".incomplete")]
    assert len(partial) == 1 and 0 < partial[0].stat().st_size < len(data)  # inventory sees "in progress"
    done = partial[0].stat().st_size
    fetcher(server)(DECIDER_REPO, "w.safetensors", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["w.safetensors"]))
    assert server.ranges[-1] == ("w.safetensors", f"bytes={done}-")
    assert (snapshot.snapshot_dir(tmp_path) / "w.safetensors").read_bytes() == data
    assert not any(p.name.endswith(".incomplete") for p in blobs.iterdir())


def test_http_fetch_restarts_when_the_server_ignores_range(tmp_path):
    data = b"abcdefghij" * 300
    server = FakeServer({"w.bin": data})
    server.drop_after["w.bin"] = 1000
    with pytest.raises(snapshot.DownloadFailed):
        fetcher(server, retries=0)(DECIDER_REPO, "w.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["w.bin"]))
    server.ignore_range = True
    fetcher(server)(DECIDER_REPO, "w.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["w.bin"]))
    assert (snapshot.snapshot_dir(tmp_path) / "w.bin").read_bytes() == data


def test_http_fetch_force_discards_partial_and_replaces_the_file(tmp_path):
    server = FakeServer({"c.json": b"{}"})
    f = fetcher(server)
    f(DECIDER_REPO, "c.json", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(server.files["c.json"]))
    server.files["c.json"] = b'{"a": 1}'
    f(DECIDER_REPO, "c.json", DECIDER_REVISION, str(tmp_path / "hub"), True, sha(server.files["c.json"]))
    assert (snapshot.snapshot_dir(tmp_path) / "c.json").read_bytes() == b'{"a": 1}'
    assert server.ranges[-1] == ("c.json", None)


def test_http_fetch_reports_progress_for_large_files(tmp_path):
    lines: list[str] = []
    server = FakeServer({"big.safetensors": b"z" * 50_000})
    fetcher(server, lines, progress_every_bytes=10_000)(DECIDER_REPO, "big.safetensors", DECIDER_REVISION,
                                                         str(tmp_path / "hub"), False, sha(server.files["big.safetensors"]))
    assert len([line for line in lines if "%" in line]) >= 3


def test_http_status_errors_fail_cleanly(tmp_path):
    def handler(request):
        return httpx.Response(404, text="Entry not found")
    f = snapshot.HttpFetcher(echo=lambda s: None, client=httpx.Client(transport=httpx.MockTransport(handler)),
                             endpoint="https://hub.test", retries=2, sleep=lambda s: None)
    with pytest.raises(snapshot.DownloadFailed, match="404"):
        f(DECIDER_REPO, "nope.json", DECIDER_REVISION, str(tmp_path / "hub"), False, "0" * 64)


def test_stale_incomplete_files_are_cleaned_after_success(hub, pins, tmp_path):
    blobs = hub.hf_home / "hub" / "models--vllm-sr--d3-mini" / "blobs"
    blobs.mkdir(parents=True)
    old = blobs / ("ab" * 32 + ".1234abcd.incomplete")  # huggingface_hub 2.x leftover, no live lock
    old.write_bytes(b"left by an interrupted huggingface_hub download")
    recent = blobs / ("cd" * 32 + ".9876fedc.incomplete")
    recent.write_bytes(b"someone is downloading right now")
    import os
    import time
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    ok, lines = ensure(hub, pins)
    assert ok and not old.exists() and recent.exists()



# --- review round 1: hash-before-link, names only from pins + manifest, locking, disk space --------------------

def test_mismatching_download_is_never_linked_or_kept(tmp_path):
    server = FakeServer({"w.safetensors": b"evil bytes" * 100})
    with pytest.raises(snapshot.HashMismatch):
        fetcher(server)(DECIDER_REPO, "w.safetensors", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(b"good"))
    repo = tmp_path / "hub" / "models--vllm-sr--d3-mini"
    assert not (snapshot.snapshot_dir(tmp_path) / "w.safetensors").exists()
    assert not (snapshot.snapshot_dir(tmp_path) / "w.safetensors").is_symlink()
    assert [p.name for p in (repo / "blobs").iterdir() if not p.name.startswith(".")] == []


@pytest.mark.parametrize("name", ["../evil.py", "a/../../evil.py", "/etc/passwd", "a//b", "./x", ""])
def test_unsafe_names_are_rejected_by_the_fetcher(tmp_path, name):
    server = FakeServer({name: b"x"})
    with pytest.raises(snapshot.DownloadFailed, match="unsafe"):
        fetcher(server)(DECIDER_REPO, name, DECIDER_REVISION, str(tmp_path / "hub"), False, sha(b"x"))
    assert server.ranges == []


def test_traversal_name_in_the_manifest_is_never_fetched(hub, pins, remote, tmp_path):
    files = {n: sha(d) for n, d in {**CODE, **WEIGHTS}.items()}
    files["../../../../evil.py"] = sha(b"evil")
    evil_manifest = json.dumps({"schema": "d3-package-manifest/1", "files_sha256": files}).encode()
    pins.write_text("".join(f"{sha(d)}  {n}\n" for n, d in {**CODE, "MODEL_MANIFEST.json": evil_manifest}.items()))
    hub.files = {**remote, "MODEL_MANIFEST.json": evil_manifest, "../../../../evil.py": b"evil"}
    ok, lines = ensure(hub, pins)
    assert not ok and any("invalid entry" in line for line in lines)
    assert all("evil" not in f for f, _ in hub.calls) and not any(f.endswith(".safetensors") for f, _ in hub.calls)


def test_files_not_in_pins_or_manifest_are_ignored(hub, pins, remote):
    hub.files = {**remote, "extra.bin": b"not listed anywhere", ".gitattributes": b"*.bin lfs"}
    ok, _ = ensure(hub, pins)
    assert ok and {f for f, _ in hub.calls}.isdisjoint({"extra.bin", ".gitattributes"})


def test_206_with_the_wrong_offset_restarts_from_zero(tmp_path):
    data = b"0123456789" * 300
    server = FakeServer({"w.bin": data})
    server.drop_after["w.bin"] = 1000
    with pytest.raises(snapshot.DownloadFailed):
        fetcher(server, retries=0)(DECIDER_REPO, "w.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(data))
    server.wrong_range_start = True  # answers 206 but from byte 0
    fetcher(server)(DECIDER_REPO, "w.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(data))
    assert (snapshot.snapshot_dir(tmp_path) / "w.bin").read_bytes() == data


def test_a_second_concurrent_download_is_refused(hub, pins):
    import fcntl
    blobs = hub.hf_home / "hub" / "models--vllm-sr--d3-mini" / "blobs"
    blobs.mkdir(parents=True)
    with open(blobs / ".dgxmm-decider.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(snapshot.DownloadFailed, match="another download"):
            ensure(hub, pins)
        assert ensure(hub, pins, verify_only=True)[0] is False  # verifying needs no lock
    assert ensure(hub, pins)[0]


def test_live_huggingface_download_is_not_swept(hub, pins):
    import fcntl
    import os
    import time
    repo = hub.hf_home / "hub" / "models--vllm-sr--d3-mini"
    (repo / "blobs").mkdir(parents=True)
    locks = hub.hf_home / "hub" / ".locks" / "models--vllm-sr--d3-mini"
    locks.mkdir(parents=True)
    blob = "ef" * 32
    live = repo / "blobs" / f"{blob}.0badc0de.incomplete"
    live.write_bytes(b"being written by another huggingface_hub process")
    os.utime(live, (time.time() - 3600, time.time() - 3600))
    with open(locks / f"{blob}.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert ensure(hub, pins)[0]
        assert live.exists()
    snapshot.clean_stale_incomplete(hub.hf_home)
    assert not live.exists()  # lock released: now a leftover


def test_tampered_blob_is_deleted_and_replaced(tmp_path):
    data = b"good weights" * 50
    server = FakeServer({"w.safetensors": data})
    f = fetcher(server)
    f(DECIDER_REPO, "w.safetensors", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(data))
    blob = tmp_path / "hub" / "models--vllm-sr--d3-mini" / "blobs" / sha(data)
    blob.write_bytes(b"tampered")
    f(DECIDER_REPO, "w.safetensors", DECIDER_REVISION, str(tmp_path / "hub"), True, sha(data))
    assert blob.read_bytes() == data
    blobs = [p for p in blob.parent.iterdir() if not p.name.startswith(".")]
    assert all(sha(p.read_bytes()) == p.name for p in blobs)


def test_not_enough_disk_fails_before_the_large_files(hub, pins):
    ok, lines = ensure(hub, pins, free_bytes=lambda path: 100)
    assert not ok and any("not enough disk" in line for line in lines)
    assert not any(f.endswith(".safetensors") for f, _ in hub.calls)


def test_symlinked_directories_cannot_redirect_writes_outside_the_snapshot(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "hub" / "models--vllm-sr--d3-mini" / "snapshots" / DECIDER_REVISION
    root.mkdir(parents=True)
    (root / "sub").symlink_to(outside)
    server = FakeServer({"sub/x.bin": b"payload"})
    with pytest.raises(snapshot.DownloadFailed, match="outside"):
        fetcher(server)(DECIDER_REPO, "sub/x.bin", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(b"payload"))
    assert list(outside.iterdir()) == []


def test_symlinked_blobs_directory_is_refused(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    repo = tmp_path / "hub" / "models--vllm-sr--d3-mini"
    repo.mkdir(parents=True)
    (repo / "blobs").symlink_to(outside)
    server = FakeServer({"c.json": b"{}"})
    with pytest.raises(snapshot.DownloadFailed, match="outside"):
        fetcher(server)(DECIDER_REPO, "c.json", DECIDER_REVISION, str(tmp_path / "hub"), False, sha(b"{}"))
    assert list(outside.iterdir()) == []


def test_another_supported_model_downloads_into_its_own_repo_at_its_pinned_revision(tmp_path, remote, pins):
    from dgxmm.engines.decision import DECIDER_MODELS
    nano = "vllm-sr/d3-nano"
    rev = DECIDER_MODELS[nano].revision
    seen = []

    def list_files(repo_id, revision):
        assert (repo_id, revision) == (nano, rev)
        return [(n, len(d)) for n, d in remote.items()]

    def fetch(repo_id, filename, revision, cache_dir, force, sha256):
        assert (repo_id, revision) == (nano, rev)
        seen.append(filename)
        path = snapshot.snapshot_dir(tmp_path / "hf", nano) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)
    assert snapshot.snapshot_dir(tmp_path, nano) == tmp_path / "hub" / "models--vllm-sr--d3-nano" / "snapshots" / rev
    lines: list[str] = []
    assert snapshot.ensure_snapshot(tmp_path / "hf", pins, fetch=fetch, list_files=list_files, echo=lines.append,
                                    repo=nano)
    assert "d3_server.py" in seen and any(f"{nano}@{rev[:12]}" in ln for ln in lines)
    assert not (tmp_path / "hf" / "hub" / "models--vllm-sr--d3-mini").exists()
    with pytest.raises(ValueError):
        snapshot.snapshot_dir(tmp_path, "vllm-sr/not-pinned")
