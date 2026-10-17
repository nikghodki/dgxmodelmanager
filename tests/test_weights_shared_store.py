"""Weights delete with the huggingface_hub >= 2 shared blob store (hub/blobs/<xx>/<xet hash>).

Repo blobs (models--org--name/blobs/<etag>) are relative symlinks into the cache-wide store, and each store
entry has a `<hash>.refs` manifest listing the repo blobs that link to it. Deleting a model must also remove the
store entries that ONLY that model references, keep every entry another repo still uses, and report the bytes
really freed (each file once).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dgxmm.inventory.service import InventoryService
from dgxmm.inventory.weights import WeightsDeleter
from tests.conftest import QWEN_TINY, write_gguf
from tests.fakes import FakeDockerClient
from tests.test_weights_delete import simulate_rm


def hub(settings) -> Path:
    return Path(settings.hf_cache.local) / "hub"


def store(settings) -> Path:
    s = hub(settings) / "blobs"
    s.mkdir(exist_ok=True)
    (s / ".huggingface-shared-blobs").write_text("1\n")
    return s


def xet(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def share(settings, repo_blob: Path, seed: str | None = None, manifest: bool = True) -> Path:
    """Move a repo blob into the shared store the way huggingface_hub does and return the store path.
    With `seed` given and the store entry already present, the repo blob just links to it (deduplicated)."""
    h = xet(seed or repo_blob.name)
    entry = store(settings) / h[:2] / h
    entry.parent.mkdir(exist_ok=True)
    if entry.exists():
        repo_blob.unlink()
    else:
        os.replace(repo_blob, entry)
        os.chmod(entry, 0o444)
    os.symlink(os.path.relpath(entry, repo_blob.parent), repo_blob)
    if manifest:
        with open(entry.with_name(h + ".refs"), "a") as f:
            f.write(repo_blob.relative_to(hub(settings)).as_posix() + "\n")
    (entry.with_name(h + ".lock")).touch()
    return entry


def share_all(settings, repo_dir: Path, seeds: dict[str, str] | None = None) -> dict[str, Path]:
    out = {}
    for b in sorted((repo_dir / "blobs").iterdir()):
        if b.is_file() and not b.is_symlink() and not b.name.endswith(".incomplete"):
            out[b.name] = share(settings, b, (seeds or {}).get(b.name))
    return out


@pytest.fixture
def make(settings, db):
    def _make(sdk=None):
        inv = InventoryService(settings)
        inv.refresh()
        from dgxmm.docker_ctl import DockerController
        return WeightsDeleter(settings, db, inv, docker=DockerController(sdk) if sdk else None), inv
    return _make


def test_exclusive_store_entries_are_removed_and_their_bytes_reported(settings, fake_hf_repo, make):
    snap = fake_hf_repo("org/alone", config=QWEN_TINY, files={"model.safetensors": 450_000})
    entries = share_all(settings, snap.parent.parent)
    real = sum(e.stat().st_size for e in entries.values())
    d, inv = make()
    assert inv.get("org/alone").complete
    p = d.preview("org/alone")
    assert p["bytes"] >= real >= 450_000
    for e in entries.values():
        assert d.to_host(e) in p["paths"]
    out = d.delete("org/alone")
    assert out["freed_bytes"] == p["bytes"]
    for e in entries.values():
        assert not os.path.lexists(e) and not os.path.lexists(e.with_name(e.name + ".refs"))
    assert (hub(settings) / "blobs").is_dir() and (hub(settings) / "blobs" / ".huggingface-shared-blobs").is_file()


def test_store_entries_used_by_another_repo_are_kept(settings, fake_hf_repo, make):
    a = fake_hf_repo("org/a", config=QWEN_TINY, files={"model.safetensors": 300_000})
    b = fake_hf_repo("org/b", config=QWEN_TINY, files={"model.safetensors": 300_000})
    seeds = {f"{'a' * 8}-model.safetensors": "same-weights"}
    ea = share_all(settings, a.parent.parent, seeds)
    eb = share_all(settings, b.parent.parent, seeds)
    shared = ea[f"{'a' * 8}-model.safetensors"]
    assert shared == eb[f"{'a' * 8}-model.safetensors"]
    d, inv = make()
    p = d.preview("org/a")
    assert d.to_host(shared) not in p["paths"]
    assert p["bytes"] < 300_000  # only a's own config entry is freed
    out = d.delete("org/a")
    assert shared.is_file() and out["freed_bytes"] == p["bytes"]
    assert inv.get("org/b") is not None and inv.get("org/b").complete


def test_a_reference_without_a_manifest_still_keeps_the_entry(settings, fake_hf_repo, make):
    a = fake_hf_repo("org/a", config=QWEN_TINY, files={"model.safetensors": 1000})
    b = fake_hf_repo("org/b", config=QWEN_TINY, files={"model.safetensors": 1000})
    name = f"{'a' * 8}-model.safetensors"
    entry = share(settings, a.parent.parent / "blobs" / name, "w")
    share(settings, b.parent.parent / "blobs" / name, "w", manifest=False)  # b's link is not in the manifest
    d, _ = make()
    d.delete("org/a")
    assert entry.is_file()


def test_a_link_created_while_deleting_keeps_the_entry(settings, fake_hf_repo, make, monkeypatch):
    """A download that links the entry between the plan and the removal records it in the manifest (under the
    entry's lock) before the link is visible: the deleter re-reads the manifest under the same lock."""
    a = fake_hf_repo("org/a", config=QWEN_TINY, files={"model.safetensors": 2000})
    name = f"{'a' * 8}-model.safetensors"
    entry = share(settings, a.parent.parent / "blobs" / name, "w")
    d, _ = make()
    real_run = d._run_helpers

    def racing(targets):
        c = hub(settings) / "models--org--c" / "blobs"
        c.mkdir(parents=True)
        with open(entry.with_name(entry.name + ".refs"), "a") as f:
            f.write("models--org--c/blobs/etag\n")
        os.symlink(os.path.relpath(entry, c), c / "etag")
        return real_run(targets)
    monkeypatch.setattr(d, "_run_helpers", racing)
    out = d.delete("org/a")
    assert entry.is_file()
    assert out["freed_bytes"] < 2000 and any("kept" in n for n in out["notes"])


def test_each_file_is_counted_once(settings, fake_hf_repo, make):
    """Two revisions whose repo blobs link to the same store entry free its bytes once."""
    s1 = fake_hf_repo("org/a", rev="a" * 40, config=QWEN_TINY, files={"model.safetensors": 200_000})
    fake_hf_repo("org/a", rev="b" * 40, config=QWEN_TINY, files={"model.safetensors": 200_000}, main=False)
    repo = s1.parent.parent
    seeds = {f"{r * 8}-model.safetensors": "w" for r in "ab"} | {f"{r * 8}-config.json": "c" for r in "ab"}
    entries = share_all(settings, repo, seeds)
    uniq = {e for e in entries.values()}
    assert len(uniq) == 2
    own = sum(f.lstat().st_size for f in repo.rglob("*") if f.is_file() and not f.is_symlink())  # refs/main
    d, _ = make()
    p = d.preview("org/a")
    assert p["bytes"] == sum(e.stat().st_size for e in uniq) + own
    assert d.delete("org/a")["freed_bytes"] == p["bytes"]
    assert not any(os.path.lexists(e) for e in uniq)


def test_store_lookalikes_outside_the_cache_are_never_removed(settings, fake_hf_repo, tmp_path, make):
    """A repo blob linking through a store prefix that is itself a symlink out of the cache: the target
    outside is not a store entry and survives; the delete of the repo still works."""
    a = fake_hf_repo("org/a", config=QWEN_TINY, files={"model.safetensors": 1000})
    outside = tmp_path / "outside"
    outside.mkdir()
    h = xet("x")
    victim = outside / h
    victim.write_bytes(b"precious")
    os.symlink(outside, store(settings) / h[:2])
    blob = a.parent.parent / "blobs" / f"{'a' * 8}-model.safetensors"
    blob.unlink()
    os.symlink(os.path.relpath(store(settings) / h[:2] / h, blob.parent), blob)
    d, _ = make()
    d.delete("org/a")
    assert victim.read_bytes() == b"precious" and (store(settings) / h[:2]).is_symlink()


def test_cleanup_of_an_incomplete_download_removes_its_exclusive_store_entries(settings, fake_hf_repo, make):
    snap = fake_hf_repo("org/partial", config=QWEN_TINY, files={"model.safetensors": 5000}, incomplete=True)
    entries = share_all(settings, snap.parent.parent)
    t = 0
    for b in hub(settings).glob("models--*/blobs/*.incomplete"):
        os.utime(b, (t, t))
    d, inv = make()
    assert not inv.get("org/partial").complete
    item = {i["ref"]: i for i in d.cleanup_preview()["items"]}["org/partial"]
    assert item["eligible"] and item["bytes"] >= 5000
    out = d.cleanup(["org/partial"])
    assert out["freed_bytes"] == item["bytes"]
    assert not any(os.path.lexists(e) for e in entries.values())


def test_gguf_in_a_snapshot_with_a_store_entry_removes_link_repo_blob_and_entry(settings, fake_hf_repo, make):
    snap = fake_hf_repo("unsloth/Qwen3-GGUF", files={}, main=True)
    blob = write_gguf(snap.parent.parent / "blobs" / "gg", vocab=10, ftype=7)
    os.symlink(os.path.relpath(blob, snap), snap / "Q8.gguf")
    entry = share(settings, blob)
    size = entry.stat().st_size
    d, _ = make()
    ref = str(snap / "Q8.gguf")
    p = d.preview(ref)
    assert p["bytes"] == size and d.to_host(entry) in p["paths"]
    out = d.delete(ref)
    assert out["freed_bytes"] == size
    assert not os.path.lexists(snap / "Q8.gguf") and not os.path.lexists(blob) and not os.path.lexists(entry)


def test_gguf_entry_shared_with_another_repo_is_kept(settings, fake_hf_repo, make):
    snap = fake_hf_repo("unsloth/Qwen3-GGUF", files={}, main=True)
    blob = write_gguf(snap.parent.parent / "blobs" / "gg", vocab=10, ftype=7)
    os.symlink(os.path.relpath(blob, snap), snap / "Q8.gguf")
    entry = share(settings, blob, "g")
    other = fake_hf_repo("mirror/Qwen3-GGUF", files={}, main=True)
    oblob = other.parent.parent / "blobs" / "gg"
    write_gguf(oblob, vocab=10, ftype=7)
    share(settings, oblob, "g")
    os.symlink(os.path.relpath(oblob, other), other / "Q8.gguf")
    d, _ = make()
    out = d.delete(str(snap / "Q8.gguf"))
    assert out["freed_bytes"] == 0 and entry.is_file() and (other / "Q8.gguf").is_file()


def test_root_owned_store_entry_is_removed_by_the_helper(settings, fake_hf_repo, make, monkeypatch):
    snap = fake_hf_repo("org/a", config=QWEN_TINY, files={"model.safetensors": 1000})
    entries = share_all(settings, snap.parent.parent)
    import dgxmm.inventory.weights as w
    real_unlink = os.unlink
    stored = {str(e) for e in entries.values()} | {str(e) + ".refs" for e in entries.values()}
    as_root = [False]

    def unlink(p, *a, **kw):
        if str(p) in stored and not as_root[0]:
            raise PermissionError(13, "Permission denied", str(p))
        return real_unlink(p, *a, **kw)
    monkeypatch.setattr(w.os, "unlink", unlink)
    sdk = FakeDockerClient()
    rm = simulate_rm(settings)

    def helper(call):  # the helper container runs as root
        as_root[0] = True
        try:
            rm(call)
        finally:
            as_root[0] = False
    sdk.helper_action = helper
    d, _ = make(sdk)
    d.delete("org/a")
    assert not any(os.path.lexists(e) for e in entries.values())
    prefixes = {d.to_host(e.parent) for e in entries.values()}
    assert prefixes <= {r["mounts"][0]["Source"] for r in sdk.helper_runs}


def test_an_entry_locked_by_a_running_download_is_kept(settings, fake_hf_repo, make, monkeypatch):
    import fcntl

    import dgxmm.inventory.weights as w
    a = fake_hf_repo("org/a", config=QWEN_TINY, files={"model.safetensors": 2000})
    entry = share(settings, a.parent.parent / "blobs" / f"{'a' * 8}-model.safetensors", "w")
    monkeypatch.setattr(w, "STORE_LOCK_TIMEOUT_S", 0.2)
    d, _ = make()
    with open(entry.with_name(entry.name + ".lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # huggingface_hub is linking or publishing this entry
        out = d.delete("org/a")
    assert entry.is_file() and any("kept" in n for n in out["notes"])
    assert d.to_host(entry) not in out["paths"]
