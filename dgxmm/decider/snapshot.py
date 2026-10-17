"""Pre-download and verify a pinned decision-model snapshot (``vllm-sr/d3-mini`` at ``DECIDER_REVISION`` by default,
or any other model of ``DECIDER_MODELS`` at its pinned revision, e.g. ``vllm-sr/d3-nano``) in the
Hugging Face cache, so enabling the decider never waits on a multi-GB download and never serves unverified files.

Trust chain: the hash list baked into the repo/image (``deploy/decider/<model>-<revision>.sha256``) pins the code,
config, tokenizer and ``MODEL_MANIFEST.json``; only once the manifest matches its pin are its own ``files_sha256``
entries (the weights, licence and card) trusted. Only files named by the pin list or the manifest are ever fetched,
every one with its expected SHA-256; anything else the Hub lists is ignored, and ``assets/`` (model-card media, never
read by the runtime) is skipped. Every name is validated (relative, normalised, no ``..``) and its link must stay
inside ``snapshots/<revision>/``.

Downloads (``HttpFetcher``) are plain HTTPS GETs of ``<endpoint>/<repo>/resolve/<revision>/<file>``, hashed while
streaming and checked *before* anything is placed in the cache: a match is moved to ``blobs/<sha256>`` and linked
(relative) from ``snapshots/<revision>/<file>``; a mismatch deletes the partial file and is never linked. They resume
with HTTP Range requests (the ``Content-Range`` offset is checked) across runs from ``blobs/<key>.incomplete`` (which
the inventory reports as "download in progress") and after dropped connections. huggingface_hub 2.x restarts every
partial file from zero and its Xet transfer path stalled on the DGX's network, so it only lists the revision's files.
The HF token is never sent (the repo is public). One download at a time (``fcntl`` lock ``blobs/.dgxmm-decider.lock``).
Nothing here runs code from the snapshot. Used by the installer, the admin CLI (``decider download``) and the
Decision model page (dgxmm/downloads.py runs it in a worker thread, with progress and cancel).
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import posixpath
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote

import httpx

from dgxmm.engines.decision import DECIDER_MODELS, DECIDER_REPO, repo_dir

MANIFEST = "MODEL_MANIFEST.json"
SKIP_PREFIXES = ("assets/",)  # model-card media: not read by the runtime, not downloaded
CHUNK = 8 * 1024 * 1024
GIB = 1024 ** 3
DISK_MARGIN = 512 * 1024 * 1024
LOCK_NAME = ".dgxmm-decider.lock"
STALE_INCOMPLETE_S = 600  # huggingface_hub leftovers older than this (and not locked) are removed after a good run
_PIN_LINE = re.compile(r"^([0-9a-f]{64})\s+\*?(\S+)$")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_OWN_PARTIAL = re.compile(r"^[0-9a-f]{32}\.incomplete$")
_HF_PARTIAL = re.compile(r"^(?P<blob>[0-9a-f]{40,64})(\.[0-9a-f]{8})?\.incomplete$")

# (repo_id, filename, revision, cache_dir, force, expected sha256)
Fetch = Callable[[str, str, str, str, bool, str], str]
ListFiles = Callable[[str, str], list[tuple[str, int]]]     # (repo_id, revision) -> [(filename, size)]


class DownloadFailed(Exception):
    pass


class HashMismatch(DownloadFailed):
    pass


class DownloadCancelled(DownloadFailed):
    """``should_stop()`` asked to stop (a cancel from the Downloads page); the partial file is kept."""


def pinned_revision(repo: str) -> str:
    """The pinned revision of a supported decision model (any other repo is refused)."""
    m = DECIDER_MODELS.get(repo)
    if m is None:
        raise ValueError(f"{repo} is not a supported decision model ({', '.join(sorted(DECIDER_MODELS))})")
    return m.revision


def snapshot_dir(hf_home: Path | str, repo: str = DECIDER_REPO) -> Path:
    return Path(hf_home) / repo_dir(repo) / "snapshots" / pinned_revision(repo)


def _safe_name(name: str) -> bool:
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name or "\0" in name:
        return False
    norm = posixpath.normpath(name)
    return norm == name and norm != "." and not norm.startswith("../") and norm != ".."


def read_pins(path: Path | str) -> dict[str, str]:
    """``sha256sum`` format: ``<64 hex>  <relative path>`` per line."""
    pins: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        m = _PIN_LINE.match(line.strip())
        if not m or not _safe_name(m.group(2)):
            raise ValueError(f"{path}:{n}: not a sha256sum line for a relative path")
        pins[m.group(2)] = m.group(1)
    if MANIFEST not in pins:
        raise ValueError(f"{path}: {MANIFEST} is not pinned")
    return pins


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _state(snap: Path, name: str, want: str) -> str:
    """"ok" | "missing" | "changed"."""
    p = snap / name
    if not p.is_file():
        return "missing"
    return "ok" if file_sha256(p) == want else "changed"


def _manifest_hashes(snap: Path, pins: dict[str, str]) -> dict[str, str]:
    """The manifest's hashes for files the pin list does not cover (call only after the manifest matched its pin)."""
    data = json.loads((snap / MANIFEST).read_text())
    listed = data.get("files_sha256") if isinstance(data, dict) else None
    if not isinstance(listed, dict):
        raise ValueError(f"{MANIFEST} has no files_sha256")
    out = {}
    for name, digest in listed.items():
        if not _safe_name(name) or not _HEX64.fullmatch(str(digest)):
            raise ValueError(f"{MANIFEST} lists an invalid entry: {name!r}")
        if name not in pins and not name.startswith(SKIP_PREFIXES):
            out[name] = digest
    return out


def verify(snap: Path, pins: dict[str, str], echo: Callable[[str], None]) -> list[str]:
    """Problems with the snapshot on disk (empty list = complete and verified)."""
    problems = [f"{n}: {s}" + (" (SHA-256 differs from the pinned hash)" if s == "changed" else "")
                for n in pins if (s := _state(snap, n, pins[n])) != "ok"]
    if problems:
        return problems
    try:
        weights = _manifest_hashes(snap, pins)
    except (ValueError, OSError) as e:
        return [f"{MANIFEST}: {e}"]
    for name, digest in sorted(weights.items()):
        s = _state(snap, name, digest)
        if s != "ok":
            problems.append(f"{name}: {s}" + (f" (SHA-256 differs from {MANIFEST})" if s == "changed" else ""))
        else:
            echo(f"    ok   {name}")
    return problems


def _fix(snap: Path, names: list[str], hashes: dict[str, str], cache_dir: str, fetch: Fetch,
         echo: Callable[[str], None], sizes: dict[str, int], repo: str = DECIDER_REPO) -> list[str]:
    problems = []
    for i, name in enumerate(names, start=1):
        want = hashes[name]  # every fetched file has an expected digest
        state = _state(snap, name, want)
        if state == "ok":
            continue
        size = sizes.get(name)
        echo(f"    [{i}/{len(names)}] {name}" + (f" ({size / GIB:.2f} GiB)" if size and size >= GIB // 10 else "")
             + (" (changed on disk: downloading again)" if state == "changed" else ""))
        for force in ([True] if state == "changed" else [False, True]):
            try:
                fetch(repo, name, pinned_revision(repo), cache_dir, force, want)
            except HashMismatch:
                state = "changed"
                continue
            except DownloadFailed:
                raise
            except Exception as e:
                raise DownloadFailed(f"downloading {name} failed: {e}") from e
            state = _state(snap, name, want)  # defence in depth: the fetcher already checked it
            if state == "ok":
                break
        if state != "ok":
            problems.append(f"{name}: {state} after download (the Hub served bytes that do not match its pinned "
                            "SHA-256; nothing was linked)")
    return problems


class HttpFetcher:
    """``Fetch`` implementation: resumable HTTPS download of one file into the HF cache layout, verified before it
    is linked."""

    def __init__(self, echo: Callable[[str], None] = print, client: httpx.Client | None = None,
                 endpoint: str | None = None, retries: int = 5, progress_every_bytes: int = 512 * 1024 * 1024,
                 progress_every_s: float = 30.0, sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] | None = None):
        self.sleep, self.echo, self.retries, self.should_stop = sleep, echo, retries, should_stop
        self.endpoint = (endpoint or default_endpoint()).rstrip("/")
        self.progress_every_bytes, self.progress_every_s = progress_every_bytes, progress_every_s
        self.client = client or httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=True,
                                             headers={"user-agent": "dgxmm-decider-download"}, trust_env=True)

    def __call__(self, repo_id: str, filename: str, revision: str, cache_dir: str, force: bool,
                 sha256: str) -> str:
        if not _safe_name(filename):
            raise DownloadFailed(f"refusing unsafe file name {filename!r}")
        if not _HEX64.fullmatch(sha256 or ""):
            raise DownloadFailed(f"no expected SHA-256 for {filename}")
        repo = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
        blobs, root = repo / "blobs", repo / "snapshots" / revision
        link = root / filename
        if not os.path.normpath(link).startswith(os.path.normpath(root) + os.sep):
            raise DownloadFailed(f"refusing unsafe file name {filename!r}")
        blobs.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        # Real paths, not just normalised names: a symlinked directory must not redirect writes elsewhere.
        real_repo = os.path.realpath(repo)
        if os.path.realpath(blobs) != os.path.join(real_repo, "blobs") or \
                os.path.realpath(root) != os.path.join(real_repo, "snapshots", revision):
            raise DownloadFailed(f"{repo}: blobs/ or snapshots/{revision} resolves outside the repository; refusing")
        self._check_inside(link.parent, root)
        key = hashlib.sha256(f"{revision}/{filename}".encode()).hexdigest()[:32]
        partial = blobs / f"{key}.incomplete"
        if force:
            partial.unlink(missing_ok=True)
            self._drop_link(link, blobs)
        url = f"{self.endpoint}/{repo_id}/resolve/{revision}/{quote(filename)}"
        attempt = 0
        while True:
            try:
                digest = self._download(url, partial, filename)
                break
            except DownloadFailed:
                raise
            except (httpx.TransportError, OSError) as e:
                if attempt >= self.retries:
                    raise DownloadFailed(f"{e} (after {attempt + 1} attempts; the partial file is kept)") from e
                attempt += 1
                self.echo(f"      connection problem ({e}); resuming (attempt {attempt + 1})")
                self.sleep(min(2 ** attempt, 30))
        if digest != sha256:
            partial.unlink(missing_ok=True)
            raise HashMismatch(f"{filename}: downloaded SHA-256 {digest[:12]}… differs from the expected "
                               f"{sha256[:12]}…; discarded")
        blob = blobs / digest
        link.parent.mkdir(parents=True, exist_ok=True)
        self._check_inside(link.parent, root)  # again: the tree may have changed while downloading
        os.replace(partial, blob)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
        return str(link)

    @staticmethod
    def _check_inside(directory: Path, root: Path) -> None:
        """``directory`` (or its nearest existing ancestor) really is ``root`` or below it."""
        d = directory
        while not d.exists() and d != root and d != d.parent:
            d = d.parent
        real, real_root = os.path.realpath(d), os.path.realpath(root)
        if real != real_root and not real.startswith(real_root + os.sep):
            raise DownloadFailed(f"{directory} resolves outside {root}; refusing")

    @staticmethod
    def _drop_link(link: Path, blobs: Path) -> None:
        """Remove a file being replaced, and the blob it points to when that blob is inside ``blobs`` (it did not
        match: keeping it would leave a mis-named blob)."""
        if link.is_symlink():
            target = (link.parent / os.readlink(link)).resolve()
            if target.parent == blobs.resolve() and target.is_file():
                target.unlink()
        if link.is_symlink() or link.exists():
            link.unlink()

    def _download(self, url: str, partial: Path, filename: str) -> str:
        """Download into ``partial`` (resuming); returns the SHA-256 of the complete file."""
        have = partial.stat().st_size if partial.exists() else 0
        h = hashlib.sha256()
        if have:
            with open(partial, "rb") as f:
                while chunk := f.read(CHUNK):
                    h.update(chunk)
        headers = {"range": f"bytes={have}-"} if have else {}
        with self.client.stream("GET", url, headers=headers) as r:
            if r.status_code == 416 and have:
                return h.hexdigest()  # the partial file already holds every byte
            if r.status_code not in (200, 206):
                raise DownloadFailed(f"GET {filename}: HTTP {r.status_code}")
            if r.status_code == 206:
                m = re.match(r"bytes (\d+)-", r.headers.get("content-range", ""))
                if not m or int(m.group(1)) != have:
                    partial.unlink(missing_ok=True)  # cannot append: start over on the next attempt
                    raise httpx.RemoteProtocolError(f"unexpected Content-Range for {filename}; restarting")
            if r.status_code == 200 and have:
                have, h = 0, hashlib.sha256()  # the server ignored the range: start over
            length = r.headers.get("content-length")
            total = have + int(length) if length and length.isdigit() else None
            next_bytes, next_t = have + self.progress_every_bytes, time.monotonic() + self.progress_every_s
            t0, start = time.monotonic(), have
            with open(partial, "ab" if have else "wb") as f:
                for chunk in r.iter_bytes():  # as received: a dropped connection keeps every byte so far
                    if self.should_stop is not None and self.should_stop():
                        raise DownloadCancelled(f"{filename}: download cancelled (the partial file is kept)")
                    f.write(chunk)
                    h.update(chunk)
                    have += len(chunk)
                    if total and (have >= next_bytes or time.monotonic() >= next_t) and have < total:
                        rate = (have - start) / max(time.monotonic() - t0, 1e-6) / 1e6
                        self.echo(f"      {have / GIB:.2f} / {total / GIB:.2f} GiB ({100 * have // total}%), "
                                  f"{rate:.0f} MB/s")
                        next_bytes, next_t = have + self.progress_every_bytes, time.monotonic() + self.progress_every_s
        return h.hexdigest()


def default_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def hf_list_files(repo_id: str, revision: str) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True, token=False)
    return [(s.rfilename, int(s.size or 0)) for s in info.siblings or []]


@contextlib.contextmanager
def download_lock(hf_home: Path | str, repo: str = DECIDER_REPO) -> Iterator[None]:
    """Exclusive, non-blocking: a second downloader fails fast instead of interleaving writes."""
    blobs = Path(hf_home) / repo_dir(repo) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    with open(blobs / LOCK_NAME, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES):
                raise DownloadFailed("another download of the decider snapshot is running; wait for it to finish")
            raise
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _hf_lock_free(hf_home: Path, blob: str, repo: str = DECIDER_REPO) -> bool:
    lock = Path(hf_home) / "hub" / ".locks" / repo_dir(repo).removeprefix("hub/") / f"{blob}.lock"
    if not lock.exists():
        return True
    try:
        with open(lock, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return True
    except OSError:
        return False


def clean_stale_incomplete(hf_home: Path | str, own: bool = False, repo: str = DECIDER_REPO) -> int:
    """Remove interrupted-download leftovers of this repo: huggingface_hub partials older than STALE_INCOMPLETE_S
    whose hf lock nobody holds, and (``own``, caller holds our lock and every file is complete) our own partials."""
    hf_home = Path(hf_home)
    blobs = hf_home / repo_dir(repo) / "blobs"
    n = 0
    if not blobs.is_dir():
        return 0
    cutoff = time.time() - STALE_INCOMPLETE_S
    for p in blobs.glob("*.incomplete"):
        try:
            if _OWN_PARTIAL.match(p.name):
                if own:
                    p.unlink()
                    n += 1
                continue
            m = _HF_PARTIAL.match(p.name)
            if m and p.stat().st_mtime < cutoff and _hf_lock_free(hf_home, m.group("blob"), repo):
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


def _partial_bytes(blobs: Path, name: str, revision: str) -> int:
    key = hashlib.sha256(f"{revision}/{name}".encode()).hexdigest()[:32]
    p = blobs / f"{key}.incomplete"
    return p.stat().st_size if p.exists() else 0


def ensure_snapshot(hf_home: Path | str, pins_path: Path | str, *, fetch: Fetch | None = None,
                    list_files: ListFiles = hf_list_files, echo: Callable[[str], None] = print,
                    verify_only: bool = False, repo: str = DECIDER_REPO,
                    free_bytes: Callable[[Path], int] = lambda p: shutil.disk_usage(p).free,
                    on_plan: Callable[[dict[str, int]], None] | None = None) -> bool:
    """Make the pinned snapshot of ``repo`` (a supported decision model; ``pins_path`` is its hash list) complete and
    verified; True on success. Raises DownloadFailed on network errors (re-running resumes) or when another
    download holds the lock. ``on_plan({file: size})``: the files the snapshot consists of, as soon as they are known
    (the pinned files first, then with the manifest's), for progress reporting."""
    pins = read_pins(pins_path)
    snap = snapshot_dir(hf_home, repo)
    label = f"{repo}@{pinned_revision(repo)[:12]}"
    echo(f"==> Checking {label} in {snap.parent.parent}")
    if verify_only:
        problems = verify(snap, pins, echo)
        for p in problems:
            echo(f"    problem: {p}")
        if not problems:
            echo(f"    verified: every file matches its pinned SHA-256 ({label})")
        return not problems
    with download_lock(hf_home, repo):
        return _ensure_locked(Path(hf_home), pins, snap, label, fetch, list_files, echo, free_bytes, repo,
                              on_plan or (lambda files: None))


def _sizes_on_disk(snap: Path, names) -> dict[str, int]:
    return {n: (snap / n).stat().st_size for n in names}


def _ensure_locked(hf_home: Path, pins, snap, label, fetch, list_files, echo, free_bytes, repo, on_plan) -> bool:
    revision = pinned_revision(repo)
    def fail(problems: list[str]) -> bool:
        for p in problems:
            echo(f"    problem: {p}")
        return False

    if not verify(snap, pins, echo):
        on_plan(_sizes_on_disk(snap, [*pins, *_manifest_hashes(snap, pins)]))
        clean_stale_incomplete(hf_home, own=True, repo=repo)
        echo(f"    verified: every file matches its pinned SHA-256 ({label})")
        return True
    try:
        listing = [(n, s) for n, s in list_files(repo, revision) if not n.startswith(SKIP_PREFIXES)]
    except Exception as e:
        raise DownloadFailed(f"could not list {label} on the Hugging Face Hub: {e}") from e
    sizes = dict(listing)
    missing_pins = sorted(set(pins) - set(sizes))
    if missing_pins:
        return fail([f"the Hub does not list pinned file(s): {', '.join(missing_pins)}"])
    on_plan({n: sizes[n] for n in pins})
    echo(f"==> Downloading {label} (pinned and manifest files only; an interrupted download resumes)")
    cache_dir = str(hf_home / "hub")
    fetch = fetch or HttpFetcher(echo=echo)
    # 1. The pinned files (code, config, tokenizer, manifest) first: nothing large is fetched unless they match.
    problems = _fix(snap, sorted(pins), pins, cache_dir, fetch, echo, sizes, repo)
    if problems:
        return fail(problems)
    # 2. The manifest's other files (weights, licence, card), each with its manifest hash; smallest first.
    try:
        weights = _manifest_hashes(snap, pins)
    except (ValueError, OSError) as e:
        return fail([f"{MANIFEST}: {e}"])
    absent = sorted(set(weights) - set(sizes))
    if absent:
        return fail([f"the Hub does not list file(s) from {MANIFEST}: {', '.join(absent)}"])
    on_plan({n: sizes[n] for n in [*pins, *weights]})
    rest = sorted(weights, key=lambda n: (sizes[n], n))
    blobs = hf_home / repo_dir(repo) / "blobs"
    todo = [n for n in rest if _state(snap, n, weights[n]) != "ok"]
    need = sum(max(sizes[n] - _partial_bytes(blobs, n, revision), 0) for n in todo)
    free = free_bytes(blobs)
    if need and free < need + DISK_MARGIN:
        return fail([f"not enough disk at {blobs}: {need / GIB:.1f} GiB still to download (+0.5 GiB margin), "
                     f"{free / GIB:.1f} GiB free"])
    echo(f"    {len(todo)} file(s), {need / GIB:.1f} GiB to download")
    problems = _fix(snap, rest, weights, cache_dir, fetch, echo, sizes, repo)
    if problems:
        return fail(problems)
    clean_stale_incomplete(hf_home, own=True, repo=repo)
    echo(f"    verified: every file matches its pinned SHA-256 ({label})")
    return True
