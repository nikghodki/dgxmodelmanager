"""Background Hugging Face downloads with progress (spec §7).

Each file is fetched by ``hf_hub_download`` in a child process (``SubprocessDownloader``), so a cancel stops the
file in flight at once: the child's process group is killed, which aborts its HTTP stream, and its partial
``blobs/*.incomplete`` files are removed. While a download runs, the repo dir holds a marker listing the files it
fetches (``DOWNLOAD_MARKER``); it is removed only when every file is on disk, and the inventory reports a snapshot
that lacks a listed file as incomplete — so a cancelled or failed download is never offered as a complete model.

A supported decision model (``DECIDER_MODELS``) never takes that path: it is fetched by the pinned snapshot
downloader (dgxmm/decider/snapshot.py: its pinned revision only, every file hashed while streaming and verified before
it is linked, resumable with HTTP Range, the HF token never sent), the same code the installer and the admin CLI use,
in a worker thread that a cancel stops between two chunks. Progress and state are the same Download rows and SSE
events, so the Downloads page and the Decision model page show it like any other download.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from dgxmm.db import Database, Download
from dgxmm.inventory.artifact import REPO_RE
from dgxmm.inventory.hf_scan import DOWNLOAD_MARKER

log = logging.getLogger(__name__)
GIB = 1024**3
DEFAULT_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "*.tiktoken", "tokenizer*", "*.md"]
EXCLUDE = ["original/*", "*.pth", "*.pt", "*.bin", "*.onnx", "*.msgpack", "*.h5", "*.gguf"]
PATTERN_RE = r"^[A-Za-z0-9._*?\[\]/-]{1,200}$"


class DownloadError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class DownloadCancelled(Exception):
    pass


# Runs in the child: fetch one file, report a failure as one JSON line on stdout. The token comes from the
# environment (never argv, which other local users can read); progress bars are off.
CHILD_CODE = """
import json, os, sys
repo, filename, revision, cache_dir = sys.argv[1:5]
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=repo, filename=filename, revision=revision or None, cache_dir=cache_dir,
                    token=os.environ.get("HF_TOKEN") or None)
except BaseException as e:
    print(json.dumps({"type": type(e).__name__, "message": str(e)[:2000],
                      "filename": str(getattr(e, "filename", "") or "")}), flush=True)
    sys.exit(1)
"""
MAX_CHILD_REPORT = 64 * 1024


class _ChildFailed(Exception):
    def __init__(self, type_name: str, message: str):
        super().__init__(message)
        self.type_name, self.message = type_name, message

    def __str__(self) -> str:
        return self.message


class _ProcessJob:
    """One file fetched by a child process; ``kill()`` stops it at once (whole process group)."""

    def __init__(self, argv: list[str], env: dict[str, str]):
        self.killed = False
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, start_new_session=True)

    def wait(self) -> None:
        out = self.proc.stdout.read(MAX_CHILD_REPORT) if self.proc.stdout else b""
        rc = self.proc.wait()
        if self.killed:
            raise DownloadCancelled()
        if rc == 0:
            return
        report = {}
        for line in reversed(out.decode("utf-8", "replace").splitlines()):
            try:
                report = json.loads(line)
                break
            except ValueError:
                continue
        name = str(report.get("type") or "DownloadError")
        msg = str(report.get("message") or f"download process exited with {rc}")
        if name == "PermissionError":
            raise PermissionError(13, msg, report.get("filename") or None)
        raise _ChildFailed(name, msg)

    def kill(self) -> None:
        with self._lock:
            self.killed = True
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class _ThreadJob:
    """A plain callable downloader (tests, embedding): cannot be interrupted, so a cancel waits for the file."""

    def __init__(self, fn: Callable, kwargs: dict):
        self.fn, self.kwargs, self.killed = fn, kwargs, False

    def wait(self) -> None:
        self.fn(**self.kwargs)
        if self.killed:
            raise DownloadCancelled()

    def kill(self) -> None:
        self.killed = True


class SubprocessDownloader:
    """Default downloader: ``hf_hub_download`` in a child process (``python -I``), one file per process."""

    def __init__(self, python: str | None = None, code: str = CHILD_CODE):
        self.python, self.code = python or sys.executable, code

    def spawn(self, repo_id: str, filename: str, revision: str | None, cache_dir: str, token: str | None):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        if token:
            env["HF_TOKEN"] = token
        argv = [self.python, "-I", "-B", "-c", self.code, repo_id, filename, revision or "", cache_dir]
        return _ProcessJob(argv, env)


def _spawn(downloader, **kw):
    if hasattr(downloader, "spawn"):
        return downloader.spawn(**kw)
    return _ThreadJob(downloader, kw)


class DownloadService:
    def __init__(self, settings, db: Database, inventory, hub, events, downloader: Callable | None = None,
                 disk_usage: Callable = shutil.disk_usage, token: str | None = None):
        self.settings, self.db, self.inventory, self.hub, self.events = settings, db, inventory, hub, events
        self.downloader = downloader or SubprocessDownloader()
        self.disk_usage, self.token = disk_usage, token
        self.hub_dir = Path(settings.hf_cache.local) / "hub"
        self._cancel: set[int] = set()
        self._tasks: dict[int, asyncio.Task] = {}
        self._sizes: dict[int, dict[str, int]] = {}
        self._jobs: dict[int, object] = {}  # download id -> file job in flight
        self.delete_guard: Callable[[str], bool] | None = None  # WeightsDeleter.is_being_deleted
        # decision models: download id -> pinned hash list; stop flags of the snapshot downloads in flight
        self._pinned: dict[int, Path] = {}
        self._stops: dict[int, threading.Event] = {}
        self.snapshot_fetch: Callable | None = None  # (echo, should_stop) -> snapshot.Fetch; None = HttpFetcher
        self.snapshot_list: Callable | None = None   # snapshot.ListFiles; None = the Hub's listing

    def view(self, dl_id: int) -> dict:
        with self.db.session() as s:
            d = s.get(Download, dl_id)
            return {"id": d.id, "repo_id": d.repo_id, "revision": d.revision, "files": d.files, "state": d.state,
                    "bytes_done": d.bytes_done, "bytes_total": d.bytes_total, "error": d.error,
                    "created_at": d.created_at.isoformat()}

    def list(self, limit: int = 50) -> list[dict]:
        with self.db.session() as s:
            ids = [d.id for d in s.scalars(select(Download).order_by(Download.id.desc()).limit(limit))]
        return [self.view(i) for i in ids]

    def _update(self, dl_id: int, **fields) -> None:
        with self.db.session() as s:
            d = s.get(Download, dl_id)
            for k, v in fields.items():
                setattr(d, k, v)
        self.events.publish("download", self.view(dl_id))

    def start(self, repo_id: str, revision: str | None = None, include: list[str] | None = None,
              user_id: int | None = None) -> dict:
        from dgxmm.engines.decision import DECIDER_MODELS
        if repo_id in DECIDER_MODELS:  # never hf_hub_download: pinned, verified, resumable (module docstring)
            return self.start_pinned(repo_id, revision, user_id)
        if not re.fullmatch(REPO_RE, repo_id) or ".." in repo_id:
            raise DownloadError("invalid_repo", "repo id must look like org/name", 422)
        if self.delete_guard is not None and self.delete_guard(repo_id):
            raise DownloadError("model_being_deleted", f"{repo_id} is being deleted from disk; try again shortly", 409)
        if revision is not None and not re.fullmatch(r"^[A-Za-z0-9._/-]{1,64}$", revision):
            raise DownloadError("invalid_revision", "invalid revision", 422)
        for p in include or []:
            if not re.fullmatch(PATTERN_RE, p) or ".." in p:
                raise DownloadError("invalid_pattern", f"invalid file pattern {p!r}", 422)
        with self.db.session() as s:
            if s.scalar(select(Download).where(Download.repo_id == repo_id, Download.state.in_(("queued", "running")))):
                raise DownloadError("already_downloading", f"{repo_id} is already downloading", 409)
        try:
            details = self.hub.details(repo_id)
        except Exception as e:
            name = type(e).__name__
            if name == "GatedRepoError":
                raise DownloadError("gated", "gated repo: accept the licence on huggingface.co and add an HF token", 403)
            if name in ("RepositoryNotFoundError", "RevisionNotFoundError"):
                raise DownloadError("not_found", f"{repo_id} not found on the Hub", 404)
            raise DownloadError("hub_error", f"could not reach the Hub: {e}", 502)
        if include:
            files = [f for f in details.files if any(fnmatch.fnmatch(f.name, p) for p in include)]
        else:
            files = [f for f in details.files if any(fnmatch.fnmatch(f.name, p) for p in DEFAULT_PATTERNS)
                     and not any(fnmatch.fnmatch(f.name, x) for x in EXCLUDE)]
            if not any(f.name.endswith(".safetensors") for f in files):
                if any(f.name.lower().endswith(".gguf") for f in details.files):
                    raise DownloadError("choose_file", "this repo contains GGUF files; choose which file to download", 422)
                raise DownloadError("no_weights", "no safetensors weights found in this repo", 422)
        if not files:
            raise DownloadError("no_files", "no files match the selection", 422)
        total = sum(f.size for f in files)
        free = self.disk_usage(str(self.hub_dir if self.hub_dir.exists() else self.hub_dir.parent)).free
        min_free = int(self.settings.min_free_disk_gib * GIB)
        if free - total < min_free:
            raise DownloadError("insufficient_disk",
                                f"needs {total / GIB:.1f} GiB; only {free / GIB:.1f} GiB free and "
                                f"{self.settings.min_free_disk_gib:.0f} GiB must stay free", 507)
        with self.db.session() as s:
            row = Download(repo_id=repo_id, revision=revision or details.sha, files=[f.name for f in files],
                           bytes_total=total, state="queued", created_by=user_id)
            s.add(row)
            s.flush()
            dl_id = row.id
        # Re-check AFTER the queued row exists (the deleter publishes markers, then re-reads downloads):
        # either it sees our row or we see its markers.
        if self.delete_guard is not None and self.delete_guard(repo_id):
            with self.db.session() as s:
                row = s.get(Download, dl_id)
                if row is not None:
                    s.delete(row)
            raise DownloadError("model_being_deleted", f"{repo_id} is being deleted from disk; try again shortly", 409)
        self._sizes[dl_id] = {f.name: f.size for f in files}
        return self.view(dl_id)

    def start_pinned(self, repo_id: str, revision: str | None = None, user_id: int | None = None) -> dict:
        """Queue the download of a decision model's pinned snapshot (no network here: the listing, disk check and
        hashes run in the download itself)."""
        from dgxmm.decider.image import DeciderImage
        from dgxmm.decider.snapshot import read_pins
        from dgxmm.engines.decision import DECIDER_MODELS, hash_list_name
        m = DECIDER_MODELS[repo_id]
        if revision is not None and revision != m.revision:
            raise DownloadError("pinned_revision", f"{repo_id} is only downloaded at its pinned revision "
                                                   f"{m.revision}", 422)
        if self.delete_guard is not None and self.delete_guard(repo_id):
            raise DownloadError("model_being_deleted", f"{repo_id} is being deleted from disk; try again shortly", 409)
        pins_path = DeciderImage(self.settings, lambda: None, events=None).build_dir / hash_list_name(repo_id,
                                                                                                    m.revision)
        try:
            pins = read_pins(pins_path)
        except (OSError, ValueError) as e:
            raise DownloadError("pins_unavailable", f"the pinned hash list of {repo_id} cannot be read ({e}); "
                                                    "reinstall DGXMM", 500)
        with self.db.session() as s:
            if s.scalar(select(Download).where(Download.repo_id == repo_id, Download.state.in_(("queued", "running")))):
                raise DownloadError("already_downloading", f"{repo_id} is already downloading", 409)
            row = Download(repo_id=repo_id, revision=m.revision, files=sorted(pins), bytes_total=0, state="queued",
                           created_by=user_id)
            s.add(row)
            s.flush()
            dl_id = row.id
        if self.delete_guard is not None and self.delete_guard(repo_id):  # same race as start()
            with self.db.session() as s:
                row = s.get(Download, dl_id)
                if row is not None:
                    s.delete(row)
            raise DownloadError("model_being_deleted", f"{repo_id} is being deleted from disk; try again shortly", 409)
        self._pinned[dl_id] = pins_path
        return self.view(dl_id)

    async def launch(self, dl_id: int) -> None:
        run = self._run_pinned if dl_id in self._pinned else self._run
        self._tasks[dl_id] = asyncio.get_running_loop().create_task(run(dl_id))
        await asyncio.sleep(0)

    def recover(self) -> int:
        """At startup: no download task survives a restart, so queued/running rows are stale."""
        with self.db.session() as s:
            rows = list(s.scalars(select(Download).where(Download.state.in_(("queued", "running")))))
            for d in rows:
                d.state, d.error = "failed", "interrupted by dgxmm restart; start the download again"
            n = len(rows)
        if n:
            log.info("marked %d interrupted download(s) failed", n)
        return n

    def cancel(self, dl_id: int) -> dict:
        with self.db.session() as s:
            if s.get(Download, dl_id) is None:
                raise DownloadError("not_found", f"download {dl_id} not found", 404)
        self._cancel.add(dl_id)
        v = self.view(dl_id)
        task = self._tasks.get(dl_id)
        live = task is not None and not task.done()
        if v["state"] == "queued" or (v["state"] == "running" and not live):
            self._update(dl_id, state="cancelled")
        job = self._jobs.get(dl_id)
        if job is not None:
            job.kill()  # stop the file in flight now; _run removes its partial file and marks the row
        stop = self._stops.get(dl_id)
        if stop is not None:
            stop.set()  # a snapshot download stops before its next chunk
        return self.view(dl_id)

    def close(self) -> None:
        """At shutdown: stop the snapshot downloads in flight (their worker threads end before the next chunk)."""
        for stop in list(self._stops.values()):
            stop.set()

    def _repo_dir(self, repo_id: str) -> Path:
        return self.hub_dir / ("models--" + repo_id.replace("/", "--"))

    def _write_marker(self, v: dict) -> None:
        """Record which files this download fetches; the inventory treats the model as incomplete until all of
        them are in its snapshot. Written atomically; a failure (a root-owned repo dir) only loses the check."""
        d = self._repo_dir(v["repo_id"])
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"{DOWNLOAD_MARKER}.{v['id']}.tmp"
            tmp.write_text(json.dumps({"download_id": v["id"], "revision": v["revision"], "files": v["files"]}))
            os.replace(tmp, d / DOWNLOAD_MARKER)
        except OSError as e:
            log.warning("cannot write the download marker for %s: %s", v["repo_id"], e)

    def _clear_marker(self, repo_id: str) -> None:
        try:
            (self._repo_dir(repo_id) / DOWNLOAD_MARKER).unlink(missing_ok=True)
        except OSError as e:
            log.warning("cannot remove the download marker for %s: %s", repo_id, e)

    def _remove_partials(self, repo_id: str) -> None:
        """Delete the partial files of a stopped download (one download per repo runs at a time)."""
        blobs = self._repo_dir(repo_id) / "blobs"
        try:
            for p in blobs.glob("*.incomplete"):
                if p.is_file() and not p.is_symlink():
                    p.unlink(missing_ok=True)
        except OSError as e:
            log.warning("cannot remove partial files of %s: %s", repo_id, e)

    async def _stopped(self, dl_id: int, repo_id: str, done: int) -> None:
        await asyncio.to_thread(self._remove_partials, repo_id)
        await asyncio.to_thread(self.inventory.refresh)
        self._update(dl_id, state="cancelled", bytes_done=done)
        self.events.publish("inventory", {"reason": "download_cancelled", "repo_id": repo_id})

    def _incomplete_bytes(self, repo_id: str) -> int:
        blobs = self.hub_dir / ("models--" + repo_id.replace("/", "--")) / "blobs"
        try:
            parts = list(blobs.glob("*.incomplete"))
            return max((p.stat().st_size for p in parts), default=0)
        except OSError:
            return 0

    def _permission_text(self, e: PermissionError, repo_id: str) -> str:
        path = getattr(e, "filename", "") or str(self.hub_dir)
        repo_dir = self._repo_dir(repo_id)
        return (f"permission denied writing {path}. The repo folder is probably owned by root (created by a "
                f"container). Fix on the host: sudo chown -R {os.getuid()}:{os.getgid()} "
                f"{repo_dir.as_posix().replace(self.settings.hf_cache.local, self.settings.hf_cache.host)}")

    def _snapshot_progress(self, repo_id: str, revision: str, plan: dict[str, int]) -> int:
        """Bytes of the planned files already in the snapshot, plus the file being downloaded."""
        snap = self._repo_dir(repo_id) / "snapshots" / revision
        done = 0
        for name, size in plan.items():
            p = snap / name
            if p.is_file():
                done += size
        return done + self._incomplete_bytes(repo_id)

    async def _run_pinned(self, dl_id: int) -> None:
        """A decision model: dgxmm/decider/snapshot.py in a daemon worker thread (a download never holds up
        shutdown), its plan and the files on disk polled for progress."""
        from dgxmm.decider import snapshot
        v = self.view(dl_id)
        pins_path = self._pinned.pop(dl_id)
        repo, revision = v["repo_id"], v["revision"]
        if v["state"] == "cancelled" or dl_id in self._cancel:
            self._update(dl_id, state="cancelled")
            return
        stop = self._stops[dl_id] = threading.Event()
        plan: dict[str, int] = {}
        lines: deque[str] = deque(maxlen=200)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def echo(line: str) -> None:
            lines.append(line)

        def on_plan(files: dict[str, int]) -> None:
            plan.clear()
            plan.update(files)

        def work() -> None:
            try:
                fetch = (self.snapshot_fetch(echo, stop.is_set) if self.snapshot_fetch is not None
                         else snapshot.HttpFetcher(echo=echo, should_stop=stop.is_set))
                ok = snapshot.ensure_snapshot(Path(self.settings.hf_cache.local), pins_path, fetch=fetch,
                                              list_files=self.snapshot_list or snapshot.hf_list_files, echo=echo,
                                              repo=repo, on_plan=on_plan,
                                              free_bytes=lambda p: self.disk_usage(str(p)).free)
                loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(ok))
            except BaseException as e:  # noqa: BLE001  (reported on the row)
                loop.call_soon_threadsafe(lambda err=e: fut.done() or fut.set_exception(err))

        self._update(dl_id, state="running")
        await asyncio.to_thread(self._write_marker, v)
        threading.Thread(target=work, name=f"dgxmm-download-{dl_id}", daemon=True).start()
        marked: set[str] = set(v["files"])
        try:
            while not fut.done():
                await asyncio.wait({fut}, timeout=0.5)
                if plan:
                    files = dict(plan)
                    if set(files) != marked:  # the manifest's files are known now: the marker lists them too
                        marked = set(files)
                        await asyncio.to_thread(self._write_marker, {**v, "files": sorted(files)})
                    done = await asyncio.to_thread(self._snapshot_progress, repo, revision, files)
                    self._update(dl_id, files=sorted(files), bytes_total=sum(files.values()),
                                 bytes_done=min(done, sum(files.values())))
            ok = fut.result()
        except snapshot.DownloadCancelled:
            await self._stopped(dl_id, repo, self.view(dl_id)["bytes_done"])
            return
        except snapshot.DownloadFailed as e:
            self._update(dl_id, state="failed", error=f"{e}. Start the download again: it resumes where it stopped.")
            return
        except PermissionError as e:
            self._update(dl_id, state="failed", error=self._permission_text(e, repo))
            return
        except Exception as e:
            self._update(dl_id, state="failed", error=f"{type(e).__name__}: {e}")
            return
        finally:
            self._stops.pop(dl_id, None)
        if dl_id in self._cancel:
            await self._stopped(dl_id, repo, self.view(dl_id)["bytes_done"])
            return
        if not ok:
            problems = [ln.strip().removeprefix("problem: ") for ln in lines if ln.strip().startswith("problem:")]
            self._update(dl_id, state="failed",
                         error=("; ".join(problems) or "the snapshot is not complete and verified")[:2000]
                         + " (nothing unverified was linked)")
            return
        total = sum(plan.values())
        await asyncio.to_thread(self._clear_marker, repo)
        await asyncio.to_thread(self.inventory.refresh)
        self._update(dl_id, state="done", files=sorted(plan) or v["files"], bytes_total=total, bytes_done=total)
        self.events.publish("inventory", {"reason": "download", "repo_id": repo})

    async def _run(self, dl_id: int) -> None:
        v = self.view(dl_id)
        if v["state"] == "cancelled" or dl_id in self._cancel:
            self._update(dl_id, state="cancelled")
            return
        sizes = self._sizes.get(dl_id, {})
        self._update(dl_id, state="running")
        await asyncio.to_thread(self._write_marker, v)
        done = 0
        for name in v["files"]:
            if dl_id in self._cancel:
                await self._stopped(dl_id, v["repo_id"], done)
                return
            try:
                job = await asyncio.to_thread(_spawn, self.downloader, repo_id=v["repo_id"], filename=name,
                                              revision=v["revision"], cache_dir=str(self.hub_dir), token=self.token)
            except Exception as e:
                self._update(dl_id, state="failed", bytes_done=done, error=f"{type(e).__name__}: {e}")
                return
            self._jobs[dl_id] = job
            if dl_id in self._cancel:  # cancelled while the child was starting
                job.kill()
            fut = asyncio.ensure_future(asyncio.to_thread(job.wait))
            try:
                while not fut.done():
                    await asyncio.sleep(0.25)
                    if not fut.done():
                        self._update(dl_id, bytes_done=done + self._incomplete_bytes(v["repo_id"]))
            finally:
                self._jobs.pop(dl_id, None)
            try:
                fut.result()
            except DownloadCancelled:
                await self._stopped(dl_id, v["repo_id"], done)
                return
            except PermissionError as e:
                self._update(dl_id, state="failed", bytes_done=done, error=self._permission_text(e, v["repo_id"]))
                return
            except _ChildFailed as e:
                self._update(dl_id, state="failed", bytes_done=done, error=f"{e.type_name}: {e.message}")
                return
            except Exception as e:
                self._update(dl_id, state="failed", bytes_done=done, error=f"{type(e).__name__}: {e}")
                return
            done += sizes.get(name, 0)
            self._update(dl_id, bytes_done=done)
        if dl_id in self._cancel:  # cancelled while the last file finished: still not a completed download
            await self._stopped(dl_id, v["repo_id"], done)
            return
        # Every file is on disk: the model may now be reported complete
        await asyncio.to_thread(self._clear_marker, v["repo_id"])
        # Refresh inventory BEFORE marking done, so tests see the new model
        await asyncio.to_thread(self.inventory.refresh)
        self._update(dl_id, state="done", bytes_done=done)
        self.events.publish("inventory", {"reason": "download", "repo_id": v["repo_id"]})
