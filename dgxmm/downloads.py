"""Background Hugging Face downloads with progress (spec §7)."""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import shutil
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from dgxmm.db import Database, Download
from dgxmm.inventory.artifact import REPO_RE

log = logging.getLogger(__name__)
GIB = 1024**3
DEFAULT_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "*.tiktoken", "tokenizer*", "*.md"]
EXCLUDE = ["original/*", "*.pth", "*.pt", "*.bin", "*.onnx", "*.msgpack", "*.h5", "*.gguf"]
PATTERN_RE = r"^[A-Za-z0-9._*?\[\]/-]{1,200}$"


class DownloadError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _default_downloader():
    from huggingface_hub import hf_hub_download
    return hf_hub_download


class DownloadService:
    def __init__(self, settings, db: Database, inventory, hub, events, downloader: Callable | None = None,
                 disk_usage: Callable = shutil.disk_usage, token: str | None = None):
        self.settings, self.db, self.inventory, self.hub, self.events = settings, db, inventory, hub, events
        self.downloader = downloader or _default_downloader()
        self.disk_usage, self.token = disk_usage, token
        self.hub_dir = Path(settings.hf_cache.local) / "hub"
        self._cancel: set[int] = set()
        self._tasks: dict[int, asyncio.Task] = {}
        self._sizes: dict[int, dict[str, int]] = {}

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
        if not re.fullmatch(REPO_RE, repo_id) or ".." in repo_id:
            raise DownloadError("invalid_repo", "repo id must look like org/name", 422)
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
        self._sizes[dl_id] = {f.name: f.size for f in files}
        return self.view(dl_id)

    async def launch(self, dl_id: int) -> None:
        self._tasks[dl_id] = asyncio.get_running_loop().create_task(self._run(dl_id))
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
        return self.view(dl_id)

    def _incomplete_bytes(self, repo_id: str) -> int:
        blobs = self.hub_dir / ("models--" + repo_id.replace("/", "--")) / "blobs"
        try:
            parts = list(blobs.glob("*.incomplete"))
            return max((p.stat().st_size for p in parts), default=0)
        except OSError:
            return 0

    async def _run(self, dl_id: int) -> None:
        v = self.view(dl_id)
        if v["state"] == "cancelled" or dl_id in self._cancel:
            self._update(dl_id, state="cancelled")
            return
        sizes = self._sizes.get(dl_id, {})
        self._update(dl_id, state="running")
        done = 0
        for name in v["files"]:
            if dl_id in self._cancel:
                self._update(dl_id, state="cancelled", bytes_done=done)
                return
            fut = asyncio.ensure_future(asyncio.to_thread(self.downloader, repo_id=v["repo_id"], filename=name,
                                                          revision=v["revision"], cache_dir=str(self.hub_dir),
                                                          token=self.token))
            while not fut.done():
                await asyncio.sleep(0.25)
                if not fut.done():
                    self._update(dl_id, bytes_done=done + self._incomplete_bytes(v["repo_id"]))
            try:
                fut.result()
            except PermissionError as e:
                path = getattr(e, "filename", "") or str(self.hub_dir)
                repo_dir = self.hub_dir / ("models--" + v["repo_id"].replace("/", "--"))
                self._update(dl_id, state="failed", bytes_done=done,
                             error=(f"permission denied writing {path}. The repo folder is probably owned by root "
                                    f"(created by a container). Fix on the DGX: sudo chown -R 1000:1000 "
                                    f"{repo_dir.as_posix().replace(self.settings.hf_cache.local, self.settings.hf_cache.host)}"))
                return
            except Exception as e:
                self._update(dl_id, state="failed", bytes_done=done, error=f"{type(e).__name__}: {e}")
                return
            done += sizes.get(name, 0)
            self._update(dl_id, bytes_done=done)
        # Refresh inventory BEFORE marking done, so tests see the new model
        await asyncio.to_thread(self.inventory.refresh)
        self._update(dl_id, state="done", bytes_done=done)
        self.events.publish("inventory", {"reason": "download", "repo_id": v["repo_id"]})
