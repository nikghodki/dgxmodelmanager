"""The decider image (``dgxmm-decider:<version>``): presence check and an on-demand local build via the docker SDK
(low-level ``api.build`` streaming), progress published on the SSE topic ``decider`` as ``{phase, message}``."""
from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import time
from pathlib import Path
from typing import Any, Callable

from dgxmm.db import utcnow
from dgxmm.engines.decision import DECIDER_MODELS, DECIDER_REPO, hash_list_name

log = logging.getLogger(__name__)
# each supported model's pinned snapshot code/config hashes (deploy/decider/<name>-<revision>.sha256)
HASH_LISTS = [hash_list_name(m.repo, m.revision) for m in DECIDER_MODELS.values()]
HASH_LIST = HASH_LISTS[0]
# The whole build context: the hash-pinned pip lock and the launcher are part of the image's supply chain.
CONTEXT_FILES = {"Dockerfile": 0o644, "entrypoint.sh": 0o755, "launch.py": 0o644, "requirements.lock": 0o644,
                 **{h: 0o644 for h in HASH_LISTS}}
MAX_LINE = 300


def default_build_dir() -> Path:
    """deploy/decider next to the dgxmm package (repo checkout, or /app in the image), else under the cwd."""
    for base in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (base / "deploy" / "decider" / "Dockerfile").is_file():
            return base / "deploy" / "decider"
    return Path(__file__).resolve().parents[2] / "deploy" / "decider"


class BuildFailed(Exception):
    pass


class DeciderImage:
    def __init__(self, settings, docker: Callable[[], Any], events, clock: Callable[[], float] = time.monotonic):
        self.settings, self._docker, self.events, self.clock = settings, docker, events, clock
        self.building = False
        self.error: str | None = None
        self.last_build: dict | None = None
        self._task: asyncio.Task | None = None

    @property
    def tag(self) -> str:
        return self.settings.images.decision

    @property
    def build_dir(self) -> Path:
        return Path(self.settings.decider_build_dir) if self.settings.decider_build_dir else default_build_dir()

    def present(self) -> bool | None:
        docker = self._docker()
        if docker is None:
            return None
        try:
            return docker.image_present(self.tag)
        except Exception:
            log.warning("could not check the decider image", exc_info=True)
            return None

    def supported_models(self) -> set[str] | None:
        """"repo@revision" of every model the built image can serve (None: no image / docker unavailable). An image
        built before models were configurable only carries d3-mini's revision label."""
        docker = self._docker()
        if docker is None:
            return None
        try:
            labels = docker.image_labels(self.tag)
        except Exception:
            log.warning("could not read the decider image labels", exc_info=True)
            return None
        if labels is None:
            return None
        if labels.get("dgxmm.decider.models"):
            return {x.strip() for x in labels["dgxmm.decider.models"].split(",") if x.strip()}
        rev = labels.get("dgxmm.decider.revision")
        return {f"{DECIDER_REPO}@{rev}"} if rev else set()

    def supports(self, repo: str) -> bool | None:
        models = self.supported_models()
        m = DECIDER_MODELS.get(repo)
        return None if models is None or m is None else f"{repo}@{m.revision}" in models

    def status(self) -> dict:
        return {"image": self.tag, "present": self.present(), "building": self.building, "error": self.error,
                "last_build": self.last_build}

    def context_tar(self) -> io.BytesIO:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name, mode in CONTEXT_FILES.items():
                data = (self.build_dir / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), mode, 0
                tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        return buf

    def start_build(self) -> bool:
        """Start a background build; False if one is already running."""
        if self.building:
            return False
        self.building, self.error = True, None
        self._task = asyncio.get_running_loop().create_task(self._run())
        return True

    def _publish(self, loop: asyncio.AbstractEventLoop, phase: str, message: str) -> None:
        loop.call_soon_threadsafe(self.events.publish, "decider", {"phase": phase, "message": message[:MAX_LINE]})

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        started, t0 = utcnow(), self.clock()
        self.events.publish("decider", {"phase": "started", "message": f"building {self.tag}"})
        ok = False
        try:
            await asyncio.to_thread(self._build_sync, loop)
            ok = True
            self.events.publish("decider", {"phase": "done", "message": f"built {self.tag}"})
        except Exception as e:
            self.error = str(e)[:MAX_LINE] or type(e).__name__
            log.warning("decider image build failed: %s", self.error)
            self.events.publish("decider", {"phase": "failed", "message": self.error})
        finally:
            self.building = False
            self.last_build = {"ok": ok, "started_at": started.isoformat(), "finished_at": utcnow().isoformat(),
                               "duration_s": round(self.clock() - t0, 1)}

    def _build_sync(self, loop: asyncio.AbstractEventLoop) -> None:
        self.build_blocking(lambda line: self._publish(loop, "progress", line))

    def build_blocking(self, on_line: Callable[[str], None]) -> None:
        """Build in the calling thread, reporting each progress line (also used by the admin CLI)."""
        docker = self._docker()
        if docker is None:
            raise BuildFailed("docker is not available")
        install = getattr(self.settings, "install_id", None)
        labels = {"dgxmm.install": install} if install else None
        for chunk in docker.build_image(self.context_tar(), self.tag, labels=labels):
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error") or chunk.get("errorDetail"):
                raise BuildFailed(chunk.get("error") or str(chunk.get("errorDetail")))
            line = (chunk.get("stream") or chunk.get("status") or "").strip()
            if line:
                on_line(line[:MAX_LINE])
        if not docker.image_present(self.tag):
            raise BuildFailed(f"build finished but {self.tag} is missing")
