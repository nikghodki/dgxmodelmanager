"""Which architectures the pinned vLLM image can serve (spec §7)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
SCRIPT = ("import json; from vllm import ModelRegistry; "
          "print('ARCHS=' + json.dumps(sorted(ModelRegistry.get_supported_archs())))")


class VllmArchRegistry:
    def __init__(self, settings, docker):
        self.image = settings.images.vllm
        self.docker = docker
        self.path = Path(settings.data_dir) / f"vllm_archs_{re.sub(r'[^A-Za-z0-9]+', '_', self.image)}.json"
        self._archs: set[str] | None = None
        self._lock = asyncio.Lock()

    def get(self) -> set[str] | None:
        if self._archs is None and self.path.is_file():
            try:
                self._archs = set(json.loads(self.path.read_text()))
            except (OSError, ValueError):
                self._archs = None
        return self._archs

    async def ensure(self) -> set[str] | None:
        if self.get() is not None:
            return self._archs
        async with self._lock:
            if self.get() is not None:
                return self._archs
            try:
                out = await asyncio.to_thread(self.docker.run_oneshot, self.image, ["python3", "-c", SCRIPT], [])
                line = next(l for l in out.splitlines() if l.startswith("ARCHS="))
                archs = json.loads(line[len("ARCHS="):])
            except Exception as e:
                log.warning("could not read vLLM model registry from %s: %s", self.image, e)
                return None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(archs))
            self._archs = set(archs)
            return self._archs


def engine_support(architecture: str | None, is_gguf: bool, archs: set[str] | None) -> dict:
    if is_gguf:
        return {"vllm": False, "llamacpp": True}
    vllm = None if archs is None or not architecture else architecture in archs
    return {"vllm": vllm, "llamacpp": False}
