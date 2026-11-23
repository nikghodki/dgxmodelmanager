"""Which architectures the vLLM images DGXMM may run can serve (spec §7)."""
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
    """The architectures each vLLM image a deployment may run can serve: the default (images.vllm) and every
    images.vllm_allowed entry. Read once per image with a one-shot container on the local node (pulled there if
    missing, as for the default image) and cached in data_dir by image. ``get()`` is the default image's."""

    def __init__(self, settings, docker):
        self.image = settings.images.vllm
        self.images = list(dict.fromkeys([self.image, *settings.images.allowed("vllm")]))
        self.docker = docker
        self.data_dir = Path(settings.data_dir)
        self._archs: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    def path_for(self, image: str) -> Path:
        return self.data_dir / f"vllm_archs_{re.sub(r'[^A-Za-z0-9]+', '_', image)}.json"

    @property
    def path(self) -> Path:
        return self.path_for(self.image)

    def get_for(self, image: str) -> set[str] | None:
        """``image``'s architectures, or None (not read yet, or not an image DGXMM may run)."""
        if image not in self.images:
            return None
        if image not in self._archs and self.path_for(image).is_file():
            try:
                self._archs[image] = set(json.loads(self.path_for(image).read_text()))
            except (OSError, ValueError):
                return None
        return self._archs.get(image)

    def get(self) -> set[str] | None:
        return self.get_for(self.image)

    def get_any(self) -> set[str] | None:
        """What some image a deployment may run can serve: the union of the known lists; None while the default
        image's list is unknown (nothing is refused then, as before the registry was read)."""
        default = self.get()
        if default is None:
            return None
        out = set(default)
        for image in self.images[1:]:
            out |= self.get_for(image) or set()
        return out

    async def _read(self, image: str) -> set[str] | None:
        try:
            out = await asyncio.to_thread(self.docker.run_oneshot, image, ["python3", "-c", SCRIPT], [])
            line = next(l for l in out.splitlines() if l.startswith("ARCHS="))
            archs = json.loads(line[len("ARCHS="):])
        except Exception as e:
            log.warning("could not read vLLM model registry from %s: %s", image, e)
            return None
        path = self.path_for(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(archs))
        self._archs[image] = set(archs)
        return self._archs[image]

    async def ensure(self) -> set[str] | None:
        """Read every image's list that is not known yet; the default image's list (None if unreadable)."""
        if all(self.get_for(i) is not None for i in self.images):
            return self.get()
        async with self._lock:
            for image in self.images:
                if self.get_for(image) is None:
                    await self._read(image)
            return self.get()


def engine_support(architecture: str | None, is_gguf: bool, archs: set[str] | None) -> dict:
    if is_gguf:
        return {"vllm": False, "llamacpp": True}
    vllm = None if archs is None or not architecture else architecture in archs
    return {"vllm": vllm, "llamacpp": False}
