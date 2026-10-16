from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable

from dgxmm.config import Settings
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.inventory.hf_scan import scan_hf_cache

log = logging.getLogger(__name__)


class InventoryService:
    def __init__(self, settings: Settings, scanners: list[Callable[[], list[ModelArtifact]]] | None = None):
        self.settings = settings
        # When HF_HOME is set (as in Dockerfile), hf_hub_download writes to HF_HOME/models--...
        # For backward compat with older downloads, also scan HF_HOME/hub/
        hf_base = Path(settings.hf_cache.local)
        hub_legacy = hf_base / "hub"
        from dgxmm.inventory.gguf_scan import scan_gguf_roots
        gguf_roots = [Path(m.local) for m in settings.gguf_roots] + [hub_legacy]
        self.scanners = scanners or [
            lambda: scan_hf_cache(hf_base),  # Scan base for new downloads (HF_HOME behavior)
            lambda: scan_hf_cache(hub_legacy) if hub_legacy.exists() else [],  # Scan hub/ for old downloads
            lambda: scan_gguf_roots(gguf_roots)
        ]
        self._items: dict[str, ModelArtifact] = {}
        self._lock = threading.Lock()

    def refresh(self) -> list[ModelArtifact]:
        found: dict[str, ModelArtifact] = {}
        for scan in self.scanners:
            try:
                for a in scan():
                    found[a.ref] = a
            except Exception:
                log.exception("inventory scanner failed")
        with self._lock:
            self._items = found
        return list(found.values())

    def list(self) -> list[ModelArtifact]:
        with self._lock:
            return sorted(self._items.values(), key=lambda a: a.ref.lower())

    def get(self, ref: str) -> ModelArtifact | None:
        with self._lock:
            return self._items.get(ref)

    async def run(self) -> None:
        while True:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(self.settings.inventory_interval_s)
