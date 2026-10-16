"""Runtime settings stored in the ``app_settings`` table, with an in-memory cache.

Values here take precedence over config.yaml / DGXMM_* env (``Settings``). Readers call the typed
accessors at use time, so a change made through the API applies without a restart.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from dgxmm.config import Settings
from dgxmm.db import AppSetting, Database

_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}

LITELLM_ENABLED = "litellm.enabled"
LITELLM_URL = "litellm.url"
LITELLM_PUBLIC_URL = "litellm.public_url"
LITELLM_MASTER_KEY = "litellm.master_key"  # write-only through the API; never returned or logged


@dataclass(frozen=True)
class LiteLLMConfig:
    enabled: bool
    url: str
    public_url: str
    master_key: str | None = field(repr=False)
    master_key_source: str  # settings | secret_file | none

    @property
    def endpoint(self) -> str:
        """The single OpenAI-compatible endpoint clients use."""
        return (self.public_url or self.url).rstrip("/") + "/v1"

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.master_key)


class SettingsStore:
    def __init__(self, db: Database, settings: Settings):
        self.db, self.settings = db, settings
        self._cache: dict[str, str] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        with self._lock:
            if self._cache is None:
                with self.db.session() as s:
                    self._cache = {r.key: r.value for r in s.query(AppSetting)}
            return self._cache

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._load().get(key, default)

    def set(self, key: str, value: str | None) -> None:
        """Store ``value``; ``None`` deletes the key (falls back to config)."""
        cache = self._load()
        with self._lock:
            with self.db.session() as s:
                row = s.get(AppSetting, key)
                if value is None:
                    if row is not None:
                        s.delete(row)
                elif row is None:
                    s.add(AppSetting(key=key, value=value))
                else:
                    row.value = value
            if value is None:
                cache.pop(key, None)
            else:
                cache[key] = value

    def get_bool(self, key: str, default: bool) -> bool:
        raw = (self.get(key) or "").strip().lower()
        return True if raw in _TRUE else False if raw in _FALSE else default

    def set_bool(self, key: str, value: bool | None) -> None:
        self.set(key, None if value is None else ("true" if value else "false"))

    # ---------- typed accessors ----------
    def litellm(self) -> LiteLLMConfig:
        key, source = self.get(LITELLM_MASTER_KEY), "settings"
        if not key:
            key, source = self.settings.read_secret(self.settings.litellm_key_file), "secret_file"
        if not key:
            key, source = None, "none"
        return LiteLLMConfig(
            enabled=self.get_bool(LITELLM_ENABLED, bool(key)),
            url=(self.get(LITELLM_URL) or self.settings.litellm_url).rstrip("/"),
            public_url=(self.get(LITELLM_PUBLIC_URL) or "").rstrip("/"),
            master_key=key, master_key_source=source)
