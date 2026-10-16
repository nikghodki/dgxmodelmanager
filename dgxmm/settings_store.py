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
LITELLM_MASTER_KEY_URL = "litellm.master_key_url"  # the LiteLLM URL the stored key was entered for
REENTER_HINT = "the master key was entered for a different LiteLLM URL; re-enter it for this URL"
FILE_KEY_HINT = ("the master key from the secret file is only used with the default LiteLLM URL ({url}); "
                 "re-enter the master key for this URL")


def norm_url(url: str) -> str:
    return url.strip().rstrip("/")


@dataclass(frozen=True)
class LiteLLMConfig:
    enabled: bool
    url: str
    public_url: str
    master_key: str | None = field(repr=False)
    master_key_source: str  # settings | secret_file | none
    master_key_hint: str | None = None  # why no key is in effect although one exists (URL binding)

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
    def set_master_key(self, key: str | None, url: str | None = None) -> None:
        """Store (or with None clear) the LiteLLM master key, bound to the URL it was entered for."""
        if key:
            self.set(LITELLM_MASTER_KEY_URL, norm_url(url or ""))
            self.set(LITELLM_MASTER_KEY, key)
        else:
            self.set(LITELLM_MASTER_KEY, None)
            self.set(LITELLM_MASTER_KEY_URL, None)

    def litellm(self) -> LiteLLMConfig:
        """Resolve the LiteLLM settings. A key is only ever used with the URL it belongs to: a stored key with
        the URL it was entered for, the secret-file key with the configured default ``litellm_url``. So
        repointing the URL can never send an existing (write-only) key to another server."""
        url = norm_url(self.get(LITELLM_URL) or self.settings.litellm_url)
        default_url = norm_url(self.settings.litellm_url)
        key, source, hint = None, "none", None
        stored = self.get(LITELLM_MASTER_KEY)
        if stored:
            if norm_url(self.get(LITELLM_MASTER_KEY_URL) or "") == url:
                key, source = stored, "settings"
            else:
                hint = REENTER_HINT
        if key is None:
            file_key = self.settings.read_secret(self.settings.litellm_key_file)
            if file_key and url == default_url:
                key, source, hint = file_key, "secret_file", None
            elif file_key and hint is None:
                hint = FILE_KEY_HINT.format(url=default_url)
        return LiteLLMConfig(
            enabled=self.get_bool(LITELLM_ENABLED, bool(key)),
            url=url,
            public_url=norm_url(self.get(LITELLM_PUBLIC_URL) or ""),
            master_key=key, master_key_source=source, master_key_hint=hint)
