"""The decider's bearer key: generated once (mode 600), read at use time, never logged or returned by the API."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from dgxmm.config import Settings


def read_key(settings: Settings) -> str | None:
    return settings.read_secret(str(settings.decider_key_path))


def _write_tmp(path: Path, key: str) -> Path:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key.encode())
    finally:
        os.close(fd)
    return tmp


def ensure_key(settings: Settings) -> str:
    """The existing key, or a new random one written atomically with mode 600 (directory 700)."""
    existing = read_key(settings)
    if existing:
        return existing
    path: Path = settings.decider_key_path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = _write_tmp(path, secrets.token_urlsafe(32))
    if path.exists():  # empty file: replace it
        os.replace(tmp, path)
    else:
        try:
            os.link(tmp, path)  # never overwrite a key another process created meanwhile
        except FileExistsError:
            pass
        finally:
            tmp.unlink(missing_ok=True)
    return read_key(settings) or ""


def rotate_key(settings: Settings) -> None:
    """Replace the key with a new random one (atomic rename, mode 600). The old key stops working as soon as the
    decider is restarted with the new one (dgxmm.decider.rotation)."""
    path: Path = settings.decider_key_path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = _write_tmp(path, secrets.token_urlsafe(32))
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
