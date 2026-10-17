"""Rotation of the decider's bearer key: new key file, then a safe restart of a running decider so its new container
gets the new key (as its tmpfs secret file). Used by POST /api/decider/rotate-key and, through a request file the
app's background watcher picks up, by ``python -m dgxmm.admin rotate-decider-key`` (a separate process cannot drive
the running app's deployer itself).

Order: stop the decider (its old key dies with it), write the new key, restart. While the decider is down every
feature falls back to rules (fail open). If the restart fails, the key is rotated anyway and the decider stays
stopped/failed; the result says so and enabling it again starts it with the new key.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path

from dgxmm.audit import audit
from dgxmm.decider.common import DECIDER_ROTATED, decider_view
from dgxmm.decider.secret import rotate_key

log = logging.getLogger(__name__)
# states the rotation stops and starts again with the new key: a queued launch would otherwise start a container with
# the old key mid-rotation, and a stopping one would never get the new key
ACTIVE = DECIDER_ROTATED
WATCH_INTERVAL_S = 2.0
IN_PROGRESS_FRESH_S = 15 * 60  # an "in_progress" marker older than this is stale (the app died mid-rotation)


class RotationInProgress(Exception):
    pass


def request_path(settings) -> Path:
    return Path(settings.decider_key_path).parent / "decider_rotate.request"


def result_path(settings) -> Path:
    return Path(settings.decider_key_path).parent / "decider_rotate.result"


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _lock_of(ctx) -> asyncio.Lock:
    lock = getattr(ctx, "_decider_rotation_lock", None)
    if lock is None:
        lock = ctx._decider_rotation_lock = asyncio.Lock()
    return lock


def in_progress(ctx) -> bool:
    """A rotation is running (enable / start of the decider must wait for it)."""
    lock = getattr(ctx, "_decider_rotation_lock", None)
    return lock is not None and lock.locked()


async def rotate_decider_key(ctx, user_id: int | None, via: str = "api") -> dict:
    """Rotate the key and restart a running decider. Never returns or logs the key.

    The decider deployment is marked as rotating (``deployer.rotating``) from before the stop until after the
    restart, so a Start / enable in between is refused (409 rotation_in_progress) instead of racing the rotation
    with a container that would get the old key."""
    lock = _lock_of(ctx)
    if lock.locked():
        raise RotationInProgress()
    async with lock:
        view = decider_view(ctx)
        running = view is not None and view["state"] in ACTIVE
        out = {"rotated": False, "restarted": False, "deployment_id": view["id"] if view else None,
               "state": view["state"] if view else None, "error": None}
        rotating = getattr(ctx.deployer, "rotating", None) if view is not None else None
        if rotating is not None:
            rotating.add(view["id"])
        try:
            if running:
                out["state"] = (await ctx.deployer.stop(view["id"], user_id, reason="decider key rotation"))["state"]
            await asyncio.to_thread(rotate_key, ctx.settings)
            out["rotated"] = True
            reset = getattr(ctx.decider, "reset_client", None)
            if reset is not None:
                reset()  # drop the cached client (its breaker may have tripped on the old key)
            if running:
                restarted = await ctx.deployer.restart(view["id"], user_id, allow_rotating=True)
                out["restarted"], out["state"] = True, restarted["state"]
        except Exception as e:  # DeployError / InsufficientMemory / docker trouble: report, never raise
            out["error"] = getattr(e, "code", None) or type(e).__name__
            log.warning("decider key rotation: %s", out["error"])
        finally:
            if rotating is not None:
                rotating.discard(view["id"])
        audit(ctx.db, user_id, "decider_rotate_key", view["name"] if view else "decider",
              {"via": via, **{k: out[k] for k in ("rotated", "restarted", "deployment_id", "error")}})
        return out


# ---------- CLI <-> app handshake (files next to the key, mode 600) ----------
def request_rotation(settings) -> str:
    """Queue a rotation for the app; returns its nonce. Re-running while a request is pending or a rotation is in
    progress returns that one's nonce instead of queueing a second rotation."""
    try:
        pending = request_path(settings).read_text().strip()[:64]
    except FileNotFoundError:
        pending = ""
    if pending:
        return pending
    current = _read(settings)
    if (current and current.get("status") == "in_progress"
            and time.time() - float(current.get("started_at") or 0) < IN_PROGRESS_FRESH_S):
        return str(current.get("nonce"))
    nonce = secrets.token_hex(8)
    result_path(settings).unlink(missing_ok=True)
    _write_private(request_path(settings), nonce + "\n")
    return nonce


def _read(settings) -> dict | None:
    try:
        data = json.loads(result_path(settings).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_result(settings, nonce: str) -> dict | None:
    """The result for `nonce`: status "in_progress" (picked up, rotating) or "done" (with ok/error/...)."""
    data = _read(settings)
    return data if data is not None and data.get("nonce") == nonce else None


def wait_result(settings, nonce: str, timeout: float) -> dict | None:
    """Wait for the "done" result; on timeout the last state seen ("in_progress" result) or None (not picked up)."""
    deadline = time.monotonic() + timeout
    last = None
    while True:
        res = read_result(settings, nonce)
        if res is not None:
            last = res
            if res.get("status") == "done":
                return res
        if time.monotonic() >= deadline:
            return last
        time.sleep(0.5)


async def process_pending(ctx) -> dict | None:
    """Run a rotation the CLI requested (if any): an "in_progress" marker first, the "done" result at the end."""
    req = request_path(ctx.settings)
    try:
        nonce = req.read_text().strip()[:64]
    except FileNotFoundError:
        return None
    await asyncio.to_thread(_write_private, result_path(ctx.settings),
                            json.dumps({"nonce": nonce, "status": "in_progress", "started_at": time.time()}))
    req.unlink(missing_ok=True)
    try:
        out = await rotate_decider_key(ctx, None, via="cli")
    except RotationInProgress:
        out = {"rotated": False, "restarted": False, "error": "rotation_in_progress"}
    res = {"nonce": nonce, "status": "done", "ok": bool(out.get("rotated")) and not out.get("error"), **out}
    await asyncio.to_thread(_write_private, result_path(ctx.settings), json.dumps(res))
    return res


async def watch(ctx, interval_s: float = WATCH_INTERVAL_S) -> None:
    """Background loop: pick up CLI rotation requests."""
    while True:
        try:
            await process_pending(ctx)
        except Exception:
            log.warning("decider key rotation request failed", exc_info=True)
        await asyncio.sleep(interval_s)
