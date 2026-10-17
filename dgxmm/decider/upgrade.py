"""Does the running decider container match the current image and hardening? A decider started by an older build
(e.g. the POC: env key, published port, root, shared network) keeps running after DGXMM is upgraded until it is
restarted; /api/decider/status reports what is out of date so the admin can rebuild the image and restart it."""
from __future__ import annotations

import logging

from dgxmm.engines.decision import KEY_FILE

log = logging.getLogger(__name__)
KEY_ENV = "DECISION_API_KEY="


def container_issues(attrs: dict, settings) -> list[str]:
    """Human-readable reasons the container (docker inspect attrs) needs a rebuild/restart. Never includes values
    of environment variables."""
    cfg, hc = attrs.get("Config") or {}, attrs.get("HostConfig") or {}
    net = attrs.get("NetworkSettings") or {}
    out: list[str] = []
    image = cfg.get("Image") or ""
    if image != settings.images.decision:
        out.append(f"runs image {image or '?'}, expected {settings.images.decision}")
    env = cfg.get("Env") or []
    if any(e.startswith(KEY_ENV) for e in env):
        out.append("its API key is in the container environment (visible in docker inspect)")
    if not any(e == f"DECISION_API_KEY_FILE={KEY_FILE}" for e in env) and not any(e.startswith(KEY_ENV) for e in env):
        out.append("it does not read its API key from the key file")
    user = (cfg.get("User") or "").split(":")[0]
    if user in ("", "0", "root"):
        out.append("runs as root")
    published = sorted({b.get("HostPort") for bs in (net.get("Ports") or {}).values() for b in (bs or [])
                        if b.get("HostPort")})
    if published:
        out.append(f"publishes host port(s) {', '.join(published)}")
    if settings.decider_network and settings.decider_network not in (net.get("Networks") or {}):
        out.append(f"is not on the internal network {settings.decider_network}")
    if not hc.get("ReadonlyRootfs"):
        out.append("its root filesystem is writable")
    return out


def check(ctx) -> dict:
    """{needed, reasons} for the current decider container (blocking: call in a thread)."""
    none = {"needed": False, "reasons": []}
    view = ctx.decider.deployment() if ctx.decider is not None else None
    if view is None or not view.get("container_id") or ctx.docker is None:
        return none
    try:
        c = ctx.docker.client.containers.get(view["container_id"])
        c.reload()
        reasons = container_issues(c.attrs or {}, ctx.settings)
    except Exception:  # container gone / docker trouble: nothing to say about its hardening
        log.debug("decider upgrade check failed", exc_info=True)
        return none
    return {"needed": bool(reasons), "reasons": reasons}
