"""Bring the DB in line with Docker after a dgxmm restart (spec §11)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.db import Deployment, DeploymentEvent

log = logging.getLogger(__name__)


def _flag(cmd: list[str], flag: str) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


async def reconcile(deployer) -> list[str]:
    notes: list[str] = []
    containers = {c.id: c for c in await asyncio.to_thread(deployer.docker.list_managed)
                  if not c.labels.get("dgxmm.oneshot")}
    with deployer.db.session() as s:
        rows = list(s.scalars(select(Deployment)))
    known = {r.container_id for r in rows if r.container_id}
    for d in rows:
        if d.state in ("stopped", "failed", "planned"):
            continue
        c = containers.get(d.container_id) if d.container_id else None
        if c is None:
            deployer._transition(d.id, "failed", message="container missing after dgxmm restart",
                                 last_error="container missing after dgxmm restart", container_id=None)
            notes.append(f"{d.name}: container missing → failed")
        elif c.status in ("exited", "dead"):
            await deployer._fail_exited(d, c.exit_code)
            notes.append(f"{d.name}: container exited ({c.exit_code}) → failed")
        elif d.state in ("starting", "loading"):
            deployer.resume_monitor(d.id)
            notes.append(f"{d.name}: still {d.state} → monitoring resumed")
        elif d.state == "stopping":
            await deployer.stop(d.id, reason="reconcile")
            notes.append(f"{d.name}: finishing interrupted stop")
    taken = {r.name for r in rows}
    for cid, c in containers.items():
        if cid in known:
            continue
        name = c.name.removeprefix("dgxmm-") or "orphan"
        base, n = name, 2
        while name in taken:
            name, n = f"{base}-{n}", n + 1
        taken.add(name)
        port = _flag(c.command, "--port")
        port_i = int(port) if port and port.isdigit() else (min(c.host_ports) if c.host_ports else None)
        with deployer.db.session() as s:
            row = Deployment(name=name, engine=c.labels.get("dgxmm.engine", "vllm"),
                             model_ref=_flag(c.command, "--model") or _flag(c.command, "-m") or "unknown",
                             config={}, state="unknown", container_id=cid, port=port_i, reserved_bytes=0,
                             estimate={}, last_error="found running without a database record")
            s.add(row)
            s.flush()
            s.add(DeploymentEvent(deployment_id=row.id, from_state="unknown", to_state="unknown",
                                  message=f"adopted orphan container {c.name}"))
        notes.append(f"{c.name}: orphan managed container → recorded as unknown")
    for n in notes:
        log.info("reconcile: %s", n)
    return notes
