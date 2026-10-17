"""Bring the DB in line with Docker after a dgxmm restart (spec §11)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.engines.decision import DECIDER_MODELS, DECIDER_REPO, MODEL_LABEL, SYSTEM_LABEL

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
    for d in rows:  # start-ups still running hold their start-up slots before anything queued resumes
        if d.state in ("starting", "loading") and d.container_id in containers:
            deployer.gate.hold(d.id, "resumed", d.name)
    for d in rows:
        if d.state in ("stopped", "failed", "planned"):
            continue
        if d.state == "queued":  # no container yet: back in the start-up queue (fit is checked at launch)
            deployer.resume_queued(d.id)
            notes.append(f"{d.name}: queued → back in the start-up queue")
            continue
        try:
            c = containers.get(d.container_id) if d.container_id else None
            if c is None:
                deployer._transition(d.id, "failed", message="container missing after dgxmm restart",
                                     last_error="container missing after dgxmm restart", container_id=None)
                notes.append(f"{d.name}: container missing → failed")
            elif c.status in ("exited", "dead"):
                await deployer._fail_exited(d, c.exit_code, c.oom_killed)
                notes.append(f"{d.name}: container exited ({c.exit_code}) → failed")
            elif d.state in ("starting", "loading"):
                deployer.resume_monitor(d.id)
                notes.append(f"{d.name}: still {d.state} → monitoring resumed")
            elif d.state == "stopping":
                await deployer.stop(d.id, reason="reconcile")
                notes.append(f"{d.name}: finishing interrupted stop")
        except Exception:
            log.exception("reconcile error for deployment %s", d.name)
            notes.append(f"{d.name}: error during reconciliation (logged)")
    taken = {r.name for r in rows}
    for cid, c in containers.items():
        if cid in known:
            continue
        try:
            name = c.name.removeprefix("dgxmm-") or "orphan"
            base, n = name, 2
            while name in taken:
                name, n = f"{base}-{n}", n + 1
            taken.add(name)
            port = _flag(c.command, "--port")
            port_i = int(port) if port and port.isdigit() else (min(c.host_ports) if c.host_ports else None)
            engine = c.labels.get("dgxmm.engine", "vllm")
            system = c.labels.get(SYSTEM_LABEL) == "true" or engine == "decision"
            labelled = c.labels.get(MODEL_LABEL)
            decider_ref = labelled if labelled in DECIDER_MODELS else DECIDER_REPO
            model_ref = _flag(c.command, "--model") or _flag(c.command, "-m") or (decider_ref if system else "unknown")
            with deployer.db.session() as s:
                row = Deployment(name=name, engine=engine, model_ref=model_ref,
                                 config={}, state="unknown", container_id=cid, port=port_i, reserved_bytes=0,
                                 estimate={}, last_error="found running without a database record",
                                 system=system or None)
                s.add(row)
                s.flush()
                s.add(DeploymentEvent(deployment_id=row.id, from_state="unknown", to_state="unknown",
                                      message=f"adopted orphan container {c.name}"))
            notes.append(f"{c.name}: orphan managed container → recorded as unknown")
        except Exception:
            log.exception("reconcile error for orphan container %s", c.name)
            notes.append(f"{c.name}: error adopting orphan (logged)")
    for n in notes:
        log.info("reconcile: %s", n)
    return notes
