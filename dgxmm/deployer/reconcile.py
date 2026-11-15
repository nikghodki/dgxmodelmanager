"""Bring the DB in line with Docker after a dgxmm restart (spec §11), on every node.

Each node is reconciled against its own containers. A remote node that is stale (no recent memory sample) or does not
answer its container listing within ``deployer.remote_liveness_timeout_s`` is skipped: its rows are left untouched and
no orphan rows are created for it. It is remembered in ``deployer.reconcile_pending`` with a snapshot of its rows, and
``reconcile(deployer, retry=True)`` (run by the app until nothing is pending) reconciles it once it answers, but only
the rows that nothing changed since the restart. A deployment whose node is no longer configured keeps its row, is
logged once and gets last_error ``node not configured: <node>`` (stop/remove then only release its DB state)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.config import LOCAL_NODE_ID
from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.engines.decision import DECIDER_MODELS, DECIDER_REPO, MODEL_LABEL, SYSTEM_LABEL

log = logging.getLogger(__name__)

_SKIP = object()  # a remote node whose containers could not be listed


def node_not_configured(node: str) -> str:
    return f"node not configured: {node}"


def _flag(cmd: list[str], flag: str) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _snapshot(d: Deployment) -> tuple:
    return d.state, d.container_id


async def _list_remote(deployer, node: str):
    """The node's managed containers, or _SKIP (stale: no Docker call at all; unreachable, failing or slow: the
    listing runs in a worker thread and is waited for at most remote_liveness_timeout_s)."""
    if deployer._node_stale(node):
        return _SKIP, "no recent memory reading"
    try:
        return await asyncio.wait_for(asyncio.to_thread(deployer._docker_for(node).list_managed),
                                      deployer.remote_liveness_timeout_s), None
    except asyncio.TimeoutError:
        return _SKIP, f"container listing timed out after {deployer.remote_liveness_timeout_s:.1f}s"
    except Exception as e:  # NodeUnreachable, or the node's Docker answering with an error
        return _SKIP, str(e) or type(e).__name__


def _owned(c, deployer) -> bool:
    """A container labelled for this install (or for no particular install)."""
    install = deployer.settings.install_id
    return not install or c.labels.get("dgxmm.install") in (None, install)


async def reconcile(deployer, retry: bool = False) -> list[str]:
    """``retry``: reconcile only the nodes skipped earlier (deployer.reconcile_pending)."""
    notes: list[str] = []
    pending: dict[str, dict[int, tuple]] = deployer.reconcile_pending
    with deployer.db.session() as s:
        rows = list(s.scalars(select(Deployment)))
    by_node: dict[str, list[Deployment]] = {}
    for r in rows:
        by_node.setdefault(deployer._node_of(r), []).append(r)
    configured = [LOCAL_NODE_ID] + [n for n in (deployer.nodes.ids() if deployer.nodes is not None else ())
                                    if n != LOCAL_NODE_ID]
    taken = {r.name for r in rows}
    if retry:
        nodes = [n for n in configured if n in pending]
    else:
        nodes = configured
        for node in sorted(set(by_node) - set(configured)):
            for d in by_node[node]:
                _mark_unconfigured(deployer, d, notes)
    for node in nodes:
        remote = node != LOCAL_NODE_ID
        node_rows = by_node.get(node, [])
        if remote:
            listed, why = await _list_remote(deployer, node)
            if listed is _SKIP:
                if node not in pending:
                    pending[node] = {d.id: _snapshot(d) for d in node_rows}
                    log.warning("reconcile: node %s skipped (%s); its %d deployment(s) left as they are until it "
                                "answers", node, why, len(node_rows))
                continue
        else:
            listed = await asyncio.to_thread(deployer.docker.list_managed)
        containers = {c.id: c for c in listed if not c.labels.get("dgxmm.oneshot")}
        snap = pending.pop(node, None)
        if snap is not None:  # a retry: only the rows nothing touched since the restart
            node_rows = [d for d in node_rows if snap.get(d.id) == _snapshot(d) and d.id not in deployer.launching
                         and not deployer.has_monitor(d.id)]
            notes.append(f"node {node}: answering again → reconciled")
        await _reconcile_node(deployer, node, remote, node_rows, containers, taken,
                              {r.id for r in rows if deployer._node_of(r) == node},
                              {r.container_id for r in rows if r.container_id}, notes)
    for n in notes:
        log.info("reconcile: %s", n)
    return notes


def _mark_unconfigured(deployer, d: Deployment, notes: list[str]) -> None:
    """R11: the row stays (its container, if any, is on a node DGXMM no longer reaches); the UI shows last_error."""
    msg = node_not_configured(deployer._node_of(d))
    if d.last_error != msg:
        with deployer.db.session() as s:
            s.get(Deployment, d.id).last_error = msg
    notes.append(f"{d.name}: on node {deployer._node_of(d)}, which is not configured → left as it is")


async def _reconcile_node(deployer, node: str, remote: bool, rows: list[Deployment], containers: dict, taken: set,
                          node_ids: set[int], known: set[str], notes: list[str]) -> None:
    by_label: dict[str, object] = {}
    if remote:  # a remote launch interrupted between docker run and storing the container id
        for c in containers.values():
            dep_label = c.labels.get("dgxmm.deployment_id")
            if dep_label and c.id not in known and _owned(c, deployer):
                by_label.setdefault(dep_label, c)
    for d in rows:  # start-ups still running hold their start-up slots before anything queued resumes
        c = containers.get(d.container_id) if d.container_id else None
        if c is None and remote and d.state in ("starting", "loading"):
            c = by_label.get(str(d.id))
        if d.state in ("starting", "loading") and c is not None:
            deployer.gate.hold(d.id, "resumed", d.name)
    linked: set[str] = set()
    for d in rows:
        if d.state in ("stopped", "failed", "planned"):
            continue
        if d.state == "queued":  # no container yet: back in the start-up queue (fit is checked at launch)
            deployer.resume_queued(d.id)
            notes.append(f"{d.name}: queued → back in the start-up queue")
            continue
        try:
            c = containers.get(d.container_id) if d.container_id else None
            if c is None and remote and (lc := by_label.get(str(d.id))) is not None:
                c = lc
                linked.add(c.id)
                with deployer.db.session() as s:
                    s.get(Deployment, d.id).container_id = c.id
                    s.add(DeploymentEvent(deployment_id=d.id, from_state=d.state, to_state=d.state,
                                          message=f"container {c.id[:12]} found on node {node} by its label"))
                d.container_id = c.id
                notes.append(f"{d.name}: container found on node {node} by its label → linked")
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
    for cid, c in containers.items():
        if cid in known or cid in linked:
            continue
        if remote and c.labels.get("dgxmm.deployment_id") in {str(i) for i in node_ids}:
            continue  # its deployment's row exists but was not reconciled now (changed since the restart)
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
                                 system=system or None, node=node)
                s.add(row)
                s.flush()
                s.add(DeploymentEvent(deployment_id=row.id, from_state="unknown", to_state="unknown",
                                      message=f"adopted orphan container {c.name}"
                                              + (f" on node {node}" if remote else "")))
            notes.append(f"{c.name}: orphan managed container" + (f" on node {node}" if remote else "")
                         + " → recorded as unknown")
        except Exception:
            log.exception("reconcile error for orphan container %s", c.name)
            notes.append(f"{c.name}: error adopting orphan (logged)")
