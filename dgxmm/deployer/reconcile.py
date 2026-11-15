"""Bring the DB in line with Docker after a dgxmm restart (spec §11), on every node.

Each node is reconciled against its own containers. A remote node that is stale (no recent memory sample) or does not
answer its container listing within ``deployer.remote_liveness_timeout_s`` is skipped: its rows are left untouched and
no orphan rows are created for it. It is remembered in ``deployer.reconcile_pending`` with a snapshot of its rows, and
``reconcile(deployer, retry=True)`` (a background task of the app, until nothing is pending) reconciles it once it
answers. A retry runs while the API serves: it lists the node first, then re-reads the rows, and acts on a row only
under the deployment's lock and only if its (state, container id) is still what it was at the restart. A deployment
whose node is no longer configured keeps its row, is logged once and gets last_error ``node not configured: <node>``
(the reason it had before is kept in brackets and restored once the node is configured again); stop/remove then only
release its DB state."""
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


def _unmarked(last_error: str | None, node: str) -> tuple[bool, str | None]:
    """(True, the last_error from before the mark) if ``last_error`` is the R11 mark for ``node``."""
    mark = node_not_configured(node)
    if last_error == mark:
        return True, None
    before = mark + " (before: "
    if last_error and last_error.startswith(before) and last_error.endswith(")"):
        return True, last_error[len(before):-1]
    return False, last_error


def _flag(cmd: list[str], flag: str) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _snapshot(d: Deployment) -> tuple:
    return d.state, d.container_id


def _rows(deployer) -> list[Deployment]:
    with deployer.db.session() as s:
        return list(s.scalars(select(Deployment)))


async def _list_remote(deployer, node: str):
    """(the node's managed containers, None) or (_SKIP, why). Stale: no Docker call at all. Otherwise the listing runs
    in a worker thread and is waited for at most remote_liveness_timeout_s; while a call that outlived its wait is
    still running (deployer._liveness_inflight, shared with the liveness checks: one such call per node), the node is
    skipped."""
    if deployer._node_stale(node):
        return _SKIP, "no recent memory reading"
    inflight = deployer._liveness_inflight.get(node)
    if inflight is not None and not inflight.done():
        return _SKIP, "an earlier call to the node is still running"
    fut = asyncio.ensure_future(asyncio.to_thread(deployer._docker_for(node).list_managed))
    deployer._liveness_inflight[node] = fut
    fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # never "exception was never retrieved"
    try:
        return await asyncio.wait_for(asyncio.shield(fut), deployer.remote_liveness_timeout_s), None
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
    configured = [LOCAL_NODE_ID] + [n for n in (deployer.nodes.ids() if deployer.nodes is not None else ())
                                    if n != LOCAL_NODE_ID]
    if retry:
        nodes = [n for n in configured if n in pending]
    else:
        nodes = configured
        for d in _rows(deployer):
            if deployer._node_of(d) not in configured:
                try:
                    _mark_unconfigured(deployer, d, notes)
                except Exception:
                    log.exception("reconcile error for deployment %s", d.name)
                    notes.append(f"{d.name}: error during reconciliation (logged)")
    for node in nodes:
        remote = node != LOCAL_NODE_ID
        before = _rows(deployer)  # a row removed while the listing runs still owns its (stale) listed container
        if remote:
            listed, why = await _list_remote(deployer, node)
            if listed is _SKIP:
                if node not in pending:
                    pending[node] = {d.id: _snapshot(d) for d in _rows(deployer) if deployer._node_of(d) == node}
                    log.warning("reconcile: node %s skipped (%s); its %d deployment(s) left as they are until it "
                                "answers", node, why, len(pending[node]))
                continue
        else:
            listed = await asyncio.to_thread(deployer.docker.list_managed)
        containers = {c.id: c for c in listed if not c.labels.get("dgxmm.oneshot")}
        snap = pending.pop(node, None)
        try:
            rows = _rows(deployer)  # after the listing: every row and container id as they are now
            node_rows = [d for d in rows if deployer._node_of(d) == node]
            if snap is not None:
                notes.append(f"node {node}: answering again → reconciled")
            try:
                _clear_unconfigured_marks(deployer, node, node_rows, notes)
            except Exception:
                log.exception("reconcile: clearing node-not-configured marks on node %s failed", node)
            # not orphans: containers and deployment ids of rows that existed before or after the listing
            known = {r.container_id for r in (*before, *rows) if r.container_id}
            dep_ids = {str(r.id) for r in (*before, *rows) if deployer._node_of(r) == node}
            await _reconcile_node(deployer, node, remote, node_rows, containers, {r.name for r in rows}, known,
                                  notes, snap, dep_ids)
        except Exception:
            log.exception("reconcile of node %s failed", node)
            notes.append(f"node {node}: error during reconciliation (logged)")
            if snap is not None:  # a retry: try again on the next pass
                pending[node] = snap
    for n in notes:
        log.info("reconcile: %s", n)
    return notes


def _mark_unconfigured(deployer, d: Deployment, notes: list[str]) -> None:
    """R11: the row stays (its container, if any, is on a node DGXMM no longer reaches); the UI shows last_error. The
    reason it had before is kept in brackets."""
    node = deployer._node_of(d)
    if not _unmarked(d.last_error, node)[0]:
        mark = node_not_configured(node)
        with deployer.db.session() as s:
            s.get(Deployment, d.id).last_error = f"{mark} (before: {d.last_error})" if d.last_error else mark
    notes.append(f"{d.name}: on node {node}, which is not configured → left as it is")


def _clear_unconfigured_marks(deployer, node: str, rows: list[Deployment], notes: list[str]) -> None:
    """The node is configured (again): its rows lose the R11 mark and get back the last_error from before it."""
    for d in rows:
        marked, before = _unmarked(d.last_error, node)
        if not marked:
            continue
        with deployer.db.session() as s:
            row = s.get(Deployment, d.id)
            if row is not None and row.last_error == d.last_error:
                row.last_error = before
        notes.append(f"{d.name}: node {node} is configured again → mark removed")


async def _act(deployer, d: Deployment, c, node: str, linked: bool, notes: list[str]) -> bool:
    """Bring one row in line with its container ``c`` (None: missing). ``linked``: ``c`` was found by its label and
    its id is stored now. True if an interrupted stop must be finished (the caller calls deployer.stop)."""
    if linked:
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
        return True
    return False


async def _reconcile_node(deployer, node: str, remote: bool, rows: list[Deployment], containers: dict, taken: set,
                          known: set[str], notes: list[str], snap: dict[int, tuple] | None = None,
                          dep_ids: set[str] | None = None) -> None:
    """``snap``: a retry (the API is serving): only rows whose (state, container id) still equal it, each under its
    deployment's lock. ``known`` / ``dep_ids``: container ids / deployment ids (of this node) of rows that existed at
    any point of this pass; a container with either is never adopted as an orphan (on a remote node)."""
    by_label: dict[str, object] = {}
    if remote:  # a remote launch interrupted between docker run and storing the container id
        for c in containers.values():
            dep_label = c.labels.get("dgxmm.deployment_id")
            if dep_label and c.id not in known and _owned(c, deployer):
                by_label.setdefault(dep_label, c)
    linked: set[str] = set()

    def container_of(d: Deployment):
        c = containers.get(d.container_id) if d.container_id else None
        if c is None and remote and d.state in ("starting", "loading", "ready", "sleeping", "unknown", "stopping"):
            lc = by_label.get(str(d.id))
            if lc is not None and lc.id not in linked:
                return lc, True
        return c, False

    if snap is None:
        for d in rows:  # start-ups still running hold their start-up slots before anything queued resumes
            if d.state in ("starting", "loading") and container_of(d)[0] is not None:
                deployer.gate.hold(d.id, "resumed", d.name)
        for d in rows:
            if d.state in ("stopped", "failed", "planned"):
                continue
            if d.state == "queued":  # no container yet: back in the start-up queue (fit is checked at launch)
                deployer.resume_queued(d.id)
                notes.append(f"{d.name}: queued → back in the start-up queue")
                continue
            try:
                c, by_lbl = container_of(d)
                if by_lbl:
                    linked.add(c.id)
                if await _act(deployer, d, c, node, by_lbl, notes):
                    await deployer.stop(d.id, reason="reconcile")
                    notes.append(f"{d.name}: finishing interrupted stop")
            except Exception:
                log.exception("reconcile error for deployment %s", d.name)
                notes.append(f"{d.name}: error during reconciliation (logged)")
    else:
        # start-ups first, so they hold their slots before anything queued resumes
        for d in sorted((d for d in rows if d.id in snap), key=lambda d: d.state not in ("starting", "loading")):
            finish_stop = False
            try:
                async with deployer._lock(d.id):
                    cur = deployer._get(d.id)
                    if cur is None or _snapshot(cur) != snap[d.id] or cur.id in deployer.launching \
                            or deployer.has_monitor(cur.id) or cur.state in ("stopped", "failed", "planned"):
                        continue  # changed since the restart (or nothing to do): the live code owns it
                    if cur.state == "queued":
                        deployer.resume_queued(cur.id)
                        notes.append(f"{cur.name}: queued → back in the start-up queue")
                        continue
                    c, by_lbl = container_of(cur)
                    if by_lbl:
                        linked.add(c.id)
                    if cur.state in ("starting", "loading") and c is not None:
                        deployer.gate.hold(cur.id, "resumed", cur.name)
                    finish_stop = await _act(deployer, cur, c, node, by_lbl, notes)
                if finish_stop:  # stop() takes the deployment's lock itself
                    await deployer.stop(d.id, reason="reconcile")
                    notes.append(f"{d.name}: finishing interrupted stop")
            except Exception:
                log.exception("reconcile error for deployment %s", d.name)
                notes.append(f"{d.name}: error during reconciliation (logged)")
    node_ids = {str(d.id) for d in rows} | (dep_ids or set())
    for cid, c in containers.items():
        if cid in known or cid in linked:
            continue
        if remote and c.labels.get("dgxmm.deployment_id") in node_ids:
            continue  # its deployment's row exists (being launched, or changed since the restart): not an orphan
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
