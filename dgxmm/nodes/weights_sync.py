"""Weights on remote nodes (Task 6).

``weights_present``: the check a remote deploy runs first. The node's probe container (dgxmm/nodes/probe.py) has the
node's HF cache mounted read-only at ``/hf``; one exec there tests every file of the snapshot (the paths are the
script's arguments, never part of its code) and prints ``present`` only when all of them are regular files (links
followed, so a link to a blob that is not there counts as missing).

``copy_argv`` / ``WeightsSync.copy_weights``: copy one snapshot from this node's HF cache to a node's cache over the
cluster link as ``tar -c | ssh <node> tar -x``: two processes started with ``asyncio.create_subprocess_exec`` (no
local shell), joined by one pipe. Only the requested snapshot, the blobs its links point to and the repo's ``refs/``
are archived (never a ``blobs/`` wildcard); the snapshot's links come last, so a copy that is cut short leaves no link
to a blob that was not fully written. The job is a Download row with ``kind="copy"`` (migration 44), so it shows on
the Downloads page and is cancelled from there (DownloadService.cancel -> CopyJob.kill: both processes are killed).
"""
from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from sqlalchemy import select

from dgxmm.config import LOCAL_NODE_ID, NodeConfig
from dgxmm.db import Download
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact
from dgxmm.nodes.registry import ssh_config_path

log = logging.getLogger(__name__)

PROBE_HF = "/hf"  # where NodeProbe mounts the node's HF cache
PRESENT = "present"
WEIGHTS_CHECK_SCRIPT = 'for f in "$@"; do [ -f "$f" ] || exit 0; done; echo ' + PRESENT
REVISION_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
REPO_DIR_RE = r"^models--[A-Za-z0-9][A-Za-z0-9._-]*$"
BLOB_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SSH_USER_RE = r"^[A-Za-z_][A-Za-z0-9._-]{0,31}$"
STDERR_LINES, STDERR_CHARS = 10, 1000
KILL_WAIT_S = 5.0


class CopyError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def repo_dir_for(ref: str) -> str:
    """The HF cache dir of a repo id (``org/name`` -> ``models--org--name``); ValueError if it is not a repo id."""
    if not isinstance(ref, str) or not re.fullmatch(REPO_RE, ref) or ".." in ref:
        raise ValueError(f"invalid repo id {ref!r}")
    return "models--" + ref.replace("/", "--")


def validate_repo_dir(repo_dir: str) -> None:
    """``models--org--name`` of a valid repo id, nothing else (no path separators, no ``..``)."""
    if not isinstance(repo_dir, str) or ".." in repo_dir or not re.fullmatch(REPO_DIR_RE, repo_dir):
        raise ValueError(f"invalid repo dir {repo_dir!r}")
    org, sep, name = repo_dir[len("models--"):].partition("--")
    if not sep or not re.fullmatch(REPO_RE, f"{org}/{name}"):
        raise ValueError(f"invalid repo dir {repo_dir!r}")


def validate_revision(revision: str) -> None:
    """A snapshot dir name (a commit hash in practice): one path component, no ``..``, never an option."""
    if not isinstance(revision, str) or ".." in revision or not re.fullmatch(REVISION_RE, revision):
        raise ValueError(f"invalid revision {revision!r}")


def _hf_relative(local_path: str, hf_local: str) -> PurePosixPath | None:
    """``local_path`` relative to this node's HF cache (None: outside it); ValueError for a ``..`` component."""
    try:
        rel = PurePosixPath(Path(local_path).relative_to(Path(hf_local)).as_posix())
    except ValueError:
        return None
    if ".." in rel.parts or not rel.parts:
        raise ValueError(f"invalid model path {local_path!r}")
    return rel


def gguf_in_hf_cache(artifact: ModelArtifact, hf_local: str) -> bool:
    return artifact.source == "gguf" and _hf_relative(artifact.local_path, hf_local) is not None


def gguf_copy_ref(artifact: ModelArtifact, hf_local: str) -> tuple[str, str] | None:
    """(repo id, revision) of a GGUF inside an HF snapshot (hub/models--org--name/snapshots/<rev>/...), the
    snapshot copy_weights copies; None if the file is not in a snapshot."""
    rel = _hf_relative(artifact.local_path, hf_local)
    if rel is None or len(rel.parts) < 5 or rel.parts[0] != "hub" or rel.parts[2] != "snapshots":
        return None
    repo_dir, revision = rel.parts[1], rel.parts[3]
    try:
        validate_repo_dir(repo_dir)
        validate_revision(revision)
    except ValueError:
        return None
    org, _sep, name = repo_dir[len("models--"):].partition("--")
    return f"{org}/{name}", revision


def weights_paths(artifact: ModelArtifact, hf_local: str | None = None) -> list[str]:
    """The probe-side paths to test: every file of an HF snapshot (config.json first), or the one GGUF file
    (``hf_local``: this node's HF cache, the GGUF must be inside it)."""
    if artifact.source == "gguf":
        rel = _hf_relative(artifact.local_path, hf_local) if hf_local else None
        if rel is None:
            raise ValueError(f"{artifact.local_path} is outside the HF cache; it cannot be on a remote node")
        return [f"{PROBE_HF}/{rel}"]
    repo_dir = repo_dir_for(artifact.ref)
    validate_revision(artifact.revision or "")
    snap = f"{PROBE_HF}/hub/{repo_dir}/snapshots/{artifact.revision}"
    files = ["config.json"] + sorted({f for f in artifact.files if f and f != "config.json"})
    for f in files:
        if f.startswith("/") or ".." in f.split("/"):
            raise ValueError(f"invalid snapshot file {f!r}")
    return [f"{snap}/{f}" for f in files]


def weights_present(probe, node_id: str, artifact: ModelArtifact, hf_local: str | None = None) -> bool:
    """True if every file of ``artifact`` (an HF snapshot, or a GGUF inside this node's HF cache ``hf_local``) is in
    node ``node_id``'s HF cache. Blocking (SSH): call it from a worker thread. Errors of the exec itself (node
    unreachable, no probe container) propagate: they are not an answer."""
    if artifact.source not in ("hf", "gguf"):
        raise ValueError(f"cannot check a {artifact.source} artifact on a node")
    out = probe.exec(node_id, ["sh", "-c", WEIGHTS_CHECK_SCRIPT, "sh", *weights_paths(artifact, hf_local)])
    return out.strip() == PRESENT


def _ssh_user(node: NodeConfig) -> str:
    user = node.ssh_user
    if not user and node.docker_url.startswith("ssh://") and "@" in node.docker_url:
        user = node.docker_url[len("ssh://"):].split("@", 1)[0]
    if not user or not re.fullmatch(SSH_USER_RE, user):
        raise ValueError(f"node {node.id} has no valid ssh user")
    return user


def _snapshot_blobs(repo: Path, snap: Path) -> list[str]:
    """Names of the blobs the snapshot's links point to; ValueError for a link that leaves ``repo/blobs``."""
    blobs: set[str] = set()
    for root, dirs, files in os.walk(snap):
        for name in dirs + files:
            p = Path(root) / name
            if not p.is_symlink():
                continue
            target = os.path.normpath(os.path.join(root, os.readlink(p)))
            rel = os.path.relpath(target, repo).split(os.sep)
            if len(rel) != 2 or rel[0] != "blobs" or not re.fullmatch(BLOB_RE, rel[1]):
                raise ValueError(f"snapshot entry {p.relative_to(snap)} does not point into the repo's blobs")
            blobs.add(rel[1])
    return sorted(blobs)


def copy_argv(local_hf: str, node: NodeConfig, repo_dir: str, revision: str) -> tuple[list[str], list[str]]:
    """The ``tar -c`` argv (run here, on this node's HF cache ``local_hf``) and the ``ssh … tar -x`` argv (into the
    node's ``hf_cache_host``) whose stdout/stdin are piped together. Reads the local snapshot's links (blocking)."""
    validate_repo_dir(repo_dir)
    validate_revision(revision)
    host = node.hf_cache_host
    if not node.cluster_ip:
        raise ValueError(f"node {node.id} has no cluster_ip")
    if not host or not host.startswith("/") or ".." in host.split("/"):
        raise ValueError(f"node {node.id} needs an absolute hf_cache_host")
    user = _ssh_user(node)
    hub = Path(local_hf) / "hub"
    repo = hub / repo_dir
    snap = repo / "snapshots" / revision
    blobs = _snapshot_blobs(repo, snap) if snap.is_dir() else []
    refs = [f"{repo_dir}/refs"] if (repo / "refs").is_dir() else []
    tar = ["tar", "-C", str(hub), "-cf", "-",
           "--no-recursion", repo_dir,  # the repo dir entry itself, not its contents
           "--recursion", *[f"{repo_dir}/blobs/{b}" for b in blobs], *refs,
           f"{repo_dir}/snapshots/{revision}"]  # links last (module docstring)
    # ssh runs the remote command through the remote user's shell: every word is a validated constant (no quoting
    # needed: the path is absolute, checked above, and node config comes from the admin's settings file)
    remote_hub = posixpath.join(host, "hub")
    if not re.fullmatch(r"^/[A-Za-z0-9._/-]+$", remote_hub):
        raise ValueError(f"node {node.id}: hf_cache_host {host!r} has characters a remote shell would interpret")
    # the config NodeRegistry wrote (key, pinned known_hosts per node), explicit: never ~ from passwd
    ssh = ["ssh", "-F", str(ssh_config_path()), "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=10", "-l", user, "--", node.cluster_ip,
           "tar", "-C", remote_hub, "-xf", "-"]
    return tar, ssh


def _snapshot_bytes(snap: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(snap):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size  # follows the link to the blob
            except OSError:
                pass
    return total


def _tail(err: bytes | None) -> str:
    text = (err or b"").decode("utf-8", errors="replace").strip()
    return "\n".join(text.splitlines()[-STDERR_LINES:])[-STDERR_CHARS:]


@dataclass
class CopyJob:
    id: int
    node: str
    ref: str
    revision: str
    task: asyncio.Task | None = None
    procs: list = field(default_factory=list)

    def kill(self) -> None:
        """Stop the copy: the task's cancellation kills both processes and marks the row cancelled."""
        if self.task is not None and not self.task.done():
            self.task.cancel()


class WeightsSync:
    """Copies of snapshots to remote nodes (admin API: POST /api/nodes/{node_id}/copy-weights)."""

    def __init__(self, settings, db, nodes, events, downloads, spawn: Callable[..., Any] | None = None):
        self.settings, self.db, self.nodes, self.events, self.downloads = settings, db, nodes, events, downloads
        self.spawn = spawn or asyncio.create_subprocess_exec
        self.jobs: dict[int, CopyJob] = {}

    def _target(self, node_id: str) -> NodeConfig:
        if node_id == LOCAL_NODE_ID:
            raise CopyError("local_node", "weights are copied from the local node to a remote node", 400)
        if self.nodes is None or node_id not in self.nodes.ids() or not self.nodes.is_remote(node_id):
            raise CopyError("unknown_node", f"node {node_id!r} is not configured", 400)
        return self.nodes.config(node_id)

    async def copy_weights(self, node_id: str, ref: str, revision: str, user_id: int | None = None) -> CopyJob:
        node = self._target(node_id)
        try:
            repo_dir = repo_dir_for(ref)
        except ValueError as e:
            raise CopyError("invalid_ref", str(e), 400)
        try:
            validate_revision(revision)
        except ValueError as e:
            raise CopyError("invalid_revision", str(e), 400)
        snap = Path(self.settings.hf_cache.local) / "hub" / repo_dir / "snapshots" / revision

        def plan():
            if not snap.is_dir():
                return None
            return copy_argv(self.settings.hf_cache.local, node, repo_dir, revision), _snapshot_bytes(snap)
        try:
            planned = await asyncio.to_thread(plan)
        except ValueError as e:
            raise CopyError("invalid_snapshot", str(e), 400)
        if planned is None:
            raise CopyError("not_local", f"{ref} revision {revision} is not in this node's HF cache", 404)
        (tar_argv, ssh_argv), total = planned
        with self.db.session() as s:
            if s.scalar(select(Download.id).where(Download.kind == "copy", Download.node == node_id,
                                                  Download.repo_id == ref,
                                                  Download.state.in_(("queued", "running")))):
                raise CopyError("already_copying", f"{ref} is already being copied to {node_id}", 409)
            row = Download(repo_id=ref, revision=revision, files=[], bytes_total=total, bytes_done=0, state="queued",
                           kind="copy", node=node_id, created_by=user_id)
            s.add(row)
            s.flush()
            job = CopyJob(id=row.id, node=node_id, ref=ref, revision=revision)
        self._publish(job.id)
        job.task = asyncio.get_running_loop().create_task(self._run(job, tar_argv, ssh_argv))
        self.jobs[job.id] = job
        job.task.add_done_callback(lambda _t: self.jobs.pop(job.id, None))  # also a task cancelled before it ran
        self.downloads.track(job.id, job.task, job)  # Downloads cancel -> job.kill()
        return job

    def _update(self, dl_id: int, **fields) -> None:
        with self.db.session() as s:
            d = s.get(Download, dl_id)
            for k, v in fields.items():
                setattr(d, k, v)
        self._publish(dl_id)

    def _publish(self, dl_id: int) -> None:
        if self.events is not None:
            self.events.publish("download", self.downloads.view(dl_id))

    async def _run(self, job: CopyJob, tar_argv: list[str], ssh_argv: list[str]) -> None:
        procs: list = []
        try:
            self._update(job.id, state="running")
            r, w = os.pipe()
            try:
                procs.append(await self.spawn(*tar_argv, stdin=asyncio.subprocess.DEVNULL, stdout=w,
                                              stderr=asyncio.subprocess.PIPE))
                procs.append(await self.spawn(*ssh_argv, stdin=r, stdout=asyncio.subprocess.DEVNULL,
                                              stderr=asyncio.subprocess.PIPE))
            finally:
                os.close(r)  # the children hold their ends; ssh sees EOF when tar exits
                os.close(w)
            job.procs = procs
            error = await self._wait(*procs)
            if error:
                self._update(job.id, state="failed", error=error)
                log.warning("copy of %s to %s failed: %s", job.ref, job.node, error)
            else:
                self._update(job.id, state="done", bytes_done=self.downloads.view(job.id)["bytes_total"])
                log.info("copied %s@%s to %s", job.ref, job.revision, job.node)
        except asyncio.CancelledError:
            await self._kill(procs)
            self._update(job.id, state="cancelled")
            raise
        except Exception as e:
            await self._kill(procs)
            self._update(job.id, state="failed", error=f"copy failed: {e}")
            log.warning("copy of %s to %s failed", job.ref, job.node, exc_info=True)

    async def _wait(self, tar, ssh) -> str | None:
        """Wait for both; when one exits non-zero the other is killed. The error text (None: both exited 0) has the
        stderr tail of each side that failed by itself (ssh first: its error is the cause when tar only saw a broken
        pipe)."""
        tasks = {asyncio.ensure_future(p.communicate()): p for p in (tar, ssh)}
        killed: set[int] = set()
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if pending and any(tasks[t].returncode != 0 for t in done):
                    for t in pending:
                        killed.add(id(tasks[t]))
                        self._kill_one(tasks[t])
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
        stderr = {id(p): t.result()[1] for t, p in tasks.items()}
        parts = []
        for name, p in (("ssh", ssh), ("tar", tar)):
            if p.returncode != 0 and id(p) not in killed:
                tail = _tail(stderr[id(p)])
                parts.append(f"{name} exited {p.returncode}" + (f": {tail}" if tail else ""))
        return "; ".join(parts) or None

    @staticmethod
    def _kill_one(p) -> None:
        if p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass

    async def _kill(self, procs: list) -> None:
        for p in procs:
            self._kill_one(p)
        for p in procs:
            try:
                await asyncio.wait_for(asyncio.shield(p.wait()), KILL_WAIT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
