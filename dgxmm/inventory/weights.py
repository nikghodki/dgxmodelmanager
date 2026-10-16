"""Delete downloaded weights and clean up incomplete ones (spec §1 + §4).

Safety: every target must live inside the HF hub dir or a configured GGUF root (after resolving symlinks),
configured roots are never deleted, models used by a non-terminal deployment or with an active download are
refused. Root-owned files are removed by a short-lived helper container (see DockerController.run_helper);
when that is unavailable we fall back to deleting in-process.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sqlalchemy import select

from dgxmm.db import Database, Deployment, Download
from dgxmm.docker_ctl import HELPER_MOUNT
from dgxmm.inventory.artifact import REPO_RE

log = logging.getLogger(__name__)
TERMINAL_STATES = frozenset({"stopped", "failed", "exited", "removed"})
DEFAULT_HELPER_IMAGE = "dgxmm:latest"


class DeleteError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


@dataclass
class Target:
    path: Path        # where it lives (parent resolved, final component not followed)
    is_dir: bool
    bytes: int


@dataclass
class DeletePlan:
    ref: str
    source: str
    name: str
    targets: list[Target]
    blocking: list[dict] = field(default_factory=list)
    affected: list[dict] = field(default_factory=list)
    download_active: bool = False
    rmdir_if_empty: Path | None = None

    @property
    def bytes(self) -> int:
        return sum(t.bytes for t in self.targets)


def _tree_bytes(p: Path) -> int:
    """Bytes of regular files under p, never following symlinks."""
    try:
        st = p.lstat()
    except OSError:
        return 0
    if p.is_symlink():
        return 0
    if not p.is_dir():
        return st.st_size
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                if not fp.is_symlink():
                    total += fp.lstat().st_size
            except OSError:
                pass
    return total


class WeightsDeleter:
    def __init__(self, settings, db: Database, inventory, docker=None):
        self.settings, self.db, self.inventory, self.docker = settings, db, inventory, docker
        self._lock = threading.Lock()

    # ------------------------------------------------------------ paths
    @property
    def hub(self) -> Path:
        return Path(self.settings.hf_cache.local) / "hub"

    def _roots(self) -> list[Path]:
        """Directories targets may live in."""
        return [self.hub.resolve()] + [Path(m.local).resolve() for m in self.settings.gguf_roots]

    def _protected(self) -> list[Path]:
        """Directories that must never be deleted (nor any of their ancestors)."""
        hub = self.hub.resolve()
        return self._roots() + [Path(self.settings.hf_cache.local).resolve(), hub / ".locks"]

    def _check(self, p: Path) -> Path:
        """Return p with its parent resolved; refuse anything outside the roots or a root itself."""
        if ".." in p.parts:
            raise DeleteError("unsafe_path", f"refusing to delete {p}: path traversal")
        loc = p.parent.resolve() / p.name
        real = p.resolve()
        roots = self._roots()
        for q in (loc, real):
            if not any(q.is_relative_to(r) for r in roots):
                raise DeleteError("unsafe_path", f"refusing to delete {p}: outside the model directories")
            if any(r == q or r.is_relative_to(q) for r in self._protected()):
                raise DeleteError("unsafe_path", f"refusing to delete {p}: it is a configured model root")
        return loc

    def _mappings(self) -> list[tuple[Path, str]]:
        ms = [self.settings.hf_cache, *self.settings.gguf_roots]
        return sorted(((Path(m.local).resolve(), m.host) for m in ms), key=lambda x: len(str(x[0])), reverse=True)

    def to_host(self, p: Path) -> str | None:
        for local, host in self._mappings():
            if p == local or p.is_relative_to(local):
                rel = p.relative_to(local)
                return str(PurePosixPath(host) / rel) if str(rel) != "." else str(PurePosixPath(host))
        return None

    def _host_or_local(self, p: Path) -> str:
        return self.to_host(p) or str(p)

    # ------------------------------------------------------------ planning
    def _repo_of(self, loc: Path) -> str | None:
        """HF repo id a path inside the hub belongs to (for the download check)."""
        hub = self.hub.resolve()
        if not loc.is_relative_to(hub):
            return None
        parts = loc.relative_to(hub).parts
        if parts and parts[0].startswith("models--") and parts[0].count("--") >= 2:
            bits = parts[0].split("--")
            return f"{bits[1]}/{'--'.join(bits[2:])}"
        return None

    def _deployments(self, ref: str) -> tuple[list[dict], list[dict]]:
        blocking, affected = [], []
        with self.db.session() as s:
            for d in s.scalars(select(Deployment).order_by(Deployment.id)):
                draft = ((d.config or {}).get("speculative") or {}).get("model") \
                    if isinstance((d.config or {}).get("speculative"), dict) else None
                if d.model_ref != ref and draft != ref:
                    continue
                row = {"id": d.id, "name": d.name, "state": d.state}
                (affected if d.state in TERMINAL_STATES else blocking).append(row)
        return blocking, affected

    def _download_active(self, repo: str | None) -> bool:
        if not repo:
            return False
        with self.db.session() as s:
            return s.scalar(select(Download.id).where(Download.repo_id == repo,
                                                      Download.state.in_(("queued", "running")))) is not None

    def plan(self, ref: str) -> DeletePlan:
        art = self.inventory.get(ref)
        if art is None:
            raise DeleteError("not_found", f"{ref} is not in the inventory", 404)
        targets: list[Target] = []
        rmdir = None
        if art.source == "hf":
            if ".." in ref or not re.fullmatch(REPO_RE, ref):
                raise DeleteError("invalid_ref", f"{ref!r} is not a valid repo id")
            dirname = "models--" + ref.replace("/", "--")
            repo_dir = self.hub / dirname
            if not os.path.lexists(repo_dir):
                raise DeleteError("not_found", f"{ref} is not in the HF cache", 404)
            loc = self._check(repo_dir)
            targets.append(Target(loc, True, _tree_bytes(repo_dir.resolve())))
            locks = self.hub / ".locks" / dirname
            if os.path.lexists(locks):
                lloc = self._check(locks)
                targets.append(Target(lloc, lloc.is_dir() and not lloc.is_symlink(), _tree_bytes(lloc)))
            repo, name = ref, ref.split("/")[-1]
        else:
            first = Path(ref)
            if not first.is_absolute():
                raise DeleteError("unsafe_path", f"refusing to delete {ref}: not an absolute path")
            shards = [first.with_name(f) for f in art.files] or [first]
            if first not in shards:
                shards.insert(0, first)
            hub = self.hub.resolve()
            for sh in shards:
                if not os.path.lexists(sh):
                    continue  # missing shard of an incomplete split GGUF
                loc = self._check(sh)
                if loc.is_dir() and not loc.is_symlink():
                    raise DeleteError("unsafe_path", f"refusing to delete {sh}: not a file")
                if loc.is_symlink():
                    targets.append(Target(loc, False, 0))
                    real = loc.resolve()
                    # HF cache snapshot entry: also delete the blob it points at, which holds the bytes
                    if real.is_file() and real.is_relative_to(hub) and real.parent.name == "blobs" \
                            and loc.is_relative_to(hub) and self._repo_of(real) == self._repo_of(loc):
                        targets.append(Target(self._check(real), False, real.stat().st_size))
                else:
                    targets.append(Target(loc, False, _tree_bytes(loc)))
            parent = first.parent.resolve()
            if not parent.is_relative_to(hub):
                rmdir = parent
            repo, name = self._repo_of(first.parent.resolve() / first.name), first.name
        if not targets:
            raise DeleteError("not_found", f"nothing left on disk for {ref}", 404)
        blocking, affected = self._deployments(ref)
        return DeletePlan(ref=ref, source=art.source, name=name, targets=targets, blocking=blocking,
                          affected=affected, download_active=self._download_active(repo), rmdir_if_empty=rmdir)

    def _view(self, plan: DeletePlan) -> dict:
        return {"ref": plan.ref, "source": plan.source, "name": plan.name,
                "paths": [self._host_or_local(t.path) for t in plan.targets], "bytes": plan.bytes,
                "deployments_blocking": plan.blocking, "deployments_affected": plan.affected,
                "download_active": plan.download_active, "blocked": bool(plan.blocking or plan.download_active)}

    def preview(self, ref: str) -> dict:
        return self._view(self.plan(ref))

    # ------------------------------------------------------------ deletion
    def _helper_image(self) -> str:
        return self.settings.helper_image or (self.docker.self_image() if self.docker else None) \
            or DEFAULT_HELPER_IMAGE

    def _run_helpers(self, targets: list[Target]) -> bool:
        """Delete via root helper containers, one per parent directory. Returns True if any helper ran."""
        if self.docker is None:
            return False
        groups: dict[Path, list[Target]] = defaultdict(list)
        for t in targets:
            groups[t.path.parent].append(t)
        ran = False
        image = None
        for parent, ts in groups.items():
            host = self.to_host(parent)
            if host is None:
                continue
            image = image or self._helper_image()
            flag = "-rf" if any(t.is_dir for t in ts) else "-f"
            cmd = ["rm", flag, "--", *[f"{HELPER_MOUNT}/{t.path.name}" for t in ts]]
            try:
                code = self.docker.run_helper(image, host, cmd, timeout=self.settings.helper_timeout_s)
                ran = True
                if code not in (0, None):
                    log.warning("delete helper for %s exited with %s", host, code)
            except Exception as e:
                log.warning("delete helper unavailable for %s (%s); deleting in-process", host, e)
        return ran

    def _execute(self, plan: DeletePlan) -> str:
        ran = self._run_helpers(plan.targets)
        remaining = [t for t in plan.targets if os.path.lexists(t.path)]
        errors: list[tuple[Path, OSError]] = []
        for t in remaining:
            try:
                if t.path.is_dir() and not t.path.is_symlink():
                    shutil.rmtree(t.path)
                else:
                    t.path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                errors.append((t.path, e))
        if errors:
            p, e = errors[0]
            host = self._host_or_local(p)
            hint = f"sudo chown -R {os.getuid()}:{os.getgid()} {host}"
            raise DeleteError("delete_failed", f"could not delete {host}: {e.strerror or e}. "
                              f"The files are probably owned by root. Fix on the host: {hint}", 500,
                              hint=hint, paths=[self._host_or_local(x) for x, _ in errors])
        if plan.rmdir_if_empty is not None and plan.rmdir_if_empty not in self._protected():
            try:
                self._check(plan.rmdir_if_empty)
                if plan.rmdir_if_empty.is_dir() and not any(plan.rmdir_if_empty.iterdir()):
                    plan.rmdir_if_empty.rmdir()
            except (OSError, DeleteError):
                pass
        if not remaining:
            return "helper"
        return "helper+in_process" if ran else "in_process"

    def _delete_planned(self, plan: DeletePlan) -> dict:
        if plan.blocking:
            names = ", ".join(f"{d['name']} ({d['state']})" for d in plan.blocking)
            raise DeleteError("model_in_use", f"{plan.ref} is used by running deployment(s): {names}. Stop them first.",
                              409, deployments=plan.blocking)
        if plan.download_active:
            raise DeleteError("download_active", f"{plan.ref} is being downloaded; cancel the download first", 409)
        view = self._view(plan)
        method = self._execute(plan)
        return {"ref": plan.ref, "freed_bytes": plan.bytes, "paths": view["paths"], "method": method,
                "warnings": plan.affected}

    def delete(self, ref: str, refresh: bool = True) -> dict:
        with self._lock:
            try:
                return self._delete_planned(self.plan(ref))
            finally:
                if refresh:
                    self.inventory.refresh()

    # ------------------------------------------------------------ incomplete cleanup
    def cleanup_preview(self) -> dict:
        items = []
        for a in self.inventory.list():
            if a.complete:
                continue
            item = {"ref": a.ref, "source": a.source, "problems": a.problems}
            try:
                v = self.preview(a.ref)
                reason = "model_in_use" if v["deployments_blocking"] else \
                    ("download_active" if v["download_active"] else None)
                item.update(name=v["name"], paths=v["paths"], bytes=v["bytes"],
                            deployments_blocking=v["deployments_blocking"],
                            deployments_affected=v["deployments_affected"], download_active=v["download_active"],
                            eligible=reason is None, skip_reason=reason)
            except DeleteError as e:
                item.update(name=a.ref.split("/")[-1], paths=[], bytes=0, deployments_blocking=[],
                            deployments_affected=[], download_active=False, eligible=False, skip_reason=e.code,
                            message=e.message)
            items.append(item)
        ok = [i for i in items if i["eligible"]]
        return {"items": items, "count": len(ok), "bytes": sum(i["bytes"] for i in ok)}

    def cleanup(self, refs: list[str] | None = None) -> dict:
        if refs is None:
            refs = [a.ref for a in self.inventory.list() if not a.complete]
        deleted, skipped, failed = [], [], []
        with self._lock:
            try:
                for ref in dict.fromkeys(refs):
                    art = self.inventory.get(ref)
                    if art is None:
                        skipped.append({"ref": ref, "code": "not_found", "message": f"{ref} is not in the inventory"})
                        continue
                    if art.complete:
                        skipped.append({"ref": ref, "code": "not_incomplete",
                                        "message": "complete models are not removed by cleanup"})
                        continue
                    try:
                        deleted.append(self._delete_planned(self.plan(ref)))
                    except DeleteError as e:
                        entry = {"ref": ref, "code": e.code, "message": e.message}
                        (failed if e.code == "delete_failed" else skipped).append(entry)
            finally:
                self.inventory.refresh()
        return {"deleted": deleted, "skipped": skipped, "failed": failed,
                "freed_bytes": sum(d["freed_bytes"] for d in deleted)}
