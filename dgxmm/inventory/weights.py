"""Delete downloaded weights and clean up incomplete ones (spec §1 + §4).

Safety: every target must live inside the HF hub dir or a configured GGUF root (after resolving symlinks),
configured roots are never deleted, models used (by path) by a non-terminal deployment or with an active
download are refused, and while a deletion runs the affected refs/paths are published in ``deleting`` so the
deployer and download service refuse to start anything on them. Root-owned files are removed by a short-lived
helper container (see DockerController.run_helper); when that is unavailable we fall back to in-process deletion.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sqlalchemy import select

from dgxmm.audit import audit
from dgxmm.db import Database, Deployment, Download
from dgxmm.docker_ctl import HELPER_MOUNT
from dgxmm.inventory.artifact import REPO_RE

log = logging.getLogger(__name__)
TERMINAL_STATES = frozenset({"stopped", "failed", "exited", "removed"})
DEFAULT_HELPER_IMAGE = "dgxmm:latest"
RECENT_PARTIAL_S = 600  # a .incomplete blob written this recently may belong to an untracked download
_PATH_ERRORS = (OSError, RuntimeError, ValueError)  # RuntimeError: symlink loop in Path.resolve()


class DeleteError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


@dataclass
class Target:
    path: Path        # where it lives (parent resolved, final component not followed)
    is_dir: bool
    bytes: int        # bytes actually freed by removing it (0 for a symlink)


@dataclass
class DeletePlan:
    ref: str
    source: str
    name: str
    targets: list[Target]
    blocking: list[dict] = field(default_factory=list)
    affected: list[dict] = field(default_factory=list)
    download_active: bool = False
    recently_active: bool = False
    rmdir_if_empty: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(t.bytes for t in self.targets)


def _unsafe(p, why: str) -> DeleteError:
    return DeleteError("unsafe_path", f"refusing to delete {p}: {why}")


def _resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except _PATH_ERRORS as e:
        raise _unsafe(p, f"cannot resolve ({e})")


def _tree_bytes(p: Path) -> int:
    """Bytes of regular files under p, never following symlinks (a symlink itself frees ~nothing)."""
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
    def __init__(self, settings, db: Database, inventory, docker=None, deployer=None, downloads=None):
        self.settings, self.db, self.inventory, self.docker = settings, db, inventory, docker
        self._lock = threading.Lock()
        self.deleting: set[str] = set()  # refs and target paths of the deletion in progress
        self.deployer = deployer
        for svc in (deployer, downloads):
            if svc is not None:
                svc.delete_guard = self.is_being_deleted

    # ------------------------------------------------------------ paths
    @property
    def hub(self) -> Path:
        return Path(self.settings.hf_cache.local) / "hub"

    def _roots(self) -> list[Path]:
        """Directories targets may live in."""
        return [_resolve(self.hub)] + [_resolve(Path(m.local)) for m in self.settings.gguf_roots]

    def _protected(self) -> list[Path]:
        """Directories that must never be deleted (nor any of their ancestors)."""
        hub = _resolve(self.hub)
        return self._roots() + [_resolve(Path(self.settings.hf_cache.local)), hub / ".locks"]

    def _check(self, p: Path) -> Path:
        """Return p with its parent resolved; refuse anything outside the roots or a root itself."""
        if ".." in p.parts:
            raise _unsafe(p, "path traversal")
        loc = _resolve(p.parent) / p.name
        real = _resolve(p)
        roots = self._roots()
        for q in (loc, real):
            if not any(q.is_relative_to(r) for r in roots):
                raise _unsafe(p, "outside the model directories")
            if any(r == q or r.is_relative_to(q) for r in self._protected()):
                raise _unsafe(p, "it is a configured model root")
        return loc

    def _mappings(self) -> list[tuple[Path, str]]:
        ms = [self.settings.hf_cache, *self.settings.gguf_roots]
        return sorted(((_resolve(Path(m.local)), m.host) for m in ms), key=lambda x: len(str(x[0])), reverse=True)

    def to_host(self, p: Path) -> str | None:
        for local, host in self._mappings():
            if p == local or p.is_relative_to(local):
                rel = p.relative_to(local)
                return str(PurePosixPath(host) / rel) if str(rel) != "." else str(PurePosixPath(host))
        return None

    def _host_or_local(self, p: Path) -> str:
        return self.to_host(p) or str(p)

    def _repo_of(self, loc: Path) -> str | None:
        """HF repo id a path inside the hub belongs to (for the download check)."""
        hub = _resolve(self.hub)
        if not loc.is_relative_to(hub):
            return None
        parts = loc.relative_to(hub).parts
        if parts and parts[0].startswith("models--") and parts[0].count("--") >= 2:
            bits = parts[0].split("--")
            return f"{bits[1]}/{'--'.join(bits[2:])}"
        return None

    def _ref_paths(self, ref: str | None) -> set[Path]:
        """Filesystem locations a model ref points at: link location and resolved file, or the HF repo dir."""
        if not ref:
            return set()
        out: set[Path] = set()
        try:
            if ref.startswith("/"):
                p = Path(ref)
                out.add(p.parent.resolve() / p.name)
                out.add(p.resolve())
            elif re.fullmatch(REPO_RE, ref) and ".." not in ref:
                d = self.hub / ("models--" + ref.replace("/", "--"))
                out.add(d.parent.resolve() / d.name)
        except _PATH_ERRORS:
            pass
        return out

    # ------------------------------------------------------------ guard
    def is_being_deleted(self, ref: str) -> bool:
        """True if ref is (or lies inside / resolves into) a deletion in progress."""
        live = set(self.deleting)
        if not live:
            return False
        if ref in live:
            return True
        zones = [Path(z) for z in live if z.startswith("/")]
        return any(p == z or p.is_relative_to(z) for p in self._ref_paths(ref) for z in zones)

    # ------------------------------------------------------------ planning
    def _deployments(self, ref: str, targets: list[Target]) -> tuple[list[dict], list[dict]]:
        zones = [t.path for t in targets]
        blocking, affected = [], []
        with self.db.session() as s:
            for d in s.scalars(select(Deployment).order_by(Deployment.id)):
                spec = (d.config or {}).get("speculative")
                draft = spec.get("model") if isinstance(spec, dict) else None
                hit = ref in (d.model_ref, draft) or any(
                    p == z or p.is_relative_to(z)
                    for r in (d.model_ref, draft) for p in self._ref_paths(r) for z in zones)
                if not hit:
                    continue
                row = {"id": d.id, "name": d.name, "state": d.state}
                launching = self.deployer is not None and d.id in dict(getattr(self.deployer, "launching", {}))
                if launching:
                    row["state"] = "launching"  # container being (re)started; DB row may still say stopped
                (affected if d.state in TERMINAL_STATES and not launching else blocking).append(row)
        return blocking, affected

    def _download_active(self, repo: str | None) -> bool:
        if not repo:
            return False
        with self.db.session() as s:
            return s.scalar(select(Download.id).where(Download.repo_id == repo,
                                                      Download.state.in_(("queued", "running")))) is not None

    @staticmethod
    def _recent_partial(repo_dir: Path) -> bool:
        cutoff = time.time() - RECENT_PARTIAL_S
        try:
            return any(b.lstat().st_mtime > cutoff for b in (repo_dir / "blobs").glob("*.incomplete"))
        except OSError:
            return False

    def _blob_shared(self, blob: Path, repo_dir: Path, excluding: set[Path]) -> bool:
        """Does any other snapshot entry of the repo resolve to this blob?"""
        for root, _dirs, files in os.walk(repo_dir / "snapshots"):
            for f in files:
                p = Path(root) / f
                if p in excluding or not p.is_symlink():
                    continue
                try:
                    if p.resolve() == blob:
                        return True
                except _PATH_ERRORS:
                    continue
        return False

    def plan(self, ref: str) -> DeletePlan:
        art = self.inventory.get(ref)
        if art is None:
            raise DeleteError("not_found", f"{ref} is not in the inventory", 404)
        try:
            return self._plan(ref, art)
        except _PATH_ERRORS as e:
            raise _unsafe(ref, f"unreadable path ({e})")

    def _plan(self, ref: str, art) -> DeletePlan:
        targets: list[Target] = []
        notes: list[str] = []
        rmdir = None
        recent = False
        if art.source == "hf":
            if ".." in ref or not re.fullmatch(REPO_RE, ref):
                raise DeleteError("invalid_ref", f"{ref!r} is not a valid repo id")
            dirname = "models--" + ref.replace("/", "--")
            repo_dir = self.hub / dirname
            if not os.path.lexists(repo_dir):
                raise DeleteError("not_found", f"{ref} is not in the HF cache", 404)
            loc = self._check(repo_dir)
            is_link = loc.is_symlink()
            targets.append(Target(loc, not is_link, _tree_bytes(loc)))
            if is_link:
                notes.append("the repo folder is a symlink; only the link is removed")
            locks = self.hub / ".locks" / dirname
            if os.path.lexists(locks):
                lloc = self._check(locks)
                targets.append(Target(lloc, lloc.is_dir() and not lloc.is_symlink(), _tree_bytes(lloc)))
            recent = not is_link and self._recent_partial(loc)
            repo, name = ref, ref.split("/")[-1]
        else:
            first = Path(ref)
            if not first.is_absolute():
                raise _unsafe(ref, "not an absolute path")
            shards = [first.with_name(f) for f in art.files] or [first]
            if first not in shards:
                shards.insert(0, first)
            hub = _resolve(self.hub)
            locs = []
            for sh in shards:
                if not os.path.lexists(sh):
                    continue  # missing shard of an incomplete split GGUF
                loc = self._check(sh)
                if loc.is_dir() and not loc.is_symlink():
                    raise _unsafe(sh, "not a file")
                locs.append(loc)
            for loc in locs:
                if not loc.is_symlink():
                    targets.append(Target(loc, False, _tree_bytes(loc)))
                    continue
                targets.append(Target(loc, False, 0))
                real = loc.resolve()
                repo_of = self._repo_of(loc)
                # HF cache snapshot entry: the blob it points at holds the bytes — delete it unless shared
                if real.is_file() and real.is_relative_to(hub) and real.parent.name == "blobs" \
                        and repo_of is not None and self._repo_of(real) == repo_of:
                    repo_dir = hub / ("models--" + repo_of.replace("/", "--"))
                    if self._blob_shared(real, repo_dir, set(locs)):
                        notes.append(f"{loc.name}: its data is shared with other files in this repo, "
                                     "so only the link is removed (0 bytes freed)")
                    else:
                        targets.append(Target(self._check(real), False, real.stat().st_size))
            parent = _resolve(first.parent)
            if not parent.is_relative_to(hub):
                rmdir = parent
            repo, name = self._repo_of(_resolve(first.parent) / first.name), first.name
        if not targets:
            raise DeleteError("not_found", f"nothing left on disk for {ref}", 404)
        blocking, affected = self._deployments(ref, targets)
        return DeletePlan(ref=ref, source=art.source, name=name, targets=targets, blocking=blocking,
                          affected=affected, download_active=self._download_active(repo), recently_active=recent,
                          rmdir_if_empty=rmdir, notes=notes)

    def _view(self, plan: DeletePlan) -> dict:
        return {"ref": plan.ref, "source": plan.source, "name": plan.name,
                "paths": [self._host_or_local(t.path) for t in plan.targets], "bytes": plan.bytes,
                "deployments_blocking": plan.blocking, "deployments_affected": plan.affected,
                "download_active": plan.download_active, "recently_active": plan.recently_active, "notes": plan.notes,
                "blocked": bool(plan.blocking or plan.download_active)}

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
            failed = {x for x, _ in errors}
            raise DeleteError("delete_failed", f"could not delete {host}: {e.strerror or e}. "
                              f"The files are probably owned by root. Fix on the host: {hint}", 500,
                              hint=hint, paths=[self._host_or_local(x) for x in failed],
                              removed=[self._host_or_local(t.path) for t in plan.targets
                                       if t.path not in failed and not os.path.lexists(t.path)])
        if plan.rmdir_if_empty is not None:
            try:
                self._check(plan.rmdir_if_empty)
                if plan.rmdir_if_empty.is_dir() and not any(plan.rmdir_if_empty.iterdir()):
                    plan.rmdir_if_empty.rmdir()
            except (OSError, DeleteError):
                pass
        if not remaining:
            return "helper"
        return "helper+in_process" if ran else "in_process"

    def _delete_one(self, ref: str, cleanup: bool = False) -> dict:
        """Plan, publish the in-progress markers, re-plan (fresh DB view), then delete. Caller holds the lock."""
        first = self.plan(ref)
        marks = {ref, *(str(t.path) for t in first.targets)}
        self.deleting |= marks
        try:
            plan = self.plan(ref)
            if plan.blocking:
                names = ", ".join(f"{d['name']} ({d['state']})" for d in plan.blocking)
                raise DeleteError("model_in_use", f"{ref} is used by deployment(s) that are not stopped: {names}. "
                                  "Stop them first.", 409, deployments=plan.blocking)
            if plan.download_active:
                raise DeleteError("download_active", f"{ref} is being downloaded; cancel the download first", 409)
            if cleanup and plan.recently_active:
                raise DeleteError("recently_active", f"{ref} has partial files written in the last "
                                  f"{RECENT_PARTIAL_S // 60} minutes; a download may still be running", 409)
            view = self._view(plan)
            method = self._execute(plan)
            return {"ref": ref, "freed_bytes": plan.bytes, "paths": view["paths"], "method": method,
                    "warnings": plan.affected, "notes": plan.notes}
        finally:
            self.deleting -= marks

    def _audit(self, user_id, action: str, target: str, detail: dict) -> None:
        try:
            audit(self.db, user_id, action, target, detail)
        except Exception:
            log.exception("could not write audit row %s for %s", action, target)

    def _audit_result(self, user_id, out: dict, cleanup: bool = False) -> None:
        detail = {"bytes": out["freed_bytes"], "paths": out["paths"], "method": out["method"],
                  "affected_deployments": [d["name"] for d in out["warnings"]]}
        if cleanup:
            detail["cleanup"] = True
        self._audit(user_id, "model_delete", out["ref"], detail)

    def _audit_error(self, user_id, ref: str, e: DeleteError) -> None:
        if e.code == "delete_failed":
            self._audit(user_id, "model_delete_failed", ref, {"error": e.message[:500],
                                                               "removed": e.extra.get("removed", []),
                                                               "failed": e.extra.get("paths", [])})
        else:
            self._audit(user_id, "model_delete_refused", ref, {"reason": e.code, "message": e.message[:500]})

    def _refresh(self) -> None:
        try:
            self.inventory.refresh()
        except Exception:
            log.exception("inventory refresh after delete failed")

    def delete(self, ref: str, user_id: int | None = None) -> dict:
        with self._lock:
            try:
                out = self._delete_one(ref)
            except DeleteError as e:
                self._audit_error(user_id, ref, e)
                if e.code == "delete_failed":
                    self._refresh()
                raise
            except Exception as e:  # unexpected: still audit, rescan what is left, and report cleanly
                log.exception("deleting %s failed", ref)
                err = DeleteError("delete_failed", f"deleting {ref} failed: {type(e).__name__}: {e}", 500)
                self._audit_error(user_id, ref, err)
                self._refresh()
                raise err from e
            self._audit_result(user_id, out)
            self._refresh()
            return out

    # ------------------------------------------------------------ incomplete cleanup
    def cleanup_preview(self) -> dict:
        items = []
        for a in self.inventory.list():
            if a.complete:
                continue
            item = {"ref": a.ref, "source": a.source, "problems": a.problems}
            try:
                plan = self.plan(a.ref)
                v = self._view(plan)
                reason = "model_in_use" if plan.blocking else "download_active" if plan.download_active \
                    else "recently_active" if plan.recently_active else None
                item.update(name=v["name"], paths=v["paths"], bytes=v["bytes"], notes=v["notes"],
                            deployments_blocking=plan.blocking, deployments_affected=plan.affected,
                            download_active=plan.download_active, eligible=reason is None, skip_reason=reason)
            except Exception as e:
                code, msg = (e.code, e.message) if isinstance(e, DeleteError) else ("error", f"{type(e).__name__}: {e}")
                item.update(name=a.ref.split("/")[-1], paths=[], bytes=0, notes=[], deployments_blocking=[],
                            deployments_affected=[], download_active=False, eligible=False, skip_reason=code,
                            message=msg)
            items.append(item)
        ok = [i for i in items if i["eligible"]]
        return {"items": items, "count": len(ok), "bytes": sum(i["bytes"] for i in ok)}

    def cleanup(self, refs: list[str] | None = None, user_id: int | None = None) -> dict:
        if refs is None:
            refs = [a.ref for a in self.inventory.list() if not a.complete]
        deleted, skipped, failed = [], [], []
        with self._lock:
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
                    out = self._delete_one(ref, cleanup=True)
                except DeleteError as e:
                    entry = {"ref": ref, "code": e.code, "message": e.message}
                    if e.code == "delete_failed":
                        self._audit_error(user_id, ref, e)
                        failed.append(entry)
                    else:
                        skipped.append(entry)
                    continue
                except Exception as e:  # never abort the batch half-way
                    log.exception("cleanup of %s failed", ref)
                    msg = f"{type(e).__name__}: {e}"
                    self._audit_error(user_id, ref, DeleteError("delete_failed", msg, 500))
                    failed.append({"ref": ref, "code": "error", "message": msg})
                    continue
                self._audit_result(user_id, out, cleanup=True)  # audit as it happens
                deleted.append(out)
            self._refresh()
        return {"deleted": deleted, "skipped": skipped, "failed": failed,
                "freed_bytes": sum(d["freed_bytes"] for d in deleted)}
