"""Delete downloaded weights and clean up incomplete ones (spec §1 + §4).

Safety: every target must live inside the HF hub dir or a configured GGUF root (after resolving symlinks),
configured roots are never deleted, models used (by path) by a non-terminal deployment or with an active
download are refused, and while a deletion runs the affected refs/paths are published in ``deleting`` so the
deployer and download service refuse to start anything on them. Root-owned files are removed by a short-lived
helper container (see DockerController.run_helper); when that is unavailable we fall back to in-process deletion.

Shared blob store (huggingface_hub >= 2): repo blobs ``models--org--name/blobs/<etag>`` may be symlinks into the
cache-wide store ``hub/blobs/<xx>/<xet hash>``, which deduplicates identical files across repos. A delete also
removes the store entries that ONLY the deleted files link to. Which entries those are is computed read-only
before anything is removed, by scanning every symlink in the cache (and the GGUF roots) — the store's own
``<hash>.refs`` manifest is only a hint and is never trusted to allow a removal. Right before an entry is removed
the deleter takes the entry's ``<hash>.lock`` (the lock huggingface_hub holds while it links an entry into a repo
and records the link in the manifest) and re-reads the manifest, so an entry a concurrent download has just
reused is kept.
"""
from __future__ import annotations

import errno
import fcntl
import logging
import os
import re
import shutil
import stat
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
STORE_DIR = "blobs"                       # hub/blobs: huggingface_hub's shared blob store
_XET_HASH = re.compile(r"[0-9a-f]{64}")   # store entry names
STORE_LOCK_TIMEOUT_S = 10.0


class DeleteError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


@dataclass
class Target:
    path: Path        # where it lives (parent resolved, final component not followed)
    is_dir: bool
    bytes: int        # bytes actually freed by removing it (0 for a symlink)
    store: bool = False  # an entry of the shared blob store (removed last, under its lock, if still unreferenced)


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


def _hop(link: Path) -> Path:
    """Where a symlink points, one level only (normalised, not resolved further)."""
    t = os.readlink(link)
    return Path(os.path.normpath(t if os.path.isabs(t) else link.parent / t))


class _StoreLock:
    """flock on a store entry's ``<hash>.lock`` — the lock huggingface_hub (filelock) takes while it links an
    entry into a repo. Never follows a planted symlink. ``held`` is False when the lock could not be taken (a
    root-owned prefix dir we cannot create the file in, or a timeout)."""

    def __init__(self, entry: Path, timeout: float | None = None):
        self.path, self.fd, self.held = entry.with_name(entry.name + ".lock"), None, False
        self.timeout = STORE_LOCK_TIMEOUT_S if timeout is None else timeout

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o666)
            if not stat.S_ISREG(os.fstat(self.fd).st_mode):
                return self
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.held = True
                    break
                except OSError as e:
                    if e.errno not in (errno.EAGAIN, errno.EACCES) or time.monotonic() > deadline:
                        break
                    time.sleep(0.05)
        except OSError as e:
            log.info("cannot lock store entry %s (%s)", self.path, e)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                if self.held:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
        return False


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
        return self._roots() + [_resolve(Path(self.settings.hf_cache.local)), hub / ".locks", hub / STORE_DIR]

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

    # ------------------------------------------------------------ shared blob store
    def _store(self) -> Path | None:
        """hub/blobs when it is a real directory inside the hub (never a symlink out of it)."""
        s = _resolve(self.hub) / STORE_DIR
        return s if os.path.isdir(s) and not os.path.islink(s) else None

    def _store_entry(self, link: Path, store: Path | None) -> Path | None:
        """The store entry a symlink finally resolves to, if it is a well-formed one: a regular file
        ``<store>/<xx>/<64 hex>`` whose real path stays inside the store (a prefix dir that is a symlink out of
        the store resolves elsewhere and is ignored)."""
        if store is None:
            return None
        try:
            real = Path(os.path.realpath(link, strict=True))
            st = real.lstat()
        except _PATH_ERRORS:
            return None
        if real.parent.parent != store or not _XET_HASH.fullmatch(real.name) or real.parent.name != real.name[:2]:
            return None
        return real if stat.S_ISREG(st.st_mode) else None

    def _store_links_in(self, top: Path, store: Path | None) -> dict[Path, int]:
        """Store entries that symlinks under `top` (a repo dir) resolve to, with the bytes removing them frees."""
        out: dict[Path, int] = {}
        if store is None:
            return out
        for root, dirs, files in os.walk(top):
            for n in files + dirs:
                p = Path(root) / n
                if p.is_symlink():
                    e = self._store_entry(p, store)
                    if e is not None and e not in out:
                        st = e.lstat()
                        out[e] = st.st_size if st.st_nlink == 1 else 0  # a hard link elsewhere keeps the data
        return out

    def _drop_referenced(self, candidates: dict[Path, int], skip_dirs: set[Path], skip_links: set[Path],
                         store: Path) -> tuple[dict[Path, int], bool]:
        """Remove from `candidates` every entry that a symlink outside the deletion still resolves to: every repo
        in the cache (models, datasets, spaces) and the GGUF roots. Read-only. Returns (exclusive, complete);
        when part of the tree could not be read the scan is incomplete and nothing may be removed."""
        left = dict(candidates)
        ok = [True]

        def failed(_e):
            ok[0] = False

        hub = _resolve(self.hub)
        tops = [c for c in sorted(hub.iterdir()) if c.name not in (STORE_DIR, ".locks")]
        tops += [r for r in self._roots()[1:] if not r.is_relative_to(hub)]
        for top in tops:
            if not left:
                break
            if top in skip_dirs:
                continue
            if top.is_symlink():  # a repo dir that is itself a link
                if top not in skip_links:
                    left.pop(self._store_entry(top, store), None)
                continue
            for root, dirs, files in os.walk(top, onerror=failed):
                r = Path(root)
                dirs[:] = [d for d in dirs if (r / d) not in skip_dirs and (r / d) != store]
                for n in files + dirs:
                    p = r / n
                    if p in skip_links or not p.is_symlink():
                        continue
                    e = self._store_entry(p, store)
                    if e is not None:
                        left.pop(e, None)
                if not left:
                    break
        return left, ok[0]

    def _store_targets(self, candidates: dict[Path, int], skip_dirs: set[Path], skip_links: set[Path],
                       notes: list[str]) -> list[Target]:
        store = self._store()
        if not candidates or store is None:
            return []
        exclusive, complete = self._drop_referenced(candidates, skip_dirs, skip_links, store)
        if not complete:
            notes.append("part of the model cache could not be read, so no shared-store data is removed")
            return []
        kept = len(candidates) - len(exclusive)
        if kept:
            notes.append(f"{kept} file(s) in the shared blob store are also used by other models and are kept")
        return [Target(self._check(e), False, b, store=True) for e, b in sorted(exclusive.items())]

    def _manifest_refs(self, entry: Path) -> bool:
        """Does the entry's ``<hash>.refs`` manifest name a repo blob that still links to it?"""
        hub = _resolve(self.hub)
        man = entry.with_name(entry.name + ".refs")
        try:
            if not stat.S_ISREG(man.lstat().st_mode):
                return False
            lines = man.read_text(errors="replace").splitlines()
        except OSError:
            return False
        store = self._store()
        for line in lines:
            rel = Path(line.strip())
            if not line.strip() or rel.is_absolute() or ".." in rel.parts:
                continue
            p = hub / rel
            if p.is_symlink() and self._store_entry(p, store) == entry:
                return True
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
            lloc = None
            if os.path.lexists(locks):
                lloc = self._check(locks)
                targets.append(Target(lloc, lloc.is_dir() and not lloc.is_symlink(), _tree_bytes(lloc)))
            if not is_link:
                cands = self._store_links_in(loc, self._store())
                targets += self._store_targets(cands, {loc} | ({lloc} if lloc else set()), set(), notes)
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
            store = self._store()
            cands: dict[Path, int] = {}
            for loc in locs:
                if not loc.is_symlink():
                    targets.append(Target(loc, False, _tree_bytes(loc)))
                    continue
                targets.append(Target(loc, False, 0))
                real = loc.resolve()
                repo_of = self._repo_of(loc)
                if repo_of is None or not real.is_file():
                    continue
                repo_dir = hub / ("models--" + repo_of.replace("/", "--"))
                # HF cache snapshot entry: the repo blob it points at holds the bytes, either itself or (shared
                # store) through one more link into hub/blobs — delete them unless other files use them
                blob = _hop(loc)
                if blob.parent != repo_dir / "blobs":
                    continue
                if self._blob_shared(real, repo_dir, set(locs)):
                    notes.append(f"{loc.name}: its data is shared with other files in this repo, "
                                 "so only the link is removed (0 bytes freed)")
                    continue
                bloc = self._check(blob)
                if not bloc.is_symlink():
                    targets.append(Target(bloc, False, bloc.stat().st_size))
                    continue
                targets.append(Target(bloc, False, 0))
                e = self._store_entry(bloc, store)
                if e is not None:
                    st = e.lstat()
                    cands.setdefault(e, st.st_size if st.st_nlink == 1 else 0)
            targets += self._store_targets(cands, set(), {t.path for t in targets}, notes)
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

    def _execute(self, plan: DeletePlan) -> tuple[str, list[Target]]:
        """Remove the plan's targets; shared-store entries last. Returns (method, store entries kept)."""
        files = [t for t in plan.targets if not t.store]
        ran = self._run_helpers(files)
        remaining = [t for t in files if os.path.lexists(t.path)]
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
        kept: list[Target] = []
        if not errors:  # the repo's own files are gone: now its store entries are unreferenced (unless reused)
            for t in plan.targets:
                if t.store:
                    try:
                        if not self._remove_store_entry(t):
                            kept.append(t)
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
            return "helper", kept
        return ("helper+in_process" if ran else "in_process"), kept

    def _remove_store_entry(self, t: Target) -> bool:
        """Remove one shared-store entry (and its manifest) if nothing links to it any more. Under the entry's
        lock the manifest is re-read: a download that reused the entry since the plan recorded its link there
        first. Returns False when the entry is kept."""
        try:
            if self._check(t.path) != t.path or not stat.S_ISREG(t.path.lstat().st_mode):
                return False  # replaced since the plan (a prefix dir swapped for a link, the entry for a link…)
        except (DeleteError, FileNotFoundError):
            return not os.path.lexists(t.path)
        with _StoreLock(t.path) as lock:
            if lock.fd is not None and not lock.held:
                log.warning("store entry %s is locked by another process; keeping it", t.path)
                return False
            if self._manifest_refs(t.path):
                return False
            man = t.path.with_name(t.path.name + ".refs")
            doomed = [t.path] + ([man] if os.path.lexists(man) else [])
            denied = []
            for p in doomed:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
                except PermissionError:
                    denied.append(p)
            if denied:  # root-owned prefix dir: the helper removes them while we still hold the lock
                self._run_helpers([Target(p, False, 0) for p in denied])
                left = [p for p in denied if os.path.lexists(p)]
                if left:
                    raise PermissionError(errno.EACCES, "Permission denied", str(left[0]))
        return True

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
            method, kept = self._execute(plan)
            notes = list(plan.notes)
            if kept:
                notes.append(f"{len(kept)} file(s) in the shared blob store were kept: another model started "
                             "using them while this one was being deleted")
            gone = {t.path for t in kept}
            return {"ref": ref, "freed_bytes": plan.bytes - sum(t.bytes for t in kept),
                    "paths": [p for p, t in zip(view["paths"], plan.targets) if t.path not in gone],
                    "method": method, "warnings": plan.affected, "notes": notes}
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
