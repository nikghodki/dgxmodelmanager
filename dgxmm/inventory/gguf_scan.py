"""Find .gguf files under configured roots and inside HF cache snapshots."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.inventory.gguf_meta import archspec_from_gguf, read_gguf_meta

log = logging.getLogger(__name__)
_SHARD = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)
_META_CACHE: dict[tuple[str, float, int], dict] = {}


def _meta(p: Path) -> dict:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key not in _META_CACHE:
        _META_CACHE[key] = read_gguf_meta(p)
    return _META_CACHE[key]


def _iter_gguf(root: Path):
    if not root.is_dir():
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".gguf"):
                yield Path(dirpath) / f


def scan_gguf_roots(roots: list[Path]) -> list[ModelArtifact]:
    out: list[ModelArtifact] = []
    for root in roots:
        files = sorted(_iter_gguf(root))
        names = {p.name: p for p in files}
        for p in files:
            m = _SHARD.match(p.name)
            if m and m.group("idx") != "00001":
                continue  # later shards are folded into the first one
            problems: list[str] = []
            shard_paths = [p]
            if m:
                n = int(m.group("n"))
                shard_paths = [p.with_name(f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
                missing = [s.name for s in shard_paths if s.name not in names or not s.exists()]
                if missing:
                    problems.append(f"missing shards: {', '.join(missing)}")
            try:
                sizes = [s.stat().st_size for s in shard_paths if s.exists()]
            except OSError:
                sizes = []
            if not p.exists():
                out.append(ModelArtifact(ref=str(p), source="gguf", local_path=str(p), revision=None, format="gguf",
                                         weights_bytes=0, size_bytes=0, complete=False, kind="other", arch=None,
                                         architecture=None, quant=None, problems=["broken symlink"]))
                continue
            arch = None
            try:
                arch = archspec_from_gguf(_meta(p))
            except Exception as e:
                problems.append(f"cannot read GGUF header: {e}")
            kind = "other" if "mmproj" in p.name.lower() else ("llm" if arch else "other")
            out.append(ModelArtifact(ref=str(p), source="gguf", local_path=str(p), revision=None, format="gguf",
                                     weights_bytes=sum(sizes), size_bytes=sum(sizes),
                                     complete=arch is not None and not problems, kind=kind, arch=arch,
                                     architecture=arch.architecture if arch else None,
                                     quant=arch.quant if arch else None, problems=problems,
                                     files=[s.name for s in shard_paths]))
    return out
