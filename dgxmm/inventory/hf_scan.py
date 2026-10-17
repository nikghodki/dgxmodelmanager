"""Scan a huggingface_hub cache directory (…/hub) without network access."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.inventory.kinds import artifact_kind, classify_kind  # noqa: F401  (classify_kind re-exported)

log = logging.getLogger(__name__)
DOWNLOAD_MARKER = ".dgxmm-download.json"  # written by dgxmm.downloads while a download has not finished
MAX_MARKER_BYTES = 1024 * 1024


def _pending_files(repo_dir: Path, snap: Path, sizes: dict[str, int]) -> tuple[list[str], int]:
    """Files an unfinished DGXMM download of this snapshot lists that are not (yet) in it: (missing, listed).
    A marker for a different revision, or an unreadable one, does not apply to this snapshot."""
    m = repo_dir / DOWNLOAD_MARKER
    try:
        if not m.is_file() or m.is_symlink() or m.stat().st_size > MAX_MARKER_BYTES:
            return [], 0
        data = json.loads(m.read_text())
    except (OSError, ValueError):
        return [], 0
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        return [], 0
    rev = data.get("revision")
    if isinstance(rev, str) and len(rev) == 40 and rev != snap.name:
        return [], 0
    files = [f for f in data["files"] if isinstance(f, str)]
    return [f for f in files if f not in sizes], len(files)


def _resolve_size(p: Path) -> int | None:
    try:
        return p.stat().st_size  # follows symlinks; raises when dangling
    except OSError:
        return None


def _inspect_snapshot(snap: Path) -> tuple[dict | None, dict[str, int], list[str]]:
    sizes: dict[str, int] = {}
    broken: list[str] = []
    for root, _dirs, files in os.walk(snap):
        for f in files:
            p = Path(root) / f
            rel = str(p.relative_to(snap))
            size = _resolve_size(p)
            if size is None:
                broken.append(rel)
            else:
                sizes[rel] = size
    config = None
    if "config.json" in sizes:
        try:
            config = json.loads((snap / "config.json").read_text())
        except (OSError, ValueError):
            broken.append("config.json (unreadable)")
    return config, sizes, broken


def _weights(sizes: dict[str, int]) -> int:
    st = {k: v for k, v in sizes.items() if k.endswith(".safetensors")}
    if any(k.startswith("model") for k in st):
        st = {k: v for k, v in st.items() if not k.startswith("consolidated")}
    if st:
        return sum(st.values())
    return sum(v for k, v in sizes.items() if k.endswith((".bin", ".pt", ".pth")) and "training_args" not in k)


def _scan_repo(d: Path) -> ModelArtifact | None:
    parts = d.name.split("--")
    ref = f"{parts[1]}/{'--'.join(parts[2:])}"
    snaps_dir = d / "snapshots"
    snaps = sorted((p for p in snaps_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) \
        if snaps_dir.is_dir() else []
    main_rev = (d / "refs" / "main").read_text().strip() if (d / "refs" / "main").is_file() else None
    if main_rev:
        snaps.sort(key=lambda p: p.name != main_rev)  # stable: main first, then newest
    problems: list[str] = []
    if any(b.name.endswith(".incomplete") for b in (d / "blobs").glob("*")) if (d / "blobs").is_dir() else False:
        problems.append("download in progress or interrupted")
    chosen = None
    for snap in snaps:
        config, sizes, broken = _inspect_snapshot(snap)
        if config is not None and not broken and _weights(sizes) > 0:
            chosen = (snap, config, sizes)
            break
        if snap.name == main_rev:
            problems.append(f"refs/main snapshot {snap.name[:8]} is incomplete ({len(broken)} broken links)")
    if chosen is None:
        if any(any(f.lower().endswith(".gguf") for f in os.listdir(s)) for s in snaps if s.is_dir()):
            return None  # GGUF repo: reported by scan_gguf_roots instead
        if not snaps:
            problems.append("no snapshot")
        elif not problems or all("refs/main" not in p for p in problems):
            problems.append("no complete snapshot (missing config.json or weights)")
        return ModelArtifact(ref=ref, source="hf", local_path=str(d), revision=None, format="other",
                             weights_bytes=0, size_bytes=0, complete=False, kind="other", arch=None,
                             architecture=None, quant=None, problems=problems)
    snap, config, sizes = chosen
    try:
        arch = parse_hf_config(config)
    except Exception as e:  # never let one odd config kill the scan
        log.warning("cannot parse %s config: %s", ref, e)
        arch = None
        problems.append(f"config.json not understood: {e}")
    missing, listed = _pending_files(d, snap, sizes)
    if missing:
        problems.append(f"download not finished: {len(missing)} of {listed} files missing ({missing[0]}"
                        + (", …)" if len(missing) > 1 else ")"))
    weights = _weights(sizes)
    fmt = "safetensors" if any(k.endswith(".safetensors") for k in sizes) else "other"
    architecture = arch.architecture if arch else None
    complete = arch is not None and "download in progress or interrupted" not in problems and not missing
    return ModelArtifact(
        ref=ref, source="hf", local_path=str(snap), revision=snap.name, format=fmt, weights_bytes=weights,
        size_bytes=sum(sizes.values()), complete=complete,
        kind=artifact_kind(architecture, config, sorted(sizes), ref), arch=arch, architecture=architecture,
        quant=arch.quant if arch else None, problems=problems, config=config, files=sorted(sizes),
    )


def scan_hf_cache(hub_dir: Path) -> list[ModelArtifact]:
    if not hub_dir.is_dir():
        return []
    out = []
    for d in sorted(hub_dir.iterdir()):
        if d.is_dir() and d.name.startswith("models--") and d.name.count("--") >= 2:
            try:
                a = _scan_repo(d)
                if a is not None:
                    out.append(a)
            except OSError as e:
                log.warning("skipping %s: %s", d, e)
    return out
