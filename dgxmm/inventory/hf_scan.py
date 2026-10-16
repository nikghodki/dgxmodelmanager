"""Scan a huggingface_hub cache directory (…/hub) without network access."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.inventory.artifact import ModelArtifact

log = logging.getLogger(__name__)
_LLM_SUFFIX = re.compile(r"(CausalLM|ConditionalGeneration|LMHeadModel)$")
_NON_LLM_PREFIX = ("Whisper", "Bart", "MBart", "T5", "MT5", "Pegasus", "Marian", "Speech", "Wav2Vec",
                   "Seamless", "Blip", "Clip", "Siglip", "Encodec")
_DRAFT_HINTS = ("Draft", "Eagle", "DFlash", "Medusa")


def classify_kind(arch: str | None) -> str:
    if not arch:
        return "other"
    if any(h in arch for h in _DRAFT_HINTS):
        return "draft"
    if _LLM_SUFFIX.search(arch) and not arch.startswith(_NON_LLM_PREFIX):
        return "llm"
    return "other"


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
    weights = _weights(sizes)
    fmt = "safetensors" if any(k.endswith(".safetensors") for k in sizes) else "other"
    architecture = arch.architecture if arch else None
    return ModelArtifact(
        ref=ref, source="hf", local_path=str(snap), revision=snap.name, format=fmt, weights_bytes=weights,
        size_bytes=sum(sizes.values()), complete=arch is not None and "download in progress or interrupted" not in problems,
        kind=classify_kind(architecture), arch=arch, architecture=architecture,
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
