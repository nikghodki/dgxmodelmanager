#!/usr/bin/env python3
"""Collect repo metadata for the parser / kind corpora -- config / tokenizer / chat-template files only, never
weights -- into a private download directory; export_corpus.py then turns it into fixtures.

    python -I scripts/decider_eval/fetch_corpus.py hub tests/fixtures/decider_corpus/repos.txt ~/eval/corpus
    python -I scripts/decider_eval/fetch_corpus.py local ~/.cache/huggingface/hub ~/eval/ondisk

`hub` downloads with huggingface_hub (anonymous: gated repos only yield their file list). `local` copies the same
files from the refs/main snapshot of every model already in an HF cache. Downloaded files are untrusted data: they
are copied and parsed as JSON / text, never executed.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

WANT = ("config.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json", "model_index.json",
        "sentence_bert_config.json", "modules.json", "decision_config.json", "decider_config.json",
        "preprocessor_config.json", "config_sentence_transformers.json", "params.json", "adapter_config.json")
MAX_FILE = 16 * 1024 * 1024


def hub(list_file: Path, out: Path) -> int:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    repos = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    for r in repos:
        d = out / r.replace("/", "--")
        if (d / "files.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            files = api.list_repo_files(r)
        except Exception as e:
            print(r, "list failed:", type(e).__name__)
            continue
        (d / "files.json").write_text(json.dumps(files))
        got = []
        for f in WANT:
            if f not in files:
                continue
            try:
                p = hf_hub_download(r, f, cache_dir=str(out / "_cache"))
            except Exception as e:  # gated: the file list is still useful for the kind rules
                print(r, f, type(e).__name__)
                continue
            if os.path.getsize(p) <= MAX_FILE:
                shutil.copyfile(p, d / f)
                got.append(f)
        print(r, "ok", got)
    return 0


def local(hub_dir: Path, out: Path, publish_check: bool = True) -> int:
    for d in sorted(hub_dir.glob("models--*")):
        parts = d.name.split("--")
        ref = parts[1] + "/" + "--".join(parts[2:])
        snaps = d / "snapshots"
        main = (d / "refs" / "main").read_text().strip() if (d / "refs" / "main").is_file() else None
        cands = sorted(snaps.glob("*/"), key=lambda p: (p.name != main, -p.stat().st_mtime)) if snaps.is_dir() else []
        cands = [s for s in cands if (s / "config.json").exists()] or cands
        if not cands:
            continue
        s, t = cands[0], out / ref.replace("/", "--")
        files = sorted(str(p.relative_to(s)) for p in s.rglob("*") if (p.is_file() or p.is_symlink())
                       and "__pycache__" not in p.parts)
        if publish_check:  # only files the Hub publishes (local runs leave caches / results next to the snapshot)
            try:
                from huggingface_hub import HfApi
                hub = set(HfApi().list_repo_files(ref))
            except Exception as e:
                print(ref, "skipped: not a public Hub repo", type(e).__name__)
                continue
            files = [f for f in files if f in hub]
        t.mkdir(parents=True, exist_ok=True)
        (t / "files.json").write_text(json.dumps(files))
        for f in WANT:
            if (s / f).exists() and os.path.getsize(s / f) <= MAX_FILE:
                shutil.copyfile(s / f, t / f)
        print(ref, s.name[:8], len(files), "files")
    return 0


if __name__ == "__main__":
    cmd, a, b = sys.argv[1], Path(sys.argv[2]).expanduser(), Path(sys.argv[3]).expanduser()
    raise SystemExit(hub(a, b) if cmd == "hub" else local(a, b))
