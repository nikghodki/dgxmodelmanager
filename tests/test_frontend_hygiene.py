"""Compiled JavaScript must never sit next to the TypeScript sources: Vite/vitest resolve ``foo.js`` before
``foo.tsx``, so a stale emitted file silently replaces the source in tests and builds."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def emitted_files() -> list[str]:
    out = []
    for js in list((FRONTEND / "src").rglob("*.js")) + list((FRONTEND / "src").rglob("*.jsx")) + \
            [p for p in FRONTEND.glob("*.js")]:
        stem = js.with_suffix("")
        if any(stem.with_suffix(s).exists() for s in (".ts", ".tsx")):
            out.append(str(js.relative_to(FRONTEND)))
    return sorted(out)


def test_no_emitted_js_next_to_typescript_sources():
    assert emitted_files() == [], "remove compiled files (tsc must not emit): " + ", ".join(emitted_files()[:10])


def test_typescript_never_emits():
    cfg = json.loads((FRONTEND / "tsconfig.json").read_text())
    assert cfg["compilerOptions"].get("noEmit") is True


def test_emitted_files_are_gitignored():
    if not shutil.which("git") or not (FRONTEND.parent / ".git").exists():
        pytest.skip("not a git checkout")
    probes = ["frontend/src/components/app/x.js", "frontend/vite.config.js", "frontend/tsconfig.tsbuildinfo"]
    p = subprocess.run(["git", "check-ignore", *probes], cwd=FRONTEND.parent, capture_output=True, text=True)
    assert sorted(p.stdout.split()) == sorted(probes)
    tracked = subprocess.run(["git", "ls-files", "frontend"], cwd=FRONTEND.parent, capture_output=True, text=True,
                             check=True).stdout.split()
    assert [f for f in tracked if f.endswith(".tsbuildinfo")] == []
