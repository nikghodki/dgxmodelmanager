"""llmfit (pip package `llmfit`) as a scoring/discovery source. Never the OOM gate."""
from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Callable


class LlmfitError(Exception):
    pass


class Llmfit:
    def __init__(self, bin: str = "llmfit", runner: Callable = subprocess.run, ttl: float = 600,
                 clock: Callable[[], float] = time.monotonic, timeout: float = 60):
        self.bin, self.runner, self.ttl, self.clock, self.timeout = bin, runner, ttl, clock, timeout
        self._cache: dict[tuple, tuple[float, Any]] = {}

    def _run(self, args: list[str]) -> Any:
        key = tuple(args)
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < self.ttl:
            return hit[1]
        try:
            proc = self.runner([self.bin, "--json", *args], capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise LlmfitError(str(e)) from e
        if proc.returncode != 0:
            raise LlmfitError((proc.stderr or "").strip()[:300] or f"exit code {proc.returncode}")
        try:
            data = json.loads(proc.stdout)
        except ValueError as e:
            raise LlmfitError("llmfit returned invalid JSON") from e
        self._cache[key] = (self.clock(), data)
        return data

    @staticmethod
    def _models(data: Any) -> list[dict]:
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
        if isinstance(data, dict):
            for k in ("models", "results", "data"):
                if isinstance(data.get(k), list):
                    return [m for m in data[k] if isinstance(m, dict)]
        return []

    @staticmethod
    def _mem(free_gib: float, ctx: int) -> list[str]:
        g = f"{max(free_gib, 1):.0f}G"
        return ["--memory", g, "--ram", g, "--max-context", str(ctx)]

    def recommend(self, free_gib: float, ctx: int, limit: int = 50) -> list[dict]:
        return self._models(self._run(self._mem(free_gib, ctx) + ["recommend", "--limit", str(limit)]))

    def search_map(self, query: str, free_gib: float, ctx: int) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if query:
            for m in self._models(self._run(self._mem(free_gib, ctx) + ["search", query])):
                if m.get("name"):
                    out[m["name"].lower()] = dict(m)
        for m in self.recommend(free_gib, ctx):
            if m.get("name"):
                out.setdefault(m["name"].lower(), {}).update(m)
        return out
