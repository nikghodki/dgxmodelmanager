#!/usr/bin/env python3
"""End-to-end check of the deploy-time advice endpoints on real data: the real FastAPI app and the real ``Decider``
answering through the live decision model, the real HF cache (read-only), the real Hub (anonymous), and this host's
memory — without touching the running DGXMM (own temporary database, no Docker access).

Run in a throwaway container on the decider's network, e.g.:

    docker run --rm --name adv-e2e --network <decider network> --entrypoint python3 \\
        -v <worktree>:/w:ro -v <HF cache>:/hf:ro -v <models dir>:/models:ro -v <decider key>:/key:ro -v <out>:/out \\
        -w /w <dgxmm image> scripts/decider_eval/live_deploy_check.py --base http://<decider>:<port> --key-file /key

Prints (and writes to --out) a JSON summary per call: status, source, confidence, verdicts / choices, latency.
The key is read at request time and never printed."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dgxmm.auth.ratelimit import SlidingWindowLimiter  # noqa: E402
from dgxmm.config import GIB, PathMapping, Settings  # noqa: E402
from dgxmm.context import AppContext  # noqa: E402
from dgxmm.db import Database, User  # noqa: E402
from dgxmm.decider.service import Decider  # noqa: E402
from dgxmm.events import EventBroker  # noqa: E402
from dgxmm.memory.hostmem import HostMemory, HostSample  # noqa: E402
from dgxmm.settings_store import SettingsStore  # noqa: E402


class LiveProbe:
    """This host's memory from /proc/meminfo (no managed containers in the temporary database)."""

    def meminfo(self) -> HostMemory:
        kv = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            kv[k] = int(v.split()[0]) * 1024
        return HostMemory(total=kv["MemTotal"], available=kv["MemAvailable"], free=kv["MemFree"],
                          cached=kv.get("Cached", 0), swap_total=kv.get("SwapTotal", 0), swap_free=kv.get("SwapFree", 0))

    def sample(self, ids):
        return HostSample(mem=self.meminfo(), managed={c: 0 for c in ids}, external_top=[], gpu={})

    def processes(self, managed=None, names=None):
        return []


class LiveDecider(Decider):
    """The real Decider (thresholds, logging, validation), pointed straight at the decider container."""

    def __init__(self, base: str, **kw):
        super().__init__(**kw)
        self._base = base

    def _base_url(self) -> str | None:
        return self._base

    def healthy(self) -> bool:
        c = self.client()
        return c is not None and not c.breaker.is_open()


def build(args, tmp: Path):
    from dgxmm.catalog.hub import HfHubClient
    from dgxmm.deployer.service import Deployer
    from dgxmm.engines.registry import build_engines
    from dgxmm.inventory.service import InventoryService
    from dgxmm.memory.budget import BudgetService
    for d in ("data", "static"):
        (tmp / d).mkdir()
    settings = Settings(data_dir=str(tmp / "data"), static_dir=str(tmp / "static"),
                        hf_cache=PathMapping(host="/hf", local=args.hf), gguf_roots=[PathMapping(host="/models",
                                                                                                 local=args.models)],
                        hf_token_file=str(tmp / "no-token"), decider_key_file=args.key_file)
    db = Database(f"sqlite:///{tmp / 'data' / 'e2e.sqlite3'}")
    events = EventBroker()
    inv = InventoryService(settings)
    inv.refresh()
    archs = set(json.loads((ROOT / "tests" / "fixtures" / "decider_corpus" / "vllm_archs.json").read_text()))
    arch_src = type("Archs", (), {"get": staticmethod(lambda: archs)})()
    engines = build_engines(settings)
    engines["vllm"].arch_source = arch_src.get
    http = httpx.AsyncClient(timeout=30)
    deployer = Deployer(settings=settings, db=db, docker=None, inventory=inv, engines=engines, http=http,
                        events=events)
    deployer.budget = BudgetService(settings, LiveProbe(), deployer.usage_rows)
    store = SettingsStore(db, settings)
    decider = LiveDecider(args.base, db=db, settings=settings, store=store, http=lambda: http)
    decider.set_enabled(True)
    ctx = AppContext(settings=settings, db=db, events=events, login_limiter=SlidingWindowLimiter(100, 60),
                     inventory=inv, budget=deployer.budget, deployer=deployer, engines=engines, settings_store=store,
                     decider=decider)
    ctx.hub, ctx.archs = HfHubClient(token=None), arch_src
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--hf", default="/hf")
    ap.add_argument("--models", default="/models")
    ap.add_argument("--out", default="/out/e2e.json")
    args = ap.parse_args()
    from dgxmm.app import create_app
    from dgxmm.auth.passwords import hash_password
    out = []
    with tempfile.TemporaryDirectory(prefix="adv-a1-e2e-") as t:
        ctx = build(args, Path(t))
        pw = secrets.token_urlsafe(18)
        with ctx.db.session() as s:
            s.add(User(username="adv-a1", pw_hash=hash_password(pw), role="admin"))
        with TestClient(create_app(ctx), base_url="http://e2e") as c:
            bp = ctx.settings.base_path
            h = {"X-CSRF-Token": c.post(f"{bp}/api/auth/login", json={"username": "adv-a1",
                                                                       "password": pw}).json()["csrf_token"]}
            calls = [("GET", "/api/catalog/compat?repo=Qwen/Qwen3-0.6B", None),
                     ("GET", "/api/catalog/compat?repo=Qwen/Qwen3-0.6B", None),  # cached
                     ("GET", "/api/catalog/compat?repo=unsloth/Qwen3-0.6B-GGUF", None),
                     ("GET", "/api/catalog/compat?repo=microsoft/Phi-3-mini-4k-instruct", None),
                     ("GET", "/api/catalog/compat?repo=meta-llama/Llama-3.1-8B-Instruct", None),  # gated, no token
                     ("GET", "/api/catalog/compat?repo=Qwen/Qwen3.8-Flash-Next", None),  # not in the image's list
                     ("POST", "/api/decider/config-advice", {"model_ref": "Qwen/Qwen3-8B", "engine": "vllm",
                                                             "goal": "long_context"}),
                     ("POST", "/api/decider/config-advice", {"model_ref": "Qwen/Qwen3-8B", "engine": "vllm",
                                                             "goal": "agent_tool_use"}),
                     ("POST", "/api/decider/draft-suggest", {"target_ref": "Qwen/Qwen3-8B", "method": "draft_model"}),
                     ("POST", "/api/decider/draft-suggest", {"target_ref": "Qwen/Qwen3.8-27B", "method": "dflash"}),
                     ("POST", "/api/decider/engine-suggest", {"model_ref": "Qwen/Qwen3-0.6B", "goal": "chat"}),
                     ("POST", "/api/decider/engine-suggest", {"model_ref": "Qwen/Qwen3-8B", "goal": "batch_throughput"}),
                     ("POST", "/api/decider/remote-code-check", {"model_ref": "vllm-sr/d3-mini"}),
                     ("POST", "/api/decider/remote-code-check", {"model_ref": "Qwen/Qwen3-8B"})]
            for method, path, body in calls:
                t0 = time.monotonic()
                r = c.request(method, f"{bp}{path}", json=body, headers=h if method == "POST" else None)
                ms = round((time.monotonic() - t0) * 1000)
                b = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                keep = {k: b.get(k) for k in ("source", "confidence", "cached", "gated", "error", "recommended", "top",
                                              "suggested", "engine", "model_ref", "reason", "verdict", "model_label")
                        if k in b}
                if "vllm" in b:
                    keep["vllm"], keep["llamacpp"] = b["vllm"], b["llamacpp"]
                if "options" in b:
                    keep["options"] = [o["summary"] for o in b["options"]]
                if "candidates" in b:
                    keep["candidates"] = [x["ref"] for x in b["candidates"]]
                if "findings" in b:
                    keep["findings"] = sorted({f["kind"] for f in b["findings"]})
                out.append({"call": f"{method} {path}", "body": body, "status": r.status_code, "ms": ms, **keep})
                print(json.dumps(out[-1]))
        with ctx.db.session() as s:
            from sqlalchemy import func, select
            from dgxmm.db import DecisionRecord
            logged = dict(s.execute(select(DecisionRecord.feature, func.count()).group_by(DecisionRecord.feature)).all())
        print(json.dumps({"decisions_logged": logged}))
        out.append({"decisions_logged": logged})
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
