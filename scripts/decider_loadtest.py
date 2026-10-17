"""Load test for a running decider (d3_server System One API) with payloads built by DGXMM's own feature code.

    python scripts/decider_loadtest.py http://127.0.0.1:PORT KEYFILE GOLDEN_DIR HF_CACHE [--levels 1,2,4,8]
        [--per-client 6] [--burst 16] [--json out.json]

1. Server capacity: for each concurrency level, that many clients send --per-client requests each, back to back,
   straight to the server (no client-side limiter): p50/p95/p99 latency and throughput.
2. Bursts through dgxmm's SystemOneClient as the app uses it (feature deadlines, priorities, client-side limiter):
   --burst simultaneous feature calls, once with the default limiter and once with it effectively off; per feature:
   answered / shed (rules) / timed out, and the worst latency against the feature's deadline.

Payloads: diagnosis states from real vLLM failure logs (GOLDEN_DIR/golden_diagnosis.jsonl) through
diagnosis.build_state; parser states from real chat templates in HF_CACHE through config_suggest.state_for (tool and
reasoning questions with their option lists); kind states from real config.json files through kind.build_state; plan
and workload states through plans.build_state / workload.build_state with realistic facts. The API key is read from
KEYFILE and never printed."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

from dgxmm.decider import config_suggest as cs
from dgxmm.decider import diagnosis as dg
from dgxmm.decider import kind as kd
from dgxmm.decider import plans as pl
from dgxmm.decider import workload as wl
from dgxmm.decider.client import DeciderError, SystemOneClient, criteria
from dgxmm.decider.types import Option

GIB = 1024**3
# feature -> (deadline used by the app, priority: 0 interactive / 1 background)
FEATURES = {"diagnosis": (dg.TIMEOUT_S, 1), "parser_tool": (cs.PARSER_TIMEOUT_S, 0),
            "parser_reasoning": (cs.PARSER_TIMEOUT_S, 0),
            "plan": (pl.TIMEOUT_S, 0), "kind": (2.0, 1), "workload": (2.0, 1)}


def payloads(golden_dir: Path, hf_cache: Path) -> list[tuple[str, str, str, list[Option]]]:
    """(feature, state, question, options) built with the feature modules."""
    out = []
    logs = [json.loads(line)["state"] for line in (golden_dir / "golden_diagnosis.jsonl").read_text().splitlines()]
    for i, log in enumerate(logs):
        if i % 3 == 0:  # worst case: a long tail (other real logs first), cut at the production caps
            log = "\n".join(logs[:i] + logs[i + 1:] + [log])
        state = dg.build_state(engine="vllm", model_ref="Qwen/Qwen3-8B", exit_code=1,
                               message="container exited with code 1", hits=[], log_text=log)
        out.append(("diagnosis", state, dg.QUESTION, dg.OPTIONS))
    from dgxmm.config import PathMapping, Settings
    from dgxmm.inventory.service import InventoryService
    settings = Settings(hf_cache=PathMapping(host=str(hf_cache), local=str(hf_cache)), gguf_roots=[])
    inv = InventoryService(settings)
    inv.refresh()
    arts = [a for a in inv.list() if a.source == "hf" and a.complete]
    for a in arts:
        info = cs.model_info(a)
        out.append(("parser_tool", cs.state_for(info), cs.QUESTION_TOOL, cs.tool_options()))
        out.append(("parser_reasoning", cs.state_for(info), cs.QUESTION_REASONING, cs.reasoning_options()))
        out.append(("kind", kd.build_state(a), kd.QUESTION, kd.OPTIONS))
    deps = {1: pl.DepFacts(1, "chat-inference", "ready", 0, last_traffic_min=600.0, traffic_known=True,
                           ready_for_min=900.0, uncovered_min=0),
            2: pl.DepFacts(2, "poc-traffic", "ready", 0, last_traffic_min=0.5, running_now=True, traffic_known=True,
                           ready_for_min=50.0, uncovered_min=0),
            3: pl.DepFacts(3, "coder", "sleeping", 10)}
    plan_list = [{"id": "kv", "title": "Use a smaller KV-cache dtype", "detail": "Set kv_cache_dtype=fp8.",
                  "new_required": 12 * GIB, "actions": []},
                 {"id": "sleep:1", "title": "Put chat-inference to sleep", "detail": "Frees about 30.0 GiB.",
                  "new_required": 16 * GIB, "actions": [{"type": "sleep", "deployment_id": 1}]},
                 {"id": "stop:1", "title": "Stop chat-inference", "detail": "Frees about 48.0 GiB.",
                  "new_required": 16 * GIB, "actions": [{"type": "stop", "deployment_id": 1}]},
                 {"id": "stop:3", "title": "Stop coder", "detail": "Frees about 20.0 GiB.",
                  "new_required": 16 * GIB, "actions": [{"type": "stop", "deployment_id": 3}]}]
    for short in (2, 6, 14):
        state = pl.build_state(shortfall=short * GIB, model_ref="Qwen/Qwen3-8B", engine="vllm",
                               cfg={"max_model_len": 65536, "max_num_seqs": 8, "kv_cache_dtype": "auto"}, deps=deps)
        out.append(("plan", state, pl.QUESTION, [pl._option(p, deps) for p in plan_list]))
    summaries = [
        {"minutes": 60, "idle_fraction": 0.1, "mean_running": 0.6, "peak_running": 2, "mean_waiting": 0.0,
         "prompt_tps": 900.0, "gen_tps": 280.0, "prompt_gen_ratio": 3.2, "prefix_hit_rate": 0.91, "ttft_p50": 0.04},
        {"minutes": 60, "idle_fraction": 0.0, "mean_running": 22.0, "peak_running": 64, "mean_waiting": 9.0,
         "prompt_tps": 4000.0, "gen_tps": 3100.0, "prompt_gen_ratio": 1.3, "prefix_hit_rate": 0.1},
        {"minutes": 60, "idle_fraction": 0.4, "mean_running": 1.2, "prompt_tps": 6500.0, "gen_tps": 90.0,
         "prompt_gen_ratio": 72.0, "prefix_hit_rate": 0.35, "kv_usage_peak": 0.92},
    ]
    for s in summaries:
        out.append(("workload", wl.build_state(s, "vllm", "Qwen/Qwen3-8B", {"max_model_len": 32768}),
                    wl.QUESTION, wl.OPTIONS))
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


async def per_feature(http: httpx.AsyncClient, base: str, key: str, items) -> dict:
    """One client, every payload once: latency per feature (and the state size that drives it)."""
    by: dict[str, list[tuple[float, int]]] = {}
    for f, state, q, opts in items:
        body = {"state": state, "questions": {"q": {"type": "choice", "instructions": q, "criteria": criteria(opts)}}}
        t0 = time.perf_counter()
        r = await http.post(base + "/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"}, timeout=120)
        r.raise_for_status()
        by.setdefault(f, []).append(((time.perf_counter() - t0) * 1000, len(state)))
    return {f: {"n": len(v), "p50_ms": round(pct([x for x, _ in v], .5), 1), "p95_ms": round(pct([x for x, _ in v], .95), 1),
                "max_ms": round(max(x for x, _ in v), 1), "state_chars_p50": int(statistics.median(n for _, n in v))}
            for f, v in by.items()}


async def raw_level(http: httpx.AsyncClient, base: str, key: str, items, clients: int, per_client: int) -> dict:
    lat: list[float] = []
    errors = 0

    async def client(ci: int):
        nonlocal errors
        for j in range(per_client):
            f, state, q, opts = items[(ci * per_client + j) % len(items)]
            body = {"state": state, "questions": {"q": {"type": "choice", "instructions": q, "criteria": criteria(opts)}}}
            t0 = time.perf_counter()
            r = await http.post(base + "/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"},
                                timeout=120)
            lat.append((time.perf_counter() - t0) * 1000)
            errors += r.status_code != 200

    t0 = time.perf_counter()
    await asyncio.gather(*(client(i) for i in range(clients)))
    wall = time.perf_counter() - t0
    return {"clients": clients, "requests": len(lat), "errors": errors, "p50_ms": round(pct(lat, .5), 1),
            "p95_ms": round(pct(lat, .95), 1), "p99_ms": round(pct(lat, .99), 1), "max_ms": round(max(lat), 1),
            "throughput_rps": round(len(lat) / wall, 2), "wall_s": round(wall, 2)}


async def burst(http: httpx.AsyncClient, base: str, key: str, items, n: int, max_in_flight: int,
                queue_timeout_s: float, only: set[str] | None = None) -> dict:
    items = [x for x in items if only is None or x[0] in only]
    c = SystemOneClient(http, base_url=lambda: base, api_key=lambda: key, revision=f"loadtest-{time.time()}",
                        max_in_flight=max_in_flight, queue_timeout_s=queue_timeout_s, cache_size=0)
    by: dict[str, dict] = {}

    async def one(i: int):
        f, state, q, opts = items[(i * 7) % len(items)]
        deadline, prio = FEATURES[f]
        t0 = time.perf_counter()
        try:
            await c.choose(feature=f, state=state + f"\n(burst {i})", question=q, options=opts, timeout_s=deadline,
                           priority=prio)
            outcome = "answered"
        except DeciderError as e:
            outcome = {"busy": "shed", "timeout": "timeout"}.get(e.code, e.code)
        ms = (time.perf_counter() - t0) * 1000
        s = by.setdefault(f, {"deadline_ms": deadline * 1000, "answered": 0, "shed": 0, "timeout": 0, "other": 0,
                              "max_ms": 0.0, "over_deadline": 0})
        s[outcome if outcome in s else "other"] += 1
        s["max_ms"] = round(max(s["max_ms"], ms), 1)
        s["over_deadline"] += ms > deadline * 1000 + 100  # 100 ms of scheduling slack
    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0
    # What the burst left behind on the server: work for requests whose callers already gave up delays the next one.
    f, state, q, opts = items[1]
    body = {"state": state + "\n(after burst)", "questions": {"q": {"type": "choice", "instructions": q,
                                                                     "criteria": criteria(opts)}}}
    t1 = time.perf_counter()
    await http.post(base + "/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"}, timeout=300)
    return {"burst": n, "mix": sorted(only) if only else "all", "max_in_flight": max_in_flight,
            "queue_timeout_s": queue_timeout_s,
            "wall_s": round(wall, 2), "next_request_ms": round((time.perf_counter() - t1) * 1000, 1),
            "features": by, "breaker_open": c.breaker.is_open()}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base"), ap.add_argument("keyfile"), ap.add_argument("golden_dir"), ap.add_argument("hf_cache")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--per-client", type=int, default=6)
    ap.add_argument("--burst", type=int, default=16)
    ap.add_argument("--json")
    ap.add_argument("--per-feature", action="store_true", help="first, one sequential pass over every payload")
    a = ap.parse_args()
    key = Path(a.keyfile).read_text().strip()
    items = payloads(Path(a.golden_dir), Path(a.hf_cache))
    random.Random(20261016).shuffle(items)  # every concurrency level sees a mix of all features
    sizes = [len(s) for _, s, _, _ in items]
    print(f"{len(items)} payloads, state chars p50 {statistics.median(sizes):.0f} max {max(sizes)}; features "
          + ", ".join(f"{f}={sum(1 for x in items if x[0] == f)}" for f in FEATURES), flush=True)
    result: dict = {"payloads": len(items), "levels": [], "bursts": []}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as http:
        if a.per_feature:
            result["per_feature"] = await per_feature(http, a.base, key, items)
            print("per_feature", json.dumps(result["per_feature"]), flush=True)
        for lvl in [int(x) for x in a.levels.split(",") if x]:
            r = await raw_level(http, a.base, key, items, lvl, a.per_client)
            print("raw", json.dumps(r), flush=True)
            result["levels"].append(r)
        if a.burst:
            from dgxmm.config import Settings
            d = Settings()
            interactive = {"parser_tool", "parser_reasoning", "plan"}
            for only in (None, interactive):
                for mif, qt in ((d.decider_max_in_flight, d.decider_queue_timeout_s), (64, 30.0)):  # defaults, "off"
                    r = await burst(http, a.base, key, items, a.burst, mif, qt, only)
                    print("burst", json.dumps(r), flush=True)
                    result["bursts"].append(r)
                    await asyncio.sleep(2)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
