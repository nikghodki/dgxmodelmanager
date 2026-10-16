"""Small load test against a deployment's OpenAI endpoint (spec §9.2)."""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

from dgxmm.db import Benchmark, BenchmarkRun, Database, utcnow

WORDS = ("memory token cache batch speed latency model server context window prompt decode prefill expert layer "
         "tensor kernel stream request response quality").split()


class BenchmarkBusy(Exception):
    pass


@dataclass
class ReqStat:
    ok: bool
    ttft: float | None = None
    itls: list[float] = field(default_factory=list)
    tokens: int = 0
    e2e: float = 0.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class BenchmarkRunner:
    def __init__(self, db: Database, events, http_factory: Callable[[], httpx.AsyncClient] | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.db, self.events, self.clock = db, events, clock
        self.http_factory = http_factory or (lambda: httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)))
        self._lock = asyncio.Lock()
        self._running: int | None = None

    def busy(self) -> int | None:
        return self._running

    async def _one(self, http: httpx.AsyncClient, base_url: str, model: str, input_len: int, output_len: int) -> ReqStat:
        prompt = f"[{uuid.uuid4().hex}] " + " ".join(random.choice(WORDS) for _ in range(input_len))
        body = {"model": model, "prompt": prompt, "max_tokens": output_len, "stream": True, "ignore_eos": True,
                "temperature": 0, "stream_options": {"include_usage": True}}
        t0 = self.clock()
        ttft, last, itls, chunks, usage = None, None, [], 0, None
        try:
            async with http.stream("POST", f"{base_url}/v1/completions", json=body) as r:
                if r.status_code != 200:
                    return ReqStat(ok=False, e2e=self.clock() - t0)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"].get("completion_tokens")
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("text"):
                        now = self.clock()
                        chunks += 1
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itls.append(now - last)
                        last = now
        except (httpx.HTTPError, ValueError):
            return ReqStat(ok=False, e2e=self.clock() - t0)
        return ReqStat(ok=True, ttft=ttft, itls=itls, tokens=int(usage or chunks), e2e=self.clock() - t0)

    async def _level(self, http, base_url, model, conc, input_len, output_len, duration_s, max_requests) -> dict:
        stats: list[ReqStat] = []
        start = self.clock()
        issued = 0

        async def worker():
            nonlocal issued
            while issued < max_requests and self.clock() - start < duration_s:
                issued += 1
                stats.append(await self._one(http, base_url, model, input_len, output_len))

        await asyncio.gather(*(worker() for _ in range(conc)))
        wall = max(self.clock() - start, 1e-9)
        ok = [s for s in stats if s.ok]
        per_req = [s.tokens / max(s.e2e - (s.ttft or 0), 1e-9) for s in ok if s.tokens]
        return {"concurrency": conc, "requests": len(ok), "errors": len(stats) - len(ok),
                "ttft_p50": percentile([s.ttft for s in ok if s.ttft is not None], 50),
                "ttft_p95": percentile([s.ttft for s in ok if s.ttft is not None], 95),
                "itl_p50": percentile([x for s in ok for x in s.itls], 50),
                "tps_total": sum(s.tokens for s in ok) / wall,
                "tps_per_req": sum(per_req) / len(per_req) if per_req else 0.0}

    async def run(self, *, deployment_id: int, engine: str, model_ref: str, config: dict, base_url: str,
                  served_name: str, levels=(1, 4, 8, 16), input_len: int = 1024, output_len: int = 256,
                  duration_s: float = 60.0, max_requests: int = 32) -> list[dict]:
        from dgxmm.deployer.service import config_hash
        if self._lock.locked():
            raise BenchmarkBusy(f"a benchmark is already running (deployment {self._running})")
        async with self._lock:
            self._running = deployment_id
            h = config_hash(config)
            with self.db.session() as s:
                run = BenchmarkRun(deployment_id=deployment_id, engine=engine, model_ref=model_ref, config_hash=h,
                                   levels=list(levels), status="running")
                s.add(run)
                s.flush()
                run_id = run.id
            results: list[dict] = []
            status, failed_at, error = "done", None, None
            try:
                async with self.http_factory() as http:
                    for conc in levels:
                        r = await self._level(http, base_url, served_name, conc, input_len, output_len, duration_s, max_requests)
                        if r["requests"] == 0:
                            # Every request failed: not a measurement. Record the failure on the run instead of
                            # storing a tps=0 row, and stop - higher concurrency against a broken endpoint is moot.
                            status, failed_at = "failed", conc
                            error = f"no successful requests at concurrency {conc} ({r['errors']} failed)"
                            r["failed"] = True
                            results.append(r)
                            self.events.publish("benchmark", {"deployment_id": deployment_id, "level": r, "done": True,
                                                              "failed": True, "error": error})
                            break
                        with self.db.session() as s:
                            s.add(Benchmark(deployment_id=deployment_id, engine=engine, model_ref=model_ref,
                                            config_hash=h, config=config, concurrency=conc,
                                            input_len=input_len, output_len=output_len, requests=r["requests"],
                                            ttft_p50=r["ttft_p50"], ttft_p95=r["ttft_p95"], itl_p50=r["itl_p50"],
                                            tps_total=r["tps_total"], tps_per_req=r["tps_per_req"]))
                        results.append(r)
                        self.events.publish("benchmark", {"deployment_id": deployment_id, "level": r,
                                                          "done": conc == levels[-1]})
            except BaseException as e:
                status, error = "failed", f"benchmark aborted: {e.__class__.__name__}: {e}"[:500]
                raise
            finally:
                self._running = None
                with self.db.session() as s:
                    run = s.get(BenchmarkRun, run_id)
                    if run is not None:
                        run.status, run.failed_concurrency, run.error = status, failed_at, error
                        run.finished_at = utcnow()
            return results
