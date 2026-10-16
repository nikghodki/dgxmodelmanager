import asyncio
import json

import httpx
import pytest
from sqlalchemy import select

from dgxmm.advisor.benchmark import BenchmarkBusy, BenchmarkRunner, percentile
from dgxmm.db import Benchmark, BenchmarkRun
from dgxmm.events import EventBroker


def sse_handler(seen):
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append(body)
        chunks = "".join(f"data: {json.dumps({'choices': [{'text': 'x'}]})}\n\n" for _ in range(5))
        chunks += f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 5}})}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, content=chunks.encode(), headers={"content-type": "text/event-stream"})
    return handler


def runner(db, seen, events=None):
    return BenchmarkRunner(db, events or EventBroker(),
                           http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(sse_handler(seen))))


async def test_runs_levels_and_saves_rows(db):
    seen = []
    r = runner(db, seen)
    rows = await r.run(deployment_id=1, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config={"a": 1},
                       base_url="http://h:1", served_name="q", levels=(1, 2), input_len=16, output_len=5,
                       duration_s=5, max_requests=4)
    assert [x["concurrency"] for x in rows] == [1, 2] and all(x["requests"] == 4 for x in rows)
    assert all(x["tps_total"] > 0 and x["ttft_p50"] is not None for x in rows)
    assert seen[0]["stream"] is True and seen[0]["ignore_eos"] is True and seen[0]["max_tokens"] == 5
    assert seen[0]["prompt"] != seen[1]["prompt"]  # nonce defeats the prefix cache
    with db.session() as s:
        assert len(s.scalars(select(Benchmark)).all()) == 2


async def test_only_one_benchmark_at_a_time(db):
    r = runner(db, [])
    kw = dict(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1", served_name="q",
              levels=(1,), input_len=8, output_len=5, duration_s=1, max_requests=50)
    first = asyncio.create_task(r.run(**kw))
    await asyncio.sleep(0)
    with pytest.raises(BenchmarkBusy):
        await r.run(**kw)
    await first


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5



def failing_handler(req: httpx.Request):
    return httpx.Response(500, json={"error": "engine crashed"})


async def test_level_with_zero_successful_requests_is_not_stored(db):
    """Bug B: a level where every request failed must not become a tps=0 benchmark row."""
    r = BenchmarkRunner(db, EventBroker(), http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(failing_handler)))
    rows = await r.run(deployment_id=7, engine="vllm", model_ref="m/m", config={"a": 1}, base_url="http://h:1",
                       served_name="q", levels=(1, 4), input_len=8, output_len=5, duration_s=2, max_requests=3)
    with db.session() as s:
        assert s.scalars(select(Benchmark)).all() == []
        run = s.scalars(select(BenchmarkRun)).one()
        assert run.status == "failed" and run.failed_concurrency == 1 and run.deployment_id == 7
        assert "concurrency 1" in run.error and run.model_ref == "m/m"
    # stops at the first failed level (higher concurrency against a broken endpoint is pointless)
    assert len(rows) == 1 and rows[0]["failed"] is True and rows[0]["requests"] == 0 and rows[0]["errors"] == 3


async def test_partial_failure_keeps_good_levels_and_records_failed_level(db):
    ok, level = sse_handler([]), {"conc": 1}

    def handler(req):
        return ok(req) if level["conc"] < 2 else httpx.Response(503, json={"error": "overloaded"})
    r = BenchmarkRunner(db, EventBroker(), http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    orig = r._level

    async def tracking_level(http, base_url, model, conc, *a):
        level["conc"] = conc
        return await orig(http, base_url, model, conc, *a)
    r._level = tracking_level
    rows = await r.run(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1",
                       served_name="q", levels=(1, 2, 4), input_len=8, output_len=5, duration_s=2, max_requests=2)
    assert [x["concurrency"] for x in rows] == [1, 2] and rows[1]["failed"] is True and not rows[0].get("failed")
    with db.session() as s:
        assert [b.concurrency for b in s.scalars(select(Benchmark))] == [1]
        run = s.scalars(select(BenchmarkRun)).one()
        assert run.status == "failed" and run.failed_concurrency == 2


async def test_successful_run_is_recorded_done(db):
    r = runner(db, [])
    await r.run(deployment_id=3, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1", served_name="q",
                levels=(1,), input_len=8, output_len=5, duration_s=2, max_requests=2)
    with db.session() as s:
        run = s.scalars(select(BenchmarkRun)).one()
        assert run.status == "done" and run.failed_concurrency is None and run.error is None


# ---- review round 1 ----
class RecordingEvents:
    def __init__(self):
        self.seen = []

    def publish(self, kind, data):
        self.seen.append((kind, data))


async def test_run_insert_failure_does_not_leave_runner_busy(db):
    class FlakyDb:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        def session(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("database is locked")
            return self.inner.session()
    r = BenchmarkRunner(FlakyDb(db), EventBroker(),
                        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(sse_handler([]))))
    kw = dict(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1", served_name="q",
              levels=(1,), input_len=8, output_len=5, duration_s=1, max_requests=1)
    with pytest.raises(RuntimeError):
        await r.run(**kw)
    assert r.busy() is None
    assert len(await r.run(**kw)) == 1  # not stuck "busy" (409) until restart


async def test_level_with_mostly_errors_is_not_a_measurement(db):
    n = {"i": 0}
    ok = sse_handler([])

    def handler(req):
        n["i"] += 1
        return ok(req) if n["i"] % 4 == 0 else httpx.Response(500)  # 75 % errors
    r = BenchmarkRunner(db, EventBroker(), http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    rows = await r.run(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1",
                       served_name="q", levels=(1, 2), input_len=8, output_len=5, duration_s=5, max_requests=8)
    assert rows[0]["failed"] is True and rows[0]["requests"] == 2 and rows[0]["errors"] == 6 and len(rows) == 1
    with db.session() as s:
        assert s.scalars(select(Benchmark)).all() == []
        run = s.scalars(select(BenchmarkRun)).one()
        assert run.failed_concurrency == 1 and "6 of 8" in run.error


async def test_stored_levels_record_their_error_count(db):
    n = {"i": 0}
    ok = sse_handler([])

    def handler(req):
        n["i"] += 1
        return httpx.Response(500) if n["i"] == 1 else ok(req)  # 1 of 4 fails
    r = BenchmarkRunner(db, EventBroker(), http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await r.run(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1", served_name="q",
                levels=(1,), input_len=8, output_len=5, duration_s=5, max_requests=4)
    with db.session() as s:
        b = s.scalars(select(Benchmark)).one()
        assert b.requests == 3 and b.errors == 1


async def test_exception_publishes_final_failed_event(db):
    ev = RecordingEvents()

    def boom():
        raise RuntimeError("no client")
    r = BenchmarkRunner(db, ev, http_factory=boom)
    with pytest.raises(RuntimeError):
        await r.run(deployment_id=5, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1", served_name="q",
                    levels=(1,), input_len=8, output_len=5, duration_s=1, max_requests=1)
    kind, data = ev.seen[-1]
    assert kind == "benchmark" and data["deployment_id"] == 5 and data["done"] is True and data["failed"] is True
    assert "no client" in data["error"]
    with db.session() as s:
        assert s.scalars(select(BenchmarkRun)).one().status == "failed"
