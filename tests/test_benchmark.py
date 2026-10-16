import asyncio
import json

import httpx
import pytest
from sqlalchemy import select

from dgxmm.advisor.benchmark import BenchmarkBusy, BenchmarkRunner, percentile
from dgxmm.db import Benchmark
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
