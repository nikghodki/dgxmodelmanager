from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from dgxmm.db import MetricRollup
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import EngineMetrics
from dgxmm.metrics.scraper import MetricsScraper
from dgxmm.metrics.store import MetricsStore
from tests.test_engines_vllm import VLLM_METRICS


def em(gen, prompt=0.0, ttft=None, hits=None, queries=None, running=1.0):
    m = EngineMetrics(gauges={"running": running, "waiting": 0.0, "kv_usage": 0.5},
                      counters={"generation_tokens": gen, "prompt_tokens": prompt})
    if ttft:
        m.histograms["ttft"] = ttft
    if hits is not None:
        m.counters.update(prefix_hits=hits, prefix_queries=queries)
    return m


def test_rates_quantiles_and_resets(db):
    s = MetricsStore(db)
    first = s.add(1, 100.0, em(1000, 500, ttft=[(0.1, 0), (1.0, 0), (float("inf"), 0)], hits=0, queries=0))
    assert first["gen_tps"] is None and first["running"] == 1.0
    second = s.add(1, 105.0, em(1500, 1000, ttft=[(0.1, 4), (1.0, 10), (float("inf"), 10)], hits=20, queries=40))
    assert second["gen_tps"] == 100.0 and second["prompt_tps"] == 100.0
    assert 0.1 < second["ttft_p50"] < 1.0 and second["prefix_hit_rate"] == 0.5
    reset = s.add(1, 110.0, em(10))  # engine restarted: counter went backwards
    assert reset["gen_tps"] is None
    assert len(s.series(1)) == 3


def test_rollup_history_and_prune(db):
    import time
    s = MetricsStore(db)
    base = int(time.time()) // 60 * 60 - 600  # ten minutes ago: inside the 30-day retention window
    s.add(1, base + 1, em(0))
    s.add(1, base + 31, em(300))
    s.add(1, base + 61, em(600))  # next minute: not complete yet when rolling at base+70
    assert s.rollup(now=base + 70) == 1
    [row] = s.history(1, hours=24 * 365 * 100)
    assert row["gen_tps"] == 10.0 and row["running"] == 1.0
    assert s.rollup(now=base + 70) == 0  # idempotent
    with db.session() as sess:
        sess.add(MetricRollup(deployment_id=1, minute=datetime.now(timezone.utc) - timedelta(days=31), data={}))
    s.rollup()
    with db.session() as sess:
        assert all(r.minute.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc) - timedelta(days=30)
                   for r in sess.scalars(select(MetricRollup)))


async def test_scraper_reads_ready_deployments(stack):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    stack.http.metrics[d["port"]] = VLLM_METRICS
    await stack.budget.refresh()
    q = stack.events.subscribe()
    t = [1000.0]
    sc = MetricsScraper(stack.settings, stack.deployer, stack.http.client(), MetricsStore(stack.db), stack.events,
                        stack.budget, clock=lambda: t[0])
    out = await sc.scrape_once()
    assert out[d["id"]]["running"] == 2.0 and out[d["id"]]["kv_usage"] == 0.25
    topics = [q.get_nowait()[0] for _ in range(q.qsize())]
    assert "metrics" in topics
    assert sc.store.series("host")[-1]["used"] > 0
