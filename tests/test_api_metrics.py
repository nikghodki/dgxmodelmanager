import time

import httpx

from dgxmm.advisor.benchmark import BenchmarkRunner
from dgxmm.advisor.rules import AdvisorService
from dgxmm.metrics.store import MetricsStore
from tests.test_api_deployments import SMALL, admin
from tests.test_benchmark import sse_handler
from tests.test_engines_vllm import VLLM_METRICS


def wire(c):
    st = c.stack
    c.ctx.metrics = MetricsStore(st.db)
    c.ctx.advisor = AdvisorService(st.deployer, st.inv, c.ctx.metrics, st.db)
    c.ctx.benchmarks = BenchmarkRunner(st.db, st.events,
                                       http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(sse_handler([]))))


def test_metrics_advice_benchmark_routes(api_stack):
    h = admin(api_stack)
    wire(api_stack)
    api_stack.stack.http.all_healthy = True
    d = api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "a", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                                                    "config": {**SMALL, "max_model_len": 65536 // 16}}).json()
    for _ in range(200):
        if api_stack.get(f"/dgxmm/api/deployments/{d['id']}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    from dgxmm.engines.vllm import VllmEngine
    api_stack.ctx.metrics.add(d["id"], 1.0, VllmEngine(api_stack.ctx.settings).parse_metrics(VLLM_METRICS))
    live = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/metrics?window=live").json()
    assert live[0]["running"] == 2.0
    assert api_stack.get(f"/dgxmm/api/deployments/{d['id']}/metrics?window=24h").json() == []
    assert isinstance(api_stack.get(f"/dgxmm/api/deployments/{d['id']}/advice").json(), list)
    r = api_stack.post(f"/dgxmm/api/deployments/{d['id']}/benchmark", headers=h,
                       json={"levels": [1], "input_len": 8, "output_len": 5, "duration_s": 2})
    assert r.status_code == 202
    for _ in range(200):
        rows = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/benchmarks").json()
        if rows:
            break
        time.sleep(0.02)
    assert rows[0]["concurrency"] == 1
    bad = api_stack.post(f"/dgxmm/api/deployments/{d['id']}/benchmark", headers=h, json={"levels": [1000]})
    assert bad.status_code == 422
    assert api_stack.get("/dgxmm/api/host/metrics?window=live").status_code == 200


def test_benchmarks_route_hides_failed_rows_and_reports_runs(api_stack):
    """Bug B: legacy requests=0 rows are hidden from results/advice; failed runs are visible via benchmark-runs."""
    from dgxmm.db import Benchmark, BenchmarkRun
    h = admin(api_stack)
    wire(api_stack)
    api_stack.stack.http.all_healthy = True
    d = api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "a", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                                                    "config": SMALL}).json()
    common = dict(deployment_id=d["id"], engine="vllm", model_ref="Qwen/Qwen3-Tiny", config_hash="h", config={},
                  input_len=8, output_len=5, ttft_p50=None, ttft_p95=None, itl_p50=None)
    with api_stack.ctx.db.session() as s:
        s.add(Benchmark(concurrency=1, requests=8, tps_total=50.0, tps_per_req=50.0, **common))
        s.add(Benchmark(concurrency=4, requests=0, tps_total=0.0, tps_per_req=0.0, **common))  # legacy failed row
        s.add(Benchmark(concurrency=8, requests=8, tps_total=0.0, tps_per_req=0.0, **common))  # zero throughput
        s.add(BenchmarkRun(deployment_id=d["id"], engine="vllm", model_ref="Qwen/Qwen3-Tiny", config_hash="h",
                           levels=[1, 4], status="failed", failed_concurrency=4, error="no successful requests at concurrency 4"))
    rows = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/benchmarks").json()
    assert [r["concurrency"] for r in rows] == [1]
    runs = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/benchmark-runs").json()
    assert runs[0]["status"] == "failed" and runs[0]["failed_concurrency"] == 4 and "concurrency 4" in runs[0]["error"]
