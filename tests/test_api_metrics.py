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
