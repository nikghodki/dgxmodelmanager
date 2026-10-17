import random
import time

import pytest

from dgxmm.config import GIB
from dgxmm.decider import plans as dp
from dgxmm.decider.plans import DepFacts
from tests.fakes import FakeDecider

DEFAULTS = {"max_model_len": 32768, "max_num_seqs": 8}


def plan(pid, fits=True, actions=(), patch=None, model_ref=None, frees=0, needs=40 * GIB):
    return {"id": pid, "title": pid.title(), "detail": f"detail of {pid}", "new_required": needs, "frees": frees,
            "fits": fits, "actions": [{"type": t, "deployment_id": i} for t, i in actions], "patch": patch,
            "model_ref": model_ref}


async def rec(plans, deps, decider, raw_cfg=None, cache=None):
    return await dp.recommend(plans, deps=deps, decider=decider, shortfall=20 * GIB, model_ref="org/new", engine="vllm",
                              cfg={"max_model_len": 65536}, raw_cfg=raw_cfg or {}, defaults=DEFAULTS,
                              context_key="max_model_len", seqs_key="max_num_seqs", request_priority=0,
                              cache=cache if cache is not None else dp._Cache())


def idle(i, name, **kw):
    return DepFacts(id=i, name=name, state=kw.pop("state", "ready"), last_traffic_min=kw.pop("last", 3 * 1440),
                    traffic_known=kw.pop("known", True), ready_for_min=kw.pop("ready_for", 3 * 1440), **kw)


@pytest.mark.parametrize("facts,why", [
    (idle(1, "a", state="loading"), "is loading"),
    (idle(1, "a", state="starting"), "is starting"),
    (idle(1, "a", state="unknown"), "unknown state"),
    (idle(1, "a", last=3), "served requests 3 min ago"),
    (idle(1, "a", running_now=True), "serving requests"),
    (idle(1, "a", system=True), "system deployment"),
    (idle(1, "a", priority=90), "higher priority"),
    (idle(1, "a", known=False), "traffic unknown"),
    (idle(1, "a", ready_for=5), "ready 5 min ago"),
    (idle(1, "a", dependents=["b"]), "b depend"),
])
def test_unsafe_reasons(facts, why):
    r = dp.unsafe_reason(plan("stop", actions=[("stop", 1)]), {1: facts}, raw_cfg={}, defaults=DEFAULTS,
                         context_key="max_model_len", seqs_key="max_num_seqs", request_priority=0)
    assert r is not None and why in r


def test_safe_cases():
    kw = dict(raw_cfg={}, defaults=DEFAULTS, context_key="max_model_len", seqs_key="max_num_seqs", request_priority=0)
    assert dp.unsafe_reason(plan("stop", actions=[("stop", 1)]), {1: idle(1, "a")}, **kw) is None
    assert dp.unsafe_reason(plan("stop", actions=[("stop", 1)]), {1: idle(1, "a", state="sleeping", last=1)}, **kw) is None
    assert "traffic unknown" in dp.unsafe_reason(plan("stop", actions=[("stop", 1)]),
                                                 {1: idle(1, "a", last=None, known=False)}, **kw)
    assert dp.unsafe_reason(plan("kv", patch={"kv_cache_dtype": "fp8"}), {}, **kw) is None
    assert dp.unsafe_reason(plan("stop", fits=False), {}, **kw) == "does not free enough memory"


def test_user_stated_context_is_kept():
    kw = dict(defaults=DEFAULTS, context_key="max_model_len", seqs_key="max_num_seqs", request_priority=0)
    p = plan("context", patch={"max_model_len": 65536})
    assert "context length you asked for (131,072)" in dp.unsafe_reason(p, {}, raw_cfg={"max_model_len": 131072}, **kw)
    assert dp.unsafe_reason(p, {}, raw_cfg={}, **kw) is None  # not typed by the user
    assert dp.unsafe_reason(plan("context", patch={"max_model_len": 16384}), {},
                            raw_cfg={"max_model_len": 32768}, **kw) is None  # the default value: not a requirement
    s = plan("seqs", patch={"max_num_seqs": 4})
    assert "concurrent sequences" in dp.unsafe_reason(s, {}, raw_cfg={"max_num_seqs": 16}, **kw)


async def test_single_allowed_plan_recommended_by_rules():
    fake = FakeDecider(answers={"plan": ("stop", 0.99)})
    deps = {1: idle(1, "busy", last=1), 2: idle(2, "old")}
    out = await rec([plan("sleep", actions=[("sleep", 1)]), plan("stop", actions=[("stop", 2)], frees=48 * GIB)],
                    deps, fake)
    by = {p["id"]: p for p in out}
    assert by["stop"]["recommended"] and by["stop"]["recommendation"]["source"] == "rules"
    assert by["stop"]["recommendation"]["reason"] == "frees 48.0 GiB; old idle 3 days"
    assert by["sleep"]["unsafe_reason"] and not by["sleep"]["recommended"] and by["sleep"]["recommendation"] is None
    assert fake.calls == []


async def test_model_ranks_only_allowed_plans():
    fake = FakeDecider(answers={"plan": ("kv", 0.8)})
    deps = {1: idle(1, "busy", running_now=True), 2: idle(2, "old")}
    out = await rec([plan("kv", patch={"kv_cache_dtype": "fp8"}), plan("stop", actions=[("stop", 2)]),
                     plan("sleep", actions=[("sleep", 1)]), plan("seqs", fits=False)], deps, fake)
    offered = [o.key for o in fake.calls[0]["options"]]
    assert offered == ["kv", "stop"]
    rec_ = [p for p in out if p["recommended"]]
    assert [p["id"] for p in rec_] == ["kv"] and rec_[0]["recommendation"]["source"] == "model"
    assert rec_[0]["recommendation"]["confidence"] == pytest.approx(0.8)
    assert "keeps other deployments running" in rec_[0]["recommendation"]["reason"]
    assert fake.calls[0]["timeout_s"] == 2.0 and "org/new" not in fake.calls[0]["question"]


async def test_below_threshold_or_unavailable_no_recommendation():
    deps = {2: idle(2, "old")}
    plans = [plan("kv", patch={"kv_cache_dtype": "fp8"}), plan("stop", actions=[("stop", 2)])]
    for d in (FakeDecider(answers={"plan": {"kv": 0.4, "stop": 0.35}}), FakeDecider(available=False),
              FakeDecider(answers={"plan": RuntimeError("x")}), FakeDecider(delay_s=5, answers={"plan": ("kv", 0.9)})):
        dp.TIMEOUT_S, old = 0.05, dp.TIMEOUT_S
        try:
            t = time.monotonic()
            out = await rec(plans, deps, d)
            assert time.monotonic() - t < 2
        finally:
            dp.TIMEOUT_S = old
        assert not any(p["recommended"] for p in out)


async def test_cached_by_input():
    fake = FakeDecider(answers={"plan": ("kv", 0.9)})
    cache = dp._Cache()
    plans = [plan("kv", patch={"kv_cache_dtype": "fp8"}), plan("stop", actions=[("stop", 2)])]
    for _ in range(3):
        await rec(plans, {2: idle(2, "old")}, fake, cache=cache)
    assert len(fake.calls) == 1


async def test_no_allowed_plan_no_recommendation():
    out = await rec([plan("stop", actions=[("stop", 1)])], {1: idle(1, "sys", system=True)}, FakeDecider())
    assert out[0]["recommended"] is False and "system" in out[0]["unsafe_reason"]


def test_facts_from_views_and_metrics(db):
    from dgxmm.engines.base import EngineMetrics
    from dgxmm.metrics.store import MetricsStore
    store = MetricsStore(db)
    now = time.time()
    store.add(1, now - 120, EngineMetrics(gauges={"running": 1.0}))
    views = [
        {"id": 1, "name": "busy", "state": "ready", "priority": 0, "engine": "vllm", "model_ref": "a/a", "config": {}},
        {"id": 2, "name": "draft", "state": "ready", "priority": 0, "engine": "vllm", "model_ref": "a/draft", "config": {}},
        {"id": 3, "name": "user", "state": "ready", "priority": 95, "engine": "vllm", "model_ref": "a/b",
         "config": {"speculative": {"method": "eagle", "model": "a/draft"}}},
        {"id": 4, "name": "dec", "state": "ready", "priority": 0, "engine": "decision", "model_ref": "x/d", "config": {}},
        {"id": 5, "name": "gone", "state": "stopped", "priority": 0, "engine": "vllm", "model_ref": "a/c", "config": {}},
    ]
    f = dp.facts_for(views, store, now=now)
    assert set(f) == {1, 2, 3, 4}
    assert f[1].last_traffic_min == pytest.approx(2, abs=0.1) and f[1].active_reason()
    assert f[2].dependents == ["user"] and f[3].priority == 95 and f[4].system
    assert f[2].last_traffic_min is None


def test_rollup_traffic_lookup(db):
    from datetime import datetime, timezone

    from dgxmm.db import MetricRollup
    from dgxmm.metrics.store import MetricsStore
    minute = datetime.fromtimestamp((int(time.time() // 60) - 600) * 60, timezone.utc)
    with db.session() as s:
        s.add(MetricRollup(deployment_id=7, minute=minute, data={"running": 0.5, "gen_tps": 10}))
        s.add(MetricRollup(deployment_id=7, minute=datetime.now(timezone.utc), data={"running": 0, "gen_tps": 0}))
    t = dp.traffic(MetricsStore(db), 7)
    assert 595 <= t.last_min <= 601 and t.running_now is False and not t.known


async def test_preview_plans_carry_contract_fields(stack):
    from dgxmm.deployer.service import DeployRequest
    small = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
    a = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="org/big-model", config=small), 1)
    stack.http.healthy.add(a["port"])
    await stack.deployer.wait_ready(a["id"], 5)
    stack.probe.available -= a["reserved_bytes"]
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    assert p["plans"]
    for x in p["plans"]:
        assert {"recommended", "recommendation", "unsafe_reason"} <= set(x)
    stop = next(x for x in p["plans"] if x["id"] == f"stop:{a['id']}")
    assert not stop["recommended"] and stop["unsafe_reason"]  # just became ready / no metrics: never recommended
    from dgxmm.engines.base import EngineMetrics
    from dgxmm.metrics.store import MetricsStore
    stack.deployer.metrics = MetricsStore(stack.db)
    stack.deployer.metrics.add(a["id"], time.time(), EngineMetrics(gauges={"running": 2.0}))
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    stop = next(x for x in p["plans"] if x["id"] == f"stop:{a['id']}")
    assert not stop["recommended"] and "serving requests" in stop["unsafe_reason"]
