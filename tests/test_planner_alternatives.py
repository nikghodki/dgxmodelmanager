"""Stop/sleep plans per eligible deployment (and minimal combinations), ranked so idle / low-priority /
smallest-sufficient come first. Live finding: the planner offered a single stop plan, picked by (priority, size),
which stopped the busy deployment; the safety filter rejected it and the recommender had no alternative."""
import random
import time
from datetime import datetime, timezone

from dgxmm.config import GIB
from dgxmm.db import DeploymentEvent
from dgxmm.planner.plans import MAX_SLEEP_PLANS, MAX_STOP_PLANS, ActiveDep, build_plans
from tests.test_plans import CFG, estimator, fit_needing


def dep(i, name=None, *, prio=0, reserved=10, actual=None, state="ready", can_sleep=False, sleep_frees=0,
        idle=None, disruptive=False):
    return ActiveDep(i, name or f"d{i}", state, prio, int(reserved * GIB), int((actual or reserved) * GIB), can_sleep,
                     int(sleep_frees * GIB), idle_min=idle, disruptive=disruptive)


def build(active, short_gib, settings):
    fit = fit_needing(40 + short_gib, 40, settings)
    return build_plans(fit=fit, cfg=CFG, context_key="max_model_len", seqs_key="max_num_seqs", kv_patches=[],
                       estimate_with=lambda p: None, active=active, variants=[], estimate_ref=lambda r: None)


def ids(plans, kind):
    return [tuple(sorted(a.deployment_id for a in p.actions)) for p in plans if p.actions and p.actions[0].type == kind]


def test_live_case_offers_the_idle_deployment_before_the_busy_one(settings):
    busy = dep(1, "poc-traffic", reserved=30, idle=0, disruptive=True)  # bigger, serving requests
    idle = dep(2, "poc-idle", reserved=22, idle=180)
    plans = build([busy, idle], 20, settings)
    assert ids(plans, "stop") == [(2,), (1,)]  # both offered, the idle one first
    by = {p.id: p for p in plans}
    assert by["stop:2"].fits and by["stop:2"].frees == 22 * GIB and by["stop:2"].title == "Stop poc-idle"
    assert by["stop:1"].fits


def test_smallest_sufficient_first_among_equally_idle(settings):
    plans = build([dep(1, reserved=40, idle=60), dep(2, reserved=25, idle=60), dep(3, reserved=10, idle=60)], 20,
                  settings)
    assert ids(plans, "stop")[:2] == [(2,), (1,)]  # 25 GiB covers 20 with the least loss; 10 alone does not


def test_lower_priority_then_longer_idle_first(settings):
    plans = build([dep(1, prio=10, reserved=30, idle=600), dep(2, prio=0, reserved=30, idle=5),
                   dep(3, prio=0, reserved=30, idle=300)], 20, settings)
    assert ids(plans, "stop")[:3] == [(3,), (2,), (1,)]


def test_sleeping_deployments_are_the_cheapest_to_stop(settings):
    plans = build([dep(1, reserved=30, idle=600), dep(2, reserved=30, actual=25, state="sleeping")], 20, settings)
    assert ids(plans, "stop")[0] == (2,)


def test_minimal_combinations_when_no_single_deployment_suffices(settings):
    plans = build([dep(1, reserved=20, idle=60), dep(2, reserved=15, idle=60), dep(3, reserved=12, idle=60)], 30,
                  settings)
    combos = ids(plans, "stop")
    assert combos and all(len(c) == 2 for c in combos)  # {1,2,3} is not minimal: {1,2} already covers 30
    assert set(combos) <= {(1, 2), (1, 3)}  # 15 + 12 = 27 < 30
    assert all(p.fits for p in plans if p.actions)
    assert combos[0] == (1, 3)  # the smaller sufficient pair (32 GiB) first


def test_sleep_plans_per_deployment(settings):
    a = dep(1, reserved=30, can_sleep=True, sleep_frees=22, idle=90)
    b = dep(2, reserved=30, can_sleep=True, sleep_frees=21, idle=900)
    c = dep(3, reserved=30, can_sleep=False, idle=900)
    plans = build([a, b, c], 20, settings)
    assert ids(plans, "sleep") == [(2,), (1,)]
    assert {p.id for p in plans if p.actions and p.actions[0].type == "sleep"} == {"sleep:1", "sleep:2"}


def test_falls_back_to_one_greedy_plan_when_nothing_suffices(settings):
    plans = build([dep(1, reserved=5, idle=60), dep(2, reserved=5, idle=60)], 30, settings)
    stops = [p for p in plans if p.actions and p.actions[0].type == "stop"]
    assert len(stops) == 1 and not stops[0].fits and len(stops[0].actions) == 2
    # many small ones: the greedy plan may need more than three deployments, and still fits
    plans = build([dep(i, reserved=6, idle=60) for i in range(1, 7)], 20, settings)
    stops = [p for p in plans if p.actions and p.actions[0].type == "stop"]
    assert len(stops) == 1 and stops[0].fits and len(stops[0].actions) == 4


def test_plan_count_is_bounded(settings):
    active = [dep(i, reserved=10 + i, can_sleep=True, sleep_frees=9 + i, idle=i * 10) for i in range(1, 13)]
    for short in (5, 15, 25, 40):
        plans = build(active, short, settings)
        assert len(ids(plans, "stop")) <= MAX_STOP_PLANS and len(ids(plans, "sleep")) <= MAX_SLEEP_PLANS


def _frees(d: ActiveDep, kind: str) -> int:
    if kind == "sleep":
        return d.sleep_frees
    return d.actual if d.state == "sleeping" else max(d.reserved, d.actual)


def test_property_plans_are_sufficient_minimal_unique_and_ranked(settings):
    rng = random.Random(20261016)
    checked = 0
    for trial in range(400):
        n = rng.randint(0, 9)
        active = []
        for i in range(1, n + 1):
            state = rng.choice(["ready", "ready", "ready", "sleeping", "loading", "unknown"])
            reserved = rng.choice([4, 8, 13, 20, 30, 45])
            can_sleep = state == "ready" and rng.random() < 0.5
            active.append(dep(i, state=state, prio=rng.choice([0, 0, 10, 50]), reserved=reserved,
                              actual=reserved * rng.choice([0.5, 1.0]) if state == "sleeping" else reserved,
                              can_sleep=can_sleep, sleep_frees=reserved * 0.6 if can_sleep else 0,
                              idle=rng.choice([None, 0, 3, 30, 600]), disruptive=rng.random() < 0.3))
        short = rng.choice([1, 10, 25, 50, 90])
        plans = build(active, short, settings)
        by_id = {d.id: d for d in active}
        for kind, bound in (("stop", MAX_STOP_PLANS), ("sleep", MAX_SLEEP_PLANS)):
            group = [p for p in plans if p.actions and p.actions[0].type == kind]
            assert len(group) <= bound
            sets = [frozenset(a.deployment_id for a in p.actions) for p in group]
            assert len(set(sets)) == len(sets), "duplicate plans"
            for p, s in zip(group, sets):
                assert all(a.type == kind for a in p.actions)
                assert p.id == f"{kind}:" + "+".join(str(i) for i in sorted(s))
                freed = sum(_frees(by_id[i], kind) for i in s)
                assert p.frees == freed and p.fits == (freed >= short * GIB)
                if kind == "sleep":
                    assert all(by_id[i].can_sleep and by_id[i].state == "ready" for i in s)
                if p.fits and len(group) > 1 or (p.fits and len(s) <= 3):
                    # minimal: dropping any member leaves the shortfall uncovered
                    for i in s:
                        assert freed - _frees(by_id[i], kind) < short * GIB, (trial, kind, s)
                checked += 1
            singles = [next(iter(s)) for s, p in zip(sets, group) if len(s) == 1 and p.fits]
            sufficient = sorted((d for d in active if (kind == "stop" or (d.can_sleep and d.state == "ready"))
                                 and _frees(d, kind) >= short * GIB),
                                key=lambda d: (d.disruptive, d.priority))
            if sufficient:  # the best-ranked deployment that alone covers the shortfall is always offered first
                assert singles and by_id[singles[0]].disruptive == sufficient[0].disruptive
                assert by_id[singles[0]].priority == min(d.priority for d in sufficient
                                                         if d.disruptive == sufficient[0].disruptive)
    assert checked > 300


async def test_preview_offers_and_recommends_stopping_the_idle_deployment(stack):
    """Through the deployer: a busy and an idle deployment that each free enough. Both stop plans are offered; the
    busy one is filtered as unsafe, so even a model that insists on it cannot get it recommended."""
    from dgxmm.deployer.service import DeployRequest
    from dgxmm.engines.base import EngineMetrics
    from dgxmm.metrics.store import MetricsStore
    from tests.fakes import FakeDecider
    stack.settings.max_concurrent_startups = 0
    stack.probe.total, stack.probe.available = 200 * GIB, 180 * GIB
    small = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
    views = [await stack.deployer.deploy(DeployRequest(name=n, engine="vllm", model_ref="org/big-model", config=small,
                                                       alias=n), 1) for n in ("busy", "idle")]
    stack.http.all_healthy = True
    for d in views:
        await stack.deployer.wait_ready(d["id"], 5)
        stack.probe.available -= d["reserved_bytes"]  # GB10: a ready model's memory is part of `used`
    busy, idle = views
    with stack.db.session() as s:  # both ready for hours
        for e in s.query(DeploymentEvent):
            e.ts = datetime.fromtimestamp(time.time() - 7200, timezone.utc)
    store = MetricsStore(stack.db)
    stack.deployer.metrics = store
    now = time.time()
    for k in range(16):  # a full window of metrics for both; only `busy` has traffic
        store.add(busy["id"], now - k * 60, EngineMetrics(gauges={"running": 1.0}))
        store.add(idle["id"], now - k * 60, EngineMetrics(gauges={"running": 0.0}))
    stack.deployer.decider_getter = lambda: FakeDecider(answers={"plan": (f"stop:{busy['id']}", 0.99)})
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    stack.probe.available -= p["fit"]["free_for_new"] - (p["fit"]["required"] - 10 * GIB)  # 10 GiB short
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    assert not p["fit"]["fits"]
    by = {x["id"]: x for x in p["plans"]}
    assert f"stop:{idle['id']}" in by and f"stop:{busy['id']}" in by
    assert "requests" in by[f"stop:{busy['id']}"]["unsafe_reason"]  # serving / served requests just now
    assert by[f"stop:{idle['id']}"]["unsafe_reason"] is None and by[f"stop:{idle['id']}"]["fits"]
    rec = [x for x in p["plans"] if x["recommended"]]
    assert rec and rec[0]["id"] != f"stop:{busy['id']}"
    stops = [x["id"] for x in p["plans"] if x["actions"] and x["actions"][0]["type"] == "stop"]
    assert stops[0] == f"stop:{idle['id']}"  # ranked by traffic: the idle deployment is offered first
