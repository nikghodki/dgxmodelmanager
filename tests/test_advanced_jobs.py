"""Integration (spec 2026-10-17, "Background engine"): every insight-engine job of the advanced features is
registered, their combined decision-model budget is small against d3-mini's measured capacity, and interactive
questions are not starved while every job spends its whole budget at once."""
from __future__ import annotations

import asyncio
import time
from datetime import timezone

import httpx

from dgxmm.context import AppContext
from dgxmm.decider.client import Limiter
from dgxmm.decider.features import wire_features
from dgxmm.decider.insight_engine import InsightEngine
from dgxmm.decider.types import Option
from tests.test_decider_capacity import SlowServer

# job -> feature (the spec's job table)
EXPECTED_JOBS = {"health_check": "health", "reading_labels": "reading", "threshold_tuning": "tuning",
                 "routing_advice": "routing", "traffic_profiles": "schedule", "schedule_proposals": "schedule",
                 "schedule_runner": "schedule"}
# d3-mini answers 1.3-3.3 requests/s on this box (docs/decider-evaluation.md, "Performance and capacity")
CAPACITY_RPS_LOW = 1.3
OPTS = [Option("gpu_oom", "a"), Option("port_in_use", "b"), Option("other", "c")]


def _wired(stack) -> AppContext:
    from dgxmm.metrics.store import MetricsStore
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, inventory=stack.inv,
                     deployer=stack.deployer, engines=stack.deployer.engines)
    ctx.metrics = MetricsStore(stack.db)
    ctx.benchmarks = None
    ctx.insight_engine = InsightEngine(ctx, tz=timezone.utc)
    wire_features(ctx)
    return ctx


def _runs_per_hour(schedule: str) -> float:
    kind, _, rest = schedule.partition(" ")
    if kind == "every":
        return 3600.0 / float(rest.split()[0])
    return {"daily": 1 / 24, "weekly": 1 / 168}[kind]


def test_every_background_job_is_registered(stack):
    jobs = {j["name"]: j["feature"] for j in _wired(stack).insight_engine.status()}
    assert jobs == EXPECTED_JOBS


def test_combined_background_budget_is_small_against_the_decider_capacity(stack):
    status = _wired(stack).insight_engine.status()
    per_hour = sum(j["budget_calls"] * _runs_per_hour(j["schedule"]) for j in status)
    # scheduled runs: at most 5 % of the low capacity (1.3 req/s = 4680 questions an hour; 129 here)
    assert per_hour <= 0.05 * CAPACITY_RPS_LOW * 3600, per_hour
    # every job spending its whole budget at once (46 questions, about 35 s at 1.3 req/s on the one lane slot) drains
    # well within the shortest cadence of a job that asks the model (health, 300 s)
    burst = sum(j["budget_calls"] for j in status)
    shortest = min(3600.0 / _runs_per_hour(j["schedule"]) for j in status if j["budget_calls"])
    assert burst / CAPACITY_RPS_LOW < 0.5 * shortest, (burst, shortest)


async def test_interactive_questions_are_not_starved_while_every_job_spends_its_budget(stack):
    from dgxmm.decider.service import Decider
    from dgxmm.settings_store import SettingsStore
    wired = _wired(stack).insight_engine.status()
    srv = SlowServer(0.1)  # serial, like the d3 server
    d = Decider(db=stack.db, settings=stack.settings, store=SettingsStore(stack.db, stack.settings),
                deployer=lambda: None, http=lambda: httpx.AsyncClient(transport=httpx.MockTransport(srv.handler)))
    d.set_enabled(True)
    d._base_url = lambda: "http://decider:8100"
    d.client().limiter = Limiter(2)
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events)
    ctx.decider = d
    eng = InsightEngine(ctx, tz=timezone.utc)

    def spender(feature: str, budget: int):
        async def job(run):  # asks more than its budget: the rest must abstain without reaching the server
            out = [await run.decider.decide(feature=feature, subject=f"{run.name}:{i}", state=f"{run.name} {i}",
                                            question="q", options=OPTS, timeout_s=30.0) for i in range(budget + 3)]
            return f"{sum(o.source == 'model' for o in out)} answered"
        return job

    for j in wired:
        eng.register_job(j["name"], j["feature"], 3600, spender(j["feature"], j["budget_calls"]),
                         budget_calls=j["budget_calls"])
        eng.trigger(j["name"])
    started = await eng.tick()
    assert sorted(started) == sorted(EXPECTED_JOBS)
    await asyncio.sleep(0.05)
    lat = []
    for i in range(5):
        t0 = time.perf_counter()
        got = await d.decide(feature="parser_tool", subject="wizard", state=f"interactive {i}", question="q",
                             options=OPTS, timeout_s=2.0)
        lat.append(time.perf_counter() - t0)
        assert got.source == "model"
        assert d.client().limiter.lane_in_flight <= 1
    # at most one background request ahead of each interactive one on the serial server, plus its own
    assert max(lat) < 0.1 * 2 + 0.15, lat
    await eng.wait_idle()
    for st in eng.status():
        assert st["calls_used"] <= st["budget_calls"] and st["last_result"].startswith("ok"), st
    assert srv.served == sum(j["budget_calls"] for j in wired) + 5


# ---- one time zone for the engine's daily/weekly jobs and the schedules (DGXMM_TZ) ----

def _zones(settings, db, events):
    from dgxmm.decider.schedule import box_zone
    ctx = AppContext(settings=settings, db=db, events=events)
    return InsightEngine(ctx).tz, box_zone(ctx)


def _same_wall_clock(a, b):
    from datetime import datetime
    for t in (datetime(2026, 1, 15, 12, tzinfo=timezone.utc), datetime(2026, 7, 15, 12, tzinfo=timezone.utc),
              datetime(2026, 3, 29, 1, 30, tzinfo=timezone.utc), datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc)):
        assert t.astimezone(a).utcoffset() == t.astimezone(b).utcoffset(), t


def test_engine_and_schedules_use_the_dgxmm_tz_setting(stack, monkeypatch):
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(stack.settings, "tz", "America/New_York")
    eng, sched = _zones(stack.settings, stack.db, stack.events)
    assert eng.key == sched.key == "America/New_York"


def test_engine_and_schedules_fall_back_to_TZ_alike(stack, monkeypatch):
    monkeypatch.setattr(stack.settings, "tz", None)
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    eng, sched = _zones(stack.settings, stack.db, stack.events)
    assert eng.key == sched.key == "Asia/Kolkata"


def test_engine_and_schedules_agree_on_a_symlinked_localtime(stack, monkeypatch, tmp_path):
    import importlib.resources

    from dgxmm.decider import insight_engine, schedule
    berlin = importlib.resources.files("tzdata").joinpath("zoneinfo", "Europe", "Berlin")
    link = tmp_path / "localtime"
    link.symlink_to(str(berlin))
    monkeypatch.setattr(stack.settings, "tz", None)
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(insight_engine, "LOCALTIME", str(link))
    monkeypatch.setattr(schedule, "LOCALTIME", str(link))
    eng, sched = _zones(stack.settings, stack.db, stack.events)
    assert sched.key == "Europe/Berlin"
    _same_wall_clock(eng, sched)  # the engine reads the file itself: the same offsets, DST changes included
