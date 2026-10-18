"""Feature 7 — the schedule runner (dgxmm/decider/schedule.py): sleep/stop/wake/start through the deployer's normal
paths, quiet-traffic checks, never on system deployments or mid-transition, never evicting, once per slot, DST gaps
and overlaps, catching up after a restart, removed deployments, colliding schedules; every run is a deployment event
and an audit row with actor schedule:<id>."""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.context import AppContext
from dgxmm.db import AuditLog, Deployment, DeploymentEvent, DeploymentSchedule, Insight
from dgxmm.decider.schedule import ScheduleIn, ScheduleService, next_run
from dgxmm.deployer.service import DeployRequest
from dgxmm.metrics.store import MetricsStore
from tests.fakes import free_on_stop

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
NOW = datetime(2026, 10, 19, 10, 0, 30, tzinfo=timezone.utc)  # a Monday
MON = 1


class Clock:
    def __init__(self, t):
        self.now = t

    def __call__(self):
        return self.now


@pytest.fixture
def env(stack):
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, deployer=stack.deployer,
                     inventory=stack.inv, budget=stack.budget, metrics=MetricsStore(stack.db))
    clock = Clock(NOW - timedelta(minutes=10))  # when the schedules are created
    svc = ScheduleService(ctx, clock=clock)
    stack.deployer.removal_hooks.append(svc.on_deployment_removed)
    stack.ctx, stack.svc, stack.clock = ctx, svc, clock
    return stack


async def ready(stack, name, *, sleep=True, **cfg):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", alias=name,
                                                  config={**SMALL, "enable_sleep_mode": sleep, **cfg}), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return stack.deployer.view(d["id"])


def quiet(stack, dep_id, now=NOW, *, minutes=16):
    """Live samples every 5 s for the last `minutes` minutes: nothing running, waiting or generating."""
    ring = stack.ctx.metrics._ring(dep_id)
    t0 = now.timestamp() - minutes * 60
    for i in range(minutes * 12):
        ring.append({"ts": t0 + 5 * i + 5, "running": 0.0, "waiting": 0.0, "prompt_tps": 0.0, "gen_tps": 0.0})


def busy(stack, dep_id, ts, running=1.0, waiting=0.0):
    stack.ctx.metrics._ring(dep_id).append({"ts": ts, "running": running, "waiting": waiting, "prompt_tps": 50.0,
                                            "gen_tps": 20.0})


def add(stack, view, action, at, days=0b1111111, tz="UTC", **kw):
    return stack.svc.create(view, [ScheduleIn(action=action, days=days, at_local=at, timezone=tz, **kw)], 1)[0]


def row(stack, sid):
    with stack.db.session() as s:
        return s.get(DeploymentSchedule, sid)


def events(stack, dep_id):
    with stack.db.session() as s:
        return [e.message for e in s.scalars(select(DeploymentEvent).where(DeploymentEvent.deployment_id == dep_id)
                                             .order_by(DeploymentEvent.id))]


def audits(stack):
    with stack.db.session() as s:
        return [(a.action, a.user_id, a.detail) for a in s.scalars(select(AuditLog).where(AuditLog.action == "schedule_run")
                                                                   .order_by(AuditLog.id))]


def state(stack, dep_id):
    return stack.deployer.view(dep_id)["state"]


# ---------------- the normal path ----------------
async def test_sleeps_a_quiet_deployment_once_per_day_with_event_and_audit(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    assert await env.svc.run_due(NOW) == "1 ran"
    assert state(env, v["id"]) == "sleeping"
    r = row(env, s["id"])
    assert r.last_result.startswith("ok:") and r.last_run_date == "2026-10-19"
    assert any(m.startswith(f"Schedule #{s['id']} (sleep at 10:00 UTC") for m in events(env, v["id"]))
    (action, user, detail), = audits(env)
    assert action == "schedule_run" and user is None and detail["actor"] == f"schedule:{s['id']}"
    assert detail["status"] == "ok" and detail["deployment_id"] == v["id"]
    # the same slot never runs twice (later passes of the day, and after a manual wake)
    await env.deployer.wake(v["id"])
    quiet(env, v["id"], NOW + timedelta(minutes=5))
    assert await env.svc.run_due(NOW + timedelta(minutes=5)) == "nothing due"
    assert state(env, v["id"]) == "ready" and len(audits(env)) == 1
    nxt = next_run(row(env, s["id"]), NOW)
    assert nxt == datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)


async def test_not_due_before_its_slot_or_on_other_days(env):
    v = await ready(env, "a")
    add(env, v, "sleep", "10:05")
    add(env, v, "sleep", "09:00", days=0b10)  # Tuesday only
    quiet(env, v["id"])
    assert await env.svc.run_due(NOW) == "nothing due"
    assert state(env, v["id"]) == "ready"


async def test_sleep_schedule_stops_a_deployment_without_sleep_mode(env):
    v = await ready(env, "a", sleep=False)
    s = add(env, v, "stop", "10:00")
    quiet(env, v["id"])
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == "stopped"
    assert any(f"stop requested (schedule:{s['id']})" in m for m in events(env, v["id"]))


async def test_wake_and_start(env):
    a = await ready(env, "a")
    b = await ready(env, "b", sleep=False)
    await env.deployer.sleep(a["id"])
    await env.deployer.stop(b["id"])
    add(env, a, "wake", "10:00")
    sb = add(env, env.deployer.view(b["id"]), "start", "10:00")
    assert await env.svc.run_due(NOW) == "1 pending, 1 ran"  # b is still starting: its outcome comes later
    assert state(env, a["id"]) == "ready"
    assert row(env, sb["id"]).last_result.startswith("pending: starting")
    assert await env.deployer.wait_ready(b["id"], 5) == "ready"
    assert await env.svc.run_due(NOW + timedelta(minutes=1)) == "1 ran"
    assert row(env, sb["id"]).last_result.startswith("ok: started (after waiting")


# ---------------- traffic ----------------
async def test_skips_a_sleep_after_a_request_in_the_last_10_minutes(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    busy(env, v["id"], NOW.timestamp() - 4 * 60)
    quiet(env, v["id"], NOW, minutes=1)  # idle right now
    assert await env.svc.run_due(NOW) == "1 deferred"
    assert state(env, v["id"]) == "ready"
    r = row(env, s["id"])
    assert r.last_result.startswith("deferred: a request arrived 4 min ago") and r.last_run_date is None
    assert audits(env)[0][2]["status"] == "deferred"
    assert any("deferred: a request arrived" in m for m in events(env, v["id"]))


@pytest.mark.parametrize("running,waiting", [(1.0, 0.0), (0.0, 2.0)])
async def test_skips_while_requests_run_or_wait(env, running, waiting):
    v = await ready(env, "a", sleep=False)
    s = add(env, v, "stop", "10:00")
    quiet(env, v["id"])
    busy(env, v["id"], NOW.timestamp() - 2, running=running, waiting=waiting)
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == "ready" and "running" in row(env, s["id"]).last_result


async def test_skips_when_the_metrics_cannot_tell(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    await env.svc.run_due(NOW)  # no samples at all
    assert state(env, v["id"]) == "ready" and "no fresh metrics" in row(env, s["id"]).last_result
    t = add(env, v, "sleep", "10:30", days=0b10)  # Tuesday: stale samples only
    tue = NOW + timedelta(days=1, minutes=30)
    quiet(env, v["id"], tue - timedelta(minutes=5))
    await env.svc.run_due(tue)
    assert state(env, v["id"]) == "ready" and "no fresh metrics" in row(env, t["id"]).last_result


async def test_skips_with_too_few_covered_minutes(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"], minutes=3)  # e.g. DGXMM just restarted: 3 of the last 15 minutes covered
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == "ready" and "not enough recent metrics" in row(env, s["id"]).last_result


# ---------------- never ----------------
async def test_never_acts_on_a_system_deployment(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    with env.db.session() as sess:
        sess.get(Deployment, v["id"]).system = True
    quiet(env, v["id"])
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == "ready" and "system deployment" in row(env, s["id"]).last_result


async def test_already_in_the_target_state_is_a_noop(env):
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    s = add(env, v, "sleep", "10:00")
    calls = len(env.http.calls)
    await env.svc.run_due(NOW)
    assert row(env, s["id"]).last_result == "skipped: already sleeping; nothing to do"
    assert [c for c in env.http.calls[calls:] if c[2] in ("/sleep", "/wake_up")] == []


@pytest.mark.parametrize("st", ["starting", "loading", "queued", "stopping", "unknown"])
async def test_skips_mid_transition(env, st):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    with env.db.session() as sess:
        sess.get(Deployment, v["id"]).state = st
    quiet(env, v["id"])
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == st and "mid-transition" in row(env, s["id"]).last_result


async def test_skips_while_waiting_for_a_startup_slot(env):
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    s = add(env, v, "wake", "10:00")
    env.deployer.gate.info = lambda dep_id: {"kind": "wake", "reason": "x"} if dep_id == v["id"] else None
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == "sleeping" and "start-up slot" in row(env, s["id"]).last_result


async def test_wake_that_does_not_fit_is_skipped_with_an_insight_and_evicts_nothing(env):
    from dgxmm.decider.insights import InsightService
    env.ctx.insights = InsightService(env.db, env.events)
    a = await ready(env, "a")
    b = await ready(env, "b")
    await env.deployer.sleep(a["id"])
    s = add(env, a, "wake", "10:00")
    env.probe.available = int(0.5 * GIB)  # the host filled up meanwhile
    await env.svc.run_due(NOW)
    assert state(env, a["id"]) == "sleeping" and state(env, b["id"]) == "ready"
    assert "does not fit" in row(env, s["id"]).last_result and "nothing was stopped" in row(env, s["id"]).last_result
    with env.db.session() as sess:
        ins = sess.scalar(select(Insight).where(Insight.dedupe_key == f"schedule_fit:{s['id']}"))
    assert ins.severity == "warning" and ins.action == {"type": "open", "path": f"/deployments/{a['id']}"}


async def test_start_that_does_not_fit_is_skipped(env):
    v = await ready(env, "a", sleep=False)
    await env.deployer.stop(v["id"])
    s = add(env, env.deployer.view(v["id"]), "start", "10:00")
    env.probe.available = int(0.5 * GIB)
    await env.svc.run_due(NOW)
    assert state(env, v["id"]) == "stopped" and "does not fit" in row(env, s["id"]).last_result


async def test_two_wakes_at_once_only_what_fits(env):
    free_on_stop(env.deployer, env.probe)
    a = await ready(env, "a")
    b = await ready(env, "b")
    for d in (a, b):
        await env.deployer.sleep(d["id"])
    add(env, a, "wake", "10:00")
    add(env, b, "wake", "10:00")
    need = env.deployer.view(a["id"])["reserved_bytes"]
    real_refresh = env.budget.refresh

    async def tight():  # room for one more wake-up only
        bud = await real_refresh()
        woken = sum(1 for d in (a, b) if state(env, d["id"]) == "ready")
        return dataclasses.replace(bud, free_for_new=need + GIB if woken == 0 else 0)
    env.budget.refresh = tight
    await env.svc.run_due(NOW)
    assert sorted(state(env, d["id"]) for d in (a, b)) == ["ready", "sleeping"]


# ---------------- once per slot, DST, restarts ----------------
async def test_dst_spring_gap_runs_once_just_after_the_gap(env):
    v = await ready(env, "a")
    env.clock.now = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
    s = add(env, v, "sleep", "02:30", tz="Europe/Berlin")  # 02:00-03:00 does not exist on 2026-03-29
    t0 = datetime(2026, 3, 29, 0, 45, tzinfo=timezone.utc)  # 01:45 CET
    quiet(env, v["id"], t0)
    assert await env.svc.run_due(t0) == "nothing due"
    t1 = datetime(2026, 3, 29, 1, 30, 20, tzinfo=timezone.utc)  # 03:30 CEST
    quiet(env, v["id"], t1)
    assert await env.svc.run_due(t1) == "1 ran"
    await env.deployer.wake(v["id"])
    t2 = t1 + timedelta(minutes=1)
    quiet(env, v["id"], t2)
    assert await env.svc.run_due(t2) == "nothing due"
    assert row(env, s["id"]).last_run_date == "2026-03-29"


async def test_dst_autumn_overlap_runs_once(env):
    v = await ready(env, "a")
    env.clock.now = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    add(env, v, "sleep", "02:30", tz="Europe/Berlin")  # 02:30 happens twice on 2026-10-25
    first = datetime(2026, 10, 25, 0, 30, 10, tzinfo=timezone.utc)  # 02:30 CEST
    quiet(env, v["id"], first)
    assert await env.svc.run_due(first) == "1 ran"
    await env.deployer.wake(v["id"])
    second = datetime(2026, 10, 25, 1, 30, 10, tzinfo=timezone.utc)  # 02:30 CET
    quiet(env, v["id"], second)
    assert await env.svc.run_due(second) == "nothing due"
    assert state(env, v["id"]) == "ready" and len(audits(env)) == 1


async def test_restart_catch_up_within_15_minutes_runs(env):
    v = await ready(env, "a")
    add(env, v, "sleep", "10:00")
    later = NOW + timedelta(minutes=12)  # DGXMM was down 09:58-10:12
    quiet(env, v["id"], later)
    assert await env.svc.run_due(later) == "1 ran"
    assert state(env, v["id"]) == "sleeping"


async def test_restart_older_than_15_minutes_is_skipped_and_logged(env, caplog):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    later = NOW + timedelta(minutes=40)
    quiet(env, v["id"], later)
    assert await env.svc.run_due(later) == "1 missed"
    assert state(env, v["id"]) == "ready"
    r = row(env, s["id"])
    assert r.last_result.startswith("skipped: missed by 40 min") and r.last_run_date == "2026-10-19"
    assert any("missed by 40 min" in m for m in events(env, v["id"]))
    assert "missed by 40 min" in caplog.text
    assert await env.svc.run_due(later + timedelta(minutes=1)) == "nothing due"  # logged once


async def test_a_slot_before_the_schedule_existed_is_not_missed(env):
    v = await ready(env, "a")
    env.clock.now = NOW + timedelta(minutes=30)
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"], NOW + timedelta(minutes=31))
    assert await env.svc.run_due(NOW + timedelta(minutes=31)) == "nothing due"
    assert row(env, s["id"]).last_result is None


async def test_paused_schedules_do_not_run(env):
    v = await ready(env, "a")
    add(env, v, "sleep", "10:00", enabled=False)
    quiet(env, v["id"])
    assert await env.svc.run_due(NOW) == "nothing due"


# ---------------- removed deployments ----------------
async def test_removing_a_deployment_deletes_its_schedules(env):
    v = await ready(env, "a")
    add(env, v, "sleep", "10:00")
    await env.deployer.stop(v["id"])
    await env.deployer.remove(v["id"])
    with env.db.session() as sess:
        assert sess.scalars(select(DeploymentSchedule)).all() == []


async def test_runner_drops_schedules_of_a_vanished_deployment(env):
    from dgxmm.deployer.service import DeployError
    v = await ready(env, "a")
    add(env, v, "sleep", "10:00")
    real = env.deployer.view

    def gone(dep_id, **kw):
        if dep_id == v["id"]:
            raise DeployError("not_found", "gone", 404)
        return real(dep_id, **kw)
    env.deployer.view = gone
    assert await env.svc.run_due(NOW) == "1 removed"
    with env.db.session() as sess:
        assert sess.scalars(select(DeploymentSchedule)).all() == []


# ---------------- collisions ----------------
async def test_colliding_slots_after_a_restart_run_only_the_latest(env):
    v = await ready(env, "a")
    sl = add(env, v, "sleep", "10:00")
    wk = add(env, v, "wake", "10:05")
    later = NOW + timedelta(minutes=7)
    quiet(env, v["id"], later)
    out = await env.svc.run_due(later)
    assert out == "1 skipped, 1 superseded"  # the wake is a no-op: it is awake
    assert row(env, sl["id"]).last_result.startswith(f"skipped: superseded by schedule #{wk['id']}")
    assert state(env, v["id"]) == "ready"


async def test_opposite_actions_at_the_same_instant_cancel(env):
    v = await ready(env, "a")
    a = add(env, v, "sleep", "10:00", tz="UTC")
    b = add(env, v, "wake", "11:00", tz="Europe/London")  # BST in October: the same instant
    quiet(env, v["id"])
    assert await env.svc.run_due(NOW) == "2 conflict"
    assert state(env, v["id"]) == "ready"
    for s in (a, b):
        assert "conflicts with schedule" in row(env, s["id"]).last_result


# ---------------- fix round 1 ----------------
async def test_feature_toggle_gates_only_profiles_and_proposals(env):
    """The schedule feature's toggle (and the decision model) never stop user-owned schedules."""
    from dgxmm.decider.features import wire_schedules
    from tests.fakes import FakeDecider

    class Off(FakeDecider):
        def rules_enabled(self, feature):
            return feature != "schedule"
    env.ctx.decider = Off({}, available=False)
    env.ctx.schedules = env.svc
    wire_schedules(env.ctx)
    eng = env.ctx.insight_engine
    jobs = {j.name: j for j in eng._jobs.values()}
    assert eng._enabled(jobs["schedule_runner"]) is True
    assert eng._enabled(jobs["traffic_profiles"]) is False and eng._enabled(jobs["schedule_proposals"]) is False
    v = await ready(env, "a")
    add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    assert await env.svc.run_due(NOW) == "1 ran" and state(env, v["id"]) == "sleeping"


async def test_pause_all_runs_nothing(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    env.svc.set_paused_all(True)
    assert await env.svc.run_due(NOW) == "paused (all schedules)"
    assert state(env, v["id"]) == "ready" and row(env, s["id"]).last_result is None and audits(env) == []
    env.svc.set_paused_all(False)
    assert await env.svc.run_due(NOW) == "1 ran"


async def test_a_transient_error_reading_the_deployment_keeps_its_schedules(env):
    from sqlalchemy.exc import OperationalError
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    real = env.deployer.view

    def locked(dep_id, **kw):
        raise OperationalError("SELECT", {}, Exception("database is locked"))
    env.deployer.view = locked
    assert await env.svc.run_due(NOW) == "1 error"
    assert row(env, s["id"]) is not None and row(env, s["id"]).last_run_date is None
    env.deployer.view = real
    quiet(env, v["id"], NOW + timedelta(minutes=1))
    assert await env.svc.run_due(NOW + timedelta(minutes=1)) == "1 ran"  # retried at the next pass


def _insight(env, key):
    with env.db.session() as sess:
        return sess.scalar(select(Insight).where(Insight.dedupe_key == key))


@pytest.fixture
def with_insights(env):
    from dgxmm.decider.insights import InsightService
    env.ctx.insights = InsightService(env.db, env.events)
    return env


async def test_failed_scheduled_sleep_pauses_sleep_schedules_and_offers_stop_start(with_insights):
    env = with_insights
    v = await ready(env, "a")
    sl = add(env, v, "sleep", "10:00", days=0b0011111)
    wk = add(env, v, "wake", "07:45", days=0b0011111)
    quiet(env, v["id"])
    env.http.fail_paths.add("/sleep")
    assert await env.svc.run_due(NOW) == "1 failed"
    assert row(env, sl["id"]).last_result.startswith("failed: sleep refused")
    assert row(env, sl["id"]).enabled is False and row(env, wk["id"]).enabled is False
    ins = _insight(env, f"schedule_failed:{sl['id']}")
    assert ins.severity == "warning" and ins.subject == f"deployment:{v['id']}"
    assert ins.action["proposal"] == {"sleep_action": "stop", "sleep_at": "10:00", "sleep_days": 31,
                                      "wake_action": "start", "wake_at": "07:45", "wake_days": 31, "timezone": "UTC"}
    # the offered stop/start pair can be created next to the paused sleep/wake pair
    env.svc.create(v, [ScheduleIn("stop", 31, "10:00", "UTC"), ScheduleIn("start", 31, "07:45", "UTC")], 1)


async def test_failed_scheduled_wake_raises_an_insight(with_insights):
    env = with_insights
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    wk = add(env, v, "wake", "10:00")
    env.http.fail_paths.add("/wake_up")
    assert await env.svc.run_due(NOW) == "1 failed"
    assert _insight(env, f"schedule_failed:{wk['id']}").title == "Schedule could not wake a"
    assert row(env, wk["id"]).enabled is False


async def test_a_queued_wake_that_later_fails_records_the_real_outcome(with_insights):
    env = with_insights
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    wk = add(env, v, "wake", "10:00")
    real = env.deployer.wake

    async def queued(dep_id, user_id=None):  # waited wake_sync_wait_s: continues in the background
        return {**env.deployer.view(dep_id), "startup_queue": {"kind": "wake", "reason": "x"}}
    env.deployer.wake = queued
    assert await env.svc.run_due(NOW) == "1 pending"
    assert row(env, wk["id"]).last_result.startswith("pending: waking up")
    with env.db.session() as sess:  # the background wake-up then failed (deployer._wake_done records both)
        sess.get(Deployment, v["id"]).last_error = "wake failed: needs 6.0 GiB but only 0.5 GiB is free"
    env.deployer.wake_failures[v["id"]] = ("insufficient_memory", "needs 6.0 GiB but only 0.5 GiB is free")
    env.deployer.wake = real
    assert await env.svc.run_due(NOW + timedelta(minutes=1)) == "1 skipped"
    assert row(env, wk["id"]).last_result.startswith("skipped: does not fit in free memory (needs 6.0 GiB")
    assert _insight(env, f"schedule_fit:{wk['id']}").severity == "warning" and row(env, wk["id"]).enabled
    assert audits(env)[-1][2]["status"] == "skipped"


async def test_traffic_skips_retry_every_minute_then_run(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    busy(env, v["id"], NOW.timestamp() - 4 * 60)
    quiet(env, v["id"], NOW, minutes=1)
    for m in range(0, 6):  # 10:00 .. 10:05: a request 4 .. 9 minutes ago
        assert await env.svc.run_due(NOW + timedelta(minutes=m)) == "1 deferred"
        quiet(env, v["id"], NOW + timedelta(minutes=m + 1), minutes=1)
    assert len([m for m in events(env, v["id"]) if "deferred" in m]) == 1  # one event, not one per retry
    assert row(env, s["id"]).last_result.endswith("(retrying until 11:00)")
    assert await env.svc.run_due(NOW + timedelta(minutes=7)) == "1 ran"  # 11 quiet minutes
    assert state(env, v["id"]) == "sleeping" and row(env, s["id"]).last_run_date == "2026-10-19"


async def test_traffic_skips_give_up_after_60_minutes(env):
    v = await ready(env, "a")
    s = add(env, v, "stop", "10:00") if not v["sleep_capable"] else add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    busy(env, v["id"], NOW.timestamp() - 1)
    assert await env.svc.run_due(NOW) == "1 deferred"
    later = NOW + timedelta(minutes=61)
    assert await env.svc.run_due(later) == "1 gave up"
    r = row(env, s["id"])
    assert r.last_result.startswith("skipped: gave up after 60 min: 1 request(s) running") and r.last_run_date
    assert await env.svc.run_due(later + timedelta(minutes=1)) == "nothing due"


async def test_wake_does_not_start_a_deployment_the_user_stopped(env):
    v = await ready(env, "a")
    await env.deployer.stop(v["id"], 1)  # stopped by hand (reason "user")
    s = add(env, env.deployer.view(v["id"]), "wake", "10:00")
    assert await env.svc.run_due(NOW) == "1 skipped"
    assert row(env, s["id"]).last_result == "skipped: stopped by user since the last scheduled sleep; not started"
    assert state(env, v["id"]) == "stopped"


async def test_wake_starts_what_a_schedule_stopped(env):
    v = await ready(env, "a", sleep=False)
    env.clock.now = NOW - timedelta(hours=2)
    add(env, v, "stop", "09:00")
    quiet(env, v["id"], NOW - timedelta(hours=1))
    await env.svc.run_due(NOW - timedelta(hours=1) + timedelta(seconds=10))
    assert state(env, v["id"]) == "stopped"
    with env.db.session() as sess:  # an old wake row from before sleep mode was turned off
        sess.add(DeploymentSchedule(deployment_id=v["id"], action="wake", days=127, at_local="10:00", timezone="UTC",
                                    enabled=True, created_at=NOW - timedelta(days=1), updated_at=NOW - timedelta(days=1)))
    assert await env.svc.run_due(NOW) == "1 pending"
    assert await env.deployer.wait_ready(v["id"], 5) == "ready"


async def test_editing_the_time_does_not_run_twice_the_same_day(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    await env.svc.run_due(NOW)
    await env.deployer.wake(v["id"])
    env.clock.now = NOW + timedelta(minutes=2)
    env.svc.update(s["id"], env.deployer.view(v["id"]), {"at_local": "10:05"})
    quiet(env, v["id"], NOW + timedelta(minutes=6))
    assert await env.svc.run_due(NOW + timedelta(minutes=6)) == "nothing due"
    assert state(env, v["id"]) == "ready"


async def test_traffic_is_rechecked_inside_the_deployer_lock(env):
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])
    runner_cls = __import__("dgxmm.decider.schedule", fromlist=["ScheduleRunner"]).ScheduleRunner
    calls = []
    orig = runner_cls._traffic_reason

    def racing(self, view, now):
        calls.append(1)
        return None if len(calls) == 1 else "a request arrived 0 min ago (needs 10 quiet minutes)"
    runner_cls._traffic_reason = racing
    try:
        assert await env.svc.run_due(NOW) == "1 deferred"
    finally:
        runner_cls._traffic_reason = orig
    assert len(calls) == 2 and state(env, v["id"]) == "ready"
    assert row(env, s["id"]).last_result.startswith("deferred: a request arrived 0 min ago")


# ---------------- fix round 2 ----------------
async def _pending_wake(env, v, sched):
    """A wake that waited wake_sync_wait_s: it continues in the background (pending)."""
    real = env.deployer.wake

    async def queued(dep_id, user_id=None):
        return {**env.deployer.view(dep_id), "startup_queue": {"kind": "wake", "reason": "x"}}
    env.deployer.wake = queued
    assert await env.svc.run_due(NOW) == "1 pending"
    env.deployer.wake = real


async def test_pending_start_is_never_failed_while_still_loading(with_insights):
    env = with_insights
    v = await ready(env, "a", sleep=False)
    await env.deployer.stop(v["id"])
    s = add(env, env.deployer.view(v["id"]), "start", "10:00")

    async def slow_start(dep_id, user_id=None, **kw):
        with env.db.session() as sess:
            sess.get(Deployment, dep_id).state = "loading"
        return env.deployer.view(dep_id)
    env.deployer.restart = slow_start
    assert await env.svc.run_due(NOW) == "1 pending"
    for m in (16, 40, 90):  # a big model loading for an hour and a half is still on its way
        assert await env.svc.run_due(NOW + timedelta(minutes=m)) == "nothing due"
    assert row(env, s["id"]).last_result.startswith("pending:") and _insight(env, f"schedule_failed:{s['id']}") is None
    with env.db.session() as sess:
        sess.get(Deployment, v["id"]).state = "ready"
    assert await env.svc.run_due(NOW + timedelta(minutes=91)) == "1 ran"
    assert row(env, s["id"]).last_result.startswith("ok: started")


async def test_overdue_wake_is_reported_once_then_its_real_outcome_recorded(with_insights):
    env = with_insights
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    s = add(env, v, "wake", "10:00")
    await _pending_wake(env, v, s)
    never = asyncio.get_running_loop().create_future()
    env.deployer._waking[v["id"]] = never  # still waiting for its start-up slot
    assert await env.svc.run_due(NOW + timedelta(minutes=10)) == "nothing due"
    assert await env.svc.run_due(NOW + timedelta(minutes=16)) == "1 overdue"
    assert await env.svc.run_due(NOW + timedelta(minutes=17)) == "nothing due"  # reported once
    ins = _insight(env, f"schedule_failed:{s['id']}")
    assert ins.status == "open" and row(env, s["id"]).enabled is True  # not blamed on sleep mode
    never.cancel()
    del env.deployer._waking[v["id"]]
    await env.deployer.wake(v["id"])  # the wake-up finally got its slot
    assert await env.svc.run_due(NOW + timedelta(minutes=20)) == "1 ran"
    assert row(env, s["id"]).last_result.startswith("ok: woken up")
    assert _insight(env, f"schedule_failed:{s['id']}").status == "expired"  # the false alarm is resolved


@pytest.mark.parametrize("code,status,paused,insight", [
    ("startup_busy", "failed: queue:", False, True),
    ("not_sleeping", "skipped: state changed:", False, False),
    ("engine_error", "failed: wake failed:", True, True),
])
async def test_background_wake_failures_are_classified(with_insights, code, status, paused, insight):
    env = with_insights
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    sl = add(env, v, "sleep", "23:00")
    s = add(env, v, "wake", "10:00")
    await _pending_wake(env, v, s)
    env.deployer.wake_failures[v["id"]] = (code, "something")
    await env.svc.run_due(NOW + timedelta(minutes=1))
    assert row(env, s["id"]).last_result.startswith(status)
    assert row(env, sl["id"]).enabled is (not paused)
    assert (_insight(env, f"schedule_failed:{s['id']}") is not None) is insight


async def test_a_wake_lost_with_a_dgxmm_restart_is_retried(with_insights):
    env = with_insights
    v = await ready(env, "a")
    await env.deployer.sleep(v["id"])
    sl = add(env, v, "sleep", "23:00")
    s = add(env, v, "wake", "10:00")
    await _pending_wake(env, v, s)
    # DGXMM restarted: no wake task, no recorded failure, the deployment still sleeps
    assert await env.svc.run_due(NOW + timedelta(minutes=2)) == "1 ran"
    assert state(env, v["id"]) == "ready" and row(env, sl["id"]).enabled is True
    assert row(env, s["id"]).last_result == "ok: woken up"


async def test_pause_all_is_rechecked_before_each_slot(env):
    a = await ready(env, "a")
    b = await ready(env, "b")
    for d in (a, b):
        add(env, d, "sleep", "10:00")
        quiet(env, d["id"])
    calls = []
    real = env.svc.paused_all

    def paused():
        calls.append(1)
        return len(calls) > 2  # paused by the admin after the first slot ran
    env.svc.paused_all = paused
    assert await env.svc.run_due(NOW) == "1 paused, 1 ran"
    env.svc.paused_all = real
    assert sorted(state(env, d["id"]) for d in (a, b)) == ["ready", "sleeping"]


async def test_a_concurrent_user_change_is_state_changed_without_a_warning(with_insights):
    from dgxmm.deployer.service import DeployError
    env = with_insights
    v = await ready(env, "a")
    s = add(env, v, "sleep", "10:00")
    quiet(env, v["id"])

    async def raced(dep_id, user_id=None, precheck=None):
        raise DeployError("not_ready", "deployment is stopping", 409)
    env.deployer.sleep = raced
    assert await env.svc.run_due(NOW) == "1 skipped"
    assert row(env, s["id"]).last_result == "skipped: state changed: deployment is stopping"
    assert _insight(env, f"schedule_failed:{s['id']}") is None and row(env, s["id"]).enabled is True


async def test_who_stopped_a_deployment_is_a_structured_field_not_message_text(env):
    """Security review L8: the wake-after-stop check reads the event's actor, never its free text."""
    from dgxmm.db import DeploymentEvent
    v = await ready(env, "a", sleep=False)
    await env.deployer.stop(v["id"], None, reason="schedule:7")
    with env.db.session() as sess:
        ev = [(e.to_state, e.actor) for e in sess.query(DeploymentEvent).filter_by(deployment_id=v["id"])
              if e.to_state in ("stopping", "stopped")]
    assert ("stopping", "schedule:7") in ev
    from dgxmm.decider.schedule import ScheduleRunner
    assert ScheduleRunner(env.svc)._stopped_by_schedule(v["id"]) is True
    with env.db.session() as sess:  # a message that merely looks like a schedule's stop, from a user's stop
        sess.add(DeploymentEvent(deployment_id=v["id"], from_state="ready", to_state="stopping",
                                 message="stop requested (schedule:7)", actor="user"))
    assert ScheduleRunner(env.svc)._stopped_by_schedule(v["id"]) is False
