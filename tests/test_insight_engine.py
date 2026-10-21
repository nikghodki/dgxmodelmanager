"""The background insight engine (dgxmm/decider/insight_engine.py): cadences, skip-while-running, budget, the
decider lane, disabled features, failure backoff with one deduplicated "degraded" insight, and job status."""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest

from dgxmm.decider import service
from dgxmm.decider.insight_engine import Cadence, InsightEngine
from dgxmm.decider.types import Option
from tests.fakes import FakeDecider

OPTS = [Option("a", "A"), Option("b", "B")]


class Clock:
    def __init__(self, t=datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)):
        self.now = t

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class Toggles(FakeDecider):
    """FakeDecider with the real Decider's rules_enabled (toggle on, decider maybe off)."""

    def __init__(self, *a, rules_off=(), **kw):
        super().__init__(*a, **kw)
        self.rules_off = set(rules_off)

    def rules_enabled(self, feature):
        return feature not in self.rules_off


@pytest.fixture
def eng(ctx):
    clock = Clock()
    ctx.decider = Toggles({"health": ("a", 0.95)})
    e = InsightEngine(ctx, clock=clock, tz=timezone.utc)
    e.clock_ = clock
    # The engine writes expires_at from its clock; the store expires rows against its own clock. In production
    # both are utcnow. With a fixed fake engine clock and a real store clock, a 6 h degraded TTL computed from
    # 2026-10-19 12:00 expired as soon as the real date passed 2026-10-19 18:00 UTC.
    ctx.insights.clock = clock
    return e


def test_engine_and_store_share_one_clock(eng):
    assert eng.ctx.insights.clock is eng.clock


async def test_degraded_insight_expires_by_the_shared_clock(eng):
    async def broken(run):
        raise RuntimeError("x")
    eng.register_job("h", "health", 60, broken)
    for _ in range(3):
        eng.trigger("h")
        await eng.tick()
        await eng.wait_idle()
    (row,) = [r for r in eng.ctx.insights.list() if r["status"] == "open"]
    eng.clock_.advance(hours=5, minutes=59)
    assert [r["status"] for r in eng.ctx.insights.list() if r["id"] == row["id"]] == ["open"]
    eng.clock_.advance(minutes=2)
    assert [r["status"] for r in eng.ctx.insights.list() if r["id"] == row["id"]] == ["expired"]


def test_cadence_parsing_and_next_times():
    c = Cadence.parse(300)
    t = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    assert c.describe() == "every 300 s" and c.next_after(t, timezone.utc) == t + timedelta(seconds=300)
    d = Cadence.parse("daily 03:30")
    assert d.next_after(t, timezone.utc) == datetime(2026, 10, 20, 3, 30, tzinfo=timezone.utc)
    w = Cadence.parse("weekly sun 04:00")  # 2026-10-19 is a Monday
    assert w.next_after(t, timezone.utc) == datetime(2026, 10, 25, 4, 0, tzinfo=timezone.utc)
    assert w.describe() == "weekly sun 04:00"
    for bad in ("hourly", "daily 25:00", "weekly xyz 01:00", 0, -5, "daily"):
        with pytest.raises(ValueError):
            Cadence.parse(bad)


async def test_interval_job_runs_when_due(eng):
    runs = []

    async def job(run):
        runs.append(run.name)
        return "checked 2 deployments"

    eng.register_job("health_check", "health", 300, job, budget_calls=4)
    assert await eng.tick() == []  # first run after min(interval, 60 s)
    eng.clock_.advance(seconds=61)
    assert await eng.tick() == ["health_check"]
    await eng.wait_idle()
    st = eng.status()[0]
    assert runs == ["health_check"] and st["last_result"] == "ok: checked 2 deployments" and st["failures"] == 0
    assert st["next_run_at"] == (eng.clock_.now + timedelta(seconds=300)).isoformat()
    assert await eng.tick() == []


async def test_skips_while_the_previous_run_is_active(eng):
    gate = asyncio.Event()
    starts = []

    async def slow(run):
        starts.append(1)
        await gate.wait()

    eng.register_job("slow", "health", 60, slow)
    eng.trigger("slow")
    assert await eng.tick() == ["slow"]
    eng.trigger("slow")
    eng.clock_.advance(seconds=600)
    for _ in range(5):
        assert await eng.tick() == []  # still running: skipped
    assert eng.status()[0]["running"] and eng.status()[0]["skipped"] == 1  # one missed slot, not one per tick
    gate.set()
    await eng.wait_idle()
    assert len(starts) == 1


async def test_budget_limits_decider_calls_and_uses_the_lane(eng):
    lanes = []
    inner = eng.ctx.decider

    async def spy(**kw):
        lanes.append(service.LANE.get().job if service.LANE.get() else None)
        return await Toggles.decide(inner, **kw)
    inner.decide = spy

    async def job(run):
        out = [await run.decider.decide(feature="health", subject="s", state=f"x{i}", question="q", options=OPTS)
               for i in range(5)]
        return f"{sum(d.source == 'model' for d in out)} answered"

    eng.register_job("budgeted", "health", 60, job, budget_calls=2)
    eng.trigger("budgeted")
    await eng.tick()
    await eng.wait_idle()
    st = eng.status()[0]
    assert st["last_result"] == "ok: 2 answered" and st["calls_used"] == 2 and st["calls_refused"] == 3
    assert lanes == ["budgeted"] * 5  # every call ran in the job context; 3 abstained inside decide()
    assert service.LANE.get() is None  # not leaked outside the job


async def test_cpu_work_of_sync_jobs_runs_off_the_event_loop(eng):
    seen = {}

    def cpu(run):
        seen["thread"] = threading.current_thread() is not threading.main_thread()
        seen["lane"] = service.LANE.get().job
        return "done"

    eng.register_job("cpu", "tuning", "weekly mon 03:00", cpu)
    eng.trigger("cpu")
    await eng.tick()
    await eng.wait_idle()
    assert seen == {"thread": True, "lane": "cpu"} and eng.status()[0]["last_result"] == "ok: done"


async def test_disabled_feature_jobs_do_not_run(eng):
    eng.ctx.decider.rules_off = {"health"}
    ran = []

    async def job(run):
        ran.append(1)

    eng.register_job("h", "health", 60, job)
    eng.register_job("needs_model", "routing", 60, job, requires_decider=True)
    eng.ctx.decider.disabled = {"routing"}
    eng.trigger("h")
    eng.trigger("needs_model")
    assert await eng.tick() == []
    assert [s["enabled"] for s in eng.status()] == [False, False] and ran == []
    eng.ctx.decider.rules_off = set()
    eng.clock_.advance(seconds=3600)
    assert await eng.tick() == []  # switched on: the dropped trigger and the missed slot do not run at once
    assert eng.status()[0]["next_run_at"] == (eng.clock_.now + timedelta(seconds=60)).isoformat()
    eng.clock_.advance(seconds=61)
    assert await eng.tick() == ["h"]
    await eng.wait_idle()


async def test_reenabled_daily_job_waits_for_its_slot(eng):
    async def job(run):
        return None
    eng.register_job("nightly", "health", "daily 03:00", job)
    eng.ctx.decider.rules_off = {"health"}
    await eng.tick()
    eng.clock_.advance(days=2)  # its slots passed while it was off
    eng.ctx.decider.rules_off = set()
    assert await eng.tick() == []
    assert eng.status()[0]["next_run_at"] == "2026-10-22T03:00:00+00:00"


async def test_repeated_failures_raise_one_degraded_insight_and_back_off(eng):
    async def broken(run):
        raise RuntimeError("boom")

    eng.register_job("broken", "health", 60, broken)
    gaps = []
    for _ in range(5):
        eng.trigger("broken")
        assert await eng.tick() == ["broken"]
        await eng.wait_idle()
        st = eng.status()[0]
        gaps.append((datetime.fromisoformat(st["next_run_at"]) - eng.clock_.now).total_seconds())
    assert st["failures"] == 5 and st["last_result"] == "error: RuntimeError"
    assert gaps == sorted(gaps) and gaps[-1] > gaps[0] and gaps[-1] <= 3600
    rows = eng.ctx.insights.list()
    assert len(rows) == 1 and rows[0]["kind"] == "degraded" and rows[0]["feature"] == "health"
    assert rows[0]["dedupe_key"] == "degraded:broken" and "boom" not in rows[0]["body"]

    async def fixed(run):
        return None
    eng._jobs["broken"].fn = fixed
    eng.trigger("broken")
    await eng.tick()
    await eng.wait_idle()
    assert eng.status()[0]["failures"] == 0 and eng.ctx.insights.list()[0]["status"] == "expired"


async def test_timeout_counts_as_a_failure(eng):
    async def hang(run):
        await asyncio.sleep(10)

    eng.register_job("hang", "health", 60, hang, timeout_s=0.05)
    eng.trigger("hang")
    await eng.tick()
    await eng.wait_idle()
    assert eng.status()[0]["last_result"] == "error: timeout" and not eng.status()[0]["running"]


def test_register_validates(eng):
    async def f(run):
        pass
    eng.register_job("a", "health", 60, f)
    with pytest.raises(ValueError):
        eng.register_job("a", "health", 60, f)
    with pytest.raises(ValueError):
        eng.register_job("b", "nope", 60, f)
    with pytest.raises(ValueError):
        eng.register_job("c", "health", 60, f, budget_calls=-1)
    with pytest.raises(KeyError):
        eng.trigger("zzz")


async def test_run_loop_starts_and_stops(eng):
    ran = asyncio.Event()

    async def job(run):
        ran.set()

    eng.tick_s = 0.01
    eng.register_job("j", "health", 60, job)
    task = asyncio.create_task(eng.run())
    eng.trigger("j")
    await asyncio.wait_for(ran.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_status_without_jobs_is_empty(eng):
    assert eng.status() == []


async def test_status_route_exposes_jobs_to_admins_only(api_stack):
    ctx = api_stack.ctx

    async def job(run):
        return "ok"
    ctx.insight_engine.register_job("probe_job", "health", 300, job)
    from dgxmm.api.decider import status_out
    out = await status_out(ctx, admin=True)
    assert [j["name"] for j in out["jobs"]] == ["probe_job"] and out["jobs"][0]["schedule"] == "every 300 s"
    assert "jobs" not in await status_out(ctx, admin=False)
    assert [f["name"] for f in out["feature_specs"]][:5] == ["diagnosis", "parser", "plan", "kind", "workload"]


# ---------------- fix round 1 ----------------
async def test_budget_covers_every_path_to_the_decider(eng):
    """Not only run.decider: ctx.decider used directly (or by a feature service) inside a job counts too."""
    async def job(run):
        out = [await eng.ctx.decider.decide(feature="health", subject="s", state=f"x{i}", question="q", options=OPTS)
               for i in range(3)]
        out += [await run.decider.decide(feature="health", subject="s", state="y", question="q", options=OPTS)]
        return f"{sum(d.source == 'model' for d in out)} answered"

    eng.register_job("svc", "health", 60, job, budget_calls=2)
    eng.trigger("svc")
    await eng.tick()
    await eng.wait_idle()
    st = eng.status()[0]
    assert st["last_result"] == "ok: 2 answered" and (st["calls_used"], st["calls_refused"]) == (2, 2)
    assert len(eng.ctx.decider.calls) == 4


async def test_timed_out_sync_job_stays_running_until_its_thread_ends(eng):
    release = threading.Event()
    starts = []

    def stuck(run):
        starts.append(threading.current_thread().name)
        release.wait(5)

    eng.register_job("stuck", "health", 60, stuck, timeout_s=0.05)
    eng.trigger("stuck")
    await eng.tick()
    await asyncio.sleep(0.2)  # well past the timeout
    assert eng.status()[0]["running"]
    eng.trigger("stuck")
    eng.clock_.advance(seconds=600)
    assert await eng.tick() == []  # no second, overlapping thread
    release.set()
    await eng.wait_idle()
    st = eng.status()[0]
    assert len(starts) == 1 and starts[0].startswith("insight-job") and st["last_result"] == "error: timeout"
    assert not st["running"]


async def test_async_job_threads_are_awaited_after_a_timeout(eng):
    release = threading.Event()

    async def job(run):
        await run.to_thread(release.wait, 5)

    eng.register_job("a", "health", 60, job, timeout_s=0.05)
    eng.trigger("a")
    await eng.tick()
    await asyncio.sleep(0.2)
    assert eng.status()[0]["running"]
    release.set()
    await eng.wait_idle()
    assert eng.status()[0]["last_result"] == "error: timeout"


async def test_hung_jobs_do_not_exhaust_the_default_thread_pool(eng):
    release = threading.Event()

    def hang(run):
        release.wait(5)

    for i in range(4):
        eng.register_job(f"hang{i}", "health", 60, hang, timeout_s=0.05)
        eng.trigger(f"hang{i}")
    await eng.tick()
    await asyncio.sleep(0.1)
    assert await asyncio.wait_for(asyncio.to_thread(lambda: 42), 1) == 42  # asyncio's own pool is untouched
    release.set()
    await eng.wait_idle()


async def test_degraded_insight_is_resolved_by_any_success_and_expires(eng):
    eng.ctx.insights.upsert("degraded", "health", "job:h", "warning", "Health triage is degraded", "b",
                            dedupe_key="degraded:h", source="rules")  # left over from before a restart

    async def ok(run):
        return None
    eng.register_job("h", "health", 60, ok)
    eng.trigger("h")
    await eng.tick()
    await eng.wait_idle()
    assert eng.ctx.insights.list()[0]["status"] == "expired"

    async def broken(run):
        raise RuntimeError("x")
    eng._jobs["h"].fn = broken
    for _ in range(3):
        eng.trigger("h")
        await eng.tick()
        await eng.wait_idle()
    live = [r for r in eng.ctx.insights.list() if r["status"] == "open"]
    assert len(live) == 1 and live[0]["expires_at"] is not None


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


@pytest.mark.parametrize("zone,spec,now,want", [
    # Europe/Berlin, autumn 2026-10-25 (03:00 CEST -> 02:00 CET): the review's case, 03:00 local is 02:00 UTC
    ("Europe/Berlin", "daily 03:00", _utc(2026, 10, 24, 12), _utc(2026, 10, 25, 2)),
    ("Europe/Berlin", "weekly mon 03:00", _utc(2026, 10, 24, 12), _utc(2026, 10, 26, 2)),
    # overlap: 02:30 happens twice; only the first (CEST) occurrence runs, then the next day
    ("Europe/Berlin", "daily 02:30", _utc(2026, 10, 24, 12), _utc(2026, 10, 25, 0, 30)),
    ("Europe/Berlin", "daily 02:30", _utc(2026, 10, 25, 0, 30), _utc(2026, 10, 26, 1, 30)),
    # spring gap 2027-03-28 02:00 -> 03:00: 02:30 does not exist and runs at 03:30 CEST (01:30 UTC)
    ("Europe/Berlin", "daily 02:30", _utc(2027, 3, 27, 12), _utc(2027, 3, 28, 1, 30)),
    ("Europe/Berlin", "daily 03:00", _utc(2027, 3, 27, 12), _utc(2027, 3, 28, 1)),
    # America/Los_Angeles: spring 2027-03-14 (02:00 -> 03:00), autumn 2026-11-01 (01:00-02:00 twice)
    ("America/Los_Angeles", "daily 02:15", _utc(2027, 3, 13, 20), _utc(2027, 3, 14, 10, 15)),
    ("America/Los_Angeles", "daily 01:30", _utc(2026, 10, 31, 20), _utc(2026, 11, 1, 8, 30)),
    ("America/Los_Angeles", "daily 01:30", _utc(2026, 11, 1, 8, 30), _utc(2026, 11, 2, 9, 30)),
    ("America/Los_Angeles", "weekly sun 04:00", _utc(2026, 10, 31, 20), _utc(2026, 11, 1, 12)),
])
def test_wall_clock_schedules_follow_dst(zone, spec, now, want):
    from zoneinfo import ZoneInfo
    assert Cadence.parse(spec).next_after(now, ZoneInfo(zone)) == want


def test_local_zone_sources(monkeypatch, tmp_path):
    from zoneinfo import ZoneInfo

    from dgxmm.decider import insight_engine as ie
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert ie.local_zone("Europe/Berlin") == ZoneInfo("Europe/Berlin")
    assert ie.local_zone(None) == ZoneInfo("America/Los_Angeles")
    assert ie.local_zone("Not/AZone") == ZoneInfo("America/Los_Angeles")
    monkeypatch.delenv("TZ")
    monkeypatch.setattr(ie, "LOCALTIME", str(tmp_path / "missing"))
    assert ie.local_zone(None) == timezone.utc


def test_engine_uses_the_settings_zone(ctx):
    from zoneinfo import ZoneInfo
    ctx.settings.tz = "Europe/Berlin"
    assert InsightEngine(ctx).tz == ZoneInfo("Europe/Berlin")


async def test_settled_triggers_coalesce_into_one_run_and_a_burst_cannot_postpone_it_forever(eng):
    """``trigger(name, settle_s)``: a burst of change notices (many deployments changing state, one LiteLLM sync per
    change) runs the job once, ``settle_s`` after the last notice; a steady stream runs it after at most
    ``SETTLE_MAX_FACTOR * settle_s``; an immediate trigger is not delayed by a settling one."""
    from dgxmm.decider.insight_engine import SETTLE_MAX_FACTOR
    runs = []

    async def job(run):
        runs.append(eng.clock_())

    eng.register_job("settled", "health", "daily 03:00", job)
    for _ in range(20):
        eng.trigger("settled", settle_s=10)
        eng.clock_.advance(seconds=1)
        assert await eng.tick() == []  # still settling
    eng.clock_.advance(seconds=8)
    assert await eng.tick() == []  # 9 s since the last notice
    eng.clock_.advance(seconds=1)
    assert await eng.tick() == ["settled"]
    await eng.wait_idle()
    assert await eng.tick() == [] and len(runs) == 1
    # a notice every 5 s, never 10 s quiet: still runs within SETTLE_MAX_FACTOR * settle_s of the first one
    first = eng.clock_()
    while True:
        eng.trigger("settled", settle_s=10)
        if await eng.tick():
            break
        eng.clock_.advance(seconds=5)
        assert eng.clock_() - first <= timedelta(seconds=SETTLE_MAX_FACTOR * 10)
    await eng.wait_idle()
    assert len(runs) == 2
    eng.trigger("settled", settle_s=10)
    eng.trigger("settled")  # someone needs it now (e.g. the API without advice)
    assert await eng.tick() == ["settled"]
    await eng.wait_idle()
    eng.trigger("settled")
    eng.trigger("settled", settle_s=10)  # a later settling notice does not delay an immediate request
    assert await eng.tick() == ["settled"]
