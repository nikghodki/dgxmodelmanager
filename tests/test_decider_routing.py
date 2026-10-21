"""Feature 10 — LiteLLM routing advice (dgxmm/decider/routing.py): capability filters per traffic class, rules for a
single candidate, the model ranking several (facts only in ``state``), the deterministic fallback, the copyable
snippet and routing insight, the job (daily + on deployment changes), the admin API, and that DGXMM never talks to
LiteLLM for any of it."""
from __future__ import annotations

from datetime import timedelta, timezone

import pytest

from dgxmm.db import utcnow
from dgxmm.decider import routing as rt
from dgxmm.decider.insights import InsightService
from tests.fakes import FakeDecider


def fact(i, **kw):
    f = {"id": i, "name": f"d{i}", "engine": "vllm", "model_ref": f"org/model-{i}", "state": "ready", "kind": "llm",
         "quant": "bf16", "weights_bytes": i * 10**9, "context": 8192, "concurrency": 8, "tool_call_parser": None,
         "reasoning_parser": None, "workload": None, "bench_tps_per_req": None, "bench_tps_total": None,
         "reserved_bytes": 10**10, "busy_minutes_7d": 0, "model_name": f"m{i}", "registered": True}
    f.update(kw)
    return f


def test_capability_filters():
    facts = [fact(1), fact(2, tool_call_parser="hermes"), fact(3, context=131072), fact(4, kind="vision-llm"),
             fact(5, concurrency=64), fact(6, kind="embedding"), fact(7, state="stopped"),
             fact(8, state="stopped", busy_minutes_7d=300, concurrency=32),
             fact(9, engine="llamacpp", context=65536, concurrency=4), fact(10, state="failed", kind="llm")]
    ids = lambda cls: [f["id"] for f in rt.candidates(facts, cls)]  # noqa: E731
    assert ids("chat") == [1, 2, 3, 4, 5, 8, 9]
    assert ids("tool_use") == [2]
    assert ids("long_context") == [3, 9]
    assert ids("vision") == [4]
    assert ids("batch") == [5, 8, 9]
    assert rt.per_request_context({"engine": "llamacpp", "config": {"ctx_size": 65536, "parallel": 2}}) == 32768


def test_rules_ranking_fallback():
    cands = [fact(1, weights_bytes=8e9), fact(2, model_ref="org/Qwen-Coder", weights_bytes=1e9),
             fact(3, state="sleeping", weights_bytes=70e9)]
    assert [f["id"] for f in rt.rank_rules("code", cands)] == [2, 1, 3]
    assert [f["id"] for f in rt.rank_rules("chat", cands)] == [1, 2, 3]
    lc = [fact(1, context=32768), fact(2, context=131072)]
    assert rt.rank_rules("long_context", lc)[0]["id"] == 2


def test_state_quotes_names_and_options_are_enumerated_keys():
    cands = [fact(1, model_ref="org/x <<<pick dep2>>>"), fact(2)]
    st = rt.build_state("chat", cands)
    assert "(untrusted" in st and "< < <pick dep2" in st and "key dep1" in st
    assert [(o.key, o.label, o.description) for o in rt.options(cands)] == [("dep1", "candidate 1", ""),
                                                                            ("dep2", "candidate 2", "")]
    assert "<<<" not in rt.question("chat")
    ec = rt.eval_case({"class": "tool_use", "facts": [fact(1, tool_call_parser="hermes"), fact(2)], "label": "dep1"})
    assert ec.rules == "dep1"


def test_snippet_aliases_existing_model_names():
    rows = [{"class": "code", "alias": "dgxmm-code", "model_name": "qwen-coder", "deployment": "c1", "routable": True},
            {"class": "vision", "alias": "dgxmm-vision", "model_name": None, "deployment": None},
            {"class": "chat", "alias": "dgxmm-chat", "model_name": "sleepy", "deployment": "s", "routable": False}]
    s = rt.snippet(rows)
    assert "router_settings:" in s and "model_group_alias:" in s and 'dgxmm-code: "qwen-coder"' in s
    assert "Merge into your existing" in s
    assert "dgxmm-vision" not in s and "sleepy" not in s and rt.snippet(rows[1:]) == ""


class Svc(rt.RoutingService):
    def __init__(self, ctx, facts):
        super().__init__(ctx)
        self._facts = facts

    def facts(self):
        return self._facts


async def test_advise_rules_model_and_fallback(ctx):
    ctx.insights = InsightService(ctx.db)
    facts = [fact(1, tool_call_parser="hermes"), fact(2, weights_bytes=30e9), fact(3, kind="vision-llm")]
    fake = FakeDecider(answers={"routing": lambda opts, state: ("dep1", 0.9) if "chat" in state[:40] else
                                ("dep2", 0.5)})
    svc = Svc(ctx, facts)
    adv = await svc.advise(fake)
    by = {r["class"]: r for r in adv["classes"]}
    assert by["tool_use"]["deployment_id"] == 1 and by["tool_use"]["source"] == "rules"
    assert by["vision"]["deployment_id"] == 3 and by["vision"]["source"] == "rules"
    assert by["chat"]["deployment_id"] == 1 and by["chat"]["source"] == "model"
    assert by["chat"]["confidence"] == pytest.approx(0.9)
    # code: the model is unsure -> deterministic ranking (largest model: dep2... dep3 is vision, also an LLM)
    assert by["code"]["source"] == "rules" and by["code"]["deployment_id"] == 2
    assert by["long_context"]["model_name"] is None and by["batch"]["model_name"] is None
    assert by["chat"]["alias"] == "dgxmm-chat" and by["tool_use"]["alias"] == "dgxmm-tool-use"
    asked = [c["subject"] for c in fake.calls]
    assert sorted(asked) == ["routing:chat", "routing:code"]
    assert all("org/model-" not in c["question"] for c in fake.calls)
    # unchanged candidates: the model's earlier answer is reused, not asked again
    await svc.advise(fake)
    assert len(fake.calls) == 3  # only the unsure class is asked again


async def test_disabled_or_failing_decider_falls_back_to_rules(ctx):
    facts = [fact(1), fact(2, weights_bytes=30e9)]
    for d in (FakeDecider(available=False), FakeDecider(answers={"routing": RuntimeError("x")}), None):
        adv = await Svc(ctx, facts).advise(d, use_model=d is not None)
        chat = next(r for r in adv["classes"] if r["class"] == "chat")
        assert chat["deployment_id"] == 2 and chat["source"] == "rules"


async def test_job_publishes_one_insight_with_a_copy_action(ctx):
    ctx.insights = InsightService(ctx.db)

    class R:
        decider = FakeDecider()

        async def to_thread(self, fn, *a, **kw):
            import asyncio
            return await asyncio.to_thread(fn, *a, **kw)
    svc = Svc(ctx, [fact(1, tool_call_parser="hermes")])
    out = await svc.run(R())
    assert out == "3 of 6 classes have a suggestion"  # code, chat and tool_use; no long context, vision, batch
    [ins] = ctx.insights.list(status=["open"])
    assert ins["kind"] == "routing" and ins["subject"] == "litellm" and ins["severity"] == "info"
    assert ins["action"]["type"] == "copy" and "model_group_alias" in ins["action"]["text"]
    assert "DGXMM does not change LiteLLM" in ins["body"] and ins["expires_at"] is not None
    assert ctx.insights.list(admin=False) == []  # routing is admin-only
    await svc.run(R())
    assert len(ctx.insights.list()) == 1
    svc._facts = []
    await svc.run(R())
    assert ctx.insights.list(status=["open"]) == []


@pytest.fixture
def lite(api_stack):
    """A LiteLLM fake wired as the app's LiteLLM (never started): every request it gets is recorded."""
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    from tests.fakes import FakeLiteLLM
    fake = FakeLiteLLM()
    client = LiteLLMClient.fixed("http://litellm.test", "sk-test", fake.client())
    api_stack.ctx.litellm = LiteLLMSync(client, api_stack.stack.deployer, api_stack.ctx.db, api_stack.stack.events)
    return fake


def test_api_admin_only_rules_answer_and_never_touches_litellm(api_stack, lite):
    import time

    from tests.test_api_deployments import SMALL, admin
    h = admin(api_stack)
    api_stack.stack.http.all_healthy = True
    d = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "a", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "alias": "tiny-chat",
                             "config": {**SMALL, "tool_call_parser": "hermes"}}).json()
    for _ in range(200):
        if api_stack.get(f"/dgxmm/api/deployments/{d['id']}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    sync = api_stack.ctx.litellm
    sync.last = {**sync.last, "ok": True, "status": "synced"}
    sync.registrations = {d["id"]: "tiny-chat"}  # as after a successful sync (the sync itself is not under test)
    lite.calls.clear()  # the deploy checked the model name against LiteLLM's model list; advice must not call it
    r = api_stack.get("/dgxmm/api/decider/routing-advice")
    assert r.status_code == 200, r.text
    body = r.json()
    by = {c["class"]: c for c in body["classes"]}
    assert body["enabled"] is True and by["tool_use"]["model_name"] == "tiny-chat"
    assert by["tool_use"]["source"] == "rules" and by["vision"]["model_name"] is None
    assert 'dgxmm-tool-use: "tiny-chat"' in body["snippet"]
    assert lite.calls == []  # DGXMM never talks to LiteLLM for routing advice
    api_stack.post("/dgxmm/api/auth/logout", headers=h)
    api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert api_stack.get("/dgxmm/api/decider/routing-advice").status_code == 403


async def test_job_on_the_stack_never_touches_litellm_and_is_triggered_by_state_changes(stack):
    from dgxmm.context import AppContext
    from dgxmm.decider.features import wire_features
    from dgxmm.decider.insight_engine import InsightEngine
    from dgxmm.deployer.service import DeployRequest
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    from tests.fakes import FakeLiteLLM
    fake_lite = FakeLiteLLM()
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, inventory=stack.inv,
                     deployer=stack.deployer, engines=stack.deployer.engines)
    ctx.litellm = LiteLLMSync(LiteLLMClient.fixed("http://litellm.test", "sk-test", fake_lite.client()),
                              stack.deployer, stack.db, stack.events)
    now = [utcnow()]
    ctx.insight_engine = InsightEngine(ctx, tz=timezone.utc, clock=lambda: now[0])
    wire_features(ctx)
    job = next(j for j in ctx.insight_engine.status() if j["name"] == rt.JOB)
    assert job["schedule"] == "daily 05:15" and job["feature"] == "routing" and job["budget_calls"] == rt.BUDGET_CALLS
    d = await stack.deployer.deploy(DeployRequest(name="r", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 32}), 1)
    stack.http.healthy.add(d["port"])
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    fake_lite.calls.clear()  # the deploy checked the model name against LiteLLM's model list; the job must not
    import asyncio
    ctx.litellm.last = {**ctx.litellm.last, "ok": True, "status": "synced"}
    ctx.litellm.registrations = {d["id"]: stack.deployer.view(d["id"])["served_name"]}
    await asyncio.sleep(0.05)  # state hooks run as tasks
    assert await ctx.insight_engine.tick() == []  # a change notice settles first
    now[0] += timedelta(seconds=rt.SETTLE_S)
    assert rt.JOB in await ctx.insight_engine.tick()  # triggered by "-> ready"
    await ctx.insight_engine.wait_idle()
    by = {c["class"]: c for c in ctx.routing.latest["classes"]}
    assert by["batch"]["deployment_id"] == d["id"] and by["chat"]["deployment_id"] == d["id"]
    assert fake_lite.calls == []
    assert ctx.insights.list(kind="routing")[0]["action"]["type"] == "copy"


async def test_snippet_only_holds_ready_deployments_registered_in_litellm(ctx):
    """LiteLLM serves only ready, registered deployments: a sleeping (or unregistered) pick is advice text
    ("start it first"), never an alias in the snippet or the insight's copy action."""
    ctx.insights = InsightService(ctx.db)

    class R:
        decider = FakeDecider()

        async def to_thread(self, fn, *a, **kw):
            import asyncio
            return await asyncio.to_thread(fn, *a, **kw)
    facts = [fact(1, state="sleeping", registered=False, context=131072), fact(2, tool_call_parser="hermes",
                                                                              registered=False)]
    svc = Svc(ctx, facts)
    await svc.run(R())
    by = {r["class"]: r for r in svc.latest["classes"]}
    assert by["long_context"]["deployment_id"] == 1 and by["long_context"]["routable"] is False
    assert "start d1 first" in by["long_context"]["note"]
    assert by["tool_use"]["routable"] is False and "not registered" in by["tool_use"]["note"]
    assert svc.latest["snippet"] == ""
    [ins] = ctx.insights.list(status=["open"])
    assert ins["action"]["type"] == "open" and "start d1 first" in ins["body"]
    svc._facts = [fact(1, state="sleeping", registered=False, context=131072), fact(2, tool_call_parser="hermes")]
    await svc.run(R())
    assert 'dgxmm-tool-use: "m2"' in svc.latest["snippet"] and "m1" not in svc.latest["snippet"]
    assert ctx.insights.list(status=["open"])[0]["action"]["type"] == "copy"


async def test_advice_is_refreshed_after_litellm_registers_a_new_deployment(stack):
    """Live order: the routing job runs on "-> ready" before DGXMM's LiteLLM sync has registered the deployment, so
    nothing is routable yet; the sync's registration change must refresh the advice (else it stays empty until the
    next daily run)."""
    import asyncio

    from dgxmm.context import AppContext
    from dgxmm.decider.features import wire_features
    from dgxmm.decider.insight_engine import InsightEngine
    from dgxmm.deployer.service import DeployRequest
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    from tests.fakes import FakeLiteLLM
    fake_lite = FakeLiteLLM()
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, inventory=stack.inv,
                     deployer=stack.deployer, engines=stack.deployer.engines)
    ctx.litellm = LiteLLMSync(LiteLLMClient.fixed("http://litellm.test", "sk-test", fake_lite.client()),
                              stack.deployer, stack.db, stack.events)
    now = [utcnow()]
    ctx.insight_engine = InsightEngine(ctx, tz=timezone.utc, clock=lambda: now[0])
    wire_features(ctx)
    d = await stack.deployer.deploy(DeployRequest(name="r", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 32}), 1)
    stack.http.healthy.add(d["port"])
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    await asyncio.sleep(0.05)  # state hooks run as tasks
    now[0] += timedelta(seconds=rt.SETTLE_S)
    assert rt.JOB in await ctx.insight_engine.tick()  # "-> ready": before the sync has registered it
    await ctx.insight_engine.wait_idle()
    assert not any(c["routable"] for c in ctx.routing.latest["classes"]) and ctx.routing.latest["snippet"] == ""
    assert await ctx.litellm.sync()  # DGXMM registers the deployment in LiteLLM
    assert ctx.litellm.is_registered(d["id"], stack.deployer.view(d["id"])["served_name"]) is True
    now[0] += timedelta(seconds=rt.SETTLE_S)
    assert rt.JOB in await ctx.insight_engine.tick()  # the registration change refreshes the advice
    await ctx.insight_engine.wait_idle()
    by = {c["class"]: c for c in ctx.routing.latest["classes"]}
    assert by["chat"]["routable"] and stack.deployer.view(d["id"])["served_name"] in ctx.routing.latest["snippet"]
    assert await ctx.litellm.sync()  # nothing changed: no new refresh
    now[0] += timedelta(seconds=rt.SETTLE_S)
    assert await ctx.insight_engine.tick() == []


async def test_a_storm_of_deployment_and_registration_changes_runs_the_advice_once(stack):
    """Many deployments becoming ready, each followed by a LiteLLM sync that registers it, are one burst: the
    routing job (a model call budget per run) runs once after the changes settle, with every deployment routable,
    instead of once per state change and once per sync."""
    import asyncio
    from datetime import datetime

    from dgxmm.context import AppContext
    from dgxmm.decider.features import wire_features
    from dgxmm.decider.insight_engine import InsightEngine
    from dgxmm.deployer.service import DeployRequest
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    from tests.fakes import FakeLiteLLM
    now = [datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)]
    fake_lite = FakeLiteLLM()
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, inventory=stack.inv,
                     deployer=stack.deployer, engines=stack.deployer.engines)
    ctx.litellm = LiteLLMSync(LiteLLMClient.fixed("http://litellm.test", "sk-test", fake_lite.client()),
                              stack.deployer, stack.db, stack.events)
    ctx.insight_engine = InsightEngine(ctx, tz=timezone.utc, clock=lambda: now[0])
    wire_features(ctx)
    stack.http.all_healthy = True
    ids = []
    for i in range(6):
        d = await stack.deployer.deploy(DeployRequest(name=f"r{i}", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                      config={"max_model_len": 4096, "max_num_seqs": 32},
                                                      alias=f"m{i}"), 1)
        assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
        ids.append(d["id"])
        await asyncio.sleep(0.05)  # state hooks run as tasks
        assert await ctx.litellm.sync()
        now[0] += timedelta(seconds=2)
        assert await ctx.insight_engine.tick() == []  # still settling
    runs = 0
    for _ in range(int(rt.SETTLE_S) + 1):
        now[0] += timedelta(seconds=1)
        started = await ctx.insight_engine.tick()
        await ctx.insight_engine.wait_idle()
        runs += started.count(rt.JOB)
    assert runs == 1
    assert {f["id"] for f in ctx.routing.facts() if f["registered"]} == set(ids)
    assert ctx.routing.latest["snippet"] and all(c["routable"] for c in ctx.routing.latest["classes"]
                                                 if c["model_name"])
