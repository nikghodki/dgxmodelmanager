"""Review round 1 fixes (written before the fixes)."""
import asyncio
import random
import time
from datetime import datetime, timezone

import pytest

from dgxmm.config import GIB
from dgxmm.context import AppContext
from dgxmm.db import Deployment, DeploymentEvent, MetricRollup
from dgxmm.decider import common, config_suggest as cs, diagnosis as dg, kind as kd, plans as dp, workload as wl
from dgxmm.decider.features import wire_features
from dgxmm.decider.types import Option
from dgxmm.engines.base import EngineMetrics
from dgxmm.metrics.store import MetricsStore
from dgxmm.planner.plans import ActiveDep, build_plans
from tests.fakes import FakeDecider
from tests.test_plans import fit_needing

W = dp.ACTIVE_WINDOW_MIN


# ---------- 1. unknown traffic is not idle ----------
def _dep(db, name, state="ready", priority=0, engine="vllm", model_ref=None, config=None, ready_ago_s=7200, now=None):
    now = time.time() if now is None else now
    with db.session() as s:
        d = Deployment(name=name, engine=engine, model_ref=model_ref or f"org/{name}", config=config or {}, state=state,
                       priority=priority)
        s.add(d)
        s.flush()
        if ready_ago_s is not None:
            s.add(DeploymentEvent(deployment_id=d.id, from_state="loading", to_state="ready",
                                  ts=datetime.fromtimestamp(now - ready_ago_s, timezone.utc)))
        return d.id


def _rollups(db, dep_id, minutes_ago, now, running=0.0, samples=12):
    with db.session() as s:
        for m in minutes_ago:
            minute = datetime.fromtimestamp((int(now // 60) - m) * 60, timezone.utc)
            s.add(MetricRollup(deployment_id=dep_id, minute=minute,
                               data={"running": running, "waiting": 0, "gen_tps": 0, "prompt_tps": 0,
                                     "samples": samples}))


def _view(db, dep_id):
    with db.session() as s:
        d = s.get(Deployment, dep_id)
        return {"id": d.id, "name": d.name, "state": d.state, "priority": d.priority, "engine": d.engine,
                "model_ref": d.model_ref, "config": d.config}


def test_empty_store_is_unknown_traffic_and_unsafe(db):
    now = time.time()
    dep = _dep(db, "lc", engine="llamacpp", now=now)
    facts = dp.facts_for([_view(db, dep)], MetricsStore(db), now=now)
    r = dp.unsafe_reason({"id": "stop", "fits": True, "actions": [{"type": "stop", "deployment_id": dep}]}, facts,
                         raw_cfg={}, defaults={}, context_key="", seqs_key="", request_priority=0)
    assert r is not None and "traffic unknown" in r


def test_coverage_rules(db):
    now = time.time()
    store = MetricsStore(db)
    full = _dep(db, "full", now=now)
    _rollups(db, full, range(1, 31), now)
    gap = _dep(db, "gap", now=now)
    _rollups(db, gap, range(1, 9), now)  # 8 of 15 minutes
    nosamples = _dep(db, "nos", now=now)
    _rollups(db, nosamples, range(1, 31), now, samples=0)
    fresh = _dep(db, "fresh", now=now, ready_ago_s=300)
    _rollups(db, fresh, range(1, 31), now)
    f = dp.facts_for([_view(db, i) for i in (full, gap, nosamples, fresh)], store, now=now)
    assert f[full].traffic_known and f[full].active_reason() is None
    assert not f[gap].traffic_known and not f[nosamples].traffic_known
    assert f[fresh].active_reason() and "ready" in f[fresh].active_reason()


async def test_sleep_action_on_unknown_traffic_is_unsafe(db):
    now = time.time()
    dep = _dep(db, "s", now=now)
    facts = dp.facts_for([_view(db, dep)], MetricsStore(db), now=now)
    plans = [{"id": "sleep", "title": "Sleep", "detail": "", "new_required": 1, "frees": 1, "fits": True,
              "actions": [{"type": "sleep", "deployment_id": dep}], "patch": None, "model_ref": None}]
    out = await dp.recommend(plans, deps=facts, decider=FakeDecider(), shortfall=1, model_ref="a/b", engine="vllm",
                             cfg={}, raw_cfg={}, defaults={}, context_key="", seqs_key="", request_priority=0,
                             cache=dp._Cache())
    assert not out[0]["recommended"] and "traffic unknown" in out[0]["unsafe_reason"]


async def test_preview_with_empty_metrics_recommends_no_stop(stack):
    from dgxmm.deployer.service import DeployRequest
    small = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
    a = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="org/big-model", config=small), 1)
    stack.http.healthy.add(a["port"])
    await stack.deployer.wait_ready(a["id"], 5)
    stack.probe.available -= a["reserved_bytes"]
    stack.deployer.metrics = MetricsStore(stack.db)
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    stop = next(x for x in p["plans"] if x["id"] == f"stop:{a['id']}")
    assert not stop["recommended"] and "became ready" in stop["unsafe_reason"]
    with stack.db.session() as s:  # ready long ago, but still no metrics at all
        for e in s.query(DeploymentEvent).filter(DeploymentEvent.deployment_id == a["id"]):
            e.ts = datetime.fromtimestamp(time.time() - 7200, timezone.utc)
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    stop = next(x for x in p["plans"] if x["id"] == f"stop:{a['id']}")
    assert not stop["recommended"] and "traffic unknown" in stop["unsafe_reason"]


# ---------- 5. priority relative to the request ----------
def test_higher_priority_than_request_is_unsafe():
    d = dp.DepFacts(id=1, name="imp", state="ready", priority=50, traffic_known=True, ready_for_min=999)
    plan = {"id": "stop", "fits": True, "actions": [{"type": "stop", "deployment_id": 1}]}
    kw = dict(raw_cfg={}, defaults={}, context_key="", seqs_key="")
    assert "higher priority" in dp.unsafe_reason(plan, {1: d}, request_priority=10, **kw)
    assert dp.unsafe_reason(plan, {1: d}, request_priority=50, **kw) is None
    assert not hasattr(dp, "PINNED_PRIORITY")


async def test_preview_uses_request_or_replaced_priority(stack):
    from dgxmm.deployer.service import DeployRequest
    small = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
    a = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="org/big-model", config=small,
                                                  priority=40), 1)
    stack.http.healthy.add(a["port"])
    await stack.deployer.wait_ready(a["id"], 5)
    stack.probe.available -= a["reserved_bytes"]
    p = await stack.deployer.preview("vllm", "org/big-model", small, priority=10)
    stop = next(x for x in p["plans"] if x["id"] == f"stop:{a['id']}")
    assert "higher priority" in stop["unsafe_reason"]


# ---------- 1. property test over real stores ----------
SCENARIOS = ["empty", "stale", "gap", "fresh_start", "busy_rollup", "busy_ring", "idle_full", "sleeping", "starting",
             "unknown", "no_samples"]
SAFE_SCENARIOS = {"idle_full", "sleeping"}


def _build(db, store, rng, now, trial, bias=False):
    views, truth = [], {}
    # biased trials draw idle/sleeping deployments more often so stop/sleep plans are actually recommendable
    pool = SCENARIOS + sorted(SAFE_SCENARIOS) * 4 if bias else SCENARIOS
    for k in range(rng.randint(1, 5)):
        scen = rng.choice(pool)
        state = {"sleeping": "sleeping", "starting": "starting", "unknown": "unknown"}.get(scen, "ready")
        prio = rng.choice([0, 10, 50, 90])
        engine = "decision" if rng.random() < 0.1 else rng.choice(["vllm", "llamacpp"])
        name = f"t{trial}-{k}"
        ref = f"org/{name}"
        cfg = {}
        if views and rng.random() < 0.25:  # this one uses an earlier one as its draft
            cfg = {"speculative": {"method": "eagle", "model": rng.choice(views)["model_ref"]}}
        dep = _dep(db, name, state=state, priority=prio, engine=engine, model_ref=ref, config=cfg, now=now,
                   ready_ago_s=300 if scen == "fresh_start" else 7200)
        if scen == "stale":
            _rollups(db, dep, range(60, 120), now)
        elif scen == "gap":
            _rollups(db, dep, rng.sample(range(1, W + 1), rng.randint(1, int(W * 0.8) - 1)), now)
        elif scen in ("fresh_start", "idle_full", "sleeping"):
            _rollups(db, dep, range(1, 40), now)
        elif scen == "busy_rollup":
            _rollups(db, dep, range(1, 40), now)
            _rollups(db, dep, [rng.randint(1, W - 1)], now, running=1.5)
        elif scen == "busy_ring":
            _rollups(db, dep, range(1, 40), now)
            store.add(dep, now - 10, EngineMetrics(gauges={"running": 1.0}))
        elif scen == "no_samples":
            _rollups(db, dep, range(1, 40), now, samples=0)
        v = _view(db, dep)
        views.append(v)
        truth[dep] = (scen, prio, engine)
    return views, truth


async def test_property_no_recommended_plan_touches_protected_deployments(db, settings):
    rng = random.Random(7)
    store = MetricsStore(db)
    now = time.time()
    action_recs = 0
    for trial in range(150):
        views, truth = _build(db, store, rng, now, trial, bias=trial % 3 == 0)
        req_prio = 90 if trial % 3 == 0 else rng.choice([0, 10, 50, 90])  # biased trials: priority never blocks
        used = {v["config"]["speculative"]["model"] for v in views
                if v["state"] in dp.ACTIVE_VIEW_STATES and v["config"].get("speculative")}
        safe = {v["id"] for v in views
                if truth[v["id"]][0] in SAFE_SCENARIOS and truth[v["id"]][1] <= req_prio
                and truth[v["id"]][2] != "decision" and v["model_ref"] not in used}
        active = [ActiveDep(v["id"], v["name"], v["state"], v["priority"], 10 * GIB, 10 * GIB,
                            v["state"] == "ready", 5 * GIB) for v in views]
        fit = fit_needing(rng.choice([30, 50, 80]), 25, settings)
        cfg = {"max_model_len": 65536, "max_num_seqs": 8, "kv_cache_dtype": "auto"}
        plans = [p.as_dict() for p in build_plans(
            fit=fit, cfg=cfg, context_key="max_model_len", seqs_key="max_num_seqs", kv_patches=[{"kv_cache_dtype": "fp8"}],
            estimate_with=lambda patch: 20 * GIB, active=active, variants=[], estimate_ref=lambda r: None)]
        ids = [v["id"] for v in views] + [999_000 + trial]  # includes an id that is not a known deployment
        for j in range(rng.randint(0, 4)):
            acts = [{"type": rng.choice(["stop", "sleep"]), "deployment_id": rng.choice(ids)}
                    for _ in range(rng.randint(1, 3))]
            plans.append({"id": f"x{j}", "title": "x", "detail": "", "new_required": 1, "frees": 1, "fits": True,
                          "actions": acts, "patch": rng.choice([None, {"max_num_seqs": 4}, {"kv_cache_dtype": "fp8"}]),
                          "model_ref": None})
        deps = dp.facts_for(views, store, now=now)
        pick = rng.random()
        with_actions = {pl["id"] for pl in plans if pl.get("actions")}

        def answer(options, state, pick=pick, force=trial % 3 == 0):
            # every third trial the adversarial model insists on a stop/sleep plan when one is offered
            pool = [o for o in options if o.key in with_actions] if force else []
            pool = pool or options
            return pool[int(pick * len(pool))].key, 0.99
        fake = FakeDecider(answers={"plan": answer})
        out = await dp.recommend(plans, deps=deps, decider=fake, shortfall=fit.required - fit.free_for_new,
                                 model_ref="org/new", engine="vllm", cfg=cfg, raw_cfg={"max_num_seqs": 16},
                                 defaults={"max_model_len": 32768, "max_num_seqs": 8}, context_key="max_model_len",
                                 seqs_key="max_num_seqs", request_priority=req_prio, cache=dp._Cache())
        rec = [p for p in out if p["recommended"]]
        assert len(rec) <= 1
        action_recs += sum(1 for p in rec if p["actions"])
        for p in rec:
            assert p["fits"]
            for a in p["actions"]:
                assert a["deployment_id"] in safe, (trial, truth.get(a["deployment_id"]), p)
            if (p.get("patch") or {}).get("max_num_seqs") is not None:
                assert False, "lowered user-stated max_num_seqs"
    assert action_recs >= 15, f"only {action_recs} recommended stop/sleep plans: the property is barely exercised"


# ---------- 2/7/8/14/15 diagnosis ----------
ENGINE_CFG_LINE = ("(EngineCore_DP0 pid=212) INFO 10-14 09:11:00 [core.py:93] Initializing a V1 LLM engine (v0.26.1) "
                   "with config: model='org/x', speculative_config=None, tokenizer='org/x', skip_tokenizer_init=False, "
                   "tokenizer_mode=auto, revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096")


def test_engine_config_line_does_not_trigger_trust_remote_code():
    oom = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.64 GiB."
    assert [c for c, _ in dg.rule_hits(ENGINE_CFG_LINE + "\n" + oom)] == ["gpu_oom"]
    assert dg.rule_hits("WARNING the model class X requires remote code") == []  # not HF's actionable message
    assert [c for c, _ in dg.rule_hits("Please pass the argument `trust_remote_code=True` to allow")] == \
        ["trust_remote_code_required"]


def test_patch_that_changes_nothing_is_dropped(settings):
    from dgxmm.engines.vllm import VllmEngine
    v = VllmEngine(settings)
    cfg = v.validate_config({"trust_remote_code": True}).model_dump(mode="json")
    assert dg.validated_patch(v, cfg, {"trust_remote_code": True}) is None
    cfg2 = v.validate_config({"kv_cache_dtype": "fp8", "max_model_len": 8192}).model_dump(mode="json")
    assert dg.validated_patch(v, cfg2, {"kv_cache_dtype": "fp8", "max_model_len": 4096}) == {"max_model_len": 4096}


def test_exit_137_alone_is_not_a_rule():
    assert dg.rule_hits("Loading weights...", 137) == []


def test_parser_patch_uses_the_matched_line():
    cfg = {"tool_call_parser": "qwen3", "enable_auto_tool_choice": True, "reasoning_parser": "qwen3"}
    log = ("(APIServer) INFO reasoning is enabled for this model\n"
           "vllm serve: error: argument --tool-call-parser: invalid choice: 'qwen3' (choose from 'hermes')")
    assert dg.config_patch("bad_parser_flag", "vllm", cfg, log, ("hermes", "qwen3")) == {"tool_call_parser": "hermes"}
    log2 = ("(APIServer) INFO tool calling: --tool-call-parser hermes enabled\n"
            "KeyError: 'reasoning helper: qwen-3 not found in registered reasoning parsers: qwen3'")
    assert dg.config_patch("bad_parser_flag", "vllm", cfg, log2, ("hermes", "qwen3")) == {"reasoning_parser": "qwen3"}


def test_old_kv_message_is_kv_cache_too_small():
    t = ("ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored "
         "in KV cache (40960). Try increasing `gpu_memory_utilization` or decreasing `max_model_len`")
    assert [c for c, _ in dg.rule_hits(t)] == ["kv_cache_too_small"]


def test_only_launch_command_shell_traces_are_echo_lines():
    assert dg.rule_hits("+ vllm serve org/x --tool-call-parser hermes --trust-remote-code") == []
    assert [c for c, _ in dg.rule_hits("+ OSError: [Errno 28] No space left on device")] == ["disk_full"]


async def test_concurrent_diagnose_stores_one_row(stack):
    from sqlalchemy import func, select

    from dgxmm.db import DiagnosisRow
    from tests.test_decider_diagnosis import fail_with, make_ctx
    ctx = make_ctx(stack, FakeDecider(answers={"diagnosis": ("other", 0.95)}, delay_s=0.05))
    dep_id = await fail_with(stack, ctx, "Something odd happened")
    with stack.db.session() as s:
        s.query(DiagnosisRow).delete()
    await asyncio.gather(*(ctx.diagnosis.diagnose(dep_id) for _ in range(5)))
    with stack.db.session() as s:
        assert s.scalar(select(func.count()).select_from(DiagnosisRow)) == 1


# ---------- 3. one deadline ----------
class SlowDecider(FakeDecider):
    async def available(self):
        await asyncio.sleep(0.6)
        return True


async def test_single_deadline_covers_availability_and_decide():
    d = SlowDecider(answers={"plan": ("a", 0.99)}, delay_s=0.6)
    t = time.monotonic()
    out = await common.safe_decide(d, feature="plan", subject="s", state="", question="q",
                                   options=[Option("a", "A"), Option("b", "B")], timeout_s=1.0)
    assert time.monotonic() - t <= 1.1 and out.source == "none"
    assert not hasattr(common, "model_ready")


async def test_suggest_config_asks_both_questions_concurrently():
    d = FakeDecider(answers={"parser_tool": ("hermes", 0.95), "parser_reasoning": ("none", 0.95)}, delay_s=0.4)
    info = cs.ModelInfo("acme/Mystery-7B", "MysteryForCausalLM", template="{{ tools }} <think>")
    t = time.monotonic()
    out = await cs.suggest_config(d, info, "vllm")
    assert time.monotonic() - t < 0.75 and out["tool_call_parser"]["value"] == "hermes"


# ---------- 4. /advice never waits on the model ----------
async def test_workload_peek_schedules_single_background_compute(stack):
    from tests.test_decider_kind_workload import ready_dep, rollups
    dep = await ready_dep(stack)
    rollups(stack.db, dep, 40, running=1, gen_tps=30)
    fake = FakeDecider(answers={"workload": ("single_user_chat", 0.9)}, delay_s=0.3)
    ctx = AppContext(settings=stack.settings, db=stack.db, events=stack.events, inventory=stack.inv,
                     deployer=stack.deployer, engines=stack.deployer.engines, decider=fake)
    ctx.metrics = MetricsStore(stack.db)
    wire_features(ctx)
    t = time.monotonic()
    assert ctx.workload.peek(dep) is None and ctx.workload.peek(dep) is None
    assert time.monotonic() - t < 0.1
    await ctx.workload.drain()
    assert len(fake.calls) == 1  # single flight
    assert ctx.workload.peek(dep)["label"] == "single_user_chat"


def test_advice_endpoint_returns_immediately_on_cold_cache(api_stack):
    from dgxmm.advisor.rules import AdvisorService
    from tests.test_api_deployments import SMALL, admin
    from tests.test_decider_kind_workload import rollups
    h = admin(api_stack)
    api_stack.ctx.metrics = MetricsStore(api_stack.ctx.db)
    api_stack.ctx.advisor = AdvisorService(api_stack.stack.deployer, api_stack.stack.inv, api_stack.ctx.metrics,
                                           api_stack.ctx.db)
    # the decider takes far longer than any plausible request (a busy box): a response that waited for it could
    # not come back within the bound below, while a response that did not wait comes back in milliseconds
    api_stack.ctx.decider = FakeDecider(answers={"workload": ("mixed", 0.99)}, delay_s=30)
    api_stack.stack.http.all_healthy = True
    d = api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "a", "engine": "vllm",
                                                                    "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    for _ in range(200):
        if api_stack.get(f"/dgxmm/api/deployments/{d['id']}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    rollups(api_stack.ctx.db, d["id"], 40, running=1, gen_tps=30)
    t = time.monotonic()
    r = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/advice")
    assert r.status_code == 200 and time.monotonic() - t < 10


# ---------- 9/10/11 parser suggestions ----------
def test_large_tokenizer_config_template_is_read(fake_hf_repo, settings):
    from dgxmm.inventory.service import InventoryService
    from tests.conftest import QWEN_TINY
    snap = fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    import json
    big = {"added_tokens_decoder": {str(i): {"content": f"<t{i}>" * 20} for i in range(4000)},
           "chat_template": "<tool_call>{\"name\": x}</tool_call>"}
    (snap / "tokenizer_config.json").write_text(json.dumps(big))
    assert (snap / "tokenizer_config.json").stat().st_size > 256 * 1024
    inv = InventoryService(settings)
    inv.refresh()
    assert "<tool_call>" in (cs.model_info(inv.get("Qwen/Qwen3-Tiny")).template or "")


async def test_name_only_rule_does_not_auto_apply():
    out = await cs.suggest_config(FakeDecider(), cs.ModelInfo("meta-llama/Llama-3.1-8B-Instruct"), "vllm")
    assert out["tool_call_parser"]["value"] == "llama3_json" and out["tool_call_parser"]["auto_apply"] is False
    tmpl = cs.ModelInfo("Qwen/Qwen3-8B", template='<tool_call>{"name": x}</tool_call> <think>')
    out = await cs.suggest_config(FakeDecider(), tmpl, "vllm")
    assert out["tool_call_parser"]["auto_apply"] is True


async def test_parser_decide_uses_real_threshold_and_survives_threshold_errors():
    info = cs.ModelInfo("acme/Mystery-7B", "MysteryForCausalLM", template="{{ tools }} <think>")
    fake = FakeDecider(answers={"parser_tool": {"hermes": 0.7, "none": 0.3}, "parser_reasoning": ("none", 0.95)})
    out = await cs.suggest_config(fake, info, "vllm")
    assert fake.calls[0]["threshold"] in (None, fake.threshold("parser"))
    assert out["tool_call_parser"]["value"] == "hermes" and out["tool_call_parser"]["auto_apply"] is False

    class Broken(FakeDecider):
        def threshold(self, feature):
            raise RuntimeError("settings unavailable")
    out = await cs.suggest_config(Broken(answers={"parser_tool": ("hermes", 0.95)}), info, "vllm")
    assert out["tool_call_parser"] is None or out["tool_call_parser"]["auto_apply"] is False


# ---------- 12/13 kinds ----------
async def test_model_llm_kind_requires_supported_architecture(settings, db, fake_hf_repo):
    from dgxmm.engines.registry import build_engines
    from dgxmm.inventory.service import InventoryService
    from tests.test_decider_kind_workload import ODD
    fake_hf_repo("acme/odd", config=ODD)
    inv = InventoryService(settings)
    engines = build_engines(settings)
    engines["vllm"].arch_source = lambda: {"Qwen3ForCausalLM"}
    ctx = AppContext(settings=settings, db=db, events=__import__("dgxmm.events", fromlist=["EventBroker"]).EventBroker(),
                     inventory=inv, engines=engines, decider=FakeDecider(answers={"kind": ("llm", 0.99)}))
    wire_features(ctx)
    inv.refresh()
    await ctx.kinds.classify_pending()
    assert inv.get("acme/odd").kind == "other"
    engines["vllm"].arch_source = lambda: {"MysteryModel"}
    inv.refresh()
    assert inv.get("acme/odd").kind == "llm"


async def test_cached_below_threshold_kind_applies_when_threshold_lowered(settings, db, fake_hf_repo):
    from dgxmm.inventory.service import InventoryService
    from tests.test_decider_kind_workload import ODD
    fake_hf_repo("acme/odd", config=ODD)
    inv = InventoryService(settings)
    fake = FakeDecider(answers={"kind": ("embedding", 0.7)})
    ctx = AppContext(settings=settings, db=db, events=__import__("dgxmm.events", fromlist=["EventBroker"]).EventBroker(),
                     inventory=inv, decider=fake)
    wire_features(ctx)
    inv.refresh()
    await ctx.kinds.classify_pending()
    assert inv.get("acme/odd").kind == "other"
    fake.thresholds["kind"] = 0.6
    inv.refresh()
    assert inv.get("acme/odd").kind == "embedding" and len(fake.calls) == 1
