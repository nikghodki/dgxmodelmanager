"""Feature 2 — deploy-time config advice (dgxmm/decider/config.py, POST /api/decider/config-advice)."""
from __future__ import annotations

import asyncio

import pytest

from dgxmm.config import GIB
from dgxmm.decider import config as ca
from tests.conftest import QWEN_TINY
from tests.fakes import FakeDecider

LONG = {**QWEN_TINY, "max_position_embeddings": 131072, "num_hidden_layers": 24, "hidden_size": 2048,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936}


@pytest.fixture
def adv(stack, fake_hf_repo):
    fake_hf_repo("org/long-ctx", config=LONG, files={"model.safetensors": 8 * GIB})
    stack.inv.refresh()
    from types import SimpleNamespace
    return SimpleNamespace(deployer=stack.deployer, budget=stack.budget, inventory=stack.inv, decider=None,
                           engines=stack.deployer.engines, stack=stack)


def run(coro):
    return asyncio.run(coro)


def test_candidate_grid_respects_the_native_context():
    assert ca.context_choices(131072) == [131072, 65536, 32768, 16384, 8192]
    assert ca.context_choices(40960) == [40960, 32768, 16384, 8192]
    assert ca.context_choices(262144) == [262144, 131072, 65536, 32768, 16384, 8192]
    assert ca.context_choices(4096) == [4096]


def test_options_come_only_from_the_memory_calculator_and_fit(adv):
    out = run(ca.advise(adv, "org/long-ctx", "vllm", "chat"))
    assert 1 <= len(out["options"]) <= ca.MAX_OPTIONS
    free = adv.budget.current().free_for_new
    for o in out["options"]:
        prep = adv.deployer.prepare("vllm", "org/long-ctx", o["config"])
        assert prep.estimate.total == o["required_bytes"] <= free
        assert o["config"]["max_model_len"] <= 131072
    assert len({o["key"] for o in out["options"]}) == len(out["options"])


def test_nothing_offered_when_nothing_fits(adv):
    adv.stack.probe.available = 3 * GIB  # less than the weights
    run(adv.budget.refresh())
    out = run(ca.advise(adv, "org/long-ctx", "vllm", "chat"))
    assert out["options"] == [] and out["recommended"] is None and out["reason"] == "nothing_fits"


def test_pareto_cut_drops_dominated_options():
    a = ca.Candidate("a", {}, 32768, 32, "auto", True, 0, 10 * GIB, 0)
    b = ca.Candidate("b", {}, 16384, 8, "auto", True, 0, 12 * GIB, 0)   # worse in every way than a
    c = ca.Candidate("c", {}, 65536, 8, "fp8", True, 0, 11 * GIB, 0)    # a trade-off
    d = ca.Candidate("d", {}, 32768, 32, "auto", False, 0, 10 * GIB, 0)  # a without prefix caching
    assert {x.key for x in ca.pareto([a, b, c, d])} == {"a", "c"}


def test_model_recommendation_at_threshold(adv):
    adv.decider = FakeDecider(answers={"config": lambda opts, st: (opts[1].key, 0.92)})
    out = run(ca.advise(adv, "org/long-ctx", "vllm", "long_context"))
    assert out["source"] == "model" and out["recommended"] == adv.decider.calls[0]["options"][1].key
    assert out["confidence"] == pytest.approx(0.92)
    q = adv.decider.calls[0]
    assert [o.key for o in q["options"]] == [o["key"] for o in out["options"]]
    assert "long_context" in q["state"] and all(o.description for o in q["options"])


def test_below_threshold_shows_the_top_three(adv):
    def probs(opts, st):
        return {o.key: (0.5 if i == 2 else 0.3 if i == 0 else 0.2 / max(len(opts) - 2, 1)) for i, o in enumerate(opts)}
    adv.decider = FakeDecider(answers={"config": probs})
    out = run(ca.advise(adv, "org/long-ctx", "vllm", "chat"))
    keys = [o.key for o in adv.decider.calls[0]["options"]]
    assert out["recommended"] is None and out["source"] == "none"
    assert out["top"][:2] == [keys[2], keys[0]] and len(out["top"]) == min(3, len(keys))


def test_decider_disabled_still_offers_options_in_rules_order(adv):
    adv.decider = FakeDecider(available=False)
    out = run(ca.advise(adv, "org/long-ctx", "vllm", "long_context"))
    assert out["options"] and out["recommended"] is None and out["source"] == "none"
    assert out["top"][0] == max(out["options"], key=lambda o: (o["context"], o["kv_cache_dtype"] == "auto",
                                                           o["seqs"]))["key"]


def test_feature_off_returns_nothing(adv):
    adv.decider = FakeDecider(answers={"config": ("x", 0.99)})
    adv.decider.rules_enabled = lambda f: f != "config"
    out = run(ca.advise(adv, "org/long-ctx", "vllm", "chat"))
    assert out["disabled"] is True and out["options"] == [] and adv.decider.calls == []


def test_llamacpp_candidates(stack, fake_hf_repo, settings):
    from types import SimpleNamespace
    from tests.conftest import write_gguf
    from pathlib import Path
    write_gguf(Path(settings.gguf_roots[0].local) / "tiny-Q4_K_M.gguf", layers=4, hidden=256, heads=4, kv=2,
               head_dim=64, ctx=32768, vocab=1000, pad_bytes=1024)
    stack.inv.refresh()
    ref = next(a.ref for a in stack.inv.list() if a.format == "gguf")
    c = SimpleNamespace(deployer=stack.deployer, budget=stack.budget, inventory=stack.inv, decider=None,
                        engines=stack.deployer.engines)
    out = run(ca.advise(c, ref, "llamacpp", "batch_throughput"))
    assert out["options"]
    for o in out["options"]:
        assert set(o["config"]) == {"ctx_size", "parallel", "cache_type_k", "cache_type_v"}
        assert o["config"]["ctx_size"] == o["context"] * o["seqs"]


def test_model_name_is_quoted_untrusted(adv, fake_hf_repo):
    adv.decider = FakeDecider(answers={"config": lambda opts, st: (opts[0].key, 0.9)})
    run(ca.advise(adv, "org/long-ctx", "vllm", "chat"))
    st = adv.decider.calls[0]["state"]
    assert "MODEL NAME (untrusted" in st and "<<<\norg/long-ctx\n>>>" in st


def test_eval_case_adapter():
    case = {"name": "x", "label": ["k"], "engine": "vllm", "goal": "chat", "ref": "org/m", "config": LONG,
            "weights_bytes": 8 * GIB, "free_bytes": 60 * GIB}
    ec = ca.eval_case(case)
    assert ec.rules is None and ec.question == ca.QUESTION and 1 <= len(ec.options) <= ca.MAX_OPTIONS
    assert "chat" in ec.state


# ---------------- API ----------------
def test_config_advice_api(api_stack, fake_hf_repo):
    from tests.test_api_deployments import admin
    fake_hf_repo("org/long-ctx", config=LONG, files={"model.safetensors": 8 * GIB})
    api_stack.stack.inv.refresh()
    h = admin(api_stack)
    url = "/dgxmm/api/decider/config-advice"
    r = api_stack.post(url, headers=h, json={"model_ref": "org/long-ctx", "engine": "vllm", "goal": "chat"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"options", "recommended", "source", "confidence", "top", "goal"} <= set(body)
    assert {"key", "config", "summary", "required_bytes"} <= set(body["options"][0])
    assert api_stack.post(url, headers=h, json={"model_ref": "org/long-ctx", "engine": "vllm",
                                                "goal": "world_domination"}).status_code == 422
    assert api_stack.post(url, headers=h, json={"model_ref": "nope/nope", "engine": "vllm",
                                                "goal": "chat"}).status_code == 404
    assert api_stack.post(url, headers=h, json={"model_ref": "org/long-ctx", "engine": "zzz",
                                                "goal": "chat"}).status_code == 422
    assert api_stack.post(url, json={"model_ref": "org/long-ctx", "engine": "vllm",
                                     "goal": "chat"}).status_code == 403  # no CSRF token
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    vh = {"X-CSRF-Token": r.json()["csrf_token"]}
    assert api_stack.post(url, headers=vh, json={"model_ref": "org/long-ctx", "engine": "vllm",
                                                 "goal": "chat"}).status_code == 403
