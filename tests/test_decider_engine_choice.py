"""Feature 5 — engine choice, vLLM vs llama.cpp (dgxmm/decider/engine.py, POST /api/decider/engine-suggest)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgxmm.config import GIB
from dgxmm.decider import engine as en
from dgxmm.inventory.service import InventoryService
from tests.conftest import QWEN_TINY, write_gguf
from tests.fakes import FakeDecider


@pytest.fixture
def inv(settings, fake_hf_repo):
    root = Path(settings.gguf_roots[0].local)
    fake_hf_repo("Qwen/Qwen3-Both", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    write_gguf(root / "Qwen3-Both-Q4_K_M.gguf", layers=2, hidden=256, heads=4, kv=2, head_dim=64, ctx=4096,
               vocab=1000, pad_bytes=1024)
    write_gguf(root / "Lonely-7B-Q8_0.gguf", layers=2, hidden=256, heads=4, kv=2, head_dim=64, ctx=4096, vocab=1000,
               pad_bytes=1024)
    fake_hf_repo("org/Only-Safetensors", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    fake_hf_repo("org/Exotic-3B", config={**QWEN_TINY, "architectures": ["ExoticForCausalLM"]},
                 files={"model.safetensors": 2 * GIB})
    write_gguf(root / "Exotic-3B-Q4_K_M.gguf", layers=2, hidden=256, heads=4, kv=2, head_dim=64, ctx=4096,
               vocab=1000, pad_bytes=1024)
    i = InventoryService(settings)
    i.refresh()
    return i


def gguf_ref(inv, name):
    return next(a.ref for a in inv.list() if a.ref.endswith(name))


def ctx_of(inv, decider=None, archs=frozenset({"Qwen3ForCausalLM"})):
    return SimpleNamespace(inventory=inv, decider=decider, db=None, budget=None,
                           archs=SimpleNamespace(get=lambda: set(archs)))


def run(coro):
    return asyncio.run(coro)


def test_gguf_only_is_llamacpp(inv):
    ref = gguf_ref(inv, "Lonely-7B-Q8_0.gguf")
    fake = FakeDecider()
    out = run(en.suggest(ctx_of(inv, fake), ref, "chat"))
    assert out["engine"] == "llamacpp" and out["source"] == "rules" and out["reason"] == "gguf_only"
    assert out["model_ref"] == ref and fake.calls == [] and fake.rules[0]["choice"] == "llamacpp"


def test_safetensors_only_is_vllm(inv):
    out = run(en.suggest(ctx_of(inv, FakeDecider(available=False)), "org/Only-Safetensors", "chat"))
    assert out["engine"] == "vllm" and out["source"] == "rules" and out["reason"] == "safetensors_only"


def test_unsupported_architecture_with_a_gguf_goes_to_llamacpp(inv):
    out = run(en.suggest(ctx_of(inv, FakeDecider()), "org/Exotic-3B", "chat"))
    assert out["engine"] == "llamacpp" and out["reason"] == "vllm_unsupported"
    assert out["model_ref"] == gguf_ref(inv, "Exotic-3B-Q4_K_M.gguf")


def test_both_formats_ask_the_model(inv):
    fake = FakeDecider(answers={"engine": ("llamacpp", 0.9)})
    out = run(en.suggest(ctx_of(inv, fake), "Qwen/Qwen3-Both", "chat"))
    assert out["engine"] == "llamacpp" and out["source"] == "model"
    assert out["model_ref"] == gguf_ref(inv, "Qwen3-Both-Q4_K_M.gguf")
    assert out["alternatives"] == {"vllm": "Qwen/Qwen3-Both", "llamacpp": gguf_ref(inv, "Qwen3-Both-Q4_K_M.gguf")}
    q = fake.calls[0]
    assert [o.key for o in q["options"]] == ["vllm", "llamacpp"] and "Goal: chat" in q["state"]


def test_both_formats_below_threshold_or_disabled(inv):
    for fake in (FakeDecider(answers={"engine": ("vllm", 0.6)}), FakeDecider(available=False), None):
        out = run(en.suggest(ctx_of(inv, fake), "Qwen/Qwen3-Both", "batch_throughput"))
        assert out["engine"] is None and out["source"] == "none"
        assert set(out["alternatives"]) == {"vllm", "llamacpp"}


def _budget(free_gib):
    b = SimpleNamespace(free_for_new=int(free_gib * GIB))
    return SimpleNamespace(current=lambda: b)


def test_fit_decides_when_only_one_format_fits(inv):
    fake = FakeDecider(answers={"engine": ("vllm", 0.99)})
    c = ctx_of(inv, fake)
    c.budget = _budget(4.5)  # safetensors (2 GiB) needs 6.5 GiB, the tiny GGUF ~2 GiB
    out = run(en.suggest(c, "Qwen/Qwen3-Both", "batch_throughput"))
    assert out["engine"] == "llamacpp" and out["source"] == "rules" and out["reason"] == "only_llamacpp_fits"
    assert fake.calls == []
    c.budget = _budget(1)
    out = run(en.suggest(c, "Qwen/Qwen3-Both", "chat"))
    assert out["engine"] == "llamacpp" and out["reason"] == "neither_fits"
    c.budget = _budget(50)
    out = run(en.suggest(c, "Qwen/Qwen3-Both", "chat"))
    assert out["source"] == "model" and len(fake.calls) == 1  # both fit: the model decides


def test_eval_case_applies_the_fit_rule():
    fmts = {"safetensors": {"ref": "Q/8B", "weights_bytes": 16 * GIB, "quant": "bf16"},
            "gguf": {"ref": "/m/q.gguf", "weights_bytes": 5 * GIB, "quant": "Q4_K_M"}}
    case = {"name": "x", "label": "llamacpp", "goal": "chat", "formats": fmts, "vllm_supported": True}
    assert en.eval_case({**case, "free_bytes": 20 * GIB}).rules == "llamacpp"
    assert en.eval_case({**case, "free_bytes": 60 * GIB}).rules is None


def test_feature_off(inv):
    fake = FakeDecider()
    fake.rules_enabled = lambda f: f != "engine"
    out = run(en.suggest(ctx_of(inv, fake), "org/Only-Safetensors", "chat"))
    assert out["disabled"] is True and out["engine"] is None


def test_eval_case_adapter():
    both = {"name": "b", "label": "vllm", "goal": "batch_throughput", "ref": "Qwen/Qwen3-8B",
            "formats": {"safetensors": {"ref": "Qwen/Qwen3-8B", "weights_bytes": 16 * GIB, "quant": "bf16"},
                        "gguf": {"ref": "/models/Qwen3-8B-Q4_K_M.gguf", "weights_bytes": 5 * GIB, "quant": "Q4_K_M"}},
            "architecture": "Qwen3ForCausalLM", "vllm_supported": True, "free_bytes": 80 * GIB}
    ec = en.eval_case(both)
    assert ec.rules is None  # both fit in 80 GiB and [o.key for o in ec.options] == ["vllm", "llamacpp"]
    one = {**both, "formats": {"gguf": both["formats"]["gguf"]}}
    assert en.eval_case(one).rules == "llamacpp"


def test_engine_suggest_api(api_stack):
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    url = "/dgxmm/api/decider/engine-suggest"
    r = api_stack.post(url, headers=h, json={"model_ref": "Qwen/Qwen3-Tiny", "goal": "chat"})
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "vllm" and r.json()["source"] == "rules"
    assert api_stack.post(url, headers=h, json={"model_ref": "Qwen/Qwen3-Tiny", "goal": "x"}).status_code == 422
    assert api_stack.post(url, headers=h, json={"model_ref": "nope/nope", "goal": "chat"}).status_code == 404
