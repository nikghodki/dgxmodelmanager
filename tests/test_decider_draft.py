"""Feature 4 — draft-model pairing for speculative decoding (dgxmm/decider/draft.py, POST /api/decider/draft-suggest)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dgxmm.config import GIB
from dgxmm.decider import draft as dr
from dgxmm.inventory.service import InventoryService
from tests.conftest import QWEN_TINY
from tests.fakes import FakeDecider

TOK = '{"model": {"vocab": {"a": 0, "b": 1}}}'
LLAMA = {**QWEN_TINY, "architectures": ["LlamaForCausalLM"], "model_type": "llama"}
QWEN_BIG = {**QWEN_TINY, "hidden_size": 1024}
DFLASH = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": 1024, "vocab_size": 1000,
          "num_hidden_layers": 2, "num_target_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 64, "intermediate_size": 512, "max_position_embeddings": 4096}
EAGLE3 = {"architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama", "hidden_size": 256,
          "vocab_size": 1000, "draft_vocab_size": 500, "num_hidden_layers": 1, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 64, "intermediate_size": 512, "max_position_embeddings": 4096,
          "speculators_config": {"verifier": {"name_or_path": "meta/Llama-Big"}}}


@pytest.fixture
def inv(settings, fake_hf_repo):
    def make(ref, cfg, size, tok=TOK):
        snap = fake_hf_repo(ref, config=cfg, files={"model.safetensors": size})
        if tok is not None:
            (snap / "tokenizer.json").write_text(tok)
    make("Qwen/Qwen3-Big", QWEN_BIG, 16 * GIB)
    make("Qwen/Qwen3-Small", QWEN_TINY, 1 * GIB)
    make("Qwen/Qwen3-Mid", QWEN_TINY, 4 * GIB, tok=None)                   # no tokenizer file, same family+vocab
    make("Qwen/Qwen3-Otherhash", QWEN_TINY, 2 * GIB, tok=TOK.replace("b", "c"))  # known, different tokenizer
    make("Qwen/Qwen3-Bigvocab", {**QWEN_TINY, "vocab_size": 1024}, 1 * GIB)     # identical tokenizer, other vocab
    make("Qwen/Qwen3-Huge", QWEN_TINY, 40 * GIB)                           # bigger than the target
    make("meta/Llama-Tiny", LLAMA, 1 * GIB, tok='{"other": 1}')            # same vocab size, other family
    make("Qwen/Qwen3-Othervocab", {**QWEN_TINY, "vocab_size": 2000}, 1 * GIB, tok=None)
    make("org/Qwen3-Big-DFlash", DFLASH, GIB // 2, tok=None)
    make("org/unrelated-DFlash", {**DFLASH, "hidden_size": 4096}, GIB // 2, tok=None)
    make("org/llama-eagle3", EAGLE3, GIB // 4, tok=None)
    i = InventoryService(settings)
    i.refresh()
    return i


def ctx_of(inv, decider=None):
    return SimpleNamespace(inventory=inv, decider=decider)


def run(coro):
    return asyncio.run(coro)


def refs(cands):
    return [c["ref"] for c in cands]


def test_tokenizer_identity_or_family_and_vocab(inv):
    target = inv.get("Qwen/Qwen3-Big")
    got = {a.ref: why for a, why in dr.candidates(target, "draft_model", inv.list())}
    assert got == {"Qwen/Qwen3-Small": "same_tokenizer", "Qwen/Qwen3-Mid": "same_family_vocab"}
    # never: bigger than the target, another family with the same vocab size, another vocab (even with an identical
    # tokenizer file), a known different tokenizer (even with the same family and vocab), draft heads
    assert dr.tokenizer_fingerprint(inv.get("Qwen/Qwen3-Small")) == dr.tokenizer_fingerprint(target)
    assert dr.tokenizer_fingerprint(inv.get("Qwen/Qwen3-Otherhash")) not in (None, dr.tokenizer_fingerprint(target))


def test_identical_tokenizer_is_enough_across_families(inv, settings, fake_hf_repo):
    snap = fake_hf_repo("org/renamed-small", config={**LLAMA, "vocab_size": 1000}, files={"model.safetensors": GIB})
    (snap / "tokenizer.json").write_text(TOK)
    inv.refresh()
    got = dict((a.ref, why) for a, why in dr.candidates(inv.get("Qwen/Qwen3-Big"), "draft_model", inv.list()))
    assert got["org/renamed-small"] == "same_tokenizer"


def test_dflash_needs_a_matching_head(inv):
    got = [a.ref for a, _ in dr.candidates(inv.get("Qwen/Qwen3-Big"), "dflash", inv.list())]
    assert got == ["org/Qwen3-Big-DFlash"]  # hidden size, vocab and target layers match; the other head does not


def test_eagle3_head_for_its_family(inv, fake_hf_repo):
    fake_hf_repo("meta/Llama-Big", config={**LLAMA, "vocab_size": 1000}, files={"model.safetensors": 8 * GIB})
    inv.refresh()
    assert [a.ref for a, _ in dr.candidates(inv.get("meta/Llama-Big"), "eagle3", inv.list())] == ["org/llama-eagle3"]
    assert dr.candidates(inv.get("meta/Llama-Big"), "eagle", inv.list()) == []
    assert dr.candidates(inv.get("Qwen/Qwen3-Big"), "eagle3", inv.list()) == []  # hidden size differs


def test_methods_without_a_draft_model(inv):
    out = run(dr.suggest(ctx_of(inv, FakeDecider()), "Qwen/Qwen3-Big", "ngram"))
    assert out["candidates"] == [] and out["suggested"] is None and out["reason"] == "no_draft_needed"


def test_single_candidate_is_a_rules_suggestion(inv):
    fake = FakeDecider()
    out = run(dr.suggest(ctx_of(inv, fake), "Qwen/Qwen3-Big", "dflash"))
    assert out["suggested"] == "org/Qwen3-Big-DFlash" and out["source"] == "rules" and fake.calls == []
    assert fake.rules[0]["feature"] == "draft"


def test_model_ranks_several_candidates(inv):
    fake = FakeDecider(answers={"draft": ("Qwen/Qwen3-Mid", 0.9)})
    out = run(dr.suggest(ctx_of(inv, fake), "Qwen/Qwen3-Big", "draft_model"))
    assert out["source"] == "model" and out["suggested"] == "Qwen/Qwen3-Mid"
    assert refs(out["candidates"])[0] == "Qwen/Qwen3-Mid"
    q = fake.calls[0]
    assert sorted(o.key for o in q["options"]) == ["Qwen/Qwen3-Mid", "Qwen/Qwen3-Small"]
    assert "Qwen/Qwen3-Mid" not in "".join(o.description for o in q["options"])  # names only in quoted state
    assert "TARGET MODEL (untrusted" in q["state"]


def test_below_threshold_or_disabled_keeps_the_rules_order(inv):
    for fake in (FakeDecider(answers={"draft": ("Qwen/Qwen3-Mid", 0.5)}), FakeDecider(available=False), None):
        out = run(dr.suggest(ctx_of(inv, fake), "Qwen/Qwen3-Big", "draft_model"))
        assert out["suggested"] is None and out["source"] == "none"
        assert refs(out["candidates"]) == ["Qwen/Qwen3-Small", "Qwen/Qwen3-Mid"]  # same tokenizer first


def test_feature_off(inv):
    fake = FakeDecider()
    fake.rules_enabled = lambda f: f != "draft"
    out = run(dr.suggest(ctx_of(inv, fake), "Qwen/Qwen3-Big", "draft_model"))
    assert out["disabled"] is True and out["candidates"] == []


def test_unknown_target(inv):
    with pytest.raises(KeyError):
        run(dr.suggest(ctx_of(inv), "nope/nope", "draft_model"))


def test_eval_case_adapter():
    t = {"ref": "Qwen/Qwen3-14B", "config": {**QWEN_TINY, "vocab_size": 151936}, "weights_bytes": 28 * GIB}
    c1 = {"ref": "Qwen/Qwen3-0.6B", "config": {**QWEN_TINY, "vocab_size": 151936}, "weights_bytes": GIB}
    c2 = {"ref": "Qwen/Qwen3-4B", "config": {**QWEN_TINY, "vocab_size": 151936}, "weights_bytes": 8 * GIB}
    ec = dr.eval_case({"name": "x", "label": ["Qwen/Qwen3-0.6B"], "method": "draft_model", "target": t,
                       "candidates": [c1, c2]})
    assert ec.rules is None and sorted(o.key for o in ec.options) == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]
    ec = dr.eval_case({"name": "y", "label": ["Qwen/Qwen3-0.6B"], "method": "draft_model", "target": t,
                       "candidates": [c1]})
    assert ec.rules == "Qwen/Qwen3-0.6B"


def test_draft_suggest_api(api_stack, fake_hf_repo):
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    url = "/dgxmm/api/decider/draft-suggest"
    r = api_stack.post(url, headers=h, json={"target_ref": "org/big-model", "method": "draft_model"})
    assert r.status_code == 200, r.text
    assert refs(r.json()["candidates"]) == ["Qwen/Qwen3-Tiny"]  # same family + vocab, smaller
    assert api_stack.post(url, headers=h, json={"target_ref": "org/big-model", "method": "magic"}).status_code == 422
    assert api_stack.post(url, headers=h, json={"target_ref": "nope/x", "method": "eagle"}).status_code == 404
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    vh = {"X-CSRF-Token": r.json()["csrf_token"]}
    assert api_stack.post(url, headers=vh, json={"target_ref": "org/big-model",
                                                 "method": "draft_model"}).status_code == 403
