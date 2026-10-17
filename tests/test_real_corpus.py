"""Regression tests over REAL data (docs/decider-evaluation.md): start-up failures induced on the DGX, configs and
chat templates of real Hugging Face repos, metric summaries of real traffic. The rules (no decision model) must be
precise on all of them; where the rules abstain the model is consulted at run time (measured in the doc)."""
from __future__ import annotations

import pytest

from dgxmm.decider import config_suggest as cs
from dgxmm.decider import diagnosis as dg
from dgxmm.decider import kind as kd
from dgxmm.decider import workload as wl
from tests import real_corpus as rc

FAILURES = rc.load_failures()
REPOS = rc.load_repos()
LABELS = rc.load_labels()

# What the rules must answer for each real failure (None = no single rule fires -> the model is asked).
EXPECTED_RULES = {
    "auto-tool-no-parser": "bad_parser_flag", "bad-reasoning-parser": "bad_parser_flag",
    "bad-tool-parser": "bad_parser_flag", "cgroup-oom": None, "ctx-above-model": "context_too_long",
    "disk-full": "disk_full", "gated-no-token": "auth_gated", "kv-none-left": "kv_cache_too_small",
    "kv-small-budget": "kv_cache_too_small", "kv-too-small": "kv_cache_too_small",
    "broken-tokenizer": "tokenizer_or_template", "offline-missing": "offline_missing_files",
    "port-in-container": "port_in_use", "port-published": "port_in_use",
    "trust-remote-code": "trust_remote_code_required", "unsupported-arch": "unsupported_architecture",
}


def test_corpora_are_present_and_labelled():
    assert len(FAILURES) >= 14
    assert {f.label for f in FAILURES} <= set(dg.CAUSES)
    assert len(REPOS) >= 100 and set(LABELS) == set(REPOS)
    kinds = {k for lab in LABELS.values() for k in lab["kind"]}
    assert kinds <= {k for k, _ in kd.KINDS}
    assert {"llm", "draft", "embedding", "reranker", "vision-llm", "speech", "decision", "other"} <= kinds


@pytest.mark.parametrize("f", FAILURES, ids=lambda f: f.name)
def test_real_failure_rules(f):
    text = f"{f.log}\n{f.message}" if f.log else f.message
    hits = [c for c, _ in dg.rule_hits(text, f.exit_code)]
    assert f.name in EXPECTED_RULES, "new fixture: add its expected rule answer"
    want = EXPECTED_RULES[f.name]
    got = hits[0] if len(hits) == 1 else None
    assert got == want, f"{f.name}: rules {hits} (label {f.label})"
    if got is not None:
        assert got == f.label  # a rule that fires is never wrong on real data


@pytest.mark.parametrize("f", FAILURES, ids=lambda f: f.name)
def test_real_failure_state_carries_the_root_cause(f):
    """The model sees the root-cause line in ERROR LINES even when the tail is long (no masked evidence)."""
    state = dg.build_state(engine=f.engine, model_ref=f.model_ref, exit_code=f.exit_code, message=f.message,
                           hits=dg.rule_hits(f.log, f.exit_code), log_text=f.log or f.message)
    assert "KeyError: '***'" not in state
    assert len(state) < 12_000


def _parser_items(which):
    rule = cs.rule_tool if which == "tool" else cs.rule_reasoning
    for ref, lab in sorted(LABELS.items()):
        if lab.get(which) is not None:
            yield ref, lab[which], rule(rc.model_info(REPOS[ref]))


@pytest.mark.parametrize("which", ["tool", "reasoning"])
def test_parser_rules_auto_apply_precision_is_total(which):
    """Strong (template) rules auto-fill the wizard: every one must be right on the real corpus."""
    wrong = [(ref, hit.value, truth) for ref, truth, hit in _parser_items(which)
             if hit is not None and hit.strong and hit.value not in truth]
    strong = [1 for _, _, hit in _parser_items(which) if hit is not None and hit.strong]
    assert not wrong, wrong
    assert len(strong) >= 40


@pytest.mark.parametrize("which", ["tool", "reasoning"])
def test_parser_rules_suggestion_precision(which):
    items = list(_parser_items(which))
    answered = [(ref, hit.value, truth) for ref, truth, hit in items if hit is not None]
    wrong = [a for a in answered if a[1] not in a[2]]
    assert len(answered) / len(items) >= 0.9
    assert not wrong, wrong


def test_kind_rules_on_real_repos():
    """The scanner's kind (plus the second rule tier for `other`) is right wherever it decides."""
    wrong, undecided = [], []
    for ref, lab in sorted(LABELS.items()):
        a = rc.as_artifact(REPOS[ref])
        kind = a.kind
        if kd.needs_kind(a):
            hit = kd.rule_kind(a.ref, a.architecture, a.config, list(a.files))
            kind = hit[0] if hit else None
        if kind is None:
            undecided.append(ref)
        elif kind not in lab["kind"]:
            wrong.append((ref, kind, lab["kind"]))
    assert not wrong, wrong
    assert len(undecided) <= 6, undecided


@pytest.mark.parametrize("ref", ["vllm-sr/d3-mini", "vllm-sr/d3-nano", "perplexity-ai/pplx-decider-flash-v1.2",
                                 "facebook/bart-large-mnli", "protectai/deberta-v3-base-prompt-injection-v2"])
def test_decision_models_are_decision(ref):
    assert rc.as_artifact(REPOS[ref]).kind == "decision"


def test_real_workload_rules():
    items = rc.load_workloads()
    assert len(items) >= 6 and {w["label"] for w in items} >= {"idle", "batch_throughput"}
    for w in items:
        hit = wl.rule_label(w["summary"], w.get("config", {}))
        if hit is not None:
            assert hit[0] == w["label"], (w["name"], hit)
        if w["label"] in ("idle", "batch_throughput"):
            assert hit is not None and hit[0] == w["label"], w["name"]


def test_no_real_chat_template_looks_like_an_injection():
    """The parser auto-fill guard (config_suggest.injection_shaped) must not fire on real repos' templates/names."""
    flagged = []
    for ref in sorted(REPOS):
        info = rc.model_info(REPOS[ref])
        if cs.injection_shaped(info.template, info.ref):
            flagged.append(ref)
    assert len(REPOS) >= 100 and not flagged, flagged
