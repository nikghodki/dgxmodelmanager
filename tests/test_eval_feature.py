"""The generalised evaluation harness (tools/eval_feature.py): Wilson bounds, the built-in corpora of the shipped
features (rules only, no decider), and adapter-based corpora under tests/fixtures/decider_corpus/<feature>/."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dgxmm.decider.common import wilson
from dgxmm.decider.types import Option

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "eval_feature.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("eval_feature", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_feature"] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def test_wilson_bounds():
    lo, hi = wilson(126, 127)
    assert round(lo, 2) == 0.96 and hi == pytest.approx(0.9986, abs=1e-3)  # the kind calibration in protocol.py
    assert wilson(0, 0) == (None, None)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.29
    lo, hi = wilson(10, 10)
    assert 0.72 < lo < 0.73 and hi == 1.0


def test_table_from_items():
    ev = load_tool()
    items = [{"rules": "a", "rules_ok": True, "model": {"choice": "a", "p": 0.95, "ok": True}, "pipeline": "a",
              "pipeline_ok": True},
             {"rules": None, "rules_ok": None, "model": {"choice": "b", "p": 0.85, "ok": False}, "pipeline": "b",
              "pipeline_ok": False},
             {"rules": None, "rules_ok": None, "model": {"choice": "a", "p": 0.55, "ok": True}, "pipeline": None,
              "pipeline_ok": None},
             {"rules": None, "rules_ok": None, "model": {"error": "rules_only"}, "pipeline": None,
              "pipeline_ok": None}]
    rep = ev.report("x", items, thresholds=[0.5, 0.9], threshold_used=0.8)
    assert rep["n"] == 4 and rep["rules"]["coverage"] == 0.25 and rep["rules"]["precision"] == 1.0
    t50, t90 = rep["curve"]
    assert (t50["n"], t50["coverage"], t50["precision"]) == (3, 1.0, round(2 / 3, 4))
    assert (t90["n"], t90["precision"]) == (1, 1.0) and t90["wilson_lo"] == round(wilson(1, 1)[0], 4)
    assert rep["pipeline"]["coverage"] == 0.5 and rep["pipeline"]["precision"] == 0.5
    text = ev.render(rep)
    assert "threshold" in text and "0.90" in text and "wilson" in text.lower()


class FakeAsker:
    offline = False

    def __init__(self, answers):
        self.answers, self.latencies, self.asked = answers, [], []

    async def ask(self, feature, state, question, options, timeout_s=30.0):
        self.asked.append((feature, state))
        choice, p = self.answers[state]
        return {"choice": choice, "p": p}


def test_adapter_corpus(tmp_path, monkeypatch):
    ev = load_tool()
    corpus = tmp_path / "health"
    corpus.mkdir()
    (corpus / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in [
        {"name": "c1", "label": "kv_pressure", "kv": 0.99},
        {"name": "c2", "label": "healthy", "kv": 0.2},
        {"name": "c3", "label": ["throughput_drop", "error_creep"], "kv": 0.5},
    ]) + "\n")
    (corpus / "README.md").write_text("provenance")

    def eval_case(case):
        opts = [Option(k, k) for k in ("kv_pressure", "healthy", "throughput_drop", "error_creep")]
        rules = "kv_pressure" if case["kv"] > 0.95 else None
        return ev.EvalCase(rules=rules, state=f"kv={case['kv']}", question="q", options=opts)

    monkeypatch.setitem(ev.ADAPTERS, "health", eval_case)
    asker = FakeAsker({"kv=0.99": ("kv_pressure", 0.97), "kv=0.2": ("healthy", 0.91), "kv=0.5": ("healthy", 0.6)})
    rep = asyncio.run(ev.evaluate("health", str(corpus), asker, thresholds=[0.5, 0.9]))
    assert rep["n"] == 3 and rep["rules"]["coverage"] == round(1 / 3, 4)
    assert rep["curve"][0]["precision"] == round(2 / 3, 4) and rep["curve"][1]["precision"] == 1.0
    assert [a[0] for a in asker.asked] == ["health"] * 3  # the model is measured on every case


def test_adapter_discovery_and_errors(tmp_path):
    ev = load_tool()
    with pytest.raises(SystemExit):
        ev.adapter_for("nope")  # not a registered feature
    from dgxmm.decider import registry
    registry.register(registry.FeatureSpec("zz_noadapter", "zz_noadapter", "x", "x", 0.8))
    try:
        with pytest.raises(SystemExit):
            ev.adapter_for("zz_noadapter")  # registered but no adapter: a clear message, not a traceback
    finally:
        registry._unregister("zz_noadapter")
    assert ev.adapter_for("compat").__module__ == "dgxmm.decider.compat"
    with pytest.raises(SystemExit):
        ev.load_cases(tmp_path / "missing")


def test_cli_rules_only_on_a_builtin_corpus():
    r = subprocess.run([sys.executable, str(TOOL), "kind", "builtin", "--rules-only"], capture_output=True,
                       text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "kind" in r.stdout and "rules" in r.stdout and "coverage" in r.stdout
