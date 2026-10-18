#!/usr/bin/env python3
"""Evaluate one decision-model feature on a corpus: rules, the model on its own, and the pipeline (rules first, then
the model at or above the feature's threshold), with coverage, precision and Wilson 95 % bounds per threshold.

    .venv/bin/python tools/eval_feature.py <feature> <corpus> [--base URL --key-file FILE | --rules-only]
                                           [--rate 2] [--thresholds 0.5,0.6,...] [--threshold 0.8] [--json OUT]

``<corpus>``:
- ``builtin``: the shipped features' real corpora (docs/decider-evaluation.md), through
  scripts/decider_eval/evaluate.py: ``diagnosis``, ``parser`` (``parser_tool`` / ``parser_reasoning``), ``kind``,
  ``workload``.
- a directory (default for a new feature: ``tests/fixtures/decider_corpus/<feature>/``) or a file of cases:
  ``*.jsonl`` (one JSON object per line) and ``*.json`` (an object or a list). Every case has ``name`` and ``label``
  (the accepted answer, or a list of accepted answers) plus whatever the feature's adapter reads.

Adapters: ``eval_case(case) -> EvalCase(rules, state, question, options)`` turns a case into DGXMM's rules answer
(or None) and the exact question DGXMM would ask. They are looked up in ``ADAPTERS`` below, else as the function
``eval_case`` of the module ``dgxmm.decider.<feature>``, so a feature lane only writes that function.

The decider is asked directly (read-only, at most --rate requests/s) with the same state / question / options,
also where a rule fired, so the model is measured on its own. The key is read from the file at request time and
never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgxmm.decider import registry  # noqa: E402
from dgxmm.decider.common import wilson  # noqa: E402
from dgxmm.decider.types import Option  # noqa: E402

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)] + [0.97, 0.99]
CORPORA = ROOT / "tests" / "fixtures" / "decider_corpus"
BUILTIN = ("diagnosis", "parser", "parser_tool", "parser_reasoning", "kind", "workload")


@dataclass
class EvalCase:
    rules: str | None   # DGXMM's rules answer, or None when the rules have no answer
    state: str
    question: str
    options: list[Option]


ADAPTERS: dict[str, Callable[[dict], EvalCase]] = {}  # feature -> adapter (else dgxmm.decider.<feature>.eval_case)


def _evaluate_module():
    spec = importlib.util.spec_from_file_location("decider_eval_evaluate",
                                                  ROOT / "scripts" / "decider_eval" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def adapter_for(feature: str) -> Callable[[dict], EvalCase]:
    if registry.find(feature) is None:
        raise SystemExit(f"unknown feature {feature!r}; registered: {', '.join(s.name for s in registry.groups())}")
    if feature in ADAPTERS:
        return ADAPTERS[feature]
    try:
        mod = importlib.import_module(f"dgxmm.decider.{feature}")
    except ImportError:
        mod = None
    fn = getattr(mod, "eval_case", None)
    if fn is None:
        raise SystemExit(f"feature {feature} has no evaluation adapter (dgxmm.decider.{feature}.eval_case)")
    return fn


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"corpus {path} not found")
    files = sorted(p for p in (path.iterdir() if path.is_dir() else [path]) if p.suffix in (".json", ".jsonl"))
    cases: list[dict] = []
    for f in files:
        if f.suffix == ".jsonl":
            cases += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        else:
            data = json.loads(f.read_text())
            cases += data if isinstance(data, list) else [data]
    for i, c in enumerate(cases):
        if not isinstance(c, dict) or "label" not in c:
            raise SystemExit(f"case {i} of {path} has no label")
        c.setdefault("name", f"case{i}")
    return cases


def _accepted(label) -> set[str]:
    return {str(x) for x in label} if isinstance(label, (list, tuple, set)) else {str(label)}


async def evaluate(feature: str, corpus: str, asker, thresholds=THRESHOLDS, threshold_used: float | None = None):
    """The report for one feature (see ``report``)."""
    thr = registry.get(registry.group_of(feature)).default_threshold if threshold_used is None else threshold_used
    if corpus == "builtin":
        return await _builtin(feature, asker, thresholds, thr)
    fn = adapter_for(feature)
    items = []
    for case in load_cases(Path(corpus)):
        ec = fn(case)
        ok = _accepted(case["label"])
        m = await asker.ask(feature, ec.state, ec.question, ec.options)
        if "choice" in m:
            m["ok"] = m["choice"] in ok
        pipeline = ec.rules or (m.get("choice") if m.get("p", 0) >= thr else None)
        items.append({"name": case["name"], "label": case["label"], "rules": ec.rules,
                      "rules_ok": (ec.rules in ok) if ec.rules else None, "model": m, "pipeline": pipeline,
                      "pipeline_ok": (pipeline in ok) if pipeline else None})
    return report(feature, items, thresholds, thr)


async def _builtin(feature: str, asker, thresholds, thr: float) -> dict:
    if feature not in BUILTIN:
        raise SystemExit(f"no built-in corpus for {feature}; built-in: {', '.join(BUILTIN)}")
    ev = _evaluate_module()
    if feature.startswith("parser"):  # parser = both questions (tool-call and reasoning parser) together
        res = await ev.eval_parser(asker, thr)
        which = ["tool", "reasoning"] if feature == "parser" else [feature.removeprefix("parser_")]
        return report(feature, [i for w in which for i in res[w]["items"]], thresholds, thr)
    res = await getattr(ev, f"eval_{feature}")(asker, thr)
    return report(feature, res["items"], thresholds, thr)


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _bounds(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"wilson_lo": None if lo is None else round(lo, 4), "wilson_hi": None if hi is None else round(hi, 4)}


def report(feature: str, items: list[dict], thresholds=THRESHOLDS, threshold_used: float = 0.8) -> dict:
    """Coverage / precision / Wilson bounds of the rules, of the model answers per threshold (coverage = share of
    the model's answers with p >= t) and of the pipeline at ``threshold_used``."""
    n = len(items)
    ruled = [i for i in items if i.get("rules")]
    rk = sum(1 for i in ruled if i.get("rules_ok"))
    piped = [i for i in items if i.get("pipeline")]
    pk = sum(1 for i in piped if i.get("pipeline_ok"))
    scored = [(i["model"]["p"], bool(i["model"].get("ok"))) for i in items if i.get("model") and "p" in i["model"]]
    curve = []
    for t in thresholds:
        kept = [ok for p, ok in scored if p >= t]
        curve.append({"threshold": t, "n": len(kept), "coverage": _rate(len(kept), len(scored)),
                      "precision": _rate(sum(kept), len(kept)), **_bounds(sum(kept), len(kept))})
    return {"feature": feature, "n": n, "threshold_used": threshold_used, "model_answers": len(scored),
            "rules": {"n": len(ruled), "coverage": _rate(len(ruled), n), "precision": _rate(rk, len(ruled)),
                      **_bounds(rk, len(ruled))},
            "pipeline": {"n": len(piped), "coverage": _rate(len(piped), n), "precision": _rate(pk, len(piped)),
                         **_bounds(pk, len(piped))},
            "curve": curve, "items": items}


def _f(v) -> str:
    return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def render(rep: dict) -> str:
    r, p = rep["rules"], rep["pipeline"]
    lines = [f"feature {rep['feature']}: {rep['n']} cases, {rep['model_answers']} model answers",
             f"rules:    n={r['n']} coverage={_f(r['coverage'])} precision={_f(r['precision'])} "
             f"wilson=[{_f(r['wilson_lo'])}, {_f(r['wilson_hi'])}]",
             f"pipeline: n={p['n']} coverage={_f(p['coverage'])} precision={_f(p['precision'])} "
             f"wilson=[{_f(p['wilson_lo'])}, {_f(p['wilson_hi'])}] at threshold {rep['threshold_used']:.2f}",
             "model alone:", f"{'threshold':>9} {'n':>5} {'coverage':>9} {'precision':>9} {'wilson_lo':>9} "
                             f"{'wilson_hi':>9}"]
    for c in rep["curve"]:
        lines.append(f"{c['threshold']:>9.2f} {c['n']:>5} {_f(c['coverage']):>9} {_f(c['precision']):>9} "
                     f"{_f(c['wilson_lo']):>9} {_f(c['wilson_hi']):>9}")
    return "\n".join(lines)


async def amain(args) -> int:
    ev = _evaluate_module()
    asker = ev.Asker(None if args.rules_only else args.base, Path(args.key_file) if args.key_file else None,
                     args.rate)
    thresholds = [float(x) for x in args.thresholds.split(",")] if args.thresholds else THRESHOLDS
    corpus = args.corpus
    if corpus == "default":
        corpus = str(CORPORA / args.feature)
    try:
        rep = await evaluate(args.feature, corpus, asker, thresholds, args.threshold)
    finally:
        await asker.http.aclose()
    print(render(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("feature")
    ap.add_argument("corpus", help="builtin | default | a directory or file of cases")
    ap.add_argument("--base", default="http://127.0.0.1:8101")
    ap.add_argument("--key-file")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--thresholds", help="comma-separated (default 0.30..0.95 step 0.05, 0.97, 0.99)")
    ap.add_argument("--threshold", type=float, help="pipeline threshold (default: the feature's)")
    ap.add_argument("--json", help="write the full report (with items) here")
    args = ap.parse_args()
    if not args.rules_only and not args.key_file:
        ap.error("--key-file is required unless --rules-only")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
