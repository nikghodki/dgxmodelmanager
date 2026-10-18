#!/usr/bin/env python3
"""Evaluate the ⌘K command interpretation (feature 6) end to end: DGXMM's own pipeline (rules, then the decision
model's action and target questions) on the hand-written corpus tests/fixtures/decider_corpus/command/ against
its fixture inventory (public model names only).

    python scripts/decider_eval/command_eval.py [--base http://<decider>:<port> --key-file FILE] [--rate 2]
                                                [--json OUT]

Without --base the rules alone are measured (decision model unavailable). The key is read from the file at request
time and never printed. Metrics per threshold t (see docs/decider-evaluation.md):

- top-1: the cases where the best interpretation has confidence >= t (shown alone): precision = its key is an
  accepted label; coverage = share of the answerable cases (label not "none").
- top-3: every case where anything is shown (best >= 0.3): one interpretation if best >= t, else up to three
  >= 0.1; accuracy = an accepted label is among those shown.
- none cases (destructive requests, injection, viewer-forbidden): correct when nothing is shown; "destructive"
  counts shown stop / restart interpretations for them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dgxmm.decider import command as cmd  # noqa: E402
from dgxmm.decider.client import DeciderError, SystemOneClient  # noqa: E402
from dgxmm.decider.protocol import NullDecider  # noqa: E402
from dgxmm.decider.types import Decision  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "decider_corpus" / "command"
THRESHOLDS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


class LiveDecider:
    """The decider interface interpret() needs, asking the decision model directly (no cache, rate-limited)."""

    def __init__(self, base: str, key_file: Path, rate: float):
        self.http = httpx.AsyncClient()
        self.client = SystemOneClient(self.http, base_url=lambda: base, api_key=lambda: key_file.read_text().strip(),
                                      revision="eval", cache_size=0)
        self.gap, self.last, self.calls, self.errors, self.latencies = 1.0 / rate, 0.0, 0, 0, []

    async def available(self) -> bool:
        return True

    def feature_enabled(self, feature: str) -> bool:
        return True

    def threshold(self, feature: str) -> float:
        return 0.0

    async def decide(self, *, feature, subject, state, question, options, threshold=None, timeout_s=2.0, log=True):
        wait = self.last + self.gap - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self.last = time.monotonic()
        self.calls += 1
        try:
            a = await self.client.choose(feature=feature, state=state, question=question, options=options,
                                         timeout_s=30.0)
        except DeciderError:
            self.errors += 1
            return Decision(feature=feature, subject=subject, source="none", choice=None, confidence=None)
        self.latencies.append(a.latency_ms)
        return Decision(feature=feature, subject=subject, source="model", choice=a.choice, confidence=a.confidence,
                        probabilities=dict(a.probabilities))

    def record_rules(self, *, feature, subject, choice, reason, options=None):
        return Decision(feature=feature, subject=subject, source="rules", choice=choice, confidence=1.0)


def key_of(i: cmd.Interpretation) -> str:
    if i.target is None:
        return i.action
    t = i.target
    return f"{i.action}:{t.kind}:{t.path if t.kind == 'page' else t.ref if t.kind == 'model' else t.name}"


def accepted(label) -> set[str]:
    return set(label) if isinstance(label, list) else {label}


async def run(decider, cases: list[dict], inv: dict) -> list[dict]:
    items = []
    for c in cases:
        targets = cmd.build_targets(inv["deployments"], inv["models"], admin=c.get("admin", True))
        ranked, early = await cmd.rank(c["text"], targets, decider, admin=c.get("admin", True))
        items.append({"name": c["name"], "category": c["category"], "text": c["text"], "admin": c.get("admin", True),
                      "label": c["label"], "refused": bool(early and early.refused),
                      "ranked": [{"key": key_of(i), "p": round(i.confidence, 4), "source": i.source}
                                 for i in ranked[:5]]})
    return items


def shown(item: dict, t: float) -> list[str]:
    r = item["ranked"]
    if not r or r[0]["p"] < cmd.FLOOR:
        return []
    if r[0]["p"] >= t:
        return [r[0]["key"]]
    return [x["key"] for x in r if x["p"] >= cmd.ALT_MIN][:3]


def metrics(items: list[dict], t: float) -> dict:
    ans = [i for i in items if i["label"] != "none"]
    none = [i for i in items if i["label"] == "none"]
    single = [i for i in ans if i["ranked"] and i["ranked"][0]["p"] >= t]
    ok1 = sum(1 for i in single if i["ranked"][0]["key"] in accepted(i["label"]))
    shown_ans = [(i, shown(i, t)) for i in ans if shown(i, t)]
    ok3 = sum(1 for i, s in shown_ans if accepted(i["label"]) & set(s))
    none_ok = sum(1 for i in none if not shown(i, t))
    destructive = sum(1 for i in none for k in shown(i, t) if k.split(":")[0] in ("stop", "restart"))
    false_single = sum(1 for i in none if i["ranked"] and i["ranked"][0]["p"] >= t)
    from dgxmm.decider.common import wilson
    lo, _ = wilson(ok1, len(single))
    return {"threshold": t, "top1_n": len(single), "top1_precision": round(ok1 / len(single), 4) if single else None,
            "top1_wilson_lo": None if lo is None else round(lo, 4),
            "top1_coverage": round(len(single) / len(ans), 4) if ans else None,
            "top3_n": len(shown_ans), "top3_accuracy": round(ok3 / len(shown_ans), 4) if shown_ans else None,
            "shown_coverage": round(len(shown_ans) / len(ans), 4) if ans else None,
            "none_n": len(none), "none_correct": none_ok, "none_destructive_shown": destructive,
            "none_single_answers": false_single}


def by_category(items: list[dict], t: float) -> dict:
    out = {}
    for cat in sorted({i["category"] for i in items}):
        sub = [i for i in items if i["category"] == cat]
        m = metrics(sub, t)
        out[cat] = {"n": len(sub), "top1_precision": m["top1_precision"], "top1_coverage": m["top1_coverage"],
                    "top3_accuracy": m["top3_accuracy"], "none_correct": m["none_correct"], "none_n": m["none_n"]}
    return out


async def amain(args) -> int:
    inv = json.loads((CORPUS / "inventory.json").read_text())
    cases = [json.loads(line) for line in (CORPUS / "utterances.jsonl").read_text().splitlines() if line.strip()]
    decider = LiveDecider(args.base, Path(args.key_file), args.rate) if args.base else NullDecider()
    try:
        items = await run(decider, cases, inv)
    finally:
        if isinstance(decider, LiveDecider):
            await decider.http.aclose()
    curve = [metrics(items, t) for t in THRESHOLDS]
    rep = {"n": len(items), "model": bool(args.base), "curve": curve,
           "by_category_at_0.7": by_category(items, 0.7), "items": items}
    if isinstance(decider, LiveDecider):
        lat = sorted(decider.latencies)
        rep.update(calls=decider.calls, errors=decider.errors,
                   latency_p50_ms=round(lat[len(lat) // 2], 1) if lat else None)
    print(f"{len(items)} utterances ({'live decider' if args.base else 'rules only'})")
    print(f"{'thr':>5} {'top1 n':>6} {'prec':>6} {'lo':>6} {'cov':>6} {'top3':>6} {'none ok':>8} {'destr':>5}")
    for m in curve:
        f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{m['threshold']:>5.2f} {m['top1_n']:>6} {f(m['top1_precision']):>6} {f(m['top1_wilson_lo']):>6} "
              f"{f(m['top1_coverage']):>6} {f(m['top3_accuracy']):>6} {m['none_correct']:>4}/{m['none_n']:<3} "
              f"{m['none_destructive_shown']:>5}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", help="decider base URL (omit: rules only)")
    ap.add_argument("--key-file", help="file with the decider's API key")
    ap.add_argument("--rate", type=float, default=2.0, help="decider requests per second (default 2)")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()
    if args.base and not args.key_file:
        ap.error("--base needs --key-file")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
