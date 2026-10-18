#!/usr/bin/env python3
"""Evaluate the schedule proposals (feature 7): the decision model's propose / skip answer per candidate window.

    python scripts/decider_eval/schedule_eval.py [--base http://<decider>:<port> --key-file FILE] [--rate 2]
                                                 [--db data/dgxmm.sqlite3] [--json OUT]

Cases (tests/fixtures/decider_corpus/schedule/patterns.jsonl): weekly traffic patterns — office hours, late office
hours, nightly batch, weekend use, a lunch gap, a short daily gap, office hours with stray night requests, irregular
use, always on — over 8 to 30 days, at three priorities and two memory-pressure levels. Each pattern is rendered as
one-minute rollups whose busy minutes are real rollups of a test deployment's traffic
(tests/fixtures/workload/raw/prod-h1-wl.rollups.json) and turned into a profile and candidates by DGXMM's own code
(profile_from_rollups, candidates, build_state); every candidate is one question. ``--db``: also profile the
deployments in a DGXMM database (read-only; real traffic; reported, never written to the corpus).

Labels (the policy the model is judged against): propose when the window hours were idle in every one of at least
two observed weeks, every window is at least 4 h long, and the deployment's priority is below 90; skip otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dgxmm.decider import schedule as sch  # noqa: E402
from tools.eval_feature import THRESHOLDS, report  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "decider_corpus" / "schedule" / "patterns.jsonl"
REAL = ROOT / "tests" / "fixtures" / "workload" / "raw" / "prod-h1-wl.rollups.json"
END = datetime(2026, 10, 19, 3, 0, tzinfo=timezone.utc)  # a Monday night: the proposals job runs then
GIB = 1024**3


def _week(loc: datetime) -> int:
    return (loc.date() - timedelta(days=loc.weekday())).toordinal() // 7


def active_fn(spec: dict):
    """(local minute) -> active hour? for a pattern."""
    p, rng = spec["pattern"], random.Random(spec.get("seed", 0))
    cache: dict = {}

    def coin(key, prob):
        if key not in cache:
            cache[key] = rng.random() < prob
        return cache[key]

    def f(loc: datetime) -> bool:
        wd, h = loc.weekday(), loc.hour
        if p == "office":
            return wd < 5 and 8 <= h < 19
        if p == "office_late":
            return wd < 5 and 10 <= h < 22
        if p == "nightly_batch":
            return 1 <= h < 4
        if p == "weekend":
            return wd >= 5 and 9 <= h < 21
        if p == "lunch":
            return wd < 5 and (9 <= h < 12 or 13 <= h < 18)
        if p == "short_gap":
            return not 17 <= h < 19
        if p == "office_stray":  # office hours plus stray night use in some weeks
            return (wd < 5 and 8 <= h < 19) or ((h >= 22 or h < 6) and coin((_week(loc), wd, h), 0.08))
        if p == "irregular":
            return coin((_week(loc), wd, h), spec.get("p", 0.35))
        if p == "always_on":
            return True
        raise ValueError(p)
    return f


def rollups(spec: dict, busy_rows: list[dict], idle_row: dict) -> list:
    act, rng = active_fn(spec), random.Random(spec.get("seed", 0) + 1)
    out, t, i = [], END - timedelta(days=spec["days"]), 0
    while t < END:
        if act(t):  # an active hour: most minutes busy (real minute shapes), some quiet like real chat traffic
            if rng.random() < 0.75:
                out.append((t, busy_rows[i % len(busy_rows)]))
                i += 1
            else:
                out.append((t, idle_row))
        else:
            out.append((t, idle_row))
        t += timedelta(minutes=1)
    return out


def label(c: sch.Candidate, priority: int) -> str:
    ok = c.min_weeks >= 2 and c.all_idle and all(ln >= 4 for _, ln in c.windows) and priority < 90
    return "propose" if ok else "skip"


def build_cases(specs: list[dict]) -> list[dict]:
    real = json.loads(REAL.read_text())
    rows = [json.loads(d) if isinstance(d, str) else d for _, d in real]
    busy = [r for r in rows if sch._busy(r) and r.get("running") is not None]
    idle = next(r for r in rows if not sch._busy(r) and r.get("samples"))
    cases = []
    for spec in specs:
        prof = sch.profile_from_rollups(rollups(spec, busy, idle), timezone.utc)
        view = {"engine": "vllm", "name": f"{spec['pattern']}-{spec['days']}d", "priority": spec["priority"],
                "reserved_bytes": spec.get("reserved_gib", 20) * GIB, "sleep_capable": True}
        cands = sch.candidates(prof)
        if not cands:
            cases.append({"name": f"{spec['name']}-none", "spec": spec, "label": "skip", "rules": "skip",
                          "state": "", "candidate": None})
            continue
        for c in cands[:sch.MAX_CANDIDATES]:
            st = sch.build_state(view, prof, c, spec["pressure"], "UTC")
            cases.append({"name": f"{spec['name']}-{c.key}", "spec": spec, "label": label(c, spec["priority"]),
                          "rules": "skip" if prof.days < sch.MIN_DAYS or sch.rule_skip(c) else None, "state": st,
                          "candidate": c.key,
                          "strong": c.strong and spec["priority"] < sch.STRONG_MAX_PRIORITY})
    return cases


def real_profiles(db: str) -> list[dict]:
    """Real deployments' profiles from a DGXMM database (read-only): days, observed hours, candidates."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for (dep,) in con.execute("SELECT DISTINCT deployment_id FROM metric_rollups WHERE deployment_id IS NOT NULL"):
        rows = [(datetime.fromisoformat(m).replace(tzinfo=timezone.utc), json.loads(d)) for m, d in
                con.execute("SELECT minute, data FROM metric_rollups WHERE deployment_id = ?", (dep,))]
        prof = sch.profile_from_rollups(rows, timezone.utc)
        out.append({"minutes": len(rows), "days": prof.days, "observed_hours": len(prof.hours),
                    "candidates": len(sch.candidates(prof)), "eligible": prof.days >= sch.MIN_DAYS})
    host = [json.loads(d) for (d,) in con.execute("SELECT data FROM metric_rollups WHERE deployment_id IS NULL")]
    con.close()
    return out + [{"host_minutes": len(host)}]


async def amain(args) -> int:
    specs = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    cases = build_cases(specs)
    ev = __import__("importlib").import_module("tools.eval_feature")._evaluate_module()
    asker = ev.Asker(args.base, Path(args.key_file) if args.key_file else None, args.rate)
    items = []
    try:
        for c in cases:
            m = await asker.ask("schedule", c["state"], sch.QUESTION, sch.OPTIONS) if c["state"] else \
                {"error": "no candidate"}
            if "choice" in m:
                m["ok"] = m["choice"] == c["label"]
            items.append({"name": c["name"], "label": c["label"], "rules": c["rules"],
                          "rules_ok": (c["rules"] == c["label"]) if c["rules"] else None, "model": m,
                          "pattern": c["spec"]["pattern"], "strong": c.get("strong")})
    finally:
        await asker.http.aclose()
    for i in items:  # the pipeline: rules where they decide, else the model's propose at/above threshold
        p = i["model"].get("p", 0)
        i["pipeline"] = i["rules"] or (i["model"].get("choice") if p >= args.threshold else None)
        i["pipeline_ok"] = (i["pipeline"] == i["label"]) if i["pipeline"] else None
    rep = report("schedule", items, THRESHOLDS, args.threshold)
    # propose precision: only "propose" answers create an insight
    prop = []
    for t in THRESHOLDS:
        kept = [i for i in items if i["model"].get("choice") == "propose" and i["model"].get("p", 0) >= t]
        ok = sum(1 for i in kept if i["label"] == "propose")
        from dgxmm.decider.common import wilson
        lo, _ = wilson(ok, len(kept))
        prop.append({"threshold": t, "n": len(kept), "precision": round(ok / len(kept), 4) if kept else None,
                     "wilson_lo": None if lo is None else round(lo, 4),
                     "recall": round(ok / max(sum(1 for i in items if i["label"] == "propose"), 1), 4)})
    rep["propose_curve"] = prop
    strong = [i for i in items if i.get("strong")]
    rep["rules_only_strong"] = {"n": len(strong), "correct": sum(1 for i in strong if i["label"] == "propose")}
    rep["labels"] = {k: sum(1 for i in items if i["label"] == k) for k in ("propose", "skip")}
    if args.db:
        rep["real"] = real_profiles(args.db)
    from tools.eval_feature import render
    print(render(rep))
    print("propose answers:", "  ".join(f"{p['threshold']:.2f}:{p['n']}/{p['precision']}" for p in prop))
    print("labels:", rep["labels"], "strong (rules-only proposals):", rep["rules_only_strong"])
    if args.db:
        print("real:", rep["real"])
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base")
    ap.add_argument("--key-file")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--db", help="a DGXMM database to profile real deployments from (read-only)")
    ap.add_argument("--json")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
