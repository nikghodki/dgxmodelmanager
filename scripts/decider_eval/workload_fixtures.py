#!/usr/bin/env python3
"""Cut DGXMM's real minute rollups of a test deployment into labelled workload fixtures.

    python scripts/decider_eval/workload_fixtures.py --db data/dgxmm.sqlite3 --deployment 5 \
        --phases ~/eval/phases.jsonl --out tests/fixtures/workload [--window 15 --step 1 --skip 2]

The traffic phases come from traffic.py's log (start/end of each profile). For every phase, windows of --window
minutes (starting --skip minutes after the phase began, so the previous profile has drained, and stepping by
--step) are summarised with DGXMM's own dgxmm.decider.workload.summarize -- the same summary the live feature sends
to the decision model, only over a shorter window than production's 1 h (an 18-minute phase cannot fill 30+
minutes; see docs/decider-evaluation.md). The database is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dgxmm.decider.workload import summarize  # noqa: E402

LABEL = {"chat": "single_user_chat", "agent": "agent_tool_use", "batch": "batch_throughput",
         "rag": "long_context_rag", "mixed": "mixed", "idle": "idle"}


def _ts(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    dt = datetime.fromisoformat(str(v))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def rollups(db: Path, dep: int) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT minute, data FROM metric_rollups WHERE deployment_id = ? ORDER BY minute",
                           (dep,)).fetchall()
    finally:
        con.close()
    return [{"ts": _ts(m), **(json.loads(d) if isinstance(d, str) else d)} for m, d in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, help="DGXMM database (read-only)")
    ap.add_argument("--rollups-json", type=Path,
                    help="rollups exported as [[minute, data], ...] (tests/fixtures/workload/raw) instead of --db")
    ap.add_argument("--deployment", type=int, default=0)
    ap.add_argument("--phases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--config", default="{}", help="deployment config (JSON) recorded with each fixture")
    args = ap.parse_args()
    if args.rollups_json:
        rows = [{"ts": _ts(m), **(json.loads(d) if isinstance(d, str) else d)}
                for m, d in json.loads(args.rollups_json.read_text())]
    else:
        rows = rollups(args.db, args.deployment)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config)
    n = 0
    for ln in args.phases.read_text().splitlines():
        ph = json.loads(ln)
        start, end = ph["start"] + args.skip * 60, ph["end"]
        w = 0
        t0 = start
        while t0 + args.window * 60 <= end + 30:
            sel = [r for r in rows if t0 <= r["ts"] < t0 + args.window * 60]
            if len(sel) >= args.window - 1:
                name = f"{ph['phase']}-w{w:02d}"
                fx = {"name": name, "label": LABEL[ph["phase"]], "window_minutes": args.window,
                      "summary": summarize(sel), "engine": "vllm", "model_ref": "Qwen/Qwen3-0.6B", "config": cfg,
                      "traffic": {k: ph[k] for k in ("ok", "err", "tool_calls", "prompt_tokens", "completion_tokens")
                                  if k in ph},
                      "source": "real minute rollups of a DGXMM test deployment on the DGX Spark, traffic from "
                                "scripts/decider_eval/traffic.py"}
                (args.out / f"{name}.json").write_text(json.dumps(fx, indent=1) + "\n")
                n += 1
                w += 1
            t0 += args.step * 60
    print(f"wrote {n} fixtures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
