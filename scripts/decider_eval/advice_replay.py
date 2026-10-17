#!/usr/bin/env python3
"""Replay DGXMM's advisor + workload ranking (this checkout's code) on a real deployment's recorded metrics.

    python scripts/decider_eval/advice_replay.py --snapshot ~/eval/advice_snapshots.jsonl --hf ~/.cache/huggingface

Each snapshot line (written while traffic ran) holds the deployment view, its live 5-second samples (last 15 min)
and the workload answer the live app returned. The advice is recomputed with dgxmm.advisor.rules.advise on the
real artifacts of the HF cache and ordered with dgxmm.decider.workload.rank, once without a label and once with
the workload label, so the relevance ordering is exercised on real traffic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dgxmm.advisor.rules import AdviceInput, advise  # noqa: E402
from dgxmm.decider.workload import rank  # noqa: E402
from dgxmm.inventory.gguf_scan import scan_gguf_roots  # noqa: E402
from dgxmm.inventory.hf_scan import scan_hf_cache  # noqa: E402
from dgxmm.planner.plans import base_model_name  # noqa: E402


def artifacts(hf: Path) -> list:
    """The inventory as DGXMM builds it: HF snapshots plus GGUF files (the HF hub is also a GGUF root)."""
    return scan_hf_cache(hf / "hub") + scan_gguf_roots([hf / "hub"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--labels", default="single_user_chat,agent_tool_use,batch_throughput,long_context_rag,idle")
    args = ap.parse_args()
    arts = artifacts(args.hf)
    for ln in args.snapshot.read_text().splitlines():
        snap = json.loads(ln)
        v = snap["view"]
        art = next((a for a in arts if a.ref == v["model_ref"]), None)
        if art is None:
            continue
        base = base_model_name(art.ref)
        others = [a for a in arts if a.ref != art.ref and a.complete]
        inp = AdviceInput(engine=v["engine"], cfg=v["config"], artifact=art, samples=snap["samples"][-180:],
                          day=snap.get("day", []), estimate=v.get("estimate") or {}, reserved=v["reserved_bytes"],
                          variants=[a for a in others if a.kind == "llm" and base_model_name(a.ref) == base],
                          drafts=[a for a in others if a.kind == "draft" and base_model_name(a.ref).startswith(base)],
                          benchmarks=[])
        items = [a.as_dict() for a in advise(inp)]
        wl = snap.get("workload") or {}
        rec = {"ts": snap["ts"], "phase": snap.get("phase"), "live_workload": wl.get("label"),
               "live_confidence": wl.get("confidence"), "advice": [a["id"] for a in rank(items, None)],
               "by_label": {lab: [(a["id"], a["relevance"], a["workload_fit"]) for a in
                                  rank(items, {"label": lab, "confidence": 1.0})] for lab in args.labels.split(",")}}
        if wl.get("label"):
            rec["ranked_live"] = [(a["id"], a["relevance"], a["workload_fit"]) for a in rank(items, wl)]
        print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
