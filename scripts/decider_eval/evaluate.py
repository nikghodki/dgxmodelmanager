#!/usr/bin/env python3
"""Evaluate DGXMM's rules and the decision model on the real corpora (docs/decider-evaluation.md).

Run on the DGX from the repository root (read-only calls to a running decider, at most --rate requests/s):

    .venv/bin/python scripts/decider_eval/evaluate.py --base http://127.0.0.1:8101 \
        --key-file data/secrets/decider_api_key --out ~/eval/results [diagnosis] [parser] [kind] [workload]

For every item the RULES answer (what DGXMM decides without the model) and the MODEL answer (the decider asked
directly with the same state / question / options DGXMM would send, even where a rule fired, so the model is
measured on its own) are recorded; the PIPELINE answer is rules first, then the model at or above the threshold.
Accuracy-vs-coverage curves of the model answers are written per feature (results/<feature>.json + summary).
The key is read from the file at request time and never printed.
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
from dgxmm.decider import config_suggest as cs  # noqa: E402
from dgxmm.decider import diagnosis as dg  # noqa: E402
from dgxmm.decider import kind as kd  # noqa: E402
from dgxmm.decider import workload as wl  # noqa: E402
from dgxmm.decider.client import DeciderError, SystemOneClient  # noqa: E402
from dgxmm.decider.protocol import DEFAULT_THRESHOLDS  # noqa: E402
from tests import real_corpus as rc  # noqa: E402

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)] + [0.97, 0.99]


class Asker:
    def __init__(self, base: str | None, key_file: Path | None, rate: float):
        self.offline = base is None  # --rules-only: no decider calls
        self.http = httpx.AsyncClient()
        self.client = SystemOneClient(self.http, base_url=lambda: base, api_key=lambda: key_file.read_text().strip(),
                                      revision="eval", cache_size=0)
        self.gap, self.last, self.latencies = 1.0 / rate, 0.0, []

    async def ask(self, feature, state, question, options, timeout_s=30.0):
        if self.offline:
            return {"error": "rules_only"}
        wait = self.last + self.gap - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self.last = time.monotonic()
        try:
            a = await self.client.choose(feature=feature, state=state, question=question, options=options,
                                         timeout_s=timeout_s)
        except DeciderError as e:
            return {"error": e.code}
        self.latencies.append(a.latency_ms)
        return {"choice": a.choice, "p": round(a.confidence, 4),
                "top3": sorted(a.probabilities.items(), key=lambda kv: -kv[1])[:3]}


def curve(items: list[dict], key: str = "model") -> list[dict]:
    """Precision / coverage of the model answers at each threshold (an answer counts at t if its p >= t)."""
    scored = [(it[key]["p"], it[key]["ok"]) for it in items if it.get(key) and "p" in it[key]]
    out = []
    for t in THRESHOLDS:
        kept = [ok for p, ok in scored if p >= t]
        out.append({"threshold": t, "coverage": round(len(kept) / len(scored), 4) if scored else None,
                    "precision": round(sum(kept) / len(kept), 4) if kept else None, "n": len(kept)})
    return out


def pick_threshold(c: list[dict], target: float) -> float | None:
    """Lowest threshold whose precision >= target (with at least 3 answers kept)."""
    for row in c:
        if row["precision"] is not None and row["n"] >= 3 and row["precision"] >= target:
            return row["threshold"]
    return None


# ---------------- diagnosis ----------------
async def eval_diagnosis(asker: Asker, thr: float) -> dict:
    items = []
    for f in rc.load_failures():
        text = f"{f.log}\n{f.message}" if f.log else f.message
        hits = dg.rule_hits(text, f.exit_code)
        rules = hits[0][0] if len(hits) == 1 else None
        state = dg.build_state(engine=f.engine, model_ref=f.model_ref, exit_code=f.exit_code, message=f.message,
                               hits=hits, log_text=f.log or f.message)
        m = await asker.ask("diagnosis", state, dg.QUESTION, dg.OPTIONS)
        if "choice" in m:
            m["ok"] = m["choice"] == f.label
        pipeline = rules or (m.get("choice") if m.get("p", 0) >= thr else None)
        items.append({"name": f.name, "label": f.label, "rule_hits": [c for c, _ in hits], "rules": rules,
                      "rules_ok": rules == f.label if rules else None, "model": m, "pipeline": pipeline,
                      "pipeline_ok": pipeline == f.label if pipeline else None})
    return summarise("diagnosis", items, thr, target=0.90)


# ---------------- parsers ----------------
async def eval_parser(asker: Asker, thr: float) -> dict:
    repos, labels = rc.load_repos(), rc.load_labels()
    out = {}
    for which, rule_fn, question, options in (
            ("tool", cs.rule_tool, cs.QUESTION_TOOL, cs.tool_options()),
            ("reasoning", cs.rule_reasoning, cs.QUESTION_REASONING, cs.reasoning_options())):
        items = []
        for ref, lab in sorted(labels.items()):
            truth = lab.get(which)
            if truth is None or ref not in repos:
                continue
            info = rc.model_info(repos[ref])
            hit = rule_fn(info)
            m = await asker.ask(f"parser_{which}", cs.state_for(info), question, options)
            if "choice" in m:
                m["ok"] = m["choice"] in truth
            auto = bool(hit and hit.strong) or (not hit and m.get("p", 0) >= thr)
            pipe = hit.value if hit else (m.get("choice") if m.get("p", 0) >= cs.SUGGEST_MIN else None)
            items.append({"name": ref, "label": truth, "rules": hit.value if hit else None,
                          "rules_strong": bool(hit and hit.strong), "rules_ok": (hit.value in truth) if hit else None,
                          "model": m, "pipeline": pipe, "pipeline_ok": (pipe in truth) if pipe else None,
                          "auto_apply": auto, "auto_ok": (pipe in truth) if auto and pipe else None})
        out[which] = summarise(f"parser_{which}", items, thr, target=0.95)
    return out


# ---------------- kind ----------------
async def eval_kind(asker: Asker, thr: float) -> dict:
    repos, labels = rc.load_repos(), rc.load_labels()
    items = []
    for ref, lab in sorted(labels.items()):
        if ref not in repos:
            continue
        a = rc.as_artifact(repos[ref])
        base = a.kind
        hit = kd.rule_kind(a.ref, a.architecture, a.config, list(a.files)) if kd.needs_kind(a) else None
        rules = hit[0] if hit else (base if not kd.needs_kind(a) else None)
        m = await asker.ask("kind", kd.build_state(a), kd.QUESTION, kd.OPTIONS) if a.config is not None else \
            {"error": "no_config"}
        if "choice" in m:
            m["ok"] = m["choice"] in lab["kind"]
        pipeline = rules or (m.get("choice") if m.get("p", 0) >= thr else None)
        items.append({"name": ref, "label": lab["kind"], "base": base, "rules": rules,
                      "rules_ok": (rules in lab["kind"]) if rules else None, "model": m,
                      "pipeline": pipeline, "pipeline_ok": (pipeline in lab["kind"]) if pipeline else None,
                      "model_consulted": rules is None})
    return summarise("kind", items, thr, target=0.90)


# ---------------- workload ----------------
# The state / options of the first version (before the real-traffic calibration), kept to measure the change.
LEGACY_OPTIONS = [cs.Option(k, k.replace("_", " "), d) for k, d in [
    ("single_user_chat", "One person chatting: one or two requests at a time, short prompts, latency matters"),
    ("agent_tool_use", "An agent / tool-calling loop: long repeated prompts (high prefix-cache hits), short outputs"),
    ("batch_throughput", "Batch or many concurrent clients: many requests running and queued, throughput matters"),
    ("long_context_rag", "Long-context / RAG: very long, mostly unique prompts with short answers"),
    ("idle", "Almost no traffic"),
    ("mixed", "A mix of the above with no dominant pattern")]]


def legacy_state(summary: dict, engine: str, model_ref: str, cfg: dict) -> str:
    from dgxmm.decider.common import quoted_block
    flags = {k: cfg.get(k) for k in ("max_model_len", "max_num_seqs", "ctx_size", "parallel", "tool_call_parser",
                                     "reasoning_parser", "enable_prefix_caching") if k in cfg}
    flags["speculative"] = bool(cfg.get("speculative"))
    old_keys = ("minutes", "idle_fraction", "mean_running", "peak_running", "mean_waiting", "peak_waiting",
                "queued_fraction", "prompt_tps", "gen_tps", "prompt_gen_ratio", "prefix_hit_rate", "spec_accept_rate",
                "kv_usage_mean", "kv_usage_peak", "ttft_p50", "e2e_p50")
    nums = "\n".join(f"- {k}: {summary.get(k)}" for k in old_keys)
    return "\n".join([f"Engine: {engine}", quoted_block("MODEL", model_ref, max_chars=300),
                      f"Config: {flags}", "Traffic summary (last 1 h, per-minute averages):", nums])


async def eval_workload(asker: Asker, thr: float, variant: str = "current") -> dict:
    items = []
    for w in rc.load_workloads():
        s = w["summary"]
        hit = wl.rule_label(s, w.get("config", {}))
        args = (s, w.get("engine", "vllm"), w.get("model_ref", "Qwen/Qwen3-0.6B"), w.get("config", {}))
        if variant == "legacy":
            state, options = legacy_state(*args), LEGACY_OPTIONS
        else:
            state, options = wl.build_state(*args), wl.OPTIONS
        m = await asker.ask("workload", state, wl.QUESTION, options)
        if "choice" in m:
            m["ok"] = m["choice"] == w["label"]
        rules = hit[0] if hit else None
        pipeline = rules or (m.get("choice") if m.get("p", 0) >= thr else None)
        items.append({"name": w["name"], "label": w["label"], "rules": rules,
                      "rules_ok": rules == w["label"] if rules else None, "model": m, "pipeline": pipeline,
                      "pipeline_ok": pipeline == w["label"] if pipeline else None})
    return summarise("workload" if variant != "legacy" else "workload_legacy", items, thr, target=0.90)


def summarise(feature: str, items: list[dict], thr: float, target: float) -> dict:
    n = len(items)
    answered = [i for i in items if i.get("pipeline")]
    model_ans = [i for i in items if i["model"].get("choice")]
    ruled = [i for i in items if i.get("rules")]
    c = curve(items)
    s = {"feature": feature, "n": n, "threshold_used": thr,
         "rules_coverage": round(len(ruled) / n, 4) if n else None,
         "rules_precision": round(sum(1 for i in ruled if i["rules_ok"]) / len(ruled), 4) if ruled else None,
         "model_accuracy": round(sum(1 for i in model_ans if i["model"]["ok"]) / len(model_ans), 4) if model_ans else None,
         "pipeline_coverage": round(len(answered) / n, 4) if n else None,
         "pipeline_precision": round(sum(1 for i in answered if i["pipeline_ok"]) / len(answered), 4) if answered else None,
         "curve": c, "suggested_threshold": pick_threshold(c, target), "target_precision": target, "items": items}
    if any("auto_apply" in i for i in items):
        auto = [i for i in items if i.get("auto_apply") and i.get("pipeline")]
        s["auto_apply_coverage"] = round(len(auto) / n, 4) if n else None
        s["auto_apply_precision"] = round(sum(1 for i in auto if i["auto_ok"]) / len(auto), 4) if auto else None
    return s


async def amain(args) -> int:
    asker = Asker(None if args.rules_only else args.base, Path(args.key_file) if args.key_file else None, args.rate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = dict(DEFAULT_THRESHOLDS)
    thr.update(json.loads(args.thresholds) if args.thresholds else {})
    jobs = {"diagnosis": lambda: eval_diagnosis(asker, thr["diagnosis"]),
            "parser": lambda: eval_parser(asker, thr["parser"]),
            "kind": lambda: eval_kind(asker, thr["kind"]), "workload": lambda: eval_workload(asker, thr["workload"]),
            "workload_legacy": lambda: eval_workload(asker, thr["workload"], "legacy")}
    for name in args.features or list(jobs):
        res = await jobs[name]()
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        for r in (res.values() if name == "parser" else [res]):
            brief = {k: v for k, v in r.items() if k not in ("items", "curve")}
            print(json.dumps(brief))
    lat = sorted(asker.latencies)
    if lat:
        print(json.dumps({"calls": len(lat), "p50_ms": round(lat[len(lat) // 2], 1),
                          "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1)}))
    await asker.http.aclose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8101")
    ap.add_argument("--key-file")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--thresholds")
    ap.add_argument("features", nargs="*")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
