"""Feature 10 — LiteLLM routing advice (spec "10. Routing advice"; advisory only).

Daily, and soon after a deployment becomes ready or stops being ready, DGXMM suggests which deployment should serve
each traffic class (``code``, ``chat``, ``tool_use``, ``long_context``, ``vision``, ``batch``):

- rules filter the candidates per class by capability: an LLM (vision: a vision LLM), a tool-call parser for
  tool use, a per-request context of at least LONG_CONTEXT tokens for long context, at least BATCH_SEQS concurrent
  sequences for batch; and a candidate must be ready, sleeping, or regularly used (REGULAR_MINUTES busy minutes in
  the last week). One candidate is the answer by rules;
- with several, the decision model ranks them from their facts (model, size, quantisation, context, sequences,
  parsers, workload label, plausible benchmark readings, memory footprint) — the facts are quoted in ``state``,
  the options are enumerated candidate keys. Below the threshold (or with the decision model off) a deterministic
  ranking decides (``source="rules"``).

Output: ``GET /api/decider/routing-advice`` (admin), a "Routing advice" section on the LiteLLM page, and one
routing insight with a copyable LiteLLM snippet (only deployments that are ready and registered by DGXMM's LiteLLM
sync; a non-ready pick is shown as "start it first") (``router_settings.model_group_alias``: ``dgxmm-<class>`` -> the
deployment's existing LiteLLM model name, its alias or served name). DGXMM never changes LiteLLM for this: the
module makes no LiteLLM call at all; the admin copies the snippet into LiteLLM's config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import timedelta

from sqlalchemy import func, or_, select

from dgxmm.db import Benchmark, MetricRollup, utcnow
from dgxmm.decider.common import gib, quoted_block, safe_decide, safe_record_rules
from dgxmm.decider.insights import PROPOSAL_TTL
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
JOB = "routing_advice"
SCHEDULE = "daily 05:15"
BUDGET_CALLS = 6
TIMEOUT_S = 10.0
SETTLE_S = 15.0  # change notices (state changes, LiteLLM registrations) within this are one refresh
CLASSES: list[tuple[str, str]] = [
    ("code", "Code generation and review"),
    ("chat", "General chat and assistants"),
    ("tool_use", "Agents and tool calling"),
    ("long_context", "Long documents and RAG (32k tokens or more per request)"),
    ("vision", "Images in the prompt"),
    ("batch", "Batch jobs and many concurrent clients"),
]
LONG_CONTEXT = 32768
BATCH_SEQS = 16
BATCH_PARALLEL = 4
REGULAR_MINUTES = 60
USAGE_WINDOW = timedelta(days=7)
ALIAS_PREFIX = "dgxmm-"
LLM_KINDS = ("llm", "vision-llm")
_CODE = re.compile(r"cod(e|er|ing)|starcoder|devstral|codestral", re.I)


def question(cls: str) -> str:
    return f"Which candidate deployment should serve {dict(CLASSES)[cls].lower()} ({cls}) traffic?"


# ---------------- facts ----------------
def per_request_context(v: dict) -> int | None:
    cfg = v.get("config") or {}
    if v["engine"] == "vllm":
        return cfg.get("max_model_len")
    if v["engine"] == "llamacpp" and cfg.get("ctx_size"):
        return int(cfg["ctx_size"]) // max(int(cfg.get("parallel") or 1), 1)
    return None


def concurrency(v: dict) -> int | None:
    cfg = v.get("config") or {}
    return cfg.get("max_num_seqs") if v["engine"] == "vllm" else cfg.get("parallel")


def capable(cls: str, f: dict) -> bool:
    """Capability filter of one traffic class for a candidate's facts."""
    if f["kind"] not in LLM_KINDS:
        return False
    if cls == "vision":
        return f["kind"] == "vision-llm"
    if cls == "tool_use":
        return bool(f.get("tool_call_parser"))
    if cls == "long_context":
        return (f.get("context") or 0) >= LONG_CONTEXT
    if cls == "batch":
        seqs = f.get("concurrency") or 0
        return seqs >= (BATCH_SEQS if f["engine"] == "vllm" else BATCH_PARALLEL)
    return True


def eligible(f: dict) -> bool:
    return f["state"] in ("ready", "sleeping") or (f.get("busy_minutes_7d") or 0) >= REGULAR_MINUTES


def candidates(facts: list[dict], cls: str) -> list[dict]:
    return [f for f in facts if eligible(f) and capable(cls, f)]


def rank_rules(cls: str, cands: list[dict]) -> list[dict]:
    """Deterministic order (the fallback): ready first, then per class code-named models (code), largest context
    (long_context), most sequences (batch); then the larger model (parameters, rounded to billions); then the faster
    plausible benchmark; then id."""
    def key(f):
        ready = 0 if f["state"] == "ready" else 1
        if cls == "code":
            primary = 0 if _CODE.search(f["model_ref"]) else 1
        elif cls == "long_context":
            primary = -(f.get("context") or 0)
        elif cls == "batch":
            primary = -(f.get("concurrency") or 0)
        else:
            primary = 0
        size = params_b(f) or (f.get("weights_bytes") or 0) / 2e9  # unknown quantisation: assume 16-bit
        return (ready, primary, -round(size), -(f.get("bench_tps_per_req") or 0), f["id"])
    return sorted(cands, key=key)


# What matters per class (deterministic text in the state; the measured answers improved with it, see
# docs/decider-evaluation.md).
CRITERIA = {
    "code": "answer quality on programming tasks first (larger, newer and code-trained models), then speed",
    "chat": "answer quality first (larger, newer models), then speed per request",
    "tool_use": "reliable tool calls and reasoning (larger, newer models with a tool-call parser), then speed",
    "long_context": "the longest context per request first, then answer quality",
    "vision": "image understanding quality first, then speed",
    "batch": "total throughput with many concurrent requests (concurrency limit and tokens/s in total)",
}
BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "nvfp4": 0.56, "mxfp4": 0.53, "int4": 0.56, "awq": 0.56,
                   "gptq": 0.56, "q8_0": 1.06, "q6_k": 0.82, "q5_k_m": 0.71, "q4_k_m": 0.6}


def params_b(f: dict) -> float | None:
    """Rough parameter count (billions) from the weights' size and quantisation."""
    bpp = BYTES_PER_PARAM.get(str(f.get("quant") or "").lower())
    w = f.get("weights_bytes")
    return round(w / bpp / 1e9, 1) if bpp and w else None


def build_state(cls: str, cands: list[dict]) -> str:
    """The candidates' facts; deployment and model names are untrusted (quoted)."""
    lines = [f"Traffic class: {cls} ({dict(CLASSES)[cls]}). What matters: {CRITERIA[cls]}.", "Candidates:"]
    for i, f in enumerate(cands, 1):
        bench = ("no benchmark" if f.get("bench_tps_per_req") is None else
                 f"benchmark {f['bench_tps_per_req']:.1f} tokens/s per request at concurrency 1, "
                 f"{f.get('bench_tps_total') or 0:.1f} tokens/s at its best level")
        lines.append(
            f"{i}. key {option_key(f)}: engine {f['engine']}, kind {f['kind']}, weights {gib(f.get('weights_bytes')) if f.get('weights_bytes') else 'unknown'}"
            f" (about {params_b(f) or 'unknown'} billion parameters), quantisation {f.get('quant') or 'unknown'}, context {f.get('context') or 'unknown'} tokens per "
            f"request, up to {f.get('concurrency') or 'unknown'} concurrent requests, tool-call parser "
            f"{'yes' if f.get('tool_call_parser') else 'no'}, reasoning parser "
            f"{'yes' if f.get('reasoning_parser') else 'no'}, workload {f.get('workload') or 'unknown'}, {bench}, "
            f"memory reserved {gib(f.get('reserved_bytes')) if f.get('reserved_bytes') else 'unknown'}, state {f['state']}, busy "
            f"{f.get('busy_minutes_7d') or 0} minutes in the last week.")
        lines.append(quoted_block(f"MODEL OF {option_key(f)}", f["model_ref"], max_chars=200))
    return "\n".join(lines)


def option_key(f: dict) -> str:
    return f"dep{int(f['id'])}"


def options(cands: list[dict]) -> list[Option]:
    return [Option(option_key(f), f"candidate {i}", "") for i, f in enumerate(cands, 1)]


def eval_case(case: dict):
    """Evaluation adapter: a case holds ``class`` and the candidates' ``facts``; label = acceptable option keys."""
    from types import SimpleNamespace
    cls = case["class"]
    cands = candidates(case["facts"], cls)
    rules = option_key(cands[0]) if len(cands) == 1 else None
    return SimpleNamespace(rules=rules, state=build_state(cls, cands), question=question(cls),
                           options=options(cands))


def snippet(rows: list[dict]) -> str:
    """LiteLLM config to copy (router aliases onto the existing model names). Only routable picks (ready and
    registered in LiteLLM by DGXMM's sync): an alias onto a model group LiteLLM does not have would fail requests."""
    picks = [r for r in rows if r.get("model_name") and r.get("routable")]
    if not picks:
        return ""
    lines = ["# Suggested by DGXMM (routing advice); DGXMM changes nothing in LiteLLM. Merge into your existing",
             "# router_settings in LiteLLM's config.yaml (do not add a second router_settings key).",
             "router_settings:", "  model_group_alias:"]
    for r in picks:
        lines.append(f"    {r['alias']}: {json.dumps(r['model_name'])}  # {r['class']} -> deployment "
                     f"{json.dumps(r['deployment'])}")
    return "\n".join(lines)


class RoutingService:
    def __init__(self, ctx, clock=utcnow):
        self.ctx, self.clock = ctx, clock
        self.latest: dict | None = None
        self._memo: dict[str, dict] = {}  # class fingerprint -> its last answer (no new model call if unchanged)

    def refresh(self) -> None:
        """A change notice: refresh the advice once the changes settled (``SETTLE_S``), so a burst of deployments
        changing state and the LiteLLM syncs registering them is one run (one model call budget). Never raises."""
        eng = getattr(self.ctx, "insight_engine", None)
        try:
            if eng is not None:
                eng.trigger(JOB, settle_s=SETTLE_S)
        except Exception:
            log.debug("routing: trigger failed", exc_info=True)

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        """Deployer state hook: a deployment became ready or stopped being ready -> refresh the advice soon."""
        if "ready" in (frm, to) and frm != to:
            self.refresh()

    # ----- facts (sync, database + inventory) -----
    def facts(self) -> list[dict]:
        ctx = self.ctx
        out = []
        views = [v for v in ctx.deployer.list_views() if not v.get("system")]
        since = self.clock() - USAGE_WINDOW
        with ctx.db.session() as s:
            busy = dict(s.execute(select(MetricRollup.deployment_id, func.count()).where(
                MetricRollup.minute >= since, MetricRollup.deployment_id.in_([v["id"] for v in views]),
                or_(func.json_extract(MetricRollup.data, "$.running") > 0,
                    func.json_extract(MetricRollup.data, "$.gen_tps") > 0)).group_by(MetricRollup.deployment_id)).all())
            for v in views:
                art = ctx.inventory.get(v["model_ref"]) if ctx.inventory is not None else None
                rows = list(s.scalars(select(Benchmark).where(
                    Benchmark.model_ref == v["model_ref"], Benchmark.engine == v["engine"], Benchmark.requests > 0,
                    Benchmark.tps_per_req > 0, or_(Benchmark.quality.is_(None), Benchmark.quality != "disputed"))
                    .order_by(Benchmark.id.desc()).limit(50)))
                c1 = [b.tps_per_req for b in rows if b.concurrency == 1]
                cfg = v.get("config") or {}
                wl = None
                workload = getattr(ctx, "workload", None)
                if workload is not None and hasattr(workload, "cached"):
                    wl = (workload.cached(v["id"]) or {}).get("label")
                out.append({"id": v["id"], "name": v["name"], "engine": v["engine"], "model_ref": v["model_ref"],
                            "state": v["state"], "kind": art.kind if art is not None else "llm",
                            "quant": art.quant if art is not None else None,
                            "weights_bytes": art.weights_bytes if art is not None else None,
                            "context": per_request_context(v), "concurrency": concurrency(v),
                            "tool_call_parser": cfg.get("tool_call_parser") if v["engine"] == "vllm" else (
                                "jinja" if "--jinja" in (cfg.get("extra_args") or []) else None),
                            "reasoning_parser": cfg.get("reasoning_parser"),
                            "workload": wl, "bench_tps_per_req": max(c1) if c1 else None,
                            "bench_tps_total": max((b.tps_total for b in rows), default=None),
                            "reserved_bytes": v.get("reserved_bytes"), "busy_minutes_7d": int(busy.get(v["id"], 0)),
                            "model_name": v.get("alias") or v.get("served_name"),
                            "registered": self._registered(v)})
        return out

    def _registered(self, v: dict) -> bool:
        """Registered in LiteLLM as of DGXMM's last successful sync (no LiteLLM call)."""
        sync = getattr(self.ctx, "litellm", None)
        name = v.get("alias") or v.get("served_name")
        if sync is None or not name or v["state"] != "ready":
            return False
        try:
            return sync.is_registered(v["id"], name) is True
        except Exception:
            return False

    # ----- advice -----
    async def advise(self, decider=None, *, use_model: bool = True, to_thread=None) -> dict:
        """The advice for every class; ``use_model=False`` = rules only (no decider call; the API's cold path).
        ``to_thread``: the job's ``run.to_thread`` (the insight engine's own threads); default asyncio's."""
        to_thread = to_thread or _to_thread
        facts = await to_thread(self.facts)
        rows = []
        for cls, label in CLASSES:
            cands = candidates(facts, cls)
            row = {"class": cls, "label": label, "alias": ALIAS_PREFIX + cls.replace("_", "-"),
                   "candidates": [{"deployment_id": f["id"], "deployment": f["name"], "model_name": f["model_name"]}
                                  for f in cands],
                   "deployment_id": None, "deployment": None, "model_name": None, "model_ref": None,
                   "source": "none", "confidence": None, "reason": "no deployment has this capability",
                   "routable": False, "note": None}
            if not cands:
                rows.append(row)
                continue
            pick, source, conf, reason = None, "rules", None, None
            if len(cands) == 1:
                pick, reason = cands[0], "the only deployment with this capability"
            elif use_model and decider is not None:
                fp = _fingerprint(cls, cands)
                memo = self._memo.get(fp)
                if memo is not None:
                    pick = next((f for f in cands if f["id"] == memo["id"]), None)
                    source, conf, reason = memo["source"], memo["confidence"], memo["reason"]
                else:
                    d = await safe_decide(decider, feature="routing", subject=f"routing:{cls}",
                                          state=build_state(cls, cands), question=question(cls),
                                          options=options(cands), timeout_s=TIMEOUT_S)
                    if d.source == "model" and d.choice:
                        pick = next(f for f in cands if option_key(f) == d.choice)
                        source, conf, reason = "model", d.confidence, "ranked first by the decision model"
                        self._memo[fp] = {"id": pick["id"], "source": source, "confidence": conf, "reason": reason}
            if pick is None:
                pick = rank_rules(cls, cands)[0]
                source, reason = "rules", _rules_reason(cls, pick, len(cands))
            if source == "rules" and decider is not None and use_model:
                await to_thread(safe_record_rules, decider, feature="routing", subject=f"routing:{cls}",
                                choice=option_key(pick), reason=reason)
            routable = pick["state"] == "ready" and pick.get("registered") is True
            note = None
            if pick["state"] != "ready":
                note = f"would suit; start {pick['name']} first (LiteLLM serves only ready deployments)"
            elif not routable:
                note = "not registered in LiteLLM yet (see the LiteLLM page)"
            row.update(deployment_id=pick["id"], deployment=pick["name"], model_name=pick["model_name"],
                       model_ref=pick["model_ref"], source=source, confidence=conf, reason=reason,
                       routable=routable, note=note)
            rows.append(row)
        advice = {"classes": rows, "snippet": snippet(rows), "computed_at": self.clock().isoformat(),
                  "source": "model" if any(r["source"] == "model" for r in rows) else "rules"}
        return advice

    async def run(self, run) -> str:
        advice = await self.advise(run.decider, to_thread=run.to_thread)
        self.latest = advice
        await run.to_thread(self._publish, advice)
        picked = sum(1 for r in advice["classes"] if r["model_name"])
        return f"{picked} of {len(CLASSES)} classes have a suggestion"

    def _publish(self, advice: dict) -> None:
        rows = [r for r in advice["classes"] if r["model_name"]]
        if not rows:
            self.ctx.insights.resolve("routing:advice")
            return
        lines = ["Suggested LiteLLM aliases per traffic class (advisory: DGXMM does not change LiteLLM):", ""]
        for r in rows:
            how = (f"decision model, {r['confidence']:.0%}" if r["source"] == "model" else "rules")
            lines.append(f"- {r['alias']} ({r['label'].lower()}): {r['model_name']} on deployment {r['deployment']} "
                         f"({how})" + (f"; {r['note']}" if r.get("note") else ""))
        missing = [r["alias"] for r in advice["classes"] if not r["model_name"]]
        if missing:
            lines += ["", "No deployment fits: " + ", ".join(missing) + "."]
        lines += ["", "Copy the snippet into LiteLLM's config to route these aliases (it holds only deployments "
                      "LiteLLM serves now); the LiteLLM page lists the same advice."]
        conf = [r["confidence"] for r in rows if r["confidence"] is not None]
        action = {"type": "copy", "text": advice["snippet"][:4096], "label": "Copy LiteLLM snippet"} \
            if advice["snippet"] else {"type": "open", "path": "/litellm", "label": "Open LiteLLM"}
        self.ctx.insights.upsert("routing", "routing", "litellm", "info",
                                 f"Routing advice for {len(rows)} traffic class{'es' if len(rows) != 1 else ''}",
                                 "\n".join(lines)[:4000], dedupe_key="routing:advice",
                                 confidence=min(conf) if conf else None,
                                 source="model" if any(r["source"] == "model" for r in rows) else "rules",
                                 action=action,
                                 expires_at=self.clock() + PROPOSAL_TTL)

    async def current(self) -> dict:
        """For the API: the latest advice, else a rules-only one now (and the job is asked to run)."""
        if self.latest is not None:
            return self.latest
        eng = getattr(self.ctx, "insight_engine", None)
        if eng is not None:
            try:
                eng.trigger(JOB)
            except Exception:
                pass
        return await self.advise(None, use_model=False)


def _rules_reason(cls: str, f: dict, n: int) -> str:
    why = {"code": "a code model" if _CODE.search(f["model_ref"]) else "the largest model",
           "long_context": f"the longest context ({f.get('context') or 0:,} tokens)",
           "batch": f"the most concurrent requests ({f.get('concurrency') or 0})"}.get(cls, "the largest model")
    return f"{why} among {n} candidates" + ("" if f["state"] == "ready" else f" (currently {f['state']})")


def _fingerprint(cls: str, cands: list[dict]) -> str:
    keys = ("id", "engine", "model_ref", "kind", "quant", "context", "concurrency", "tool_call_parser",
            "reasoning_parser", "workload", "bench_tps_per_req", "state")
    raw = json.dumps([cls, [[f.get(k) for k in keys] for f in cands]], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


async def _to_thread(fn, *a, **kw):
    import asyncio
    return await asyncio.to_thread(fn, *a, **kw)
