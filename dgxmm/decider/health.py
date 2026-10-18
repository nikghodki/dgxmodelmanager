"""Feature 1 — health triage of running deployments (spec "1. Health triage").

Every CHECK_INTERVAL_S (insight-engine job ``health_check``) each ready, non-system deployment is checked over the
last WINDOW_S: the live 5-second metric samples (KV-cache usage p50/p95, preemptions, running / waiting, generation
throughput per running sequence against the deployment's own 24 h baseline at similar load) and the access / error
lines of its container log in the same window (counts only; a redacted, capped excerpt goes to the model).

Rules first (high precision): sustained KV usage >= 95 % with preemptions -> ``kv_pressure``; preemptions per
minute above a limit -> ``preemption_storm``; throughput below half the baseline at similar load ->
``throughput_drop``; 5xx / server error lines rising -> ``error_creep`` (never a client's 4xx: vLLM's per-request
error echoes are counted by the request's access line); a high share of 4xx answers (prompts over the
context limit, malformed tool definitions, unknown models) -> ``client_misuse``; every metric well inside its limits
-> ``healthy``. Anything in between is ambiguous and only then is the decision model asked (rules-less window:
``healthy`` or one of the labels); below the threshold nothing changes. Rules run with the decision model disabled.

Each finding is a warning insight (``health:<dep>:<label>``, templated title/body) with, where one applies, a
remedy as a ``reconfigure`` action (fp8 KV cache, fewer sequences, a larger KV pool only when the memory calculator
says it fits, prefix caching). A label clears once the deployment has been checked conclusively without it for
CLEAR_AFTER (15 min): the insight is resolved with ``HEALTH_CLEAR_GRACE`` (it stays visible an hour). A deployment
that is no longer ready has its health insights resolved the same way.

Decisions: model consultations are logged by the Decider; rules decisions are logged when a deployment's findings
change (its first check, a new label, a return to healthy) — not every five minutes of "still healthy".
"""
from __future__ import annotations

import logging
import re
import statistics
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

from dgxmm.db import as_aware, utcnow
from dgxmm.decider.common import (busy_minute, last_span, percentile, plural, quoted_block, safe_decide,
                                  safe_record_rules)
from dgxmm.decider.insights import HEALTH_CLEAR_GRACE, deployment_subject
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
JOB = "health_check"
CHECK_INTERVAL_S = 300
WINDOW_S = 300
MIN_SAMPLES = 12            # one minute of 5-second samples
BUDGET_CALLS = 8            # model questions per check cycle (ambiguous deployments only)
TIMEOUT_S = 10.0            # background lane: queue behind interactive questions
CLEAR_AFTER = timedelta(minutes=15)          # conclusive checks without the finding: resolve
INCONCLUSIVE_CLEAR_AFTER = timedelta(hours=1)  # only inconclusive ones (model off / unsure): resolve after this
LOG_TAIL = 4000

# Limits (calibrated on the real windows in tests/fixtures/decider_corpus/health; docs/decider-evaluation.md)
KV_HIGH = 0.95              # "full" KV cache
KV_SUSTAINED = 0.25         # share of busy samples at or above KV_HIGH that counts as sustained
KV_OK = 0.80                # p95 below this is well inside
PREEMPT_STORM_PER_MIN = 10.0
THROUGHPUT_DROP = 0.5       # per-sequence generation speed below this share of the baseline
THROUGHPUT_OK = 0.75        # at or above this share: well inside
SIMILAR_LOAD = (0.5, 1.5)   # baseline minutes whose running count is within this factor of the window's
MIN_BASELINE_MINUTES = 10
MIN_BUSY_FRACTION = 0.5     # throughput is judged only on a mostly busy window
ERRORS_OK = 1               # a stray server error (5xx answer or error line) in five minutes is not a trend
ERRORS_MIN = 3              # 5xx answers + server error lines in the window
ERRORS_RISE = 2.0           # ... and at least this many times the previous window's
MISUSE_MIN = 5              # 4xx answers in the window
MISUSE_SHARE = 0.10         # ... and at least this share of the requests
MISUSE_OK_SHARE = 0.02

LABELS: list[tuple[str, str]] = [
    ("healthy", "Serving normally: the KV cache has room, nothing is preempted, throughput is in line with this "
                "deployment's usual speed at this load, and there are no server errors or client errors to speak of"),
    ("kv_pressure", "The KV cache is full or nearly full (95 % or more) for much of the time and requests are "
                    "being preempted: more concurrent context than the KV pool holds"),
    ("preemption_storm", "Requests are preempted (evicted and recomputed) many times per minute, wasting compute"),
    ("throughput_drop", "Generation throughput per running request is far below this deployment's own 24 h "
                        "baseline at a similar load, without KV pressure explaining it (contention, throttling)"),
    ("error_creep", "Server errors (HTTP 5xx answers, ERROR lines, tracebacks) are appearing and rising"),
    ("client_misuse", "Clients send bad requests: many HTTP 4xx answers, e.g. prompts over the context limit, "
                      "malformed tool definitions or unknown model names"),
]
OPTIONS = [Option(k, k.replace("_", " "), d) for k, d in LABELS]
PROBLEMS = [k for k, _ in LABELS if k != "healthy"]
QUESTION = "Which best describes the health of this deployment over the last five minutes?"

# vLLM / uvicorn access lines and llama.cpp request lines
_ACCESS = re.compile(r'"(?:GET|POST|PUT|DELETE|PATCH) (\S{1,512}) HTTP/[\d.]{1,4}" (\d{3})')
_LLAMA = re.compile(r"request: (?:GET|POST|PUT|DELETE|PATCH) (\S{1,512}) \S{1,64} (\d{3})\b")
_ERROR = re.compile(r"\b(?:ERROR|CRITICAL|FATAL)\b|Traceback \(most recent call last\)|\bRuntimeError\b")
# error lines that only report a client's mistake (vLLM logs a 404 for an unknown model as ERROR)
_CLIENT_ERROR = re.compile(r"code=4\d\d|BadRequestError|NotFoundError|\b4\d\d (?:Bad Request|Not Found)")
# vLLM's request exception handlers (on with --log-error-stack, which VLLM_SERVER_DEV_MODE, set for sleep mode, turns
# on) echo every rejected request as ERROR, logger.exception ones with a traceback whose lines all carry the same
# record prefix. The request's access line already counts it (4xx or 5xx), so the echo is not an error line.
_RECORD_TAG = re.compile(r"\b(?:ERROR|WARNING|CRITICAL) \d\d-\d\d \d\d:\d\d:\d\d (\[[\w.-]{1,80}:\d{1,6}\])")
_REQUEST_ECHO = re.compile(r"(?:Exception|Error) caught\. Request id: ")
_IGNORED_PATHS = ("/metrics", "/health", "/v1/models", "/ping", "/version", "/load")


# ---------------- window summary ----------------
def _vals(samples, key):
    return [s[key] for s in samples if isinstance(s.get(key), (int, float))]


def _per_seq(s: dict) -> float | None:
    """Generated tokens per second per running sequence (the speed of one request at this load)."""
    run, gen = s.get("running"), s.get("gen_tps")
    if not isinstance(run, (int, float)) or not isinstance(gen, (int, float)) or run < 0.5:
        return None
    return gen / run


def summarize(samples: list[dict], baseline_rows: list[dict] | None = None) -> dict:
    """The window's metrics: ``samples`` are live 5-second samples (dgxmm/metrics/store.py), ``baseline_rows`` the
    deployment's minute rollups of the previous 24 h (excluding the window)."""
    n = len(samples)
    ts = _vals(samples, "ts")
    minutes = round((max(ts) - min(ts) + 5) / 60, 2) if ts else 0.0
    busy = [s for s in samples if busy_minute(s)]
    kv = _vals(samples, "kv_usage")
    kv_busy = _vals(busy, "kv_usage")
    pre = sum(_vals(samples, "preemptions"))
    per_seq = [v for v in (_per_seq(s) for s in busy) if v is not None]
    running = _vals(busy, "running")
    out = {
        "samples": n, "minutes": minutes,
        "busy_fraction": round(len(busy) / n, 4) if n else None,
        "kv_p50": percentile(kv, 0.5), "kv_p95": percentile(kv, 0.95),
        "kv_high_fraction": round(sum(1 for v in kv_busy if v >= KV_HIGH) / len(kv_busy), 4) if kv_busy else None,
        "preemptions": round(pre, 1),
        "preemptions_per_min": round(pre / minutes, 2) if minutes else None,
        "running_mean": round(statistics.fmean(running), 3) if running else None,
        "running_peak": max(_vals(samples, "running"), default=None),
        "waiting_mean": round(statistics.fmean(_vals(busy, "waiting")), 3) if _vals(busy, "waiting") else None,
        "waiting_peak": max(_vals(samples, "waiting"), default=None),
        "gen_tps": round(statistics.fmean(_vals(busy, "gen_tps")), 2) if _vals(busy, "gen_tps") else None,
        "prompt_tps": round(statistics.fmean(_vals(busy, "prompt_tps")), 2) if _vals(busy, "prompt_tps") else None,
        "per_seq_tps": round(statistics.median(per_seq), 3) if per_seq else None,
    }
    out.update(baseline(baseline_rows or [], out["running_mean"]))
    return out


def baseline(rows: list[dict], running: float | None) -> dict:
    """The deployment's usual per-sequence generation speed at a similar load (median over the busy minute rollups
    whose mean running count is within SIMILAR_LOAD of the window's); None with fewer than MIN_BASELINE_MINUTES."""
    if not running or running < 0.5:
        return {"baseline_per_seq_tps": None, "baseline_minutes": 0, "throughput_ratio": None}
    lo, hi = running * SIMILAR_LOAD[0], running * SIMILAR_LOAD[1]
    vals = [v for r in rows if isinstance(r.get("running"), (int, float)) and lo <= r["running"] <= hi
            and (v := _per_seq(r)) is not None and v > 0]
    if len(vals) < MIN_BASELINE_MINUTES:
        return {"baseline_per_seq_tps": None, "baseline_minutes": len(vals), "throughput_ratio": None}
    return {"baseline_per_seq_tps": round(statistics.median(vals), 3), "baseline_minutes": len(vals),
            "throughput_ratio": None}


def with_ratio(s: dict) -> dict:
    b, v = s.get("baseline_per_seq_tps"), s.get("per_seq_tps")
    s["throughput_ratio"] = round(v / b, 3) if b and v is not None else None
    return s


# ---------------- logs ----------------
def scan_logs(text: str | None, *, excerpt_lines: int = 40) -> dict:
    """Counts of the window's log: client requests (access lines for /v1/... other than the model list), 4xx and
    5xx answers, server error lines; plus an excerpt of the non-2xx and error lines for the model (redacted and
    quoted by ``build_state``)."""
    requests = c4 = c5 = errors = 0
    interesting: list[str] = []
    echo_tags: set[str] = set()  # record tags ([file.py:line]) of request-exception echoes: they log nothing else
    for line in (text or "").splitlines()[-LOG_TAIL:]:
        m = _ACCESS.search(line) or _LLAMA.search(line)
        if m:
            path, status = m.group(1), int(m.group(2))
            if path.split("?")[0].rstrip("/") in _IGNORED_PATHS:
                continue
            requests += 1
            if 400 <= status < 500:
                c4 += 1
                interesting.append(line)
            elif status >= 500:
                c5 += 1
                interesting.append(line)
            continue
        tag = _RECORD_TAG.search(line)
        if tag and _REQUEST_ECHO.search(line):
            echo_tags.add(tag.group(1))
            continue
        if tag and tag.group(1) in echo_tags:
            continue  # the echo's traceback (other records may be interleaved)
        if _ERROR.search(line) and not _CLIENT_ERROR.search(line):
            errors += 1
            interesting.append(line)
    return {"requests": requests, "status_4xx": c4, "status_5xx": c5, "error_lines": errors,
            "server_errors": c5 + errors, "excerpt": "\n".join(l[:300] for l in interesting[-excerpt_lines:])}


# ---------------- rules ----------------
def rule_labels(s: dict, logs: dict, prev_server_errors: int | None = None) -> list[str]:
    """Every rule that fires, most specific first: kv_pressure, preemption_storm, error_creep, client_misuse,
    throughput_drop. Empty when no rule fires (healthy or ambiguous: see ``rules_healthy``)."""
    out = []
    pre = s.get("preemptions") or 0
    if (s.get("kv_high_fraction") or 0) >= KV_SUSTAINED and pre > 0:
        out.append("kv_pressure")
    if (s.get("preemptions_per_min") or 0) >= PREEMPT_STORM_PER_MIN:
        out.append("preemption_storm")
    err = logs.get("server_errors", 0)
    # "rising" needs a previous window to compare with (none on the first check after a start or a restart)
    if prev_server_errors is not None and err >= ERRORS_MIN and err >= ERRORS_RISE * max(prev_server_errors, 1) \
            and err > prev_server_errors:
        out.append("error_creep")
    req, c4 = logs.get("requests", 0), logs.get("status_4xx", 0)
    if c4 >= MISUSE_MIN and req and c4 / req >= MISUSE_SHARE:
        out.append("client_misuse")
    r = s.get("throughput_ratio")
    if r is not None and r < THROUGHPUT_DROP and (s.get("busy_fraction") or 0) >= MIN_BUSY_FRACTION \
            and "kv_pressure" not in out and "preemption_storm" not in out:
        out.append("throughput_drop")
    return out


def rules_healthy(s: dict, logs: dict) -> bool:
    """Every metric well inside its limits (an idle deployment included)."""
    if s.get("kv_p95") is not None and s["kv_p95"] >= KV_OK:
        return False
    if (s.get("preemptions") or 0) > 0 or logs.get("server_errors", 0) > ERRORS_OK:
        return False
    req, c4 = logs.get("requests", 0), logs.get("status_4xx", 0)
    if c4 > 1 and req and c4 / req >= MISUSE_OK_SHARE:
        return False
    r = s.get("throughput_ratio")
    if r is not None and r < THROUGHPUT_OK and (s.get("busy_fraction") or 0) >= MIN_BUSY_FRACTION:
        return False
    return True


def rules_answer(s: dict, logs: dict, prev_server_errors: int | None = None) -> tuple[list[str], bool]:
    """(labels that fired, conclusive): ``[]``/True = healthy by rules, ``[]``/False = ambiguous (ask the model)."""
    labels = rule_labels(s, logs, prev_server_errors)
    if labels:
        return labels, True
    return [], rules_healthy(s, logs)


# ---------------- the question ----------------
def _n(v, digits: int = 1, pct: bool = False) -> str:
    if v is None:
        return "unknown"
    return f"{v * 100:.0f} %" if pct else f"{v:,.{digits}f}"


def build_state(s: dict, logs: dict, engine: str, model_ref: str, cfg: dict, prev_server_errors=None) -> str:
    flags = {k: cfg.get(k) for k in ("max_model_len", "max_num_seqs", "kv_tokens", "kv_cache_dtype",
                                     "enable_prefix_caching", "ctx_size", "parallel", "cache_type_k") if k in cfg}
    lines = [
        f"- Window: {_n(s.get('minutes'), 1)} minutes, requests in flight in {_n(s.get('busy_fraction'), pct=True)} "
        f"of the samples.",
        f"- Load: {_n(s.get('running_mean'), 2)} requests running on average while busy (peak "
        f"{_n(s.get('running_peak'), 0)}), {_n(s.get('waiting_mean'), 1)} waiting (peak {_n(s.get('waiting_peak'), 0)}).",
        f"- KV cache usage: p50 {_n(s.get('kv_p50'), pct=True)}, p95 {_n(s.get('kv_p95'), pct=True)}; at or above "
        f"95 % in {_n(s.get('kv_high_fraction'), pct=True)} of the busy samples.",
        f"- Preemptions: {_n(s.get('preemptions'), 0)} in the window ({_n(s.get('preemptions_per_min'), 1)} per "
        f"minute).",
        f"- Generation: {_n(s.get('gen_tps'))} tokens/s in total, {_n(s.get('per_seq_tps'), 2)} tokens/s per running "
        f"request; the 24 h baseline at a similar load is {_n(s.get('baseline_per_seq_tps'), 2)} tokens/s per request "
        f"(ratio {_n(s.get('throughput_ratio'), 2)}, from {s.get('baseline_minutes', 0)} minutes).",
        f"- Requests answered in the window: {logs.get('requests', 0)}; HTTP 4xx: {logs.get('status_4xx', 0)}; "
        f"HTTP 5xx: {logs.get('status_5xx', 0)}; server error lines: {logs.get('error_lines', 0)} (previous window: "
        f"{'unknown' if prev_server_errors is None else prev_server_errors} server errors).",
    ]
    parts = [f"Engine: {engine}", quoted_block("MODEL", model_ref, max_chars=300), f"Config: {flags}",
             "Metrics over the last five minutes (from the engine's metrics and its log):", *lines]
    if logs.get("excerpt"):
        parts.append(quoted_block("LOG EXCERPT (non-2xx and error lines)", logs["excerpt"], max_chars=4000,
                                  max_lines=40))
    return "\n".join(parts)


CORPUS_OK_LINE = '(APIServer pid=1) INFO:     192.0.2.10:50000 - "POST /v1/chat/completions HTTP/1.1" 200 OK'


def eval_case(case: dict):
    """Evaluation adapter (tools/eval_feature.py): a recorded window -> DGXMM's rules answer and its question.
    The rules answer is the first label that fired, ``healthy`` when every metric is well inside, else None."""
    s = with_ratio(summarize(case["samples"], case.get("baseline") or []))
    text = case.get("log", "")
    if case.get("ok_requests"):  # recorded windows keep only the non-2xx / error lines plus the count of the rest
        text += "\n" + "\n".join([CORPUS_OK_LINE] * int(case["ok_requests"]))
    logs = scan_logs(text)
    prev = case.get("prev_server_errors")
    labels, conclusive = rules_answer(s, logs, prev)
    rules = labels[0] if labels else ("healthy" if conclusive else None)
    cfg = case.get("config") or {}
    return SimpleNamespace(rules=rules, state=build_state(s, logs, case.get("engine", "vllm"),
                                                          case.get("model_ref", ""), cfg, prev),
                           question=QUESTION, options=OPTIONS)


# ---------------- remedies and texts ----------------
GIB = 1024**3


def remedies(label: str, engine: str, cfg: dict, s: dict) -> list[tuple[dict, str, bool]]:
    """(patch, text, needs_fit_check) candidates for a finding, in order of preference. ``needs_fit_check``: the
    patch grows the reservation, so it is offered only when the memory calculator says it fits."""
    out: list[tuple[dict, str, bool]] = []
    if engine == "vllm":
        seqs = int(cfg.get("max_num_seqs") or 8)
        fewer = max(1, seqs // 2)
        if label in ("kv_pressure", "preemption_storm"):
            if cfg.get("kv_cache_dtype", "auto") == "auto":
                out.append(({"kv_cache_dtype": "fp8"}, "use an fp8 KV cache (about twice the tokens in the same "
                                                       "memory)", False))
            pool = cfg.get("kv_tokens")
            if isinstance(pool, int) and pool > 0:
                out.append(({"kv_tokens": int(pool * 1.5)}, f"grow the KV pool from {pool:,} to "
                                                            f"{int(pool * 1.5):,} tokens", True))
            if label == "kv_pressure" and cfg.get("enable_prefix_caching") is False:
                out.append(({"enable_prefix_caching": True}, "enable prefix caching (shared prompt prefixes use "
                                                             "the KV cache once)", False))
            if seqs > 1:
                out.append(({"max_num_seqs": fewer}, f"lower max_num_seqs from {seqs} to {fewer} (fewer requests "
                                                     "compete for the cache; the rest queue)", False))
    elif engine == "llamacpp" and label in ("kv_pressure", "preemption_storm"):
        if cfg.get("cache_type_k", "f16") == "f16":
            out.append(({"cache_type_k": "q8_0"}, "quantise the K cache to q8_0", False))
        par = int(cfg.get("parallel") or 4)
        if par > 1:
            out.append(({"parallel": max(1, par // 2)}, f"lower parallel from {par} to {max(1, par // 2)}", False))
    return out


TITLES = {
    "kv_pressure": "KV cache pressure on {name}",
    "preemption_storm": "Frequent preemptions on {name}",
    "throughput_drop": "Throughput drop on {name}",
    "error_creep": "Rising server errors on {name}",
    "client_misuse": "Clients send bad requests to {name}",
}


def explain(label: str, s: dict, logs: dict) -> str:
    """The deterministic explanation of a finding (numbers only; never model or log text)."""
    window = last_span(s.get("minutes"), "minute")
    if label == "kv_pressure":
        pre = s.get("preemptions")
        pre = ("an unknown number of requests were" if pre is None else
               f"{plural(int(pre), 'request')} {'was' if int(pre) == 1 else 'were'}")
        return (f"KV-cache usage was at or above 95 % in {_n(s.get('kv_high_fraction'), pct=True)} of the busy time "
                f"over {window} (p95 {_n(s.get('kv_p95'), pct=True)}) and {pre} preempted.")
    if label == "preemption_storm":
        return (f"{_n(s.get('preemptions_per_min'), 1)} preemptions per minute over {window}: preempted requests "
                "are evicted and recomputed, which wastes compute and adds latency.")
    if label == "throughput_drop":
        return (f"Each running request generated {_n(s.get('per_seq_tps'), 1)} tokens/s, "
                f"{_n(s.get('throughput_ratio'), pct=True)} of this deployment's usual "
                f"{_n(s.get('baseline_per_seq_tps'), 1)} tokens/s at a similar load (24 h baseline, "
                f"{plural(s.get('baseline_minutes') or 0, 'minute')}). Other work on the GPU or host is the usual cause.")
    if label == "error_creep":
        return (f"{plural(logs.get('status_5xx', 0), 'HTTP 5xx answer')} and "
                f"{plural(logs.get('error_lines', 0), 'server error line')} in the last five minutes, more than before. The deployment's log has the details.")
    if label == "client_misuse":
        req, c4 = logs.get("requests", 0), logs.get("status_4xx", 0)
        return (f"{c4} of {plural(req, 'request')} ({_n(c4 / req if req else None, pct=True)}) "
                f"{'was' if c4 == 1 else 'were'} rejected with HTTP 4xx in "
                "the last five minutes: typically prompts longer than the context limit, malformed tool "
                "definitions or a wrong model name. Check the clients; the server is fine.")
    return "This deployment is serving normally."


# ---------------- the job ----------------
def dedupe_key(dep_id: int, label: str) -> str:
    return f"health:{int(dep_id)}:{label}"


class HealthService:
    def __init__(self, ctx, clock=utcnow, now_ts=time.time):
        self.ctx, self.clock, self.now_ts = ctx, clock, now_ts
        self._prev_errors: dict[int, int] = {}        # dep -> server errors in the previous window
        self._last_bad: dict[tuple[int, str], datetime] = {}
        self._last_labels: dict[int, tuple[str, ...]] = {}

    def forget(self, dep_id: int) -> None:
        self._prev_errors.pop(dep_id, None)
        self._last_labels.pop(dep_id, None)
        for k in [k for k in self._last_bad if k[0] == dep_id]:
            self._last_bad.pop(k, None)

    # ----- data -----
    def _load(self, v: dict) -> tuple[dict, dict] | None:
        store = getattr(self.ctx, "metrics", None)
        if store is None:
            return None
        now = self.now_ts()
        start = now - WINDOW_S
        samples = [x for x in store.series(v["id"]) if (x.get("ts") or 0) >= start]
        if len(samples) < MIN_SAMPLES:
            return None
        rows = [r for r in store.history(v["id"], 24) if (r.get("ts") or 0) < start]
        s = with_ratio(summarize(samples, rows))
        text = ""
        docker = getattr(self.ctx.deployer, "docker", None)
        if docker is not None and v.get("container_id"):
            try:
                text = docker.logs(v["container_id"], LOG_TAIL, since=start)
            except TypeError:  # a controller without ``since``: the tail, as a best effort
                text = docker.logs(v["container_id"], LOG_TAIL)
            except Exception:
                log.debug("health: could not read the log of %s", v["name"], exc_info=True)
        return s, scan_logs(text)

    # ----- one deployment -----
    async def check(self, run, v: dict) -> dict | None:
        """Check one ready deployment; returns {labels, source, conclusive} or None (not enough data)."""
        dep_id = v["id"]
        loaded = await run.to_thread(self._load, v)
        if loaded is None:
            return None
        s, logs = loaded
        prev = self._prev_errors.get(dep_id)
        self._prev_errors[dep_id] = logs["server_errors"]
        labels, conclusive = rules_answer(s, logs, prev)
        subject = deployment_subject(dep_id)
        source, confidence, decision_id = "rules", None, None
        if not labels and not conclusive:
            d = await safe_decide(run.decider, feature="health", subject=subject,
                                  state=build_state(s, logs, v["engine"], v["model_ref"], v["config"] or {}, prev),
                                  question=QUESTION, options=OPTIONS, timeout_s=TIMEOUT_S)
            if d.source == "model" and d.choice:
                conclusive, source, confidence, decision_id = True, "model", d.confidence, d.decision_id
                labels = [] if d.choice == "healthy" else [d.choice]
        elif tuple(labels) != self._last_labels.get(dep_id):
            choice = labels[0] if labels else "healthy"
            reason = ", ".join(labels) if labels else "every metric well inside its limits"
            decision_id = (await run.to_thread(lambda: safe_record_rules(
                run.decider, feature="health", subject=subject, choice=choice, reason=reason,
                options=OPTIONS))).decision_id
        if conclusive:
            self._last_labels[dep_id] = tuple(labels)
        # clearing runs on every check: a finding no rule (or model) reports any more clears after CLEAR_AFTER of
        # conclusive checks, or after INCONCLUSIVE_CLEAR_AFTER when the checks are inconclusive (decider off/unsure)
        await run.to_thread(self._apply, v, labels, conclusive)
        if labels:
            await self._offer_remedies(run, v, s, logs, labels, source, confidence, decision_id)
        return {"labels": labels, "source": source, "conclusive": conclusive, "summary": s,
                "logs": {k: logs[k] for k in ("requests", "status_4xx", "status_5xx", "error_lines")}}

    def _apply(self, v, labels, conclusive: bool) -> None:
        """Clear labels not seen for CLEAR_AFTER (conclusive check) or INCONCLUSIVE_CLEAR_AFTER (inconclusive);
        raising happens in _offer_remedies, after the fit check."""
        dep_id, now = v["id"], self.clock()
        for label in labels:
            self._last_bad[(dep_id, label)] = now
        open_rows = self.ctx.insights.list(status=["open", "snoozed", "dismissed"],
                                           subject=deployment_subject(dep_id), kind="health")
        for row in open_rows:
            label = row["dedupe_key"].rsplit(":", 1)[-1]
            if label in labels:
                continue
            last = self._last_bad.get((dep_id, label))
            if last is None:  # e.g. after a restart: the insight's last refresh
                last = datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else now
                self._last_bad[(dep_id, label)] = last
            if now - as_aware(last) >= (CLEAR_AFTER if conclusive else INCONCLUSIVE_CLEAR_AFTER):
                self.ctx.insights.resolve(row["dedupe_key"], grace=HEALTH_CLEAR_GRACE)

    async def _offer_remedies(self, run, v, s, logs, labels, source, confidence, decision_id) -> None:
        for label in labels:
            fixes = []
            for patch, text, needs_fit in remedies(label, v["engine"], v["config"] or {}, s):
                if needs_fit and not await self._fits(v, patch):
                    continue
                fixes.append((patch, text))
            action = None
            if fixes:
                action = {"type": "reconfigure", "deployment_id": int(v["id"]), "patch": fixes[0][0],
                          "label": "Reconfigure…"}
            elif label in ("error_creep", "throughput_drop", "client_misuse"):
                action = {"type": "open", "path": f"/deployments/{int(v['id'])}", "label": "Open deployment"}
            body = explain(label, s, logs)
            if fixes:
                body += " Suggested: " + "; ".join(t for _, t in fixes) + "."
            if source == "model":
                body += f" (Classified by the decision model, confidence {confidence:.0%}.)"
            title = TITLES[label].format(name=v["name"])[:256]
            await run.to_thread(self.ctx.insights.upsert, "health", "health", deployment_subject(v["id"]), "warning",
                               title, body[:4000], dedupe_key=dedupe_key(v["id"], label), confidence=confidence,
                               source=source, decision_id=decision_id, action=action)

    async def _fits(self, v: dict, patch: dict) -> bool:
        """The memory calculator (deployer.preview, replacing this deployment) says the patched config fits."""
        try:
            out = await self.ctx.deployer.preview(v["engine"], v["model_ref"], {**(v["config"] or {}), **patch},
                                                  replacing=v["id"], consult=lambda: False)
            return bool((out.get("fit") or {}).get("fits"))
        except Exception:
            log.debug("health: fit check of %s failed", patch, exc_info=True)
            return False

    # ----- every deployment -----
    async def run(self, run) -> str:
        views = [v for v in self.ctx.deployer.list_views() if not v.get("system")]
        flagged_subjects = await run.to_thread(self._subjects_with_findings)
        checked = flagged = 0
        for v in views:
            if v["state"] != "ready":
                self._prev_errors.pop(v["id"], None)
                self._last_labels.pop(v["id"], None)
                if deployment_subject(v["id"]) in flagged_subjects:  # only those with live health insights
                    await run.to_thread(self._clear_not_ready, v["id"])
                continue
            try:
                res = await self.check(run, v)
            except Exception:
                log.exception("health check of deployment %s failed", v["id"])
                continue
            if res is not None:
                checked += 1
                flagged += bool(res["labels"])
        return f"{plural(checked, 'deployment')} checked, {flagged} with findings"

    def _subjects_with_findings(self) -> set[str]:
        """Subjects with a live health insight (one query per run)."""
        return {r["subject"] for r in self.ctx.insights.list(status=["open", "snoozed", "dismissed"], kind="health",
                                                              limit=1000)}

    def _clear_not_ready(self, dep_id: int) -> None:
        for row in self.ctx.insights.list(status=["open", "snoozed", "dismissed"],
                                          subject=deployment_subject(dep_id), kind="health"):
            self.ctx.insights.resolve(row["dedupe_key"], grace=HEALTH_CLEAR_GRACE)
