"""Feature 1 — failure diagnosis: why did a deployment fail, and how to fix it.

On every transition to `failed` a background task (never the monitor itself) runs high-precision regex rules over
the redacted log tail; exactly one matching cause -> source "rules". Otherwise the decision model is asked (log tail
as a quoted untrusted block in `state`); below threshold the diagnosis is stored with source "none" ("couldn't
diagnose confidently"). The result is stored in `diagnoses` and announced on SSE topic `diagnosis`.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from dgxmm.db import DecisionRecord, Deployment, DeploymentEvent, DiagnosisRow
from dgxmm.decider.common import finite, quoted_block, safe_decide, safe_record_rules
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.retention import prune_diagnoses_of
from dgxmm.decider.types import Option
from dgxmm.deployer.logtail import KERNEL_OOM, MAX_LINE_CHARS, cap_line, is_error_line  # noqa: F401 (MAX_LINE_CHARS re-exported)
from dgxmm.memory.redact import redact_text

log = logging.getLogger(__name__)
LOG_LINES, LOG_BYTES = 120, 6 * 1024
TIMEOUT_S = 10.0
# Untrusted log text is bounded before any rule runs (an echoed 44 KB architecture name made unbounded patterns
# backtrack for minutes): the last MAX_TEXT_CHARS of the text, every line cut at MAX_LINE_CHARS (logtail.cap_line).
# Every rule is also written with bounded quantifiers (tests/test_regex_safety.py).
MAX_TEXT_CHARS = 512 * 1024
QUESTION = "What is the root cause of this model-server startup failure?"


@dataclass(frozen=True)
class Cause:
    key: str
    label: str
    description: str  # one line; also the option description sent to the model
    fix_hint: str


TAXONOMY: list[Cause] = [
    Cause("gpu_oom", "Out of memory",
          "GPU / unified memory ran out while loading weights, profiling or capturing CUDA graphs",
          "Free memory (stop or sleep other deployments), reduce the context length, or use a smaller quantization."),
    Cause("kv_cache_too_small", "KV cache too small",
          "Weights loaded, but the memory left for the KV cache is too small for max_model_len",
          "Reduce the context length (max_model_len / ctx_size) or use an fp8 KV cache."),
    Cause("context_too_long", "Context longer than the model supports",
          "max_model_len / context size is larger than the model's maximum position embeddings",
          "Lower the context length to at most the model's maximum (or pick a model with a longer context)."),
    Cause("unsupported_architecture", "Unsupported architecture",
          "The engine does not support this model architecture / file format",
          "Use another engine (e.g. llama.cpp with a GGUF of this model) or a newer engine image."),
    Cause("trust_remote_code_required", "Needs trust_remote_code",
          "The model needs custom code and trust_remote_code was not enabled",
          "Enable trust_remote_code — only if you trust the code in this repository."),
    Cause("auth_gated", "Gated / private model",
          "Hugging Face authentication failed: gated or private repo, missing or invalid token (401/403)",
          "Accept the model's licence on huggingface.co and add a Hugging Face token in Settings."),
    Cause("bad_parser_flag", "Invalid parser flag",
          "Invalid or misconfigured tool-call / reasoning parser flags",
          "Pick a tool-call / reasoning parser that exists in this vLLM version, or clear it."),
    Cause("tokenizer_or_template", "Tokenizer or chat template problem",
          "Tokenizer files or chat template missing or broken",
          "Check that the tokenizer files and a chat template are present (re-download the model)."),
    Cause("port_in_use", "Port in use",
          "The server port is already in use",
          "Another process holds the port; start the deployment again (a free port is chosen) or free the port."),
    Cause("offline_missing_files", "Model files missing",
          "Model files are not available locally and cannot be downloaded (offline mode, wrong id or revision)",
          "Download the model (Catalog) and check the repository id and revision."),
    Cause("driver_mismatch", "Driver / CUDA mismatch",
          "CUDA driver / toolkit / NCCL / kernel image incompatible with the GPU or runtime",
          "Use an engine image built for this GPU (sm_121 / CUDA 13) or update the NVIDIA driver."),
    Cause("disk_full", "Disk full", "No space left on the disk",
          "Free disk space (delete unused models on the Models page) and try again."),
    # The decision model's own start-up refusals (deploy/decider/entrypoint.sh and launch.py exit 3 / 4). Rules
    # only: when the decider fails it cannot be asked, and the model's option set stays the one it was evaluated on.
    Cause("integrity_check_failed", "Decision model files fail the integrity check",
          "A file of the pinned decision-model snapshot is missing or differs from the hash pinned in the image",
          "Re-download the decision model, which re-verifies every file (`docker exec dgxmm python -m dgxmm.admin "
          "decider download`); if the image has no hash list for this model, rebuild it (`decider build`)."),
    Cause("api_key_missing", "Decision model key not delivered",
          "The decision model refused to start without its API key file (or with a key in its environment)",
          "Disable and enable the decision model so DGXMM starts it again and delivers a fresh key; remove any "
          "DECISION_API_KEY environment override."),
    Cause("other", "Other", "Something else", "See the log for details."),
]
CAUSES = {c.key: c for c in TAXONOMY}
RULES_ONLY_CAUSES = frozenset({"integrity_check_failed", "api_key_missing"})
OPTIONS = [Option(c.key, c.label, c.description) for c in TAXONOMY if c.key not in RULES_ONLY_CAUSES]

# High-precision rules: each pattern is specific to one cause. Lines echoing the configuration (vLLM's
# "non-default args: {...}", our own "usage:" / argument lists) are removed first, so a flag NAME appearing in the
# arguments (e.g. 'max_model_len', 'tool_call_parser') never triggers a rule.
RULES: list[tuple[str, re.Pattern]] = [(k, re.compile(p, re.I | re.M)) for k, p in [
    ("disk_full", r"No space left on device|\[Errno 28\]"),
    ("port_in_use", r"Address already in use|\[Errno 98\]|couldn't bind HTTP server socket|port is already allocated"),
    ("auth_gated", r"GatedRepoError|Cannot access gated repo|Invalid user token|Access to model \S+ is restricted"),
    ("trust_remote_code_required", r"pass the argument `?trust_remote_code=True`?|execute the configuration file in "
                                   r"that repo|contains custom code which must be executed"),
    ("bad_parser_flag", r"argument --(tool-call|reasoning)-parser: invalid choice|not found in registered reasoning "
                        r"parsers|--enable-auto-tool-choice requires --tool-call-parser|tool call parser\b[^\n]{0,200}?\bnot "
                        r"(found|registered)|invalid (tool[-_ ]call|reasoning) parser|(?:Reasoning|Tool[- ]call|Tool) "
                        r"parser '[^']{0,64}' not found"),
    ("context_too_long", r"User-specified max_model_len \(\d+\) is greater than the derived max_model_len"),
    ("kv_cache_too_small", r"No available memory for the cache blocks|larger than the available KV cache memory|"
                           r"estimated maximum model length is \d+|insufficient memory for the KV cache|"
                           r"maximum number of tokens that can be stored in KV cache"),
    ("gpu_oom", r"CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory|cudaMalloc failed: out of "
                r"memory|unable to allocate CUDA\d* buffer|failed to allocate CUDA\d* buffer|CUBLAS_STATUS_ALLOC_FAILED|"
                + KERNEL_OOM),
    ("unsupported_architecture", r"Model architectures? \[[^\]\n]{0,2000}\] (is|are) not supported|unknown model architecture|"
                                 r"are not supported for now"),
    ("tokenizer_or_template", r"Can't load tokenizer|tokenizer\.chat_template is not set|Cannot use chat template|"
                              r"jinja2\.exceptions\.\w*Error|/tokenizers/registry\.py\", line \d+, in get_tokenizer|"
                              r"tokenization_utils\w*\.py\", line \d+, in convert_to_native_format"),
    ("offline_missing_files", r"LocalEntryNotFoundError|outgoing traffic has been disabled|RevisionNotFoundError|"
                              r"RepositoryNotFoundError|couldn't find them in the cached files|Invalid rev id|"
                              r"dgxmm-decider: snapshot \S+ not found under|the pinned model snapshot is missing"),
    ("integrity_check_failed", r"dgxmm-decider: snapshot files do not match the pinned hashes|"
                               r"dgxmm-decider: no pinned hash list for revision|snapshot integrity check failed|"
                               r"^[\w./-]+: FAILED(?: open or read)?$|"
                               r"sha256sum: WARNING: \d+ computed checksums? did NOT match"),
    ("api_key_missing", r"dgxmm-decider: DECISION_API_KEY (?:is set in the environment|must not be set)|"
                        r"dgxmm-decider: no key in \S+ after|dgxmm-decider: cannot read the key file|"
                        r"no API key was delivered"),
    ("driver_mismatch", r"no kernel image is available|NVIDIA driver on your system is too old|driver version is "
                        r"insufficient|is not compatible with the current PyTorch installation|"
                        r"CUDA driver version is insufficient"),
]]
_ECHO_LINE = re.compile(r"non-default args:|^\s*usage: |^\s*\[entrypoint\]|Initializing a [^\n]{0,400}? engine [^\n]{0,2000}? with config:|"
                        r"^\s*\++ +(?:exec +)?(?:\S*/)?(?:vllm|python3?|llama-server)\b", re.I)


def fix_hint(cause: str, model_ref: str | None) -> str:
    """The cause's fix hint; for a failed integrity check it names the failing decision model and how to re-fetch it
    (the CLI verifies every file against that model's pinned hash list)."""
    c = CAUSES[cause]
    if cause != "integrity_check_failed" or not model_ref:
        return c.fix_hint
    from dgxmm.engines.decision import DECIDER_REPO
    cmd = "decider download" + ("" if model_ref == DECIDER_REPO else f" --model {model_ref}")
    page = "" if model_ref == DECIDER_REPO else " or with Download on the Decision model page"
    return (f"Re-download {model_ref}, which re-verifies every file against its pinned hashes (`docker exec dgxmm "
            f"python -m dgxmm.admin {cmd}`{page}); if the image has no hash list for this model, rebuild it "
            "(`decider build`).")


def bounded(text: str | None) -> str:
    """The last MAX_TEXT_CHARS of the text (whole lines), every line cut at MAX_LINE_CHARS."""
    text = text or ""
    if len(text) > MAX_TEXT_CHARS:
        text = text[-MAX_TEXT_CHARS:]
        text = text[text.find("\n") + 1:] if "\n" in text else text
    return "\n".join(cap_line(ln) for ln in text.splitlines())


def evidence(text: str) -> str:
    """Bounded log text without configuration-echo lines."""
    return "\n".join(ln for ln in bounded(text).splitlines() if not _ECHO_LINE.search(ln))


OOM_KILLED_LINE = "Docker reports the container was OOM-killed (State.OOMKilled)"


def rule_hits(text: str, exit_code: int | None = None, engine: str | None = None,
              oom_killed: bool = False) -> list[tuple[str, str]]:
    """[(cause, matched line)] for every rule that fires (one entry per cause). The decider's own start-up refusals
    (RULES_ONLY_CAUSES) are recognised only for the decision engine: other engines' logs can contain the same words
    (e.g. a "Status: FAILED" line)."""
    ev = evidence(text)
    out: list[tuple[str, str]] = []
    for cause, rx in RULES:
        if cause in RULES_ONLY_CAUSES and engine != "decision":
            continue
        m = rx.search(ev)
        if m:
            line_start = ev.rfind("\n", 0, m.start()) + 1
            line_end = ev.find("\n", m.end())
            out.append((cause, ev[line_start:line_end if line_end >= 0 else None].strip()[:240]))
    if oom_killed and not any(c == "gpu_oom" for c, _ in out):
        out.append(("gpu_oom", OOM_KILLED_LINE))  # the same evidence the deployer labels "out of memory" on
    return out  # exit code 137 alone is not evidence of a cause (a SIGKILL): the model sees it in the state


def _half(v: int | None, floor: int) -> int | None:
    if not v:
        return None
    h = int(v) // 2
    return h if h >= floor else None


def _rule_line(cause: str, ev: str) -> str:
    rx = dict(RULES)[cause]
    m = rx.search(ev)
    if not m:
        return ""
    start, end = ev.rfind("\n", 0, m.start()) + 1, ev.find("\n", m.end())
    return ev[start:end if end >= 0 else None]


def config_patch(cause: str, engine: str, cfg: dict, text: str, parser_hint=None) -> dict | None:
    """A config patch that addresses the cause, or None. Callers validate it against the engine config model."""
    ev = evidence(text)
    if engine == "vllm":
        if cause == "trust_remote_code_required":
            return {"trust_remote_code": True}
        if cause == "context_too_long":
            m = re.search(r"derived max_model_len \((?:max_position_embeddings|model_max_length)=(\d+)", ev)
            if m and 256 <= int(m.group(1)) < int(cfg.get("max_model_len") or 0):
                new = int(m.group(1))
                return {"max_model_len": new, **({"kv_tokens": None} if cfg.get("kv_tokens") else {})}
            return None
        if cause == "kv_cache_too_small":
            m = re.search(r"estimated maximum model length is (\d+)", ev)
            if m and int(m.group(1)) >= 1024:
                new = int(m.group(1)) // 1024 * 1024
                if new < int(cfg.get("max_model_len") or 0):
                    return {"max_model_len": new, **({"kv_tokens": None} if cfg.get("kv_tokens") else {})}
            if cfg.get("kv_cache_dtype", "auto") == "auto":
                return {"kv_cache_dtype": "fp8"}
            h = _half(cfg.get("max_model_len"), 4096)
            return {"max_model_len": h, **({"kv_tokens": None} if cfg.get("kv_tokens") else {})} if h else None
        if cause == "gpu_oom":
            h = _half(cfg.get("max_model_len"), 4096)
            return {"max_model_len": h, **({"kv_tokens": None} if cfg.get("kv_tokens") else {})} if h else None
        if cause == "bad_parser_flag":
            tool, reasoning = parser_hint or (None, None)
            line = _rule_line("bad_parser_flag", ev)  # decide from the line that matched, not the whole log
            if re.search(r"reasoning", line, re.I) and not re.search(r"tool-call-parser|tool call parser", line, re.I):
                return {"reasoning_parser": reasoning}
            if re.search(r"--enable-auto-tool-choice requires --tool-call-parser", line):
                return {"tool_call_parser": tool} if tool else {"enable_auto_tool_choice": False}
            return {"tool_call_parser": tool} if tool else {"tool_call_parser": None, "enable_auto_tool_choice": False}
    if engine == "llamacpp" and cause in ("gpu_oom", "kv_cache_too_small"):
        h = _half(cfg.get("ctx_size"), 4096)
        return {"ctx_size": h} if h else None
    return None


# Causes whose fix weakens a security boundary: offered as a config patch only when a high-precision rule matched the
# engine's own error message, never on the decision model's word alone (its input is untrusted log text that an
# attacker can shape; see docs/security.md). The fix hint still explains what to do.
RULES_ONLY_PATCH = frozenset({"trust_remote_code_required"})


def model_patch(cause: str, engine: str, cfg: dict, text: str, parser_hint=None) -> dict | None:
    """config_patch() for a cause the decision model chose (not a rule)."""
    if cause in RULES_ONLY_PATCH:
        return None
    return config_patch(cause, engine, cfg, text, parser_hint)


def validated_patch(engine_obj, cfg: dict, patch: dict | None) -> dict | None:
    """Keep the patch only if the patched config validates against the engine's config model."""
    if not patch or engine_obj is None:
        return None
    patch = {k: v for k, v in patch.items() if cfg.get(k, object()) != v}  # setting a value already there fixes nothing
    if not patch:
        return None
    try:
        engine_obj.validate_config({**cfg, **patch})
    except (ValidationError, ValueError, TypeError):
        return None
    return patch


def log_excerpt(text: str | None) -> str:
    lines = bounded(text).splitlines()[-LOG_LINES:]
    out = redact_text("\n".join(lines))
    return out[-LOG_BYTES:]


ERROR_LINES, ERROR_BYTES = 25, 2048
_CONDENSE_MARKER = re.compile(r"^\[(?:\d+ earlier lines omitted|last \d+ lines:)")  # dgxmm/deployer/logtail.py


def error_lines(text: str | None) -> list[str]:
    """Distinct error/exception lines of a log in order (traceback frames, vLLM's ERROR-level traceback dumps and
    config echo dropped), at most ERROR_LINES. vLLM's engine-core root cause usually sits above a long wrapper
    traceback that a plain tail cuts off."""
    out: list[str] = []
    for ln in evidence(text or "").splitlines():
        s = ln.strip()
        if is_error_line(ln) and not _CONDENSE_MARKER.match(s) and s not in out:
            out.append(s[:300])
    return out[-ERROR_LINES:]


def build_state(*, engine: str, model_ref: str, exit_code: int | None, message: str, hits: list[tuple[str, str]],
                log_text: str) -> str:
    parts = [f"Engine: {engine}", f"Exit code: {exit_code if exit_code is not None else 'unknown'}",
             quoted_block("MODEL REFERENCE", model_ref, max_chars=300),
             quoted_block("DGXMM STATUS MESSAGE", message, max_chars=600)]
    if hits:
        parts.append("Several log patterns matched: " + ", ".join(c for c, _ in hits))
    errs = error_lines(log_text)
    if errs:
        parts.append(quoted_block("ERROR LINES", "\n".join(errs), max_chars=ERROR_BYTES, keep="head"))
    parts.append(quoted_block("LOG", bounded(log_text), max_chars=LOG_BYTES, max_lines=LOG_LINES))
    return "\n".join(parts)


class DiagnosisService:
    def __init__(self, ctx, timeout_s: float = TIMEOUT_S):
        self.ctx, self.timeout_s = ctx, timeout_s
        self._tasks: set[asyncio.Task] = set()

    @property
    def decider(self):
        return decider_of(self.ctx)

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        """Deployer state hook: diagnose in the background so nothing waits on the decider."""
        if to != "failed":
            return
        task = asyncio.get_running_loop().create_task(self.diagnose(dep_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:  # tests
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await asyncio.sleep(0)  # let done-callbacks remove finished tasks

    def _parser_hint(self, model_ref: str):
        try:
            from dgxmm.decider.config_suggest import model_info, rule_reasoning, rule_tool
            a = self.ctx.inventory.get(model_ref) if self.ctx.inventory is not None else None
            if a is None:
                return None
            info = model_info(a)  # reads files: callers run this in a thread
            t, r = rule_tool(info), rule_reasoning(info)
            norm = lambda h: None if h is None or h.value == "none" else h.value  # noqa: E731
            return norm(t), norm(r)
        except Exception:
            log.debug("parser hint failed", exc_info=True)
            return None

    async def diagnose(self, dep_id: int) -> dict | None:
        try:
            return await self._diagnose(dep_id)
        except Exception:
            log.exception("diagnosis of deployment %s failed", dep_id)
            return None

    def _load(self, dep_id: int):
        with self.ctx.db.session() as s:
            dep = s.get(Deployment, dep_id)
            if dep is None or dep.state != "failed":
                return None
            ev = s.scalars(select(DeploymentEvent).where(DeploymentEvent.deployment_id == dep_id,
                                                         DeploymentEvent.to_state == "failed")
                           .order_by(DeploymentEvent.id.desc()).limit(1)).first()
            if ev is None:
                return None
            if s.scalar(select(DiagnosisRow.id).where(DiagnosisRow.event_id == ev.id)) is not None:
                return None  # already diagnosed
            return (dep.engine, dep.model_ref, dict(dep.config or {}), dep.exit_code, bool(dep.oom_killed), ev.id,
                    ev.message or "", ev.log_tail or "")

    def _insert(self, **fields) -> dict | None:
        try:
            with self.ctx.db.session() as s:
                row = DiagnosisRow(**fields)
                s.add(row)
                s.flush()
                prune_diagnoses_of(s, row.deployment_id)  # keep the latest DIAGNOSES_PER_DEPLOYMENT
                return row_view(row)
        except IntegrityError:
            return None  # a concurrent diagnose() stored this event first

    async def _diagnose(self, dep_id: int) -> dict | None:
        loaded = await asyncio.to_thread(self._load, dep_id)
        if loaded is None:
            return None
        engine, model_ref, cfg, exit_code, oom_killed, event_id, message, log_tail = loaded
        text = f"{log_tail}\n{message}" if log_tail else message
        # every regex over the untrusted log runs in a worker thread, never on the event loop
        hits = await asyncio.to_thread(rule_hits, text, exit_code, engine, oom_killed)
        decider = self.decider
        subject = f"deployment:{dep_id}:event:{event_id}"
        cause, source, conf, decision_id = None, "none", None, None
        summary_extra = ""
        if len(hits) == 1:
            cause, line = hits[0]
            d = safe_record_rules(decider, feature="diagnosis", subject=subject, choice=cause,
                                  reason=f"log matched: {redact_text(line)}", options=OPTIONS)
            source, conf, decision_id = "rules", 1.0, d.decision_id
            summary_extra = f" Matched: “{redact_text(line)[:200]}”."
        else:
            state = build_state(engine=engine, model_ref=model_ref, exit_code=exit_code, message=message, hits=hits,
                                log_text=log_tail or message)
            d = await safe_decide(decider, feature="diagnosis", subject=subject, state=state, question=QUESTION,
                                  options=OPTIONS, timeout_s=self.timeout_s)
            decision_id, conf = d.decision_id, d.confidence
            if d.source == "model" and d.choice in CAUSES:
                cause, source = d.choice, "model"
        info = CAUSES.get(cause) if cause else None
        patch = None
        if info is not None:
            hint = await asyncio.to_thread(self._parser_hint, model_ref) if cause == "bad_parser_flag" else None
            make = config_patch if source == "rules" else model_patch
            patch = validated_patch(self.ctx.engines.get(engine) if self.ctx.engines else None, cfg,
                                    await asyncio.to_thread(make, cause, engine, cfg, text, hint))
        summary = (info.description + "." + summary_extra) if info else \
            "Couldn't diagnose this failure confidently — see the log."
        excerpt = await asyncio.to_thread(log_excerpt, log_tail or message)
        out = await asyncio.to_thread(
            self._insert, deployment_id=dep_id, event_id=event_id, cause=cause, source=source, confidence=conf,
            decision_id=decision_id, summary=summary, fix_hint=fix_hint(cause, model_ref) if info else "", patch=patch,
            log_excerpt=excerpt)
        if out is None:
            return None
        self.ctx.events.publish("diagnosis", {"deployment_id": dep_id})
        return out

    def forget(self, dep_id: int) -> None:
        """Deployer removal hook: a removed deployment's diagnoses go with it (also an FK cascade)."""
        with self.ctx.db.session() as s:
            s.execute(delete(DiagnosisRow).where(DiagnosisRow.deployment_id == dep_id))

    def latest(self, dep_id: int) -> dict | None:
        with self.ctx.db.session() as s:
            row = s.scalars(select(DiagnosisRow).where(DiagnosisRow.deployment_id == dep_id)
                            .order_by(DiagnosisRow.id.desc()).limit(1)).first()
            if row is None:
                return None
            rec = s.get(DecisionRecord, row.decision_id) if row.decision_id is not None else None
            return row_view(row, rec.feedback if rec is not None else None)


def row_view(row: DiagnosisRow, feedback: str | None = None) -> dict:
    """``feedback``: the admin's 👍/👎 on its decision (accepted | rejected), so the card shows it after a reload."""
    c = CAUSES.get(row.cause) if row.cause else None
    return {"cause": row.cause, "label": c.label if c else "Couldn't diagnose confidently", "source": row.source,
            "confidence": finite(row.confidence), "summary": row.summary, "fix_hint": row.fix_hint, "patch": row.patch,
            "decision_id": row.decision_id, "created_at": row.created_at.isoformat() if row.created_at else None,
            "log_excerpt": row.log_excerpt, "feedback": feedback}
