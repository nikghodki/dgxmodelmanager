"""Advanced feature 11 — ``trust_remote_code`` risk triage: what does a repository's custom code do?

Triggered whenever a deploy / reconfigure would enable ``trust_remote_code`` (wizard, reconfigure dialog, a
diagnosis patch) for a model whose config has ``auto_map``. The snapshot's ``.py`` files are scanned statically by
dgxmm/decider/code_scan.py in a separate ``python -I -S`` process with CPU, address-space, file-size and wall-clock
limits and an empty environment: the files are parsed with ``ast`` and never imported or executed.

Every rule finding (network / process / native imports, process calls, eval / exec, decoded-then-executed data,
writes outside the model directory, credential environment reads, long encoded literals, …) shows a warning. The
decision model labels the scan summary and the finding excerpts ``routine`` or ``unusual``: ``unusual`` at or above
the threshold adds a warning; ``routine`` is never a verdict (it is only logged): a complete scan without findings
is the neutral ``no_findings`` ("no suspicious patterns found, static scan only") whatever the model says, and the
generic trust_remote_code warning always stays. Findings are informational: they never block a deploy. The scan is
static: code assembled at run time in ways the rules do not name (code_scan.py lists what they do) goes unseen.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from dgxmm.decider.common import abstain, arecord_rules, quoted_block, safe_decide, threshold_of
from dgxmm.decider.compat import rules_on
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
FEATURE = "remote_code"
SCANNER = str(Path(__file__).with_name("code_scan.py"))
SCAN_TIMEOUT_S = 30.0         # wall clock
SCAN_OUTPUT_MAX = 4 * 1024**2
FILE_TIMEOUT_S = 10.0         # per file, when the whole-repo scan failed
MAX_FILES_ONE_BY_ONE = 60
TIMEOUT_S = 4.0
NEVER = 1.01
LABELS = {"routine": "Ordinary model code: architecture, tokenizer or processor definitions; any imports and file "
                     "access serve loading the model",
          "unusual": "Code that does more than define a model: contacts the network, runs processes, executes "
                     "hidden or decoded code, touches files or credentials outside the model directory"}
OPTIONS = [Option(k, k, d) for k, d in LABELS.items()]
QUESTION = "Is this model repository's custom Python code routine model code, or does it do something unusual?"
KIND_TEXT = {"network_import": "imports a network module", "process_import": "imports a process module",
             "native_import": "loads native code (ctypes / cffi)", "process_call": "runs a process",
             "dynamic_exec": "calls eval / exec / compile / __import__", "encoded_exec": "executes decoded data",
             "file_write_outside": "writes a file outside the model directory",
             "sensitive_path": "names a credential or shell start-up file", "credential_env":
             "reads a credential from the environment", "environment_dump": "copies the whole environment",
             "long_encoded_literal": "embeds a long encoded string", "unparseable": "could not be parsed",
             "oversized_file": "is too large to scan", "unanalysable": "could not be analysed",
             "dynamic_import": "imports a module by a computed name or a file path",
             "builtins_access": "reaches the builtins",
             "dynamic_attribute": "looks up a risky function by name", "obfuscated_name":
             "assembles a risky name from pieces", "unsafe_deserialize": "unpickles / deserialises data",
             "external_code": "loads code from another repository", "missing_code":
             "references code that is not in the snapshot", "code_object": "builds or patches a code object",
             "object_walk": "walks the object graph (__subclasses__ / __globals__)",
             "native_load": "loads native code at run time",
             "remote_fetch": "downloads files or code from the Hub at run time"}
_cache: OrderedDict[tuple, dict] = OrderedDict()
CACHE_MAX = 32
FAILED_SCAN_TTL_S = 120.0  # a failed scan (up to 1 + MAX_FILES_ONE_BY_ONE processes) is not repeated sooner
SCAN_CONCURRENCY = 1       # scans running at once, across all checks
_failed: dict[tuple, tuple[float, dict]] = {}
_slots = threading.BoundedSemaphore(SCAN_CONCURRENCY)


def _now() -> float:
    return time.monotonic()


def auto_maps(artifact) -> list[dict]:
    """The config's auto_map tables (top level and text_config)."""
    cfg = artifact.config or {}
    t = cfg.get("text_config")
    return [m for m in (cfg.get("auto_map"), t.get("auto_map") if isinstance(t, dict) else None)
            if isinstance(m, dict) and m]


def has_auto_map(artifact) -> bool:
    return bool(auto_maps(artifact))


def auto_map_findings(artifact) -> list[dict]:
    """transformers loads ``repo--module.Class`` references from ANOTHER repository (never in this snapshot, so not
    scanned): ``external_code``; a module that is not in the snapshot: ``missing_code``."""
    out, seen = [], set()
    have = set()
    try:
        for dirpath, _dirs, files in os.walk(artifact.local_path):
            have |= {os.path.relpath(os.path.join(dirpath, f), artifact.local_path) for f in files}
    except OSError:
        pass
    for m in auto_maps(artifact):
        for v in m.values():
            for ref in (v if isinstance(v, list) else [v]):
                if not isinstance(ref, str) or not ref or ref in seen:
                    continue
                seen.add(ref)
                if "--" in ref:
                    out.append({"file": "config.json", "line": 0, "kind": "external_code", "detail": ref[:120],
                                "excerpt": ""})
                    continue
                module = ref.rsplit(".", 1)[0].replace(".", "/") + ".py"
                if module not in have:
                    out.append({"file": "config.json", "line": 0, "kind": "missing_code", "detail": module[:120],
                                "excerpt": ""})
    return out


def repo_root(snapshot: str) -> str:
    """The HF cache repo directory (models--org--name) of a snapshot; symlinks may only point inside it."""
    p = Path(snapshot)
    return str(p.parent.parent) if p.parent.name == "snapshots" else str(p)


def _run_scanner(root: str, snapshot: str, only: str | None = None) -> dict:
    """One scanner process (it limits its own CPU, memory and file writes, code_scan.apply_limits; no preexec_fn,
    which is unsafe in a threaded parent): its JSON result or ``{"error": ...}``."""
    cmd = [sys.executable, "-I", "-S", SCANNER, root, snapshot] + ([only] if only is not None else [])
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=SCAN_TIMEOUT_S if only is None else FILE_TIMEOUT_S,
                           env={}, cwd="/", stdin=subprocess.DEVNULL, start_new_session=True, check=False)
    except subprocess.TimeoutExpired:
        return {"error": "scan_timeout"}
    except OSError as e:
        return {"error": f"scan_failed: {type(e).__name__}"}
    if r.returncode != 0 or len(r.stdout) > SCAN_OUTPUT_MAX:
        return {"error": f"scan_failed: exit {r.returncode}"}
    try:
        out = json.loads(r.stdout)
    except ValueError:
        return {"error": "scan_failed: output"}
    return out if isinstance(out, dict) and isinstance(out.get("findings"), list) else {"error": "scan_failed"}


def scan_snapshot(snapshot: str) -> dict:
    """Scan `snapshot` (blocking): ``{"findings": [...], "files": [...], ...}``. When the whole-repo scan fails (a
    file that exhausts the limits, a crash), every file is scanned in its own process, so one bad file cannot hide
    the others' findings; a file that still fails is an ``unanalysable`` finding. ``{"error": ...}`` only when no
    file list can be made."""
    root = repo_root(snapshot)
    whole = _run_scanner(root, snapshot)
    if "error" not in whole:
        return whole
    from dgxmm.decider.code_scan import py_files
    try:
        files = py_files(snapshot)
    except OSError:
        return whole
    if not files:  # nothing to retry one by one: the failure stands
        return whole
    out = {"findings": [], "files": [], "scanned_files": 0, "bytes": 0, "truncated": False}
    for rel in files[:MAX_FILES_ONE_BY_ONE]:
        one = _run_scanner(root, snapshot, rel)
        if "error" in one:
            out["findings"].append({"file": rel, "line": 0, "kind": "unanalysable", "detail": one["error"][:120],
                                    "excerpt": ""})
            continue
        out["findings"] += one["findings"]
        out["files"] += one.get("files") or []
        out["scanned_files"] += int(one.get("scanned_files") or 0)
    out["truncated"] = len(files) > MAX_FILES_ONE_BY_ONE
    return out


def _cached_scan(artifact) -> dict:
    try:
        mtime = os.stat(artifact.local_path).st_mtime
    except OSError:
        mtime = 0.0
    key = (artifact.local_path, artifact.revision, mtime)
    with _slots:  # one scan at a time; a check waiting here finds the result the one before it stored
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        hit = _failed.get(key)
        if hit is not None and _now() < hit[0]:
            return hit[1]
        out = scan_snapshot(artifact.local_path)
        if "error" not in out:
            _cache[key] = out
            _failed.pop(key, None)
            while len(_cache) > CACHE_MAX:
                _cache.popitem(last=False)
        else:
            _failed[key] = (_now() + FAILED_SCAN_TTL_S, out)
            for k in [k for k, (until, _) in _failed.items() if until <= _now()]:
                del _failed[k]
    return out


def build_state(ref: str, scan: dict) -> str:
    files = scan.get("files") or []
    findings = scan.get("findings") or []
    lines = [f"Python files scanned: {scan.get('scanned_files', len(files))}; rule findings: {len(findings)}"
             + (" (truncated)" if scan.get("truncated") else ""),
             quoted_block("REPOSITORY", ref, max_chars=300)]
    summary = "\n".join(f"{f.get('file')}: {f.get('lines')} lines, {f.get('classes')} classes, "
                        f"{f.get('functions')} functions; imports {', '.join(f.get('imports') or []) or 'none'}"
                        for f in files[:40])
    lines.append(quoted_block("FILES", summary or "(none)", max_chars=2500, keep="head"))
    if findings:
        ex = "\n".join(f"{x.get('file')}:{x.get('line')} [{x.get('kind')}] {x.get('excerpt') or x.get('detail', '')}"
                       for x in findings[:30])
        lines.append(quoted_block("FINDINGS", ex, max_chars=3000, keep="head"))
    return "\n".join(lines)


def combine(findings: list, d, truncated: bool = False) -> tuple[str, str]:
    """(verdict, source) of a check: rule findings always make it ``unusual``; the model's ``unusual`` at or above
    the threshold too; a scan that did not cover everything is ``unknown``; otherwise ``no_findings`` (never
    ``routine``: the model can only add a warning, never reassure)."""
    if findings:
        if d is not None and d.source == "model" and d.choice == "unusual":
            return "unusual", "model"
        return "unusual", "rules"
    if d is not None and d.source == "model" and d.choice == "unusual":
        return "unusual", "model"
    if truncated:
        return "unknown", "none"
    return "no_findings", "rules"


def _out(**kw) -> dict:
    return {"verdict": "unknown", "findings": [], "source": "none", "confidence": None, "model_label": None,
            "decision_id": None, "scanned_files": 0, "truncated": False, "error": None, "disabled": False,
            "remote_code": True, **kw}


async def check(ctx, model_ref: str) -> dict:
    """``{verdict: unusual|no_findings|unknown|no_custom_code, findings: [{file, line, kind, detail}], source,
    confidence, model_label, ...}``. KeyError for a model that is not on disk.

    The one exception to "a feature that is switched off is off": the static scan always runs and its findings are
    always shown whenever trust_remote_code would be enabled; the switch only skips the model's question
    (``disabled`` is therefore always false)."""
    decider = decider_of(ctx)
    ask_model = rules_on(decider, FEATURE)
    artifact = ctx.inventory.get(model_ref)
    if artifact is None:
        raise KeyError(model_ref)
    if not has_auto_map(artifact):
        return _out(verdict="no_custom_code", source="rules", confidence=1.0, remote_code=False)
    scan = await asyncio.to_thread(_cached_scan, artifact)
    error = None
    if "error" in scan:  # nothing could be analysed: that is itself a finding (never "routine")
        log.warning("remote-code scan of %s failed: %s", model_ref, scan["error"])
        error = scan["error"]
        scan = {"findings": [{"file": "*", "line": 0, "kind": "unanalysable", "detail": error[:120], "excerpt": ""}],
                "files": [], "scanned_files": 0, "truncated": True}
    findings = await asyncio.to_thread(auto_map_findings, artifact) + scan["findings"]
    subject = f"model:{model_ref}"[:256]
    rules_d = None
    if findings:
        kinds = sorted({f["kind"] for f in findings})
        rules_d = await arecord_rules(decider, feature=FEATURE, subject=subject, choice="unusual",
                                    reason="findings: " + ", ".join(kinds))
    state = await asyncio.to_thread(build_state, model_ref, scan)
    if ask_model:
        d = await safe_decide(decider, feature=FEATURE, subject=subject, state=state, question=QUESTION,
                              options=OPTIONS, threshold=threshold_of(decider, FEATURE, NEVER), timeout_s=TIMEOUT_S)
    else:
        d = abstain(FEATURE, subject)
    verdict, source = combine(findings, d, truncated=bool(scan.get("truncated")))
    label = "unusual" if d.source == "model" and d.choice == "unusual" else None  # "routine" is never shown
    conf = (1.0 if findings else None) if source == "rules" else d.confidence
    did = rules_d.decision_id if source == "rules" and rules_d is not None else d.decision_id
    return _out(verdict=verdict, source=source, model_label=label, confidence=conf, decision_id=did,
                findings=[{k: f.get(k) for k in ("file", "line", "kind", "detail")} for f in findings],
                scanned_files=int(scan.get("scanned_files") or 0), truncated=bool(scan.get("truncated")), error=error)


# ---------------- evaluation adapter ----------------
_CORPUS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "decider_corpus" / "remote_code"


def eval_case(case: dict):
    """A corpus case: ``scan`` (a stored scan of a public repo's code) or ``dir`` (a synthetic sample under the
    corpus directory, scanned now) -> the question DGXMM asks; the rules say ``unusual`` when they find anything."""
    scan = case.get("scan")
    if scan is None:
        d = str(_CORPUS / case["dir"])
        scan = scan_snapshot(d)
        if "error" in scan:
            raise RuntimeError(f"scan of {case['dir']} failed: {scan['error']}")
    return SimpleNamespace(rules="unusual" if scan["findings"] else None,
                           state=build_state(case.get("ref") or case["name"], scan), question=QUESTION,
                           options=OPTIONS)
