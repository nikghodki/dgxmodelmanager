"""The log text stored with a failed deployment's event (shown on the Events tab and read by the diagnosis).

Engines often print the root cause far above the end of the log: a vLLM start-up that failed on a full disk ends
in ~480 lines of wrapper tracebacks below `ld: final link failed: No space left on device` (measured on the DGX,
tests/fixtures/real_failures/disk-full.txt). So the deployer reads more than it stores and keeps, above the last
TAIL_LINES lines, the distinct error lines of the part that was cut.
"""
from __future__ import annotations

import re

from dgxmm.memory.redact import TRUNC, mask_pem_blocks, redact_log

TAIL_LINES = 200     # lines stored verbatim (what the deployer always stored)
SCAN_LINES = 5000    # lines read from the container to look for earlier error lines
EARLY_ERRORS = 40    # earlier error lines kept at most (the last ones, distinct)
MAX_LINE_CHARS = 2048  # every line is cut here before any regex runs on it (untrusted text, docs/security.md)
_ERROR = re.compile(r"\b\w*(?:Error|Exception)\b[:(]|\bERROR\b|\bFATAL\b|\bfailed\b|\berror:|No space left|"
                    r"out of memory|Killed", re.I)
# "(EngineCore pid=148) ERROR 10-17 01:50:53 [core.py:1368] <message>": vLLM re-logs whole tracebacks at ERROR level,
# so the logger prefix is removed before deciding whether a line is an error message or a traceback frame.
_PREFIX = re.compile(r"^\s*(?:\(\w+ pid=\d+\)\s?)?(?:(?:ERROR|WARNING|INFO|DEBUG|CRITICAL) +\d\d-\d\d "
                     r"[\d:]+ \[[^\]]*\] ?)?")
_FRAME = re.compile(r"^(?:\s{2,}|File \"|\^+\s*$|~+[\^~]*\s*$|Traceback \(most recent call last\))")


# The kernel OOM killer's own lines (dmesg wording, also seen in container logs): evidence of an out-of-memory kill.
# A bare "Killed" is not: a shell prints it for any SIGKILL (docker kill, a stop timeout).
KERNEL_OOM = (r"Out of memory: Killed process \d+|Memory cgroup out of memory: Killed process|oom-kill:constraint=|"
              r"oom_reaper: reaped process \d+")
KERNEL_OOM_RE = re.compile(KERNEL_OOM)


def oom_evidence(text: str | None) -> bool:
    """Does the (capped) log text carry the kernel OOM killer's own line?"""
    return any(KERNEL_OOM_RE.search(cap_line(ln)) for ln in (text or "").splitlines())


def cap_line(line: str, limit: int = MAX_LINE_CHARS) -> str:
    """The line cut at `limit` characters. The cut drops the trailing partial word, so a secret straddling the
    boundary never survives as a fragment that is too short for the redaction rules to recognise."""
    if len(line) <= limit:
        return line
    head = line[:limit]
    if not line[limit].isspace():
        head = head[:max(head.rfind(" "), head.rfind("\t"), 0)]
    return head + TRUNC


def message_of(line: str) -> str:
    """The line without vLLM's process / logger prefix."""
    return _PREFIX.sub("", line, count=1)


def is_error_line(line: str) -> bool:
    m = message_of(line)
    return bool(m.strip()) and not _FRAME.search(m) and bool(_ERROR.search(m))


def condense(text: str, tail: int = TAIL_LINES, early: int = EARLY_ERRORS) -> str:
    lines = [cap_line(ln) for ln in (text or "").splitlines()]
    if len(lines) <= tail:
        return "\n".join(lines)
    head, last = lines[:-tail], lines[-tail:]
    errs: list[str] = []
    for ln in head:
        s = ln.strip()
        if is_error_line(ln):
            if s in errs:
                errs.remove(s)
            errs.append(s[:400])
    errs = errs[-early:]
    out = [f"[{len(head)} earlier lines omitted; {len(errs)} error line(s) among them:]", *errs,
           f"[last {tail} lines:]", *last]
    return "\n".join(out)


def stored_tail(text: str) -> str:
    """What a failed event persists: the condensed log, redacted (secrets far above the kept tail included). Whole
    PEM key blocks are masked before condensing (which could keep a block's body but cut its BEGIN line); what is
    left of one is masked as a run of base64 lines by redact_log."""
    return redact_log(condense(mask_pem_blocks(text)))
