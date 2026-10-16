"""Redact secrets from process command lines before they leave the server (spec §5).

Fail closed: any flag or KEY=VALUE whose name mentions key/token/secret/password/passwd/auth/credential has its
value replaced, with a tiny allowlist of well-known non-secret words (``tokenizer``, the plural ``tokens`` used by
count flags such as ``--max-num-batched-tokens``). Known token shapes (URL userinfo, Bearer/Basic, ``hf_…``,
``sk-…``) are redacted wherever they appear, including inside ``sh -c`` strings and ``python -c`` code.
"""
from __future__ import annotations

import re
import shlex

MASK = "***"
MAX_COMMAND_BYTES = 4096

_SENSITIVE = re.compile(r"key|token|secret|password|passwd|auth|credential", re.I)
_SAFE_SEGMENTS = frozenset({"tokenizer", "tokenizers", "tokens"})
_EXACT_SEGMENTS = frozenset({"pass", "pw", "pwd"})
_FLAG = re.compile(r"^(--?)([A-Za-z][\w.-]*)(?:(=)(.*))?$", re.S)

_URL_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@'\"]+(?::[^/\s@'\"]*)?@")
_AUTH_SCHEME = re.compile(r"(?i)\b(bearer|basic)(\s+)[A-Za-z0-9._~+/=-]+")
_HF = re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}")
_SK = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")
_VALUE = r"(\"[^\"]*\"|'[^']*'|[^\s&'\",;)}\]]+)"
_INLINE_FLAG = re.compile(r"(?<![\w-])(--?([A-Za-z][\w.-]*))(=|[ \t]+)(?!-)" + _VALUE)
# Only keys that might be sensitive start a match, so a harmless pair (e.g. a URL scheme) never swallows the
# text after it; the callback applies the exact rule (allowlist included).
_PAIR = re.compile(r"(?<![\w-])([\"']?)((?=[\w.-]*?(?i:key|token|secret|pass|pw|auth|credential))[A-Za-z_][\w.-]*)\1"
                   r"(\s*[=:]\s*)" + _VALUE)
_SHELL_SAFE = re.compile(r"[\w@%+=:,./*-]+")


def is_sensitive_name(name: str) -> bool:
    segments = [s for s in re.split(r"[-_.]", name.lower()) if s]
    if any(s in _EXACT_SEGMENTS for s in segments):
        return True
    return bool(_SENSITIVE.search("-".join(s for s in segments if s not in _SAFE_SEGMENTS)))


def _mask_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[0] + MASK + value[0]
    return MASK


def redact_text(text: str) -> str:
    text = _URL_USERINFO.sub(lambda m: m.group(1) + MASK + "@", text)
    text = _AUTH_SCHEME.sub(lambda m: m.group(1) + m.group(2) + MASK, text)
    text = _HF.sub(MASK, text)
    text = _SK.sub(MASK, text)
    text = _INLINE_FLAG.sub(
        lambda m: m.group(1) + m.group(3) + _mask_value(m.group(4)) if is_sensitive_name(m.group(2)) else m.group(0),
        text)

    def _pair(m: re.Match) -> str:
        value = m.group(4)
        if not is_sensitive_name(m.group(2)) or value.strip("'\"").lower() in ("bearer", "basic", MASK):
            return m.group(0)
        return m.group(1) + m.group(2) + m.group(1) + m.group(3) + _mask_value(value)
    return _PAIR.sub(_pair, text)


def _looks_like_flag(arg: str) -> bool:
    return len(arg) > 1 and arg[0] == "-" and not re.fullmatch(r"-\d[\d.]*", arg)


def redact_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    pending = False  # previous arg was a sensitive flag without "=": this arg is its value
    for arg in argv:
        if pending:
            pending = False
            if not _looks_like_flag(arg):
                out.append(MASK)
                continue
        m = _FLAG.match(arg)
        if m and is_sensitive_name(m.group(2)):
            if m.group(3):
                out.append(f"{m.group(1)}{m.group(2)}={MASK}")
            else:
                out.append(arg)
                pending = True
            continue
        out.append(redact_text(arg))
    return out


def _quote(arg: str) -> str:
    return arg if _SHELL_SAFE.fullmatch(arg) else shlex.quote(arg)


def format_command(argv: list[str]) -> tuple[str, bool]:
    """Redacted, shell-quoted command line, truncated to MAX_COMMAND_BYTES. Returns (command, truncated)."""
    text = " ".join(_quote(a) for a in redact_argv(argv))
    raw = text.encode()
    if len(raw) <= MAX_COMMAND_BYTES:
        return text, False
    return raw[:MAX_COMMAND_BYTES].decode("utf-8", "ignore") + "…", True
