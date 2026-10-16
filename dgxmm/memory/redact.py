"""Redact secrets from process command lines before they leave the server (spec §5).

Fail closed — over-redaction is the safe failure:
- a flag or KEY=VALUE whose name contains key/token/secret/pass/pwd/auth/credential/cookie (or is sig/signature)
  has its value masked, except for a few exact non-secret names (``tokenizer…``, count flags such as
  ``--max-num-batched-tokens``, ``passthrough``/``bypass``…);
- after a sensitive flag without ``=`` the next argument is masked unless it starts with ``--``;
- header-like keys (Authorization, Proxy-Authorization, Cookie, X-…-Key/Token) are masked to the end of the header;
- URL userinfo, Bearer/Basic, and known token shapes (hf_, sk-, ghp_/github_pat_, AKIA…, xox?-, JWT) anywhere;
- tool-specific short options (``mysql -pS``, ``redis-cli -a S``, ``curl -u u:S``, ``sshpass -p S``);
- ``sh -c`` / ``python -c`` strings get the text rules, and are also tokenised (shlex) and redacted as argv, with any
  masked token removed from the text.
"""
from __future__ import annotations

import os
import re
import shlex

MASK = "***"
MAX_COMMAND_BYTES = 4096

_SENSITIVE = re.compile(r"key|token|secret|pass|pwd|auth|credential|cookie", re.I)
_EXACT_SENSITIVE = frozenset({"pw", "sig", "signature"})
_COUNT_WORDS = frozenset({"max", "min", "num", "batched", "new", "prompt", "draft", "speculative", "budget", "output",
                          "input", "context", "total", "lookahead", "reasoning"})
_SAFE_PASS_WORDS = frozenset({"passthrough", "bypass", "compass", "passive", "passes", "multipass", "pass-through"})
_FLAG = re.compile(r"^(--?)([A-Za-z][\w.-]*)(?:(=)(.*))?$", re.S)

# Tool-specific options (argv[0] basename, or a later token naming the tool inside sh -c)
_TOOL_VALUE_FLAGS = {  # option → the NEXT argument is a secret
    "redis-cli": {"-a"}, "curl": {"-u", "--user", "-U", "--proxy-user"}, "sshpass": {"-p"},
    "wget": {"--user", "--http-user"}, "htpasswd": set(),
}
_TOOL_EQ_FLAGS = {"curl": {"--user", "--proxy-user"}}  # --user=u:S
_TOOL_GLUED = {  # option glued to its secret value (mysql -pS); a bare option means "prompt"
    t: "-p" for t in ("mysql", "mysqldump", "mysqladmin", "mysqlimport", "mysqlcheck", "mariadb", "mariadb-dump",
                      "mariadb-admin", "mysqlsh")
}
_KNOWN_TOOLS = set(_TOOL_VALUE_FLAGS) | set(_TOOL_GLUED)

_URL_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s'\"]*@")  # greedy: up to the LAST @ in the token
_HEADER = re.compile(r"(?i)(?<![\w-])((?:proxy-)?authorization|(?:set-)?cookie|x-[\w-]*(?:key|token|auth|secret)[\w-]*"
                     r"|api-?key)(\s*:\s*)([^'\"\n]*)")
_AUTH_SCHEME = re.compile(r"(?i)\b(bearer|basic|digest)(\s+)[A-Za-z0-9._~+/=:-]{6,}")
_TOKEN_SHAPES = [re.compile(p) for p in (
    r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}",
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}",
    r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}",
    r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}",
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])",
    r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}",
    r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}",
)]
_QUOTED = r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"
_VALUE = r"(" + _QUOTED + r"|[^\s&'\",;(){}\[\]]+)"
_INLINE_FLAG = re.compile(r"(?<![\w-])(--?([A-Za-z][\w.-]*))(=|[ \t]+)(?!--)" + _VALUE)
_LIST_FLAG = re.compile(r"([\"'])(--?([A-Za-z][\w.-]*))\1(\s*,\s*)(" + _QUOTED + r")")  # ['--api-key', 'S']
# Only keys that might be sensitive start a match, so a harmless pair (e.g. a URL scheme) never swallows the
# text after it; the callback applies the exact rule (allowlist included).
_PAIR = re.compile(r"(?<![\w-])([\"']?)((?=[\w.-]*?(?i:key|token|secret|pass|pw|auth|credential|cookie|sig))"
                   r"[A-Za-z_][\w.-]*)\1(\s*[=:]\s*)" + _VALUE)
_SHELL_SAFE = re.compile(r"[\w@%+=:,./*-]+")


def is_sensitive_name(name: str) -> bool:
    lower = name.lower()
    segments = [s for s in re.split(r"[-_.]", lower) if s]
    if any(s in _EXACT_SENSITIVE for s in segments):
        return True
    kept: list[str] = []
    for i, s in enumerate(segments):
        if s in ("tokenizer", "tokenizers") or s in _SAFE_PASS_WORDS:
            continue
        if s == "tokens" and any(p in _COUNT_WORDS for p in segments[:i]):
            continue  # --max-num-batched-tokens, --num-speculative-tokens: counts, not credentials
        kept.append(s)
    if "pass-through" in lower:
        return bool(_SENSITIVE.search(lower.replace("pass-through", "")))
    return bool(_SENSITIVE.search("-".join(kept)))


def _mask_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[0] + MASK + value[0]
    return MASK


def redact_text(text: str) -> str:
    text = _URL_USERINFO.sub(lambda m: m.group(1) + MASK + "@", text)
    text = _HEADER.sub(lambda m: m.group(1) + m.group(2) + (MASK if m.group(3).strip() else m.group(3)), text)
    text = _AUTH_SCHEME.sub(lambda m: m.group(1) + m.group(2) + MASK, text)
    for rx in _TOKEN_SHAPES:
        text = rx.sub(MASK, text)
    text = _LIST_FLAG.sub(
        lambda m: m.group(0)[:-len(m.group(5))] + _mask_value(m.group(5)) if is_sensitive_name(m.group(3)) else m.group(0),
        text)
    text = _INLINE_FLAG.sub(
        lambda m: m.group(1) + m.group(3) + _mask_value(m.group(4)) if is_sensitive_name(m.group(2)) else m.group(0),
        text)

    def _pair(m: re.Match) -> str:
        value = m.group(4)
        if not is_sensitive_name(m.group(2)) or value.strip("'\"") == MASK:
            return m.group(0)
        return m.group(1) + m.group(2) + m.group(1) + m.group(3) + _mask_value(value)
    return _PAIR.sub(_pair, text)


def _redact_script(text: str, redacted: str) -> str:
    """Tokenise a shell/code string and redact it as argv; remove every masked token's original text."""
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        return redacted
    if len(tokens) < 2:
        return redacted
    for orig, new in zip(tokens, redact_argv(tokens, _depth=1)):
        if new == orig or not orig:
            continue
        secret = orig[len(new) - len(MASK):] if new.endswith(MASK) and orig.startswith(new[:-len(MASK)]) else orig
        if secret:
            redacted = re.sub(r"(?<![\w-])" + re.escape(secret) + r"(?![\w-])", MASK, redacted)
            redacted = redacted.replace(secret, MASK) if len(secret) >= 6 else redacted
    return redacted


def _looks_like_long_flag(arg: str) -> bool:
    return arg.startswith("--") and len(arg) > 2


def redact_argv(argv: list[str], _depth: int = 0) -> list[str]:
    out: list[str] = []
    pending = False  # previous arg was a sensitive option without "=": this arg is its value
    tool = os.path.basename(argv[0]) if argv else ""
    for idx, arg in enumerate(argv):
        base = os.path.basename(arg)
        if idx and base in _KNOWN_TOOLS:
            tool = base
        if pending:
            pending = False
            if not _looks_like_long_flag(arg):
                out.append(MASK)
                continue
        glued = _TOOL_GLUED.get(tool)
        if glued and arg.startswith(glued) and len(arg) > len(glued) and not arg.startswith("--"):
            out.append(glued + MASK)
            continue
        m = _FLAG.match(arg)
        if m:
            flag = m.group(1) + m.group(2)
            if flag in _TOOL_VALUE_FLAGS.get(tool, ()) and not m.group(3):
                out.append(arg)
                pending = True
                continue
            if is_sensitive_name(m.group(2)) or (m.group(3) and flag in _TOOL_EQ_FLAGS.get(tool, ())):
                if m.group(3):
                    out.append(f"{flag}={MASK}")
                else:
                    out.append(arg)
                    pending = True
                continue
        red = redact_text(arg)
        if _depth == 0 and any(c.isspace() for c in arg):
            red = _redact_script(arg, red)
        out.append(red)
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
