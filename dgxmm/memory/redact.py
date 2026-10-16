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
_FLAG = re.compile(r"^(--?)([A-Za-z][\w.-]*)(?:([=:])(.*))?$", re.S)  # --password=S and --password:S

# Tool-specific options (argv[0] basename, or a later token naming the tool inside sh -c)
_TOOL_VALUE_FLAGS = {  # option → the NEXT argument is a secret
    "redis-cli": {"-a"}, "curl": {"-u", "--user", "-U", "--proxy-user"}, "sshpass": {"-p"},
    "wget": {"--user", "--http-user"}, "openssl": {"-k"},
}
_TOOL_EQ_FLAGS = {"curl": {"--user", "--proxy-user"}}  # --user=u:S
_TOOL_GLUED: dict[str, tuple[str, ...]] = {  # option glued to its secret value (mysql -pS, curl -uu:S)
    **{t: ("-p",) for t in ("mysql", "mysqldump", "mysqladmin", "mysqlimport", "mysqlcheck", "mariadb", "mariadb-dump",
                            "mariadb-admin", "mysqlsh")},
    "sshpass": ("-p",), "curl": ("-u", "-U"), "redis-cli": ("-a",),
}
_KNOWN_TOOLS = set(_TOOL_VALUE_FLAGS) | set(_TOOL_GLUED) | {"htpasswd"}
# Words that run the following word as a command (tool detection inside sh -c strings)
_PREFIX_COMMANDS = frozenset({"sudo", "exec", "env", "nohup", "time", "nice", "command", "xargs", "timeout"})
_SHELL_OP_CHARS = ";|&()\n"
MAX_SCRIPT_DEPTH = 3  # nested sh -c "sh -c '…'" strings are tokenised and redacted this many levels deep

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
# ['--api-key', 'S'], setdefault('HF_TOKEN', 'S'), ("password", "S")
_LIST_FLAG = re.compile(r"([\"'])(-{0,2}([A-Za-z_][\w.-]*))\1(\s*,\s*)(" + _QUOTED + r")")
_SUBSCRIPT = re.compile(r"\[\s*([\"']?)([A-Za-z_][\w.-]*)\1\s*\](\s*=\s*)" + _VALUE)  # os.environ['API_KEY'] = 'S'
_LIST_LITERAL = re.compile(r"\[([^\[\]]*)\]")
_PASS_ARG = re.compile(r"(?<![\w-])pass:[^\s'\"]+")  # openssl -passin pass:S
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
    text = _PASS_ARG.sub("pass:" + MASK, text)
    text = _LIST_LITERAL.sub(_redact_list_literal, text)
    text = _SUBSCRIPT.sub(
        lambda m: m.group(0)[:-len(m.group(4))] + _mask_value(m.group(4)) if is_sensitive_name(m.group(2)) else m.group(0),
        text)
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


def _redact_list_literal(m: re.Match) -> str:
    """A Python/JSON list of strings is an argv (subprocess.run([...])): redact it item by item."""
    inner = m.group(1)
    items = list(re.finditer(_QUOTED, inner))
    if len(items) < 2 or re.sub(_QUOTED, "", inner).strip(" \t,"):
        return m.group(0)
    values = [it.group(0)[1:-1].replace("\\" + it.group(0)[0], it.group(0)[0]) for it in items]
    red = redact_argv(values, _depth=MAX_SCRIPT_DEPTH)
    out, last = [], 0
    for it, v, r in zip(items, values, red):
        if r != v:
            q = it.group(0)[0]
            out += [inner[last:it.start()], q + r.replace(q, "\\" + q) + q]
            last = it.end()
    return "[" + "".join(out) + inner[last:] + "]"


def shell_tokens(text: str) -> list[tuple[str, int, int]]:
    """Lenient POSIX-shell tokenizer: (value, start, end) per word, with quotes/backslashes resolved and control
    operators (; | & ( ) newline) as their own tokens. Unterminated quotes run to the end (never raises)."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace() and c != "\n":
            i += 1
            continue
        start = i
        if c in _SHELL_OP_CHARS:
            while i < n and text[i] in _SHELL_OP_CHARS:
                i += 1
            out.append((text[start:i], start, i))
            continue
        buf: list[str] = []
        while i < n and not text[i].isspace() and text[i] not in _SHELL_OP_CHARS:
            c = text[i]
            if c == "'":
                j = text.find("'", i + 1)
                j = n if j < 0 else j
                buf.append(text[i + 1:j])
                i = j + 1
            elif c == '"':
                i += 1
                while i < n and text[i] != '"':
                    if text[i] == "\\" and i + 1 < n and text[i + 1] in '"\\$`\n':
                        i += 1
                    buf.append(text[i])
                    i += 1
                i += 1
            elif c == "\\":
                buf.append(text[i + 1:i + 2])
                i += 2
            else:
                buf.append(c)
                i += 1
        out.append(("".join(buf), start, min(i, n)))
    return out


def _secret_of(orig: str, new: str) -> str:
    if new.endswith(MASK) and orig.startswith(new[:-len(MASK)]):
        return orig[len(new) - len(MASK):]
    return orig


def _redact_script(text: str, depth: int) -> str:
    """Redact a shell/code string: tokenise it (with source spans), redact the tokens as argv (tool options,
    sensitive flags, nested sh -c up to MAX_SCRIPT_DEPTH) and rewrite the changed spans, then apply the text rules.
    The result must be a fixed point of redaction; if it is not, every changed token span is masked (fail closed)."""
    toks = shell_tokens(text)
    if len(toks) < 2:
        return redact_text(text)
    values = [v for v, _, _ in toks]
    red = redact_argv(values, _depth=depth + 1)

    def rewrite(spans_only_structural: bool) -> str:
        out, last, secrets = [], 0, []
        for (v, s, e), r in zip(toks, red):
            if r == v or (spans_only_structural and r == redact_text(v)):
                continue
            out += [text[last:s], _quote(r)]
            last = e
            secrets.append(_secret_of(v, r))
        result = redact_text("".join(out) + text[last:])
        for sec in secrets:
            if len(sec) >= 6:
                result = result.replace(sec, MASK)
        return result

    result = rewrite(True)
    check = [v for v, _, _ in shell_tokens(result)]
    if redact_argv(check, _depth=depth + 1) != check:
        result = rewrite(False)
    return result


def _looks_like_long_flag(arg: str) -> bool:
    return arg.startswith("--") and len(arg) > 2


def redact_argv(argv: list[str], _depth: int = 0) -> list[str]:
    out: list[str] = []
    pending = False  # previous arg was a sensitive option without "=": this arg is its value
    tool = os.path.basename(argv[0]) if argv else ""
    new_cmd = False  # previous token was a shell operator / prefix command: this token names the command
    ht = {"b": False, "n": False, "pos": 0}  # htpasswd -b FILE USER PASSWORD / -nb USER PASSWORD
    for idx, arg in enumerate(argv):
        if arg and all(c in _SHELL_OP_CHARS for c in arg):
            out.append(arg)
            pending, new_cmd, tool = False, True, ""
            continue
        base = os.path.basename(arg)
        cmd_pos = (idx == 0 or new_cmd) and not arg.startswith("-")
        if cmd_pos and re.match(r"^\w+=", arg):  # VAR=value before the command: still in command position
            out.append(redact_text(arg))
            continue
        if cmd_pos or ("/" not in arg and base in _KNOWN_TOOLS):
            if base != tool or idx == 0:
                tool, ht = base, {"b": False, "n": False, "pos": 0}
            new_cmd = base in _PREFIX_COMMANDS
            if cmd_pos and not any(c.isspace() for c in arg):
                out.append(redact_text(arg))
                continue
        if pending:
            pending = False
            if not _looks_like_long_flag(arg):
                out.append(MASK)
                continue
        if tool == "htpasswd" and idx:
            if re.fullmatch(r"-[A-Za-z]+", arg):
                ht["b"] |= "b" in arg
                ht["n"] |= "n" in arg
            else:
                ht["pos"] += 1
                if ht["b"] and ht["pos"] >= (2 if ht["n"] else 3):
                    out.append(MASK)
                    continue
        glued = next((g for g in _TOOL_GLUED.get(tool, ()) if arg.startswith(g) and len(arg) > len(g)), None)
        if glued and not arg.startswith("--"):
            out.append(glued + MASK)
            continue
        m = _FLAG.match(arg)
        if m:
            flag = m.group(1) + m.group(2)
            if flag in _TOOL_VALUE_FLAGS.get(tool, ()) and not m.group(3):
                out.append(arg)
                pending = True
                continue
            if is_sensitive_name(m.group(2)) or (m.group(3) == "=" and flag in _TOOL_EQ_FLAGS.get(tool, ())):
                if m.group(3):
                    out.append(f"{flag}{m.group(3)}{MASK}")
                else:
                    out.append(arg)
                    pending = True
                continue
        if _depth < MAX_SCRIPT_DEPTH and any(c.isspace() for c in arg):
            out.append(_redact_script(arg, _depth))
        else:
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
