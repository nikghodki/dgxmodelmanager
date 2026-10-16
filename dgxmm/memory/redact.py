"""Redact secrets from process command lines before they leave the server (spec §5).

Fail closed — over-redaction is the safe failure:
- a flag or KEY=VALUE whose name contains key/token/secret/pass/pwd/auth/credential/cookie (or is sig/signature)
  has its value masked, except for a few exact non-secret names (``tokenizer…``, count flags such as
  ``--max-num-batched-tokens``, ``passthrough``/``bypass``…);
- after a sensitive flag without ``=`` the next argument is masked unless it starts with ``--``;
- header-like keys (Authorization, Proxy-Authorization, Cookie, X-…-Key/Token) are masked to the end of the header;
- URL userinfo, Bearer/Basic, and known token shapes (hf_, sk-, ghp_/github_pat_, AKIA…, xox?-, JWT) anywhere;
- tool-specific options (``mysql -pS``, ``redis-cli -a S``, ``curl -u u:S``, ``docker login -p S``…), with the tool
  found at command position (after shell operators/keywords and prefix commands such as ``sudo -u x``, ``timeout 10``);
- shell/code strings (``sh -c``, ``python -c``, ``perl -e``) are tokenised with source spans and redacted as argv,
  recursively up to MAX_SCRIPT_DEPTH (deeper strings are masked whole); list/tuple literals are redacted as argv.

Cost is bounded (any local user controls argv, and the budget loop redacts the top processes every tick): every
regex quantifier that can rescan is bounded, each argument is cut at MAX_TOKEN_CHARS and the whole command at
MAX_INPUT_CHARS before any rule runs; cuts drop the trailing partial word so no fragment of a secret survives.
"""
from __future__ import annotations

import os
import re
import shlex

MASK = "***"
MAX_COMMAND_BYTES = 4096   # output cap (the API contract)
MAX_TOKEN_CHARS = 2048     # per-argument input cap before redaction
MAX_INPUT_CHARS = 8192     # whole-command input cap before redaction
TRUNC = "…[truncated]"
MAX_SCRIPT_DEPTH = 3       # nested sh -c "sh -c '…'" strings are redacted this many levels deep; deeper: masked
_N = "{0,64}"              # bound for name-like runs inside patterns (keeps every rule linear)

_SENSITIVE = re.compile(r"key|token|secret|pass|pwd|auth|credential|cookie", re.I)
_EXACT_SENSITIVE = frozenset({"pw", "sig", "signature"})
_COUNT_WORDS = frozenset({"max", "min", "num", "batched", "new", "prompt", "draft", "speculative", "budget", "output",
                          "input", "context", "total", "lookahead", "reasoning"})
_SAFE_PASS_WORDS = frozenset({"passthrough", "bypass", "compass", "passive", "passes", "multipass"})
_FLAG = re.compile(r"^(--?)([A-Za-z][\w.-]*)(?:([=:])(.*))?$", re.S)  # --password=S and --password:S

# Tool-specific options; the tool is the basename of the word at command position
_TOOL_VALUE_FLAGS = {  # option → the NEXT argument is a secret
    "redis-cli": {"-a"}, "curl": {"-u", "--user", "-U", "--proxy-user"}, "sshpass": {"-p"},
    "wget": {"--user", "--http-user"}, "openssl": {"-k", "-psk"}, "docker-login": {"-p"},
    "mongosh": {"-p"}, "mongo": {"-p"}, "lftp": {"-u", "--user"}, "smbclient": {"-U", "--user"},
}
_TOOL_EQ_FLAGS = {"curl": {"--user", "--proxy-user"}, "smbclient": {"--user"}, "lftp": {"--user"}}  # --user=u:S
_TOOL_GLUED: dict[str, tuple[str, ...]] = {  # option glued to its secret value (mysql -pS, curl -uu:S)
    **{t: ("-p",) for t in ("mysql", "mysqldump", "mysqladmin", "mysqlimport", "mysqlcheck", "mariadb", "mariadb-dump",
                            "mariadb-admin", "mysqlsh")},
    "sshpass": ("-p",), "curl": ("-u", "-U"), "redis-cli": ("-a",), "docker-login": ("-p",), "smbclient": ("-U",),
    "mongosh": ("-p",), "mongo": ("-p",),
}
_LOGIN_TOOLS = frozenset({"docker", "podman", "nerdctl", "buildah", "skopeo", "helm", "oras"})  # `<tool> login`
_KNOWN_TOOLS = set(_TOOL_VALUE_FLAGS) | set(_TOOL_GLUED) | {"htpasswd"} | _LOGIN_TOOLS
# Prefix commands run their operand as a command: name → (options that take a value, operands before the command)
_PREFIX_COMMANDS: dict[str, tuple[frozenset[str], int]] = {
    "env": (frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}), 0),
    "sudo": (frozenset({"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T", "--user", "--group", "--host",
                        "--prompt", "--chdir", "--role", "--type", "--other-user", "--command-timeout"}), 0),
    "doas": (frozenset({"-u", "-C"}), 0),
    "timeout": (frozenset({"-s", "-k", "--signal", "--kill-after"}), 1),
    "nice": (frozenset({"-n", "--adjustment"}), 0),
    "nohup": (frozenset(), 0), "exec": (frozenset({"-a"}), 0), "setsid": (frozenset(), 0),
    "time": (frozenset({"-f", "-o", "--format", "--output"}), 0), "command": (frozenset(), 0),
    "builtin": (frozenset(), 0),
    "stdbuf": (frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}), 0),
    "ionice": (frozenset({"-c", "-n", "-p", "-P", "-u", "--class", "--classdata", "--pid", "--pgid", "--uid"}), 0),
    "chrt": (frozenset({"-T", "-P", "-D", "--sched-runtime", "--sched-period", "--sched-deadline"}), 1),
    "taskset": (frozenset(), 1),
    "xargs": (frozenset({"-n", "-I", "-i", "-P", "-d", "-s", "-L", "-l", "-E", "-e", "-a", "--max-args", "--replace",
                         "--max-procs", "--delimiter", "--arg-file", "--max-chars", "--max-lines", "--eof"}), 0),
}
_SHELL_KEYWORDS = frozenset({"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}", "fi", "done",
                             "esac", "in"})
_SHELL_OP_CHARS = ";|&()\n"
_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
_KV_ARG = re.compile(r"^([A-Za-z_][\w.-]*)([=:])(.+)$", re.S)  # whole-argument NAME=VALUE (any name length)

_URL_SCHEME = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{0,31}://")
_RUN_END = re.compile(r"[\s'\"]")
_HEADER = re.compile(r"(?i)(?<![\w-])((?:proxy-)?authorization|(?:set-)?cookie|x-[\w-]" + _N + r"?(?:key|token|auth|secret)"
                     r"[\w-]" + _N + r"|api-?key)(\s{0,8}:\s{0,8})([^'\"\n]*)")
_AUTH_SCHEME = re.compile(r"(?i)\b(bearer|basic|digest)(\s{1,8})[A-Za-z0-9._~+/=:-]{6,}")
_TOKEN_SHAPES = [re.compile(p) for p in (
    r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}",
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}",
    r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}",
    r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}",
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])",
    r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}",
    r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}",
)]
_QUOTED = r"\"(?:[^\"\\]|\\.){0,2048}\"|'(?:[^'\\]|\\.){0,2048}'"  # bound = MAX_TOKEN_CHARS
_VALUE = r"([bBrRuUfF]{0,2}(?:" + _QUOTED + r")|[^\s&'\",;(){}\[\]]+)"
_INLINE_FLAG = re.compile(r"(?<![\w.-])(--?([A-Za-z][\w.-]" + _N + r"))(=|:|[ \t]{1,8})(?!--)" + _VALUE)
# ['--api-key', 'S'], setdefault('HF_TOKEN', 'S'), ("password", "S")
_LIST_FLAG = re.compile(r"([\"'])(-{0,2}([A-Za-z_][\w.-]" + _N + r"))\1(\s{0,8},\s{0,8})([bBrRuUfF]{0,2}(?:" + _QUOTED + r"))")
_SUBSCRIPT = re.compile(r"\[\s{0,8}[bBrRuU]{0,2}([\"']?)([A-Za-z_][\w.-]" + _N + r")\1\s{0,8}\](\s{0,8}=\s{0,8})" + _VALUE)
_SEQ_LITERAL = re.compile(r"[\[(]([^\[\]()]{0,4096})[\])]")  # [...] or (...) without nesting
_SEQ_ITEM = re.compile(r"([bBrRuUfF]{0,2})(" + _QUOTED + r")|[^,'\"]+|,")
_PASS_ARG = re.compile(r"(?<![\w-])pass:[^\s'\"]+")  # openssl -passin pass:S
# Only keys that might be sensitive start a match (bounded lookahead), so a harmless pair never swallows the text
# after it; the callback applies the exact rule (allowlist included).
_PAIR = re.compile(r"(?<![\w.-])([\"']?)((?=[\w.-]" + _N + r"?(?i:key|token|secret|pass|pw|auth|credential|cookie|sig))"
                   r"[A-Za-z_][\w.-]" + _N + r")\1(\s{0,8}[=:]\s{0,8})" + _VALUE)
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


def _unquote(q: str) -> str:
    return q[1:-1].replace("\\" + q[0], q[0])


def _seq_literal(m: re.Match, depth: int) -> str:
    """A list/tuple/call-argument sequence with >= 2 string items is an argv (subprocess.run([...]),
    Perl system(...)): redact it item by item. Non-string items (variables, numbers) are kept as they are."""
    inner = m.group(1)
    items: list[tuple[re.Match | None, str]] = []
    pos = 0
    for it in _SEQ_ITEM.finditer(inner):
        if it.start() != pos:
            return m.group(0)  # stray quote: not a clean sequence
        pos = it.end()
        if it.group(2):
            items.append((it, _unquote(it.group(2))))
        elif it.group(0) != "," and it.group(0).strip():
            items.append((None, it.group(0).strip()))
    if pos != len(inner) or sum(1 for it, _ in items if it is not None) < 2:
        return m.group(0)
    values = [v for _, v in items]
    red = redact_argv(values, _depth=depth + 1)
    out, last = [], 0
    for (it, v), r in zip(items, red):
        if it is not None and r != v:
            q = it.group(2)[0]
            out += [inner[last:it.start(2)], q + r.replace(q, "\\" + q) + q]
            last = it.end(2)
    return m.group(0)[0] + "".join(out) + inner[last:] + m.group(0)[-1]


def _redact_userinfo(text: str) -> str:
    """scheme://…@ → scheme://***@, masking up to the LAST @ of the URL's non-space run (over-redaction accepted).
    Linear: each run's end and last @ are computed once, however many schemes the run contains."""
    out, last = [], 0
    run_end, run_at = -1, -1
    for m in _URL_SCHEME.finditer(text):
        if m.start() < last:
            continue
        if m.end() > run_end:
            e = _RUN_END.search(text, m.end())
            run_end = e.start() if e else len(text)
            run_at = text.rfind("@", m.end(), run_end)
        if run_at >= m.end():
            out += [text[last:m.end()], MASK]
            last = run_at
    return "".join(out) + text[last:] if out else text


def redact_text(text: str, _depth: int = 0) -> str:
    text = _redact_userinfo(text)
    text = _HEADER.sub(lambda m: m.group(1) + m.group(2) + (MASK if m.group(3).strip() else m.group(3)), text)
    text = _AUTH_SCHEME.sub(lambda m: m.group(1) + m.group(2) + MASK, text)
    for rx in _TOKEN_SHAPES:
        text = rx.sub(MASK, text)
    text = _PASS_ARG.sub("pass:" + MASK, text)
    if _depth < MAX_SCRIPT_DEPTH:
        text = _SEQ_LITERAL.sub(lambda m: _seq_literal(m, _depth), text)
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


def shell_tokens(text: str) -> list[tuple[str, int, int]]:
    """Lenient POSIX-shell tokenizer: (value, start, end) per word, with quotes/backslashes resolved and control
    operators (; | & ( ) newline) as their own tokens. Unterminated quotes run to the end (never raises). Linear."""
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
    sensitive flags, nested strings one level deeper) and rewrite the changed spans, then apply the text rules.
    The result must be a fixed point of redaction (checked once); if not, every changed token span is masked."""
    toks = shell_tokens(text)
    if len(toks) < 2:
        return redact_text(text, depth)
    values = [v for v, _, _ in toks]
    red = redact_argv(values, _depth=depth + 1)
    text_only = [redact_text(v, depth + 1) == r for v, r in zip(values, red)]

    def rewrite(structural_only: bool) -> str:
        out, last, secrets = [], 0, []
        for (v, s, e), r, plain in zip(toks, red, text_only):
            if r == v or (structural_only and plain):
                continue
            out += [text[last:s], _quote(r)]
            last = e
            secrets.append(_secret_of(v, r))
        result = redact_text("".join(out) + text[last:], depth)
        for sec in secrets:
            if len(sec) >= 6:
                result = result.replace(sec, MASK)
        return result

    result = rewrite(True)
    check = [v for v, _, _ in shell_tokens(result)]
    if redact_argv(check, _depth=MAX_SCRIPT_DEPTH, _check=True) != check:
        result = rewrite(False)
    return result


def _looks_like_long_flag(arg: str) -> bool:
    return arg.startswith("--") and len(arg) > 2


def redact_argv(argv: list[str], _depth: int = 0, _check: bool = False) -> list[str]:
    """Redact an argv. `_check` (fixed-point check) treats nested scripts as text instead of masking them."""
    out: list[str] = []
    pending = False   # previous arg was a sensitive option without "=": this arg is its value
    tool = ""
    cmd_pos = True    # this word names the command (start, after ; | && ( { do then … or a prefix command)
    prefix: list | None = None  # [value_opts, operands_left, expecting_value] while skipping a prefix command's args
    ht = {"b": False, "n": False, "pos": 0}  # htpasswd -b FILE USER PASSWORD / -nb USER PASSWORD
    for idx, arg in enumerate(argv):
        if arg and all(c in _SHELL_OP_CHARS for c in arg):
            out.append(arg)
            pending, cmd_pos, prefix, tool = False, True, None, ""
            continue
        if prefix is not None:
            value_opts, operands, expecting = prefix
            if expecting:
                prefix[2] = False
                out.append(redact_text(arg, _depth))
                continue
            if arg == "--":
                out.append(arg)
                continue
            if len(arg) > 1 and arg[0] == "-":
                prefix[2] = arg.split("=", 1)[0] in value_opts and "=" not in arg
                out.append(redact_text(arg, _depth))
                continue
            if _ASSIGNMENT.match(arg):  # env VAR=value
                out.append(redact_text(arg, _depth))
                continue
            if operands > 0:  # timeout DURATION, chrt PRIORITY, taskset MASK
                prefix[1] -= 1
                out.append(redact_text(arg, _depth))
                continue
            prefix, cmd_pos = None, True
        base = os.path.basename(arg)
        if cmd_pos and not pending:
            if arg in _SHELL_KEYWORDS:
                out.append(arg)
                continue
            if _ASSIGNMENT.match(arg):  # VAR=value before the command: still in command position
                out.append(redact_text(arg, _depth))
                continue
            cmd_pos = False
            if base in _PREFIX_COMMANDS:
                spec = _PREFIX_COMMANDS[base]
                prefix = [spec[0], spec[1], False]
                out.append(arg)
                continue
            if not arg.startswith("-"):
                tool, ht = base, {"b": False, "n": False, "pos": 0}
                if not any(c.isspace() for c in arg):
                    out.append(redact_text(arg, _depth))
                    continue
        elif base in _KNOWN_TOOLS and base != tool and not pending:  # e.g. docker exec c /usr/bin/redis-cli
            tool, ht = base, {"b": False, "n": False, "pos": 0}
        if tool in _LOGIN_TOOLS and arg == "login":
            tool = "docker-login"
        if pending:
            pending = False
            if not _looks_like_long_flag(arg):
                out.append(MASK)
                continue
        if tool == "htpasswd" and idx:
            if re.fullmatch(r"-[A-Za-z]{1,8}", arg):
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
        kv = _KV_ARG.match(arg)
        if kv and kv.group(2) == "=" and is_sensitive_name(kv.group(1)):
            out.append(kv.group(1) + "=" + MASK)  # LITELLM_MASTER_KEY=S (names of any length)
            continue
        if any(c.isspace() for c in arg):
            if _check:
                out.append(redact_text(arg, MAX_SCRIPT_DEPTH))
            elif _depth < MAX_SCRIPT_DEPTH:
                out.append(_redact_script(arg, _depth))
            else:
                out.append(MASK)  # nested deeper than MAX_SCRIPT_DEPTH: fail closed
        else:
            out.append(redact_text(arg, _depth))
    return out


def _quote(arg: str) -> str:
    return arg if _SHELL_SAFE.fullmatch(arg) else shlex.quote(arg)


def _cap_input(argv: list[str]) -> tuple[list[str], bool]:
    """Bound redaction work: cut each argument at MAX_TOKEN_CHARS and the command at MAX_INPUT_CHARS. A cut drops
    the trailing partial word, so a secret straddling the boundary never shows even partly."""
    out: list[str] = []
    total, cut = 0, False
    for arg in argv:
        room = min(MAX_TOKEN_CHARS, MAX_INPUT_CHARS - total)
        if room <= 0:
            out.append(TRUNC)
            return out, True
        if len(arg) > room:
            head = arg[:room]
            head = head[:max(head.rfind(" "), head.rfind("\t"), head.rfind("\n"), 0)]
            out.append(head + TRUNC if head else TRUNC)
            cut = True
            total += room
            continue
        out.append(arg)
        total += len(arg)
    return out, cut


def format_command(argv: list[str]) -> tuple[str, bool]:
    """Redacted, shell-quoted command line, truncated to MAX_COMMAND_BYTES. Returns (command, truncated)."""
    capped, cut = _cap_input(argv)
    text = " ".join(_quote(a) for a in redact_argv(capped))
    raw = text.encode()
    if len(raw) <= MAX_COMMAND_BYTES:
        return text, cut
    return raw[:MAX_COMMAND_BYTES].decode("utf-8", "ignore") + "…", True
