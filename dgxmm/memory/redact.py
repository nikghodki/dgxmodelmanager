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

_SENSITIVE = re.compile(r"key|token|secret|pass|pwd|auth|credential|creds|cookie", re.I)
_EXACT_SENSITIVE = frozenset({"pw", "sig", "signature", "cred"})
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
_PAIR = re.compile(r"(?<![\w.-])([\"']?)((?=[\w.-]" + _N + r"?(?i:key|token|secret|pass|pw|auth|cred|cookie|sig))"
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


def _redact_script(text: str, depth: int, base: frozenset[str] = frozenset()) -> str:
    """Redact a shell/code string: tokenise it (with source spans), redact the tokens as argv (tool options,
    sensitive flags, nested strings one level deeper) and rewrite the changed spans, then apply the text rules.
    The result must be a fixed point of redaction (checked once); if not, every changed token span is masked."""
    toks = shell_tokens(text)
    if len(toks) < 2:
        return redact_text(text, depth)
    values = [v for v, _, _ in toks]
    red = redact_argv(values, _depth=depth + 1, _base=base)
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
    if redact_argv(check, _depth=MAX_SCRIPT_DEPTH, _check=True, _base=base) != check:
        result = rewrite(False)
    return result


def _looks_like_long_flag(arg: str) -> bool:
    return arg.startswith("--") and len(arg) > 2


_WRAPPERS = frozenset({"docker", "podman", "nerdctl", "kubectl", "oc", "crictl", "ctr", "ssh", "su", "runuser",
                       "nsenter", "chroot", "unshare", "systemd-run", "machinectl", "lxc", "incus", "flatpak-spawn"})
_CMD_VALUE_OPTS = frozenset({"--entrypoint"})  # docker run --entrypoint redis-cli: this option's value IS a command
_TOOL_SCRIPT_FLAGS = {"lftp": {"-e", "-c"}}     # lftp -e 'open -u user,S h': the value is a script for the tool
# tools with their own secret options: once one is active, later words never switch the context
_OWN_FLAG_TOOLS = frozenset(set(_TOOL_VALUE_FLAGS) | set(_TOOL_GLUED) | set(_TOOL_EQ_FLAGS) | {"htpasswd"}
                            | set(_TOOL_SCRIPT_FLAGS))


class _Argv:
    """Walks one argv (or a script's words), tracking which tool(s) the current simple command runs.

    The tool is taken at command positions only (start, after shell operators/keywords, after a prefix command and
    its options/operands). Mid-command, a known tool's basename switches context only from a wrapper or an unknown
    tool (docker exec c redis-cli, ssh host mysql, kubectl exec p -- redis-cli), never from a tool with its own secret
    options, never for an option value (the word after an option, or after "="); a switch keeps the union of rules."""

    def __init__(self, depth: int, check: bool, base: frozenset[str] = frozenset()):
        self.depth, self.check, self.base = depth, check, base
        self.out: list[str] = []
        self.mask_rest = False
        self.after_opt = False   # previous word was an option: this word may be its value
        self.cmd_value = False   # previous word was an option whose value is a command (--entrypoint)
        self.prev = ""
        self._reset()

    def _reset(self) -> None:
        self.pending = False     # previous arg was a sensitive option without "=": this arg is its value
        self.tools: set[str] = set(self.base)
        self.cmd_pos = True      # this word names the command
        self.prefix: list | None = None  # [value_opts, operands_left, expecting, name] while skipping a prefix
        self.ht = {"b": False, "n": False, "pos": 0}  # htpasswd -b FILE USER PASSWORD / -nb USER PASSWORD
        self.kv_mode = False     # aws configure set / git config / npm config set: NAME VALUE positionals
        self.script_next = False  # previous arg was lftp -e/-c: this arg is a script for the tool

    def _state(self) -> dict:
        return {k: getattr(self, k) for k in ("pending", "tools", "cmd_pos", "prefix", "ht", "kv_mode",
                                              "script_next", "mask_rest")}

    def _set_tool(self, base: str, add: bool) -> None:
        self.tools = (self.tools | {base}) if add else (set(self.base) | {base})
        self.ht = {"b": False, "n": False, "pos": 0}

    def _can_switch(self, arg: str, base: str) -> bool:
        if base not in _KNOWN_TOOLS or base in self.tools or self.tools & _OWN_FLAG_TOOLS:
            return False
        if arg.startswith("-") or "=" in arg:
            return False
        if self.cmd_value:
            return True
        return not (self.after_opt and self.tools & _WRAPPERS)  # docker run --name mongo: an option value

    def feed(self, arg: str) -> None:
        self._feed(arg)
        self.cmd_value = arg in _CMD_VALUE_OPTS
        self.after_opt = len(arg) > 1 and arg[0] == "-" and arg != "--" and "=" not in arg
        self.prev = arg

    def _split(self, arg: str, lead: str) -> None:
        """env -S 'cmd args' (and shebang "-S cmd args"): the value is split into words and run as a command;
        the words' end state (tool, pending option) carries over to the following arguments."""
        value = arg[len(lead):]
        if self.depth >= MAX_SCRIPT_DEPTH:
            self.out.append(lead + MASK)
            self.mask_rest = True
            return
        sub = _Argv(self.depth + 1, self.check, self.base)
        for v, _, _ in shell_tokens(value):
            sub.feed(v)
        text = redact_text(value, MAX_SCRIPT_DEPTH) if self.check else _redact_script(value, self.depth, self.base)
        self.out.append(lead + text)
        for k, v in sub._state().items():
            setattr(self, k, v)

    def _feed(self, arg: str) -> None:
        out, depth = self.out, self.depth
        if self.mask_rest:
            out.append(MASK)
            return
        if arg and all(c in _SHELL_OP_CHARS for c in arg):
            out.append(arg)
            self._reset()
            return
        if self.prefix is not None:
            value_opts, operands, expecting, name = self.prefix
            if expecting == "split":
                self._split(arg, "")
                return
            if expecting:
                self.prefix[2] = False
                out.append(redact_text(arg, depth))
                return
            if arg == "--":
                out.append(arg)
                return
            if len(arg) > 1 and arg[0] == "-":
                if name == "env" and arg in ("-S", "--split-string"):
                    self.prefix[2] = "split"
                    out.append(arg)
                    return
                if name == "env" and (arg.startswith("--split-string=") or (arg.startswith("-S") and len(arg) > 2)):
                    self._split(arg, "--split-string=" if arg.startswith("--") else "-S")
                    return
                self.prefix[2] = arg.split("=", 1)[0] in value_opts and "=" not in arg
                out.append(redact_text(arg, depth))
                return
            if _ASSIGNMENT.match(arg):  # env VAR=value
                out.append(redact_text(arg, depth))
                return
            if operands > 0:  # timeout DURATION, chrt PRIORITY, taskset MASK
                self.prefix[1] -= 1
                out.append(redact_text(arg, depth))
                return
            self.prefix, self.cmd_pos = None, True
        if self.script_next:
            self.script_next = False
            if self.check:
                out.append(redact_text(arg, MAX_SCRIPT_DEPTH))
            elif depth >= MAX_SCRIPT_DEPTH:
                out.append(MASK)
            else:
                out.append(_redact_script(arg, depth, frozenset(self.base | (self.tools & set(_TOOL_SCRIPT_FLAGS)))))
            return
        base = os.path.basename(arg)
        if self.cmd_pos and not self.pending:
            if arg in _SHELL_KEYWORDS:
                out.append(arg)
                return
            if _ASSIGNMENT.match(arg):  # VAR=value before the command: still in command position
                out.append(redact_text(arg, depth))
                return
            self.cmd_pos = False
            if base in _PREFIX_COMMANDS:
                spec = _PREFIX_COMMANDS[base]
                self.prefix = [spec[0], spec[1], False, base]
                out.append(arg)
                return
            if not arg.startswith("-"):
                self._set_tool(base, add=False)
                if not any(c.isspace() for c in arg):
                    out.append(redact_text(arg, depth))
                    return
        elif not self.pending and self._can_switch(arg, base):
            self._set_tool(base, add=True)
            if not any(c.isspace() for c in arg):
                out.append(redact_text(arg, depth))
                return
        if arg.startswith("--entrypoint=") and self.tools & _WRAPPERS:
            ep = os.path.basename(arg.split("=", 1)[1])
            if ep in _KNOWN_TOOLS and not self.tools & _OWN_FLAG_TOOLS:
                self.tools.add(ep)
        tools = self.tools
        if tools & _LOGIN_TOOLS and arg == "login" and not (self.after_opt and tools & _WRAPPERS):
            tools.add("docker-login")
        if self.pending:
            self.pending = False
            if not _looks_like_long_flag(arg):
                out.append(MASK)
                return
        if (arg == "set" and self.prev in ("configure", "config")) or (arg == "config" and "git" in tools):
            self.kv_mode = True
            out.append(arg)
            return
        if self.kv_mode and not arg.startswith("-") and "=" not in arg and is_sensitive_name(arg):
            out.append(redact_text(arg, depth))
            self.pending = True  # aws configure set aws_secret_access_key S, git config user.password S
            return
        if "htpasswd" in tools:
            ht = self.ht
            if re.fullmatch(r"-[A-Za-z]{1,8}", arg):
                ht["b"] |= "b" in arg
                ht["n"] |= "n" in arg
            else:
                ht["pos"] += 1
                if ht["b"] and ht["pos"] >= (2 if ht["n"] else 3):
                    out.append(MASK)
                    return
        glued = next((g for t in tools for g in _TOOL_GLUED.get(t, ()) if arg.startswith(g) and len(arg) > len(g)),
                     None)
        if glued and not arg.startswith("--"):
            out.append(glued + MASK)
            return
        m = _FLAG.match(arg)
        if m:
            flag = m.group(1) + m.group(2)
            if not m.group(3):
                if any(flag in _TOOL_SCRIPT_FLAGS.get(t, ()) for t in tools):
                    out.append(arg)
                    self.script_next = True
                    return
                if any(flag in _TOOL_VALUE_FLAGS.get(t, ()) for t in tools):
                    out.append(arg)
                    self.pending = True
                    return
            if is_sensitive_name(m.group(2)) or (
                    m.group(3) == "=" and any(flag in _TOOL_EQ_FLAGS.get(t, ()) for t in tools)):
                if m.group(3):
                    out.append(f"{flag}{m.group(3)}{MASK}")
                else:
                    out.append(arg)
                    self.pending = True
                return
        kv = _KV_ARG.match(arg)
        if kv and kv.group(2) == "=" and is_sensitive_name(kv.group(1)):
            out.append(kv.group(1) + "=" + MASK)  # LITELLM_MASTER_KEY=S (names of any length)
            return
        if any(c.isspace() for c in arg):
            if self.check:
                out.append(redact_text(arg, MAX_SCRIPT_DEPTH))
            elif depth < MAX_SCRIPT_DEPTH:
                out.append(_redact_script(arg, depth))
            else:
                out.append(MASK)  # nested deeper than MAX_SCRIPT_DEPTH: fail closed
        else:
            out.append(redact_text(arg, depth))


def redact_argv(argv: list[str], _depth: int = 0, _check: bool = False,
                _base: frozenset[str] = frozenset()) -> list[str]:
    """Redact an argv. `_check` (fixed-point check) treats nested scripts as text instead of masking them;
    `_base` are tools whose rules apply to every command (an lftp -e script)."""
    w = _Argv(_depth, _check, _base)
    for arg in argv:
        w.feed(arg)
    return w.out


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
        total += len(arg) + 1  # an empty argument still costs a separator
    return out, cut


def format_command(argv: list[str]) -> tuple[str, bool]:
    """Redacted, shell-quoted command line, truncated to MAX_COMMAND_BYTES. Returns (command, truncated)."""
    capped, cut = _cap_input(argv)
    text = " ".join(_quote(a) for a in redact_argv(capped))
    raw = text.encode()
    if len(raw) <= MAX_COMMAND_BYTES:
        return text, cut
    return raw[:MAX_COMMAND_BYTES].decode("utf-8", "ignore") + "…", True
