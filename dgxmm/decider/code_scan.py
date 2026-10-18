"""Static scan of a model repository's Python files (advanced feature 11, remote-code risk). STANDALONE: standard
library only, run as its own process by dgxmm/decider/remote_code.py (``python -I -S code_scan.py <repo dir>
<snapshot dir>``) under CPU / memory / file-size limits. It parses files with ``ast`` and never imports or executes
them. Output: one JSON object on stdout.

Findings (``kind``):
- ``network_import`` / ``process_import`` / ``native_import``: imports of network, process or native-code modules;
- ``process_call``: os.system / os.popen / os.exec* / os.spawn* / subprocess.* / pty.spawn / pty.fork / os.fork,
  asyncio.create_subprocess_*, multiprocessing.* and ProcessPoolExecutor calls;
- ``code_object``: code objects built or patched by hand (types.FunctionType / types.CodeType, ``code.replace(co_*=)``,
  ``__code__``); ``object_walk``: ``__subclasses__`` / ``__globals__`` walks (reaching Popen or os without importing);
- ``native_load``: native code loaded at run time (torch.utils.cpp_extension.load*, torch.ops / torch.classes
  load_library, ctypes CDLL / cdll.LoadLibrary, cffi dlopen);
- ``remote_fetch``: files or code fetched from the Hub at run time (hf_hub_download, snapshot_download,
  get_class_from_dynamic_module, transformers.dynamic_module_utils): another repository's code, never scanned;
- ``external_code``: other code pulled in at run time: ``from_pretrained("org/repo", trust_remote_code=True)`` (any
  call with that literal keyword and a literal repo id), torch.hub.load, pip installs (``import pip``, pip.main, a
  process running pip install);
- ``dynamic_exec``: calls of the builtins eval / exec / compile / __import__, and of the ``code`` module's
  interpreters (InteractiveInterpreter / InteractiveConsole / interact / compile_command);
- ``encoded_exec``: such a call fed by base64 / zlib / marshal / codecs decoding;
- ``file_write_outside``: a file opened for writing (or Path.write_*) at an absolute or home-relative path;
- ``sensitive_path``: a string naming credentials or shell start-up files (~/.ssh, .aws/credentials, .netrc, …);
- ``credential_env``: an environment variable read whose name looks like a credential; ``environment_dump``: the
  whole environment copied or iterated;
- ``long_encoded_literal``: a string of 2000+ base64 / hex characters;
- ``dynamic_import``: importlib.import_module with a computed or risky name, ``sys.modules[...]``, and every
  loader that runs a file by its path, whatever the module name (importlib spec_from_file_location /
  SourceFileLoader / SourcelessFileLoader / ExtensionFileLoader, runpy.run_path / run_module, zipimport);
- ``builtins_access``: ``__builtins__`` / the ``builtins`` module (a way round the eval / exec names);
- ``dynamic_attribute``: getattr / setattr on builtins or a risky module (os, subprocess, socket, …);
- ``obfuscated_name``: a risky module or builtin name assembled from pieces ("so" + "cket", "".join([...]));
- ``unsafe_deserialize``: pickle / dill / marshal / joblib / shelve loads, torch.load without weights_only=True,
  numpy.load with allow_pickle not literally False, pandas.read_pickle (unpickling runs code);
- ``unparseable`` / ``oversized_file``: a file that could not be scanned.

Names are resolved through a per-file alias map built from the file's imports (``import x as y``, ``from x import
y [as z]``), so a renamed call (``from runpy import run_path``, ``hf_hub_download as d``) is matched by its real name.

Run as a script, the process first limits itself (``apply_limits``: CPU, address space, no file writes, no core
dumps) — the parent sets no ``preexec_fn``. ``code_scan.py --limits`` prints the limits it would apply.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

LIMIT_CPU_S = 20
LIMIT_AS_BYTES = 1024**3

MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_FILES = 300
MAX_FINDINGS = 400
LONG_LITERAL = 2000
NETWORK = {"socket", "requests", "urllib", "urllib2", "urllib3", "http", "httpx", "aiohttp", "ftplib", "smtplib",
           "telnetlib", "paramiko", "websocket", "websockets", "pycurl", "xmlrpc", "socketserver"}
PROCESS = {"subprocess", "pty", "pexpect", "multiprocessing"}
NATIVE = {"ctypes", "cffi"}
DYNAMIC = {"eval", "exec", "compile", "__import__"}
DESERIALIZE = {"pickle", "cPickle", "_pickle", "dill", "marshal", "shelve", "joblib", "cloudpickle"}
DESERIALIZE_CALLS = {"load", "loads", "Unpickler", "open"}
RISKY_MODULES = NETWORK | PROCESS | NATIVE | DESERIALIZE | {"os", "sys", "builtins", "importlib", "posix", "runpy"}
RISKY_NAMES = RISKY_MODULES | DYNAMIC | {"system", "popen", "spawn", "fork", "__builtins__", "getattr", "setattr",
                                         "import_module", "check_output", "Popen"}
IMPORT_CALLS = {"importlib.import_module", "import_module", "importlib.__import__"}
# loaders that run a file (or module) chosen by a path: flagged whatever the module name; the value is the
# position of the path argument (shown in the finding)
LOADERS = {"importlib.util.spec_from_file_location": 1, "importlib.machinery.SourceFileLoader": 1,
           "importlib.machinery.SourcelessFileLoader": 1, "importlib.machinery.ExtensionFileLoader": 1,
           "runpy.run_path": 0, "runpy.run_module": 0, "zipimport.zipimporter": 0}
LOADER_HEADS = {"util", "machinery", "runpy", "zipimport"}
LOADER_BARE = {"spec_from_file_location", "SourceFileLoader", "SourcelessFileLoader", "ExtensionFileLoader",
               "zipimporter"}
LOADER_MODULES = {"runpy", "zipimport"}
CODE_INTERP = {"code.InteractiveInterpreter", "code.InteractiveConsole", "code.interact", "code.compile_command",
               "codeop.compile_command", "codeop.Compile", "codeop.CommandCompiler"}
PIP_CALLS = {"pip.main", "pip._internal.main", "pip._internal.cli.main.main"}
PIP_WORD = re.compile(r"(^|[\s/])pip3?(\.exe)?($|\s)")
DECODERS = {("base64", "b64decode"), ("base64", "b32decode"), ("base64", "b16decode"), ("base64", "a85decode"),
            ("base64", "b85decode"), ("base64", "decodebytes"), ("base64", "urlsafe_b64decode"),
            ("zlib", "decompress"), ("marshal", "loads"), ("codecs", "decode"), ("bz2", "decompress"),
            ("lzma", "decompress"), ("gzip", "decompress"), ("binascii", "unhexlify"), ("binascii", "a2b_base64")}
OS_PROCESS = re.compile(r"^(system|popen|popen[234]|exec[lv]p?e?|spawn[lv]p?e?|fork|forkpty|posix_spawnp?)$")
# Credential-like environment variable names, matched per "_"-separated word (D3_GRAPH_TOKENS is a count, HF_TOKEN a
# credential; tuned on the real corpus, docs/decider-evaluation.md).
CRED_WORDS = {"TOKEN", "SECRET", "PASSWORD", "PASSWD", "PASS", "CREDENTIAL", "CREDENTIALS", "APIKEY", "AUTH",
              "COOKIE", "PAT"}
CRED_PAIRS = {("API", "KEY"), ("ACCESS", "KEY"), ("PRIVATE", "KEY"), ("SECRET", "KEY"), ("SESSION", "KEY")}
CRED_PREFIX = ("AWS_", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API", "AZURE_OPENAI", "GITHUB_TOKEN", "GH_TOKEN",
               "GOOGLE_APPLICATION_CREDENTIALS", "ANTHROPIC_API", "SSH_AUTH_SOCK")
SENSITIVE = re.compile(r"(\.ssh(/|$)|authorized_keys|id_rsa|id_ed25519|\.aws/credentials|\.netrc|\.bashrc|"
                       r"\.bash_profile|\.profile$|\.zshrc|/etc/passwd|/etc/shadow|/etc/sudoers|\.docker/config\.json|"
                       r"\.kube/config|\.git-credentials|crontab|/etc/cron)")
ENCODED = re.compile(r"^[A-Za-z0-9+/=_\-\s]+$")
MP_CALLS = {"Process", "Pool", "get_context", "Manager"}
CODE_CALLS = ("FunctionType", "CodeType", "LambdaType")
WALK_ATTRS = {"__subclasses__", "__globals__", "f_globals", "gi_frame", "f_back"}
NATIVE_LOAD = ("cpp_extension.load", "cpp_extension.load_inline", "ops.load_library", "classes.load_library",
               "ctypes.CDLL", "ctypes.PyDLL", "ctypes.WinDLL", "cdll.LoadLibrary", "pydll.LoadLibrary", "ffi.dlopen")
REMOTE_FETCH = ("hf_hub_download", "snapshot_download", "get_class_from_dynamic_module", "get_cached_module_file",
                "cached_download", "hf_hub_url")
REMOTE_MODULES = ("transformers.dynamic_module_utils", "huggingface_hub.file_download")


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _aliases(tree) -> dict[str, str]:
    """Local name -> full dotted name, from every import in the file (``import a.b as c``: c -> a.b; ``import a.b``:
    a -> a; ``from a import b as c``: c -> a.b). Relative imports and ``*`` bind nothing here."""
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    out.setdefault(top, top)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


def _strings(node) -> list[str]:
    """The string literals inside an expression (a process call's arguments)."""
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _folded(node) -> str | None:
    """A string assembled at parse time from literal pieces: "a" + "b", "".join(["a", "b"]); None otherwise."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _folded(node.left), _folded(node.right)
        return a + b if a is not None and b is not None else None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" \
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str) \
            and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
        parts = [_folded(e) for e in node.args[0].elts]
        return node.func.value.value.join(parts) if all(p is not None for p in parts) else None
    return None


def _outside_path(node) -> bool:
    """A path expression that points outside the model directory: an absolute or ~ literal, expanduser(),
    Path.home(), or an f-string / join starting with one of these."""
    s = _const_str(node)
    if s is not None:
        return s.startswith(("/", "~")) or bool(SENSITIVE.search(s))
    if isinstance(node, ast.JoinedStr) and node.values:
        return _outside_path(node.values[0])
    if isinstance(node, ast.BinOp):
        return _outside_path(node.left)
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name.endswith(("expanduser", "Path.home", "gethomedir")) or name in ("Path.home",):
            return True
        if name.endswith(("path.join", "Path")) and node.args:
            return _outside_path(node.args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("joinpath", "__truediv__"):
            return _outside_path(node.func.value)
    if isinstance(node, ast.Attribute) and _dotted(node).endswith("home"):
        return True
    return False


def _write_mode(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "mode"), None)
    m = _const_str(mode) if mode is not None else None
    return m is not None and any(c in m for c in "wax+")


class Scanner(ast.NodeVisitor):
    def __init__(self, rel: str, lines: list[str] | None = None, aliases: dict[str, str] | None = None):
        self.rel, self.out, self.lines = rel, [], lines or []
        self.aliases = aliases or {}
        self.imports: list[str] = []
        self.decodes = 0
        self.execs: list[ast.Call] = []

    def add(self, node, kind: str, detail: str) -> None:
        line = getattr(node, "lineno", 0) or 0
        text = self.lines[line - 1].strip()[:160] if 0 < line <= len(self.lines) else ""
        self.out.append({"file": self.rel, "line": line, "kind": kind, "detail": detail[:120], "excerpt": text})

    def resolve(self, name: str) -> str:
        """`name` with its first part replaced by what the file imported under that name."""
        head, dot, rest = name.partition(".")
        full = self.aliases.get(head) if head else None
        return f"{full}{dot}{rest}" if full else name

    @staticmethod
    def shown(written: str, name: str) -> str:
        """How a call is named in a finding: as written, plus its real name when an alias renamed it."""
        return written if name == written or name.endswith("." + written) else f"{name} (as {written})"

    def _module(self, node, name: str) -> None:
        top = name.split(".")[0]
        self.imports.append(name)
        if top == "pip":
            self.add(node, "external_code", f"imports {name}")
        elif top in LOADER_MODULES:
            self.add(node, "dynamic_import", f"imports {name}")
        elif name.startswith(REMOTE_MODULES):
            self.add(node, "remote_fetch", name)
        elif name.startswith("torch.utils.cpp_extension"):
            self.add(node, "native_load", name)
        elif top == "builtins":
            self.add(node, "builtins_access", name)
        elif top in DESERIALIZE:
            self.add(node, "unsafe_deserialize", f"imports {name}")
        elif top in NETWORK:
            self.add(node, "network_import", name)
        elif top in PROCESS:
            self.add(node, "process_import", name)
        elif top in NATIVE:
            self.add(node, "native_import", name)

    def visit_Import(self, node):
        for a in node.names:
            self._module(node, a.name)

    def visit_ImportFrom(self, node):
        if node.module and not node.level:
            self._module(node, node.module)
            if node.module == "os":
                for a in node.names:
                    if OS_PROCESS.match(a.name):
                        self.add(node, "process_call", f"from os import {a.name}")
            for a in node.names:
                full = f"{node.module}.{a.name}"
                if full.startswith(REMOTE_MODULES):
                    self.add(node, "remote_fetch", full)
                elif full == "torch.utils.cpp_extension":
                    self.add(node, "native_load", full)
                elif full in LOADERS:
                    self.add(node, "dynamic_import", f"from {node.module} import {a.name}")

    def visit_BinOp(self, node):
        v = _folded(node)
        if v is not None and isinstance(node.op, ast.Add) and v.strip() in RISKY_NAMES:
            self.add(node, "obfuscated_name", v.strip())
            return  # the pieces are not separate findings
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "__builtins__":
            self.add(node, "builtins_access", "__builtins__")

    def visit_Attribute(self, node):
        if node.attr in WALK_ATTRS:
            self.add(node, "object_walk", node.attr)
        elif node.attr == "__code__":
            self.add(node, "code_object", "__code__")
        self.generic_visit(node)

    def _loader(self, node, name: str) -> None:
        """A loader that runs a file or module by its path / name: always a finding, the path shown."""
        head, _, last = name.rpartition(".")
        if not (name in LOADERS or (last in LOADER_BARE | {"run_path", "run_module"}
                                    and head.split(".")[-1] in LOADER_HEADS) or (not head and last in LOADER_BARE)):
            return
        full = name if name in LOADERS else next((k for k in LOADERS if k.endswith("." + last)), name)
        pos = LOADERS.get(full, 0)
        arg = node.args[pos] if len(node.args) > pos else next(
            (k.value for k in node.keywords if k.arg in ("location", "path", "path_name", "archivepath", "mod_name")),
            None)
        shown = _folded(arg) if arg is not None else None
        self.add(node, "dynamic_import", f"{full}({shown or '<computed>'})")

    def _call_extra(self, node, name: str, written: str) -> None:
        # str.join of literal pieces naming something risky
        if isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            v = _folded(node)
            if v is not None and v.strip() in RISKY_NAMES:
                self.add(node, "obfuscated_name", v.strip())
        # importlib & friends with a computed or risky module name
        self._loader(node, name)
        if name in IMPORT_CALLS or name.endswith((".import_module", ".exec_module")):
            arg = _folded(node.args[0]) if node.args else None
            if arg is None or arg.split(".")[0] in RISKY_MODULES:
                self.add(node, "dynamic_import", f"{name}({arg or '<computed>'})")
        # getattr / setattr on builtins or a risky module
        if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr", "delattr") and node.args:
            target = node.args[0]
            tname = self.resolve(_dotted(target))
            if tname in ("__builtins__", "builtins"):
                self.add(node, "builtins_access", f"{node.func.id}({tname}, ...)")
            elif tname.split(".")[0] in RISKY_MODULES or (isinstance(target, ast.Call) and _dotted(target.func) in
                                                          ("__import__", "importlib.import_module", "import_module")):
                attr = _folded(node.args[1]) if len(node.args) > 1 else None
                self.add(node, "dynamic_attribute", f"{node.func.id}({tname or 'module'}, {attr or '<computed>'})")
        if name in ("vars", "globals", "locals") and node.args and _dotted(node.args[0]) in ("__builtins__", "builtins"):
            self.add(node, "builtins_access", f"{name}(builtins)")
        # deserialisation that runs code
        head, _, attr = name.rpartition(".")
        if head.split(".")[-1] in DESERIALIZE and attr in DESERIALIZE_CALLS:
            self.add(node, "unsafe_deserialize", name)
        elif name == "torch.load" and not any(k.arg == "weights_only" and isinstance(k.value, ast.Constant)
                                              and k.value.value is True for k in node.keywords):
            self.add(node, "unsafe_deserialize", "torch.load without weights_only=True")
        elif name in ("numpy.load", "np.load") and _pickle_allowed(node):
            self.add(node, "unsafe_deserialize", "numpy.load(allow_pickle=<not False>)")
        elif attr == "read_pickle":
            self.add(node, "unsafe_deserialize", self.shown(written, name))

    def visit_Call(self, node):
        written = _dotted(node.func)
        name = self.resolve(written)
        self._call_extra(node, name, written)
        self._foreign(node, name, written)
        head, _, attr = name.rpartition(".")
        if (head.split(".")[-1], attr) in DECODERS:
            self.decodes += 1
        if isinstance(node.func, ast.Name) and node.func.id in DYNAMIC:
            self.add(node, "dynamic_exec", node.func.id)
            self.execs.append(node)
            if any((_dotted(n.func).rpartition(".")[0].split(".")[-1], _dotted(n.func).rpartition(".")[2]) in DECODERS
                   for n in ast.walk(node) if isinstance(n, ast.Call) and n is not node):
                self.add(node, "encoded_exec", f"{node.func.id} of decoded data")
        elif (head in ("os", "posix") and OS_PROCESS.match(attr)) or head.split(".")[0] == "subprocess" \
                or name in ("pty.spawn", "pty.fork", "commands.getoutput") \
                or attr in ("create_subprocess_shell", "create_subprocess_exec") \
                or (name.split(".")[0] == "multiprocessing" and attr in MP_CALLS) or attr == "ProcessPoolExecutor":
            self.add(node, "process_call", self.shown(written, name))
            strings = [v for a in [*node.args, *(k.value for k in node.keywords)] for v in _strings(a)]
            if any(PIP_WORD.search(v) for v in strings) and any("install" in v for v in strings):
                self.add(node, "external_code", f"pip install via {self.shown(written, name)}")
        if name in CODE_INTERP and written.split(".")[0] in self.aliases:
            self.add(node, "dynamic_exec", name)
        last = name.rpartition(".")[2]
        if last in CODE_CALLS and (name == last or head.split(".")[-1] == "types"):
            self.add(node, "code_object", f"types.{last}")
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "replace" \
                and any((k.arg or "").startswith("co_") for k in node.keywords):
            self.add(node, "code_object", "code.replace(co_*)")
        if name.endswith(NATIVE_LOAD) or name in ("CDLL", "PyDLL"):
            self.add(node, "native_load", self.shown(written, name))
        if last in REMOTE_FETCH:
            self.add(node, "remote_fetch", self.shown(written, name))
        if name in ("os.open", "posix.open") and node.args and _outside_path(node.args[0]):
            self.add(node, "file_write_outside", "os.open() outside the model directory")
        if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args and _write_mode(node) \
                and _outside_path(node.args[0]):
            self.add(node, "file_write_outside", "open() for writing outside the model directory")
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("write_text", "write_bytes", "touch",
                                                                       "symlink_to", "mkdir") \
                and _outside_path(node.func.value):
            self.add(node, "file_write_outside", f"{node.func.attr}() outside the model directory")
        if name in ("shutil.copy", "shutil.copyfile", "shutil.copy2", "shutil.move", "os.rename", "os.replace",
                    "os.symlink") and len(node.args) > 1 and _outside_path(node.args[1]):
            self.add(node, "file_write_outside", f"{name}() outside the model directory")
        if name in ("os.environ.get", "os.getenv", "environ.get", "getenv") and node.args:
            self._env(node, node.args[0])
        if name in ("os.environ.items", "os.environ.copy", "os.environ.keys", "os.environ.values",
                    "environ.items", "environ.copy") or (name in ("dict", "json.dumps", "str") and node.args
                                                         and _dotted(node.args[0]) in ("os.environ", "environ")):
            self.add(node, "environment_dump", name)
        self.generic_visit(node)

    def _foreign(self, node, name: str, written: str) -> None:
        """Code from elsewhere pulled in at run time: trust_remote_code=True, torch.hub.load, pip."""
        if any(k.arg == "trust_remote_code" and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in node.keywords):
            # only with a literal repo id: a computed one is how repos load their own tokenizer / model (real corpus)
            ref_kw = ("pretrained_model_name_or_path", "model", "repo_id")
            target = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ref_kw), None)
            if target is not None and _folded(target) is not None:
                self.add(node, "external_code", f"{self.shown(written, name)}(trust_remote_code=True)")
        if name == "torch.hub.load":
            self.add(node, "external_code", "torch.hub.load")
        if name in PIP_CALLS:
            self.add(node, "external_code", name)

    def finish(self) -> None:
        """exec / eval in a file that also decodes data (decoded into a variable first)."""
        if self.decodes and not any(f["kind"] == "encoded_exec" for f in self.out):
            for node in self.execs:
                if node.func.id in ("exec", "eval", "compile"):
                    self.add(node, "encoded_exec", f"{node.func.id} in a file that decodes data")

    def _env(self, node, key) -> None:
        k = _const_str(key)
        if k is not None and _credential_name(k):
            self.add(node, "credential_env", k)

    def visit_Subscript(self, node):
        if _dotted(node.value) in ("os.environ", "environ"):
            self._env(node, node.slice)
        if _dotted(node.value) == "sys.modules":
            self.add(node, "dynamic_import", "sys.modules[...]")
        self.generic_visit(node)

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, bytes):
            try:
                v = v.decode("ascii")
            except UnicodeDecodeError:
                v = None
        if isinstance(v, str):
            if len(v) >= LONG_LITERAL and ENCODED.match(v[:20000]):
                self.add(node, "long_encoded_literal", f"{len(v)} characters")
            elif len(v) < 512 and SENSITIVE.search(v):
                self.add(node, "sensitive_path", v)
            elif v == "__builtins__":
                self.add(node, "builtins_access", "'__builtins__'")


def _pickle_allowed(call: ast.Call) -> bool:
    """numpy.load's allow_pickle (keyword or 3rd positional argument) given and not literally False."""
    v = call.args[2] if len(call.args) > 2 else next((k.value for k in call.keywords if k.arg == "allow_pickle"), None)
    return v is not None and not (isinstance(v, ast.Constant) and v.value is False)


def _credential_name(name: str) -> bool:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name.upper()) if w]
    return (any(w in CRED_WORDS for w in words) or any(p in CRED_PAIRS for p in zip(words, words[1:]))
            or name.upper().startswith(CRED_PREFIX))


def _inside(path: str, root: str) -> bool:
    real = os.path.realpath(path)
    return real == root or real.startswith(root + os.sep)


def py_files(snapshot: str) -> list[str]:
    out = []
    for dirpath, dirnames, files in os.walk(snapshot):
        dirnames.sort()
        out += [os.path.relpath(os.path.join(dirpath, f), snapshot) for f in sorted(files) if f.endswith(".py")]
    return out


def scan(root: str, snapshot: str, only: str | None = None) -> dict:
    """Scan every ``.py`` file under `snapshot` (only the file `only`, relative, when given)."""
    root = os.path.realpath(root)
    findings, summary, total, nfiles, truncated = [], [], 0, 0, False
    for dirpath, dirnames, files in os.walk(snapshot):
        dirnames.sort()
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, snapshot)
            if only is not None and rel != only:
                continue
            if not _inside(path, root):
                findings.append({"file": rel, "line": 0, "kind": "sensitive_path", "detail": "links outside the repo"})
                continue
            if nfiles >= MAX_FILES or total >= MAX_TOTAL_BYTES:
                truncated = True
                continue
            try:
                size = os.path.getsize(path)
                if size > MAX_FILE_BYTES:
                    findings.append({"file": rel, "line": 0, "kind": "oversized_file", "detail": f"{size} bytes"})
                    continue
                with open(path, "rb") as f:
                    src = f.read(MAX_FILE_BYTES + 1)
            except OSError as e:
                findings.append({"file": rel, "line": 0, "kind": "unparseable", "detail": type(e).__name__})
                continue
            nfiles += 1
            total += len(src)
            try:
                tree = ast.parse(src, filename=rel)
            except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
                findings.append({"file": rel, "line": getattr(e, "lineno", 0) or 0, "kind": "unparseable",
                                 "detail": type(e).__name__})
                continue
            s = Scanner(rel, src.decode("utf-8", "replace").splitlines(), _aliases(tree))
            try:
                s.visit(tree)
                s.finish()
            except RecursionError:
                findings.append({"file": rel, "line": 0, "kind": "unparseable", "detail": "RecursionError"})
            findings += s.out
            top = [type(n).__name__ for n in tree.body]
            summary.append({"file": rel, "lines": src.count(b"\n") + 1, "imports": sorted(set(s.imports))[:60],
                            "classes": top.count("ClassDef"), "functions": top.count("FunctionDef")
                            + top.count("AsyncFunctionDef"),
                            "top_level_other": sum(1 for t in top if t not in ("ClassDef", "FunctionDef",
                                                                                "AsyncFunctionDef", "Import",
                                                                                "ImportFrom", "Expr", "Assign",
                                                                                "AnnAssign", "If", "Try"))})
    if len(findings) > MAX_FINDINGS:
        findings, truncated = findings[:MAX_FINDINGS], True
    return {"findings": findings, "files": summary, "scanned_files": nfiles, "bytes": total, "truncated": truncated}


def apply_limits() -> dict:
    import resource
    limits = {"cpu": (resource.RLIMIT_CPU, LIMIT_CPU_S), "as": (resource.RLIMIT_AS, LIMIT_AS_BYTES),
              "fsize": (resource.RLIMIT_FSIZE, 0), "core": (resource.RLIMIT_CORE, 0)}
    for res, value in limits.values():
        resource.setrlimit(res, (value, value))
    return {k: resource.getrlimit(r)[0] for k, (r, _) in limits.items()}


def main(argv: list[str]) -> int:
    lim = apply_limits()  # before any untrusted file is read
    if argv[1:] == ["--limits"]:
        print(json.dumps(lim))
        return 0
    if len(argv) not in (3, 4):
        print(json.dumps({"error": "usage"}))
        return 2
    sys.setrecursionlimit(3000)
    print(json.dumps(scan(argv[1], argv[2], argv[3] if len(argv) == 4 else None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
