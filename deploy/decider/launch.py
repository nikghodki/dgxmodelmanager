"""dgxmm-decider launcher: serve the pinned d3 snapshot through its own ``d3_server.serve()``, from a private,
hash-verified copy of its code, with the bearer key read from a file instead of the environment.

The snapshot directory is not trusted as an import path: it lives in the Hugging Face cache, which other containers
(vLLM engines running repositories' remote code) can write to. So the launcher (run with ``python3 -I``: no script
directory, current directory, user site or PYTHON* variables on the import path):

1. copies exactly the files of the image's pinned hash list from the snapshot into a fresh directory on the
   container's own tmpfs and checks every copy against its pinned hash (exit 3 on any difference): what is checked
   is exactly what runs, with no window between the check and the import, and a file planted next to the snapshot's
   code (``json.py``, ``numpy/``, ...) is never on the import path;
2. links the weight / tokenizer data files the pinned ``MODEL_MANIFEST.json`` lists (data suffixes only, never
   code) from the snapshot into that directory, makes it read-only and imports only from it;
3. waits (up to --key-wait-s) for the file named by ``DECISION_API_KEY_FILE`` (DGXMM writes it into a tmpfs right
   after the container starts), reads it and deletes it; refuses to start without a key (exit 4): the decider is
   never served unauthenticated (d3_server's ``main()`` would read the key from the environment, where it shows in
   ``docker inspect`` and is inherited by every child process);
4. loads the package (``--verify fast`` checks it against the pinned MODEL_MANIFEST.json), warms up, serves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time

KEY_FILE_VARIABLE = "DECISION_API_KEY_FILE"
DEFAULT_KEY_FILE = "/run/dgxmm-decider/api_key"  # a tmpfs (see dgxmm/engines/decision.py)
MAX_KEY_BYTES = 4096
MANIFEST = "MODEL_MANIFEST.json"
# Files of the manifest that are linked (not copied) into the package directory: data the runtime reads, never
# anything Python could import (.py, .pyc, .so, .pth, package directories).
DATA_SUFFIXES = (".safetensors", ".json", ".txt", ".jinja", ".model", ".tiktoken")
_HASH_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*)")
_DATA_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
CHUNK = 1 << 20


class Refused(Exception):
    """A start-up refusal: message for the log, exit code for the deployer."""

    def __init__(self, code: int, lines: list[str]):
        super().__init__("; ".join(lines))
        self.code, self.lines = code, lines


def read_hash_list(path: str) -> list[tuple[str, str]]:
    """[(sha256, relative name)] of a pinned hash list (``sha256sum`` format, safe relative names only)."""
    out = []
    with open(path, encoding="ascii") as f:
        for line in f.read().splitlines():
            m = _HASH_LINE.fullmatch(line)
            if not m or ".." in m.group(2).split("/"):
                raise Refused(3, [f"dgxmm-decider: malformed pinned hash list {path}"])
            out.append((m.group(1), m.group(2)))
    if not out:
        raise Refused(3, [f"dgxmm-decider: malformed pinned hash list {path}"])
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_package(snapshot: str, hash_list: str, dest: str) -> None:
    """Copy the pinned files into `dest` (an empty directory), verify the copies, link the manifest's data files and
    make `dest` read-only. Raises Refused(3) when a pinned file is missing or differs."""
    entries = read_hash_list(hash_list)
    problems = []
    for digest, name in entries:
        out = os.path.join(dest, name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            with open(os.path.join(snapshot, name), "rb") as src, open(out, "xb") as dst:
                while chunk := src.read(CHUNK):
                    dst.write(chunk)
        except OSError:
            problems.append(f"{name}: FAILED open or read")
            continue
        if _sha256(out) != digest:  # the copy is what runs, so the copy is what is checked
            problems.append(f"{name}: FAILED")
    if problems:
        n = len(problems)
        raise Refused(3, problems + [f"sha256sum: WARNING: {n} computed checksum{'s' if n != 1 else ''} did NOT match",
                                     "dgxmm-decider: snapshot files do not match the pinned hashes; refusing to start"])
    pinned = {name for _, name in entries}
    try:
        with open(os.path.join(dest, MANIFEST), encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        raise Refused(3, [f"dgxmm-decider: the pinned hash list does not cover a readable {MANIFEST}",
                          "dgxmm-decider: snapshot files do not match the pinned hashes; refusing to start"])
    listed = set(manifest.get("files_sha256") or {}) | set(manifest.get("files_bytes") or {})
    for name in sorted(listed - pinned):
        if _DATA_NAME.fullmatch(name) and name.endswith(DATA_SUFFIXES) and os.path.exists(os.path.join(snapshot, name)):
            os.symlink(os.path.join(snapshot, name), os.path.join(dest, name))
    for root, dirs, files in os.walk(dest):
        for n in files:
            p = os.path.join(root, n)
            if not os.path.islink(p):
                os.chmod(p, 0o444)
        os.chmod(root, 0o555)


def read_key_file(path: str, wait_s: float, poll_s: float = 0.2) -> str | None:
    """The key in `path` once it exists and is non-empty (the file is then deleted), or None after `wait_s`."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            with open(path, "rb") as f:
                key = f.read(MAX_KEY_BYTES).decode("ascii", "strict").strip()
        except FileNotFoundError:
            key = ""
        except (OSError, UnicodeDecodeError) as e:
            print(f"dgxmm-decider: cannot read the key file: {type(e).__name__}", file=sys.stderr, flush=True)
            return None
        if key:
            try:
                os.unlink(path)  # the key now lives only in this process
            except OSError:
                pass
            return key
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve the pinned d3 snapshot (System One) with a file-delivered key.")
    ap.add_argument("--snapshot", required=True, help="snapshot directory (read-only model mount)")
    ap.add_argument("--hashes", required=True, help="the image's pinned hash list for this snapshot")
    ap.add_argument("--work-dir", default="/tmp", help="tmpfs under which the verified package copy is made")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--key-wait-s", type=float, default=120.0, help="how long to wait for the key file")
    args = ap.parse_args(argv)

    if not sys.flags.isolated:
        print("dgxmm-decider: launch.py must run with python3 -I", file=sys.stderr, flush=True)
        return 3
    try:
        package = tempfile.mkdtemp(prefix="dgxmm-decider-pkg.", dir=args.work_dir)
        stage_package(args.snapshot, args.hashes, package)
    except Refused as e:
        for line in e.lines:
            print(line, file=sys.stderr, flush=True)
        return e.code
    if os.environ.get("DECISION_API_KEY"):
        print("dgxmm-decider: DECISION_API_KEY must not be set; use DECISION_API_KEY_FILE", file=sys.stderr)
        return 4
    path = os.environ.get(KEY_FILE_VARIABLE) or DEFAULT_KEY_FILE
    key = read_key_file(path, args.key_wait_s)
    if not key:
        print(f"dgxmm-decider: no key in {path} after {args.key_wait_s:.0f} s; refusing to serve without a key",
              file=sys.stderr, flush=True)
        return 4

    sys.path.insert(0, package)
    import d3_server  # noqa: E402  (the verified copy of the snapshot's server module)
    from d3_runtime import D3  # noqa: E402

    model = D3.from_pretrained(package, device=args.device, batch_size=args.batch_size, verify="fast")
    model.warmup()
    server = d3_server.serve(model, args.host, args.port, api_key=key)
    del key
    host, port = server.server_address[:2]
    print(f"d3 System One server: {model.model_name} on http://{host}:{port}{d3_server.SYSTEM_ONE_PATH}; "
          "bearer key required", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
