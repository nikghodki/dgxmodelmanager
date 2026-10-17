"""deploy/decider/launch.py: the decider runs only a private, hash-verified copy of the pinned snapshot files
(docs/security.md, "Decision model code integrity")."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dgxmm.decider import diagnosis

D = Path(__file__).resolve().parents[1] / "deploy" / "decider"


def _load():
    spec = importlib.util.spec_from_file_location("dgxmm_decider_launch", D / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launch = _load()

FAKE_SERVER = '''import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uuid  # a module the pinned code imports after start-up: a planted uuid.py must never shadow it
SYSTEM_ONE_PATH = "/v1/decide"
class _S:
    server_address = ("127.0.0.1", 1)
    def serve_forever(self):
        print("served-from", HERE, flush=True)
    def server_close(self):
        pass
def serve(model, host, port, api_key=None):
    assert api_key == "k-123"
    return _S()
'''
FAKE_RUNTIME = '''import os
class D3:
    model_name = "fake"
    @classmethod
    def from_pretrained(cls, root, **kw):
        assert os.path.isfile(os.path.join(root, "model.safetensors")), "weights are linked into the package"
        assert open(os.path.join(root, "model.safetensors"), "rb").read() == b"weights"
        return cls()
    def warmup(self):
        pass
'''


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    snap = tmp_path / "snap"
    snap.mkdir()
    files = {"d3_server.py": FAKE_SERVER, "d3_runtime.py": FAKE_RUNTIME, "config.json": "{}",
             "tokenizer.json": "{}"}
    for n, t in files.items():
        (snap / n).write_text(t)
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "model.safetensors.index.json").write_text("{}")
    manifest = {"files_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()},
                "files_bytes": {"model.safetensors": 7, "model.safetensors.index.json": 2, "README.md": 1,
                                "extra_code.py": 1, "assets/banner.png": 1}}
    (snap / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    (snap / "extra_code.py").write_text("raise SystemExit('unpinned code imported')\n")
    files["MODEL_MANIFEST.json"] = json.dumps(manifest)
    hashes = tmp_path / "d3-test-rev.sha256"
    hashes.write_text("".join(f"{hashlib.sha256(t.encode()).hexdigest()}  {n}\n" for n, t in sorted(files.items())))
    return snap, hashes


def _plant(snap: Path, marker: Path) -> None:
    """What a container with write access to the HF cache could add next to the pinned code."""
    code = f"open({str(marker)!r}, 'w').write('planted code ran')\n"
    (snap / "uuid.py").write_text(code)
    (snap / "json.py").write_text(code)
    (snap / "sitecustomize.py").write_text(code)
    (snap / "numpy").mkdir()
    (snap / "numpy" / "__init__.py").write_text(code)
    (snap / "evil.pth").write_text(f"import os; {code}")


def test_stage_copies_exactly_the_pinned_files_and_links_only_manifest_data(tmp_path):
    snap, hashes = _snapshot(tmp_path)
    _plant(snap, tmp_path / "marker")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    launch.stage_package(str(snap), str(hashes), str(pkg))
    names = sorted(p.name for p in pkg.iterdir())
    assert names == sorted(["MODEL_MANIFEST.json", "config.json", "d3_runtime.py", "d3_server.py", "tokenizer.json",
                            "model.safetensors", "model.safetensors.index.json"])
    for n in ("d3_server.py", "d3_runtime.py", "config.json", "tokenizer.json", "MODEL_MANIFEST.json"):
        assert not (pkg / n).is_symlink() and (pkg / n).read_bytes() == (snap / n).read_bytes()
    assert (pkg / "model.safetensors").is_symlink()  # data: linked, never code
    assert os.stat(pkg).st_mode & 0o777 == 0o555 and os.stat(pkg / "d3_server.py").st_mode & 0o777 == 0o444


def test_the_copy_not_the_original_is_what_runs(tmp_path):
    """No check-then-use window: changing the snapshot after staging changes nothing that is imported."""
    snap, hashes = _snapshot(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    launch.stage_package(str(snap), str(hashes), str(pkg))
    (snap / "d3_server.py").write_text("raise SystemExit('swapped after the check')\n")
    assert (pkg / "d3_server.py").read_text() == FAKE_SERVER


@pytest.mark.parametrize("damage", ["tamper", "missing"])
def test_a_differing_or_missing_pinned_file_refuses_with_exit_3_and_is_diagnosed(tmp_path, damage):
    snap, hashes = _snapshot(tmp_path)
    if damage == "tamper":
        (snap / "d3_runtime.py").write_text(FAKE_RUNTIME + "\n# changed\n")
    else:
        (snap / "tokenizer.json").unlink()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    with pytest.raises(launch.Refused) as e:
        launch.stage_package(str(snap), str(hashes), str(pkg))
    assert e.value.code == 3
    log = "\n".join(e.value.lines)
    assert ("d3_runtime.py: FAILED" if damage == "tamper" else "tokenizer.json: FAILED open or read") in log
    assert [c for c, _ in diagnosis.rule_hits(log, 3, engine="decision")] == ["integrity_check_failed"]


@pytest.mark.parametrize("line", ["0" * 64 + "  ../escape.py", "0" * 64 + "  /abs.py", "nonsense", ""])
def test_malformed_hash_lists_refuse(tmp_path, line):
    h = tmp_path / "h.sha256"
    h.write_text(line + "\n" if line else "")
    with pytest.raises(launch.Refused) as e:
        launch.read_hash_list(str(h))
    assert e.value.code == 3


def _run(tmp_path, snap, hashes, *flags):
    key = tmp_path / "key"
    key.write_text("k-123")
    env = {"PATH": os.environ.get("PATH", ""), "DECISION_API_KEY_FILE": str(key),
           "PYTHONPATH": str(snap)}  # ignored under -I
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, *flags, str(D / "launch.py"), "--snapshot", str(snap), "--hashes",
                           str(hashes), "--work-dir", str(work), "--key-wait-s", "2"],
                          cwd=snap, env=env, capture_output=True, text=True, timeout=60)


def test_launcher_never_imports_files_planted_in_the_snapshot(tmp_path):
    snap, hashes = _snapshot(tmp_path)
    marker = tmp_path / "marker"
    _plant(snap, marker)
    p = _run(tmp_path, snap, hashes, "-I", "-B")
    assert p.returncode == 0, p.stderr
    assert "served-from " + str(tmp_path / "work") in p.stdout  # the verified copy on the work tmpfs
    assert not marker.exists(), "planted code ran"


def test_launcher_refuses_to_run_without_isolated_mode(tmp_path):
    snap, hashes = _snapshot(tmp_path)
    p = _run(tmp_path, snap, hashes)
    assert p.returncode == 3 and "python3 -I" in p.stderr


def test_launcher_tampered_snapshot_exits_3_before_waiting_for_the_key(tmp_path):
    snap, hashes = _snapshot(tmp_path)
    (snap / "d3_server.py").write_text(FAKE_SERVER + "\n")
    p = _run(tmp_path, snap, hashes, "-I")
    assert p.returncode == 3
    assert "d3_server.py: FAILED" in p.stderr and "do not match the pinned hashes" in p.stderr
    assert (tmp_path / "key").exists()  # refused before reading (and deleting) the key


def test_entrypoint_runs_the_launcher_isolated_with_the_hash_list():
    ep = (D / "entrypoint.sh").read_text()
    assert "exec python3 -I -B" in ep and '--hashes "$hashes"' in ep and "--work-dir /tmp" in ep
    assert "sys.path.insert(0, args.snapshot)" not in (D / "launch.py").read_text()
