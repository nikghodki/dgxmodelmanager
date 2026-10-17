"""H3 supply chain of the decider image (static checks of deploy/decider; the real double build is
scripts/decider_image_check.sh)."""
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "deploy" / "decider"


def test_base_image_is_pinned_by_digest():
    froms = re.findall(r"^FROM\s+(\S+)", (D / "Dockerfile").read_text(), re.M)
    assert froms and all(re.fullmatch(r"[\w./-]+@sha256:[0-9a-f]{64}", f) for f in froms), froms


def test_pip_installs_only_the_hash_pinned_lock():
    text = (D / "Dockerfile").read_text().replace("\\\n", " ")  # join continuation lines
    installs = re.findall(r"pip install[^;\n]*", text)
    assert installs, "no pip install"
    for cmd in installs:
        assert "--require-hashes" in cmd and "--no-deps" in cmd and "-r /usr/local/share/dgxmm-decider/requirements.lock" in cmd
        assert "--only-binary=:all:" in cmd  # no sdist builds (no code runs at install time)
    assert "pip check" in text and "sbom-pip.txt" in text


def test_lock_pins_every_package_with_a_sha256():
    lines = [ln.strip() for ln in (D / "requirements.lock").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    reqs = " ".join(lines).replace("\\", " ").split()
    pkgs = [t for t in reqs if not t.startswith("--hash=")]
    hashes = [t for t in reqs if t.startswith("--hash=")]
    assert pkgs and len(pkgs) == len(hashes)
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][\w.]*", p) for p in pkgs), pkgs
    assert all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", h) for h in hashes)
    names = {p.split("==")[0] for p in pkgs}
    assert {"transformers", "flash-linear-attention"} <= names
    assert "transformers==5.17.0" in pkgs and "flash-linear-attention==0.5.2" in pkgs


def test_runtime_is_unprivileged_and_key_comes_from_a_file():
    text = (D / "Dockerfile").read_text()
    assert re.search(r"^USER 10001:10001$", text, re.M)
    assert "DECISION_API_KEY=" not in text
    ep = (D / "entrypoint.sh").read_text()
    assert "python3 -I" in ep and '--hashes "$hashes"' in ep and "launch.py" in ep and "exit 4" in ep
    launch = (D / "launch.py").read_text()
    assert "serve(model, args.host, args.port, api_key=key)" in launch and "os.unlink(path)" in launch
    assert "os.environ[" not in launch and "putenv" not in launch  # the key never enters the environment


def test_build_verifies_the_hash_list_of_every_supported_model():
    """The image refuses to build unless every model it claims to serve (label dgxmm.decider.models) has a
    well-formed pinned hash list covering the snapshot's code, manifest, config and tokenizer."""
    from dgxmm.engines.decision import DECIDER_MODELS
    text = (D / "Dockerfile").read_text().replace("\\\n", " ")
    arg = re.search(r'^ARG DGXMM_DECIDER_MODELS="([^"]+)"', text, re.M)
    assert arg, "the model list is one build arg"
    assert sorted(arg.group(1).split(",")) == sorted(f"{r}@{m.revision}" for r, m in DECIDER_MODELS.items())
    assert 'dgxmm.decider.models="${DGXMM_DECIDER_MODELS}"' in text  # the label is that same list
    verify = re.search(r"^RUN .*DGXMM_DECIDER_MODELS.*$", text, re.M)
    assert verify, "a RUN step checks every listed model's hash list"
    for need in ("d3_server.py", "MODEL_MANIFEST.json", "config.json", "tokenizer.json", "exit 1"):
        assert need in verify.group(0), need
    assert text.index(verify.group(0)) < text.index("USER 10001:10001")
