"""Hermetic tests for install.sh: a temp copy of the repo, a fake HOME and a PATH that holds only
stub docker/nvidia-smi/id/stat/ip/hostname/ss/uname scripts plus symlinks to real coreutils."""
from __future__ import annotations

import os
import shutil
import stat as st
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STUBS = Path(__file__).resolve().parent / "fixtures" / "install_stubs"
TOOLS = ["bash", "env", "sed", "awk", "grep", "cat", "mkdir", "chmod", "cp", "mv", "rm", "rmdir",
         "dirname", "basename", "df", "cut", "tr", "head", "tail", "sort", "date", "sleep", "wc",
         "mktemp", "readlink", "realpath", "tee", "touch", "ls", "seq", "printf", "true", "false", "uniq", "find"]
PW = "Zx9_fake-Admin-pw-123"
FIRST_RUN = f"2026-10-16 WARNING dgxmm.bootstrap: DGXMM first run: created user 'admin' with password {PW} (change it at first login)"


class Box:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.home = tmp / "home"
        self.bin = tmp / "bin"
        self.tools = tmp / "tools"
        self.log = tmp / "calls.log"
        for d in (self.repo, self.home, self.bin, self.tools):
            d.mkdir()
        for name in ("install.sh", "config.example.yaml", "docker-compose.yaml"):
            shutil.copy2(ROOT / name, self.repo / name)
        for stub in STUBS.iterdir():
            (self.bin / stub.name).symlink_to(stub)
        for tool in TOOLS:
            real = shutil.which(tool)
            if real:
                (self.tools / tool).symlink_to(real)
        self.log.touch()

    def remove_stub(self, name: str) -> None:
        (self.bin / name).unlink()

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        base = {
            "PATH": f"{self.bin}:{self.tools}",
            "HOME": str(self.home),
            "FAKE_LOG": str(self.log),
            "REAL_STAT": shutil.which("stat") or "/usr/bin/stat",
            "DGXMM_HEALTH_POLL": "0",
            "LANG": "C.UTF-8",
        }
        base.update(env)
        p = subprocess.run(["bash", str(self.repo / "install.sh"), *args], cwd=self.repo, env=base,
                           capture_output=True, text=True, timeout=60)
        p.out = p.stdout + p.stderr  # type: ignore[attr-defined]
        return p

    def calls(self) -> str:
        return self.log.read_text()

    def snapshot(self) -> dict[str, tuple[int, bytes]]:
        snap = {}
        for base in (self.repo, self.home):
            for p in sorted(base.rglob("*")):
                snap[str(p)] = (p.stat().st_mode, p.read_bytes() if p.is_file() else b"")
        return snap


@pytest.fixture
def box(tmp_path) -> Box:
    return Box(tmp_path)


def mode(p: Path) -> int:
    return st.S_IMODE(p.stat().st_mode)


# --- help / argument handling -----------------------------------------------------------------

def test_help_lists_commands(box):
    p = box.run("--help")
    assert p.returncode == 0
    for word in ("install", "upgrade", "status", "logs", "uninstall", "--dry-run", "--purge",
                 "--allow-root", "DGXMM_PORT", "DGXMM_BIND", "HF_TOKEN", "DGXMM_PULL_ENGINES"):
        assert word in p.stdout
    assert box.calls() == ""


def test_unknown_option_fails(box):
    p = box.run("--bogus")
    assert p.returncode == 2 and "unknown option" in p.out


# --- preflight ---------------------------------------------------------------------------------

def test_refuses_root(box):
    p = box.run("--dry-run", FAKE_UID="0")
    assert p.returncode != 0
    assert "refusing to run as root" in p.out and "--allow-root" in p.out


def test_allow_root(box):
    p = box.run("--dry-run", "--allow-root", FAKE_UID="0")
    assert p.returncode == 0, p.out


def test_no_docker(box):
    box.remove_stub("docker")
    p = box.run("--dry-run")
    assert p.returncode != 0
    assert "docker is not installed" in p.out


def test_docker_permission_denied(box):
    p = box.run("--dry-run", FAKE_DAEMON="denied")
    assert p.returncode != 0
    assert "usermod -aG docker" in p.out


def test_docker_daemon_down(box):
    p = box.run("--dry-run", FAKE_DAEMON="down")
    assert p.returncode != 0
    assert "cannot reach the Docker daemon" in p.out


def test_no_compose_v2(box):
    p = box.run("--dry-run", FAKE_COMPOSE="missing")
    assert p.returncode != 0
    assert "docker compose v2" in p.out


def test_no_nvidia_runtime(box):
    p = box.run("--dry-run", FAKE_RUNTIMES='{"runc":{"path":"runc"}}')
    assert p.returncode != 0
    assert "nvidia runtime" in p.out and "nvidia-container-toolkit" in p.out


def test_no_gpu(box):
    p = box.run("--dry-run", FAKE_GPU="broken")
    assert p.returncode != 0
    assert "nvidia-smi" in p.out


def test_port_in_use_by_something_else(box):
    p = box.run("--dry-run", FAKE_PORT_BUSY="8090")
    assert p.returncode != 0
    assert "port 8090 is already in use" in p.out


def test_port_in_use_by_our_own_container_is_fine(box):
    p = box.run("--dry-run", FAKE_PORT_BUSY="8090", FAKE_DGXMM_STATE="running",
                FAKE_DGXMM_WORKDIR=str(box.repo.resolve()))
    assert p.returncode == 0, p.out


def test_refuses_second_instance_from_another_checkout(box):
    p = box.run("--dry-run", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR="/srv/other/dgxmm")
    assert p.returncode != 0
    assert "another DGXMM" in p.out and "/srv/other/dgxmm" in p.out


def test_reports_arch(box):
    p = box.run("--dry-run", FAKE_ARCH="x86_64")
    assert p.returncode == 0 and "x86_64" in p.out


# --- dry run -----------------------------------------------------------------------------------

def test_dry_run_writes_nothing_and_shows_detected_values(box):
    before = box.snapshot()
    p = box.run("--dry-run")
    assert p.returncode == 0, p.out
    assert box.snapshot() == before
    h = box.home
    for line in ("DGXMM_UID=1234", "DGXMM_GID=2345", "DOCKER_GID=977", "DGXMM_PORT=8090",
                 "DGXMM_BIND=0.0.0.0", "DGXMM_NETWORK=dgxmm", f"DGXMM_HOST_HOME={h}",
                 f"DGXMM_HF_CACHE={h}/.cache/huggingface", f"DGXMM_MODELS_DIR={h}/models"):
        assert line in p.out
    assert f"host: {h}/.cache/huggingface" in p.out
    assert f"host: {h}/models" in p.out
    assert f"host: {h}/.triton" in p.out
    assert "docker_network: dgxmm" in p.out
    calls = box.calls()
    assert "compose up" not in calls and "network create" not in calls
    assert "would" in p.out


def test_dry_run_honours_env_settings(box):
    p = box.run("--dry-run", DGXMM_PORT="9100", DGXMM_BIND="127.0.0.1", DGXMM_NETWORK="gateway",
                DGXMM_HF_CACHE="/data/hf-cache", DGXMM_MODELS_DIR="/data/gguf",
                LITELLM_URL="http://litellm.example:4000")
    assert p.returncode == 0, p.out
    for line in ("DGXMM_PORT=9100", "DGXMM_BIND=127.0.0.1", "DGXMM_NETWORK=gateway",
                 "DGXMM_HF_CACHE=/data/hf-cache", "DGXMM_MODELS_DIR=/data/gguf",
                 "host: /data/hf-cache", "host: /data/gguf", "docker_network: gateway",
                 "litellm_url: http://litellm.example:4000"):
        assert line in p.out


def test_flags_override_env(box):
    p = box.run("--dry-run", "--port", "9200", "--bind", "127.0.0.1", "--network", "lab", DGXMM_PORT="9100")
    assert p.returncode == 0, p.out
    assert "DGXMM_PORT=9200" in p.out and "DGXMM_NETWORK=lab" in p.out


def test_rejects_bad_values(box):
    assert box.run("--dry-run", DGXMM_PORT="99999").returncode != 0
    assert box.run("--dry-run", DGXMM_BIND="not-an-ip").returncode != 0
    p = box.run("--dry-run", DGXMM_HF_CACHE="relative/path")
    assert p.returncode != 0 and "absolute" in p.out


def test_existing_files_are_kept(box):
    env = "DGXMM_UID=1000\nDGXMM_GID=1000\nDOCKER_GID=988\nDGXMM_PORT=8090\nDGXMM_BIND=127.0.0.1\n" \
          "DGXMM_HF_CACHE=/srv/hf\nDGXMM_MODELS_DIR=/srv/models\nDGXMM_NETWORK=gateway\nDGXMM_HOST_HOME=/srv\n"
    (box.repo / ".env").write_text(env)
    (box.repo / "config.yaml").write_text("docker_network: gateway\n")
    before = box.snapshot()
    p = box.run("--dry-run", DGXMM_PORT="9999")
    assert p.returncode == 0, p.out
    assert "kept existing .env" in p.out and "kept existing config.yaml" in p.out
    assert "DGXMM_PORT=9999" in p.out and "edit .env" in p.out  # warns that .env wins
    assert box.snapshot() == before


def test_warns_when_config_network_differs_from_env(box):
    (box.repo / "config.yaml").write_text("docker_network: other\n")
    p = box.run("--dry-run")
    assert p.returncode == 0, p.out
    assert "docker_network" in p.out and "other" in p.out and "WARN" in p.out


# --- real install against the fake docker ---------------------------------------------------------

def test_install_generates_files_and_prints_password_once(box, monkeypatch):
    p = box.run(FAKE_LOGS=FIRST_RUN, HF_TOKEN="hf_secret_token_value")
    assert p.returncode == 0, p.out
    repo, h = box.repo, box.home
    env = (repo / ".env").read_text()
    assert "DGXMM_UID=1234" in env and "DOCKER_GID=977" in env and f"DGXMM_HF_CACHE={h}/.cache/huggingface" in env

    monkeypatch.setenv("DGXMM_HOST_HOME", "/nowhere")  # rendered paths must be absolute, not ~
    from dgxmm.config import load_settings
    s = load_settings(str(repo / "config.yaml"))
    assert s.hf_cache.host == f"{h}/.cache/huggingface" and s.hf_cache.local == "/data/hf"
    assert s.gguf_roots[0].host == f"{h}/models"
    assert [m.host for m in s.vllm_cache_mounts] == [f"{h}/.cache/vllm", f"{h}/.cache/flashinfer", f"{h}/.triton"]
    assert s.docker_network == "dgxmm"

    sec = repo / "secrets"
    assert mode(sec) == 0o700
    for f in ("hf_token", "litellm_master_key", "admin_password"):
        assert mode(sec / f) == 0o600
    assert (sec / "hf_token").read_text() == "hf_secret_token_value"
    assert (sec / "litellm_master_key").read_text() == ""
    assert (sec / "admin_password").read_text().strip() == PW
    assert (repo / "data").is_dir()
    for d in (".cache/huggingface", "models", ".cache/vllm", ".cache/flashinfer", ".triton"):
        assert (h / d).is_dir()

    assert p.out.count(PW) == 1
    assert "hf_secret_token_value" not in p.out
    assert "http://192.0.2.20:8090/dgxmm/" in p.out and "http://198.51.100.2:8090/dgxmm/" in p.out
    assert "http://spark-test:8090/dgxmm/" in p.out
    assert "172.17.0.1" not in p.out and "172.23.0.1" not in p.out
    calls = box.calls()
    assert "docker network create" in calls and "docker compose up -d --build" in calls

    # Re-run = upgrade: keeps everything, never reprints the password.
    before = (repo / ".env").read_text(), (repo / "config.yaml").read_text(), (sec / "admin_password").read_text()
    p2 = box.run(FAKE_LOGS="", FAKE_NETWORK="exists", FAKE_DGXMM_STATE="running",
                 FAKE_DGXMM_WORKDIR=str(repo.resolve()), HF_TOKEN="hf_other")
    assert p2.returncode == 0, p2.out
    assert PW not in p2.out and "see secrets/admin_password" in p2.out
    assert "kept existing .env" in p2.out and "kept existing config.yaml" in p2.out
    assert before == ((repo / ".env").read_text(), (repo / "config.yaml").read_text(), (sec / "admin_password").read_text())
    assert (sec / "hf_token").read_text() == "hf_secret_token_value"


def test_install_uses_existing_network(box):
    p = box.run(FAKE_NETWORK="exists")
    assert p.returncode == 0, p.out
    assert "network create" not in box.calls()


def test_install_fails_when_unhealthy(box):
    p = box.run(FAKE_HEALTH="running/unhealthy")
    assert p.returncode != 0
    assert "unhealthy" in p.out


def test_install_times_out_while_starting(box):
    p = box.run(FAKE_HEALTH="running/starting", DGXMM_HEALTH_TIMEOUT="1")
    assert p.returncode != 0
    assert "did not become healthy" in p.out


def test_litellm_key_autodetected_never_printed(box):
    p = box.run(FAKE_LITELLM="1", FAKE_LITELLM_KEY="sk-very-secret-master", FAKE_LITELLM_NET="dgxmm")
    assert p.returncode == 0, p.out
    assert (box.repo / "secrets" / "litellm_master_key").read_text() == "sk-very-secret-master"
    assert "sk-very-secret-master" not in p.out


def test_litellm_key_from_env(box):
    p = box.run(LITELLM_MASTER_KEY="sk-from-env")
    assert p.returncode == 0, p.out
    assert (box.repo / "secrets" / "litellm_master_key").read_text() == "sk-from-env"
    assert "sk-from-env" not in p.out


def test_pull_engines(box):
    p = box.run(DGXMM_PULL_ENGINES="1")
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "docker pull vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8" in calls
    assert "docker pull ghcr.io/ggml-org/llama.cpp@sha256:e4cfff28678327b0ae7eda129e700e56a0fa55aa23db2f016256ea155c15c2b5" in calls


def test_loopback_bind_prints_loopback_url(box):
    p = box.run(DGXMM_BIND="127.0.0.1")
    assert p.returncode == 0, p.out
    assert "http://127.0.0.1:8090/dgxmm/" in p.out and "192.0.2.20" not in p.out


# --- status / logs / upgrade / uninstall --------------------------------------------------------------

def test_status_running(box):
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()))
    assert p.returncode == 0, p.out
    assert "healthy" in p.out


def test_status_not_installed(box):
    p = box.run("status")
    assert p.returncode != 0 and "not installed" in p.out


def test_logs(box):
    p = box.run("logs", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()))
    assert p.returncode == 0, p.out
    assert "docker logs --tail" in box.calls()


def test_upgrade_outside_git_checkout_dry_run(box):
    before = box.snapshot()
    p = box.run("upgrade", "--dry-run")
    assert p.returncode == 0, p.out
    assert "not a git checkout" in p.out
    assert box.snapshot() == before


def _installed(box):
    for d in ("data", "secrets"):
        (box.repo / d).mkdir()
    (box.repo / "data" / "dgxmm.sqlite3").write_text("db")
    (box.repo / "secrets" / "hf_token").write_text("t")
    (box.repo / "config.yaml").write_text("docker_network: dgxmm\n")
    (box.repo / ".env").write_text("DGXMM_NETWORK=dgxmm\n")
    (box.home / ".cache" / "huggingface").mkdir(parents=True)
    (box.home / ".cache" / "huggingface" / "model.bin").write_text("weights")
    return dict(FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()),
                FAKE_MANAGED_IDS="c1 c2", FAKE_NETWORK="exists")


def test_uninstall_keeps_data_and_engines(box):
    fake = _installed(box)
    p = box.run("uninstall", **fake)
    assert p.returncode == 0, p.out
    for f in ("data/dgxmm.sqlite3", "secrets/hf_token", "config.yaml", ".env"):
        assert (box.repo / f).exists()
    calls = box.calls()
    assert "docker rm dgxmm" in calls or "docker rm -f dgxmm" in calls
    assert "docker image rm dgxmm:latest" in calls
    assert "c1" not in calls.replace("ps -aq", "")  # engine containers untouched
    assert (box.home / ".cache" / "huggingface" / "model.bin").exists()


def test_uninstall_purge_removes_state_and_managed_engines_only(box):
    fake = _installed(box)
    (box.repo / ".env").write_text("DGXMM_NETWORK=dgxmm\nDGXMM_INSTALL_ID=zz99yy88xx77\n")
    p = box.run("uninstall", "--purge", **fake, FAKE_NETWORK_LABEL="true", FAKE_OWN_MANAGED_IDS="c1 c2")
    assert p.returncode == 0, p.out
    for f in ("data", "secrets", "config.yaml", ".env"):
        assert not (box.repo / f).exists()
    calls = box.calls()
    assert "ps -aq --filter label=dgxmm.managed=true --filter label=dgxmm.install=zz99yy88xx77" in calls
    assert "docker rm -f c1 c2" in calls
    assert "docker network rm dgxmm" in calls
    assert (box.home / ".cache" / "huggingface" / "model.bin").exists()
    assert (box.repo / "install.sh").exists()


def test_uninstall_purge_keeps_network_it_did_not_create(box):
    fake = _installed(box)
    p = box.run("uninstall", "--purge", **fake, FAKE_NETWORK_LABEL="")
    assert p.returncode == 0, p.out
    assert "network rm" not in box.calls()


def _installed_with_decider_net(box):
    fake = _installed(box)
    (box.repo / ".env").write_text("DGXMM_NETWORK=dgxmm\nDGXMM_INSTALL_ID=zz99yy88xx77\n")
    return {**fake, "FAKE_OWN_MANAGED_IDS": "c1 c2", "FAKE_DECIDER_NET": "exists"}


def test_uninstall_purge_removes_the_decider_network(box):
    fake = _installed_with_decider_net(box)
    p = box.run("uninstall", "--purge", **fake)
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "docker network rm dgxmm-decider" in calls
    # after the managed containers (the decider) are gone
    assert calls.index("docker rm -f c1 c2") < calls.index("docker network rm dgxmm-decider")


def test_uninstall_purge_uses_the_configured_decider_network_and_checks_its_label(box):
    fake = _installed_with_decider_net(box)
    (box.repo / "config.yaml").write_text("docker_network: dgxmm\ndecider_network: \"my-dec\"  # internal\n")
    p = box.run("uninstall", "--purge", **fake)
    assert p.returncode == 0, p.out
    assert "docker network rm my-dec" in box.calls()


def test_uninstall_purge_keeps_a_same_named_network_dgxmm_did_not_create(box):
    fake = _installed_with_decider_net(box)
    p = box.run("uninstall", "--purge", **fake, FAKE_DECIDER_NET_PURPOSE="")
    assert p.returncode == 0, p.out
    assert "network rm dgxmm-decider" not in box.calls()


def test_uninstall_purge_keeps_the_decider_network_of_another_install(box):
    # same purpose label, but created by another install (label dgxmm.install differs): reported, kept
    fake = _installed_with_decider_net(box)
    p = box.run("uninstall", "--purge", **fake, FAKE_DECIDER_NET_INSTALL="other0install")
    assert p.returncode == 0, p.out
    assert "network rm dgxmm-decider" not in box.calls()
    assert "kept network dgxmm-decider" in p.out


def test_uninstall_purge_keeps_an_unlabelled_decider_network(box):
    # created before install labels (no dgxmm.install label): reported, kept
    fake = _installed_with_decider_net(box)
    p = box.run("uninstall", "--purge", **fake, FAKE_DECIDER_NET_INSTALL="")
    assert p.returncode == 0, p.out
    assert "network rm dgxmm-decider" not in box.calls() and "kept network dgxmm-decider" in p.out


def test_uninstall_purge_without_an_install_id_keeps_the_decider_network(box):
    fake = _installed(box)  # .env without DGXMM_INSTALL_ID
    p = box.run("uninstall", "--purge", **fake, FAKE_DECIDER_NET="exists")
    assert p.returncode == 0, p.out
    assert "network rm dgxmm-decider" not in box.calls()


def test_uninstall_keeps_the_decider_network_without_purge_and_in_dry_run(box):
    fake = _installed_with_decider_net(box)
    assert box.run("uninstall", "--purge", "--dry-run", **fake).returncode == 0
    assert "docker network rm dgxmm-decider" not in box.calls()
    assert box.run("uninstall", **fake).returncode == 0
    assert "network rm dgxmm-decider" not in box.calls()


def test_uninstall_help_mentions_the_decider_network(box):
    p = box.run("--help")
    assert "decider" in p.out and "network" in p.out


def test_uninstall_refuses_other_checkouts_container(box):
    fake = _installed(box)
    fake["FAKE_DGXMM_WORKDIR"] = "/srv/other"
    p = box.run("uninstall", **fake)
    assert p.returncode != 0 and "/srv/other" in p.out
    assert "docker rm" not in box.calls()


def test_uninstall_dry_run_changes_nothing(box):
    fake = _installed(box)
    before = box.snapshot()
    p = box.run("uninstall", "--purge", "--dry-run", **fake)
    assert p.returncode == 0, p.out
    assert box.snapshot() == before
    calls = box.calls()
    assert "docker rm" not in calls and "image rm" not in calls


# --- portability ---------------------------------------------------------------------------------

def test_compose_is_parameterised():
    text = (ROOT / "docker-compose.yaml").read_text()
    assert "/home/" not in text
    for var in ("${DGXMM_UID", "${DGXMM_GID", "${DOCKER_GID", "${DGXMM_BIND:-0.0.0.0}", "${DGXMM_PORT:-8090}",
                "${DGXMM_HF_CACHE", "${DGXMM_MODELS_DIR", "${DGXMM_NETWORK:-dgxmm}", "${DGXMM_HOST_HOME"):
        assert var in text


def test_no_hardcoded_home_in_tracked_files():
    if not (ROOT / ".git").exists() or not shutil.which("git"):
        pytest.skip("not a git checkout")
    needle = "/home/" + "nikghodki"  # split so this file does not match itself
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    bad = [f for f in files if not f.startswith("docs/superpowers/") and (ROOT / f).is_file()
           and needle in (ROOT / f).read_text(errors="ignore")]
    assert bad == []


def test_install_script_is_executable():
    assert os.access(ROOT / "install.sh", os.X_OK)


@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not installed (run `make lint`)")
def test_shellcheck_clean():
    subprocess.run(["shellcheck", str(ROOT / "install.sh")], check=True)


# --- review fix round 1 -------------------------------------------------------------------------

def test_port_busy_detected_with_large_ss_output(box):
    # grep -q closing the pipe early must not make a busy port look free (SIGPIPE under pipefail)
    p = box.run("--dry-run", FAKE_PORT_BUSY="8090", FAKE_SS_FILLER="30000")
    assert p.returncode != 0 and "port 8090 is already in use" in p.out


def test_port_busy_detected_ipv6_listener(box):
    p = box.run("--dry-run", FAKE_PORT_BUSY="8090", FAKE_SS_V6="1")
    assert p.returncode != 0 and "port 8090 is already in use" in p.out


def test_other_port_not_reported_busy(box):
    p = box.run("--dry-run", FAKE_PORT_BUSY="18090")
    assert p.returncode == 0, p.out


def test_compose_up_failure_has_hint(box):
    p = box.run(FAKE_COMPOSE_UP="fail")
    assert p.returncode != 0
    assert "docker compose up failed" in p.out and "./install.sh logs" in p.out


def test_engine_pull_failure_has_hint(box):
    p = box.run(DGXMM_PULL_ENGINES="1", FAKE_PULL="fail")
    assert p.returncode != 0
    assert "could not pull" in p.out and "compose up" not in box.calls()


def test_stale_password_not_captured_when_container_not_recreated(box):
    p = box.run(FAKE_RECREATE="0", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()),
                FAKE_LOGS=FIRST_RUN)
    assert p.returncode == 0, p.out
    assert PW not in p.out and not (box.repo / "secrets" / "admin_password").exists()
    assert "docs/operations/troubleshooting.md" in p.out


def test_password_line_older_than_this_run_not_captured(box):
    p = box.run(FAKE_LOGS=FIRST_RUN, FAKE_LOGS_OLD="1")
    assert p.returncode == 0, p.out
    assert PW not in p.out and not (box.repo / "secrets" / "admin_password").exists()
    assert "docs/operations/troubleshooting.md" in p.out


def test_password_captured_with_since_this_run(box):
    p = box.run(FAKE_LOGS=FIRST_RUN)
    assert p.returncode == 0, p.out
    assert "docker logs --since " in box.calls()
    assert p.out.count(PW) == 1


@pytest.mark.parametrize("value", ["abc", "0", "-5", "1.5", ""])
def test_health_timeout_must_be_positive_integer(box, value):
    p = box.run("--dry-run", DGXMM_HEALTH_TIMEOUT=value)
    if value == "":
        assert p.returncode == 0, p.out  # unset/empty -> default
    else:
        assert p.returncode != 0 and "DGXMM_HEALTH_TIMEOUT" in p.out


@pytest.mark.parametrize("value", ["0", "abc", "-1", "65536"])
def test_port_must_be_valid(box, value):
    p = box.run("--dry-run", DGXMM_PORT=value)
    assert p.returncode != 0 and "DGXMM_PORT" in p.out


def test_mdns_url_printed_after_ip_urls(box):
    p = box.run()
    assert p.returncode == 0, p.out
    assert "http://spark-test.local:8090/dgxmm/" in p.out
    assert p.out.index("http://192.0.2.20:8090/dgxmm/") < p.out.index("http://spark-test.local:8090/dgxmm/")


def test_compose_healthcheck_works_on_old_engines():
    text = (ROOT / "docker-compose.yaml").read_text()
    assert "start_interval" not in text and "start_period" in text


def test_trusted_proxies_env_accepted(box):
    """DGXMM_TRUSTED_PROXIES env var is accepted and doesn't break install."""
    p = box.run("--dry-run", DGXMM_TRUSTED_PROXIES="172.23.0.0/16")
    assert p.returncode == 0, p.out


def test_trusted_proxies_written_to_env(box):
    """DGXMM_TRUSTED_PROXIES is written to .env when provided."""
    p = box.run("--dry-run", DGXMM_TRUSTED_PROXIES="198.51.100.1")
    assert p.returncode == 0, p.out
    assert "DGXMM_TRUSTED_PROXIES=198.51.100.1" in p.out


def test_trusted_proxies_hint_when_bind_loopback(box):
    """Installer prints hint when BIND=127.0.0.1 and TRUSTED_PROXIES unset."""
    p = box.run("--dry-run", DGXMM_BIND="127.0.0.1", FAKE_NETWORK="exists")
    assert p.returncode == 0, p.out
    # Should print specific hint about reverse proxy
    assert "Behind a reverse proxy?" in p.out
    assert "DGXMM_TRUSTED_PROXIES" in p.out
    assert "172.99.0.1" in p.out  # stub gateway IP from fake docker
    # Hint should not have [dry-run] prefix
    assert "[dry-run] Behind a reverse proxy?" not in p.out


def test_trusted_proxies_hint_not_shown_when_set(box):
    """No hint when DGXMM_TRUSTED_PROXIES is already set."""
    p = box.run("--dry-run", DGXMM_BIND="127.0.0.1", DGXMM_TRUSTED_PROXIES="198.51.100.1", FAKE_NETWORK="exists")
    assert p.returncode == 0, p.out
    assert "Behind a reverse proxy?" not in p.out


def test_trusted_proxies_hint_not_shown_for_lan_bind(box):
    """No hint when BIND is 0.0.0.0 (LAN accessible)."""
    p = box.run("--dry-run", DGXMM_BIND="0.0.0.0")
    assert p.returncode == 0, p.out
    assert "Behind a reverse proxy?" not in p.out


def test_trusted_proxies_from_existing_env_file_suppresses_hint(box):
    """A value saved in .env counts as set: no hint, and compose receives it."""
    (box.repo / ".env").write_text("DGXMM_BIND=127.0.0.1\nDGXMM_TRUSTED_PROXIES=172.99.0.1\n")
    p = box.run("--dry-run", FAKE_NETWORK="exists")
    assert p.returncode == 0, p.out
    assert "Behind a reverse proxy?" not in p.out
    assert "DGXMM_TRUSTED_PROXIES=" not in p.out  # already present, nothing appended


def test_trusted_proxies_env_appended_when_missing_from_env_file(box):
    """Env value is appended to an existing .env that lacks the key (same rule as other keys)."""
    (box.repo / ".env").write_text("DGXMM_BIND=127.0.0.1\n")
    p = box.run("--dry-run", DGXMM_TRUSTED_PROXIES="198.51.100.1", FAKE_NETWORK="exists")
    assert p.returncode == 0, p.out
    assert "| DGXMM_TRUSTED_PROXIES=198.51.100.1" in p.out
    assert "Behind a reverse proxy?" not in p.out


def test_trusted_proxies_existing_env_file_wins_over_environment(box):
    """Like DGXMM_PORT: an existing .env value is kept, with a warning, over a different env value."""
    (box.repo / ".env").write_text("DGXMM_TRUSTED_PROXIES=172.99.0.1\n")
    p = box.run("--dry-run", DGXMM_TRUSTED_PROXIES="198.51.100.1")
    assert p.returncode == 0, p.out
    assert "keeping .env" in p.out


# --- bundled LiteLLM (--with-litellm) -----------------------------------------------------------

LITELLM_SECRETS = ("litellm_bundled_master_key", "litellm_salt_key", "litellm_db_password")


def _copy_litellm_files(box):
    (box.repo / "deploy" / "litellm").mkdir(parents=True)
    shutil.copy2(ROOT / "deploy" / "litellm" / "config.yaml", box.repo / "deploy" / "litellm" / "config.yaml")


def test_help_documents_litellm(box):
    p = box.run("--help")
    assert p.returncode == 0
    for word in ("--with-litellm", "DGXMM_WITH_LITELLM", "DGXMM_LITELLM_PORT", "--litellm-port"):
        assert word in p.stdout


def test_with_litellm_install_generates_secrets_env_and_uses_profile(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", FAKE_LOGS=FIRST_RUN)
    assert p.returncode == 0, p.out
    repo, sec = box.repo, box.repo / "secrets"
    for name in LITELLM_SECRETS:
        assert mode(sec / name) == 0o600
        value = (sec / name).read_text()
        assert len(value) >= 32 and value not in p.out
    master = (sec / "litellm_bundled_master_key").read_text()
    assert master.startswith("sk-") and len(master) >= 35 and master.isascii()
    # Fresh config wired to the bundled LiteLLM: DGXMM gets the same key, zero touch.
    assert (sec / "litellm_master_key").read_text() == master and mode(sec / "litellm_master_key") == 0o600
    assert len(set((sec / n).read_text() for n in LITELLM_SECRETS)) == 3
    env = (repo / ".env").read_text()
    assert "DGXMM_WITH_LITELLM=1" in env and "DGXMM_LITELLM_PORT=4000" in env and "COMPOSE_PROFILES=litellm" in env
    assert "litellm_url: http://dgxmm-litellm:4000" in (repo / "config.yaml").read_text()
    db = repo / "data" / "litellm-db"
    assert db.is_dir() and mode(db) == 0o700
    calls = box.calls()
    assert "docker compose --profile litellm up -d --build" in calls
    # LiteLLM health is checked after DGXMM's, and the single endpoint is printed.
    assert "http://192.0.2.20:4000/v1" in p.out and "LiteLLM is running" in p.out
    assert "secrets/litellm_master_key" in p.out and "LiteLLM page" in p.out
    assert p.out.index("DGXMM is running") < p.out.index("LiteLLM is running")


def test_with_litellm_profile_remembered_and_secrets_kept(box):
    _copy_litellm_files(box)
    assert box.run("--with-litellm").returncode == 0
    sec = box.repo / "secrets"
    before = {n: (sec / n).read_text() for n in LITELLM_SECRETS}
    p = box.run(FAKE_NETWORK="exists", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()))
    assert p.returncode == 0, p.out
    assert {n: (sec / n).read_text() for n in LITELLM_SECRETS} == before
    assert box.calls().count("docker compose --profile litellm up -d --build") == 2
    assert "port 4000 (in use by this DGXMM's LiteLLM" in p.out


def test_with_litellm_keeps_existing_bundled_secrets(box):
    _copy_litellm_files(box)
    sec = box.repo / "secrets"
    sec.mkdir()
    (sec / "litellm_bundled_master_key").write_text("sk-existing-bundled-key-value")
    (sec / "litellm_salt_key").write_text("sk-existing-salt")
    p = box.run("--with-litellm")
    assert p.returncode == 0, p.out
    assert (sec / "litellm_bundled_master_key").read_text() == "sk-existing-bundled-key-value"
    assert (sec / "litellm_master_key").read_text() == "sk-existing-bundled-key-value"  # wired, fresh config
    assert (sec / "litellm_salt_key").read_text() == "sk-existing-salt"
    assert len((sec / "litellm_db_password").read_text()) >= 32
    assert "sk-existing" not in p.out


def test_with_litellm_existing_external_key_is_not_reused(box):
    """An existing install's key belongs to the external LiteLLM: never make it the bundled admin key."""
    _copy_litellm_files(box)
    sec = box.repo / "secrets"
    sec.mkdir()
    (sec / "litellm_master_key").write_text("sk-external-production-key")
    (box.repo / "config.yaml").write_text("docker_network: dgxmm\nlitellm_url: http://litellm-proxy:4000\n")
    p = box.run("--with-litellm", FAKE_LITELLM="1", FAKE_LITELLM_KEY="sk-external-production-key")
    assert p.returncode == 0, p.out
    assert (sec / "litellm_master_key").read_text() == "sk-external-production-key"
    bundled = (sec / "litellm_bundled_master_key").read_text()
    assert bundled.startswith("sk-") and bundled != "sk-external-production-key"
    assert "sk-external" not in p.out and bundled not in p.out
    assert "secrets/litellm_bundled_master_key" in p.out and "Integrations" in p.out
    assert "litellm_url to http://dgxmm-litellm:4000" in p.out


def test_with_litellm_fresh_config_external_key_env_not_used_for_dgxmm_when_external_url(box):
    """LITELLM_URL pointing elsewhere: DGXMM keeps the external key; the bundled key is separate."""
    _copy_litellm_files(box)
    p = box.run("--with-litellm", LITELLM_URL="http://litellm-proxy:4000", LITELLM_MASTER_KEY="sk-ext-from-env")
    assert p.returncode == 0, p.out
    sec = box.repo / "secrets"
    assert (sec / "litellm_master_key").read_text() == "sk-ext-from-env"
    assert (sec / "litellm_bundled_master_key").read_text() != "sk-ext-from-env"
    assert "litellm_url: http://litellm-proxy:4000" in (box.repo / "config.yaml").read_text()


def test_with_litellm_from_env_var_and_custom_port(box):
    _copy_litellm_files(box)
    p = box.run(DGXMM_WITH_LITELLM="1", DGXMM_LITELLM_PORT="4100")
    assert p.returncode == 0, p.out
    env = (box.repo / ".env").read_text()
    assert "DGXMM_WITH_LITELLM=1" in env and "DGXMM_LITELLM_PORT=4100" in env
    assert "http://192.0.2.20:4100/v1" in p.out


def test_without_litellm_nothing_changes(box):
    p = box.run()
    assert p.returncode == 0, p.out
    env = (box.repo / ".env").read_text()
    assert "LITELLM" not in env and "COMPOSE_PROFILES" not in env
    assert "--profile" not in box.calls()
    assert not (box.repo / "secrets" / "litellm_salt_key").exists()
    assert not (box.repo / "data" / "litellm-db").exists()
    assert "litellm_url: http://litellm-proxy:4000" in (box.repo / "config.yaml").read_text()


def test_with_litellm_dry_run_writes_nothing(box):
    _copy_litellm_files(box)
    before = box.snapshot()
    p = box.run("--with-litellm", "--dry-run")
    assert p.returncode == 0, p.out
    assert box.snapshot() == before
    assert "DGXMM_WITH_LITELLM=1" in p.out and "COMPOSE_PROFILES=litellm" in p.out
    assert "litellm_url: http://dgxmm-litellm:4000" in p.out
    assert "would write secrets/litellm_salt_key" in p.out
    assert "--profile litellm up -d --build" in p.out
    assert " up -d" not in box.calls() and "network create" not in box.calls()


def test_with_litellm_existing_config_pointing_elsewhere_is_kept(box):
    _copy_litellm_files(box)
    cfg = "docker_network: dgxmm\nlitellm_url: http://litellm-proxy:4000\n"
    (box.repo / "config.yaml").write_text(cfg)
    p = box.run("--with-litellm")
    assert p.returncode == 0, p.out
    assert (box.repo / "config.yaml").read_text() == cfg
    assert "keeps litellm_url: http://litellm-proxy:4000" in p.out


@pytest.mark.parametrize("line", ['litellm_url: "http://dgxmm-litellm:4000"', "litellm_url: 'http://dgxmm-litellm:4000'"])
def test_with_litellm_quoted_internal_url_counts_as_bundled(box, line):
    _copy_litellm_files(box)
    (box.repo / "config.yaml").write_text(f"docker_network: dgxmm\n{line}\n")
    p = box.run("--with-litellm")
    assert p.returncode == 0, p.out
    assert "keeps litellm_url" not in p.out
    sec = box.repo / "secrets"
    assert (sec / "litellm_master_key").read_text() == (sec / "litellm_bundled_master_key").read_text()


def test_with_litellm_port_conflict(box):
    p = box.run("--with-litellm", "--dry-run", FAKE_PORT_BUSY="4000")
    assert p.returncode != 0
    assert "port 4000 (LiteLLM) is already in use" in p.out and "DGXMM_LITELLM_PORT" in p.out


def test_with_litellm_same_port_as_ui_rejected(box):
    p = box.run("--with-litellm", "--dry-run", "--litellm-port", "8090")
    assert p.returncode != 0 and "must differ" in p.out


def test_with_litellm_bad_port_rejected(box):
    p = box.run("--with-litellm", "--dry-run", DGXMM_LITELLM_PORT="70000")
    assert p.returncode != 0 and "DGXMM_LITELLM_PORT" in p.out


def test_with_litellm_warns_about_external_litellm_proxy(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", "--dry-run", FAKE_LITELLM="1")
    assert p.returncode == 0, p.out
    assert "litellm-proxy is already running" in p.out and "second" in p.out
    p = box.run("--with-litellm", "--dry-run", FAKE_LITELLM="1", FAKE_PORT_BUSY="4000")
    assert p.returncode != 0 and "port 4000 (LiteLLM) is already in use" in p.out


def test_with_litellm_requires_litellm_config_file(box):
    p = box.run("--with-litellm", "--dry-run")
    assert p.returncode != 0 and "deploy/litellm/config.yaml" in p.out


def test_with_litellm_unhealthy_fails_with_hint(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", FAKE_LITELLM_HEALTH="running/unhealthy",
                FAKE_LITELLM_LOGS="boom sk-leakedkeyvalue123")
    assert p.returncode != 0
    assert "unhealthy" in p.out and "DGXMM itself is running" in p.out
    assert "sk-leakedkeyvalue123" not in p.out


def test_status_shows_litellm_containers(box):
    (box.repo / ".env").write_text("DGXMM_WITH_LITELLM=1\nDGXMM_LITELLM_PORT=4000\n")
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()), FAKE_BUNDLED="1")
    assert p.returncode == 0, p.out
    assert "repo-dgxmm-litellm-1" in p.out and "repo-dgxmm-litellm-db-1" in p.out
    assert ":4000/v1" in p.out


def test_status_flags_unhealthy_litellm(box):
    (box.repo / ".env").write_text("DGXMM_WITH_LITELLM=1\n")
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()),
                FAKE_BUNDLED="1", FAKE_LITELLM_HEALTH="exited/")
    assert p.returncode != 0


def _installed_with_litellm(box):
    fake = _installed(box)
    (box.repo / "data" / "litellm-db").mkdir()
    (box.repo / "data" / "litellm-db" / "PG_VERSION").write_text("16")
    (box.repo / "secrets" / "litellm_salt_key").write_text("s")
    (box.repo / ".env").write_text("DGXMM_NETWORK=dgxmm\nDGXMM_WITH_LITELLM=1\nCOMPOSE_PROFILES=litellm\n")
    fake["FAKE_BUNDLED"] = "1"
    return fake


def test_uninstall_removes_litellm_containers_keeps_db(box):
    fake = _installed_with_litellm(box)
    p = box.run("uninstall", **fake)
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "docker stop -t 30 id-dgxmm-litellm id-dgxmm-litellm-db" in calls
    assert "docker rm id-dgxmm-litellm id-dgxmm-litellm-db" in calls
    assert (box.repo / "data" / "litellm-db" / "PG_VERSION").exists()
    assert (box.repo / "secrets" / "litellm_salt_key").exists()
    assert "data/litellm-db" in p.out


def test_uninstall_purge_removes_litellm_db(box):
    fake = _installed_with_litellm(box)
    p = box.run("uninstall", "--purge", **fake)
    assert p.returncode == 0, p.out
    assert not (box.repo / "data").exists() and not (box.repo / "secrets").exists()
    assert "docker rm id-dgxmm-litellm id-dgxmm-litellm-db" in box.calls()


def test_uninstall_purge_dry_run_keeps_litellm_db(box):
    fake = _installed_with_litellm(box)
    before = box.snapshot()
    p = box.run("uninstall", "--purge", "--dry-run", **fake)
    assert p.returncode == 0, p.out
    assert box.snapshot() == before
    assert "docker rm" not in box.calls() and "docker stop" not in box.calls()


def test_compose_litellm_profile_is_pinned_and_parameterised():
    text = (ROOT / "docker-compose.yaml").read_text()
    assert "profiles: [litellm]" in text
    assert "dgxmm-litellm:" in text and "dgxmm-litellm-db:" in text
    assert "ghcr.io/berriai/litellm:v1." in text and "@sha256:" in text and "main-latest" not in text
    assert "image: postgres:16." in text and "@sha256:" in text.split("image: postgres:16.")[1].split("\n")[0]
    assert "${DGXMM_BIND:-0.0.0.0}:${DGXMM_LITELLM_PORT:-4000}:4000" in text
    assert "./data/litellm-db:/var/lib/postgresql/data" in text
    # Only the dgxmm service has a fixed container name; profile services derive theirs from the project.
    assert text.count("container_name:") == 1
    cfg = (ROOT / "deploy" / "litellm" / "config.yaml").read_text()
    assert "store_model_in_db: true" in cfg and "os.environ/LITELLM_MASTER_KEY" in cfg
    assert "os.environ/DATABASE_URL" in cfg


def _compose_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _compose_available(), reason="docker compose not available")
@pytest.mark.parametrize("profile", [[], ["--profile", "litellm"]])
def test_compose_file_validates(tmp_path, profile):
    shutil.copy2(ROOT / "docker-compose.yaml", tmp_path / "docker-compose.yaml")
    (tmp_path / ".env").write_text(
        "DGXMM_UID=1000\nDGXMM_GID=1000\nDOCKER_GID=988\nDGXMM_HOST_HOME=/tmp\nDGXMM_HF_CACHE=/tmp/hf\n"
        "DGXMM_MODELS_DIR=/tmp/models\nDGXMM_NETWORK=dgxmmtest\nDGXMM_LITELLM_PORT=14000\nDGXMM_BIND=127.0.0.1\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DGXMM_", "COMPOSE_"))}
    p = subprocess.run(["docker", "compose", *profile, "config", "-q"], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    services = subprocess.run(["docker", "compose", *profile, "config", "--services"], cwd=tmp_path, env=env,
                              capture_output=True, text=True, check=True).stdout.split()
    expected = {"dgxmm", "dgxmm-litellm", "dgxmm-litellm-db"} if profile else {"dgxmm"}
    assert set(services) == expected


# --- review fix round 1 (bundled LiteLLM) ---------------------------------------------------------

def test_with_litellm_compose_up_failure_still_shows_password(box):
    """A DB that never gets healthy fails `compose up`; DGXMM is up, so finish its setup first."""
    _copy_litellm_files(box)
    p = box.run("--with-litellm", FAKE_LOGS=FIRST_RUN, FAKE_COMPOSE_UP="fail-litellm",
                FAKE_LITELLM_LOGS="db: FATAL postgresql://litellm:hunter2pw@dgxmm-litellm-db/litellm sk-abcdef123456")
    assert p.returncode != 0
    assert p.out.count(PW) == 1 and (box.repo / "secrets" / "admin_password").read_text().strip() == PW
    assert "DGXMM is running" in p.out
    assert "bundled LiteLLM" in p.out and "logs --tail" in p.out
    assert "docker compose --profile litellm logs --tail 30 dgxmm-litellm-db dgxmm-litellm" in box.calls()
    assert "hunter2pw" not in p.out and "sk-abcdef123456" not in p.out and "://***@" in p.out
    assert p.out.index(PW) < p.out.index("did not start")


def test_with_litellm_compose_up_failure_without_dgxmm_dies(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", FAKE_COMPOSE_UP="fail", FAKE_HEALTH="missing/")
    assert p.returncode != 0 and "docker compose up failed" in p.out


def test_with_litellm_crash_loop_fails_early(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", FAKE_LITELLM_HEALTH="restarting/", FAKE_LITELLM_RESTARTS="4",
                DGXMM_LITELLM_HEALTH_TIMEOUT="300", FAKE_LITELLM_LOGS="crash")
    assert p.returncode != 0
    assert "restarted 4 times" in p.out and "crash" in p.out


def test_compose_profiles_ignores_stray_shell_value(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", COMPOSE_PROFILES="something-else")
    assert p.returncode == 0, p.out
    assert "COMPOSE_PROFILES=litellm" in (box.repo / ".env").read_text()


def test_status_accepts_truthy_with_litellm(box):
    (box.repo / ".env").write_text("DGXMM_WITH_LITELLM=true\n")
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()), FAKE_BUNDLED="1")
    assert p.returncode == 0, p.out
    assert "LiteLLM endpoint" in p.out


def test_disabling_bundle_stops_containers_and_drops_profile(box):
    (box.repo / ".env").write_text("DGXMM_WITH_LITELLM=0\nCOMPOSE_PROFILES=litellm,extra\n")
    p = box.run(FAKE_BUNDLED="1")
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "docker stop -t 30 id-dgxmm-litellm id-dgxmm-litellm-db" in calls
    assert "docker rm id-dgxmm-litellm id-dgxmm-litellm-db" in calls
    assert "--profile" not in calls
    assert "COMPOSE_PROFILES=extra" in (box.tmp / "calls.log.profiles").read_text()
    assert "DGXMM_WITH_LITELLM=0" in p.out and "data/litellm-db" in p.out


def test_disabled_without_bundled_containers_touches_nothing(box):
    p = box.run()
    assert p.returncode == 0, p.out
    assert "docker stop" not in box.calls()


def test_bundled_key_mounted_only_into_bundled_litellm():
    text = (ROOT / "docker-compose.yaml").read_text()
    assert "/run/secrets/litellm_bundled_master_key" in text
    litellm_svc = text.split("  dgxmm-litellm:\n", 1)[1].split("\nnetworks:", 1)[0]
    assert "secrets: [litellm_bundled_master_key, litellm_salt_key, litellm_db_password]" in litellm_svc
    dgxmm_svc = text.split("  dgxmm:\n", 1)[1].split("  dgxmm-litellm-db:", 1)[0]
    assert "litellm_bundled_master_key" not in dgxmm_svc


def _docs(*sections: str) -> str:
    """The text of documentation-site sections: docs/<s>.md and every page under docs/<s>/ (mkdocs.yml)."""
    docs = ROOT / "docs"
    files = [f for s in sections for f in [docs / f"{s}.md", *sorted((docs / s).glob("*.md"))] if f.is_file()]
    return "\n".join(f.read_text() for f in files)


def test_docs_cover_bundled_litellm():
    ops = _docs("operations", "getting-started")
    assert "pg_dump" in ops and "dgxmm-litellm-db" in ops and "data/litellm-db" in ops
    cfg = _docs("configuration")
    for word in ("DGXMM_WITH_LITELLM", "DGXMM_LITELLM_PORT", "COMPOSE_PROFILES", "Bundled LiteLLM",
                 "litellm_bundled_master_key"):
        assert word in cfg
    assert "--with-litellm" in (ROOT / "README.md").read_text()


# --- LiteLLM public URL seeding -------------------------------------------------------------------------

def _cfg_lines(box, key):
    return [l for l in (box.repo / "config.yaml").read_text().splitlines() if l.startswith(f"{key}:")]


def test_with_litellm_fresh_config_seeds_public_url(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm")
    assert p.returncode == 0, p.out
    assert _cfg_lines(box, "litellm_public_url") == ["litellm_public_url: http://192.0.2.20:4000"]


def test_with_litellm_public_url_uses_custom_port_and_bind_ip(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", "--bind", "192.0.2.20", DGXMM_LITELLM_PORT="4100")
    assert p.returncode == 0, p.out
    assert _cfg_lines(box, "litellm_public_url") == ["litellm_public_url: http://192.0.2.20:4100"]


def test_with_litellm_loopback_bind_does_not_seed_public_url(box):
    """127.0.0.1 is not reachable by clients; leave the public URL for the admin to set."""
    _copy_litellm_files(box)
    p = box.run("--with-litellm", "--bind", "127.0.0.1")
    assert p.returncode == 0, p.out
    assert _cfg_lines(box, "litellm_public_url") == []


def test_with_litellm_external_url_does_not_seed_public_url(box):
    _copy_litellm_files(box)
    p = box.run("--with-litellm", LITELLM_URL="http://litellm-proxy:4000")
    assert p.returncode == 0, p.out
    assert _cfg_lines(box, "litellm_public_url") == []


def test_with_litellm_dry_run_shows_public_url_and_writes_nothing(box):
    _copy_litellm_files(box)
    before = box.snapshot()
    p = box.run("--with-litellm", "--dry-run")
    assert p.returncode == 0, p.out
    assert box.snapshot() == before
    assert "litellm_public_url: http://192.0.2.20:4000" in p.out


def test_without_litellm_no_public_url(box):
    assert box.run().returncode == 0
    assert _cfg_lines(box, "litellm_public_url") == []


def test_with_litellm_existing_config_public_url_untouched(box):
    _copy_litellm_files(box)
    cfg = "docker_network: dgxmm\nlitellm_url: http://dgxmm-litellm:4000\n"
    (box.repo / "config.yaml").write_text(cfg)
    p = box.run("--with-litellm")
    assert p.returncode == 0, p.out
    assert (box.repo / "config.yaml").read_text() == cfg


def test_example_config_documents_public_url():
    text = (ROOT / "config.example.yaml").read_text()
    assert "# litellm_public_url:" in text


# --- decision model (--with-decider) ------------------------------------------------------------------

DECIDER_SNAP = ".cache/huggingface/hub/models--vllm-sr--d3-mini/snapshots/97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e"


def _meminfo(box, gib: float) -> dict[str, str]:
    path = box.tmp / "meminfo"
    path.write_text(f"MemTotal:       127000000 kB\nMemAvailable:   {int(gib * 1024 * 1024)} kB\n")
    return {"DGXMM_MEMINFO": str(path)}


def _decider_calls(box) -> list[str]:
    return [line.split("dgxmm.admin decider ", 1)[1] for line in box.calls().splitlines()
            if "dgxmm.admin decider " in line]


def test_help_documents_decider(box):
    p = box.run("--help")
    for word in ("--with-decider", "DGXMM_WITH_DECIDER", "--purge-decider-model", "DGXMM_DECIDER_TIMEOUT"):
        assert word in p.stdout


def test_with_decider_install_builds_downloads_enables_and_remembers(box):
    p = box.run("--with-decider", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert "DGXMM_WITH_DECIDER=1" in (box.repo / ".env").read_text()
    calls = _decider_calls(box)
    assert calls == ["build", "download", "enable --wait --timeout 900", "status"]
    log = box.calls()
    assert log.index("compose up") < log.index("dgxmm.admin decider build")  # after DGXMM is healthy
    assert "memory: 40 GiB available" in p.out and "Decision model" in p.out
    # remembered: a plain re-run keeps it on (each step is idempotent in the CLI), unless an admin disabled it
    box.log.write_text("")
    p = box.run(**_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert _decider_calls(box) == ["admin-disabled", "build", "download", "enable --wait --timeout 900", "status"]


def test_with_decider_from_env_and_custom_timeout(box):
    p = box.run(DGXMM_WITH_DECIDER="1", DGXMM_DECIDER_TIMEOUT="120", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert "enable --wait --timeout 120" in _decider_calls(box)


def test_without_decider_nothing_decider_happens(box):
    p = box.run(**_meminfo(box, 2))  # low memory does not matter without the decider
    assert p.returncode == 0, p.out
    assert _decider_calls(box) == [] and "DGXMM_WITH_DECIDER" not in (box.repo / ".env").read_text()


def test_with_decider_dry_run_writes_nothing(box):
    before = box.snapshot()
    p = box.run("--with-decider", "--dry-run", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert box.snapshot() == before and _decider_calls(box) == []
    for sub in ("build", "download", "enable --wait --timeout 900"):
        assert f"would run: docker exec dgxmm python -m dgxmm.admin decider {sub}" in p.out


@pytest.mark.parametrize("value", ["maybe", "2"])
def test_with_decider_value_validated(box, value):
    p = box.run("--dry-run", DGXMM_WITH_DECIDER=value)
    assert p.returncode != 0 and "DGXMM_WITH_DECIDER" in p.out


@pytest.mark.parametrize("value", ["0", "-5", "abc"])
def test_decider_timeout_validated(box, value):
    p = box.run("--dry-run", "--with-decider", DGXMM_DECIDER_TIMEOUT=value, **_meminfo(box, 40))
    assert p.returncode != 0 and "DGXMM_DECIDER_TIMEOUT" in p.out


def test_with_decider_preflight_needs_16_gib_free(box):
    p = box.run("--with-decider", **_meminfo(box, 12.5))
    assert p.returncode != 0
    assert "16 GiB" in p.out and "12 GiB" in p.out
    assert "compose up" not in box.calls()  # fails before changing anything


def test_with_decider_preflight_memory_skipped_when_decider_already_runs(box):
    p = box.run("--with-decider", FAKE_DECIDER_RUNNING="d1", **_meminfo(box, 3))
    assert p.returncode == 0, p.out
    assert "already running" in p.out


def _pinned_files() -> list[str]:
    pins = ROOT / "deploy" / "decider" / "d3-mini-97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e.sha256"
    return [line.split()[1] for line in pins.read_text().splitlines() if line.strip()]


def _with_pins(box) -> None:
    (box.repo / "deploy" / "decider").mkdir(parents=True, exist_ok=True)
    name = "d3-mini-97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e.sha256"
    shutil.copy2(ROOT / "deploy" / "decider" / name, box.repo / "deploy" / "decider" / name)


def test_with_decider_preflight_disk_for_the_model(box):
    _with_pins(box)
    df = box.bin / "df"
    df.write_text("#!/usr/bin/env bash\necho 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
                  "echo '/dev/x 100000000 99000000 4194304 99% /'\n")  # 4 GiB free
    df.chmod(0o755)
    p = box.run("--with-decider", **_meminfo(box, 40))
    assert p.returncode != 0 and "model snapshot" in p.out and "4 GiB" in p.out
    snap = box.home / DECIDER_SNAP
    snap.mkdir(parents=True)
    (snap / "MODEL_MANIFEST.json").write_text("{}")  # only the manifest: not a complete snapshot
    p = box.run("--with-decider", **_meminfo(box, 40))
    assert p.returncode != 0 and "model snapshot" in p.out
    for name in _pinned_files():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_text("x")
    p = box.run("--with-decider", **_meminfo(box, 40))
    assert p.returncode == 0, p.out  # every pinned file present: no space needed for it


def test_with_decider_image_disk_warning(box):
    df = box.bin / "df"
    df.write_text("#!/usr/bin/env bash\necho 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
                  "echo '/dev/x 100000000 50000000 20971520 50% /'\n")  # 20 GiB free
    df.chmod(0o755)
    p = box.run("--with-decider", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert "decider image" in p.out and "30 GiB" in p.out  # warning only: layers may already be cached
    box.log.write_text("")
    p = box.run("--with-decider", FAKE_DECIDER_IMAGE="present", **_meminfo(box, 40))
    assert "30 GiB" not in p.out


def test_with_decider_reads_image_tag_from_config(box):
    (box.repo / "config.yaml").write_text("docker_network: dgxmm\nimages:\n  vllm: vllm/vllm-openai:x\n"
                                          "  decision: custom-decider:7  # local tag\n")
    p = box.run("--with-decider", FAKE_DECIDER_IMAGE="present", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert "image inspect custom-decider:7" in box.calls()


def test_with_decider_not_enough_memory_at_enable_fails_with_hint(box):
    p = box.run("--with-decider", FAKE_DECIDER_ENABLE_RC="3", **_meminfo(box, 40))
    assert p.returncode != 0
    assert "does not fit" in p.out and "DGXMM itself is running" in p.out


def test_with_decider_timeout_fails_with_hint(box):
    p = box.run("--with-decider", FAKE_DECIDER_ENABLE_RC="5", **_meminfo(box, 40))
    assert p.returncode != 0 and "not healthy" in p.out and "decider status" in p.out


def test_with_decider_download_failure_says_rerun_resumes(box):
    p = box.run("--with-decider", FAKE_DECIDER_DOWNLOAD_RC="1", **_meminfo(box, 40))
    assert p.returncode != 0 and "resume" in p.out
    assert "decider enable" not in box.calls()


def test_with_decider_build_failure_stops_before_download(box):
    p = box.run("--with-decider", FAKE_DECIDER_BUILD_RC="1", **_meminfo(box, 40))
    assert p.returncode != 0 and "decider image" in p.out
    assert "decider download" not in box.calls()


def test_status_shows_decider(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()))
    assert p.returncode == 0, p.out
    assert "Decision model: enabled, healthy" in p.out
    assert "status --check" in box.calls()
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()),
                FAKE_DECIDER_STATUS="enabled, not healthy (loading, deployment 3)", FAKE_DECIDER_STATUS_RC="1")
    assert p.returncode == 1 and "not healthy" in p.out


def test_status_decider_not_requested_does_not_fail_status(box):
    p = box.run("status", FAKE_DGXMM_STATE="running", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()),
                FAKE_DECIDER_STATUS="disabled", FAKE_DECIDER_STATUS_RC="1")
    assert p.returncode == 0, p.out
    assert "Decision model: disabled" in p.out


def test_status_lists_decider_container_when_dgxmm_is_down(box):
    p = box.run("status", FAKE_DGXMM_STATE="exited", FAKE_DGXMM_WORKDIR=str(box.repo.resolve()),
                FAKE_DECIDER_IDS="d1")
    assert "dgxmm-decider-d1" in p.out and "decider status" not in box.calls()


def _installed_with_id(box, install_id="zz99yy88xx77"):
    fake = _installed(box)
    (box.repo / ".env").write_text(f"DGXMM_NETWORK=dgxmm\nDGXMM_INSTALL_ID={install_id}\n")
    return fake


def test_uninstall_removes_only_this_installs_decider_container_and_keeps_its_setting(box):
    fake = _installed_with_id(box)
    p = box.run("uninstall", **fake, FAKE_DECIDER_IDS="d1", FAKE_DECIDER_ALL_IDS="d1 dother")
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "ps -aq --filter label=dgxmm.system=true --filter label=dgxmm.install=zz99yy88xx77" in calls
    assert "docker rm -f d1" in calls and "dother" not in calls.replace("ps -aq", "")
    assert "decider disable" not in calls  # decider.enabled is not flipped: a reinstall brings it back
    assert calls.index("docker rm -f d1") < calls.index("docker stop -t 30 dgxmm")
    assert "c1" not in calls.replace("ps -aq", "")  # user engines still untouched


def test_uninstall_reports_a_decider_container_from_before_install_labels(box):
    fake = _installed_with_id(box)
    p = box.run("uninstall", **fake, FAKE_DECIDER_IDS="", FAKE_DECIDER_ALL_IDS="dlegacy")
    assert p.returncode == 0, p.out
    assert "rm -f dlegacy" not in box.calls() and "dlegacy" in p.out


def test_uninstall_purge_removes_only_this_installs_decider_images_and_keeps_model(box):
    fake = _installed_with_id(box)
    snap = box.home / DECIDER_SNAP
    snap.mkdir(parents=True)
    (snap / "MODEL_MANIFEST.json").write_text("{}")
    p = box.run("uninstall", "--purge", **fake, FAKE_DECIDER_IMAGE="present", FAKE_BUILT_REFS="mine-decider:1")
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "images --filter label=dgxmm.built=true --filter label=dgxmm.install=zz99yy88xx77" in calls
    assert "docker image rm mine-decider:1" in calls
    assert "image rm dgxmm-decider:3" not in calls and "dgxmm-decider:3" in p.out  # not ours: kept, reported
    assert snap.exists() and "--purge-decider-model" in p.out


def test_install_id_generated_once_and_kept(box):
    p = box.run(**_meminfo(box, 40))
    assert p.returncode == 0, p.out
    line = [x for x in (box.repo / ".env").read_text().splitlines() if x.startswith("DGXMM_INSTALL_ID=")]
    assert len(line) == 1 and len(line[0].split("=", 1)[1]) == 12
    p = box.run(**_meminfo(box, 40))
    assert [x for x in (box.repo / ".env").read_text().splitlines() if x.startswith("DGXMM_INSTALL_ID=")] == line


def test_compose_passes_the_install_id():
    assert "DGXMM_INSTALL_ID: \"${DGXMM_INSTALL_ID:-}\"" in (ROOT / "docker-compose.yaml").read_text()


def test_persisted_decider_low_memory_only_warns_and_lets_the_app_decide(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run(**_meminfo(box, 3))
    assert p.returncode == 0, p.out
    assert "WARN" in p.out and "16 GiB" in p.out
    assert "compose up" in box.calls() and "enable --wait --timeout 900" in _decider_calls(box)


def test_persisted_decider_failures_do_not_fail_the_upgrade(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run(FAKE_DECIDER_ENABLE_RC="3", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert "does not fit" in p.out and "WARN" in p.out


def test_persisted_decider_respects_an_admin_disable(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run(FAKE_DECIDER_ADMIN_DISABLED="0", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert _decider_calls(box) == ["admin-disabled"] and "--with-decider" in p.out


def test_explicit_with_decider_re_enables_after_an_admin_disable(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run("--with-decider", FAKE_DECIDER_ADMIN_DISABLED="0", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert _decider_calls(box) == ["build", "download", "enable --wait --timeout 900", "status"]


def test_decider_timeout_capped_by_the_cli_token_lifetime(box):
    p = box.run("--dry-run", "--with-decider", DGXMM_DECIDER_TIMEOUT="10201", **_meminfo(box, 40))
    assert p.returncode != 0 and "10200" in p.out
    assert box.run("--dry-run", "--with-decider", DGXMM_DECIDER_TIMEOUT="10200", **_meminfo(box, 40)).returncode == 0


def test_uninstall_purge_decider_model(box):
    fake = _installed(box)
    snap = box.home / DECIDER_SNAP
    snap.mkdir(parents=True)
    (snap / "MODEL_MANIFEST.json").write_text("{}")
    other = box.home / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-8B"
    other.mkdir(parents=True)
    p = box.run("uninstall", "--purge", "--dry-run", "--purge-decider-model", **fake)
    assert p.returncode == 0 and snap.exists()
    p = box.run("uninstall", "--purge", "--purge-decider-model", **fake)
    assert p.returncode == 0, p.out
    assert not (box.home / ".cache/huggingface/hub/models--vllm-sr--d3-mini").exists()
    assert other.exists() and (box.home / ".cache" / "huggingface" / "model.bin").exists()


def _hf_repo(hub, repo_dir, rev, files):
    """A repo dir in the huggingface_hub >= 2 layout: snapshot links -> repo blobs -> shared store hub/blobs/<xx>/."""
    snap = hub / repo_dir / "snapshots" / rev
    snap.mkdir(parents=True)
    (hub / repo_dir / "blobs").mkdir(exist_ok=True)
    for name, digest in files.items():
        store = hub / "blobs" / digest[:2] / digest
        store.parent.mkdir(parents=True, exist_ok=True)
        if not store.exists():
            store.write_text(name)
        blob = hub / repo_dir / "blobs" / ("b" + digest[:39])
        if not blob.is_symlink():
            blob.symlink_to(f"../../blobs/{digest[:2]}/{digest}")
        (snap / name).symlink_to(f"../../blobs/{blob.name}")
    return snap


def test_purge_decider_model_removes_every_decider_and_only_their_exclusive_blobs(box):
    fake = _installed(box)
    hub = box.home / ".cache" / "huggingface" / "hub"
    mini = _hf_repo(hub, "models--vllm-sr--d3-mini", "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e",
                    {"config.json": "a1" * 32, "shared.json": "c3" * 32})
    nano = _hf_repo(hub, "models--vllm-sr--d3-nano", "aeb70d4dd53ed25b855c3b0a66394b8fe265db7f",
                    {"model.safetensors": "b2" * 32, "tokenizer.json": "d4" * 32})
    other = _hf_repo(hub, "models--Qwen--Qwen3-8B", "e" * 40, {"tokenizer.json": "d4" * 32, "w.bin": "f5" * 32})
    p = box.run("uninstall", "--purge", "--dry-run", "--purge-decider-model", **fake)
    assert p.returncode == 0, p.out
    assert mini.exists() and nano.exists() and (hub / "blobs" / "b2" / ("b2" * 32)).exists()
    assert f"rm -f -- {hub}/blobs/b2/{'b2' * 32}" in p.out and f"blobs/d4/{'d4' * 32}" not in p.out
    p = box.run("uninstall", "--purge", "--purge-decider-model", **fake)
    assert p.returncode == 0, p.out
    assert not (hub / "models--vllm-sr--d3-mini").exists() and not (hub / "models--vllm-sr--d3-nano").exists()
    for d in ("a1", "c3", "b2"):  # used only by the decision models
        assert not (hub / "blobs" / d / (d * 32)).exists(), d
    for d in ("d4", "f5"):  # also (or only) used by another model
        assert (hub / "blobs" / d / (d * 32)).exists(), d
    assert (other / "tokenizer.json").read_text() == "tokenizer.json"
    assert "removed 2 decision model snapshot(s) and 3 shared blob(s)" in p.out


def test_installer_purges_every_supported_decision_model():
    from dgxmm.engines.decision import DECIDER_MODELS, repo_dir
    text = (ROOT / "install.sh").read_text()
    dirs = " ".join(repo_dir(r).removeprefix("hub/") for r in DECIDER_MODELS)
    assert f"DECIDER_REPO_DIRS=({dirs})" in text


def test_purge_decider_model_needs_purge(box):
    p = box.run("uninstall", "--purge-decider-model")
    assert p.returncode == 2 and "--purge" in p.out


def test_installer_pins_the_same_decider_revision_as_the_engine():
    from dgxmm.engines.decision import DECIDER_REPO, DECIDER_REVISION, repo_dir
    text = (ROOT / "install.sh").read_text()
    assert f"DECIDER_REVISION={DECIDER_REVISION}\n" in text
    assert f"DECIDER_REPO_DIR={repo_dir(DECIDER_REPO).removeprefix('hub/')}\n" in text
    from dgxmm.config import Images
    assert f"DECIDER_DEFAULT_IMAGE={Images().decision} " in text


def test_docs_cover_decider_install_upgrade_and_backup():
    ops = _docs("operations", "getting-started")
    for needle in ("--with-decider", "decider download", "decider enable --wait", "schema_version",
                   "decider_api_key", "--purge-decider-model", "Rolling back"):
        assert needle in ops, needle
    conf = _docs("configuration")
    assert "DGXMM_WITH_DECIDER" in conf and "DGXMM_DECIDER_TIMEOUT" in conf
    changelog = (ROOT / "CHANGELOG.md").read_text()
    # the decision-model release entry (the newest section) documents --with-decider
    assert changelog.index("## [2026-10-17]") < changelog.index("--with-decider") < changelog.index("## [2026-10-16]")



def test_uninstall_purge_keeps_managed_containers_of_other_installs(box):
    fake = _installed(box)
    (box.repo / ".env").write_text("DGXMM_NETWORK=dgxmm\nDGXMM_INSTALL_ID=zz99yy88xx77\n")
    fake["FAKE_MANAGED_IDS"] = "c1 c2 legacy9"
    p = box.run("uninstall", "--purge", **fake, FAKE_OWN_MANAGED_IDS="c1")
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "docker rm -f c1\n" in calls + "\n" and "rm -f c1 c2" not in calls and "legacy9" not in calls.split("ps -aq")[0]
    assert "c2 legacy9" in p.out and "kept" in p.out


def test_uninstall_purge_without_install_id_removes_no_engine_containers(box):
    fake = _installed(box)
    p = box.run("uninstall", "--purge", **fake)
    assert p.returncode == 0, p.out
    assert "docker rm -f c1" not in box.calls() and "c1 c2" in p.out


def test_purged_images_are_removed_by_tag_without_force(box):
    fake = _installed(box)
    (box.repo / ".env").write_text("DGXMM_NETWORK=dgxmm\nDGXMM_INSTALL_ID=zz99yy88xx77\n")
    p = box.run("uninstall", "--purge", **fake, FAKE_BUILT_REFS="dgxmm-decider:3 other-tag:1 <none>:<none>@sha256:dd")
    assert p.returncode == 0, p.out
    calls = box.calls()
    assert "image rm -f" not in calls
    assert "docker image rm dgxmm-decider:3" in calls and "docker image rm other-tag:1" in calls
    assert "docker image rm sha256:dd" in calls  # untagged: by id


def test_remembered_decider_with_unreadable_meminfo_only_warns(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run(DGXMM_MEMINFO=str(box.tmp / "missing"))
    assert p.returncode == 0, p.out
    assert "MemAvailable" in p.out and "WARN" in p.out


def test_explicit_decider_with_unreadable_meminfo_dies(box):
    p = box.run("--with-decider", DGXMM_MEMINFO=str(box.tmp / "missing"))
    assert p.returncode != 0 and "MemAvailable" in p.out


def test_remembered_decider_with_bad_timeout_only_warns_and_uses_the_default(box):
    (box.repo / ".env").write_text("DGXMM_WITH_DECIDER=1\n")
    p = box.run(DGXMM_DECIDER_TIMEOUT="99999", **_meminfo(box, 40))
    assert p.returncode == 0, p.out
    assert "DGXMM_DECIDER_TIMEOUT" in p.out and "enable --wait --timeout 900" in _decider_calls(box)


def test_unreadable_source_files_are_refused_before_anything_runs(box):
    (box.repo / "config.example.yaml").chmod(0o600)
    p = box.run("--dry-run")
    assert p.returncode != 0
    assert "not readable by the container user" in p.out
    assert "config.example.yaml" in p.out and "chmod -R o+rX" in p.out
    assert "docker" not in box.calls()


def test_private_data_and_secrets_dirs_are_not_flagged(box):
    for d in ("data", "secrets"):
        (box.repo / d).mkdir()
        f = box.repo / d / "x"
        f.write_text("x")
        f.chmod(0o600)
    p = box.run("--dry-run")
    assert "not readable by the container user" not in p.out


# --- host time zone (DGXMM_TZ: wall-clock schedules inside the container) -----------------------------
def _tz_env(box, *, tzfile=None, localtime=None):
    env = {"DGXMM_TIMEZONE_FILE": str(box.tmp / "no-timezone"), "DGXMM_LOCALTIME": str(box.tmp / "no-localtime")}
    if tzfile is not None:
        (box.tmp / "timezone").write_text(tzfile)
        env["DGXMM_TIMEZONE_FILE"] = str(box.tmp / "timezone")
    if localtime is not None:
        target = box.tmp / "usr/share/zoneinfo" / localtime
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"TZif")
        (box.tmp / "localtime").symlink_to(target)
        env["DGXMM_LOCALTIME"] = str(box.tmp / "localtime")
    return env


def test_tz_detected_from_etc_timezone(box):
    p = box.run("--dry-run", **_tz_env(box, tzfile="Europe/Berlin\n"))
    assert p.returncode == 0, p.out
    assert "DGXMM_TZ=Europe/Berlin" in p.out


def test_tz_detected_from_the_localtime_link(box):
    p = box.run("--dry-run", **_tz_env(box, localtime="America/Los_Angeles"))
    assert p.returncode == 0, p.out
    assert "DGXMM_TZ=America/Los_Angeles" in p.out


def test_tz_defaults_to_utc(box):
    p = box.run("--dry-run", **_tz_env(box))
    assert p.returncode == 0, p.out
    assert "DGXMM_TZ=UTC" in p.out


def test_tz_from_env_and_validated(box):
    p = box.run("--dry-run", DGXMM_TZ="Asia/Tokyo", **_tz_env(box, tzfile="Europe/Berlin\n"))
    assert p.returncode == 0 and "DGXMM_TZ=Asia/Tokyo" in p.out
    for bad in ("../../etc/passwd", "Europe/Ber lin", "x;rm"):
        p = box.run("--dry-run", DGXMM_TZ=bad, **_tz_env(box))
        assert p.returncode != 0 and "DGXMM_TZ" in p.out, bad
    p = box.run("--dry-run", **_tz_env(box, tzfile="not a zone!\n"))  # junk in the host file: UTC
    assert p.returncode == 0 and "DGXMM_TZ=UTC" in p.out


def test_tz_written_to_env_and_kept_on_upgrade(box):
    p = box.run(FAKE_LOGS=FIRST_RUN, **_tz_env(box, tzfile="Europe/Berlin\n"))
    assert p.returncode == 0, p.out
    assert "DGXMM_TZ=Europe/Berlin" in (box.repo / ".env").read_text()
    p2 = box.run(FAKE_LOGS="", FAKE_NETWORK="exists", FAKE_DGXMM_STATE="running",
                 FAKE_DGXMM_WORKDIR=str(box.repo.resolve()), **_tz_env(box, tzfile="Asia/Tokyo\n"))
    assert p2.returncode == 0, p2.out
    assert "DGXMM_TZ=Europe/Berlin" in (box.repo / ".env").read_text()


def test_tz_added_to_an_older_env(box):
    (box.repo / ".env").write_text("DGXMM_UID=1234\nDGXMM_GID=2345\nDOCKER_GID=977\n")
    p = box.run("--dry-run", **_tz_env(box, tzfile="Europe/Berlin\n"))
    assert p.returncode == 0, p.out
    assert "would append missing keys" in p.out and "DGXMM_TZ=Europe/Berlin" in p.out


def test_compose_passes_the_zone_to_the_container():
    compose = (ROOT / "docker-compose.yaml").read_text()
    assert 'DGXMM_TZ: "${DGXMM_TZ:-}"' in compose and 'TZ: "${DGXMM_TZ:-UTC}"' in compose


def test_compose_shares_the_host_pid_namespace_for_nvml():
    """NVML only lists processes of the caller's PID namespace. Without `pid: host` DGXMM sees no per-process
    GPU memory of its own engine containers and their CUDA memory reads as reservation credit / external."""
    text = (ROOT / "docker-compose.yaml").read_text()
    dgxmm = text.split("\n  dgxmm-litellm-db:")[0]
    assert "\n    pid: host" in dgxmm
    assert text.count("pid: host") == 1  # only the DGXMM service
