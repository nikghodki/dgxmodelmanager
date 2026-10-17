"""H3 hardening of the decider container: key delivered as a file (never env / inspect), internal network without a
published port, non-root read-only container with dropped capabilities and cgroup limits, a single read-only mount."""
import pytest

from dgxmm.config import GIB, Mount
from dgxmm.docker_ctl import DockerController, SecretDeliveryFailed
from dgxmm.engines.base import ContainerSpec, LaunchContext, managed_labels
from dgxmm.engines.decision import DECIDER_REPO, KEY_FILE, KEY_DIR
from tests.fakes import FakeDockerClient
from tests.test_decider_engine import d3, deploy_decider  # noqa: F401  (fixture)


def spec_for(stack):
    eng = stack.deployer.engines["decision"]
    art = stack.inv.get(DECIDER_REPO)
    return eng.container_spec(art, eng.validate_config({}), None, LaunchContext(7, "decider", 8123, 13 * GIB, 121 * GIB),
                              stack.settings)


# ---------- engine spec ----------
def test_spec_never_puts_the_key_in_env_or_command(d3):
    spec = spec_for(d3)
    key = d3.settings.decider_key_path.read_text()
    assert len(key) >= 32
    assert "DECISION_API_KEY" not in spec.environment
    assert spec.environment["DECISION_API_KEY_FILE"] == KEY_FILE and KEY_FILE.startswith(KEY_DIR + "/")
    assert key not in repr(spec) and key not in " ".join(spec.command) and key not in str(spec.environment)
    assert spec.secret_files == {KEY_FILE: key.encode()}


def test_spec_is_hardened(d3):
    spec, s = spec_for(d3), d3.settings
    assert spec.user == "10001:10001" and spec.read_only
    assert spec.cap_drop == ["ALL"] and "no-new-privileges" in spec.security_opt
    assert spec.pids_limit == s.decider_pids_limit and spec.mem_limit == int(s.decider_memory_limit_gib * GIB)
    assert not spec.ipc_host and spec.gpus
    # writable paths are tmpfs only; the key directory is private to the decider user and not executable
    assert set(spec.tmpfs) == {"/tmp", KEY_DIR}
    assert "noexec" in spec.tmpfs[KEY_DIR] and "mode=0700" in spec.tmpfs[KEY_DIR] and "uid=10001" in spec.tmpfs[KEY_DIR]
    assert "nosuid" in spec.tmpfs["/tmp"] and "nodev" in spec.tmpfs["/tmp"]
    assert "exec" in spec.tmpfs["/tmp"].split(",")  # docker's tmpfs default is noexec; Triton dlopens from /tmp
    assert f"size={int(s.decider_tmp_gib * GIB)}" in spec.tmpfs["/tmp"]
    # one read-only bind: the d3-mini repo dir under the image's HF_HOME; never the docker socket or other host paths
    assert [(m.target, m.mode) for m in spec.mounts] == [("/models/hf/hub/models--vllm-sr--d3-mini", "ro")]
    assert all("docker.sock" not in m.host for m in spec.mounts)


def test_spec_uses_the_internal_network_and_publishes_nothing(d3):
    spec = spec_for(d3)
    assert spec.publish_bind is None
    assert spec.network == "dgxmm-decider" and spec.internal_network
    d3.settings.decider_network = ""  # opt out: the shared engine network (key-only protection)
    spec = spec_for(d3)
    assert spec.network == d3.settings.docker_network and not spec.internal_network and spec.publish_bind is None


# ---------- docker controller ----------
def hardened_spec(**kw):
    base = dict(name="dgxmm-decider", image="img", command=["--port", "8100"], environment={"A": "1"},
                mounts=[Mount(host="/h", target="/t", mode="ro")], labels=managed_labels(1, "decision"),
                network="dgxmm-decider", port=8100, publish_bind=None, ipc_host=False, user="10001:10001",
                read_only=True, tmpfs={"/tmp": "size=1"}, cap_drop=["ALL"], security_opt=["no-new-privileges"],
                pids_limit=512, mem_limit=4 * GIB, internal_network=True,
                secret_files={"/run/dgxmm-decider/api_key": b"s3cret-key-value"})
    base.update(kw)
    return ContainerSpec(**base)


def controller(sdk, own="selfid"):
    d = DockerController(sdk)
    d.own_container_id = lambda: own
    return d


def test_run_passes_hardening_kwargs_and_no_ports():
    sdk = FakeDockerClient()
    sdk.add_network("dgxmm-decider", internal=True)
    cid = controller(sdk).run(hardened_spec())
    call = sdk.run_calls[-1]
    assert "ports" not in call or call["ports"] is None
    assert call["user"] == "10001:10001" and call["read_only"] is True and call["tmpfs"] == {"/tmp": "size=1"}
    assert call["cap_drop"] == ["ALL"] and call["security_opt"] == ["no-new-privileges"]
    assert call["pids_limit"] == 512 and call["mem_limit"] == 4 * GIB and call["memswap_limit"] == 4 * GIB
    assert "ipc_mode" not in call
    assert "s3cret-key-value" not in str(call)
    # the key was written after start through exec stdin, as a 0400 file
    assert sdk.secret_writes == [(cid, "/run/dgxmm-decider/api_key", b"s3cret-key-value")]
    ex = sdk.api.exec_calls[-1]
    assert ex["stdin"] is True and "s3cret-key-value" not in str(ex["cmd"]) and "umask 0377" in " ".join(ex["cmd"])


def test_default_specs_are_unchanged_for_other_engines():
    sdk = FakeDockerClient()
    controller(sdk).run(hardened_spec(labels=managed_labels(2, "vllm"), user=None, read_only=False, tmpfs={},
                                      cap_drop=[], security_opt=[], pids_limit=None, mem_limit=None,
                                      internal_network=False, secret_files={}, ipc_host=True,
                                      publish_bind="127.0.0.1", network="gateway"))
    call = sdk.run_calls[-1]
    for k in ("user", "read_only", "tmpfs", "cap_drop", "security_opt", "pids_limit", "mem_limit"):
        assert k not in call, k
    assert call["ipc_mode"] == "host" and call["ports"] == {"8100/tcp": ("127.0.0.1", 8100)}
    assert sdk.api.exec_calls == []


def test_failed_secret_delivery_removes_the_container():
    sdk = FakeDockerClient()
    sdk.add_network("dgxmm-decider", internal=True)
    sdk.api.exec_exit_code = 1
    with pytest.raises(SecretDeliveryFailed):
        controller(sdk).run(hardened_spec())
    assert not sdk._containers  # never left running without its key


def test_internal_network_is_created_internal_and_dgxmm_joins_it():
    sdk = FakeDockerClient()
    controller(sdk, own="me123").run(hardened_spec())
    net = sdk.networks.get("dgxmm-decider")
    assert net.attrs["Internal"] is True and net.created_kwargs["internal"] is True
    assert net.created_kwargs["labels"]["dgxmm.managed"] == "true"
    assert "me123" in net.connected
    controller(sdk, own="me123").run(hardened_spec(name="dgxmm-decider2"))
    assert net.connected.count("me123") == 1  # already attached: not connected twice


def test_internal_network_carries_the_install_label_of_the_spec():
    sdk = FakeDockerClient()
    controller(sdk).run(hardened_spec(labels={**managed_labels(1, "decision"), "dgxmm.install": "inst42abc"}))
    labels = sdk.networks.get("dgxmm-decider").created_kwargs["labels"]
    assert labels == {"dgxmm.managed": "true", "dgxmm.purpose": "decider", "dgxmm.install": "inst42abc"}


def test_refuses_a_non_internal_network_for_an_internal_spec():
    sdk = FakeDockerClient()
    sdk.add_network("dgxmm-decider", internal=False)
    with pytest.raises(PermissionError):
        controller(sdk).run(hardened_spec())
    assert sdk.run_calls == []


def test_refuses_when_dgxmm_cannot_join_the_network():
    sdk = FakeDockerClient()
    with pytest.raises(RuntimeError):
        controller(sdk, own=None).run(hardened_spec())  # not in a container: it could never reach the decider
    assert sdk.run_calls == []


def test_attach_self_at_startup_only_if_the_network_exists():
    sdk = FakeDockerClient()
    d = controller(sdk, own="me1")
    assert d.attach_self("dgxmm-decider") is False  # nothing to join yet; never created at startup
    sdk.add_network("dgxmm-decider", internal=True)
    assert d.attach_self("dgxmm-decider") is True and sdk.networks.get("dgxmm-decider").connected == ["me1"]


# ---------- deployer integration ----------
async def test_deployed_decider_has_no_key_in_container_env_or_view(d3):
    d3.sdk.add_network("dgxmm-decider", internal=True)
    d3.deployer.docker.own_container_id = lambda: "selfid"
    dep = await deploy_decider(d3)
    key = d3.settings.decider_key_path.read_text()
    c = d3.sdk.container(dep["container_id"])
    assert key not in str(c.attrs) and "DECISION_API_KEY" not in c.env
    assert d3.sdk.secret_writes[-1] == (c.id, KEY_FILE, key.encode())
    assert dep["host_endpoint"] is None and key not in str(dep)
    assert c.ports == []


def test_decider_base_url_is_the_container_name_even_with_engine_base_host(d3):
    d3.settings.engine_base_host = "127.0.0.1"
    assert d3.deployer.base_url("decider", 8123, engine="decision") == "http://dgxmm-decider:8123"
    assert d3.deployer.base_url("tiny", 8100, engine="vllm") == "http://127.0.0.1:8100"


@pytest.mark.parametrize("field,value", [("decider_user", "root"), ("decider_user", "0:0"), ("decider_user", "1000"),
                                         ("decider_memory_limit_gib", 0.5), ("decider_pids_limit", 10),
                                         ("decider_tmp_gib", 0), ("decider_network", "bad name")])
def test_decider_hardening_settings_are_validated(field, value):
    from pydantic import ValidationError

    from dgxmm.config import Settings
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})


# ---------- a decider that exits before its key arrives (fix round 1) ----------
def test_exited_container_reports_exit_code_and_redacted_log_tail():
    sdk = FakeDockerClient()
    sdk.add_network("dgxmm-decider", internal=True)
    sdk.exit_on_start["dgxmm-decider"] = 3
    token = "hf_" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2"
    sdk.next_log_text = ("dgxmm-decider: snapshot files do not match the pinned hashes; refusing to start\n"
                         f"HF_TOKEN={token}\n")
    with pytest.raises(SecretDeliveryFailed) as ei:
        controller(sdk).run(hardened_spec())
    e = ei.value
    assert e.exit_code == 3 and "integrity" in str(e) and "exited with code 3" in str(e)
    assert "do not match the pinned hashes" in e.log_tail and token not in e.log_tail and token not in str(e)
    assert not sdk._containers  # removed after its details were captured


async def test_deploy_records_the_early_exit_as_the_failure(d3):
    d3.sdk.add_network("dgxmm-decider", internal=True)
    d3.sdk.exit_on_start["dgxmm-decider"] = 2
    d3.sdk.next_log_text = "dgxmm-decider: snapshot vllm-sr/d3-mini@97dd054 not found under /models/hf (download it first)\n"
    from dgxmm.deployer.service import DeployError
    with pytest.raises(DeployError) as ei:
        await deploy_decider(d3)
    assert ei.value.code == "container_exited"
    dep = next(v for v in d3.deployer.list_views() if v["engine"] == "decision")
    assert dep["state"] == "failed" and dep["exit_code"] == 2
    assert "snapshot" in dep["last_error"] and "exited with code 2" in dep["last_error"]
    full = d3.deployer.view(dep["id"], with_events=True)
    assert any("not found under" in (ev.get("log_tail") or "") for ev in full["events"])
