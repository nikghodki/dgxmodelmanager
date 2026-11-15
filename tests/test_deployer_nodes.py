"""Task 5: the deployer places, monitors and stops deployments on the chosen node."""
import docker.errors
import pytest
import requests

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory
from dgxmm.nodes.registry import NodeUnreachable, is_transport_error


# ---------- from the brief ----------
async def test_deploy_on_remote_runs_container_on_that_node_only(deployer_two_nodes, qwen_req):
    await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert len(deployer_two_nodes.fake("dgx-2").containers.list()) == 1
    assert deployer_two_nodes.fake("local").containers.list() == []


async def test_remote_spec_publishes_on_cluster_ip_and_mounts_hf_read_only(deployer_two_nodes, qwen_req):
    await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    c = deployer_two_nodes.fake("dgx-2").containers.list()[0]
    assert c.ports_bind_ip == "192.0.2.2"
    assert ("/home/dgxuser/.cache/huggingface", "ro") in c.mount_sources_modes


async def test_fit_uses_target_node_budget(deployer_two_nodes, qwen_req):
    deployer_two_nodes.set_free("local", 0); deployer_two_nodes.set_free("dgx-2", 100 * 2**30)
    # distinct aliases: model names stay globally unique across nodes (spec R3), so the brief's two deploys of the
    # same model need their own names to reach the memory check
    await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "q2", "alias": "q2"}),
                                    user_id=1)
    with pytest.raises(InsufficientMemory):
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "local", "name": "q1", "alias": "q1"}),
                                        user_id=1)


async def test_unknown_node_400(deployer_two_nodes, qwen_req):
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-9"}), user_id=1)
    assert (e.value.code, e.value.status) == ("unknown_node", 400)


async def test_stale_node_refuses_start(deployer_two_nodes, qwen_req):
    deployer_two_nodes.mark_stale("dgx-2")
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert (e.value.code, e.value.status) == ("node_stale", 503)


async def test_remote_transport_errors_do_not_fail_running_deployment(deployer_two_nodes, ready_remote_dep):
    deployer_two_nodes.fake("dgx-2").raise_on_all = NodeUnreachable("ssh: connect timed out")
    for _ in range(6):
        await deployer_two_nodes.monitor_tick(ready_remote_dep.id)
    assert deployer_two_nodes.state(ready_remote_dep.id) == "ready"


# ---------- placement details ----------
def test_deploy_request_node_defaults_to_local_and_is_validated():
    r = DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny")
    assert r.node == "local"
    assert DeployRequest(name="a", engine="vllm", model_ref="x/y", node="dgx-2").node == "dgx-2"
    with pytest.raises(ValueError):
        DeployRequest(name="a", engine="vllm", model_ref="x/y", node="DGX 2")


async def test_remote_spec_uses_default_bridge_and_no_dgx1_cache_mounts(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    call = d.fake("dgx-2").run_calls[-1]
    assert call["network"] == "bridge"  # never DGXMM's gateway network on a remote node (R8)
    assert call["ports"] == {f"{v['port']}/tcp": ("192.0.2.2", v["port"])}
    assert call["volumes"] == {"/home/dgxuser/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "ro"}}
    assert call["labels"]["dgxmm.managed"] == "true"
    assert d.fake("dgx-2").networks.items == {}  # no network created there
    assert d._get(v["id"]).node == "dgx-2"


async def test_local_deploy_on_two_node_install_is_unchanged(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    v = await d.deploy(qwen_req, user_id=1)
    call = d.fake("local").run_calls[-1]
    s = d.settings
    assert call["network"] == s.docker_network
    assert call["ports"] == {f"{v['port']}/tcp": (s.publish_bind, v["port"])}
    expected = {s.hf_cache.host: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                **{m.host: {"bind": m.target, "mode": m.mode} for m in s.vllm_cache_mounts}}
    assert call["volumes"] == expected
    assert d.fake("dgx-2").run_calls == []
    assert d._get(v["id"]).node == "local"


def test_base_url_local_unchanged_remote_uses_cluster_ip(deployer_two_nodes):
    d = deployer_two_nodes
    assert d.base_url("tiny", 8100) == "http://engines.test:8100"
    assert d.base_url("tiny", 8100, "vllm", node="local") == "http://engines.test:8100"
    assert d.base_url("tiny", 8100, "vllm", node="dgx-2") == "http://192.0.2.2:8100"


async def test_remote_health_check_goes_to_cluster_ip(deployer_two_nodes, ready_remote_dep):
    assert ready_remote_dep.view["state"] == "ready"


async def test_ports_are_allocated_per_node(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    a = await d.deploy(qwen_req.model_copy(update={"name": "a", "alias": "a"}), user_id=1)
    b = await d.deploy(qwen_req.model_copy(update={"name": "b", "alias": "b", "node": "dgx-2"}), user_id=1)
    assert a["port"] == b["port"] == d.settings.port_range[0]


async def test_stop_and_remove_act_on_the_deployment_node(deployer_two_nodes, ready_remote_dep):
    d = deployer_two_nodes
    local_before = list(d.fake("local").run_calls)
    await d.stop(ready_remote_dep.id)
    assert d.state(ready_remote_dep.id) == "stopped"
    assert d.fake("dgx-2").containers.list() == []
    await d.remove(ready_remote_dep.id)
    assert d._get(ready_remote_dep.id) is None
    assert d.fake("local").run_calls == local_before


async def test_restart_on_remote_launches_there_again(deployer_two_nodes, ready_remote_dep):
    d = deployer_two_nodes
    await d.stop(ready_remote_dep.id)
    v = await d.restart(ready_remote_dep.id)
    assert v["state"] in ("starting", "loading")
    assert len(d.fake("dgx-2").containers.list()) == 1 and d.fake("local").containers.list() == []


async def test_unreachable_at_deploy_is_503_and_creates_nothing(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.fake("dgx-2").raise_on_all = requests.exceptions.ConnectionError("ssh: connect to host 192.0.2.2: timed out")
    with pytest.raises(DeployError) as e:
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert (e.value.code, e.value.status) == ("node_unreachable", 503)
    assert d.list_views() == []
    assert d.gate.holders == {}


async def test_stop_while_unreachable_is_503(deployer_two_nodes, ready_remote_dep):
    d = deployer_two_nodes
    d.fake("dgx-2").raise_on_all = ConnectionResetError("ssh: connection reset")
    with pytest.raises(DeployError) as e:
        await d.stop(ready_remote_dep.id)
    assert (e.value.code, e.value.status) == ("node_unreachable", 503)
    # the container is still known: a later stop (node back) removes it
    assert d._get(ready_remote_dep.id).container_id
    d.fake("dgx-2").raise_on_all = None
    await d.stop(ready_remote_dep.id)
    assert d.state(ready_remote_dep.id) == "stopped" and d.fake("dgx-2").containers.list() == []


async def test_loading_remote_deployment_survives_unreachable_ticks(deployer_two_nodes, qwen_req):
    """While starting/loading, transport errors from a remote node never count toward 5-errors-then-failed."""
    d = deployer_two_nodes
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    await d.aclose()  # drive the monitor by hand
    d.fake("dgx-2").raise_on_all = OSError("ssh: Broken pipe")
    for _ in range(7):
        assert await d.monitor_tick(v["id"]) is False
    st = d.monitor_states[v["id"]]
    assert st.errors == 0 and st.unreachable == 7
    assert d.state(v["id"]) in ("starting", "loading")
    d.fake("dgx-2").raise_on_all = None  # back: the next tick sees it running and resets the counter
    await d.monitor_tick(v["id"])
    assert st.unreachable == 0 and d.state(v["id"]) == "loading"


async def test_local_monitor_errors_still_fail_after_five(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    v = await d.deploy(qwen_req, user_id=1)
    await d.aclose()
    d.fake("local").raise_on_all = docker.errors.APIError("persistent error")
    done = [await d.monitor_tick(v["id"]) for _ in range(5)]
    assert done == [False] * 4 + [True]
    assert d.state(v["id"]) == "failed"


def test_monitor_backoff_is_capped(deployer_two_nodes, monkeypatch):
    from dgxmm.deployer import service
    from dgxmm.deployer.service import MonitorState
    d = deployer_two_nodes
    st = MonitorState(start=0.0)
    assert d._monitor_wait(st) == d.settings.monitor_poll_s
    st.unreachable = 50
    assert d._monitor_wait(st) == service.UNREACHABLE_BACKOFF_MAX_S == 60.0


async def test_remote_monitor_loop_backs_off_and_keeps_state(deployer_two_nodes, qwen_req, monkeypatch):
    from dgxmm.deployer import service
    monkeypatch.setattr(service, "UNREACHABLE_BACKOFF_MAX_S", 0.02)
    d = deployer_two_nodes
    d.fake("dgx-2").raise_on_all = ConnectionRefusedError("ssh: connection refused")
    with pytest.raises(DeployError):  # unreachable at deploy
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    d.fake("dgx-2").raise_on_all = None
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    fake = d.fake("dgx-2")
    fake.raise_on_all = TimeoutError("ssh: timed out")
    import asyncio
    for _ in range(200):
        if fake.raised >= 8:
            break
        await asyncio.sleep(0.01)
    assert fake.raised >= 8  # the monitor keeps retrying
    assert d.state(v["id"]) in ("starting", "loading")


async def test_sleep_and_wake_on_remote_use_remote_url_and_node_budget(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    req = qwen_req.model_copy(update={"node": "dgx-2", "config": {**qwen_req.config, "enable_sleep_mode": True}})
    v = await d.deploy(req, user_id=1)
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    await d.sleep(v["id"])
    assert v["port"] in d.engines_http.sleeping
    d.set_free("local", 0)  # the local box is full: waking a dgx-2 deployment must not care
    out = await d.wake(v["id"])
    assert out["state"] == "ready" and v["port"] not in d.engines_http.sleeping


async def test_wake_refused_when_remote_node_is_full(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    req = qwen_req.model_copy(update={"node": "dgx-2", "config": {**qwen_req.config, "enable_sleep_mode": True}})
    v = await d.deploy(req, user_id=1)
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    await d.sleep(v["id"])
    d.set_free("dgx-2", 0)
    with pytest.raises(InsufficientMemory):
        await d.wake(v["id"])


async def test_liveness_fails_vanished_remote_container(deployer_two_nodes, ready_remote_dep):
    d = deployer_two_nodes
    fake = d.fake("dgx-2")
    fake._containers.clear()
    assert await d.check_liveness() == [ready_remote_dep.id]
    assert d.state(ready_remote_dep.id) == "failed"


# ---------- transport-error mapping (one boundary) ----------
def _chained(cause: Exception) -> Exception:
    try:
        try:
            raise cause
        except Exception as e:
            raise docker.errors.DockerException(f"Error while fetching server API version: {e}") from e
    except docker.errors.DockerException as wrapped:
        return wrapped


@pytest.mark.parametrize("exc, expected", [
    (requests.exceptions.ConnectionError("refused"), True),
    (requests.exceptions.ReadTimeout("timed out"), True),
    (ConnectionResetError("reset"), True),
    (TimeoutError("timed out"), True),
    (OSError("ssh: Broken pipe"), True),
    (_chained(requests.exceptions.ConnectionError("refused")), True),
    (docker.errors.DockerException("bad config"), False),
    (docker.errors.APIError("500 Server Error: port is already allocated"), False),
    (docker.errors.NotFound("no such container"), False),
    (PermissionError("container x is not managed by dgxmm"), False),
    (ValueError("nope"), False),
])
def test_is_transport_error(exc, expected):
    assert is_transport_error(exc) is expected


def test_remote_controller_maps_transport_errors(two_node_registry):
    from dgxmm.nodes.registry import RemoteDocker
    two_node_registry.fake("dgx-2").raise_on_all = requests.exceptions.ConnectionError("refused")
    rd = RemoteDocker(two_node_registry.docker("dgx-2"), "dgx-2")
    with pytest.raises(NodeUnreachable, match="dgx-2"):
        rd.get("abc")
    two_node_registry.fake("dgx-2").raise_on_all = docker.errors.APIError("boom")
    with pytest.raises(docker.errors.APIError):
        rd.get("abc")
    two_node_registry.fake("dgx-2").raise_on_all = None
    assert rd.get("abc") is None
    assert rd.client is two_node_registry.fake("dgx-2")


async def test_single_node_deployer_refuses_remote(stack):
    req = DeployRequest(name="tiny", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                        config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048},
                        node="dgx-2")
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req, user_id=1)
    assert (e.value.code, e.value.status) == ("unknown_node", 400)
    assert stack.sdk.run_calls == []


# ---------- llama.cpp on a remote node ----------
def _gguf(path):
    from tests.conftest import write_gguf
    from dgxmm.inventory.gguf_scan import scan_gguf_roots
    write_gguf(path, pad_bytes=GIB)
    [a] = scan_gguf_roots([path.parent])
    return a


def test_llamacpp_remote_serves_hf_cache_gguf_from_node_cache(two_node_settings):
    from pathlib import Path
    from dgxmm.engines.base import LaunchContext
    from dgxmm.engines.llamacpp import LlamaCppEngine
    s = two_node_settings
    art = _gguf(Path(s.hf_cache.local) / "hub" / "models--org--g" / "snapshots" / "abc" / "m-Q4_K_M.gguf")
    e = LlamaCppEngine(s)
    spec = e.container_spec(art, e.validate_config({}), None,
                            LaunchContext(3, "q", 8110, 10 * GIB, 120 * GIB, "dgx-2"), s)
    assert [(m.host, m.target, m.mode) for m in spec.mounts] == [
        ("/home/dgxuser/.cache/huggingface", "/root/.cache/huggingface", "ro")]
    c = spec.command
    assert c[c.index("-m") + 1] == "/root/.cache/huggingface/hub/models--org--g/snapshots/abc/m-Q4_K_M.gguf"
    assert spec.network == "bridge" and spec.publish_bind == "192.0.2.2"
    local = e.container_spec(art, e.validate_config({}), None, LaunchContext(3, "q", 8110, 10 * GIB, 120 * GIB), s)
    assert (local.network, local.publish_bind) == (s.docker_network, s.publish_bind)
    assert local.mounts[0].host == s.hf_cache.host


def test_llamacpp_remote_refuses_gguf_outside_hf_cache(two_node_settings):
    from pathlib import Path
    from dgxmm.engines.base import LaunchContext, SpecError
    from dgxmm.engines.llamacpp import LlamaCppEngine
    s = two_node_settings
    art = _gguf(Path(s.gguf_roots[0].local) / "qwen" / "m-Q4_K_M.gguf")
    e = LlamaCppEngine(s)
    with pytest.raises(SpecError) as err:
        e.container_spec(art, e.validate_config({}), None, LaunchContext(3, "q", 8110, 10 * GIB, 120 * GIB, "dgx-2"), s)
    assert err.value.code == "not_on_node"
