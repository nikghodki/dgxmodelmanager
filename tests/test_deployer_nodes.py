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
    health = [h for h in deployer_two_nodes.engines_http.hosts if h[1] == ready_remote_dep.port]
    assert health and {h[0] for h in health} == {"192.0.2.2"}


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
    hosts = {path: host for host, port, path in d.engines_http.hosts if port == v["port"]}
    assert hosts["/sleep"] == hosts["/wake_up"] == "192.0.2.2"


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


# ---------- follow-up: base_url callers use the deployment's node ----------
async def _ready_local(d, qwen_req):
    v = await d.deploy(qwen_req.model_copy(update={"name": "loc", "alias": "loc"}), user_id=1)
    d.engines_http.healthy.add(v["port"])
    assert await d.wait_ready(v["id"], 5) == "ready"
    return d.view(v["id"])


async def test_base_url_for_rows_and_views(deployer_two_nodes, ready_remote_dep, qwen_req):
    d = deployer_two_nodes
    loc = await _ready_local(d, qwen_req)
    assert d.base_url_for(d.view(ready_remote_dep.id)) == f"http://192.0.2.2:{ready_remote_dep.port}"
    assert d.base_url_for(d._get(ready_remote_dep.id)) == f"http://192.0.2.2:{ready_remote_dep.port}"
    assert d.base_url_for(loc) == d.base_url(loc["name"], loc["port"]) == f"http://engines.test:{loc['port']}"


async def test_views_endpoints_remote_and_local_unchanged(deployer_two_nodes, ready_remote_dep, qwen_req):
    d = deployer_two_nodes
    loc = await _ready_local(d, qwen_req)
    r = d.view(ready_remote_dep.id)
    assert r["endpoint"] == r["host_endpoint"] == f"http://192.0.2.2:{ready_remote_dep.port}/v1"
    assert loc["endpoint"] == f"http://dgxmm-loc:{loc['port']}/v1"
    assert loc["host_endpoint"] == f"http://{d.settings.publish_bind}:{loc['port']}/v1"
    assert list(r) == list(loc)  # same keys, same order


async def test_litellm_api_base_uses_node(deployer_two_nodes, ready_remote_dep, qwen_req):
    from dgxmm.litellm import LiteLLMSync
    d = deployer_two_nodes
    loc = await _ready_local(d, qwen_req)
    want = LiteLLMSync(None, d, d.db, d.events).desired()
    bases = {k[0]: val[0] for k, val in want.items()}
    assert bases[str(ready_remote_dep.id)] == f"http://192.0.2.2:{ready_remote_dep.port}/v1"
    assert bases[str(loc["id"])] == f"http://engines.test:{loc['port']}/v1"


async def test_metrics_scrape_url_uses_node(deployer_two_nodes, ready_remote_dep, qwen_req):
    import httpx
    from dgxmm.metrics.scraper import MetricsScraper
    d = deployer_two_nodes
    loc = await _ready_local(d, qwen_req)
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, text="")

    class Store:
        def add(self, dep_id, now, m):
            return {}

        def add_host(self, *a):
            pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await MetricsScraper(d.settings, d, http, Store(), d.events, d.budget).scrape_once()
    assert sorted(urls) == sorted([f"http://192.0.2.2:{ready_remote_dep.port}/metrics",
                                   f"http://engines.test:{loc['port']}/metrics"])


async def test_ondemand_gateway_proxy_target_uses_node(deployer_two_nodes, ready_remote_dep, qwen_req):
    from types import SimpleNamespace
    from dgxmm.ondemand.gateway import Gateway, _Exchange
    d = deployer_two_nodes
    loc = await _ready_local(d, qwen_req)
    gw = Gateway(SimpleNamespace(deployer=d, settings=d.settings))
    targets = []
    gw.lease = lambda dep_id, base: targets.append(base) or object()
    gw.unlease = lambda client: None
    for v in (d.view(ready_remote_dep.id), loc):
        ex = object.__new__(_Exchange)
        ex.gw, ex.v = gw, v

        async def _no_exchange(client, v):
            return None
        ex._exchange = _no_exchange
        await ex._forward()
    assert targets == [f"http://192.0.2.2:{ready_remote_dep.port}", f"http://engines.test:{loc['port']}"]


def test_decider_base_url_uses_base_url_for():
    from dgxmm.decider.service import Decider
    calls = []

    class Dep:
        def base_url_for(self, v):
            calls.append(v["id"])
            return "http://192.0.2.2:8150"
    d = object.__new__(Decider)
    d.deployment = lambda: {"id": 5, "name": "decider", "state": "ready", "port": 8150, "engine": "decision"}
    d._deployer = lambda: Dep()
    assert d._base_url() == "http://192.0.2.2:8150" and calls == [5]


# ---------- follow-up: liveness never stalls the budget loop on a down node ----------
async def test_liveness_skips_stale_remote_node(deployer_two_nodes, ready_remote_dep):
    d = deployer_two_nodes
    fake = d.fake("dgx-2")
    fake.raise_on_all = NodeUnreachable("would block")
    d.mark_stale("dgx-2")
    assert await d.check_liveness() == []
    assert fake.raised == 0  # never asked
    assert d.state(ready_remote_dep.id) == "ready"


async def test_blocked_remote_docker_does_not_stall_budget_loop(deployer_two_nodes, ready_remote_dep):
    import asyncio
    import threading
    import time
    d = deployer_two_nodes
    d.remote_liveness_timeout_s = 0.05
    release = threading.Event()
    blocked = []

    def blocking_get(key):
        blocked.append(key)
        release.wait(5)  # dgx-2's Docker over SSH hangs
        raise TimeoutError("ssh: timed out")
    d.fake("dgx-2").containers.get = blocking_get

    interval = 0.1
    refreshed: list[float] = []

    async def budget_loop():  # what context.budget_loop does: refresh, then listeners (liveness)
        while True:
            await d.budget.refresh()
            refreshed.append(time.monotonic())
            await d.check_liveness()
            await asyncio.sleep(interval)

    task = asyncio.create_task(budget_loop())
    try:
        await asyncio.sleep(0.6)
    finally:
        task.cancel()
        release.set()
    gaps = [b - a for a, b in zip(refreshed, refreshed[1:])]
    assert len(refreshed) >= 4 and max(gaps) < 2 * interval  # every refresh within one interval (+ bounded wait)
    assert len(blocked) == 1  # one hung lookup in flight; later passes skip the node instead of piling up threads
    assert d.state(ready_remote_dep.id) == "ready"


# ---------- fix round 1 ----------
async def _until(pred, timeout=5.0):
    import asyncio
    for _ in range(int(timeout / 0.01)):
        if pred():
            return True
        await asyncio.sleep(0.01)
    return pred()


async def test_unreachable_remote_startup_times_out_and_frees_the_gate(deployer_two_nodes, qwen_req):
    """A remote start-up whose node stays unreachable must not hold the global start-up slot forever."""
    from dgxmm.deployer.service import MonitorState
    d = deployer_two_nodes
    r = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "r", "alias": "r"}), user_id=1)
    d._monitors.pop(r["id"]).cancel()  # driven by hand below
    loc = await d.deploy(qwen_req.model_copy(update={"name": "l", "alias": "l"}), user_id=1)
    assert loc["state"] == "queued" and d.gate.holds(r["id"])
    d.fake("dgx-2").raise_on_all = NodeUnreachable("ssh: connect timed out")
    st = MonitorState(start=d.clock() - d.settings.startup_timeout_s - 1)
    d.monitor_states[r["id"]] = st
    assert await d._monitor_tick(r["id"], st) is True
    row = d._get(r["id"])
    assert row.state == "failed" and "unreachable" in row.last_error
    assert row.container_id  # kept: a stop once the node is back removes the container
    assert not d.gate.holds(r["id"])
    assert await _until(lambda: d.state(loc["id"]) in ("starting", "loading"))
    assert len(d.fake("local").containers.list()) == 1


async def test_unreachable_remote_before_timeout_keeps_slot(deployer_two_nodes, qwen_req):
    from dgxmm.deployer.service import MonitorState
    d = deployer_two_nodes
    r = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    d._monitors.pop(r["id"]).cancel()
    d.fake("dgx-2").raise_on_all = NodeUnreachable("down")
    st = MonitorState(start=d.clock())
    assert await d._monitor_tick(r["id"], st) is False
    assert d.state(r["id"]) in ("starting", "loading") and d.gate.holds(r["id"])


def _block(fake, attr, release, calls):
    """Make fake.containers.<attr> hang until ``release`` (dgx-2's Docker over SSH stuck), then time out."""
    def blocked(*a, **kw):
        calls.append(attr)
        release.wait(5)
        raise TimeoutError("ssh: timed out")
    setattr(fake.containers, attr, blocked)


@pytest.mark.parametrize("stuck", ["list", "run"])
async def test_hung_remote_docker_does_not_delay_local_deploy(deployer_two_nodes, qwen_req, stuck):
    """Remote SSH I/O never happens under mem_lock: a hung dgx-2 (port listing before the lock, or the container
    launch after it) leaves local deploys free to proceed."""
    import asyncio
    import threading
    import time
    d = deployer_two_nodes
    release, calls = threading.Event(), []
    _block(d.fake("dgx-2"), stuck, release, calls)
    remote = asyncio.create_task(d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "r", "alias": "r"}),
                                          user_id=1))
    try:
        assert await _until(lambda: calls)  # dgx-2 is now hanging
        t0 = time.monotonic()
        v = await asyncio.wait_for(d.deploy(qwen_req.model_copy(update={"name": "l", "alias": "l"}), user_id=1), 2)
        took = time.monotonic() - t0
        assert took < 0.5
        # list stuck: the remote deploy has not claimed anything yet, local starts; run stuck: remote holds the
        # start-up slot, so the local deploy is queued (not blocked)
        assert v["state"] in (("starting", "loading") if stuck == "list" else ("queued",))
    finally:
        release.set()
        with pytest.raises(DeployError) as e:
            await asyncio.wait_for(remote, 10)
        assert e.value.code == "node_unreachable"


async def test_remote_launch_reservation_counts_while_off_the_lock(deployer_two_nodes, qwen_req):
    """Between mem_lock and the container start, a remote deployment's memory stays committed on its node."""
    import asyncio
    import threading
    d = deployer_two_nodes
    release, calls = threading.Event(), []
    _block(d.fake("dgx-2"), "run", release, calls)
    remote = asyncio.create_task(d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1))
    try:
        assert await _until(lambda: calls)
        await d.budget.refresh()
        b = d.budget.current("dgx-2")
        [u] = [u for u in b.deployments]
        assert u.state == "queued" and u.reserved > 0 and b.committed >= u.reserved
    finally:
        release.set()
        with pytest.raises(DeployError):
            await asyncio.wait_for(remote, 10)
    await d.budget.refresh()
    assert d.budget.current("dgx-2").deployments == ()  # the failed deploy left nothing behind


async def test_remote_port_conflict_retry_runs_off_the_event_loop(deployer_two_nodes, qwen_req):
    import threading
    d = deployer_two_nodes
    lo = d.settings.port_range[0]
    d.fake("dgx-2").busy_host_ports = {lo}
    threads = []
    orig = d._next_port

    def spy(dep_id, tried):
        threads.append(threading.current_thread() is threading.main_thread())
        return orig(dep_id, tried)
    d._next_port = spy
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert v["port"] == lo + 1 and threads == [False]


def test_used_ports_does_not_swallow_a_down_remote_node(deployer_two_nodes):
    d = deployer_two_nodes
    d.fake("dgx-2").raise_on_all = requests.exceptions.ConnectionError("refused")
    with pytest.raises(NodeUnreachable):
        d._used_ports(None, "dgx-2")
    d.fake("local").raise_on_all = docker.errors.APIError("boom")  # local: logged, as before
    assert d._used_ports(None, "local") == set()


async def test_one_address_for_bind_base_url_and_views(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.nodes.config("dgx-2").publish_bind = "192.0.2.12"  # a second address on the cluster link
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    call = d.fake("dgx-2").run_calls[-1]
    assert call["ports"] == {f"{v['port']}/tcp": ("192.0.2.12", v["port"])}
    assert d.base_url_for(v) == f"http://192.0.2.12:{v['port']}"
    assert v["endpoint"] == v["host_endpoint"] == f"http://192.0.2.12:{v['port']}/v1"


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "127.0.0.1", "224.0.0.1"])
async def test_remote_spec_refuses_unsafe_bind(deployer_two_nodes, qwen_req, bind):
    """Defensive: even if the node config were changed after validation, no engine publishes on such an address."""
    d = deployer_two_nodes
    d.nodes.config("dgx-2").publish_bind = bind
    with pytest.raises(DeployError) as e:
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert e.value.code == "unsafe_publish_bind"
    assert d.fake("dgx-2").run_calls == []


# ---------- fix round 2: ports of remote launches in flight after mem_lock ----------
def _block_first_run(fake, release, calls):
    """dgx-2's first containers.run hangs until ``release``, then proceeds normally (later calls are not blocked)."""
    orig = fake.containers.run

    def run(*a, **kw):
        if not calls:
            calls.append(kw.get("name"))
            release.wait(5)
        return orig(*a, **kw)
    fake.containers.run = run


async def test_remote_restart_in_flight_keeps_its_port(deployer_two_nodes, ready_remote_dep, qwen_req):
    """A's restart writes its port into a still-stopped row and launches after mem_lock: a concurrent deploy on the
    same node must not get that port."""
    import asyncio
    import threading
    d = deployer_two_nodes
    d.settings.max_concurrent_startups = 0  # no start-up limit: B is not queued behind A
    a_id = ready_remote_dep.id
    await d.stop(a_id)
    release, calls = threading.Event(), []
    _block_first_run(d.fake("dgx-2"), release, calls)
    restart = asyncio.create_task(d.restart(a_id))
    try:
        assert await _until(lambda: calls)  # A's launch is in flight, its row still "stopped"
        assert d.state(a_id) == "stopped"
        b = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "b", "alias": "b"}), user_id=1)
    finally:
        release.set()
    a = await asyncio.wait_for(restart, 10)
    assert a["port"] != b["port"]
    assert a["state"] in ("starting", "loading") and b["state"] in ("starting", "loading")


async def test_remote_deploy_in_flight_keeps_its_port(deployer_two_nodes, qwen_req):
    """Same for a fresh remote deploy launching after mem_lock (row "planned") and a concurrent one."""
    import asyncio
    import threading
    d = deployer_two_nodes
    d.settings.max_concurrent_startups = 0
    release, calls = threading.Event(), []
    _block_first_run(d.fake("dgx-2"), release, calls)
    first = asyncio.create_task(d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "a", "alias": "a"}),
                                         user_id=1))
    try:
        assert await _until(lambda: calls)
        b = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "b", "alias": "b"}), user_id=1)
    finally:
        release.set()
    a = await asyncio.wait_for(first, 10)
    assert a["port"] != b["port"]


async def test_remote_queued_launch_in_flight_keeps_its_port(deployer_two_nodes, qwen_req):
    """And for a queued remote deployment launching after mem_lock (row "queued") while another deploy runs."""
    import asyncio
    import threading
    d = deployer_two_nodes
    blocker = await d.deploy(qwen_req.model_copy(update={"name": "l", "alias": "l"}), user_id=1)  # holds the slot
    q = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "a", "alias": "a"}), user_id=1)
    assert q["state"] == "queued"
    release, calls = threading.Event(), []
    _block_first_run(d.fake("dgx-2"), release, calls)
    d.engines_http.healthy.add(blocker["port"])  # l becomes ready: the queued remote one takes the slot
    try:
        assert await _until(lambda: calls)
        d.settings.max_concurrent_startups = 0
        b = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "b", "alias": "b"}), user_id=1)
    finally:
        release.set()
    assert await _until(lambda: d.state(q["id"]) in ("starting", "loading"))
    assert d._get(q["id"]).port != b["port"]


def test_db_ports_counts_rows_being_launched(deployer_two_nodes):
    from tests.nodes_fixtures import insert_deployment
    d = deployer_two_nodes
    insert_deployment(d, node="dgx-2", port=8100, name="a")
    with d.db.session() as s:
        from dgxmm.db import Deployment
        row = s.query(Deployment).filter_by(name="a").one()
        row.state = "stopped"
        dep_id = row.id
    assert 8100 not in d._db_ports(None, "dgx-2")
    d.launching[dep_id] = ("Qwen/Qwen3-0.6B",)
    assert 8100 in d._db_ports(None, "dgx-2")
    assert 8100 not in d._db_ports(dep_id, "dgx-2")  # its own port stays available to it
    assert 8100 not in d._db_ports(None, "local")
