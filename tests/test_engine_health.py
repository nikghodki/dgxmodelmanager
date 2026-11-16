"""Periodic engine health probe of ready deployments (dgxmm/deployer/health.py): engine.health() of each ready
deployment (a cluster: its head) every engine_health_interval_s; after engine_health_failures failed probes in a row
it is failed through the normal path (containers / ranks removed, last_error set; LiteLLM deregisters it through the
state hooks). Never probed: deployments that are not ready (starting, loading, waking, sleeping, stopping, stopped),
system deployments (the decider has its own supervisor), one with an operation in progress (its lock is held) or on
a node whose memory reading is stale."""
import asyncio

import pytest

from dgxmm.config import Settings
from dgxmm.deployer.health import EngineHealth
from dgxmm.deployer.service import DeployRequest

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def ready(stack, name="a", **cfg):
    d = await stack.deployer.deploy(DeployRequest(name=name, alias=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={**SMALL, **cfg}), 1)
    stack.http.healthy.add(d["port"])
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return stack.deployer.view(d["id"])


def probes(http, port):
    return [p for _, prt, p in http.calls if prt == port and p == "/health"]


@pytest.fixture
def health(stack):
    return EngineHealth(stack.settings, stack.deployer)


def test_settings_defaults():
    s = Settings()
    assert s.engine_health_interval_s == 30.0 and s.engine_health_failures == 3
    with pytest.raises(ValueError):
        Settings(engine_health_failures=0)
    with pytest.raises(ValueError):
        Settings(engine_health_interval_s=-1)


async def test_healthy_deployment_stays_ready(stack, health):
    v = await ready(stack)
    before = len(probes(stack.http, v["port"]))
    assert await health.check_once() == []
    assert len(probes(stack.http, v["port"])) == before + 1
    assert stack.deployer.view(v["id"])["state"] == "ready"


async def test_fails_after_n_consecutive_failures_and_removes_the_container(stack, health):
    seen = []

    async def hook(dep_id, frm, to):
        seen.append((frm, to))
    stack.deployer.state_hooks.append(hook)
    v = await ready(stack)
    stack.http.healthy.discard(v["port"])  # e.g. the EngineCore died: /health stops answering 200
    for _ in range(stack.settings.engine_health_failures - 1):
        assert await health.check_once() == []
        assert stack.deployer.view(v["id"])["state"] == "ready"
    assert await health.check_once() == [v["id"]]
    after = stack.deployer.view(v["id"])
    assert after["state"] == "failed" and after["container_id"] is None
    assert "health" in after["last_error"] and "3" in after["last_error"]
    assert stack.sdk.containers.list() == []
    await asyncio.sleep(0)
    assert ("stopping", "failed") in seen  # the state hooks (LiteLLM sync, on-demand) see the failure


async def test_one_slow_or_failed_probe_is_tolerated(stack, health):
    v = await ready(stack)
    for _ in range(5):  # fail, fail, ok, fail, fail, ok ...: never N in a row
        stack.http.healthy.discard(v["port"])
        await health.check_once()
        await health.check_once()
        stack.http.healthy.add(v["port"])
        await health.check_once()
    assert stack.deployer.view(v["id"])["state"] == "ready"


async def test_a_probe_that_raises_counts_as_failed(stack, health, monkeypatch):
    import httpx
    v = await ready(stack)

    async def hang(*a, **kw):
        raise httpx.ReadTimeout("no answer")
    monkeypatch.setattr(stack.deployer.engines["vllm"], "health", hang)
    for _ in range(3):
        await health.check_once()
    assert stack.deployer.view(v["id"])["state"] == "failed"


async def test_not_ready_deployments_are_never_probed(stack, health):
    stack.settings.vllm_sleep = "level1"
    a = await ready(stack, "a", enable_sleep_mode=True)
    b = await ready(stack, "b")
    await stack.deployer.sleep(a["id"])
    await stack.deployer.stop(b["id"])
    stack.http.healthy.clear()
    na, nb = len(probes(stack.http, a["port"])), len(probes(stack.http, b["port"]))
    for _ in range(5):
        assert await health.check_once() == []
    assert len(probes(stack.http, a["port"])) == na and len(probes(stack.http, b["port"])) == nb
    assert stack.deployer.view(a["id"])["state"] == "sleeping"


async def test_starting_deployments_are_left_to_the_startup_monitor(stack, health):
    d = await stack.deployer.deploy(DeployRequest(name="s", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=SMALL), 1)
    for _ in range(5):
        await health.check_once()
    assert stack.deployer.view(d["id"])["state"] in ("starting", "loading")


async def test_a_deployment_with_an_operation_in_progress_is_skipped(stack, health):
    v = await ready(stack)
    stack.http.healthy.discard(v["port"])
    async with stack.deployer._lock(v["id"]):  # e.g. a stop or sleep under way
        for _ in range(5):
            await health.check_once()
    assert stack.deployer.view(v["id"])["state"] == "ready"


async def test_a_state_change_resets_the_count(stack, health):
    stack.settings.vllm_sleep = "level1"
    v = await ready(stack, enable_sleep_mode=True)
    stack.http.healthy.discard(v["port"])
    await health.check_once()
    await health.check_once()
    await stack.deployer.sleep(v["id"])
    await stack.deployer.wake(v["id"])  # woken: a fresh start for the count
    await health.check_once()
    await health.check_once()
    assert stack.deployer.view(v["id"])["state"] == "ready"
    await health.check_once()
    assert stack.deployer.view(v["id"])["state"] == "failed"


async def test_a_transition_during_the_probe_discards_its_result(stack, health, monkeypatch):
    v = await ready(stack)
    stack.http.healthy.discard(v["port"])
    await health.check_once()
    await health.check_once()
    engine = stack.deployer.engines["vllm"]
    orig = engine.health

    async def slow(*a, **kw):  # the user stops it while the probe waits for an answer
        await stack.deployer.stop(v["id"])
        return await orig(*a, **kw)
    monkeypatch.setattr(engine, "health", slow)
    assert await health.check_once() == []
    assert stack.deployer.view(v["id"])["state"] == "stopped"


async def test_system_deployments_are_not_probed(stack, health):
    from dgxmm.db import Deployment
    v = await ready(stack)
    with stack.db.session() as s:
        s.get(Deployment, v["id"]).system = True
    stack.http.healthy.clear()
    for _ in range(5):
        await health.check_once()
    assert stack.deployer.view(v["id"])["state"] == "ready"


async def test_probes_run_concurrently(stack, health, monkeypatch):
    a, b = await ready(stack, "a"), await ready(stack, "b")
    engine = stack.deployer.engines["vllm"]
    inflight, peak = [0], [0]

    async def slow(*args, **kw):
        inflight[0] += 1
        peak[0] = max(peak[0], inflight[0])
        await asyncio.sleep(0.05)
        inflight[0] -= 1
        return True
    monkeypatch.setattr(engine, "health", slow)
    await health.check_once()
    assert peak[0] == 2  # one slow engine never delays the others' probes


async def test_disabled_with_interval_zero(stack):
    stack.settings.engine_health_interval_s = 0
    await asyncio.wait_for(EngineHealth(stack.settings, stack.deployer).run(), 1)  # returns at once


# ---------- clusters and remote nodes ----------
async def test_cluster_probes_its_head_and_fails_every_rank(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    h = EngineHealth(d.settings, d)
    port = ready_cluster_dep.port
    before = len(probes(d.engines_http, port))
    await h.check_once()
    assert len(probes(d.engines_http, port)) == before + 1
    d.engines_http.healthy.discard(port)
    for _ in range(3):
        await h.check_once()
    assert d.state(ready_cluster_dep.id) == "failed"
    assert all(d.fake(n).containers.list() == [] for n in ("local", "dgx-2"))
    assert "health" in d.view(ready_cluster_dep.id)["last_error"]


async def test_stale_remote_node_is_not_probed(deployer_two_nodes, ready_remote_dep):
    d = deployer_two_nodes
    h = EngineHealth(d.settings, d)
    d.engines_http.healthy.discard(ready_remote_dep.port)
    d.mark_stale("dgx-2")
    for _ in range(5):
        await h.check_once()
    assert d.state(ready_remote_dep.id) == "ready"


async def test_build_context_runs_the_health_loop(settings):
    from dgxmm.context import build_context
    ctx = build_context(settings)
    try:
        assert isinstance(ctx.health, EngineHealth)
        assert ctx.health.run in ctx.background
    finally:
        await ctx.http.aclose()


async def test_the_probe_sends_no_completion_so_idle_stays_idle(stack, health):
    """A completion every interval would show as prompt/generation tokens in the metrics (decider/common.py
    BUSY_KEYS): schedules, plans and proposals would never see the deployment idle. The periodic probe is the
    health() check without its test completion (the start-up check keeps it)."""
    v = await ready(stack)
    stack.http.calls.clear()
    for _ in range(3):
        await health.check_once()
    assert [p for _, prt, p in stack.http.calls if prt == v["port"]] == ["/health"] * 3


@pytest.mark.parametrize("engine", ["vllm", "llamacpp"])
async def test_health_without_completion_checks_only_the_health_route(stack, engine):
    import httpx
    from tests.fakes import FakeEngines
    fe = FakeEngines()
    fe.healthy.add(1)
    async with fe.client() as http:
        assert await stack.deployer.engines[engine].health(http, "http://h:1", "q", completion=False) is True
        fe.healthy.clear()
        assert await stack.deployer.engines[engine].health(http, "http://h:1", "q", completion=False) is False
    assert {p for _, _, p in fe.calls} == {"/health"}
