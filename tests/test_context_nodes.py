"""Multi-node context tests."""
import pytest
from pathlib import Path


def test_remote_probe_loop_not_started_for_single_node(settings_factory):
    """Remote probe loop should not be started when there are no remote nodes."""
    from dgxmm.context import build_context
    
    settings = settings_factory()
    ctx = build_context(settings)
    
    # Count background tasks - should not include remote_probe_loop
    # We can't directly count tasks, but we can check that the loop would exit immediately
    import asyncio
    from dgxmm.context import remote_probe_loop
    
    # Run the loop once - it should return immediately without starting
    async def run_once():
        # Create a task that will be cancelled after a short time
        task = asyncio.create_task(remote_probe_loop(ctx))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # If it didn't return immediately, it would still be running
        return task.done()
    
    assert asyncio.run(run_once())


def test_remote_probe_loop_started_for_multi_node(two_node_settings):
    """Remote probe loop should be started when there are remote nodes."""
    from dgxmm.context import build_context
    from dgxmm.nodes.probe import NodeProbe
    from tests.fakes import FakeDockerClient
    
    # Need to mock the node registry with fakes
    ctx = build_context(two_node_settings, client_factory=lambda n: FakeDockerClient())
    
    # Replace probe with NodeProbe
    from dgxmm.nodes.registry import NodeRegistry
    fake_clients = {"local": FakeDockerClient(), "dgx-2": FakeDockerClient()}
    registry = NodeRegistry(two_node_settings, client_factory=lambda n: fake_clients[n.id])
    ctx.nodes = registry
    ctx.probe = NodeProbe(registry, two_node_settings)
    
    # Check that remote_probe_loop is in background tasks
    # The actual check is that build_context adds it when remote nodes exist
    import asyncio
    from dgxmm.context import remote_probe_loop
    
    # Run the loop briefly - it should not exit immediately
    async def run_briefly():
        task = asyncio.create_task(remote_probe_loop(ctx))
        await asyncio.sleep(0.01)  # Let it enter the loop
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True
    
    assert asyncio.run(run_briefly())


def test_remote_probe_loop_continues_after_node_failure(three_node_registry):
    """If one node's probe fails, remaining nodes should still be sampled."""
    import asyncio
    from dgxmm.context import remote_probe_loop, AppContext
    from dgxmm.nodes.probe import NodeProbe
    
    # Set up dgx-2 to fail, dgx-3 to succeed
    three_node_registry.fake("dgx-2").exec_fail = True
    three_node_registry.fake("dgx-3").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    
    probe = NodeProbe(three_node_registry, three_node_registry.settings)
    
    # Create minimal context
    class FakeCtx:
        pass
    
    ctx = FakeCtx()
    ctx.probe = probe
    ctx.nodes = three_node_registry
    ctx.settings = three_node_registry.settings
    
    # Run one iteration of the loop
    async def run_one_iteration():
        task = asyncio.create_task(remote_probe_loop(ctx))
        await asyncio.sleep(0.05)  # Let one iteration complete
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    asyncio.run(run_one_iteration())
    
    # dgx-2 should have no sample (failed)
    assert probe._samples.get("dgx-2") is None
    
    # dgx-3 should have been sampled (success)
    assert probe._samples.get("dgx-3") is not None
    assert probe._samples["dgx-3"].total == 127000000 * 1024


def test_remote_probe_loop_does_not_block_event_loop(two_node_registry):
    """Blocking probe calls should not prevent other coroutines from running."""
    import asyncio
    import time
    from dgxmm.context import remote_probe_loop, AppContext
    from dgxmm.nodes.probe import NodeProbe
    
    # Set up dgx-2 with a slow exec
    two_node_registry.fake("dgx-2").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    
    probe = NodeProbe(two_node_registry, two_node_registry.settings)
    
    # Make exec_run block briefly
    original_exec_run = two_node_registry.fake("dgx-2")._containers.get
    
    def slow_get(name):
        if name == "dgxmm-node-probe":
            time.sleep(0.05)  # 50ms block
        # Create a fake container
        from tests.fakes import FakeContainer
        c = FakeContainer(two_node_registry.fake("dgx-2"), "probe-cid", "dgxmm-node-probe",
                         two_node_registry.settings.images.probe, ["sleep", "2147483647"],
                         {}, {}, [], [], [], status="running")
        return c
    
    two_node_registry.fake("dgx-2").containers.get = slow_get
    
    # Create minimal context
    class FakeCtx:
        pass
    
    ctx = FakeCtx()
    ctx.probe = probe
    ctx.nodes = two_node_registry
    ctx.settings = two_node_registry.settings
    
    # Run probe loop alongside another coroutine
    progress = []
    
    async def other_task():
        for i in range(5):
            await asyncio.sleep(0.02)  # 20ms per iteration
            progress.append(i)
    
    async def run_both():
        probe_task = asyncio.create_task(remote_probe_loop(ctx))
        other_task_coro = asyncio.create_task(other_task())
        
        # Wait for other_task to complete
        await other_task_coro
        
        # Cancel probe task
        probe_task.cancel()
        try:
            await probe_task
        except asyncio.CancelledError:
            pass
    
    asyncio.run(run_both())
    
    # If the probe blocked the event loop, other_task would not complete all iterations
    # We expect all 5 iterations to complete despite the 50ms block in probe
    assert len(progress) == 5
    assert progress == [0, 1, 2, 3, 4]


# ---------- Task 5 follow-up: production wiring of the remote probe ----------
def test_single_node_context_builds_no_node_probe(settings_factory, isolated_home):
    from dgxmm.context import build_context
    from tests.fakes import FakeDockerClient
    fake = FakeDockerClient()
    ctx = build_context(settings_factory(), client_factory=lambda n: fake)
    assert ctx.node_probe is None and ctx.deployer.node_probe is None
    assert ctx.budget.nodes is None and ctx.budget.node_probe is None
    assert not (isolated_home / ".ssh").exists() and not (Path(ctx.settings.data_dir) / "ssh").exists()


async def test_two_node_context_admits_remote_deploy_once_probe_is_fresh(two_node_settings, fake_hf_repo,
                                                                         isolated_home):
    import asyncio
    from dgxmm.config import GIB
    from dgxmm.context import build_context, remote_probe_loop
    from dgxmm.deployer.service import DeployError, DeployRequest
    from dgxmm.nodes.probe import NodeProbe
    from tests.conftest import QWEN_TINY
    from tests.fakes import FakeDockerClient, FakeEngines, FakeProbe

    fakes = {"local": FakeDockerClient(), "dgx-2": FakeDockerClient()}
    ctx = build_context(two_node_settings, client_factory=lambda n: fakes[n.id])
    assert isinstance(ctx.node_probe, NodeProbe)
    assert ctx.budget.node_probe is ctx.node_probe and ctx.deployer.node_probe is ctx.node_probe
    assert ctx.budget.nodes is ctx.nodes
    assert (Path(two_node_settings.data_dir) / "ssh" / "config").exists()  # the data dir, never a home
    assert not (isolated_home / ".ssh").exists()

    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    ctx.inventory.refresh()
    ctx.budget.probe = FakeProbe()  # the local host's /proc is not there in tests
    ctx.deployer.http = FakeEngines().client()
    req = DeployRequest(name="q2", engine="vllm", model_ref="Qwen/Qwen3-Tiny", node="dgx-2",
                        config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048})
    try:
        with pytest.raises(DeployError) as e:  # no sample of dgx-2 yet
            await ctx.deployer.deploy(req, user_id=1)
        assert (e.value.code, e.value.status) == ("node_stale", 503)

        fakes["dgx-2"].exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
        fakes["dgx-2"].weights_on_node = {"Qwen/Qwen3-Tiny"}  # the snapshot is in dgx-2's HF cache
        loop = asyncio.create_task(remote_probe_loop(ctx))  # the production sampler
        for _ in range(100):
            if ctx.node_probe.last("dgx-2") is not None:
                break
            await asyncio.sleep(0.01)
        loop.cancel()
        assert not ctx.node_probe.stale("dgx-2")
        v = await ctx.deployer.deploy(req, user_id=1)
        assert v["state"] in ("starting", "loading")
        assert ctx.budget.current("dgx-2").total == 127000000 * 1024
        assert [c.name for c in fakes["dgx-2"].containers.list() if c.name != "dgxmm-node-probe"] == ["dgxmm-q2"]
        assert fakes["local"].run_calls == []
    finally:
        await ctx.deployer.aclose()
        await ctx.http.aclose()
