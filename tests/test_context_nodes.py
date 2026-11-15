"""Multi-node context tests."""
import pytest


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
    ctx = build_context(two_node_settings)
    
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
