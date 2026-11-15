"""Remote memory probe tests."""
from dgxmm.nodes.probe import NodeProbe


def test_probe_parses_remote_meminfo_via_exec(two_node_registry):
    two_node_registry.fake("dgx-2").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    p = NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: 0.0)
    assert p.sample("dgx-2").total == 127000000 * 1024


def test_probe_stale_after_three_intervals(two_node_registry):
    t = [0.0]
    p = NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: t[0])
    p.sample("dgx-2")
    t[0] = 3 * two_node_registry.settings.remote_probe_interval_s + 0.1
    assert p.stale("dgx-2") is True and p.stale("local") is False


def test_exec_failure_keeps_old_sample_and_node_goes_stale(two_node_registry):
    """Exec failure returns the last sample without updating timestamp, so the node goes stale by age."""
    t = [0.0]
    p = NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: t[0])
    
    # First sample succeeds
    two_node_registry.fake("dgx-2").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    two_node_registry.fake("dgx-2").exec_fail = False
    mem1 = p.sample("dgx-2")
    assert mem1.total == 127000000 * 1024
    assert not p.stale("dgx-2")
    
    # Advance time and exec fails
    t[0] = 1.0
    two_node_registry.fake("dgx-2").exec_fail = True
    mem2 = p.sample("dgx-2")
    
    # Returns old sample, timestamp not updated
    assert mem2.total == 127000000 * 1024
    
    # Advance time past 3 intervals from the first sample (not the failed attempt)
    t[0] = 3 * two_node_registry.settings.remote_probe_interval_s + 0.1
    assert p.stale("dgx-2") is True


def test_ensure_starts_stopped_probe_container(two_node_registry):
    """ensure() should start a stopped dgxmm-node-probe container."""
    from tests.fakes import FakeContainer
    
    fake = two_node_registry.fake("dgx-2")
    
    # Create a stopped probe container
    stopped = FakeContainer(fake, "probe-cid", "dgxmm-node-probe", 
                          two_node_registry.settings.images.probe, 
                          ["sleep", "2147483647"], {"dgxmm.managed": "true", "dgxmm.role": "probe"}, {}, [], [], [],
                          status="exited")
    fake._containers["probe-cid"] = stopped
    
    p = NodeProbe(two_node_registry, two_node_registry.settings)
    p.ensure("dgx-2")
    
    # Container should be started
    assert stopped.status == "running"


def test_reachable_follows_the_last_sample_attempt(two_node_registry):
    """GET /api/nodes reads reachable(): the last attempt failed = unreachable at once (stale only after 3 intervals)."""
    p = NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: 0.0)
    assert p.reachable("dgx-2") is None and p.reachable("local") is True
    two_node_registry.fake("dgx-2").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    p.sample("dgx-2")
    assert p.reachable("dgx-2") is True
    two_node_registry.fake("dgx-2").exec_fail = True
    p.sample("dgx-2")
    assert p.reachable("dgx-2") is False and p.stale("dgx-2") is False
