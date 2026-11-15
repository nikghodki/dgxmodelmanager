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
