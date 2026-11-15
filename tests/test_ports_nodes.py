"""Per-node port allocation."""
import pytest
from tests.nodes_fixtures import deployer_two_nodes, insert_deployment


def test_ports_unique_per_node(deployer_two_nodes):
    """Ports are unique per (node, port), not globally unique."""
    # Allocate port on local and dgx-2 - should get same port
    a = deployer_two_nodes.allocate_port("local")
    b = deployer_two_nodes.allocate_port("dgx-2")
    
    assert a == b == 8100
    
    # Insert deployment on local with port 8100
    insert_deployment(deployer_two_nodes, node="local", port=a)
    
    # Next allocation on local should skip 8100, but dgx-2 can still use it
    assert deployer_two_nodes.allocate_port("local") == 8101
    assert deployer_two_nodes.allocate_port("dgx-2") == 8100  # Still available on dgx-2
