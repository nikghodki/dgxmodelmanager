"""Per-node budget tests."""
import asyncio


def test_budget_per_node_filters_usages_by_node(budget_service_two_nodes):
    asyncio.run(budget_service_two_nodes.refresh())
    b = budget_service_two_nodes.all()
    assert {u.name for u in b["dgx-2"].deployments} == {"remote-model"}
    assert {u.name for u in b["local"].deployments} == {"local-model"}
