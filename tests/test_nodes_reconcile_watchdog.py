"""Task 7: reconcile, the memory watchdog and LiteLLM per node."""
import logging
import time

import pytest
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import Deployment
from dgxmm.deployer.reconcile import reconcile
from dgxmm.nodes.registry import NodeUnreachable


# ---------- reconcile ----------

async def test_reconcile_adopts_remote_managed_container_by_label(deployer_two_nodes, starting_remote_dep):
    deployer_two_nodes.fake("dgx-2").add_managed(deployment_id=starting_remote_dep.id, name="dgxmm-q2")
    await reconcile(deployer_two_nodes)
    assert deployer_two_nodes.count_rows() == 1
    assert deployer_two_nodes.row(starting_remote_dep.id).container_id is not None
    assert deployer_two_nodes.state(starting_remote_dep.id) == "starting"
    await deployer_two_nodes.aclose()


async def test_reconcile_skips_unreachable_node_without_touching_its_rows(deployer_two_nodes, ready_remote_dep):
    deployer_two_nodes.fake("dgx-2").raise_on_all = NodeUnreachable("down")
    deployer_two_nodes.fake("dgx-2").add_managed(deployment_id=999, name="dgxmm-ghost")
    await reconcile(deployer_two_nodes)
    assert deployer_two_nodes.state(ready_remote_dep.id) == "ready"
    assert deployer_two_nodes.count_rows() == 1  # no orphan rows from the unreachable node
    assert "dgx-2" in deployer_two_nodes.reconcile_pending


async def test_reconcile_skips_stale_node_without_any_docker_call(deployer_two_nodes, ready_remote_dep, caplog):
    fake = deployer_two_nodes.fake("dgx-2")
    fake._containers.clear()  # would fail the row as "container missing" if reconciled
    fake.raise_on_all = RuntimeError("no docker call may reach a stale node")
    deployer_two_nodes.mark_stale("dgx-2")
    with caplog.at_level(logging.INFO, logger="dgxmm.deployer.reconcile"):
        await reconcile(deployer_two_nodes)
    assert deployer_two_nodes.state(ready_remote_dep.id) == "ready"
    assert fake.raised == 0
    assert len([r for r in caplog.records if "dgx-2" in r.getMessage() and "skipped" in r.getMessage()]) == 1


async def test_reconcile_remote_listing_is_bounded(deployer_two_nodes, ready_remote_dep):
    fake = deployer_two_nodes.fake("dgx-2")
    orig = fake.containers.list

    def slow_list(*a, **kw):
        time.sleep(1.0)
        return orig(*a, **kw)
    fake.containers.list = slow_list
    deployer_two_nodes.remote_liveness_timeout_s = 0.05
    t0 = time.monotonic()
    await reconcile(deployer_two_nodes)
    assert time.monotonic() - t0 < 0.8
    assert deployer_two_nodes.state(ready_remote_dep.id) == "ready"
    assert "dgx-2" in deployer_two_nodes.reconcile_pending


async def test_reconcile_retries_a_skipped_node_once_it_is_back(deployer_two_nodes, starting_remote_dep):
    d = deployer_two_nodes
    d.fake("dgx-2").add_managed(deployment_id=starting_remote_dep.id, name="dgxmm-q2")
    d.mark_stale("dgx-2")
    await reconcile(d)
    assert d.row(starting_remote_dep.id).container_id is None
    d.probe.stale_nodes.discard("dgx-2")
    await reconcile(d, retry=True)
    assert d.row(starting_remote_dep.id).container_id is not None
    assert d.reconcile_pending == {}
    await d.aclose()


async def test_reconcile_retry_leaves_rows_changed_since_startup(deployer_two_nodes, starting_remote_dep):
    d = deployer_two_nodes
    d.mark_stale("dgx-2")
    await reconcile(d)
    d._transition(starting_remote_dep.id, "failed", message="changed meanwhile", last_error="changed meanwhile")
    d.probe.stale_nodes.discard("dgx-2")
    await reconcile(d, retry=True)  # the row is not "container missing"-failed again, nor anything else
    assert d.row(starting_remote_dep.id).last_error == "changed meanwhile"


async def test_reconcile_records_remote_orphan_on_its_node(deployer_two_nodes):
    deployer_two_nodes.fake("dgx-2").add_managed(deployment_id=999, name="dgxmm-ghost")
    await reconcile(deployer_two_nodes)
    with deployer_two_nodes.db.session() as s:
        rows = s.scalars(select(Deployment)).all()
    assert [(r.name, r.state, r.node) for r in rows] == [("ghost", "unknown", "dgx-2")]


async def test_reconcile_local_unchanged_when_remote_node_is_down(deployer_two_nodes):
    deployer_two_nodes.fake("dgx-2").raise_on_all = NodeUnreachable("down")
    deployer_two_nodes.fake("local").add_managed(deployment_id=999, name="dgxmm-ghost")
    await reconcile(deployer_two_nodes)
    with deployer_two_nodes.db.session() as s:
        rows = s.scalars(select(Deployment)).all()
    assert [(r.name, r.node) for r in rows] == [("ghost", "local")]


# ---------- R11: a deployment whose node is no longer configured ----------

async def test_reconcile_marks_deployment_on_unconfigured_node(deployer_two_nodes, caplog):
    d = deployer_two_nodes
    dep_id = d.insert_row(node="dgx-9", state="ready", container_id="c" * 64)
    with caplog.at_level(logging.INFO, logger="dgxmm.deployer.reconcile"):
        await reconcile(d)
    row = d.row(dep_id)
    assert row.state == "ready" and row.container_id == "c" * 64  # row left as it is
    assert row.last_error == "node not configured: dgx-9"
    assert d.view(dep_id)["last_error"] == "node not configured: dgx-9"
    assert d.count_rows() == 1
    assert len([r for r in caplog.records if "dgx-9" in r.getMessage()]) == 1


@pytest.mark.parametrize("state", ["ready", "failed", "starting", "sleeping"])
async def test_stop_and_remove_of_deployment_on_unconfigured_node_release_db_state(deployer_two_nodes, state):
    d = deployer_two_nodes
    for n in ("local", "dgx-2"):
        d.fake(n).raise_on_all = RuntimeError("no docker call for a node that is not configured")
    dep_id = d.insert_row(node="dgx-9", state=state, container_id="c" * 64)
    v = await d.stop(dep_id, reason="user")
    assert v["state"] == "stopped" and v["container_id"] is None
    await d.remove(dep_id)
    assert d.count_rows() == 0
    assert d.fake("local").raised == 0 and d.fake("dgx-2").raised == 0


async def test_liveness_skips_deployment_on_unconfigured_node(deployer_two_nodes, caplog):
    d = deployer_two_nodes
    dep_id = d.insert_row(node="dgx-9", state="ready", container_id="c" * 64)
    with caplog.at_level(logging.WARNING):
        assert await d.check_liveness() == []
    assert d.state(dep_id) == "ready"
    assert not [r for r in caplog.records if "liveness" in r.getMessage()]


# ---------- watchdog ----------

async def test_watchdog_only_stops_victims_on_pressured_node(watchdog_two_nodes):
    watchdog_two_nodes.set_available("dgx-2", 1 * 2**30)
    stopped = await watchdog_two_nodes.tick()
    assert {d.node for d in stopped} == {"dgx-2"}
    assert watchdog_two_nodes.state_of(watchdog_two_nodes.local_dep) == "ready"
    assert not watchdog_two_nodes.deployer.fake("dgx-2")._containers  # stopped through dgx-2's controller


async def test_watchdog_local_pressure_stops_only_local(watchdog_two_nodes):
    watchdog_two_nodes.set_available("local", 1 * 2**30)
    stopped = await watchdog_two_nodes.tick()
    assert {d.node for d in stopped} == {"local"}
    assert watchdog_two_nodes.state_of(watchdog_two_nodes.remote_dep) == "ready"


async def test_watchdog_ignores_stale_remote_budget(watchdog_two_nodes):
    watchdog_two_nodes.set_available("dgx-2", 1 * 2**30)
    watchdog_two_nodes.deployer.mark_stale("dgx-2")
    assert await watchdog_two_nodes.tick() == []
    assert watchdog_two_nodes.state()["level"] == "ok"


async def test_watchdog_state_reports_the_pressured_node(watchdog_two_nodes):
    watchdog_two_nodes.deployer.settings.auto_protect = False
    watchdog_two_nodes.set_available("dgx-2", 1 * 2**30)
    assert await watchdog_two_nodes.tick() == []
    st = watchdog_two_nodes.state()
    assert st["level"] == "critical" and st["message"].startswith("dgx-2: ")
    assert st["nodes"]["dgx-2"]["level"] == "critical" and st["nodes"]["local"]["level"] == "ok"


# ---------- LiteLLM ----------

async def test_litellm_entry_for_remote_uses_cluster_ip(litellm_sync_two_nodes, ready_remote_dep):
    want = litellm_sync_two_nodes.desired()
    api_base = next(v[0] for k, v in want.items() if k[0] == str(ready_remote_dep.id))
    assert api_base.startswith("http://192.0.2.2:")


async def test_litellm_leaves_out_deployment_on_unconfigured_node(litellm_sync_two_nodes, deployer_two_nodes,
                                                                  ready_remote_dep):
    gone = deployer_two_nodes.insert_row(node="dgx-9", state="ready", container_id="c" * 64, name="gone", port=8160)
    with deployer_two_nodes.db.session() as s:
        s.get(Deployment, gone).alias = "gone"
    want = litellm_sync_two_nodes.desired()
    assert {k[0] for k in want} == {str(ready_remote_dep.id)}


async def test_single_node_watchdog_state_is_unchanged(stack):
    from dgxmm.memory.budget import compute_budget
    from dgxmm.memory.hostmem import HostMemory
    from dgxmm.watchdog import Watchdog
    w = Watchdog(stack.settings, stack.deployer, stack.events, stack.db)
    assert await w.check_nodes(compute_budget(HostMemory(121 * GIB, 5 * GIB, 0, 0, 0, 0), [], stack.settings)) \
        == "critical"
    assert set(w.state()) == {"level", "message", "since", "action"}
