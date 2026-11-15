"""Task 14: feature parity for remote and cluster deployments (dgxmm/deployer/cluster.py). On-demand mode, schedules,
the watchdog, switch, reconfigure, sleep and the planner work for a two-node cluster (head local, worker dgx-2) the way
they do for a single deployment: decisions about one node count only the cluster's share there, and stopping a cluster
frees every node. vLLM sleep across nodes failed the hardware check, so a cluster "sleep" stops every rank."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from dgxmm.config import GIB
from dgxmm.deployer import cluster
from dgxmm.deployer.service import DeployError, InsufficientMemory
from dgxmm.planner.plans import PlanAction
from tests.nodes_fixtures import CLUSTER, SMALL_VLLM

pytestmark = pytest.mark.timeout(60)
NODES = ("local", "dgx-2")


def _containers(d, node):
    return d.fake(node).containers.list()


async def _ready(d, req, **update):
    d.engines_http.all_healthy = True
    v = await d.deploy(req.model_copy(update=update) if update else req, user_id=1)
    assert await d.wait_ready(v["id"], 5) == "ready"
    return SimpleNamespace(id=v["id"], name=v["name"], port=v["port"], view=d.view(v["id"]))


# ---------- fixtures ----------
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
async def ondemand_two_nodes(deployer_two_nodes):
    """The real on-demand controller over the two-node deployer. reap_tick(): one reaper pass; request(name): one
    completion through the controller (a slot, then the engine at base_url_for, as the gateway proxies it);
    health_calls: (host, port) of every /health request since the last reset_calls()."""
    from dgxmm.ondemand.controller import OnDemandController
    d = deployer_two_nodes
    clock = Clock()

    class Controller(OnDemandController):
        @property
        def health_calls(self) -> list[tuple[str, int]]:
            return [(h, p) for h, p, path in d.engines_http.hosts if path == "/health"]
    ctl = Controller(SimpleNamespace(settings=d.settings, deployer=d, db=d.db, events=d.events,
                                             budget=d.budget), clock=clock, retry_s=0.005, settle_poll_s=0.001)
    ctl.attach(d)
    ctl.deployer, ctl.clock_ = d, clock
    ctl.fake = d.fake
    ctl.reap_tick = ctl.reap_once

    async def request(name: str):
        dep_id = d.row_by_name(name).id
        async with await ctl.acquire(dep_id) as _slot:
            r = await d.http.post(f"{d.base_url_for(d.view(dep_id))}/v1/completions",
                                  json={"model": name, "prompt": "hi"})
        return SimpleNamespace(status=r.status_code)
    ctl.request = request
    ctl.reset_calls = d.engines_http.hosts.clear
    yield ctl
    await d.aclose()


@pytest.fixture
async def ondemand_cluster(ondemand_two_nodes, deepseek_cluster_req):
    """A ready on-demand vLLM cluster deployment."""
    return await _ready(ondemand_two_nodes.deployer, deepseek_cluster_req, mode="on_demand")


@pytest.fixture
async def idle_cluster_dep(ondemand_two_nodes, ondemand_cluster):
    """The on-demand cluster, idle for longer than its timeout."""
    ondemand_two_nodes._observe(ondemand_two_nodes.deployer.view(ondemand_cluster.id))
    ondemand_two_nodes.clock_.t += ondemand_two_nodes.deployer.settings.ondemand_idle_timeout_s + 1
    return ondemand_cluster


@pytest.fixture
async def stopped_cluster_dep(ondemand_two_nodes, ondemand_cluster):
    """The on-demand cluster, stopped (every rank gone), with no /health calls recorded yet; its head answers
    health checks."""
    d = ondemand_two_nodes.deployer
    await d.stop(ondemand_cluster.id)
    assert all(_containers(d, n) == [] for n in NODES)
    d.engines_http.all_healthy = False
    d.engines_http.healthy.add(ondemand_cluster.port)
    ondemand_two_nodes.reset_calls()
    return ondemand_cluster


@pytest.fixture
async def watchdog_two_nodes(deployer_two_nodes, qwen_req):
    """A Watchdog (auto-protect on) over the two-node deployer with one ready local deployment of priority 50.
    tick(): a budget refresh, one watchdog pass and a refresh again; returns the deployments it stopped.
    budget(node): that node's current budget."""
    from dgxmm.watchdog import Watchdog
    d = deployer_two_nodes
    local = await _ready(d, qwen_req, name="qwen-local", alias="qwen-local", priority=50)
    d.settings.auto_protect = True
    w = Watchdog(d.settings, d, d.events, d.db)

    async def tick():
        before = {x["id"]: x["state"] for x in d.list_views()}
        await w.check_nodes(await d.budget.refresh())
        await d.budget.refresh()
        return [SimpleNamespace(id=x["id"], name=x["name"]) for x in d.list_views()
                if x["state"] == "stopped" and before.get(x["id"]) != "stopped"]
    w.tick = tick
    w.set_available = d.set_free
    w.budget = d.budget.current
    w.deployer, w.local_dep = d, local.id
    yield w
    await d.aclose()


@pytest.fixture
async def low_priority_cluster_dep(watchdog_two_nodes, deepseek_cluster_req):
    return await _ready(watchdog_two_nodes.deployer, deepseek_cluster_req, priority=0)


@pytest.fixture
async def scheduler_two_nodes(deployer_two_nodes):
    """The schedule service over the two-node deployer (quiet engine metrics for every deployment). run_due(at):
    one runner pass at HH:MM UTC (22:00 on 2026-10-19, earlier times on the next day), then waits for a start it
    made; state(dep_id)."""
    from dgxmm.decider.schedule import ScheduleService
    from dgxmm.metrics.store import MetricsStore
    d = deployer_two_nodes
    created = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    ctx = SimpleNamespace(settings=d.settings, db=d.db, events=d.events, deployer=d, budget=d.budget,
                          metrics=MetricsStore(d.db), decider=None)
    svc = ScheduleService(ctx, clock=lambda: created)

    def when(at: str) -> datetime:
        h, m = map(int, at.split(":"))
        day = datetime(2026, 10, 19, tzinfo=timezone.utc) + timedelta(days=0 if h >= 12 else 1)
        return day.replace(hour=h, minute=m, second=30)

    def quiet(dep_id: int, now: datetime) -> None:
        ring = ctx.metrics._ring(dep_id)
        t0 = now.timestamp() - 16 * 60
        for i in range(16 * 12):
            ring.append({"ts": t0 + 5 * i + 5, "running": 0.0, "waiting": 0.0, "prompt_tps": 0.0, "gen_tps": 0.0})

    async def run_due(at: str) -> str:
        now = when(at)
        for v in d.list_views():
            quiet(v["id"], now)
        out = await svc.run_due(now)
        for v in d.list_views():
            if v["state"] in ("queued", "starting", "loading"):
                await d.wait_ready(v["id"], 5)
        return out
    yield SimpleNamespace(run_due=run_due, state=d.state, svc=svc, deployer=d)
    await d.aclose()


@pytest.fixture
async def ready_single_dep(deployer_two_nodes, qwen_req):
    return await _ready(deployer_two_nodes, qwen_req)


# ---------- the brief's tests ----------
async def test_ondemand_reap_stops_all_ranks(ondemand_two_nodes, idle_cluster_dep):
    await ondemand_two_nodes.reap_tick()
    assert all(ondemand_two_nodes.fake(n).containers.list() == [] for n in ("local", "dgx-2"))


async def test_ondemand_start_waits_for_head_health(ondemand_two_nodes, stopped_cluster_dep):
    resp = await ondemand_two_nodes.request(stopped_cluster_dep.name)
    assert resp.status == 200 and ondemand_two_nodes.health_calls == [("192.0.2.1", stopped_cluster_dep.port)]


async def test_watchdog_cluster_victim_frees_both_nodes(watchdog_two_nodes, low_priority_cluster_dep):
    watchdog_two_nodes.set_available("local", 1 * 2**30)
    stopped = await watchdog_two_nodes.tick()
    assert [d.id for d in stopped] == [low_priority_cluster_dep.id]
    assert watchdog_two_nodes.budget("dgx-2").committed == 0


async def test_schedule_stop_then_wake_cluster(scheduler_two_nodes, ready_cluster_dep):
    from dgxmm.decider.schedule import ScheduleIn
    s, d = scheduler_two_nodes, scheduler_two_nodes.deployer
    view = d.view(ready_cluster_dep.id)
    s.svc.create(view, [ScheduleIn(action="sleep", days=0b1111111, at_local="22:00", timezone="UTC"),
                        ScheduleIn(action="wake", days=0b1111111, at_local="07:00", timezone="UTC")], 1)
    await scheduler_two_nodes.run_due(at="22:00")   # stop
    assert scheduler_two_nodes.state(ready_cluster_dep.id) == "stopped"
    assert all(_containers(d, n) == [] for n in NODES)
    await scheduler_two_nodes.run_due(at="07:00")   # wake
    assert scheduler_two_nodes.state(ready_cluster_dep.id) == "ready"
    assert all(len(_containers(d, n)) == 1 for n in NODES)


async def test_switch_single_to_cluster(deployer_two_nodes, ready_single_dep, deepseek_cluster_req):
    # the brief's actions=["stop"]: the plan action that stops the old deployment
    await deployer_two_nodes.switch(ready_single_dep.id, deepseek_cluster_req,
                                    actions=[{"type": "stop", "deployment_id": ready_single_dep.id}], user_id=1)
    assert deployer_two_nodes.row_by_name(deepseek_cluster_req.name).placement["kind"] == "cluster"


async def test_cluster_sleep_follows_hardware_check(deployer_two_nodes, ready_cluster_dep):
    await deployer_two_nodes.sleep(ready_cluster_dep.id)
    expected = "sleeping" if deployer_two_nodes.supports_sleep(ready_cluster_dep) else "stopped"
    assert deployer_two_nodes.state(ready_cluster_dep.id) == expected


# ---------- sleep ----------
async def test_supports_sleep_single_and_cluster(deployer_two_nodes, ready_cluster_dep, qwen_req):
    d = deployer_two_nodes
    single = await _ready(d, qwen_req, config={**SMALL_VLLM, "enable_sleep_mode": True})
    plain = await _ready(d, qwen_req, name="plain", alias="plain", config={**SMALL_VLLM, "enable_sleep_mode": False})
    assert d.supports_sleep(d.row(single.id)) and d.supports_sleep(d.view(single.id))
    assert not d.supports_sleep(d.row(plain.id))
    assert d.row(ready_cluster_dep.id).config.get("enable_sleep_mode")  # vLLM's default: configured with sleep mode
    assert cluster.SLEEP_ACROSS_NODES is False and not d.supports_sleep(d.row(ready_cluster_dep.id))
    v = d.view(ready_cluster_dep.id)
    assert v["sleep_capable"] is False and v["sleep_label"] == "sleep (stops all nodes)"
    assert "sleep_label" not in d.view(single.id)  # single deployments' views are unchanged


async def test_cluster_sleep_stops_every_rank_and_frees_both_nodes(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    v = await d.sleep(ready_cluster_dep.id)
    assert v["state"] == "stopped"
    assert all(_containers(d, n) == [] for n in NODES)
    await d.budget.refresh()
    assert all(d.budget.current(n).committed == 0 for n in NODES)
    assert not any(path in ("/sleep", "/wake_up") for _, _, path in d.engines_http.calls)
    await d.restart(ready_cluster_dep.id)  # Start brings it back
    assert await d.wait_ready(ready_cluster_dep.id, 5) == "ready"


async def test_cluster_sleep_precheck_and_state_checks(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes

    async def busy():
        return "requests running"
    with pytest.raises(DeployError) as e:
        await d.sleep(ready_cluster_dep.id, precheck=busy)
    assert e.value.code == "precheck" and d.state(ready_cluster_dep.id) == "ready"
    await d.stop(ready_cluster_dep.id)
    with pytest.raises(DeployError) as e:
        await d.sleep(ready_cluster_dep.id)
    assert e.value.code == "not_ready"


async def test_can_sleep_agrees_with_supports_sleep(deployer_two_nodes, ready_cluster_dep):
    """The planner never credits a sleep action on a cluster (its sleep is a stop): _can_sleep, _action_frees and
    the plan candidates all agree with supports_sleep."""
    d = deployer_two_nodes
    row = d.row(ready_cluster_dep.id)
    assert row.config.get("enable_sleep_mode") and not d._can_sleep(row)
    await d.budget.refresh()
    assert d._action_frees(PlanAction(type="sleep", deployment_id=row.id), d.budget.current()) == 0
    [cand] = [a for a in d.active_deps() if a.id == row.id]
    assert cand.can_sleep is False


# ---------- planner: the share on the node checked ----------
async def test_plans_credit_only_the_clusters_share_on_the_node(deployer_two_nodes, ready_cluster_dep, qwen_req):
    d = deployer_two_nodes
    single = await _ready(d, qwen_req, node="dgx-2")  # holds nothing on the local node
    await d.budget.refresh()
    row = d.row(ready_cluster_dep.id)
    local_share = ready_cluster_dep.ranks_reserved("local")
    assert local_share < row.reserved_bytes
    stop = PlanAction(type="stop", deployment_id=row.id)
    assert d._action_frees(stop, d.budget.current()) == max(local_share, d._actual(row.id) or 0)
    assert d._action_frees(stop, d.budget.current("dgx-2"), "dgx-2") == ready_cluster_dep.ranks_reserved("dgx-2")
    assert d._action_frees(PlanAction(type="stop", deployment_id=single.id), d.budget.current()) == 0
    cands = {a.id: a for a in d.active_deps()}
    assert cands[row.id].reserved == local_share and single.id not in cands
    assert {a.id: a.reserved for a in d.active_deps(node="dgx-2")} == {
        row.id: ready_cluster_dep.ranks_reserved("dgx-2"), single.id: d.row(single.id).reserved_bytes}
    assert d._replacing_credit(row.id) == max(local_share, d._actual(row.id) or 0)
    assert d.reserved_on(d.view(row.id), "dgx-2") == ready_cluster_dep.ranks_reserved("dgx-2")
    assert d.frees_on(d.view(row.id), "local") == max(local_share, d._actual(row.id) or 0)
    assert d.frees_on(d.view(single.id), "local") == 0


# ---------- on-demand ----------
async def test_cluster_on_demand_mode_set_and_deploy(deployer_two_nodes, ready_cluster_dep, deepseek_cluster_req):
    d = deployer_two_nodes
    v = d.set_mode(ready_cluster_dep.id, "on_demand", None)
    assert v["mode"] == "on_demand"
    other = await d.deploy(deepseek_cluster_req.model_copy(update={"name": "od", "alias": "od", "mode": "on_demand"}),
                           1)
    assert other["mode"] == "on_demand" and other["placement"]["kind"] == "cluster"


async def test_ondemand_victim_counts_only_the_share_on_the_short_node(ondemand_two_nodes, idle_cluster_dep):
    ctl, d = ondemand_two_nodes, ondemand_two_nodes.deployer
    share = d.reserved_on(d.view(idle_cluster_dep.id), "dgx-2")
    assert ctl._victims(-1, share, set(), "dgx-2") and ctl._victims(-1, share + 1, set(), "dgx-2") == []
    assert ctl._short_node(SimpleNamespace(extra={"node": "dgx-2"}), d.view(idle_cluster_dep.id)) == "dgx-2"


# ---------- watchdog ----------
async def test_watchdog_weighs_the_clusters_share_on_the_node(watchdog_two_nodes, low_priority_cluster_dep):
    """Same priority: the deployment holding more on the pressured node goes first, whatever the cluster's total."""
    d, w = watchdog_two_nodes.deployer, watchdog_two_nodes
    from dgxmm.db import Deployment
    with d.db.session() as s:
        s.get(Deployment, w.local_dep).priority = 0
        row = s.get(Deployment, low_priority_cluster_dep.id)
        ranks = [dict(r) for r in row.ranks]
        for r in ranks:
            r["reserved"] = 1 * GIB if r["node"] == "local" else 60 * GIB
        row.ranks, row.reserved_bytes = ranks, 61 * GIB
    assert d.row(w.local_dep).reserved_bytes > 1 * GIB
    assert w._victim("local")["id"] == w.local_dep
    assert w._victim("dgx-2")["id"] == low_priority_cluster_dep.id


# ---------- switch / reconfigure ----------
async def test_switch_cluster_to_single(deployer_two_nodes, ready_cluster_dep, qwen_req):
    d = deployer_two_nodes
    d.engines_http.all_healthy = True
    v = await d.switch(ready_cluster_dep.id, qwen_req.model_copy(update={"name": "single"}), [], user_id=1)
    assert v["state"] == "ready" and v["placement"] == {"kind": "single", "nodes": ["local"]}
    assert d.state(ready_cluster_dep.id) == "stopped"  # the cluster cannot sleep: it was stopped
    assert _containers(d, "dgx-2") == [] and len(_containers(d, "local")) == 1


async def test_switch_to_cluster_refused_per_node_before_touching_anything(deployer_two_nodes, ready_single_dep,
                                                                         deepseek_cluster_req):
    d = deployer_two_nodes
    d.set_free("dgx-2", 2 * GIB)  # dgx-2 cannot hold its rank; stopping the local deployment frees nothing there
    with pytest.raises(InsufficientMemory) as e:
        await d.switch(ready_single_dep.id, deepseek_cluster_req, [], user_id=1)
    assert e.value.extra["node"] == "dgx-2" and e.value.message.startswith("node dgx-2: ")
    assert d.state(ready_single_dep.id) == "ready" and d.row_by_name(deepseek_cluster_req.name) is None


async def test_reconfigure_cluster_restarts_every_rank(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    d.engines_http.all_healthy = True
    old = {n: [c.id for c in _containers(d, n)] for n in NODES}
    v = await d.reconfigure(ready_cluster_dep.id, dict(SMALL_VLLM, max_num_seqs=1))
    assert v["state"] == "ready" and v["config"]["max_num_seqs"] == 1
    row = d.row(ready_cluster_dep.id)
    assert sorted(r["node"] for r in row.ranks) == sorted(NODES) and sum(r["reserved"] for r in row.ranks) == \
        row.reserved_bytes
    for n in NODES:
        [c] = _containers(d, n)
        assert c.id not in old[n]


async def test_reconfigure_cluster_refused_per_node(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    d.set_free("dgx-2", 0)
    with pytest.raises(InsufficientMemory) as e:
        await d.reconfigure(ready_cluster_dep.id, dict(SMALL_VLLM, max_model_len=8192))
    assert e.value.extra["node"] == "dgx-2" and d.state(ready_cluster_dep.id) == "ready"


# ---------- remove ----------
async def test_cluster_remove_is_as_strict_as_remove_ranks(deployer_two_nodes, ready_cluster_dep, monkeypatch):
    """A rank container that remove_leftover could not remove (False) and that is still there keeps the row."""
    from dgxmm.docker_ctl import DockerController
    d = deployer_two_nodes
    await d.stop(ready_cluster_dep.id)
    name = cluster.rank_name(ready_cluster_dep.name, 1)
    left = SimpleNamespace(labels={"dgxmm.deployment_id": str(ready_cluster_dep.id)})
    monkeypatch.setattr(DockerController, "remove_leftover", lambda self, n, dep_id=None: False)
    monkeypatch.setattr(DockerController, "get", lambda self, ref: left if ref == name else None)
    with pytest.raises(DeployError) as e:
        await d.remove(ready_cluster_dep.id)
    assert e.value.code == "docker_error" and d.row(ready_cluster_dep.id) is not None
    monkeypatch.setattr(DockerController, "get", lambda self, ref: None)  # gone after all: removal goes ahead
    await d.remove(ready_cluster_dep.id)
    assert d.row(ready_cluster_dep.id) is None


# ---------- schedules / decider ----------
def test_schedules_allow_sleep_and_wake_for_clusters():
    from dgxmm.decider.schedule import allowed_actions, effective_action
    v = {"state": "ready", "sleep_capable": False, "sleep_label": "sleep (stops all nodes)"}
    assert allowed_actions(v) == ("sleep", "wake", "stop", "start")
    assert effective_action("sleep", v) == "stop" and effective_action("wake", {**v, "state": "stopped"}) == "start"
    assert allowed_actions({"state": "ready", "sleep_capable": False}) == ("stop", "start")


async def test_decider_facts_carry_placement_not_ranks(deployer_two_nodes, ready_cluster_dep):
    from dgxmm.decider.plans import build_state, facts_for
    d = deployer_two_nodes
    facts = facts_for(d.list_views(), None, db=d.db)
    f = facts[ready_cluster_dep.id]
    assert f.placement == CLUSTER and f.placement_label == "dgx-1 + dgx-2 (cluster)"
    assert not hasattr(f, "ranks")
    state = build_state(shortfall=GIB, model_ref="m", engine="vllm", cfg={}, deps=facts)
    assert "runs on dgx-1 + dgx-2 (cluster)" in state and "rank" not in state


async def test_reconfigure_remote_single_checks_its_own_node(deployer_two_nodes, ready_remote_dep):
    """A single deployment on dgx-2 is fitted against dgx-2's memory (the local node's free memory is no help)."""
    d = deployer_two_nodes
    d.set_free("dgx-2", 0)
    with pytest.raises(InsufficientMemory):
        await d.reconfigure(ready_remote_dep.id, dict(SMALL_VLLM, max_model_len=8192))
    assert d.state(ready_remote_dep.id) == "ready"



# ---------- fix round 1 ----------
def _free_for(d, node: str, free: int) -> None:
    """Make ``node``'s free_for_new about ``free`` (available minus margin and OS reserve while memory is in use)."""
    d.set_free(node, free + d.settings.margin_bytes + d.settings.os_reserve_bytes)


def _plan_ids(plans) -> set[int]:
    return {a["deployment_id"] for p in plans for a in p["actions"]}


async def test_preview_cluster_short_on_dgx2_plans_free_dgx2_only(deployer_two_nodes, qwen_req, deepseek_cluster_req):
    d = deployer_two_nodes
    local = await _ready(d, qwen_req, name="ql", alias="ql")
    remote = await _ready(d, qwen_req, name="qr", alias="qr", node="dgx-2")
    _free_for(d, "dgx-2", 1 * GIB)
    req = deepseek_cluster_req
    pv = await d.preview(req.engine, req.model_ref, req.config, placement=req.placement)
    assert pv["fit"]["fits"] is False and pv["node"] == "dgx-2"
    stops = [p for p in pv["plans"] if p["id"].startswith("stop:")]
    assert stops and _plan_ids(pv["plans"]) == {remote.id} and local.id not in _plan_ids(pv["plans"])
    assert pv["notes"] == []  # the decision model's memory is on the local node: no help on dgx-2


async def test_preview_single_on_dgx2_plans_free_dgx2_only(deployer_two_nodes, qwen_req):
    from dgxmm.deployer.service import Placement
    d = deployer_two_nodes
    local = await _ready(d, qwen_req, name="ql", alias="ql")
    remote = await _ready(d, qwen_req, name="qr", alias="qr", node="dgx-2")
    _free_for(d, "dgx-2", 0)
    pv = await d.preview("vllm", "Qwen/Qwen3-Tiny", dict(SMALL_VLLM),
                         placement=Placement(kind="single", nodes=["dgx-2"]))
    assert pv["fit"]["fits"] is False and pv["node"] == "dgx-2"
    assert _plan_ids(pv["plans"]) == {remote.id} and local.id not in _plan_ids(pv["plans"])
    plain = await d.preview("vllm", "Qwen/Qwen3-Tiny", dict(SMALL_VLLM))  # the local node: fits, no node key
    assert plain["fit"]["fits"] is True and "node" not in plain


async def test_preview_replacing_uses_its_placement(deployer_two_nodes, ready_cluster_dep, qwen_req):
    d = deployer_two_nodes
    remote = await _ready(d, qwen_req, name="qr", alias="qr", node="dgx-2")
    _free_for(d, "dgx-2", 0)
    row = d.row(ready_cluster_dep.id)
    pv = await d.preview(row.engine, row.model_ref, dict(SMALL_VLLM, max_model_len=8192), replacing=row.id)
    assert pv["node"] == "dgx-2" and _plan_ids(pv["plans"]) <= {remote.id}


class _LaggingProbe:
    """Freed memory of a container stopped on a remote node shows up in that node's sample only ``lag`` samples
    later (the remote probe interval)."""

    def __init__(self, d, lag: int):
        self.d, self.lag, self.pending = d, lag, []  # (node, bytes, samples left)
        orig = d.probe.sample

        def sample(arg):
            if isinstance(arg, str):
                keep = []
                for node, n, left in self.pending:
                    if node != arg:
                        keep.append((node, n, left))
                    elif left <= 1:
                        d.probe.remote_available[node] = d.probe.remote_available.get(
                            node, d.probe.default_available) + n
                    else:
                        keep.append((node, n, left - 1))
                self.pending = keep
            return orig(arg)
        d.probe.sample = sample
        stop = d.stop

        async def stopping(dep_id, *a, **kw):
            row = d.row(dep_id)
            before = row.state if row else None
            out = await stop(dep_id, *a, **kw)
            if before == "ready" and row.node == "dgx-2" and not cluster.is_cluster(row):
                self.pending.append(("dgx-2", row.reserved_bytes, self.lag))
            return out
        d.stop = stopping


async def _ondemand_singles(ctl, qwen_req, names_nodes):
    d, out = ctl.deployer, {}
    for name, node in names_nodes:
        out[name] = await _ready(d, qwen_req, name=name, alias=name, node=node, mode="on_demand")
        ctl._observe(d.view(out[name].id))
    return out


async def test_ondemand_make_room_for_cluster_short_on_dgx2_evicts_exactly_there(ondemand_two_nodes, qwen_req,
                                                                                 ondemand_cluster):
    ctl, d = ondemand_two_nodes, ondemand_two_nodes.deployer
    await d.stop(ondemand_cluster.id)
    singles = await _ondemand_singles(ctl, qwen_req, [("r1", "dgx-2"), ("r2", "dgx-2"), ("l1", "local")])
    ctl.clock_.t += d.settings.ondemand_idle_timeout_s + 1  # all idle
    share = d.row(ondemand_cluster.id).ranks[1]["reserved"]
    victim = d.row(singles["r1"].id).reserved_bytes
    assert share > victim // 2
    _free_for(d, "dgx-2", share - victim // 2)  # one dgx-2 victim is enough
    _LaggingProbe(d, lag=3)
    d.engines_http.all_healthy = True
    resp = await ctl.request(ondemand_cluster.name)
    assert resp.status == 200 and d.state(ondemand_cluster.id) == "ready"
    states = {n: d.state(v.id) for n, v in singles.items()}
    assert sorted(states[n] for n in ("r1", "r2")) == ["ready", "stopped"] and states["l1"] == "ready"


async def test_ondemand_cluster_never_fitting_on_second_node_refused_before_evicting(ondemand_two_nodes, qwen_req,
                                                                                    ondemand_cluster):
    from dgxmm.db import Deployment
    from dgxmm.ondemand.controller import OnDemandUnavailable
    ctl, d = ondemand_two_nodes, ondemand_two_nodes.deployer
    await d.stop(ondemand_cluster.id)
    singles = await _ondemand_singles(ctl, qwen_req, [("l1", "local")])
    ctl.clock_.t += d.settings.ondemand_idle_timeout_s + 1
    await d.budget.refresh()
    usable = d.budget.current("dgx-2").usable
    with d.db.session() as s:  # an always-on deployment holding nearly all of dgx-2
        s.add(Deployment(name="big", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=dict(SMALL_VLLM),
                         state="ready", port=8190, reserved_bytes=usable - GIB, estimate={}, node="dgx-2",
                         mode="always_on", placement={"kind": "single", "nodes": ["dgx-2"]}))
    _free_for(d, "local", 0)  # the cluster is short on local first (where l1 could be evicted)
    with pytest.raises(OnDemandUnavailable) as e:
        await ctl.request(ondemand_cluster.name)
    assert e.value.code == "does_not_fit" and "on node dgx-2" in e.value.message
    assert d.state(singles["l1"].id) == "ready" and d.state(ondemand_cluster.id) == "stopped"


async def test_undo_of_a_cluster_sleep_action_restarts_it(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    d.engines_http.all_healthy = True
    applied = await d.apply_actions([{"type": "sleep", "deployment_id": ready_cluster_dep.id}], user_id=1)
    assert applied == [("stop", ready_cluster_dep.id)] and d.state(ready_cluster_dep.id) == "stopped"
    assert await d.undo_actions(applied) == []
    assert await d.wait_ready(ready_cluster_dep.id, 5) == "ready"


async def test_wake_restarts_a_cluster_its_sleep_stopped(deployer_two_nodes, ready_cluster_dep):
    d = deployer_two_nodes
    d.engines_http.all_healthy = True
    await d.sleep(ready_cluster_dep.id)
    await d.wake(ready_cluster_dep.id)
    assert await d.wait_ready(ready_cluster_dep.id, 5) == "ready"
    assert all(len(_containers(d, n)) == 1 for n in NODES)
    await d.stop(ready_cluster_dep.id)  # stopped by the user: wake does not start it
    with pytest.raises(DeployError) as e:
        await d.wake(ready_cluster_dep.id)
    assert e.value.code == "not_sleeping" and d.state(ready_cluster_dep.id) == "stopped"


def _stopping_events(d, dep_id: int) -> list[str]:
    from sqlalchemy import select
    from dgxmm.db import DeploymentEvent
    with d.db.session() as s:
        return list(s.scalars(select(DeploymentEvent.message).where(DeploymentEvent.deployment_id == dep_id,
                                                                    DeploymentEvent.to_state == "stopping")))


@pytest.mark.parametrize("build", ["validated", "model_copy"])
async def test_switch_to_remote_single_fitted_against_dgx2(deployer_two_nodes, ready_single_dep, qwen_req, build):
    """The pre-check fits the new deployment on dgx-2 and refuses before touching the old one. ``model_copy``: a
    request whose placement was not re-derived from its node (pydantic skips model_post_init) is judged where
    deploy() will put it (its node), too."""
    from dgxmm.deployer.service import DeployRequest
    d = deployer_two_nodes
    d.engines_http.all_healthy = True
    if build == "validated":
        req = DeployRequest(**{**qwen_req.model_dump(exclude={"placement", "node"}), "name": "q2", "alias": None,
                               "node": "dgx-2"})
        assert req.placement.nodes == ["dgx-2"]
    else:
        req = qwen_req.model_copy(update={"name": "q2", "alias": None, "node": "dgx-2"})
        assert req.placement.nodes == ["local"]
    _free_for(d, "dgx-2", 0)  # local is roomy, dgx-2 is not: stopping the local one frees nothing there
    with pytest.raises(InsufficientMemory) as e:
        await d.switch(ready_single_dep.id, req, [], user_id=1)
    assert e.value.extra["node"] == "dgx-2" and e.value.message.startswith("node dgx-2: ")
    assert _stopping_events(d, ready_single_dep.id) == []  # refused at the pre-check: never stopped
    assert d.state(ready_single_dep.id) == "ready" and d.row_by_name("q2") is None
    d.set_free("dgx-2", 100 * GIB)
    _free_for(d, "local", 0)  # local is full, dgx-2 roomy: fits
    v = await d.switch(ready_single_dep.id, req, [], user_id=1)
    assert v["state"] == "ready" and d.row(v["id"]).node == "dgx-2"


async def test_wake_after_sleep_failed_start_and_user_stop_is_refused(deployer_two_nodes, ready_cluster_dep):
    """sleep -> stopped -> a start that fails -> the user stops the failed cluster (straight to "stopped", no
    "stopping" event): the latest stop is the user's, so wake is 409."""
    d = deployer_two_nodes
    await d.sleep(ready_cluster_dep.id)
    d.fake("dgx-2").fail_run = True
    with pytest.raises(DeployError):
        await d.restart(ready_cluster_dep.id)
    assert d.state(ready_cluster_dep.id) == "failed"
    d.fake("dgx-2").fail_run = False
    await d.stop(ready_cluster_dep.id)
    assert d.state(ready_cluster_dep.id) == "stopped"
    with pytest.raises(DeployError) as e:
        await d.wake(ready_cluster_dep.id)
    assert e.value.code == "not_sleeping" and d.state(ready_cluster_dep.id) == "stopped"


async def test_single_rows_placement_matches_node(deployer_two_nodes, ready_remote_dep):
    """A request copied with a new node (model_copy keeps the old placement) still gets a consistent row."""
    row = deployer_two_nodes.row(ready_remote_dep.id)
    assert row.node == "dgx-2" and row.placement == {"kind": "single", "nodes": ["dgx-2"]}
    assert deployer_two_nodes.nodes_for(deployer_two_nodes.view(row.id)) == ["dgx-2"]
