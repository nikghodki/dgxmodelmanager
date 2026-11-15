"""Task 11: per-rank estimates and placement options (the "it is a big model" suggestion)."""
import asyncio

import pytest

from dgxmm.deployer.service import DeployRequest, Placement
from dgxmm.memory.budget import Budget
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.planner.placement import PlacementOption, build_options, rank_bytes

GiB = 2**30
RUNTIME = 4 * GiB


def est(weights=0, kv=0, runtime=0, **kw) -> MemoryEstimate:
    """The brief's MemoryEstimate(weights, kv, runtime) on the real fields (kv -> kv_cache)."""
    base = dict(weights=weights, draft_weights=0, kv_cache=kv, state=0, activations=0, runtime=runtime, margin=0,
                confidence="estimated")
    base.update(kw)
    return MemoryEstimate(**base)


def budget(free: int) -> Budget:
    return Budget(total=128 * GiB, available=free, margin=0, os_reserve=0, usable=free, committed=0,
                  managed_actual=0, external=0, free_for_new=free, deployments=())


BUDGETS = {"local": budget(100 * GiB), "dgx-2": budget(100 * GiB)}


@pytest.fixture
def options_two_nodes(two_node_settings):
    """Options for a vLLM model of `model_gib` weights (+4 GiB runtime) on local + dgx-2 with the given free memory."""
    def _make(model_gib, free_gib, engine="vllm", stale=()):
        budgets = {n: budget(int(g * GiB)) for n, g in free_gib.items()}
        return build_options(engine, est(weights=int(model_gib * GiB), runtime=RUNTIME), two_node_settings,
                             budgets, stale=lambda n: n in stale)
    return _make


def test_big_model_recommends_cluster(options_two_nodes):
    opts = options_two_nodes(model_gib=167, free_gib={"local": 110, "dgx-2": 112})
    rec = [o for o in opts if o.recommended]
    assert [o.placement.kind for o in rec] == ["cluster"]
    assert all(not o.fits and o.reason == "too_big_for_one_node" for o in opts if o.placement.kind == "single")


def test_small_model_recommends_single_node_with_most_free(options_two_nodes):
    opts = options_two_nodes(model_gib=22, free_gib={"local": 40, "dgx-2": 110})
    assert [o.placement.nodes for o in opts if o.recommended] == [["dgx-2"]]


def test_nothing_fits_recommends_none(options_two_nodes):
    opts = options_two_nodes(model_gib=510, free_gib={"local": 110, "dgx-2": 110})
    assert not any(o.recommended for o in opts)
    assert all(o.reason == "insufficient_memory" for o in opts if o.placement.kind == "cluster")


def test_vllm_rank_bytes_halve_weights_and_kv():
    e = est(weights=100 * GiB, kv=20 * GiB, runtime=4 * GiB)
    assert rank_bytes("vllm", e, Placement(kind="cluster", nodes=["local", "dgx-2"]), BUDGETS) == \
        {"local": 64 * GiB, "dgx-2": 64 * GiB}


def test_vllm_rank_bytes_margin_split_activations_once_per_rank():
    # margin is proportional (split like the weights); activations and runtime are absolute (full on every rank)
    e = est(weights=100 * GiB, kv=20 * GiB, runtime=4 * GiB, draft_weights=2 * GiB, state=2 * GiB,
            activations=3 * GiB, margin=5 * GiB)
    per_rank = (51 + 11) * GiB + int(2.5 * GiB) + (4 + 3) * GiB
    assert rank_bytes("vllm", e, Placement(kind="cluster", nodes=["local", "dgx-2"]), BUDGETS) == \
        {"local": per_rank, "dgx-2": per_rank}


def test_single_rank_bytes_is_the_single_node_total():
    e = est(weights=100 * GiB, kv=20 * GiB, runtime=4 * GiB, activations=1 * GiB, margin=3 * GiB)
    for engine in ("vllm", "llamacpp"):
        assert rank_bytes(engine, e, Placement(kind="single", nodes=["dgx-2"]), BUDGETS) == {"dgx-2": e.total}


def test_llamacpp_rank_bytes_split_by_free_memory():
    e = est(weights=90 * GiB, kv=30 * GiB, runtime=1 * GiB)
    budgets = {"local": budget(40 * GiB), "dgx-2": budget(80 * GiB)}
    assert rank_bytes("llamacpp", e, Placement(kind="cluster", nodes=["local", "dgx-2"]), budgets) == \
        {"local": 41 * GiB, "dgx-2": 81 * GiB}


def test_options_shape_and_labels(options_two_nodes):
    opts = options_two_nodes(model_gib=22, free_gib={"local": 40, "dgx-2": 110})
    assert [(o.placement.kind, o.placement.nodes) for o in opts] == \
        [("single", ["local"]), ("single", ["dgx-2"]), ("cluster", ["local", "dgx-2"])]
    assert [o.label for o in opts] == ["dgx-1", "dgx-2", "dgx-1 + dgx-2 (cluster)"]
    single = opts[0]
    assert isinstance(single, PlacementOption)
    assert single.per_node == {"local": 26 * GiB} and single.free == {"local": 40 * GiB}
    assert single.fits and single.reason == "fits" and not single.recommended
    assert opts[2].per_node == {"local": 15 * GiB, "dgx-2": 15 * GiB} and opts[2].fits
    assert not opts[2].recommended  # a single node fits: the cluster is never recommended over it
    d = opts[1].as_dict()
    assert d["placement"] == {"kind": "single", "nodes": ["dgx-2"]} and d["recommended"] is True


def test_stale_node_never_recommended(options_two_nodes):
    opts = options_two_nodes(model_gib=22, free_gib={"local": 40, "dgx-2": 110}, stale={"dgx-2"})
    by = {(o.placement.kind, tuple(o.placement.nodes)): o for o in opts}
    remote = by[("single", ("dgx-2",))]
    assert not remote.fits and remote.reason == "node_stale" and not remote.recommended
    cluster = by[("cluster", ("local", "dgx-2"))]
    assert not cluster.fits and cluster.reason == "node_stale"
    assert [o.placement.nodes for o in opts if o.recommended] == [["local"]]


def test_stale_node_with_big_model_recommends_none(options_two_nodes):
    opts = options_two_nodes(model_gib=167, free_gib={"local": 110, "dgx-2": 112}, stale={"dgx-2"})
    assert not any(o.recommended for o in opts)


def test_no_cluster_option_for_unsupported_engine(options_two_nodes):
    opts = options_two_nodes(model_gib=167, free_gib={"local": 110, "dgx-2": 112}, engine="decision")
    assert [o.placement.kind for o in opts] == ["single", "single"]
    assert all(o.reason == "insufficient_memory" for o in opts)  # no cluster option: nothing could need less


def test_no_cluster_option_without_two_cluster_ready_nodes(three_node_settings):
    # three_node_settings: local has no cluster_ip/nccl; dgx-2 and dgx-3 are cluster-ready but local must be head
    budgets = {n: budget(110 * GiB) for n in ("local", "dgx-2", "dgx-3")}
    opts = build_options("vllm", est(weights=167 * GiB, runtime=RUNTIME), three_node_settings, budgets)
    assert [o.placement.kind for o in opts] == ["single", "single", "single"]
    assert not any(o.recommended for o in opts)


def test_missing_budget_is_stale(options_two_nodes):
    opts = options_two_nodes(model_gib=22, free_gib={"local": 40})
    remote = next(o for o in opts if o.placement.nodes == ["dgx-2"])
    assert remote.reason == "node_stale" and not remote.fits and remote.free == {"dgx-2": 0}


# ---------- through the deployer (real engine estimate, real budgets) ----------

def _req(**kw):
    return DeployRequest(name="tiny", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                         config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}, **kw)


def test_deployer_placement_options_uses_real_estimate(deployer_two_nodes):
    d = deployer_two_nodes
    d.set_free("local", 40 * GiB)
    d.set_free("dgx-2", 100 * GiB)
    asyncio.run(d.budget.refresh())
    opts = d.placement_options(_req(placement={"kind": "cluster", "nodes": ["dgx-2", "local"]}))  # placement ignored
    total = d.prepare("vllm", "Qwen/Qwen3-Tiny", _req().config).estimate.total
    assert [o.placement.nodes for o in opts] == [["local"], ["dgx-2"], ["local", "dgx-2"]]
    assert opts[0].per_node == {"local": total}
    assert [o.placement.nodes for o in opts if o.recommended] == [["dgx-2"]]


def test_deployer_placement_options_stale_remote(deployer_two_nodes):
    d = deployer_two_nodes
    d.set_free("local", 40 * GiB)
    d.set_free("dgx-2", 100 * GiB)
    asyncio.run(d.budget.refresh())
    d.mark_stale("dgx-2")
    opts = d.placement_options(_req())
    remote = next(o for o in opts if o.placement.nodes == ["dgx-2"])
    assert remote.reason == "node_stale" and not remote.recommended
    assert [o.placement.nodes for o in opts if o.recommended] == [["local"]]


# ---------- API ----------

def test_api_placement_options(api_stack):
    from tests.test_api_deployments import SMALL, admin
    h = admin(api_stack)
    r = api_stack.post("/dgxmm/api/deployments/placement-options", headers=h,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 200, r.text
    opts = r.json()
    assert [o["placement"] for o in opts] == [{"kind": "single", "nodes": ["local"]}]
    assert opts[0]["recommended"] and opts[0]["fits"] and opts[0]["reason"] == "fits"
    assert set(opts[0]) >= {"label", "per_node", "free"}


def test_api_placement_options_errors_and_viewer(api_stack):
    from tests.test_api_deployments import SMALL, admin
    h = admin(api_stack)
    r = api_stack.post("/dgxmm/api/deployments/placement-options", headers=h,
                       json={"name": "x", "engine": "vllm", "model_ref": "org/missing", "config": SMALL})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "model_not_found"
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    vh = {"X-CSRF-Token": r.json()["csrf_token"]}
    r = api_stack.post("/dgxmm/api/deployments/placement-options", headers=vh,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 403


# ---------- fix round 1: proportional margin split, edge cases, API errors ----------

def test_deepseek_like_low_confidence_model_recommends_cluster(two_node_settings):
    """A 167 GB-class model (weights ~155 GiB, KV ~10 GiB) with a low-confidence 20% margin, then scaled by a
    calibration ratio: the proportional margin is split across ranks, so the cluster fits at 110 GiB free per node."""
    kv, act = 10 * GiB, 2 * GiB
    e = est(weights=155 * GiB, kv=kv, activations=act, runtime=RUNTIME, margin=int(0.20 * (kv + act)),
            confidence="low")
    e = e.with_total(int(e.total * 1.15), "calibrated", "scaled x1.15")
    budgets = {"local": budget(110 * GiB), "dgx-2": budget(110 * GiB)}
    opts = build_options("vllm", e, two_node_settings, budgets)
    cluster = next(o for o in opts if o.placement.kind == "cluster")
    assert cluster.fits and cluster.recommended and cluster.reason == "fits"
    assert all(o.reason == "too_big_for_one_node" and not o.fits for o in opts if o.placement.kind == "single")
    shard = e.weights + e.kv_cache + e.margin
    assert cluster.per_node == {n: -(-shard // 2) + act + RUNTIME for n in ("local", "dgx-2")}


def test_proportional_margin_split_llamacpp_by_free():
    e = est(weights=60 * GiB, kv=0, runtime=1 * GiB, activations=1 * GiB, margin=30 * GiB)
    budgets = {"local": budget(30 * GiB), "dgx-2": budget(60 * GiB)}
    assert rank_bytes("llamacpp", e, Placement(kind="cluster", nodes=["local", "dgx-2"]), budgets) == \
        {"local": (30 + 2) * GiB, "dgx-2": (60 + 2) * GiB}


def test_llamacpp_node_with_no_or_negative_free_gets_no_share():
    e = est(weights=90 * GiB, kv=10 * GiB, runtime=1 * GiB)
    pl = Placement(kind="cluster", nodes=["local", "dgx-2"])
    for low in (0, -5 * GiB):
        budgets = {"local": budget(low), "dgx-2": budget(80 * GiB)}
        assert rank_bytes("llamacpp", e, pl, budgets) == {"local": 1 * GiB, "dgx-2": 101 * GiB}


def test_llamacpp_all_zero_free_falls_back_to_even_split():
    e = est(weights=90 * GiB, kv=10 * GiB, runtime=1 * GiB)
    budgets = {"local": budget(0), "dgx-2": budget(-1 * GiB)}
    assert rank_bytes("llamacpp", e, Placement(kind="cluster", nodes=["local", "dgx-2"]), budgets) == \
        {"local": 51 * GiB, "dgx-2": 51 * GiB}


def test_equal_free_tie_break_is_first_in_config_order(options_two_nodes):
    opts = options_two_nodes(model_gib=22, free_gib={"local": 100, "dgx-2": 100})
    assert [o.placement.nodes for o in opts if o.recommended] == [["local"]]


def test_api_placement_options_unknown_engine_and_invalid_ref(api_stack):
    from tests.test_api_deployments import SMALL, admin
    h = admin(api_stack)
    calls = []
    orig = api_stack.ctx.deployer.budget.refresh

    async def counting_refresh():
        calls.append(1)
        return await orig()
    api_stack.ctx.deployer.budget.refresh = counting_refresh
    url = "/dgxmm/api/deployments/placement-options"
    r = api_stack.post(url, headers=h, json={"name": "x", "engine": "nope", "model_ref": "Qwen/Qwen3-Tiny",
                                             "config": SMALL})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_engine"
    r = api_stack.post(url, headers=h, json={"name": "x", "engine": "vllm", "model_ref": "../../etc/passwd",
                                             "config": SMALL})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_model_ref"
    assert calls == []  # validation runs before the budget refresh
    r = api_stack.post(url, headers=h, json={"name": "x", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                             "config": SMALL})
    assert r.status_code == 200 and calls == [1]
