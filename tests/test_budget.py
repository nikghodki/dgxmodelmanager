from dgxmm.config import GIB
from dgxmm.memory.budget import BudgetService, DeploymentUsage, compute_budget
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.memory.hostmem import HostMemory, HostSample
from dgxmm.planner.fit import check_fit


def mem(total_gib=120, avail_gib=90):
    return HostMemory(total=total_gib * GIB, available=avail_gib * GIB, free=0, cached=0, swap_total=0, swap_free=0)


def est(total):
    return MemoryEstimate(weights=total, draft_weights=0, kv_cache=0, state=0, activations=0, runtime=0,
                          margin=0, confidence="estimated")


def test_budget_formula(settings):
    usages = [DeploymentUsage(1, "a", "ready", 40 * GIB, 38 * GIB, "c1"),
              DeploymentUsage(2, "b", "starting", 20 * GIB, 2 * GIB, "c2"),
              DeploymentUsage(3, "c", "stopped", 10 * GIB, 0, None)]
    b = compute_budget(mem(120, 50), usages, settings)
    assert b.usable == 120 * GIB - 8 * GIB - 4 * GIB
    assert b.committed == 60 * GIB
    assert b.managed_actual == 40 * GIB
    # used - managed actual - the ready deployment's unattributed reservation (40 - 38)
    assert b.external == 70 * GIB - 40 * GIB - 2 * GIB
    assert b.free_for_new == b.usable - b.committed - b.external == 20 * GIB


def test_gb10_ready_reservation_gap_is_not_external(settings):
    """GB10: CUDA allocations are not in the cgroup and NVML reports 0, so a ready model's
    reservation shows up in `used` but only partly in `actual`. The gap must not be counted twice."""
    ready = DeploymentUsage(1, "a", "ready", int(9.5 * GIB), 4 * GIB, "c1")
    starting = DeploymentUsage(2, "b", "starting", 20 * GIB, 2 * GIB, "c2")
    # used = 10 GiB real external + 9.5 GiB (ready, full reservation) + 5 GiB (starting, still growing)
    used = int(24.5 * GIB)
    m = HostMemory(total=121 * GIB, available=121 * GIB - used, free=0, cached=0, swap_total=0, swap_free=0)
    b = compute_budget(m, [ready, starting], settings)
    assert b.committed == int(9.5 * GIB) + 20 * GIB
    assert b.managed_actual == 6 * GIB
    # ready gap (5.5 GiB) credited; starting gap (3 GiB) NOT credited -> stays conservative
    assert b.external == used - 6 * GIB - int(5.5 * GIB) == 13 * GIB
    assert b.free_for_new == b.usable - b.committed - b.external


def test_gb10_starting_and_sleeping_get_no_credit(settings):
    used = 30 * GIB
    m = HostMemory(total=121 * GIB, available=121 * GIB - used, free=0, cached=0, swap_total=0, swap_free=0)
    for state in ("starting", "loading", "sleeping", "unknown"):
        b = compute_budget(m, [DeploymentUsage(1, "a", state, 20 * GIB, 2 * GIB, "c1")], settings)
        assert b.external == used - 2 * GIB, state


async def test_planner_capacity_ignores_ready_deployments(stack):
    """Capacity = usable - external must not shrink when a model becomes ready (GB10 accounting)."""
    from dgxmm.planner.colocate import planner_view
    from dgxmm.deployer.service import DeployRequest
    b0 = await stack.budget.refresh()
    cap0 = planner_view(stack.deployer, stack.inv, b0, set())["capacity"]
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 2,
                                                          "max_num_batched_tokens": 2048}), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    stack.probe.managed_actual[d["container_id"]] = 1 * GIB
    stack.probe.available -= d["reserved_bytes"]  # the model's memory now shows up in `used`
    b1 = await stack.budget.refresh()
    assert planner_view(stack.deployer, stack.inv, b1, set())["capacity"] == cap0


def test_sleeping_counts_actual_not_reserved(settings):
    b = compute_budget(mem(120, 100), [DeploymentUsage(1, "a", "sleeping", 40 * GIB, 16 * GIB, "c1")], settings)
    assert b.committed == 16 * GIB


def test_external_never_negative(settings):
    b = compute_budget(mem(120, 119), [DeploymentUsage(1, "a", "ready", 10 * GIB, 30 * GIB, "c")], settings)
    assert b.external == 0


def test_check_fit(settings):
    b = compute_budget(mem(120, 100), [], settings)
    ok = check_fit(est(50 * GIB), b)
    assert ok.fits and ok.headroom == b.free_for_new - 50 * GIB
    no = check_fit(est(b.free_for_new + 1), b)
    assert not no.fits and no.headroom == -1
    assert check_fit(est(b.free_for_new + 1), b, extra_free=1).fits


class FakeProbe:
    def __init__(self, sample):
        self._s = sample

    def sample(self, ids):
        return self._s

    def meminfo(self):
        return self._s.mem


async def test_budget_service_maps_container_actuals(settings):
    sample = HostSample(mem=mem(120, 80), managed={"c1": 30 * GIB}, external_top=[])
    svc = BudgetService(settings, FakeProbe(sample), lambda: [(1, "a", "ready", 32 * GIB, "c1")])
    b = await svc.refresh()
    assert b.deployments[0].actual == 30 * GIB and svc.current() is b and svc.last_sample is sample


# ---- per-process GPU memory attributed (NVML works) ----
def _used(gib):
    return HostMemory(total=121 * GIB, available=121 * GIB - int(gib * GIB), free=0, cached=0, swap_total=0,
                      swap_free=0)


def test_ready_measured_above_reservation_gets_no_gap(settings):
    """GPU 47.1 + host 6.6 GiB measured for a 48.7 GiB reservation: the whole footprint is the deployment's,
    no reservation credit on top, and external is what is left of `used`."""
    actual = int(53.7 * GIB)
    u = DeploymentUsage(1, "a", "ready", int(48.7 * GIB), actual, "c1")
    b = compute_budget(_used(64.9), [u], settings)
    assert b.managed_actual == actual
    assert b.external == int(64.9 * GIB) - actual  # ~11.2 GiB, not used - reserved (16.2)
    assert b.committed == actual  # max(reserved, measured)
    assert b.free_for_new == b.usable - actual - b.external


def test_ready_measured_below_reservation_credits_only_the_rest(settings):
    u = DeploymentUsage(1, "a", "ready", 60 * GIB, 50 * GIB, "c1")
    b = compute_budget(_used(75), [u], settings)
    assert b.external == 75 * GIB - 50 * GIB - 10 * GIB  # gap = reserved - measured, never counted twice
    assert b.committed == 60 * GIB


def test_gap_and_external_are_consistent_for_both_paths(settings):
    """Same physical state, measured with and without NVML: external is the same once the deployment is fully
    allocated (fallback credits reserved - cgroup, NVML path measures it)."""
    reserved = 50 * GIB
    with_nvml = DeploymentUsage(1, "a", "ready", reserved, reserved, "c1")
    cgroup_only = DeploymentUsage(1, "a", "ready", reserved, 6 * GIB, "c1")
    b1 = compute_budget(_used(61), [with_nvml], settings)
    b2 = compute_budget(_used(61), [cgroup_only], settings)
    assert b1.external == b2.external == 11 * GIB
