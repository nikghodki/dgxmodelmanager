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
    assert b.external == 70 * GIB - 40 * GIB  # used - managed actual
    assert b.free_for_new == b.usable - b.committed - b.external == 18 * GIB


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
