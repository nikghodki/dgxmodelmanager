import asyncio
from pathlib import Path

from dgxmm.config import GIB
from dgxmm.db import MemoryCalibration
from dgxmm.deployer.service import DeployRequest, config_hash
from dgxmm.memory.calibration import Calibration, CalibrationRecorder
from dgxmm.memory.estimate import MemoryEstimate


def est(total, margin=0):
    return MemoryEstimate(total - margin, 0, 0, 0, 0, 0, margin, "estimated")


def test_adjust_exact_ratio_and_oom(db):
    cal = Calibration(db)
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == 10 * GIB  # nothing recorded yet
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB, method="engine_log", quality="verified")
    exact = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    assert exact.total == int(12 * GIB * 1.05) and exact.confidence == "measured"
    other = cal.adjust("vllm", "r", "h2", est(20 * GIB))
    assert other.total == int(20 * GIB * 1.2) and other.confidence == "calibrated"
    cal.record("vllm", "r2", "x", 10 * GIB, 30 * GIB, method="engine_log", quality="verified")
    assert cal.adjust("vllm", "r2", "y", est(10 * GIB)).total == int(10 * GIB * 1.5)  # ratio clamped
    cal.record_oom("vllm", "r", "h1", 13 * GIB)
    after_oom = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    # pending penalty (12.6 x 1.10 = 13.9 GiB) is below the M1 floor (13 GiB OOM x 1.10 = 14.3 GiB)
    assert after_oom.total == max(int(int(12 * GIB * 1.05) * 1.10), int(13 * GIB * 1.10)) and "OOM" in after_oom.notes[-1]
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB, method="engine_log", quality="verified")  # a later success clears the pending penalty...
    # ...but M1: never swing back below the size that was OOM-killed (13 GiB) + 10%
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == int(13 * GIB * 1.10)


def test_oom_floor_survives_later_success(db):
    """M1: after an OOM at 20 GiB, a success measured at 8 GiB must not bring the estimate back to <=20 GiB."""
    cal = Calibration(db)
    cal.record_oom("vllm", "m", "h", 20 * GIB)
    cal.record("vllm", "m", "h", 22 * GIB, 8 * GIB, method="engine_log", quality="verified")
    adj = cal.adjust("vllm", "m", "h", est(18 * GIB))
    assert adj.total == int(20 * GIB * 1.10) and any("OOM" in n for n in adj.notes)
    # other configs of the same model are not floored by this config's OOM
    assert cal.adjust("vllm", "m", "other", est(18 * GIB)).total == 18 * GIB


def test_measurement_below_estimate_never_lowers(db):
    """Controller ruling: calibration may only RAISE estimates, never lower them."""
    cal = Calibration(db)
    # Record a measurement that's much lower than the estimate
    cal.record("vllm", "test/model", "cfg1", 10 * GIB, 4 * GIB, method="engine_log", quality="verified")
    # Adjust with the original estimate
    adjusted = cal.adjust("vllm", "test/model", "cfg1", est(10 * GIB))
    # The total should NOT be lowered below the original estimate
    assert adjusted.total == 10 * GIB, f"Expected {10 * GIB}, got {adjusted.total}"
    assert adjusted.confidence == "measured"

    # Test ratio adjustment also respects the floor of 1.0
    cal.record("vllm", "test/model2", "cfg2", 10 * GIB, 4 * GIB, method="engine_log", quality="verified")
    # Try to adjust a different config with the same model (ratio would be 0.4)
    adjusted2 = cal.adjust("vllm", "test/model2", "cfg3", est(10 * GIB))
    # Ratio should be clamped to 1.0 minimum, so estimate unchanged
    assert adjusted2.total == 10 * GIB


# ---- Bug A: only reliable (engine-log) measurements are learned from ----
FIXTURES = Path(__file__).parent / "fixtures" / "engine_logs"
SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def test_unreliable_and_legacy_rows_are_ignored(db):
    """Legacy rows (no method: cgroup+NVML, which on GB10 misses CUDA memory) must not drive calibration."""
    with db.session() as s:
        s.add(MemoryCalibration(engine="vllm", repo="r", config_hash="h", estimated_bytes=10 * GIB,
                                measured_bytes=14 * GIB, oom=False))  # legacy row, method NULL
    cal = Calibration(db)
    assert cal.adjust("vllm", "r", "h", est(10 * GIB)).total == 10 * GIB
    assert cal.adjust("vllm", "r", "other", est(10 * GIB)).total == 10 * GIB
    cal.record("vllm", "r", "h", 10 * GIB, 13 * GIB, method="cgroup_nvml")
    assert cal.adjust("vllm", "r", "h", est(10 * GIB)).total == 10 * GIB
    cal.record("vllm", "r", "h", 10 * GIB, 12 * GIB, method="engine_log", quality="verified")
    assert cal.adjust("vllm", "r", "h", est(10 * GIB)).total == int(12 * GIB * 1.05)
    s = cal.summary("vllm", "r")
    assert [(x["method"], x["used"]) for x in s] == [("engine_log", True), ("cgroup_nvml", False), (None, False)]


def test_legacy_oom_rows_still_floor_estimates(db):
    with db.session() as s:
        s.add(MemoryCalibration(engine="vllm", repo="r", config_hash="h", estimated_bytes=20 * GIB,
                                measured_bytes=0, oom=True))
    assert Calibration(db).adjust("vllm", "r", "h", est(10 * GIB)).total == int(20 * GIB * 1.10)


def test_legacy_success_row_does_not_clear_pending_oom_penalty(db):
    cal = Calibration(db)
    cal.record("vllm", "r", "h", 10 * GIB, 12 * GIB, method="engine_log")
    cal.record_oom("vllm", "r", "h", 9 * GIB)
    with db.session() as s:  # an unreliable success after the OOM
        s.add(MemoryCalibration(engine="vllm", repo="r", config_hash="h", estimated_bytes=10 * GIB,
                                measured_bytes=4 * GIB, oom=False, method="cgroup_nvml"))
    adj = cal.adjust("vllm", "r", "h", est(10 * GIB))
    assert adj.total == int(int(12 * GIB * 1.05) * 1.10) and "OOM" in adj.notes[-1]


async def _deploy_and_measure(stack, *, log_text: str, actual: int, drop: int, noisy: bool = False):
    cal = Calibration(stack.db)
    stack.deployer.calibration = cal
    stack.deployer.state_hooks.append(CalibrationRecorder(stack.deployer, cal, delay_s=0).on_state)
    stack.http.all_healthy = False
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL), 1)
    await asyncio.sleep(0.05)  # "starting" hook snapshots MemAvailable
    if noisy:  # another deployment changes state inside the measurement window
        for hook in stack.deployer.state_hooks:
            await hook(d["id"] + 1000, "ready", "stopping")
    stack.probe.available -= drop
    stack.sdk.container(d["container_id"]).log_text = log_text
    stack.probe.managed_actual[d["container_id"]] = actual
    stack.http.all_healthy = True
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    await asyncio.sleep(0.1)
    return cal, d


async def test_recorder_does_not_record_cgroup_nvml_only(stack):
    """GB10: without engine accounting in the logs, the cgroup+NVML number is NOT recorded."""
    cal, _ = await _deploy_and_measure(stack, log_text="", actual=3 * GIB, drop=8 * GIB)
    assert cal.summary("vllm", "Qwen/Qwen3-Tiny") == []


async def test_recorder_measures_from_engine_log_plus_host_memory(stack):
    log_text = (FIXTURES / "vllm_profiled.txt").read_text()
    engine = int((2.31 + 0.36 + 0.21 + 4.41) * GIB)
    host = int(3.9 * GIB)  # cgroup anon: python heap etc. (CUDA allocations are not in it on GB10)
    cal, d = await _deploy_and_measure(stack, log_text=log_text, actual=host, drop=engine + host + GIB // 4)
    rows = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "engine_log" and r["quality"] == "verified" and r["used"] is True
    det = r["detail"]
    assert abs(det["observed_bytes"] - (engine + host)) < 16 * 1024**2 and r["config_hash"] == config_hash(d["config"])
    # vLLM: the KV it filled is replaced by the formula KV for this config (see the no-ratchet tests below)
    assert r["measured"] == det["learned_bytes"] == det["observed_bytes"] - det["parts"]["kv_cache"] + det["formula_kv_bytes"]
    assert r["detail"]["memavailable_delta"] == engine + host + GIB // 4 and r["detail"]["complete"] is True


async def test_recorder_skips_delta_when_another_deployment_changed_state(stack):
    log_text = (FIXTURES / "vllm_profiled.txt").read_text()
    cal, _ = await _deploy_and_measure(stack, log_text=log_text, actual=GIB, drop=40 * GIB, noisy=True)
    (r,) = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    assert r["quality"] == "unverified" and r["detail"]["memavailable_delta"] is None


# ---- review round 1 ----
def vllm_log(kv_gib: float, consumed=2.31, act=0.36, graphs=0.21) -> str:
    """A vLLM startup log whose KV pool filled whatever --gpu-memory-utilization allowed."""
    return ("(EngineCore pid=211) INFO 10-12 09:14:38 [model_runner.py:380] Model loading took 1.12 GiB memory and 6.2 seconds\n"
            f"(EngineCore pid=211) INFO 10-12 09:14:52 [gpu_worker.py:578] Available KV cache memory: {kv_gib:.2f} GiB\n"
            f"(EngineCore pid=211) INFO 10-12 09:14:56 [model_runner.py:906] Graph capturing finished in 3 secs, took {graphs:.2f} GiB\n"
            f"(EngineCore pid=211) INFO 10-12 09:14:56 [gpu_worker.py:804] Free memory on device (88.50/121.69 GiB) on startup. "
            f"Actual usage is {consumed:.2f} GiB for consumed memory (weights + non-torch), {act:.2f} GiB for peak activation, "
            f"and {graphs:.2f} GiB for CUDAGraph memory. Current kv cache memory in use is {kv_gib:.2f} GiB.\n")


async def _cycle(stack, *, dep_id=None, req=None, make_log, host: int, extra_drop: int = 0):
    """One launch: (re)start, let the engine 'allocate', reach ready, get measured, stop. Returns the reservation."""
    from dgxmm.memory.footprint import parse_engine_log
    stack.http.all_healthy = False
    d = await (stack.deployer.restart(dep_id) if dep_id else stack.deployer.deploy(req, 1))
    await asyncio.sleep(0.05)
    text = make_log(d)
    observed = parse_engine_log(d["engine"], text).total + host
    stack.probe.available -= observed + extra_drop
    stack.sdk.container(d["container_id"]).log_text = text
    stack.probe.managed_actual[d["container_id"]] = host
    stack.http.all_healthy = True
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    await asyncio.sleep(0.1)
    await stack.deployer.stop(d["id"])
    stack.probe.available += observed + extra_drop
    await asyncio.sleep(0.05)
    return d["id"], d["reserved_bytes"]


def _wire(stack):
    cal = Calibration(stack.db)
    stack.deployer.calibration = cal
    stack.deployer.state_hooks.append(CalibrationRecorder(stack.deployer, cal, delay_s=0).on_state)
    return cal


async def test_vllm_reservation_is_stable_when_kv_fills_the_budget(stack):
    """vLLM sizes its KV pool to fill --gpu-memory-utilization (derived from the reservation), so the logged
    footprint ~= the reservation. Learning that verbatim would grow the reservation on every restart."""
    from dgxmm.memory.estimate import vllm_gpu_util
    cal = _wire(stack)
    total = stack.probe.total
    non_kv = 2.31 + 0.36 + 0.21

    def make_log(d):
        budget_gib = vllm_gpu_util(d["reserved_bytes"], total) * total / GIB
        return vllm_log(budget_gib - non_kv)
    req = DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL)
    dep_id, r1 = await _cycle(stack, req=req, make_log=make_log, host=4 * GIB)
    reservations = [r1]
    for _ in range(3):
        reservations.append((await _cycle(stack, dep_id=dep_id, make_log=make_log, host=4 * GIB))[1])
    assert reservations[1] >= reservations[0]  # the host term is learned once...
    assert reservations[1] == reservations[2] == reservations[3], [x / GIB for x in reservations]  # ...then no ratchet
    rows = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    assert len(rows) == 4 and all(r["quality"] == "verified" for r in rows)
    assert len({r["measured"] for r in rows}) == 1
    # recorded = log non-KV + formula KV (+ host), not the KV vLLM chose to fill
    est = stack.deployer.view(dep_id)["estimate"]
    assert abs(rows[0]["measured"] - (int(non_kv * GIB) + est["kv_cache"] + est["state"] + 4 * GIB)) < 16 * 1024**2


async def test_llamacpp_reservation_is_stable_across_restarts(stack, settings):
    from tests.conftest import write_gguf
    cal = _wire(stack)
    write_gguf(Path(settings.gguf_roots[0].local) / "Qwen3-8B-Q4_K_M.gguf", pad_bytes=1 * GIB)
    stack.inv.refresh()
    ref = next(a.ref for a in stack.inv.list() if a.source == "gguf")
    big = (FIXTURES / "llamacpp.txt").read_text().replace("896.00 MiB", "12000.00 MiB")  # fixed n_ctx: KV is fixed
    req = DeployRequest(name="g", engine="llamacpp", model_ref=ref, config={"ctx_size": 8192, "parallel": 2})
    dep_id, r1 = await _cycle(stack, req=req, make_log=lambda d: big, host=GIB // 2)
    r = [r1] + [(await _cycle(stack, dep_id=dep_id, make_log=lambda d: big, host=GIB // 2))[1] for _ in range(3)]
    assert r[1] > r[0] and r[1] == r[2] == r[3], [x / GIB for x in r]
    assert len(cal.summary("llamacpp", ref)) == 4


async def test_vllm_without_formula_kv_is_not_recorded(stack):
    """An adopted/legacy deployment without a stored estimate cannot be normalised: record nothing."""
    cal = _wire(stack)
    stack.http.all_healthy = False
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL), 1)
    from dgxmm.db import Deployment
    with stack.db.session() as s:
        s.get(Deployment, d["id"]).estimate = {}
    stack.sdk.container(d["container_id"]).log_text = vllm_log(40.0)
    stack.http.all_healthy = True
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    await asyncio.sleep(0.1)
    assert cal.summary("vllm", "Qwen/Qwen3-Tiny") == []


async def test_memavailable_overshoot_is_learned_conservatively(stack, caplog):
    """Clean window, MemAvailable dropped far more than the engine accounted for: something was missed.
    Record measured + the unexplained excess, flag it, and don't present it as 'measured' confidence."""
    from dgxmm.memory.estimate import vllm_gpu_util
    cal = _wire(stack)
    total = stack.probe.total

    def make_log(d):
        return vllm_log(vllm_gpu_util(d["reserved_bytes"], total) * total / GIB - 2.88)
    req = DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL)
    import logging
    with caplog.at_level(logging.WARNING, logger="dgxmm.memory.calibration"):
        dep_id, _ = await _cycle(stack, req=req, make_log=make_log, host=GIB, extra_drop=10 * GIB)
    (r,) = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    det = r["detail"]
    assert r["quality"] == "disputed_high" and any("MemAvailable" in m for m in caplog.messages)
    assert det["unexplained_bytes"] == 10 * GIB and r["measured"] == det["learned_bytes"] + 10 * GIB
    est = stack.deployer.prepare("vllm", "Qwen/Qwen3-Tiny", SMALL).estimate
    assert est.confidence == "calibrated" and est.total >= int(r["measured"] * 1.05)


def test_quality_tolerance_is_capped():
    from dgxmm.memory.calibration import delta_quality
    assert delta_quality(10 * GIB, 10 * GIB + GIB) == "verified"            # 1 GiB floor
    assert delta_quality(10 * GIB, 10 * GIB + int(1.2 * GIB)) == "disputed_high"
    assert delta_quality(60 * GIB, 60 * GIB + 5 * GIB) == "disputed_high"   # 10 % would be 6 GiB; capped at 3
    assert delta_quality(60 * GIB, 60 * GIB + int(2.9 * GIB)) == "verified"
    assert delta_quality(60 * GIB, 50 * GIB) == "disputed_low"
    assert delta_quality(60 * GIB, None) == "unverified"


def test_only_verified_exact_rows_claim_measured_confidence(db):
    cal = Calibration(db)
    cal.record("vllm", "r", "h", 10 * GIB, 12 * GIB, method="engine_log", quality="unverified")
    a = cal.adjust("vllm", "r", "h", est(10 * GIB))
    assert a.total == int(12 * GIB * 1.05) and a.confidence == "calibrated" and "unverified" in a.notes[-1]
    cal.record("vllm", "r", "h", 10 * GIB, 12 * GIB, method="engine_log", quality="verified")
    assert cal.adjust("vllm", "r", "h", est(10 * GIB)).confidence == "measured"
