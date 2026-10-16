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
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB, method="engine_log")
    exact = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    assert exact.total == int(12 * GIB * 1.05) and exact.confidence == "measured"
    other = cal.adjust("vllm", "r", "h2", est(20 * GIB))
    assert other.total == int(20 * GIB * 1.2) and other.confidence == "calibrated"
    cal.record("vllm", "r2", "x", 10 * GIB, 30 * GIB, method="engine_log")
    assert cal.adjust("vllm", "r2", "y", est(10 * GIB)).total == int(10 * GIB * 1.5)  # ratio clamped
    cal.record_oom("vllm", "r", "h1", 13 * GIB)
    after_oom = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    # pending penalty (12.6 x 1.10 = 13.9 GiB) is below the M1 floor (13 GiB OOM x 1.10 = 14.3 GiB)
    assert after_oom.total == max(int(int(12 * GIB * 1.05) * 1.10), int(13 * GIB * 1.10)) and "OOM" in after_oom.notes[-1]
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB, method="engine_log")  # a later success clears the pending penalty...
    # ...but M1: never swing back below the size that was OOM-killed (13 GiB) + 10%
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == int(13 * GIB * 1.10)


def test_oom_floor_survives_later_success(db):
    """M1: after an OOM at 20 GiB, a success measured at 8 GiB must not bring the estimate back to <=20 GiB."""
    cal = Calibration(db)
    cal.record_oom("vllm", "m", "h", 20 * GIB)
    cal.record("vllm", "m", "h", 22 * GIB, 8 * GIB, method="engine_log")
    adj = cal.adjust("vllm", "m", "h", est(18 * GIB))
    assert adj.total == int(20 * GIB * 1.10) and any("OOM" in n for n in adj.notes)
    # other configs of the same model are not floored by this config's OOM
    assert cal.adjust("vllm", "m", "other", est(18 * GIB)).total == 18 * GIB


def test_measurement_below_estimate_never_lowers(db):
    """Controller ruling: calibration may only RAISE estimates, never lower them."""
    cal = Calibration(db)
    # Record a measurement that's much lower than the estimate
    cal.record("vllm", "test/model", "cfg1", 10 * GIB, 4 * GIB, method="engine_log")
    # Adjust with the original estimate
    adjusted = cal.adjust("vllm", "test/model", "cfg1", est(10 * GIB))
    # The total should NOT be lowered below the original estimate
    assert adjusted.total == 10 * GIB, f"Expected {10 * GIB}, got {adjusted.total}"
    assert adjusted.confidence == "measured"

    # Test ratio adjustment also respects the floor of 1.0
    cal.record("vllm", "test/model2", "cfg2", 10 * GIB, 4 * GIB, method="engine_log")
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
    assert abs(r["measured"] - (engine + host)) < 16 * 1024**2 and r["config_hash"] == config_hash(d["config"])
    assert r["detail"]["memavailable_delta"] == engine + host + GIB // 4 and r["detail"]["complete"] is True


async def test_recorder_flags_disagreeing_memavailable_delta(stack):
    log_text = (FIXTURES / "vllm_profiled.txt").read_text()
    cal, _ = await _deploy_and_measure(stack, log_text=log_text, actual=GIB, drop=40 * GIB)
    (r,) = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    # the engine's own accounting is kept (a sum of real allocations: a safe lower bound); the delta only verifies
    assert r["quality"] == "disputed" and abs(r["measured"] - int(8.29 * GIB)) < 16 * 1024**2


async def test_recorder_skips_delta_when_another_deployment_changed_state(stack):
    log_text = (FIXTURES / "vllm_profiled.txt").read_text()
    cal, _ = await _deploy_and_measure(stack, log_text=log_text, actual=GIB, drop=40 * GIB, noisy=True)
    (r,) = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    assert r["quality"] == "unverified" and r["detail"]["memavailable_delta"] is None
