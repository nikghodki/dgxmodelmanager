from dgxmm.config import GIB
from dgxmm.deployer.service import DeployRequest, config_hash
from dgxmm.memory.calibration import Calibration, CalibrationRecorder
from dgxmm.memory.estimate import MemoryEstimate


def est(total, margin=0):
    return MemoryEstimate(total - margin, 0, 0, 0, 0, 0, margin, "estimated")


def test_adjust_exact_ratio_and_oom(db):
    cal = Calibration(db)
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == 10 * GIB  # nothing recorded yet
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB)
    exact = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    assert exact.total == int(12 * GIB * 1.05) and exact.confidence == "measured"
    other = cal.adjust("vllm", "r", "h2", est(20 * GIB))
    assert other.total == int(20 * GIB * 1.2) and other.confidence == "calibrated"
    cal.record("vllm", "r2", "x", 10 * GIB, 30 * GIB)
    assert cal.adjust("vllm", "r2", "y", est(10 * GIB)).total == int(10 * GIB * 1.5)  # ratio clamped
    cal.record_oom("vllm", "r", "h1", 13 * GIB)
    after_oom = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    assert after_oom.total == int(int(12 * GIB * 1.05) * 1.10) and "OOM" in after_oom.notes[-1]
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB)  # a later success clears the OOM penalty
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == int(12 * GIB * 1.05)


def test_measurement_below_estimate_never_lowers(db):
    """Controller ruling: calibration may only RAISE estimates, never lower them."""
    cal = Calibration(db)
    # Record a measurement that's much lower than the estimate
    cal.record("vllm", "test/model", "cfg1", 10 * GIB, 4 * GIB)
    # Adjust with the original estimate
    adjusted = cal.adjust("vllm", "test/model", "cfg1", est(10 * GIB))
    # The total should NOT be lowered below the original estimate
    assert adjusted.total == 10 * GIB, f"Expected {10 * GIB}, got {adjusted.total}"
    assert adjusted.confidence == "measured"

    # Test ratio adjustment also respects the floor of 1.0
    cal.record("vllm", "test/model2", "cfg2", 10 * GIB, 4 * GIB)
    # Try to adjust a different config with the same model (ratio would be 0.4)
    adjusted2 = cal.adjust("vllm", "test/model2", "cfg3", est(10 * GIB))
    # Ratio should be clamped to 1.0 minimum, so estimate unchanged
    assert adjusted2.total == 10 * GIB


async def test_recorder_measures_after_ready(stack):
    cal = Calibration(stack.db)
    stack.deployer.calibration = cal
    stack.deployer.state_hooks.append(CalibrationRecorder(stack.deployer, cal, delay_s=0).on_state)
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}), 1)
    stack.probe.managed_actual[d["container_id"]] = 3 * GIB
    await stack.deployer.wait_ready(d["id"], 5)
    import asyncio
    await asyncio.sleep(0.05)
    rows = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    assert rows and rows[0]["measured"] == 3 * GIB and rows[0]["config_hash"] == config_hash(d["config"])
