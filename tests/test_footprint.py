"""Engine-log memory footprint parsing (Bug A: GB10 cgroup/NVML under-report CUDA allocations)."""
from pathlib import Path

import pytest

from dgxmm.config import GIB
from dgxmm.memory.footprint import parse_engine_log, parse_llamacpp, parse_vllm

LOGS = Path(__file__).parent / "fixtures" / "engine_logs"
MIB = 1024**2


def log(name: str) -> str:
    return (LOGS / name).read_text()


def test_vllm_profiled_log_uses_the_actual_usage_breakdown():
    fp = parse_vllm(log("vllm_profiled.txt"))
    # consumed (weights + non-torch) + peak activation + CUDA graphs + KV in use
    assert fp.total == pytest.approx((2.31 + 0.36 + 0.21 + 4.41) * GIB, rel=1e-6)
    assert fp.complete is True and fp.parts["kv_cache"] == pytest.approx(4.41 * GIB, rel=1e-6)


def test_vllm_basic_log_sums_weights_kv_and_largest_graph_capture():
    fp = parse_vllm(log("vllm_basic.txt"))
    assert fp.total == pytest.approx((25.11 + 26.22 + 2.73) * GIB, rel=1e-6)
    assert fp.complete is False  # no non-torch / activation accounting in this log format
    assert set(fp.parts) == {"weights", "kv_cache", "cuda_graphs"}


def test_vllm_log_without_kv_allocation_is_not_a_measurement():
    assert parse_vllm(log("vllm_loading.txt")) is None
    assert parse_vllm("") is None


def test_vllm_uses_the_latest_startup_when_the_log_holds_several():
    text = log("vllm_basic.txt") + log("vllm_profiled.txt")
    assert parse_vllm(text).total == pytest.approx((2.31 + 0.36 + 0.21 + 4.41) * GIB, rel=1e-6)


def test_llamacpp_sums_every_backend_buffer():
    fp = parse_llamacpp(log("llamacpp.txt"))
    assert fp.total == pytest.approx((604.15 + 157.65 + 2.32 + 896.00 + 298.75 + 20.01) * MIB, rel=1e-6)
    assert fp.parts["model"] == pytest.approx((604.15 + 157.65) * MIB, rel=1e-6)
    assert fp.parts["KV"] == pytest.approx(896 * MIB, rel=1e-6)
    assert fp.complete is False  # CUDA context / host heap are not in the buffer list


def test_llamacpp_without_compute_buffer_is_not_a_measurement():
    assert parse_llamacpp(log("llamacpp_loading.txt")) is None


def test_llamacpp_uses_the_latest_startup():
    text = log("llamacpp.txt").replace("896.00 MiB", "4480.00 MiB") + log("llamacpp.txt")
    assert parse_llamacpp(text).parts["KV"] == pytest.approx(896 * MIB, rel=1e-6)


def test_dispatch_by_engine():
    assert parse_engine_log("vllm", log("vllm_profiled.txt")) is not None
    assert parse_engine_log("llamacpp", log("llamacpp.txt")) is not None
    assert parse_engine_log("other", log("llamacpp.txt")) is None
