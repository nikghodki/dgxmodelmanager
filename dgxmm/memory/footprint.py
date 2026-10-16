"""Measure a deployment's real memory footprint from the engine's own startup accounting.

Why not cgroup + NVML: on GB10 unified memory, CUDA allocations are charged to neither the container
cgroup nor NVML per-process usage (see budget.py), so that number reads far below reality (a vLLM
Qwen3-0.6B reads ~3.9 GiB, a llama.cpp one ~0.55 GiB). Both engines log what they allocate:

* vLLM:  "Model loading took X GiB", "Available KV cache memory: Y GiB", "Graph capturing finished in
  N secs, took Z GiB"; newer builds also log "Actual usage is A GiB for consumed memory (weights +
  non-torch), B GiB for peak activation, and C GiB for CUDAGraph memory ... Current kv cache memory in
  use is D GiB" which is the complete device-side breakdown.
* llama.cpp: one "<backend> <kind> buffer size = N MiB" line per allocation (model, KV, compute, output).

Every number is an allocation the engine actually made, so their sum is a lower bound on the device-side
footprint (it misses e.g. the CUDA context when `complete` is False). Values are parsed from the latest
startup in the log only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GIB = 1024**3
MIB = 1024**2
_UNITS = {"GiB": GIB, "GB": GIB, "MiB": MIB, "MB": MIB}  # vLLM's "GB" was always GiB-based

_V_WEIGHTS = re.compile(r"(?:Model loading took|Loading model weights took) ([\d.]+) ?(GiB|GB)")
_V_KV = re.compile(r"Available KV cache memory: ([\d.]+) ?(GiB|GB)")
_V_GRAPH = re.compile(r"Graph capturing finished in [\d.]+ secs?, took ([\d.]+) ?(GiB|GB)")
_V_ACTUAL = re.compile(r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), ([\d.]+) GiB for "
                       r"peak activation, and ([\d.]+) GiB for CUDAGraph memory.*?Current kv cache memory in use is "
                       r"([\d.]+) GiB")
_L_START = re.compile(r"^build: \d+", re.M)
_L_BUF = re.compile(r"^\S+:\s+(\S+)\s+(\w+) buffer size =\s*([\d.]+) (MiB|GiB)", re.M)


@dataclass(frozen=True)
class LogFootprint:
    total: int
    complete: bool               # True when the log accounts for non-torch / runtime overhead too
    parts: dict[str, int] = field(default_factory=dict)


def _b(value: str, unit: str) -> int:
    return int(float(value) * _UNITS[unit])


def parse_vllm(text: str) -> LogFootprint | None:
    starts = [m.start() for m in _V_WEIGHTS.finditer(text)]
    if not starts:
        return None
    seg = text[starts[-1]:]  # latest engine startup only
    actual = list(_V_ACTUAL.finditer(seg))
    if actual:
        consumed, act, graphs, kv = (_b(v, "GiB") for v in actual[-1].groups())
        parts = {"weights_and_non_torch": consumed, "activations": act, "cuda_graphs": graphs, "kv_cache": kv}
        return LogFootprint(sum(parts.values()), True, parts)
    kv = list(_V_KV.finditer(seg))
    if not kv:
        return None  # KV cache not allocated yet: not a complete startup
    weights = _b(*_V_WEIGHTS.search(seg).groups())
    # CUDA graphs can be captured twice (memory profiling, then for real); take the larger: conservative
    graphs = max((_b(*m.groups()) for m in _V_GRAPH.finditer(seg)), default=0)
    parts = {"weights": weights, "kv_cache": _b(*kv[-1].groups()), "cuda_graphs": graphs}
    return LogFootprint(sum(parts.values()), False, parts)


def parse_llamacpp(text: str) -> LogFootprint | None:
    starts = [m.start() for m in _L_START.finditer(text)]
    seg = text[starts[-1]:] if starts else text
    parts: dict[str, int] = {}
    for m in _L_BUF.finditer(seg):
        _backend, kind, value, unit = m.groups()
        parts[kind] = parts.get(kind, 0) + _b(value, unit)
    if "model" not in parts or "compute" not in parts:
        return None  # context (KV + compute buffers) not created yet
    return LogFootprint(sum(parts.values()), False, parts)


_PARSERS = {"vllm": parse_vllm, "llamacpp": parse_llamacpp}


def parse_engine_log(engine: str, text: str) -> LogFootprint | None:
    parser = _PARSERS.get(engine)
    return parser(text) if parser and text else None
