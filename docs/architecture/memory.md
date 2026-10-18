---
description: "The unified-memory budget, per-engine estimates, calibration and the plans offered when a model does not fit."
---

# Memory budget and fit planner

## Budget

On the GB10, CPU and GPU share one pool of memory, so DGXMM budgets host memory as a whole
(`memory/budget.py`):

```text
usable       = MemTotal − safety_margin_gib − os_reserve_gib            (defaults 8 GiB and 4 GiB)
committed    = Σ max(reserved, measured)  for starting / loading / ready / unknown deployments
             + Σ measured                for sleeping / stopping deployments
external     = max(used − measured_by_managed − ready_gap, 0)           used = MemTotal − MemAvailable
free_for_new = usable − committed − external
```

- *measured* is the engine container's cgroup `anon` + `kernel` memory.
- *external* is everything DGXMM does not manage (desktop, notebooks, other containers). DGXMM never
  touches it. The Dashboard lists the largest external processes with their container, user, masked
  command line, RSS and GPU memory (see [process view](processes.md)).
- **GB10 unified memory:** CUDA allocations are not charged to the container's cgroup, so a running
  model's memory shows up in `used` but only partly in *measured*. For `ready` deployments the gap
  between reservation and measurement (`ready_gap`) is credited back, so it does not appear as external
  usage. It is never credited for deployments that are still starting, which keeps the check
  conservative while memory is still growing.

The budget is refreshed every `budget_interval_s` (2 s) and pushed to browsers over SSE.

## Estimates

Each deploy gets an estimate (`memory/estimate.py`). The total is the sum of its parts:

| Part | vLLM | llama.cpp |
|---|---|---|
| weights | size of the safetensors files (plus a speculative draft model's weights) | size of the GGUF file |
| KV cache | from `config.json`: layers, KV heads, head size, KV dtype, attention window, for the KV pool (`kv_tokens`, default `min(context × seqs, 2 × context)`) | from GGUF metadata for `ctx_size` across `parallel` slots, K/V cache types |
| state | recurrent state of SSM and hybrid layers, per sequence | same |
| activations | from batched tokens, hidden and intermediate sizes, and vocabulary | compute buffers from `n_ubatch` |
| runtime | `vllm_runtime_fixed_gib` + `vllm_cuda_graph_gib` (3.0 + 1.5 GiB) | `llamacpp_runtime_fixed_gib` (0.5 GiB) |
| margin | +20 % of KV + state + activations when `config.json` lacks fields the formula needs | same |

The estimate's **confidence** is `low` (fields were missing), `estimated`, `calibrated` or `measured`.

## Calibration

Fifteen seconds after a deployment becomes ready, DGXMM records its measured memory next to the formula
estimate (`memory/calibration.py`). Later estimates for the same model are corrected:

- **Same model and exact same configuration:** `max(measured × 1.05, formula)`. Confidence: `measured`.
- **Same model, other configuration:** the formula is scaled by the mean measured/estimated ratio of
  the last 5 measurements, clamped to 1.0–1.5. Confidence: `calibrated`.
- **Out-of-memory kill** (Docker's `OOMKilled`, or the kernel OOM killer's line in the log; not a bare
  exit 137): the next estimate for that configuration gets +10 %. It never drops
  back below 1.1 × the size that was killed.

Calibration only raises estimates. It never lowers one below the formula.

## Plans when it does not fit

`planner/plans.py` offers, in this order (options that fit are listed first):

1. a smaller KV cache dtype (vLLM `kv_cache_dtype=fp8`; llama.cpp `q8_0` K and V cache);
2. a shorter context: the largest of context/2, /4, … (at least 4096) that fits;
3. fewer concurrent sequences / slots;
4. a smaller quantization of the same base model that is already on disk;
5. putting ready, sleep-capable vLLM deployments to sleep (frees roughly KV cache and activations);
6. stopping deployments.

For 5 and 6 every deployment that frees enough on its own gets its own plan (ids `sleep:<id>`,
`stop:<id>`), then the minimal combinations of up to three deployments (`stop:3+5`; no member can be
dropped), drawn from the eight best-ranked candidates; at most three sleep and four stop plans. The
ranking is least disruptive first: not busy, lower priority, then the longest idle (asleep counts as
idle). When no set within those bounds frees enough, one greedy plan in rank order is offered instead.

Sleep and stop plans run their actions first, then deploy. If the deploy still fails, those actions are
undone. Before the plans are returned, the [plan recommendation](decision-model.md#decision-model-decider) marks unsafe
plans and may recommend one.

## Next steps

- [Deployer and start-up gate](deployer.md)
- [Tuning the memory budget](../configuration/models-and-engines.md#tuning-the-memory-budget)
