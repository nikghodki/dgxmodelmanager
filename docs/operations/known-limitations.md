---
description: "What DGXMM does not do, or does only on a best-effort basis."
---

# Known limitations

- **Command-line masking is best-effort.** The Processes page and the dashboard's external memory card
  mask passwords, tokens and keys, and mask a whole value when it cannot be parsed reliably (fail
  closed). Some rare wrapper or quoting forms may still not be recognised, so a secret passed as a
  command-line argument can appear unmasked. Do not pass secrets as arguments; use files or
  environment variables, which DGXMM never reads.
- **Deletion only knows about DGXMM's deployments.** A model is protected from deletion when a DGXMM
  deployment uses it. Containers or processes that DGXMM does not manage (a vLLM started by hand, a
  notebook) and use the same HF cache or GGUF directory are not detected. Check the
  [Processes](../user-guide/processes.md) page, or `docker ps`, before you delete a model.
- **Clearing the master key does not deregister models.** Only turning the integration off or changing
  the internal URL removes DGXMM's models from LiteLLM. Clearing a master key stored on the LiteLLM
  page (or removing the secret file) leaves the existing registrations in LiteLLM; turn the
  integration off first if you want them removed.
- **Virtual-key limits cannot be cleared.** LiteLLM's key update cannot remove a budget or rate limit once it is
  set, so **Edit** can only change it. Set a higher value, or create a new key
  and revoke the old one.
- **Per-process GPU memory on GB10.** Per-process GPU figures come from NVML (driver 580 and later report
  them on the GB10). When NVML reports no value for a process, DGXMM shows `0`, so the GPU column on the
  Processes page may read 0 for a process that does use the GPU, and a deployment's measured memory is its
  cgroup memory only. The budget then credits the rest of a ready deployment's reservation instead (see
  [memory budget](../architecture/memory.md#budget)).
- **An on-demand request that cannot fit may wait the full queue limit.** The immediate `does_not_fit`
  answer compares the model with the memory DGXMM's deployments could free; memory held by other programs
  on the host is not counted. With a lot of such memory in use, a request whose model can never fit is
  queued and fails only after `ondemand_queue_timeout_s` (15 minutes) with `ondemand_queue_timeout`.
- **The on-demand queue does not survive a restart.** Restarting DGXMM drops waiting requests (their
  connections close) and resets the in-flight counters; ready on-demand models then wait a full idle
  timeout before they are stopped.
- **No fast sleep on DGX Spark.** On unified memory vLLM's level-1 sleep frees nothing, so a single-node vLLM
  "sleep" stops the model (`vllm_sleep: stop`) and its wake is a full model load (minutes for a large model),
  like a cluster's. `vllm_sleep: level1` is meant for hosts with separate GPU memory; on GB10 it fails
  (DGXMM then restarts the deployment). See [sleep and wake](../user-guide/deployments.md#sleep-and-wake).
- **The health probe checks the engine's health route only.** A ready deployment is failed after
  `engine_health_failures` (3) failed `GET /health` probes `engine_health_interval_s` (30 s) apart, so about 1 to
  1.5 minutes after its engine stopped answering; requests sent to it meanwhile hang or fail. The probe sends no
  test completion (it would count as traffic and keep every deployment from ever looking idle), so an engine whose
  `/health` still answers 200 while it cannot serve completions is not detected (vLLM's `/health` does fail once
  its EngineCore is dead). The failure counts are kept in memory only.
- **NCCL all-reduce throughput with GPU Direct RDMA disabled.** Measured cluster all-reduce bandwidth is
  approximately 3 GB/s on 2x DGX Spark over ConnectX-7 direct link with RoCE, because GPU Direct RDMA is
  disabled (the vLLM image's libmlx5 lacks `mlx5dv_reg_dmabuf_mr`). With GPU Direct RDMA enabled, the
  expected bandwidth is significantly higher. This affects multi-node vLLM cluster performance.
- **GB10 unified memory and cgroup accounting.** On GB10, CUDA allocations are not charged to
  cgroups or RSS, so the container's measured memory is only its host-side memory (the process, page
  cache, tmpfs). Keep a host-level GPU-aware OOM guard to prevent the host from running out of
  unified memory. This is a known GB10 driver behavior. See
  [Host memory protection](../operations/host-memory-protection.md).
- **Memory estimates for attention the parser does not model.** DGXMM estimates KV-cache memory from
  `config.json`. Attention fields its parser does not model give the estimate low confidence and a 20% margin
  (split per rank like the rest of it), so such a model may reserve more memory than it uses, or, if the
  architecture needs much more KV than its base, still too little. DeepSeek-V4's KV pool is modelled after the
  vLLM image configured as `images.vllm` (default `vllm/vllm-openai@sha256:22d3b121…`); after changing or
  upgrading that image, re-run `tools/vllm_dsv4_pool_sim.py` in the new image and compare it with
  `tests/test_deepseek_v4_kv.py`. A deployment keeps the estimate stored at its last start (see
  [upgrading an existing install](../getting-started/upgrade.md#upgrading-an-existing-install)).

## Next steps

- [Security hardening](../security/hardening.md)
