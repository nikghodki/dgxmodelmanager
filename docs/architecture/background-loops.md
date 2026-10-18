---
description: "What DGXMM does at startup (bootstrap, reconcile) and the background loops that keep it current."
---

# Startup and background loops

At startup, before the first request is served (`context.py`):

1. **Bootstrap.** If there are no users, create `admin` with a random password and
   `must_change_pw=True`, and log it once.
2. **Recover downloads.** Downloads that were queued or running are marked failed (they do not survive
   a restart).
3. **Inventory.** Scan the HF cache and GGUF roots.
4. **Reconcile** (`deployer/reconcile.py`). Compare the database with the managed containers that
   exist:
   - a missing container marks its deployment `failed`;
   - an exited container is failed with its exit code;
   - a deployment still `starting` or `loading` gets its monitor back;
   - an interrupted `stopping` is finished;
   - a running managed container with no record is recorded as an `unknown` deployment.

   Because DGXMM adopts every container labelled `dgxmm.managed=true`, only one DGXMM instance may run
   per Docker host.

Before these, the database is upgraded (tables, columns, indexes, versioned migrations). DGXMM also
rejoins the decider's network if it exists, reads that network's subnets, and, after reconciling, logs a
warning when a running decider was started by an older build.

Background loops:

| Loop | Interval | Does |
|---|---|---|
| Budget | `budget_interval_s` (2 s) | Refresh memory, push SSE `budget`, run the watchdog and the liveness check. |
| Metrics scraper | `scrape_interval_s` (5 s) | Scrape each ready engine's `/metrics` and record host samples. Roll up to 1-minute rows every 60 s and drop rollups older than 30 days. |
| Inventory | `inventory_interval_s` (300 s) | Rescan model directories. |
| LiteLLM sync | when a deployment enters or leaves `ready`, its model name changes or the LiteLLM settings change, else every 60 s | Add missing registrations and remove stale ones (one entry per ready deployment). After a failure it retries in 5 s, doubling up to 600 s. |
| Container directory | 30 s | Refresh the read-only container id → name/image map used by the process view. |
| vLLM architectures | once | Run a one-off container from the vLLM image to list supported architectures. The result is cached in `data/` per image reference. |
| Deployment monitor | `monitor_poll_s` (3 s), per deployment | Only while starting or loading. |
| Calibration | 15 s after ready | Record measured memory. |
| Decider supervisor | `decider_probe_interval_s` (30 s) | Probe the ready decider, restart it after a crash or hang (see [decision model](decision-model.md#decision-model-decider)). |
| Model kinds | 60 s | Ask the decision model about models no rule recognises (once per snapshot). |
| Decider retention | with the metrics rollups (60 s) | Prune `decisions` (rated rows after `decider_feedback_retention_days`), `diagnoses` and `compat_cache` (7 days). |
| Decider network subnets | 30 s | Re-read the decider network's subnets, whose forwarding headers are ignored. |
| Key rotation requests | continuous | Pick up `rotate-decider-key` requests from the admin CLI. |
| Insight engine | ticks every 5 s | Runs the background jobs of the decision-model features (health, reading labels, traffic profiles, schedule proposals, the schedule runner, routing advice, threshold tuning; see [background job engine](insights-jobs.md#background-job-engine)). |
| Insights retention | with the metrics rollups (60 s) | Reopen ended snoozes, expire proposals, delete closed insights after 30 days. |

## Next steps

- [Data model](data-model.md)
- [Insights and the job engine](insights-jobs.md)
