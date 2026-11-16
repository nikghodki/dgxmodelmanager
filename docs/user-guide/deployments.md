---
description: "A deployment's page: lifecycle actions, sleep and wake, switch and reconfigure, logs, metrics, advice, benchmarks and API access."
---

# Deployments

![Deployment page with a KV-cache pressure insight, the Schedules card and the API access card](../images/deployment-detail-light.png#only-light) ![Deployment page with a KV-cache pressure insight, the Schedules card and the API access card](../images/deployment-detail-dark.png#only-dark)

Header actions depend on the state (admins only):

| Action | When | What happens |
|---|---|---|
| **Sleep** | ready, vLLM with sleep mode | On DGX Spark (the default `vllm_sleep: stop`): **Sleep (stops the model)** stops the container and frees all of its memory, after a confirmation. With `vllm_sleep: level1`: frees the KV cache, weights stay loaded for a fast wake-up. LiteLLM deregisters it while asleep. See [sleep and wake](#sleep-and-wake). |
| **Wake** | sleeping, or stopped by its Sleep | Takes the memory back (only if it is free). Waking re-allocates the KV cache, so it waits for the [start-up queue](deploying.md#the-start-up-queue) like a start: the page then says *Waiting to wake up*, and the wake-up continues in the background (the API answers `202` after 50 s). A deployment its Sleep stopped is started again (the page offers **Start**). |
| **Start** | stopped or failed | Starts again with the stored configuration, after a fresh memory check. |
| **Reconfigure** | ready, sleeping, stopped, failed | Change the configuration with a live memory check. Applying it restarts the deployment. If the new configuration does not come up, the old one is restored. |
| **Logs** | any | The last lines of the engine container log. |
| **Switch model…** (`⋯` menu) | ready or sleeping | Replace this deployment with another model. DGXMM sleeps or stops this one as needed, moves the LiteLLM model name, and rolls back if the new model does not become ready. |
| **Stop** (`⋯`) | running and queued states | Stops and removes the engine container (a queued deployment just leaves the queue). The deployment and its settings are kept. An [on-demand](on-demand.md) deployment is started again by its next request. |
| **Remove** (`⋯`) | stopped or failed | Deletes the deployment record. |

Stop and Remove ask for confirmation. **Start** and **Wake** are refused when another running
deployment has taken this one's model name in the meantime; the notice links to that deployment and
offers to set a different model name.

Tabs:

- **Overview**: for a failed deployment, a **Diagnosis** card first (see
  [failure diagnosis](decision-model.md#failure-diagnosis)); then the **Insights** card with the open
  [insights](insights.md) about this deployment (when there are any), the **Schedules** card (see
  [schedules](schedules.md)), the **Mode** card (*On demand* or *Always on* and the idle minutes; admins
  change them with **Save mode**, see [on-demand models](on-demand.md#choosing-and-changing-the-mode)),
  the **API access** card, memory (reserved, measured, estimate parts and
  confidence), and the full configuration. When multiple nodes are configured, **Runs on** shows
  the placement: the node label (e.g. *dgx-1*) for single deployments, or *dgx-1 + dgx-2 (cluster)* for
  clusters. API access shows how clients call this deployment: the single LiteLLM
  **Endpoint** (`…/v1`) and the **Model** name, each with a copy button, the LiteLLM status, and
  ready-to-copy `curl` and Python (OpenAI SDK) snippets that read the key from `LITELLM_API_KEY`. A
  warning appears when the integration is off, the model is not registered yet, or the last sync
  failed. **Direct engine endpoints (advanced)**, collapsed by default, lists the served name and the
  engine URLs on the Docker network and on the host. Those bypass LiteLLM, so no keys, budgets or rate
  limits apply.

- **LiteLLM**: the **Model name** (admins can change it; empty means the served name), the
  **Registration** status with a **Re-sync** button, and, for admins, **Keys with access**: the virtual
  keys that can call this model (keys limited to it, and keys for all models), with **Create key for
  this deployment** (the model is preselected). Renaming re-registers the deployment under the new name
  and removes the old one. If keys are limited to the old name, **Also update N key(s)** (on by default)
  rewrites them to the new name, and the result says exactly which keys were updated. Viewers see this
  tab read-only, without keys.
- **Performance**: throughput, time to first token, KV-cache usage, queue, inter-token latency and
  speculative-decoding acceptance. View it live or over 1 h, 24 h, 7 d or 30 d.
- **Advice**: suggestions from the measured metrics and benchmarks. Examples: an fp8 KV cache, an
  NVFP4 variant that is already on disk, speculative decoding with a local draft model, more concurrent
  sequences, a smaller KV pool, chunked prefill, and llama.cpp offload, flash-attention and KV-cache
  settings. **Apply** opens Reconfigure with the change filled in. With 30 minutes or more of traffic
  metrics, a **Workload** label orders the advice (see [workload-aware advice](decision-model.md#workload-aware-advice)).
- **Benchmarks**: **Run benchmark** load-tests the ready deployment at concurrency 1, 4, 8 and 16
  (1024-token prompts, 256-token outputs, at most 60 s and 32 requests per level). Results (total and
  per-request tok/s, TTFT p50/p95, ITL p50) appear as each level finishes and are kept per
  configuration, so you can compare settings. Only one benchmark runs at a time. Each level gets a
  [reading label](decision-model.md#reading-labels) badge (*Plausible*, *Contended*, *Cold cache*, *Broken*); only
  plausible and unlabelled levels are charted and used for advice.
- **Events**: every state change with its message. Failures include the container's last log lines
  (secrets in them are masked). An engine that exits with code 137 is labelled *out of memory* only
  when Docker reports an OOM kill or the log has the kernel OOM killer's line; otherwise it says *killed
  by SIGKILL* (for example `docker kill`, or a stop that timed out), and the memory estimate is not
  raised.

## Cluster deployments

![Cluster deployment deepseek-v4-flash on dgx-1 + dgx-2: the Sleep (stops all nodes) action, the Mode card and the Ranks card with the head on dgx-1 and the worker on dgx-2, both running](../images/deployment-cluster-light.png#only-light) ![Cluster deployment deepseek-v4-flash on dgx-1 + dgx-2: the Sleep (stops all nodes) action, the Mode card and the Ranks card with the head on dgx-1 and the worker on dgx-2, both running](../images/deployment-cluster-dark.png#only-dark)

A deployment that runs on several nodes ([multi-node deployments](deploying.md#multi-node-deployments-and-cluster-limits))
says *runs on dgx-1 + dgx-2 (cluster)* under its name. Its **Ranks** card lists one container per node: the rank,
the node, its role (the **head** serves the API, **workers** run headless) and the rank container's state. Its
Sleep reads **Sleep (stops all nodes)**: a cluster cannot sleep across nodes, so it stops every rank and frees
the memory on each node, after a confirmation.

The Deployments list has a **Runs on** column whenever more than one node is configured:

![Deployments list with the Runs on column: deployments on dgx-1, on dgx-2 and one on dgx-1 + dgx-2 (cluster), and an on-demand model asleep](../images/deployments-light.png#only-light) ![Deployments list with the Runs on column: deployments on dgx-1, on dgx-2 and one on dgx-1 + dgx-2 (cluster), and an on-demand model asleep](../images/deployments-dark.png#only-dark)

## Sleep and wake

DGX Spark (GB10) has unified memory: the GPU and the CPU share one pool of RAM. vLLM's level-1 sleep frees the
KV cache and copies the weights to CPU memory, which on GB10 is the same memory, so it frees nothing; vLLM then
refuses the sleep itself (`Memory usage increased after sleeping`). So by default (`vllm_sleep: stop` in
[config.yaml](../configuration/index.md#engines)) a single-node vLLM deployment's **Sleep** is a stop, exactly
like a cluster's *Sleep (stops all nodes)*:

- the action reads **Sleep (stops the model)**, the deployment shows `sleep_capable: false` and
  `sleep_label: "sleep (stops the model)"`, and it asks for confirmation like Stop;
- the container is removed and its whole reservation released; the deployment becomes **Stopped** (event
  *sleep (stops the model): stopping the container*, actor `sleep`);
- **Wake** (or **Start**) starts it again through the memory check and the start-up queue: a full model load,
  not a fast wake-up. A deployment you stopped yourself is not started by a wake;
- engines are started without `--enable-sleep-mode` and without vLLM's development routes, whatever the
  deployment's *sleep mode* setting says. That setting is kept, so existing deployments keep working, and it
  takes effect again with `vllm_sleep: level1`.

On a host with separate GPU memory, `vllm_sleep: level1` restores vLLM's own sleep (it takes effect at each
deployment's next start; one asleep at the switch can still be woken). If a sleep fails there for any reason,
DGXMM never leaves the deployment *Ready* on an engine that may be asleep or broken: it stops it and starts it
again, and the deployment's last error says *sleep failed: …*; if that start fails too, the deployment ends
**Failed**.

## Health checks

While a deployment is ready, DGXMM probes its engine every 30 s (`engine_health_interval_s`): its `GET /health`
(a cluster: on its head). The probe sends no test completion, so it never counts as traffic. After 3 failed probes in a row (`engine_health_failures`) it is
**Failed**: its container (every rank of a cluster) is removed, LiteLLM drops it, and the last error says *engine
not answering its health check*; the event keeps the engine's last log lines. Start it again, or, for an
[on-demand](on-demand.md) deployment, its next request starts it. A single slow answer never fails a deployment,
and deployments that are starting, waking, asleep, stopping or stopped, the decision model (it has its own
supervisor) and deployments on a node that is not answering are not probed.

## Next steps

- [On-demand models](on-demand.md)
- [Insights](insights.md)
- [Schedules](schedules.md)
- [Calling your models](calling-models.md)
