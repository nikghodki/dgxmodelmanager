---
description: "The deploy path step by step, the start-up gate, the deployment state machine, switch and reconfigure with rollback, and the watchdog."
---

# Deployer and start-up gate

## Deploying a model, step by step

The deploy wizard calls `preview` repeatedly while you edit the form (with a 350 ms debounce). Only the
final **Deploy** call changes anything.

```mermaid
sequenceDiagram
  autonumber
  actor U as Admin (browser)
  participant API as FastAPI
  participant D as Deployer
  participant B as BudgetService
  participant K as Docker
  participant E as Engine container
  participant L as LiteLLM

  U->>API: POST /api/deployments/preview {engine, model_ref, config}
  API->>D: prepare(): validate config, find artifact, estimate, calibrate
  D->>B: refresh() budget
  D-->>API: estimate + fit (+ plans if it does not fit)
  API-->>U: memory check panel
  Note over U: If it does not fit, the user picks a plan<br/>(config patch, other model, or sleep/stop actions)
  U->>API: POST /api/deployments {name, ..., actions}
  API->>D: apply_actions() (sleep / stop chosen deployments)
  API->>D: deploy()
  D->>B: refresh() and check fit again (under the memory lock)
  alt does not fit any more
    D-->>API: 409 insufficient_memory
    API->>D: undo actions, compute fresh plans
    API-->>U: 409 with fit + plans
  else fits
    D->>D: allocate host port, insert row (planned)
    D->>K: run container (labels dgxmm.managed=true, network, mounts)
    D-->>U: 201 deployment (starting)
    loop every monitor_poll_s (3 s)
      D->>K: container status
      D->>E: GET /health, then POST /v1/completions (1 token)
    end
    D->>D: transition to ready (SSE "deployment" event)
    D-)L: state hook triggers sync: POST /model/new
  end
```

Details (`dgxmm/deployer/service.py`):

- **prepare** validates the engine config (unknown keys are rejected, and extra flags must be on the
  allowlist). It checks that the model reference is a Hugging Face repo id or a `.gguf` path under an
  allowed root, and that the artifact is fully downloaded and supported by the engine. Then it estimates
  memory and applies calibration.
- **Fit** is `estimate.total <= free_for_new (+ memory the chosen actions free)` (`planner/fit.py`).
- **Container creation** runs under a global memory lock, so two deploys cannot both claim the same
  free memory. vLLM gets `--gpu-memory-utilization = floor(reservation / MemTotal × 10000) / 10000`, so its
  memory pool never exceeds the reservation (each vLLM estimate carries 16 MiB for the flooring). vLLM containers run with `HF_HUB_OFFLINE=1`: they only use files
  already in the cache.
- **Readiness**: the monitor marks the deployment `loading` once the container runs, and `ready` once
  `/health` returns 200 and a one-token completion succeeds. It fails the deployment if the container
  exits (it is reported as out of memory, and recorded for calibration, only with evidence: Docker's
  `OOMKilled`, or the kernel OOM killer's own line in the log; a plain exit 137 is reported as a SIGKILL,
  e.g. `docker kill` or a stop timeout, and never recorded), if
  `MemAvailable` falls below the safety margin during startup, or after `startup_timeout_s` (900 s).
  On failure the container is removed and its last 200 log lines are kept in the event history.
- **Start-up gate** (`deployer/gate.py`): at most `max_concurrent_startups` (default 1) engines are
  `starting`/`loading` at once. vLLM sizes its KV cache from a memory profile taken while it starts and
  asserts that free memory did not change meanwhile; on unified memory another engine starting, waking
  or stopping changes it ("Error in memory profiling", "No available memory for the cache blocks").
  A deploy/start that finds the slot taken passes the fit check, keeps its reservation and port, and
  waits in state `queued` (FIFO) with a reason (`startup_queue` in the deployment view: position, what it
  waits for). When its turn comes the fit is checked again (memory outside DGXMM may have grown);
  it fails with a clear message if it no longer fits or after `startup_queue_timeout_s` (1800 s).
  Stopping a queued deployment cancels it. A wake-up (which re-allocates the KV cache) waits for the
  slot too: the request waits up to `wake_sync_wait_s` (50 s), then answers 202 (`startup_queue` set) while the
  wake-up continues in the background, up to `wake_queue_timeout_s` (300 s). Time spent queued
  does not count against switch/reconfigure waits. After a DGXMM restart, deployments still starting
  hold their slots and queued ones re-enter the queue. Stops and sleeps are never delayed (freeing
  memory must stay immediate), so a stop during another engine's start-up can still disturb it.
- **LiteLLM**: a state hook wakes the sync loop whenever a deployment enters or leaves `ready`.

## Deployment state machine

States and transitions come from `dgxmm/deployer/states.py`. `unknown` is the state of a running managed
container that DGXMM found without a database record (see [reconcile](background-loops.md)).

```mermaid
stateDiagram-v2
  [*] --> planned: deploy
  [*] --> stopped: adopt settings
  [*] --> unknown: orphan container found at startup
  planned --> starting: container created
  planned --> queued: another engine is starting
  queued --> starting: its turn (fit checked again)
  queued --> failed: queue timeout / no longer fits
  queued --> stopped: cancelled
  planned --> failed
  planned --> stopped
  starting --> loading: container running
  starting --> ready
  starting --> failed
  starting --> stopping
  loading --> ready: health + test completion OK
  loading --> failed: exited, timeout, memory low
  loading --> stopping
  ready --> sleeping: sleep (vLLM, vllm_sleep level1)
  ready --> stopping: stop, a sleep that stops, a failed sleep, health probes failed
  ready --> failed: container exited or vanished
  sleeping --> ready: wake
  sleeping --> stopping
  sleeping --> failed
  stopping --> stopped
  stopping --> failed
  stopped --> starting: start
  stopped --> queued: start while another engine is starting
  stopped --> failed
  failed --> starting: start
  failed --> queued: start while another engine is starting
  failed --> stopped
  failed --> failed
  unknown --> ready
  unknown --> stopping
  unknown --> failed
```

Every transition writes a `deployment_events` row (with a log tail on failures) and publishes an SSE
event. A deployment can only be removed when it is `stopped` or `failed`. On every budget refresh, a
liveness check moves `ready`, `sleeping` and `unknown` deployments whose container exited or vanished
to `failed`.

**Sleep** (vLLM only, and only when started with `enable_sleep_mode`, which is the default) depends on
`vllm_sleep`. With `level1` it calls the engine's `/sleep?level=1`. That frees the KV cache while the model
stays loaded for a fast wake-up. **Wake** first checks that the memory it will take back is free. With `stop`
(the default: on GB10 unified memory a level-1 sleep frees nothing and vLLM refuses it) a sleep is a stop
(`Deployer._stop_locked`, actor `sleep`), exactly like a cluster's (`cluster.sleep_as_stop`): the view has
`sleep_capable: false` and `sleep_label: "sleep (stops the model)"` (`SLEEP_AS_STOP_LABEL` in
`deployer/service.py`; a cluster's is `"sleep (stops all nodes)"`; `null` for a real sleep or none), the container
gets neither `--enable-sleep-mode` nor `VLLM_SERVER_DEV_MODE` (`VllmEngine.container_spec`), and a **wake** of a
deployment whose last stop was its sleep (`cluster.stopped_by_sleep`) is a `restart`. A level-1 sleep whose engine call
fails for any reason never leaves the deployment `ready`: it is stopped (`ready → stopping → stopped`, actor
`sleep failed`) and restarted through `restart`, keeping `last_error` *sleep failed: …*; if the restart fails,
it ends `failed`.

**Health probe** (`deployer/health.py`, its own background loop, every `engine_health_interval_s`): each `ready`
deployment that is not a system deployment, has no operation in progress (its lock is free) and whose nodes are
configured and fresh gets `engine.health(completion=False)`: the start-up check without its one-token test
completion, i.e. the engine's `GET /health` (5 s timeout; a cluster: on its head), all concurrently and outside the
lock. A completion every interval would count as request traffic in the metrics (`decider/common.py`
`BUSY_KEYS`), so schedules, plans and proposals would never see the deployment idle. A result is discarded when the deployment had a
new event meanwhile, and any new event restarts the count. `engine_health_failures` failures in a row:
`Deployer.fail_unhealthy` under the lock: `ready → stopping`, the container removed (a cluster: `cluster.fail`,
every rank), `→ failed` with `last_error` and the log tail; the state hooks deregister it from LiteLLM.

## Switch and reconfigure with rollback

**Switch** (`Deployer.switch`) replaces deployment A with a new deployment B:

1. Estimate B. The memory A frees counts as available. DGXMM sleeps A when that frees enough, and stops
   it otherwise, unless you chose what happens to A yourself. Any other plan actions are added.
2. If it still does not fit, return `409` with fresh plans and change nothing.
3. Apply the actions. Move A's LiteLLM model name (its alias) to B. Deploy B and wait for `ready` (up to
   `startup_timeout_s` + 30 s).
4. **Rollback** if B does not become ready: stop and remove B, give the alias back to A, and undo every
   action in reverse order (wake what was slept, restart what was stopped). If any undo step fails,
   the error is `rollback_incomplete` and lists what could not be restored.

**Reconfigure** (`Deployer.reconfigure`) changes the configuration (or model) of one deployment in
place. It stops the deployment and restarts it with the new config. If that does not become ready, the
old config is written back and the deployment is restarted with it, then any plan actions are undone.

## Watchdog

`dgxmm/watchdog.py` runs after every budget refresh and looks at `MemAvailable`:

| Level | Condition |
|---|---|
| `ok` | `MemAvailable` ≥ 1.5 × safety margin |
| `warning` | `MemAvailable` < 1.5 × safety margin (12 GiB by default) |
| `critical` | `MemAvailable` < safety margin (8 GiB by default) |

Level changes are logged and sent to the UI as an alert banner. If `auto_protect: true` and the level is
`critical`, the watchdog stops the lowest-priority `ready`, `sleeping` or `unknown` deployment (the
largest reservation goes first within the same priority). It acts at most once every 60 seconds and
audits the stop as `watchdog_stop`. With the default `auto_protect: false` it only warns.

System deployments are never stopped, except the decision model: it is advisory and fails open, so it is
sacrificed before any user deployment. It is the first victim of `auto_protect`, and, whatever `auto_protect`
says, the watchdog stops it when its measured memory exceeds its reservation by more than
`decider_overuse_margin_gib` in two samples in a row (its GPU memory is not capped by its cgroup). The stop is a
deployment event and a `watchdog_stop` audit entry; `/api/decider/status` shows it as the supervisor's
`last_action: watchdog_stopped` with the reason, and the supervisor restarts it after its backoff, never while
memory is under pressure.

## Next steps

- [Memory budget and fit planner](memory.md)
- [Startup and background loops](background-loops.md)
