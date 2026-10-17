# Architecture

This page explains how DGXMM works inside. You do not need it to use DGXMM. Read it to understand
DGXMM's decisions or to change the code. Each section names the source files it describes.

- [Components](#components)
- [Backend module map](#backend-module-map)
- [Deploying a model, step by step](#deploying-a-model-step-by-step)
- [Memory model](#memory-model)
- [Deployment state machine](#deployment-state-machine)
- [Switch and reconfigure with rollback](#switch-and-reconfigure-with-rollback)
- [Watchdog](#watchdog)
- [Decision model (decider)](#decision-model-decider)
- [LiteLLM integration](#litellm-integration)
- [Downloads](#downloads)
- [Deleting weights](#deleting-weights)
- [Process view](#process-view)
- [Startup and background loops](#startup-and-background-loops)
- [Data model](#data-model)
- [HTTP API](#http-api)
- [Frontend](#frontend)
- [Security model](#security-model)

## Components

DGXMM is a single container (`dgxmm`). It holds a FastAPI backend, the built React single-page app
and a SQLite database. Everything else it uses is an existing service or a container it starts,
including the optional decision model (`dgxmm-decider`) on its own internal network.

```mermaid
flowchart LR
  browser["Browser<br/>React SPA"]
  subgraph dgxmm["dgxmm container"]
    api["FastAPI<br/>REST + SSE"]
    db[("SQLite<br/>data/dgxmm.sqlite3")]
    llmfit["llmfit CLI"]
    sock["admin unix socket<br/>data/run/admin.sock"]
  end
  cli["admin CLI<br/>docker exec dgxmm python -m dgxmm.admin"]
  subgraph host["Host"]
    docker["Docker Engine<br/>/var/run/docker.sock"]
    probes["/proc, cgroup v2,<br/>/etc/passwd, NVML"]
    disk[("HF cache +<br/>GGUF directories")]
    helper["delete helper<br/>(short-lived, dgxmm.helper=true,<br/>no network)"]
  end
  subgraph net["Docker network (default: dgxmm)"]
    vllm["vLLM engine<br/>dgxmm-NAME"]
    llama["llama.cpp engine<br/>dgxmm-NAME"]
    litellm["LiteLLM proxy<br/>(external or bundled)"]
  end
  subgraph dnet["internal network dgxmm-decider (no egress, no published port)"]
    decider["decision model<br/>dgxmm-decider (optional)"]
  end
  clients["Clients<br/>(OpenAI SDK, curl)"]
  hub["Hugging Face Hub"]

  browser -- "HTTP /dgxmm/api, SSE /dgxmm/api/stream" --> api
  api --> db
  api --> llmfit
  api -- "create / stop / remove<br/>(label dgxmm.managed=true)" --> docker
  api -- "list containers (read-only)" --> docker
  docker --> vllm
  docker --> llama
  docker -- "run once per delete" --> helper
  helper -- "rm (one bind-mounted directory)" --> disk
  api -- "read memory, cgroups, processes, GPU" --> probes
  api -- "scan" --> disk
  vllm -- "mount" --> disk
  llama -- "mount (read-only)" --> disk
  api -- "health, /metrics, sleep/wake" --> vllm
  api -- "health, /metrics" --> llama
  api -- "/model/new, /model/delete,<br/>/key/*, /health" --> litellm
  clients -- "one /v1 endpoint,<br/>model = model name" --> litellm
  litellm -- "OpenAI API" --> vllm
  litellm -- "OpenAI API" --> llama
  api -- "search, download" --> hub
  docker --> decider
  api -- "key file via docker exec,<br/>/health, /v1/systemone (bearer key)" --> decider
  decider -- "mount (read-only, its snapshot only)" --> disk
  cli --> sock
  sock --> api
```

- **Browser.** The UI is served from `/dgxmm/` by the same process. A server-sent event (SSE) stream
  pushes memory, deployment, download, metric and alert updates.
- **FastAPI.** Uvicorn listens on port 8090 in the container (`dgxmm/__main__.py`). All routes live
  under the base path `/dgxmm` (`dgxmm/app.py`), and `/` redirects there.
- **SQLite.** One file, `data/dgxmm.sqlite3`, in WAL mode. On start, new tables are created
  (`Base.metadata.create_all`), missing nullable columns and non-unique indexes are added, and then the
  versioned migrations in `dgxmm/migrations.py` run once each, in a transaction, recorded in
  `schema_version` (see [operations](operations.md#upgrading-an-existing-install)). It also holds the
  runtime settings (`app_settings`) that the UI can change.
- **Docker API.** DGXMM starts engine containers through the mounted Docker socket. Engines join the
  configured Docker network, where DGXMM reaches them as `http://dgxmm-<name>:<port>`. Their ports are
  also published on the host at `publish_bind` (default `127.0.0.1`).
- **Host probes.** `/proc`, `/sys/fs/cgroup` and `/etc/passwd` are mounted read-only. DGXMM reads
  `/proc/meminfo`, each engine container's `memory.stat` (cgroup v2 path `system.slice/docker-<id>.scope`)
  and per-process data (see [process view](#process-view)). NVML (via the `nvidia` runtime) adds GPU
  utilisation, temperature, power and per-process GPU memory.
- **Delete helper.** To delete model files that engine containers left owned by root, DGXMM starts a
  short-lived, locked-down container that runs `rm` on one bind-mounted directory (see
  [deleting weights](#deleting-weights)).
- **LiteLLM** (optional; an existing proxy or the bundled one). Each ready deployment is registered once
  through LiteLLM's model management API, so clients use a single OpenAI-compatible endpoint. DGXMM
  also proxies LiteLLM's virtual-key API for the UI (see [LiteLLM integration](#litellm-integration)).
- **Hugging Face Hub.** Catalog search, model details and downloads (with the optional HF token).
- **llmfit.** A Python CLI bundled in the image. It adds fit levels, scores and expected tok/s to
  catalog results. It never decides whether a deploy may proceed.
- **Decision model** (optional). A small classifier served from a locally built, hardened image on an
  internal network. DGXMM asks it typed questions with a fixed set of answers and uses its answer only
  above a confidence threshold, after its own rules (see [decision model](#decision-model-decider)).
- **Admin unix socket.** A second listener on `data/run/admin.sock` (mode 0600 in a 0700 directory) for
  the admin CLI (`docker exec dgxmm python -m dgxmm.admin decider …`). Short-lived CLI tokens are only
  accepted there, never on the TCP port.

## Backend module map

| Module | Responsibility |
|---|---|
| `__main__.py` | Entry point: loads settings and runs Uvicorn on `0.0.0.0:8090` with proxy headers enabled. |
| `admin.py` | Admin command line (`python -m dgxmm.admin`): `reset-password`, `list-users`, `reset-calibration`, `decider …`, `rotate-decider-key`. |
| `admin_listener.py` | The admin unix socket next to the TCP listener (`DualServer`), without proxy-header handling. |
| `proxy_headers.py` | Applies `X-Forwarded-For` / `-Proto` only from `trusted_proxies`, and never from the decider's network. |
| `lifecycle.py` | Shutdown signal shared by the SSE streams and background loops, so `docker stop` ends cleanly. |
| `migrations.py` | Versioned, run-once schema migrations (`schema_version`) and automatic non-unique indexes. |
| `app.py` | FastAPI factory: lifespan (startup tasks, background loops), routers, root redirect, SPA fallback. |
| `config.py` | `Settings` model: optional YAML file, then `DGXMM_<KEY>` environment overrides. |
| `context.py` | Wires every service into an `AppContext` and defines the startup and background tasks. |
| `db.py` | SQLAlchemy models and the SQLite engine (WAL, foreign keys, busy timeout), plus `add_missing_columns`. |
| `events.py` | In-process publish/subscribe used by the SSE stream (slow clients drop their oldest events). |
| `audit.py` | Writes audit log rows. |
| `auth/` | `bootstrap` (first admin), `passwords` (argon2id), `sessions`, `ratelimit`, `deps` (auth, CSRF, roles). |
| `settings_store.py` | Runtime settings in the `app_settings` table, with a cache and typed accessors (LiteLLM URLs, master key and its URL binding, enabled). |
| `api/` | HTTP routers: auth, users, host (incl. processes), inventory (incl. delete and clean-up), engines, deployments, integrations (LiteLLM), stream, audit, containers, catalog, downloads, planner, metrics, decider (status, settings, enable/disable, image, decisions, key rotation) and decider features (diagnosis, workload, suggest-config). `ratelimit.py` holds the per-user API rate limits. |
| `deployer/service.py` | Deployment lifecycle: prepare, preview, deploy, monitor, stop, restart, sleep/wake, switch, reconfigure. |
| `deployer/states.py` | Deployment states and allowed transitions. |
| `deployer/reconcile.py` | After a restart: brings the database in line with the containers that actually exist. |
| `deployer/adopt.py` | Copies the settings of an unmanaged vLLM container into a new, stopped deployment. |
| `deployer/gate.py` | The start-up gate: at most `max_concurrent_startups` engines starting or loading, the others `queued` (FIFO). |
| `deployer/logtail.py` | The log tail stored with a failure: line cap, error lines from the cut part, redaction, OOM evidence. |
| `docker_ctl.py` | The only module that calls Docker. Every write requires the `dgxmm.managed=true` label, except the delete helpers it creates itself (`dgxmm.helper=true`). Also holds the read-only, cached container directory (id → name, image, deployment). |
| `engines/` | Engine plug-ins (`vllm`, `llamacpp`, `decision`): config schema, container spec, health check, metrics parsing, sleep. `decision.py` holds the supported decision models (pinned revisions, fixed reservations) and the hardened decider container spec. `argguard` holds the extra-flag allowlist and `prom` parses Prometheus metrics. |
| `memory/estimate.py`, `kv.py` | Memory estimates per engine (weights, KV cache, state, activations, runtime). |
| `memory/budget.py` | The host memory budget and `free_for_new`. |
| `memory/hostmem.py` | Reads `/proc/meminfo`, container cgroups, processes (with container, user and command) and NVML. |
| `memory/redact.py` | Masks secrets in process command lines before they leave the probe. |
| `memory/calibration.py` | Corrects estimates with measured usage and out-of-memory kills. |
| `planner/fit.py`, `plans.py` | Fit check and "ways to make it fit" plans. |
| `planner/colocate.py` | "What runs together" combinations for the Planner page. |
| `inventory/` | Scans the HF cache and GGUF roots; parses `config.json` and GGUF headers into architecture specs. `weights.py` plans and performs weight deletion and incomplete clean-up. |
| `catalog/` | Hugging Face search and details (`hub`), llmfit wrapper, vLLM-supported architectures (`support`), fit badges. |
| `downloads.py` | Background Hugging Face downloads with a disk-space guard. |
| `metrics/` | Engine `/metrics` scraper and storage (live ring buffer plus 1-minute rollups). |
| `advisor/` | Tuning rules (`rules`) and the built-in load test (`benchmark`). |
| `watchdog.py` | Memory alerts and optional automatic protection. |
| `decider/` | The decision model: `service.py` (the `Decider`: settings, thresholds, decision log), `client.py` (`SystemOneClient`: back-pressure, circuit breaker, answer validation), `supervisor.py` (probes and restarts), `image.py` (in-app image build), `snapshot.py` (verified snapshot download), `secret.py` / `rotation.py` (bearer key), `upgrade.py` (detects a decider from an older build), `retention.py`, `cli.py` (`admin decider …`), and one module per feature: `diagnosis.py`, `config_suggest.py`, `plans.py`, `kind.py`, `workload.py`. |
| `litellm.py` | LiteLLM client (URL and key resolved per call, errors scrubbed of keys, cached status) and the sync that keeps LiteLLM's model list in line with ready deployments. |

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
  free memory. vLLM gets `--gpu-memory-utilization = ceil(reservation / MemTotal × 100) / 100`, so its
  memory pool matches the reservation. vLLM containers run with `HF_HUB_OFFLINE=1`: they only use files
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

## Memory model

### Budget

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
  command line, RSS and GPU memory (see [process view](#process-view)).
- **GB10 unified memory:** CUDA allocations are not charged to the container's cgroup, so a running
  model's memory shows up in `used` but only partly in *measured*. For `ready` deployments the gap
  between reservation and measurement (`ready_gap`) is credited back, so it does not appear as external
  usage. It is never credited for deployments that are still starting, which keeps the check
  conservative while memory is still growing.

The budget is refreshed every `budget_interval_s` (2 s) and pushed to browsers over SSE.

### Estimates

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

### Calibration

Fifteen seconds after a deployment becomes ready, DGXMM records its measured memory next to the formula
estimate (`memory/calibration.py`). Later estimates for the same model are corrected:

- **Same model and exact same configuration:** `max(measured × 1.05, formula)`. Confidence: `measured`.
- **Same model, other configuration:** the formula is scaled by the mean measured/estimated ratio of
  the last 5 measurements, clamped to 1.0–1.5. Confidence: `calibrated`.
- **Out-of-memory kill** (Docker's `OOMKilled`, or the kernel OOM killer's line in the log; not a bare
  exit 137): the next estimate for that configuration gets +10 %. It never drops
  back below 1.1 × the size that was killed.

Calibration only raises estimates. It never lowers one below the formula.

### Plans when it does not fit

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
undone. Before the plans are returned, the [plan recommendation](#decision-model-decider) marks unsafe
plans and may recommend one.

## Deployment state machine

States and transitions come from `dgxmm/deployer/states.py`. `unknown` is the state of a running managed
container that DGXMM found without a database record (see [reconcile](#startup-and-background-loops)).

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
  ready --> sleeping: sleep (vLLM)
  ready --> stopping
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

**Sleep** (vLLM only, and only when started with `enable_sleep_mode`, which is the default) calls the
engine's `/sleep?level=1`. That frees the KV cache while the model stays loaded for a fast wake-up.
**Wake** first checks that the memory it will take back is free.

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

## Decision model (decider)

The decision model is optional and advisory. Its code is in `dgxmm/decider/` and
`dgxmm/engines/decision.py`, its image in `deploy/decider/`. The threat model, with every control and
what was measured, is in [security.md](security.md#decision-model-decider); accuracy and capacity are in
[decider-evaluation.md](decider-evaluation.md).

**Deployment.** The decider is a deployment of the `decision` engine named `decider`, created by
`POST /api/decider/enable` through the normal deploy path (fit check, plans, start-up gate) with a fixed
model (`decider.model`: d3-mini or d3-nano) and a fixed, measured reservation (13 or 9 GiB, never
calibrated). Its row has `system = true`: plans never stop or sleep it, LiteLLM sync skips it, the UI
shows a **System** badge, and the watchdog stops it before any user deployment (see
[watchdog](#watchdog)). Disabling stops it; `{"remove": true}` also deletes the row.

**Container** (`DecisionEngine.container_spec`, `DockerController.run`). Built locally as
`images.decision` (`dgxmm-decider:3`) from a digest-pinned vLLM base with hash-pinned wheels, by
`dgxmm/decider/image.py` (the **Build image** button, progress on the admin-only SSE topic `decider`)
or `admin decider build`. It runs as `decider_user` (10001), with a read-only root filesystem, every
capability dropped, `no-new-privileges`, private IPC, `decider_pids_limit`, a cgroup memory limit
(`decider_memory_limit_gib`), `/tmp` as a `decider_tmp_gib` tmpfs and `restart: no`. The only mounts
are the selected model's repo directory of the HF cache, read-only, plus one read-only bind per
shared-store blob that its snapshot links to. At start the launcher copies the pinned files into a
private directory on its tmpfs, checks each copy against the image's hash list and refuses to start
(exit 3) on a mismatch; exit 2 means the snapshot is missing, exit 4 that no key arrived.

**Network.** The decider joins only `decider_network` (`dgxmm-decider`), an internal Docker network
that DGXMM creates on the first start and then attaches its own container to (again at every DGXMM
start, so an upgraded container rejoins). It publishes no port. DGXMM reaches it as
`http://dgxmm-decider:<port>`. The network's subnets are read at start and every 30 s, and forwarding
headers from them are always ignored (`proxy_headers.py`).

**Key.** A bearer key is generated once in `data/secrets/decider_api_key` (mode 600). Right after the
container starts, DGXMM writes it through `docker exec` stdin into a 0400 file on the container's
tmpfs (`/run/dgxmm-decider/api_key`); the launcher reads and deletes it and hands it to the server, so
it is never in the environment, the command line or `docker inspect`. If the write fails the
container is removed at once. Rotation (`dgxmm/decider/rotation.py`, `POST /api/decider/rotate-key`
or the CLI through a request file the app watches) stops the decider, writes a new key with an atomic
rename and starts it again.

**Asking a question** (`Decider.decide`, `SystemOneClient`). A feature sends a fixed question, a set of
options (at most 128) and a *state*: facts DGXMM assembled, with untrusted text (log lines, chat
template, model name, `config.json`) quoted in labelled blocks, length-capped and redacted. The client:

1. answers at once with *no answer* when the feature is switched off, the decider is not `ready`, or the
   circuit breaker is open (3 failures in a row open it for 60 s; a good supervisor probe closes it);
2. queues behind at most `decider_max_in_flight` (2) questions at the server, interactive ones (plan,
   parser) ahead of background ones (diagnosis, kind, workload), and gives up (*busy*, answered by
   rules, not counted as a failure) after `decider_queue_timeout_s` (1 s; background: half the deadline)
   or when the remaining deadline is shorter than the measured service time;
3. posts to `/v1/systemone` with the bearer key, reads at most 64 KiB of the reply, and validates it:
   finite probabilities in [0, 1] for offered options only, summing to about 1, and the choice is the
   most probable option. Anything else is a `bad_reply` failure;
4. returns the choice with its confidence. The feature uses it only at or above its threshold
   (`decider.feature.<f>.threshold`); below it the decision is logged as abstained.

Each question, answered or not, is a `decisions` row with the input's SHA-256 (never the state text),
probabilities, latency and model revision; answers are cached in memory by feature, input hash and
model revision (256 entries).

**Rules first.** Each feature runs its own deterministic rules before it asks anything, in worker
threads and with bounded regexes over capped input:

| Feature | Rules | Model asked | Effect |
|---|---|---|---|
| Diagnosis (`diagnosis.py`) | 14 causes from log patterns, Docker's OOM flag and the decider's own exit codes, on a failed deployment's stored log tail | when no rule matched (10 s, background) | a `diagnoses` row (cause, fix hint, validated patch, redacted excerpt), SSE `diagnosis` |
| Parser (`config_suggest.py`) | chat-template evidence (strong) and model-family names (weak) | when no rule matched (4 s per question) | wizard suggestion; auto-fill only for strong rules or model ≥ threshold |
| Plan (`plans.py`) | safety filter: `unsafe_reason` per plan; exactly one safe plan is recommended by rules | two or more safe plans (2 s, cached 2 min) | `recommended`, `recommendation` on the plan |
| Kind (`kind.py`) | scanner rules (`inventory/kinds.py`) | models no rule recognises, once per snapshot (background loop) | `artifact_kinds` cache, `kind_source` / `kind_confidence` |
| Workload (`workload.py`) | idle and saturated-batch patterns from the last hour of rollups (≥ 30 minutes needed) | otherwise (2 s, cached 15 min) | advice `relevance` and `workload_fit` |

Rules decisions are logged too (source `rules`). Nothing the decider answers runs anything: diagnoses
and suggestions only prefill dialogs, plans are still chosen and applied by the admin.

**Supervisor** (`supervisor.py`). Every `decider_probe_interval_s` (30 s) while enabled it probes a ready
decider (`/health` plus one question). `decider_hang_probes` (3) failures in a row while the container
runs, or a failed deployment, lead to a restart through the normal deployer path, with backoff
(`decider_restart_backoff_s` doubling to `decider_restart_backoff_max_s`, reset after `decider_stable_s`),
at most `decider_max_restarts_per_hour` attempts (failed ones included), never under memory pressure, never during a key rotation and
never after an admin stopped it. A watchdog stop counts as a failure. After a model switch it rolls back
to the previous model if the new one fails to start. Status, breaker, load and supervisor state are in
`GET /api/decider/status`.

**Retention** (`retention.py`). Once a minute, with the metrics rollups, `decisions` rows older than
`decider_retention_days` (30) or beyond the newest `decider_max_rows` (50 000) are deleted, and
`diagnoses` keep the latest 20 per deployment (they are also deleted with their deployment).

## LiteLLM integration

**Settings store** (`settings_store.py`). The LiteLLM connection is resolved on every call from the
`app_settings` table, falling back to `config.yaml`: internal URL (`litellm.url` > `litellm_url`), public
URL (`litellm.public_url` > `litellm_public_url`), master key (`litellm.master_key` > the
`litellm_key_file` secret) and `litellm.enabled` (default: on when a key is available). Values are cached
in memory and the cache is updated on write, so a change made on the LiteLLM page applies to the next
call without a restart.

**Master key bound to its URL.** A stored key is saved together with the internal URL in effect at that
moment (`litellm.master_key_url`) and is used only while the internal URL is still that URL. The
secret-file key is used only with the configured default `litellm_url`. Pointing the internal URL at
another host therefore never sends an existing write-only key there; the API reports
`master_key_hint` instead and the UI asks for the key. The key is never returned by the API, never
logged and never audited, and error texts from LiteLLM are scrubbed of the key, `sk-…` tokens and key
hashes before they are logged or returned.

**One registration per deployment** (`LiteLLMSync`). The desired state is one entry per `ready`
deployment, keyed by deployment id and model name (`alias` or served name), with
`model_info.dgxmm_id` marking it as DGXMM's. Each sync lists the models carrying a `dgxmm_id`, deletes
every one that is not desired (stopped deployments, an old model name after a rename, duplicates, and
the second, served-name entry that older versions registered next to an alias), and adds what is
missing. That makes the migration from the old two-name scheme automatic on the first sync, and a
rename re-registers exactly once. Syncs are serialised by a lock. If two ready deployments end up with
one name anyway (the deployer prevents it when names are claimed), the oldest keeps it and the others
get `litellm_status: error` with a `model_name_conflict` reason. The last sync's result and conflicts
feed `litellm_error` in deployment views and the per-deployment `GET /deployments/{id}/litellm`, so
those read paths never call LiteLLM.

**Model names are unique.** The deployer checks names under its memory lock when a deployment is
deployed, started, woken or renamed: a name is taken if it is another deployment's alias (in any state)
or the model name of another deployment that is planned, starting, loading, ready or unknown.

**Virtual keys** are proxied to LiteLLM's `/key/*` API with the master key. Before a key operation DGXMM
checks LiteLLM's `/health/readiness`; without a connected database it answers `409 litellm_no_db`.
Keys are identified by their hash; the full key is returned only in the create response. Keys created
by DGXMM carry `metadata.created_by = "dgxmm"` and the user name.

**Status** (`GET /api/integrations/litellm`, or on its own `GET /api/integrations/litellm/status`) uses
only unauthenticated calls (`/health/liveliness`, `/health/readiness`, and `/openapi.json` for the
version), cached for 15 seconds. The probe has a 4 s deadline; a slower version fetch finishes in the
background and fills in the cached status. `GET /api/integrations/litellm?probe=false` returns the
settings at once with the cached status (or `null`); the LiteLLM page uses it so the probe never holds
the page up. **Test connection** is always fresh and also checks the key with an authenticated
`/model/info`. `GET /api/integrations/litellm/models` caches LiteLLM's model list for 15 seconds (DGXMM's
own registrations invalidate it).

**Leaving a LiteLLM.** A `PUT /api/integrations/litellm` that disables the integration or changes the
internal URL first deletes every model with `model_info.dgxmm_id` from the LiteLLM in effect before the
change, using the key bound to that URL, under the sync lock (so the sync loop cannot re-register in
between), then saves. It is best effort (15 s bound): the response carries `deregistered`,
`deregister_error` and a `warning` naming the old URL when some may remain; the audit entry records the
count and error. The settings are saved either way.


## Downloads

`downloads.py` fetches the selected files one at a time. Each file is fetched by `hf_hub_download` in a
child process (`python -I -B`, progress bars off; the HF token is passed in its environment, never on its
command line). Cancel kills the child's process group, which aborts the HTTP stream at once, then removes
the repo's `blobs/*.incomplete` partial files (only one download per repo runs at a time) and rescans the
inventory. When the download starts it writes `.dgxmm-download.json` in the repo dir with the revision and
the file list, and removes it only once every file is on disk. The inventory reports a snapshot of that
revision that lacks a listed file as incomplete ("download not finished"), so a cancelled or failed
download is never offered as a complete model; incomplete cleanup offers it instead. A marker for another
revision, or an unreadable one, is ignored.

A supported decision model (d3-mini, d3-nano) never takes that path, whether it is started from the Decision
model page or `POST /api/downloads`: `DownloadService.start_pinned` runs `decider/snapshot.py` (the installer's
and the CLI's downloader) in a daemon worker thread. It fetches only the pinned revision's files named by the
checked-in hash list and the verified manifest, hashes each while streaming and links it only when it matches,
resumes a partial file with HTTP Range on the next attempt, and never sends the HF token. A snapshot that is
already complete is only verified. Progress (the planned files and their sizes, then the bytes on disk) goes to
the same Download row and `download` events; Cancel stops the worker before its next chunk. Another revision is
refused (`pinned_revision`).

## Deleting weights

`inventory/weights.py` plans and performs deletions; the API (`DELETE /api/inventory?ref=`, with
`GET /api/inventory/delete-preview` and `/api/inventory/cleanup-incomplete[/preview]`) is admin-only.

**Plan.** A Hugging Face ref maps to `<hub>/models--org--name` plus its `.locks` directory. A GGUF path
maps to the file and all its split shards. When a GGUF inside the HF cache is a snapshot symlink, the
blob it points to is deleted too, unless another snapshot link still uses that blob (then only the link
goes, and 0 bytes are reported).

**Shared blob store (huggingface_hub >= 2).** A repo blob may itself be a symlink into the cache-wide
store `<hub>/blobs/<xx>/<hash>`, which holds each distinct file once for all repos (with a
`<hash>.refs` manifest and a `<hash>.lock`). The plan collects the store entries that the deleted links
resolve to and then, read-only and before anything is removed, scans every symlink in the cache (all
`models--`, `datasets--` and `spaces--` repos) and in the GGUF roots outside it. Entries another link
still resolves to are kept; if part of the tree cannot be read, no store entry is removed. Only
well-formed entries count (a regular file `<xx>/<64 hex>` whose real path stays inside the store), and
the store directory itself is a protected root. Freed bytes count every real file once: repo files
that are not links, plus the exclusive store entries (0 for an entry with another hard link). Store
entries are removed after the repo's own files, one at a time under the entry's `.lock` (the lock
huggingface_hub holds while it links an entry into a repo and records the link in the manifest). Under
the lock the manifest is re-read, so an entry that a concurrent download has just reused is kept, and
the result says so. A `.lock` held for over 10 s also keeps the entry. Removing an entry also removes
its manifest; root-owned entries go through the helper while the lock is held. Incomplete-download
cleanup uses the same plan.

**Safety rules**, enforced on the server:

- **Containment.** Refs containing `..` are refused. Both the link location and the fully resolved path
  must lie inside the HF hub or a configured GGUF root, so symlinks cannot escape. A target that is, or
  contains, a configured root (hub, `hf_cache.local`, `.locks`, the shared store `blobs`, a GGUF root) is refused (`400
  unsafe_path`). Symlink loops and other path errors are refused the same way.
- **In use.** Every deployment's model and draft model are mapped to paths. A non-terminal deployment
  whose paths lie in a target blocks the delete (`409 model_in_use`, listing the deployments). Stopped
  and failed ones are returned as warnings.
- **Downloads.** A queued or running download of the repo blocks it (`409 download_active`).
- **Races.** The deleter publishes the targets it is about to delete, then plans again with a fresh
  database read. Deploy, start, restart, switch, reconfigure and download starts check those markers
  (and check again after creating their row), so either the deleter sees the new deployment or the
  deployment sees the deletion and is refused (`409 model_being_deleted`). Deletions are serialised.

**Execution.** One helper container runs per parent directory, from DGXMM's own image (`helper_image`;
it is never pulled):

- user `0:0`, so files left owned by root by engine containers can be removed;
- one bind mount: the target's parent host directory at `/x`, and nothing else;
- command `rm -rf -- /x/<name>…` (`rm -f` for files) as the entrypoint;
- `network_mode: none`, read-only root filesystem, all capabilities dropped except `DAC_OVERRIDE` and
  `FOWNER`, `no-new-privileges`, a process limit, `auto_remove`;
- label `dgxmm.helper=true` (never `dgxmm.managed`), and a `helper_timeout_s` (120 s) limit after which
  only that container is force-removed.

Anything the helper could not remove, or everything when the helper cannot run (no Docker, image
missing), is deleted in-process. A permission error there fails with `500 delete_failed` and a
`sudo chown -R <uid>:<gid> <path>` hint. Afterwards the inventory is refreshed. Every deletion, refusal
and failure is audited (`model_delete`, `model_delete_refused`, `model_delete_failed`; clean-ups add one
`incomplete_cleanup` summary row).

**Incomplete clean-up** applies the same rules to artifacts that are not complete only. Complete models
are never deleted through it, and Hugging Face repos with a partial blob written in the last 10 minutes
are skipped as `recently_active`.

## Process view

`memory/hostmem.py` scans `/proc` for `GET /api/host/processes` and for the external processes in the
budget. Per process it reads `status` first and skips kernel threads, zombies and processes with no
RSS and no GPU memory, then reads `cgroup` (to find the Docker container id) and `cmdline`.
`/proc/<pid>/environ` is never opened. GPU memory comes from NVML.

- **User.** Host processes get their name from the host's `/etc/passwd` (re-read when it changes).
  Processes in containers show the uid as seen inside the container, because container uids are not
  host accounts.
- **Container.** The cgroup's container id is resolved to name and image through the container
  directory in `docker_ctl.py`: one read-only `docker ps` listing, cached for 30 seconds and refreshed by
  a background loop. A miss may refresh early, at most every 5 seconds; Docker errors keep the last
  map. The budget loop only reads the cache, so it never waits for Docker.
- **Managed.** A process is DGXMM-managed when its container is an active deployment's container. A
  container with DGXMM labels that is not an active deployment is reported as `dgxmm_label`, and its
  memory counts as external.
- **Command.** `memory/redact.py` masks passwords, tokens and keys in the argv before it leaves the
  probe. Values that cannot be parsed reliably are masked as a whole (fail closed), and input size is
  capped so masking cost stays bounded. The result is shell-quoted and cut at 4 KiB. Masking is
  best-effort and favours over-masking; `tests/test_redact.py` documents the cases it covers.

Viewers and admins get the same masked data. The scan runs in a worker thread and takes tens of
milliseconds on a DGX Spark.

## Startup and background loops

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
| Decider supervisor | `decider_probe_interval_s` (30 s) | Probe the ready decider, restart it after a crash or hang (see [decision model](#decision-model-decider)). |
| Model kinds | 60 s | Ask the decision model about models no rule recognises (once per snapshot). |
| Decider retention | with the metrics rollups (60 s) | Prune `decisions` and `diagnoses`. |
| Decider network subnets | 30 s | Re-read the decider network's subnets, whose forwarding headers are ignored. |
| Key rotation requests | continuous | Pick up `rotate-decider-key` requests from the admin CLI. |

## Data model

SQLite tables (`dgxmm/db.py`):

| Table | Contents |
|---|---|
| `users` | username, argon2id hash, role (`admin` / `viewer`), `must_change_pw` |
| `cli_tokens` | SHA-256 of a short-lived admin CLI token, user, scope, expiry (deleted when the command ends) |
| `sessions` | SHA-256 of the session token, user, CSRF token, client IP, expiry |
| `deployments` | name, engine, model reference, validated config, state, container id, port, priority, alias (the LiteLLM model name), reservation and estimate, LiteLLM status, last error, exit code, whether the last exit was an OOM kill (`oom_killed`), `system` (the decision model) |
| `app_settings` | runtime settings changed in the UI (key, value, updated at): the LiteLLM URLs, enabled flag, master key and the URL it is bound to; the decision model's `decider.*` settings |
| `decisions` | one row per decision-model question: feature, subject, source (`rules` / `model` / `none`), choice, confidence, probabilities, latency, model and revision, SHA-256 of the input, error code, feedback (30 days / 50 000 rows) |
| `diagnoses` | per failure event: cause, source, confidence, summary, fix hint, patch, redacted log excerpt (latest 20 per deployment; unique per deployment and event) |
| `artifact_kinds` | model-derived kinds per artifact and snapshot time, with source and confidence |
| `schema_version` | applied migrations |
| `deployment_events` | state transitions with message and optional log tail |
| `memory_calibration` | estimated vs measured bytes per engine, model and config hash; OOM records |
| `benchmarks` | one row per concurrency level: TTFT p50/p95, ITL p50, total and per-request tok/s |
| `benchmark_runs` | one row per benchmark run: levels, status, the level that failed and its error |
| `metric_rollups` | 1-minute aggregates per deployment and for the host (30-day retention) |
| `downloads` | repo, revision, files, progress, state, error |
| `favourites` | model references starred in the Planner |
| `audit_log` | user, action, target, details, timestamp |

Live metrics (about the last hour at the scrape interval) are kept in memory only.

## HTTP API

Every route is under `/dgxmm/api` (`dgxmm/api/`). All of them except login need a session; *user* means any
signed-in user (`viewer` or `admin`). With a session, every request other than GET must also send the
session's CSRF token in `X-CSRF-Token`. The UI uses exactly
these routes, so this is also the API for scripts.

| Area | Routes | Role |
|---|---|---|
| Auth | `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`, `POST /auth/change-password` | login: none; others: user |
| Host | `GET /host` (budget, GPU, disk), `GET /host/processes?scope=all\|external\|managed&limit=50` | user |
| Deployments | `GET /deployments`, `GET /deployments/{id}`, `GET /deployments/{id}/litellm` | user |
| | `POST /deployments/preview`, `POST /deployments`, `PATCH /deployments/{id}` (`{"alias": …, "rewrite_keys": true}`), `POST /deployments/{id}/start\|stop\|sleep\|wake\|switch\|reconfigure`, `DELETE /deployments/{id}`, `GET /deployments/{id}/logs` | admin |
| Metrics and advice | `GET /deployments/{id}/metrics`, `GET /host/metrics`, `GET /deployments/{id}/advice`, `GET /deployments/{id}/benchmarks` | user |
| | `POST /deployments/{id}/benchmark` | admin |
| Inventory | `GET /inventory` | user |
| | `POST /inventory/refresh`, `GET /inventory/delete-preview?ref=`, `DELETE /inventory?ref=`, `GET /inventory/cleanup-incomplete/preview`, `POST /inventory/cleanup-incomplete` | admin |
| Catalog and downloads | `GET /catalog/search`, `GET /catalog/details`, `GET /downloads` | user |
| | `POST /downloads`, `POST /downloads/{id}/cancel` | admin |
| Planner | `GET /planner` | user |
| | `PUT /planner/favourites/{ref}`, `DELETE /planner/favourites/{ref}` | admin |
| LiteLLM | `GET /integrations/litellm` (`?probe=false` answers at once with the cached status), `GET /integrations/litellm/status`, `GET /integrations/litellm/models` (API bases only for admins) | user |
| | `PUT /integrations/litellm`, `POST /integrations/litellm/test`, `POST /integrations/litellm/sync`, `GET\|POST /integrations/litellm/keys`, `PATCH\|DELETE /integrations/litellm/keys/{token}` | admin |
| Decision model | `GET /decider/status`, `GET /decider/image`, `GET /deployments/{id}/diagnosis`, `GET /deployments/{id}/workload` | user |
| | `PUT /decider/settings`, `POST /decider/enable` (`{"actions": […]}`), `POST /decider/disable` (`{"remove": true}`), `POST /decider/models/download`, `POST /decider/image/build`, `POST /decider/rotate-key`, `GET /decider/decisions?feature=&limit=`, `POST /decider/decisions/{id}/feedback`, `POST /decider/suggest-config` | admin |
| Engines | `GET /engines` | user |
| Containers | `GET /containers/adoptable`, `POST /containers/{container}/adopt` | admin |
| Users and audit | `GET\|POST /users`, `PATCH\|DELETE /users/{id}`, `GET /audit` | admin |
| Events | `GET /stream` (server-sent events) | user |

Errors are JSON, `{"detail": {"code": …, "message": …}}`, with a stable `code` (for example
`model_name_taken`, `model_in_use`, `insufficient_memory`, `litellm_no_db`, `image_missing`,
`rate_limited`) and sometimes extra fields such as the deployment that holds a model name. Some
routes are rate-limited and decider request bodies are capped (see
[API limits](configuration.md#api-limits)).

Responses that changed for the decision model: plan options carry `recommended`, `recommendation`
(`source`, `confidence`, `reason`) and `unsafe_reason`; inventory artifacts `kind_source` and
`kind_confidence`; advice items `relevance` and `workload_fit`; deployment views `system` and
`startup_queue` (position and reason while queued). `POST /deployments/{id}/wake` answers `202` when
the wake-up continues in the background.

## Frontend

`frontend/` is a React 18 + TypeScript app built with Vite (`base: /dgxmm/`, router basename
`/dgxmm`). The Docker image builds it and serves it from `/app/static`.

- **Pages** (`src/pages/`): Login, Change password, Dashboard, Deployments, Deployment detail, Deploy
  wizard, Models (on disk), Catalog, Downloads, Planner, Processes, LiteLLM, Decision model, Users, Audit
  log. Deploy wizard, Catalog, Planner, Processes, LiteLLM, Decision model, Deployment detail and Audit
  are lazy-loaded.
- **Design system** (`src/components/ui/`): shadcn/ui components on Radix primitives, styled with
  Tailwind CSS tokens. There is a light theme and a dark theme. Status is always shown as icon plus
  text, and Inter and JetBrains Mono fonts are self-hosted. App building blocks (shell, sidebar, data
  tables, memory bar, plan chooser, command palette) live in `src/components/app/`.
- **Data**: TanStack Query for every API call (`src/api.ts` adds the CSRF header). `src/stream.ts`
  opens one `EventSource` on `/dgxmm/api/stream`: `budget` events update the host data in place,
  `metrics` events are appended to the live chart series, and `deployment`, `download` and `inventory`
  events refresh the matching queries; `diagnosis` refreshes a deployment's diagnosis and the
  admin-only `decider` topic carries image build progress.
- **Charts**: Recharts with a fixed palette that has light and dark variants.

## Security model

The threat model (assets, trust boundaries, the decision model's container, network, key and supply chain,
residual risks) is in [security.md](security.md).

- **Authentication** (`auth/`): username and password with argon2id hashes. New passwords need at
  least 12 characters. For unknown users, login still verifies a dummy hash, so response timing does not reveal which usernames exist. Failed and rate-limited
  logins are audited.
- **Sessions**: a random token in an HTTP-only, `SameSite=Strict` cookie `dgxmm_session` scoped to
  `/dgxmm`. Only its SHA-256 is stored. Sessions expire after `session_ttl_s` (12 h) of inactivity
  (the expiry slides on each request). The cookie is `Secure` when the request arrived over HTTPS
  (`secure_cookies: auto`). Changing a password signs out the user's other sessions. An admin reset
  signs out all of them.
- **CSRF**: every non-GET request must send the session's token in `X-CSRF-Token`.
- **Admin CLI tokens**: `python -m dgxmm.admin decider status|enable|disable` creates a short-lived admin
  token in `cli_tokens` (hashed), uses it over the admin unix socket only and deletes it afterwards. It
  is refused on the TCP port and on requests with forwarding headers, accepted only on the decider
  routes and the inventory refresh, and audited with `"via": "cli"`.
- **Rate limiting**: 5 login attempts per 60 seconds, per client IP and per username; per-user limits on
  the decision-model routes (see [API limits](configuration.md#api-limits)).
- **First login**: users created by bootstrap, by an admin or by a reset must change their password
  before any other API call.
- **Roles**: `viewer` has read-only access. Viewers can see deployments (with their LiteLLM endpoint
  and model name), models, metrics, the catalog, the planner and the masked process list. They cannot
  run deployment previews (`POST /deployments/preview`, which may ask the decision model), see users, the
  live container logs, the audit log, the adoptable-container list,
  delete previews (they contain host paths), LiteLLM API bases or virtual keys, and the UI hides the
  admin pages. `admin` is needed for every change. The last admin cannot be deleted or demoted.
- **Audit**: logins (also failed and rate-limited ones), password changes, deploy, start, stop, remove,
  sleep, wake, switch and reconfigure (plus their failures), model name changes, adopt, downloads and
  cancellations, inventory rescans, weight deletions and clean-ups (also refused and failed ones),
  LiteLLM settings changes, syncs and virtual-key changes, benchmarks, user changes, watchdog stops and
  command-line password resets are written to `audit_log`. Audit rows never contain the master key or a
  virtual key.
- **Secrets**: the LiteLLM master key is write-only (stored in `app_settings` or read from the secret
  file), bound to its URL, and never returned, logged or audited. A full virtual key appears only in the
  response that creates it. Process command lines are masked before they reach any client, and process
  environments are never read.
- **Container guard**: `docker_ctl.py` refuses to create a container without `dgxmm.managed=true` and
  to stop or remove one that lacks it. The one exception is the delete helper it creates itself
  (`dgxmm.helper=true`, see [deleting weights](#deleting-weights)). Containers it does not manage are
  only ever read.
- **Engine arguments**: configs are strict schemas, and extra CLI flags must match a per-engine
  allowlist (`engines/argguard.py`). Flags DGXMM controls (`--model`, `--port`, `--host`,
  `--gpu-memory-utilization`, …) cannot be overridden. Model references must be a repo id or a `.gguf`
  file under a configured root.
- **Trust boundary**: the Docker socket gives DGXMM root-equivalent control of the host. Only give the
  admin role to people you would trust with root, and keep the UI off the public internet (see
  [operations](operations.md#security-hardening)).
