---
description: "Every DGXMM setting with its default: config.yaml, environment overrides, .env and installer variables, and secrets."
---

# Configuration

DGXMM reads its settings from three places:

| File | Written by | Used by | Purpose |
|---|---|---|---|
| `.env` | `install.sh` (once) | Docker Compose | Host-specific values: user and group ids, port, bind address, model paths, Docker network |
| `config.yaml` | `install.sh` (once, from `config.example.yaml`) | the DGXMM app | Everything DGXMM itself does: memory gate, engines, paths, LiteLLM, polling |
| `secrets/*` | `install.sh` | Compose secrets → `/run/secrets/` | Hugging Face token, LiteLLM master key, first admin password |

The installer never overwrites `.env`, `config.yaml` or a non-empty secret file. After you edit `.env`,
apply the change with `./install.sh` (running it again is safe): Compose recreates the `dgxmm`
container when its image or settings changed. DGXMM reads `config.yaml` and the HF token only at
start, and `./install.sh` does not recreate an unchanged container, so after editing `config.yaml` run
`docker restart dgxmm`. Engine containers keep running either way.

A few settings can also be changed at runtime in the UI. They are stored in DGXMM's database and win
over `config.yaml`: the [LiteLLM connection](litellm.md#runtime-settings-and-precedence) (URLs, master key,
on/off) and the [decision model](decision-model.md#decision-model-runtime-settings) (on/off, model, per-feature switches
and thresholds, and **Pause all schedules**). They apply immediately, without a restart.

## config.yaml reference

Every key is optional. Anything left out takes the default below (from `dgxmm/config.py`).
`config.example.yaml` lists the same keys with comments. In host paths, a leading `~` means the home
directory of the host user DGXMM runs as (`DGXMM_HOST_HOME` in `.env`), not the home directory inside
the container.

### Memory gate

| Key | Type | Default | Meaning |
|---|---|---|---|
| `safety_margin_gib` | float ≥ 0 | `8` | Memory that is never given to models. It is also the watchdog's critical threshold and the startup-abort threshold. |
| `os_reserve_gib` | float ≥ 0 | `4` | Kept back for the OS, desktop and other processes. |
| `auto_protect` | bool | `false` | `true`: when memory turns critical, the watchdog stops the lowest-priority deployment by itself, and the decision model before any user deployment. `false`: it only warns (a decision model that exceeds its own reservation is stopped either way, see `decider_overuse_margin_gib`). |
| `vllm_runtime_fixed_gib` | float | `3.0` | Fixed vLLM overhead added to every vLLM estimate. |
| `vllm_cuda_graph_gib` | float | `1.5` | CUDA graph memory added to every vLLM estimate. |
| `llamacpp_runtime_fixed_gib` | float | `0.5` | Fixed llama.cpp overhead added to every llama.cpp estimate. |

### Engines

| Key | Type | Default | Meaning |
|---|---|---|---|
| `images.vllm` | string | `vllm/vllm-openai@sha256:22d3b121…352c8` | vLLM image (public multi-arch image, pinned by digest). |
| `images.llamacpp` | string | `ghcr.io/ggml-org/llama.cpp@sha256:e4cfff28…c2b5` | llama.cpp server image (build b11669, multi-arch, pinned by digest). |
| `images.vllm_allowed` | list of strings | `[]` | Images a single-node vLLM deployment may choose instead of `images.vllm` (its config's `image`; the **Engine image** select). Each entry must be pinned by digest (`<repository>[:<tag>]@sha256:<digest>`), else DGXMM refuses to start. Their model registries are read like the default image's (pulled on the DGXMM host for that). See [engine images](../user-guide/deploying.md#engine-images) and [security](../security.md#engine-images). |
| `images.llamacpp_allowed` | list of strings | `[]` | The same for llama.cpp deployments (instead of `images.llamacpp`; cluster deployments always run `images.llamacpp_rpc`). |
| `images.llamacpp_rpc` | string | `dgxmm-llamacpp-rpc:61dbf49-u1` | llama.cpp image for cluster deployments (`ggml-rpc-server` workers, `llama-server --rpc` head): built locally from `deploy/llamacpp-rpc` (commit 61dbf49, `GGML_RPC=ON`, CUDA sm_121) and loaded on every node of the cluster (`docker save … \| ssh <node> docker load`). A local build has no registry digest, so it is referenced by tag; the Dockerfile pins its bases by digest and the source by commit. |
| `docker_network` | string | `dgxmm` | Docker network that engine containers join. **Must equal `DGXMM_NETWORK` in `.env`.** |
| `port_range` | `[low, high]` | `[8100, 8199]` | Host ports handed to engine containers (1024–65535, ascending). Ports used by other containers are skipped. |
| `cluster_port_range` | `[low, high]` | `[29500, 29599]` | One cluster port per cluster deployment: vLLM's master port (`--master-port`) or the RPC port the llama.cpp workers listen on. A cluster's API and cluster ports are held on every node of its placement. |
| `publish_bind` | string | `127.0.0.1` | Host address that engine ports are published on. Keep it on loopback: clients should go through LiteLLM or the Docker network. |
| `engine_base_host` | string or null | `null` | `null`: DGXMM (and LiteLLM) reach engines as `dgxmm-<name>:<port>` on `docker_network`. Set a hostname or IP to use the published ports on that host instead. |
| `startup_timeout_s` | int | `900` | Seconds an engine may take to become ready before DGXMM fails it. |
| `max_concurrent_startups` | int ≥ 0 | `1` | Engines starting/loading at the same time (the decision model included); the others wait in state `queued`. `0` = no limit, which lets vLLM's start-up memory profile fail when two engines start together on unified memory. |
| `startup_queue_timeout_s` | float > 0 | `1800` | A queued deployment fails after waiting this long for a start-up slot. |
| `wake_queue_timeout_s` | float > 0 | `300` | A wake-up waiting for a running start-up gives up after this long (409 `startup_busy`, or the reason in the deployment's last error if it was already continuing in the background). |
| `wake_sync_wait_s` | float ≥ 0 | `50` | The wake request waits this long for its turn; after that it answers 202 with `startup_queue` and the wake-up continues in the background. |
| `vllm_sleep` | `stop` / `level1` | `stop` | What **Sleep** does to a single-node vLLM deployment. `stop`: it stops the deployment (the action reads *Sleep (stops the model)* and asks first, like a cluster's) and **Wake** starts it again (a full model load); engines never get `--enable-sleep-mode` or vLLM's development routes. On unified memory (every DGX Spark / GB10) vLLM's level-1 sleep copies the weights into the same RAM, so it frees nothing and vLLM refuses it (`Memory usage increased after sleeping`). `level1`: vLLM's own sleep (KV cache freed, weights offloaded to CPU RAM), for hosts with separate GPU memory; it takes effect at each deployment's next start. A stored `enable_sleep_mode` is kept either way. Clusters always stop. See [deployments](../user-guide/deployments.md#sleep-and-wake). |
| `engine_health_interval_s` | float ≥ 0 | `30` | How often every ready deployment's engine is probed: its `GET /health` (5 s timeout; a cluster: on its head; no test completion, which would count as traffic). `0` turns the probe off. |
| `engine_health_failures` | int ≥ 1 | `3` | Failed probes in a row after which a ready deployment is failed: its container (every rank of a cluster) is removed, `last_error` says why and LiteLLM drops it. One slow or failed probe alone never fails it. |
| `vllm_cache_mounts` | list of `{host, target, mode}` | `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.triton` → `/root/...` | Host directories mounted into vLLM containers so compiled kernels survive restarts. `mode` defaults to `rw`. |
| `llmfit_bin` | string | `llmfit` | llmfit executable used for catalog scores. |

### Decision model (decider)

Whether the decider runs, which model it runs and the per-feature switches and thresholds are not in
`config.yaml`: they are set on the **Decision model** page and stored in the database (see
[decision model runtime settings](decision-model.md#decision-model-runtime-settings)). The model is `vllm-sr/d3-mini`
(default, 13 GiB reserved) or `vllm-sr/d3-nano` (9 GiB), both pinned to a revision whose files are
checked against hashes built into the decider image. Using it: [usage](../user-guide/decision-model.md);
measurements: [decider-evaluation.md](../decider-evaluation.md). The keys below tune its container,
supervision and data retention; the defaults fit a DGX Spark.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `images.decision` | string | `dgxmm-decider:3` | Local image the decider runs from (built from `deploy/decider`, never pulled). The tag changes whenever the image's contract with DGXMM changes, so an older image counts as missing and is rebuilt. The image lists the models it can serve in its `dgxmm.decider.models` label; enabling a model the image does not list answers 409 `image_outdated` (rebuild it). |
| `decider_network` | string | `dgxmm-decider` | Docker network of the decider. DGXMM creates it as an *internal* network (no outbound traffic, no published port, labelled `dgxmm.purpose=decider` and `dgxmm.install`), refuses a non-internal network of that name, and attaches its own container to it. Empty: the decider joins `docker_network` instead, where every container on that network can reach it and only its key protects it. Letters, digits and `_ . -`, up to 63 characters. |
| `decider_user` | `uid:gid` | `10001:10001` | User the decider container runs as (the image's unprivileged user). Root (`0`) is refused. |
| `decider_memory_limit_gib` | float ≥ 2 | `16` | cgroup memory limit of the decider container (swap off). On GB10 GPU memory is not charged to the cgroup, so this caps the host-side memory: the process, `/tmp` and the page cache of the weights it reads (measured peaks 9.3–12.7 GiB). The 13 GiB budget reservation is separate. |
| `decider_pids_limit` | int ≥ 64 | `512` | Processes and threads in the decider container (measured peak about 100). |
| `decider_tmp_gib` | float > 0 | `0.75` | Size of the decider's `/tmp` tmpfs (GPU kernel caches, about 0.23 GiB used). It counts as used memory, so it is capped to stay inside the reservation. |
| `decider_key_file` | path or null | `null` | File of the decider's bearer key, generated on first use (mode 600). `null`: `data/secrets/decider_api_key`. |
| `decider_build_dir` | path or null | `null` | Docker build context of the decider image. `null`: the `deploy/decider` directory shipped with DGXMM (`/app/deploy/decider` in the container). |
| `install_id` | string or null | `null` | Set from `DGXMM_INSTALL_ID` in `.env` (generated by the installer). Labels the containers DGXMM creates and the images it builds `dgxmm.install=<id>`, so `uninstall --purge` removes only this install's. |
| `decider_max_in_flight` | int | `2` | Questions sent to the decider at once. d3_server answers one request at a time and finishes requests whose caller already gave up, so the rest wait in DGXMM. |
| `decider_queue_timeout_s` | float | `1.0` | Longest wait for a slot for interactive questions (plan, parser); background ones (diagnosis, kind, workload) may wait up to half their deadline. A question that cannot start in time, or whose remaining deadline is shorter than the measured service time, is answered by rules at once. |
| `decider_probe_interval_s` | float | `30` | How often the supervisor probes a ready decider (`/health` plus one question). |
| `decider_probe_timeout_s` | float | `10` | Probe timeout. |
| `decider_hang_probes` | int | `3` | Failed probes in a row (container still running) after which the decider is restarted. |
| `decider_restart_backoff_s` | float | `30` | Delay before the next automatic restart; doubles after each restart. |
| `decider_restart_backoff_max_s` | float | `900` | Upper bound of that delay. |
| `decider_max_restarts_per_hour` | int | `5` | After this many automatic restart attempts in an hour (failed ones included) the supervisor gives up until the decider is enabled again (**Retry** on the Decision model page). |
| `decider_stable_s` | float | `600` | Healthy this long after a restart resets the backoff. |
| `decider_retention_days` | int ≥ 1 | `30` | Decision-log rows (`decisions`) older than this are deleted (once a minute, by the metrics loop). |
| `decider_max_rows` | int ≥ 1000 | `50000` | At most this many of the newest decision-log rows are kept (rows with feedback excepted). Diagnoses keep the latest 20 per deployment and are deleted with it. |
| `decider_feedback_retention_days` | int ≥ 1 | `180` | Decision-log rows with feedback (what threshold tuning learns from) are never removed by the row cap or the `decider_retention_days` rule; they are deleted after this many days. A value below `decider_retention_days` is treated as `decider_retention_days`. |
| `decider_overuse_margin_gib` | float ≥ 0 | `2` | The memory watchdog stops the decider when its measured memory exceeds its 13 GiB reservation by more than this in two budget samples in a row. |

Automatic restarts happen only for a decider that crashed, hangs or was stopped by the memory watchdog while
enabled (never after an admin stopped it), never while the memory watchdog reports pressure, and through the
normal fit check and start-up queue. A watchdog stop counts as a failure: the restart waits out the (doubling)
backoff, so a decider that keeps running away is not restarted in a loop.

**Watchdog and `auto_protect`.** The decision model is advisory, so the watchdog gives it up first. Whatever
`auto_protect` says, it stops the decider when its measured memory exceeds its reservation by more than
`decider_overuse_margin_gib` in two budget samples in a row. With `auto_protect: true` it is also the first
deployment stopped when memory turns critical. With the default `auto_protect: false`, critical memory only raises
the alert, also for the decider. If stopping the decider fails, it is skipped and the normal order applies.

Each feature waits a fixed time for an answer before it falls back to its rules (not configurable): failure
diagnosis 10 s (in the background), parser suggestions 4 s per question, plan recommendation 2 s, model kind and
workload 2 s (in the background); config advice, draft pairing, engine choice, the compatibility check and the
remote-code check 4 s; ⌘K commands 3 s per question and 8 s for the whole interpretation; health triage, reading
labels, routing advice and schedule proposals 10 s (in the background). Plan rankings are cached for 2 minutes and
workload labels for 15 minutes.

### Nodes

DGXMM can manage models on one or more nodes (DGX Spark boxes or similar). With no `nodes:` key in
`config.yaml`, DGXMM behaves exactly as today: one implicit local node. With a `nodes:` list, one
DGXMM instance (on the node it runs on, the local node) controls every node; other nodes run no DGXMM.

Per deployment you choose where it runs:

- **On the local node** (the node running DGXMM) — single-node deployment
- **On a remote node** — single-node deployment there, managed from the local node's DGXMM
- **Across nodes** (cluster) — one model split across the nodes' GPUs (vLLM tensor parallel over NCCL/RDMA,
  or llama.cpp RPC layer split)

**Configuration.** The list holds exactly one local node: the entry whose `docker_url` starts with `unix://`.
Its `id` must be `local` (deployments and the API name it so); its `label` is free (shown in the UI) and its place
in the list does not matter. Every other entry is a remote node (`docker_url: ssh://<user>@<cluster_ip>`, the
docker_url host must be the node's `cluster_ip`; SSH must listen on port 22, since a `:port` in the URL is refused at
load), with an `id` of lowercase letters, digits and `-` (1–32
characters, starting with a letter or digit), its `cluster_ip`, a pinned `known_hosts` line and the SSH key file
(`node_ssh_key_file`, below). `nccl` is needed on every node of a vLLM cluster; llama.cpp clusters need only
`cluster_ip`.

```yaml
nodes:
  - id: local                      # the node running DGXMM: always id "local"
    label: DGX-1
    docker_url: unix:///var/run/docker.sock
    cluster_ip: 192.0.2.1
    nccl:
      ib_hca: mlx5_0:1,mlx5_1:1
      socket_ifname: eth0          # also used as GLOO_SOCKET_IFNAME

  - id: dgx-2
    label: DGX-2
    docker_url: ssh://dgxuser@192.0.2.2
    cluster_ip: 192.0.2.2
    publish_bind: 192.0.2.2        # default: cluster_ip
    hf_cache_host: /home/dgxuser/.cache/huggingface
    ssh_user: dgxuser              # default: the user in docker_url
    known_hosts: |
      192.0.2.2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...
    nccl:
      ib_hca: mlx5_2:1,mlx5_3:1
      socket_ifname: eth1
```

| Node field | Type | Meaning |
|---|---|---|
| `id` | string | `local` for the local node; a remote node: 1–32 characters, lowercase letters, digits and `-`. Used in placements, the API and the database. |
| `label` | string | Name shown in the UI. |
| `docker_url` | string | Docker connection. Local: `unix:///var/run/docker.sock`. Remote: `ssh://<user>@<cluster_ip>` (the host must equal `cluster_ip`; port 22 only, no `:port`). |
| `cluster_ip` | IP or null | The node's address on the cluster link (point-to-point RDMA network). Required on remote nodes and on every node of a cluster. |
| `publish_bind` | IP or null | Remote nodes: address their engine API ports are published on. Defaults to `cluster_ip`; must be a concrete unicast address (never `0.0.0.0`, loopback or a hostname). Not used for the local node (its engines use `publish_bind` of the main settings, `127.0.0.1`). |
| `hf_cache_host` | path or null | The node's Hugging Face cache on its host filesystem (mounted read-only into its engines and its probe). Local node default: `hf_cache.host`. It must exist on a remote node, or its probe cannot start and the node stays stale. |
| `ssh_user` | string or null | Remote nodes: SSH user for Docker over SSH; default the user in `docker_url`. |
| `known_hosts` | string or null | Remote nodes (required): the pinned `known_hosts` line(s) of the node, see [SSH setup](#node-ssh-key-and-known-hosts). |
| `nccl.ib_hca` | string | NCCL InfiniBand HCAs (`NCCL_IB_HCA`, e.g. `mlx5_0:1,mlx5_1:1`). Required with `nccl`. |
| `nccl.socket_ifname` | string | Interface for NCCL and Gloo bootstrap traffic (`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`). Required with `nccl`. |

**The implicit local node.** When `nodes:` is not set, DGXMM uses one implicit local node with id and label
`local`, Docker socket `unix:///var/run/docker.sock`, `hf_cache_host` from `hf_cache.host`, no `cluster_ip` and no
NCCL settings; its engines publish on `publish_bind` (`127.0.0.1`). Existing deployments (from before multi-node)
are mapped to this node.

**Per-node settings.** Each node uses its own `hf_cache_host`, `publish_bind` and `cluster_ip`. The Docker
network (`docker_network`) exists only on the local node; remote engines run on their node's default bridge
network and publish their API port on `publish_bind` (typically the cluster IP). Settings like `port_range`,
`startup_timeout_s`, and `vllm_cache_mounts` apply globally; `vllm_cache_mounts` (local host paths) are not
mounted on remote nodes.

**Node health.** DGXMM keeps one long-lived probe container per remote node, `dgxmm-node-probe` (`images.probe`,
pinned by digest; read-only root filesystem, no network, no capabilities; the node's `/proc` and `hf_cache_host`
mounted read-only; labels `dgxmm.managed=true` and `dgxmm.role=probe`, so it is never shown or adopted as a
deployment). It samples `meminfo` every `remote_probe_interval_s` (default 5 s) and checks weights on the node. A
node whose last reading is older than `3 × interval` is marked **stale**: new starts are refused there (fail
closed), running deployments are left alone, and DGXMM retries the probe in the background. `GET /api/nodes` gives
the reason of the last failed attempt (`reason`, e.g. the probe could not be created because `hf_cache_host` does
not exist there), and the Dashboard shows it. When the node answers again it is marked fresh and new deployments
are admitted again.

**Sleep mode on remote nodes.** (Only with `vllm_sleep: level1`; with the default `stop` no engine gets sleep
mode.) A remote engine publishes on its node's cluster IP, which every container of the
DGXMM host can reach (local engines publish on `127.0.0.1`), and vLLM's sleep mode turns on development routes
(`/sleep`, `/wake_up`, `/collective_rpc`, ...) that do not check the engine key. So a single-node vLLM deployment on
a remote node gets no sleep mode unless `remote_dev_endpoints: true` (default `false`): it then has no Sleep
action (Stop frees its memory). Clusters never get sleep mode. See [security](../security/overview.md) for the firewall rule that keeps
other containers off the cluster link.

**Cluster port allocation.** vLLM clusters need a master port (`--master-port`) on the head node; llama.cpp
clusters need RPC ports on worker nodes. These ports are allocated from `cluster_port_range` (default
`[29500, 29599]`), separate from the main `port_range`. The head's API port and master port are reserved on
every node in the cluster placement.

**llama.cpp RPC image.** Cluster deployments using llama.cpp need a special image with RPC support, built
locally from `deploy/llamacpp-rpc` (commit 61dbf49, `GGML_RPC=ON`, CUDA sm_121) and loaded onto every node.
Every rank runs as DGXMM's own uid:gid (`DGXMM_UID:DGXMM_GID`, never root; the image's default user is the
unprivileged uid 10001), so each node's `hf_cache_host` must be readable by that uid (it owns the cache when the
same user runs DGXMM's host and the remote node's account). Workers run without a local tensor cache
(`ggml-rpc-server` without `-c`): nothing is written on a worker, at the cost of sending its share of the weights
over the cluster link at every load (about 100 GB for a model split across two nodes).
The image is referenced by tag (`images.llamacpp_rpc`, default `dgxmm-llamacpp-rpc:61dbf49-u1`) since it has no
registry digest:

```yaml
images:
  llamacpp_rpc: dgxmm-llamacpp-rpc:61dbf49-u1
```

To build and load it on both nodes (from the node running DGXMM):

```bash
# Build locally
docker build -t dgxmm-llamacpp-rpc:61dbf49-u1 deploy/llamacpp-rpc

# Load on remote node
docker save dgxmm-llamacpp-rpc:61dbf49-u1 | ssh dgxuser@192.0.2.2 docker load
```

<a id="node-ssh-key-and-known-hosts"></a>**Node SSH key and known_hosts.** Remote nodes need an SSH key for Docker-over-SSH. The key is a file under
the data directory, pointed to by `node_ssh_key_file` in `config.yaml`. At start DGXMM writes the ssh client
configuration for the remote nodes to `<data_dir>/ssh/config` (inside the container `/app/data/ssh/config`, set by
`DGXMM_SSH_CONFIG`; directories mode 0700, files 0600): per node its user, the key, the pinned `known_hosts` line,
`StrictHostKeyChecking yes`, `BatchMode yes`, `ConnectTimeout 10` and keep-alives. It never writes a `~/.ssh`.
Every ssh DGXMM runs uses that file: the image's `ssh` (`deploy/ssh/dgxmm-ssh`, first on `PATH`) passes it with `-F`,
because OpenSSH reads `~/.ssh/config` only from the passwd home directory, and makes ssh work when `DGXMM_UID` has no
passwd entry in the image.

```yaml
# node_ssh_key_file: null  # null -> <data_dir>/secrets/node_ssh_key
```

Default: `null`, which resolves to `<data_dir>/secrets/node_ssh_key` (inside the container: `/app/data/secrets/node_ssh_key`;
on the host: `data/secrets/node_ssh_key` relative to `docker-compose.yaml`). The key must have mode 0600. To set it up:

1. **Create a dedicated SSH key** (no passphrase):
   ```bash
   ssh-keygen -t ed25519 -f data/secrets/node_ssh_key -N '' -C 'dgxmm-node-access'
   chmod 600 data/secrets/node_ssh_key
   ```

2. **Install the public key on the remote node**:
   ```bash
   ssh-copy-id -i data/secrets/node_ssh_key.pub dgxuser@192.0.2.2
   ```

3. **Get the `known_hosts` line** with `ssh-keyscan` and verify it against the host's real key:
   ```bash
   # Scan the host key
   ssh-keyscan -t ed25519 192.0.2.2
   
   # Verify it matches the host's actual key fingerprint
   ssh-keyscan -t ed25519 192.0.2.2 | ssh-keygen -lf -
   # Compare with:
   ssh dgxuser@192.0.2.2 'ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub'
   ```

4. **Add the verified line to the node config** in `config.yaml`:
   ```yaml
   nodes:
     - id: dgx-2
       docker_url: ssh://dgxuser@192.0.2.2
       known_hosts: |
         192.0.2.2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyHere...
   ```

The key file must be readable by the user the DGXMM container runs as (set by `DGXMM_UID` in `.env`). Every
connection checks the host key against the pinned line (no trust on first use): a node whose key does not match,
or that cannot be reached, is not a start-up error; it is marked unreachable and stale (`GET /api/nodes` and the
Dashboard show why) and nothing is started there until it answers. Always verify the key fingerprint out of band.

**Remote probe interval and cluster port range.**

```yaml
remote_probe_interval_s: 5.0   # Sample remote node memory every 5 seconds
cluster_port_range: [29500, 29599]  # Master/RPC ports for clusters
remote_dev_endpoints: false    # sleep mode (vLLM dev routes) for single deployments on remote nodes
```

**Memory budget per node.** Fit checks, memory budgets, the watchdog and the planner work per node. A cluster
deployment's memory is counted on every node it uses: the per-rank share on each. A single-node deployment
(local or remote) reserves memory on that node only. The Dashboard shows one memory bar per node; a cluster
deployment appears on both nodes' bars.

**Weights on remote nodes.** Deploying on a remote node whose HF cache lacks the model is refused with
`weights_missing` and a `copy_weights` action. For llama.cpp clusters, only the head node needs the GGUF
(workers receive tensors over RPC); for vLLM clusters, every node needs the model.

### Models and storage

| Key | Type | Default | Meaning |
|---|---|---|---|
| `hf_cache.host` | path | `~/.cache/huggingface` | Hugging Face cache on the host (mounted into engines). |
| `hf_cache.local` | path | `/data/hf` | The same directory inside the `dgxmm` container (set by `docker-compose.yaml`). |
| `gguf_roots` | list of `{host, local}` | `[{host: ~/models, local: /data/gguf/0}]` | Directories scanned for `.gguf` files. See [Model directories](models-and-engines.md#model-directories). |
| `min_free_disk_gib` | float | `50` | Downloads are refused when they would leave less than this free on the HF cache disk. |
| `helper_image` | string or null | `null` | Image of the short-lived helper container that [deletes weights](../architecture/storage.md#deleting-weights) as root. `null`: the image of the running `dgxmm` container, else `dgxmm:latest`. It is never pulled; if it is missing, DGXMM deletes in-process. |
| `helper_timeout_s` | float | `120` | Seconds a delete helper may run before it is killed. |

### LiteLLM and secrets

| Key | Type | Default | Meaning |
|---|---|---|---|
| `litellm_url` | URL | `http://litellm-proxy:4000` | LiteLLM proxy that DGXMM registers models with (the internal URL). The LiteLLM page can override it. |
| `litellm_key_file` | path | `/run/secrets/litellm_master_key` | LiteLLM master key file. Used only with the default `litellm_url`. Read each time DGXMM talks to LiteLLM. |
| `litellm_public_url` | URL | `""` | Base URL clients use to reach LiteLLM, for example `https://llm.example.com` (no `/v1`). Shown in the UI as the endpoint `<url>/v1`; empty means the internal URL is shown. Default for the LiteLLM page's **Public endpoint URL**. `./install.sh --with-litellm` sets it on a fresh `config.yaml`. |
| `hf_token_file` | path | `/run/secrets/hf_token` | Hugging Face token file. Empty or missing means anonymous Hub access. |

### On-demand models

How [on-demand deployments](../user-guide/on-demand.md) behave and how LiteLLM reaches DGXMM's on-demand
gateway. The two timeouts are the defaults; a deployment's own idle timeout (Mode card, or
`PATCH /api/deployments/{id}/mode`) wins over `ondemand_idle_timeout_s`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ondemand_idle_timeout_s` | int, 60–86400 | `300` (5 min) | An on-demand deployment with no request for this long (nothing in flight, no activity in the engine's metrics) is stopped. Raise it when restarts are slow compared with the gaps between your requests; the deploy wizard pre-fills it. |
| `ondemand_queue_timeout_s` | int, 300–3600 | `900` (15 min) | A request waiting for memory (or for its model to start) gives up after this long with 503 `ondemand_queue_timeout` and `Retry-After: 60`. The limit counts from the request's arrival, so it includes the model's start-up (about 2 minutes even for a small model); a value below 300 is raised to 300, with a warning in the log (earlier versions allowed 60, so an older config still loads). DGXMM sets the `timeout` and `stream_timeout` of the on-demand LiteLLM entries to this value plus 600 s, at least 1200 s (1500 s with the default), and re-registers the entries when it changes, so LiteLLM never gives up first; every proxy in front of LiteLLM needs a read timeout of at least that entry timeout, 1800 s with the default (see [troubleshooting](../operations/troubleshooting.md#requests-to-an-on-demand-model-fail-after-about-60-seconds)). |
| `ondemand_max_queued` | int, 1–10000 | `64` | At most this many requests wait for on-demand models at a time, all deployments together (queued for memory or waiting for a start); each holds its request body, up to 16 MiB, in DGXMM's memory. One more is refused at once with 503 `ondemand_queue_full` and `Retry-After: 60`. Requests for a ready model are never refused. |
| `ondemand_key_file` | path | `/run/secrets/dgxmm_ondemand_key` | Bearer key LiteLLM sends to the gateway (`secrets/dgxmm_ondemand_key`, see [secrets](#secrets)). Missing or empty: the gateway answers every request 503 `ondemand_not_configured` and on-demand deployments are not registered in LiteLLM. |
| `ondemand_allowed_cidrs` | `auto` or comma-separated CIDRs (or a YAML list) | `auto` | Who may connect to the gateway. `auto` (or empty): the subnets of the Docker networks DGXMM's container is attached to, except the decider's network, without each network's gateway address and the first address of each subnet, where connections to published host ports arrive from. Set a list only when LiteLLM reaches DGXMM from an address auto does not cover. The listed ranges get the same exclusions as auto: the gateway and first address of every network DGXMM's container is attached to, and the whole decider network, are refused even when the list covers them (DGXMM logs one warning naming them), so the host and the LAN never reach the key check. Invalid CIDRs stop DGXMM from starting. Requests carrying `X-Forwarded-For`, `Forwarded` or `X-Real-IP` are always refused. |
| `dgxmm_internal_url` | URL | `""` | The URL LiteLLM uses to reach DGXMM, including the base path. Empty: `http://dgxmm:8090` + `base_path`, i.e. `http://dgxmm:8090/dgxmm`, the `dgxmm` container on the Docker network. Change it only when LiteLLM reaches the DGXMM container under another name or network alias. It must be a direct connection: a URL through nginx or another proxy is refused by the gateway. |

### Web UI and sessions

| Key | Type | Default | Meaning |
|---|---|---|---|
| `base_path` | string | `/dgxmm` | URL prefix of the UI and API. The frontend is built for `/dgxmm`, so leave this alone. |
| `session_ttl_s` | int | `43200` (12 h) | Login session lifetime. It is extended on every request. |
| `secure_cookies` | `auto` / `always` / `never` | `auto` | `auto`: the cookie is `Secure` when the request is HTTPS (directly or via `X-Forwarded-Proto`). Use `always` behind an HTTPS proxy. |
| `tz` | IANA zone or null | `null` | Time zone of daily/weekly schedules (background insight jobs, sleep/wake schedules), e.g. `Europe/Berlin`. Set from `DGXMM_TZ` in `.env`, which the installer detects from the host. `null`: `$TZ`, then `/etc/localtime`, then UTC. Across DST changes a time that does not exist runs just after the gap (02:30 becomes 03:30), and a time that occurs twice runs once, at its first occurrence. |
| `auth.oidc.*` | mapping | off | Single sign-on with OpenID Connect: issuer, client id, role mapping, auto-redirect, password sign-in on or off. All keys are in the [single sign-on guide](../user-guide/single-sign-on.md#options). |
| `trusted_proxies` | comma-separated IPs/CIDRs (or a YAML list) | `127.0.0.1,::1` | IPs/subnets trusted to set `X-Forwarded-For` (for correct client IPs in logs and rate limiting). Default is loopback-only; an empty value means the default. Set it to your proxy's IP or Docker network subnet, for example `172.18.0.0/16`. `*` trusts every client (logged as a warning; not recommended). Invalid entries stop DGXMM from starting. Peers on the decider's internal network (`decider_network`) are never trusted, even when a range here covers them. `DGXMM_TRUSTED_PROXIES` in `.env` overrides it. |

### Polling intervals

| Key | Type | Default | Meaning |
|---|---|---|---|
| `monitor_poll_s` | float | `3` | How often a starting deployment is checked. |
| `budget_interval_s` | float | `2` | Memory budget refresh. It also drives the live UI, the watchdog and the liveness checks. |
| `scrape_interval_s` | float | `5` | Engine Prometheus metrics scrape. |
| `inventory_interval_s` | float | `300` | Rescan of the HF cache and GGUF roots. The **Rescan** button on Models runs one immediately. |

### Paths inside the container

These are set by the image. You rarely need to change them. `docker-compose.yaml` mounts the host's
`/proc`, `/sys/fs/cgroup` and `/etc/passwd` read-only at the `host_*` paths below.

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `/app/data` | SQLite database and caches (`./data` on the host). |
| `static_dir` | `/app/static` | Built frontend. |
| `host_proc` | `/host/proc` | Host `/proc` (read-only mount). |
| `host_cgroup` | `/host/sys/fs/cgroup` | Host cgroup tree (read-only mount). |
| `host_passwd` | `/host/etc/passwd` | Host `/etc/passwd` (read-only mount), used only to show host user names on the Processes page. No other account file is mounted. Without it, numeric uids are shown. |

## Environment variable overrides

Any key can be set with `DGXMM_<KEY_IN_UPPER_CASE>`, which wins over `config.yaml`. The value is
parsed as YAML, so lists and objects work. An object replaces the whole key, and its missing fields
take their defaults:

```bash
DGXMM_AUTO_PROTECT=true
DGXMM_PORT_RANGE="[8200, 8299]"
DGXMM_IMAGES="{vllm: vllm/vllm-openai:latest}"     # llamacpp keeps its default
```

Two more variables are not config keys: `DGXMM_CONFIG` (path of the YAML file, default `config.yaml`;
`/app/config.yaml` in the container) and `DGXMM_HOST_HOME` (what `~` means in host paths).

The container only receives the variables listed in `docker-compose.yaml`. To pass an override, add a
`docker-compose.override.yaml` next to it, which Compose merges automatically:

```yaml
services:
  dgxmm:
    environment:
      DGXMM_AUTO_PROTECT: "true"
```

Then run `./install.sh`. Editing `config.yaml` is usually simpler.

## .env and installer variables

`./install.sh` writes `.env` on first install from flags, environment variables and detected values.
Once `.env` exists, its values win: a different flag or variable only produces a warning. Keys missing
from an older `.env` are appended.

| `.env` key | Installer flag / variable | Default | Meaning |
|---|---|---|---|
| `DGXMM_UID`, `DGXMM_GID` | (detected) | `id -u`, `id -g` | User the container runs as. It owns `data/` and files DGXMM writes to the model caches. |
| `DOCKER_GID` | (detected) | group of `/var/run/docker.sock` | Added to the container so it can use the Docker socket. |
| `DGXMM_PORT` | `--port` / `DGXMM_PORT` | `8090` | Host port of the UI. |
| `DGXMM_BIND` | `--bind` / `DGXMM_BIND` | `0.0.0.0` | Host IPv4 address to bind. Use `127.0.0.1` behind a reverse proxy. |
| `DGXMM_NETWORK` | `--network` / `DGXMM_NETWORK` | `dgxmm` | Docker network for DGXMM and engines. It is created (labelled `dgxmm.created=true`) if it does not exist. |
| `DGXMM_HF_CACHE` | `--hf-cache` / `DGXMM_HF_CACHE` | `$HOME/.cache/huggingface` | HF cache, mounted at `/data/hf`. |
| `DGXMM_MODELS_DIR` | `--models-dir` / `DGXMM_MODELS_DIR` | `$HOME/models` | GGUF directory, mounted at `/data/gguf/0`. |
| `DGXMM_HOST_HOME` | `DGXMM_HOST_HOME` | `$HOME` | Home directory that `~` in `config.yaml` expands to. |
| `DGXMM_WITH_LITELLM` | `--with-litellm` / `DGXMM_WITH_LITELLM=1` | `0` (key not written) | Run the [bundled LiteLLM](litellm.md#bundled-litellm). Written only when enabled; set `0` to turn it off. |
| `DGXMM_LITELLM_PORT` | `--litellm-port` / `DGXMM_LITELLM_PORT` | `4000` | Host port of the bundled LiteLLM, bound to `DGXMM_BIND`. |
| `DGXMM_WITH_DECIDER` | `--with-decider` / `DGXMM_WITH_DECIDER=1` | `0` (key not written) | Set up the [decision model](../operations/decision-model.md) on every install/upgrade: build its image, download and verify its model snapshot, enable it and wait until it is healthy. Written only when enabled. Set `0` so the installer stops re-enabling it (disabling it is done in the UI or with the admin CLI). |
| `DGXMM_INSTALL_ID` | (generated once) | 12 random letters/digits | Labels the containers DGXMM creates and the images it builds (`dgxmm.install`), so `uninstall` removes only this install's decider container and image. |
| `DGXMM_TZ` | `DGXMM_TZ` | the host's zone (`timedatectl`, `/etc/timezone`, or the `/etc/localtime` link), else `UTC` | Passed to the container as [`tz`](#web-ui-and-sessions) and `TZ`: the zone of daily/weekly schedules. Must be an IANA name such as `Europe/Berlin`. |
| `DGXMM_HOST_PID` | `DGXMM_HOST_PID` | `1` | `1`: the DGXMM container runs in the host PID namespace (`pid: host`), so NVML reports the GPU memory of the engine containers and it counts as each deployment's measured memory (see [memory budget](../architecture/memory.md#budget)). `0`: the container's own PID namespace; deployments' GPU memory is then not measured and the budget credits the rest of a ready deployment's reservation instead. The installer passes it to compose as `DGXMM_PID_MODE` (`host` or empty); a plain `docker compose up -d` without the installer uses `host`. Apply a change with `./install.sh`. See [hardening](../security/hardening.md). |
| `DGXMM_TRUSTED_PROXIES` | `DGXMM_TRUSTED_PROXIES` | empty | Passed to the container as an override of [`trusted_proxies`](#web-ui-and-sessions). Empty means `config.yaml` (or the loopback default) applies. |
| `COMPOSE_PROFILES` | (written with `DGXMM_WITH_LITELLM=1`) | `litellm` | Makes a plain `docker compose up -d` start the bundled LiteLLM too. A stray shell value is never adopted. |

Paths must be absolute and must not contain whitespace, a backtick, or any of `:` `"` `'` `\` `$` `#`.
Binds must be IPv4.

Variables and flags that only affect one installer run:

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Written to `secrets/hf_token` when that file is missing or empty. |
| `LITELLM_URL` | Written as `litellm_url` when `config.yaml` is generated. With `--with-litellm`, a URL other than the bundled one also means `litellm_public_url` is not set. |
| `LITELLM_MASTER_KEY` | Written to `secrets/litellm_master_key` when that file is missing or empty. When DGXMM is wired to the bundled LiteLLM, it also seeds `secrets/litellm_bundled_master_key` if that is missing. |
| `--dry-run` / `-n` | Print what would be done and change nothing. |
| `--allow-root` | Allow running the installer as root (not recommended: DGXMM's files should belong to a normal user). |
| `DGXMM_PULL_ENGINES=1` | Pull the engine images listed in `config.yaml` before starting. |
| `DGXMM_HEALTH_TIMEOUT` | Seconds to wait for the container to become healthy (default 300). |
| `DGXMM_LITELLM_HEALTH_TIMEOUT` | Same for the bundled LiteLLM (default 300). `DGXMM_LITELLM_MAX_RESTARTS` (default 3) fails sooner on a crash loop. |
| `DGXMM_DECIDER_TIMEOUT` | With `--with-decider`: seconds to wait for the decider to become healthy (default 900, at most 10200; its first start compiles GPU kernels for a few minutes). |
| `--purge-decider-model` | With `uninstall --purge`: also delete the snapshots of every supported decision model (`<HF cache>/hub/models--vllm-sr--d3-mini`, `models--vllm-sr--d3-nano`) and the shared-store blobs (`<HF cache>/hub/blobs`) only they use. Other models, and blobs another model also uses, are kept. |
| `DGXMM_LOG_LINES` | Lines shown by `./install.sh logs` (default 200). |

When the installer generates `config.yaml`, it puts your HF cache, models directory, home directory
and network into the copy of `config.example.yaml`. If you change `DGXMM_HF_CACHE`, `DGXMM_MODELS_DIR`
or `DGXMM_NETWORK` in `.env` later, update `hf_cache.host`, `gguf_roots[0].host` and `docker_network` in
`config.yaml` to match. Engine containers mount the *host* paths from `config.yaml`. The installer warns
if the networks differ.

## Secrets

`secrets/` has mode 700 and every file in it has mode 600. Compose mounts `hf_token`,
`litellm_master_key`, `dgxmm_ondemand_key` and `oidc_client_secret` into the `dgxmm` container at
`/run/secrets/`.

| File | Meaning |
|---|---|
| `secrets/hf_token` | Hugging Face token (optional; needed for gated repos). |
| `secrets/litellm_master_key` | Master key of the LiteLLM DGXMM registers models with (optional). |
| `secrets/dgxmm_ondemand_key` | Key LiteLLM uses to call DGXMM's [on-demand gateway](../architecture/ondemand.md). The installer generates it when it is missing or empty (`sk-` and 48 random letters and digits, about 286 bits) and never overwrites it. DGXMM puts it on the LiteLLM entries of on-demand deployments; you never enter it anywhere. |
| `secrets/oidc_client_secret` | The OpenID Connect client secret for [single sign-on](../user-guide/single-sign-on.md) (optional). The installer creates it empty; DGXMM reads it at every sign-in, so a new secret written into it applies without a restart. Empty: a public client (PKCE only). |
| `secrets/litellm_bundled_master_key` | Bundled LiteLLM only: its own master key. Mounted only into `dgxmm-litellm`. |
| `secrets/litellm_salt_key` | Bundled LiteLLM only: encrypts credentials in its database. Never change it once set. |
| `secrets/litellm_db_password` | Bundled LiteLLM only: Postgres password. |
| `secrets/admin_password` | The first-run `admin` password, saved by the installer. DGXMM never reads it, and it goes stale once you change the password. |

The HF token is read when DGXMM starts: after you change it, restart DGXMM with `docker restart dgxmm`.
The LiteLLM master key file is read each time DGXMM talks to LiteLLM, so a key written in place (for
example with `printf … > secrets/litellm_master_key`) applies without a restart; an editor that
replaces the file needs `docker restart dgxmm`. A master key entered on the LiteLLM page is stored in DGXMM's database (`data/`) instead and
wins over the file (see [runtime settings](litellm.md#runtime-settings-and-precedence)).

**Rotating the on-demand key.** Write a new key into the existing file; no restart is needed:

```bash
printf 'sk-%s' "$(head -c 4096 /dev/urandom | LC_ALL=C tr -dc A-Za-z0-9 | head -c 48)" > secrets/dgxmm_ondemand_key
```

The gateway checks the file at most once a second and re-reads it when it changed, so the old key gets
401 from then on. When it picks up the new key it starts a LiteLLM sync, which notices the new key by its
fingerprint and re-registers every on-demand entry with it, adding the new entry before it removes the
old one. LiteLLM itself may keep using the old key until it reloads its model list, so expect up to about
a minute of 401 (and, through LiteLLM, its own error or retry) for on-demand models right after a
rotation. Requests with the previous key are not counted as failed attempts, and the rotation clears the
failed-attempt counts, so LiteLLM's address is not rate limited (429) meanwhile. Rotate when a minute of
errors is acceptable. Write
the file in place as above: the file is mounted on its own, so an editor that replaces it is not seen
inside the container until `docker restart dgxmm`. If the file goes missing or empty, the gateway refuses
every request (503 `ondemand_not_configured`) until a key is back.

## Next steps

- [Models, engines and memory](models-and-engines.md)
- [Networking and reverse proxy](networking.md)
