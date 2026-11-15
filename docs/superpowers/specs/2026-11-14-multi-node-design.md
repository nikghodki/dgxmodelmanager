# Multi-node deployments (DGX-1, DGX-2, cluster): design

Date: 2026-11-14 · Status: proposed; awaiting review.

## Intent

DGXMM manages models on one DGX Spark. The owner now has two Sparks (`dgx-1` runs DGXMM; `dgx-2` is a second
GB10) joined by a ConnectX-7 200 GbE point-to-point link with RDMA (cluster subnets 192.0.2.0/25 and
192.0.2.128/25, set up by NVIDIA Sync). They want to pick, per deployment, where a model runs:

1. **on dgx-1** (today's behaviour),
2. **on dgx-2** (a single-node deployment on the second box, still managed from dgx-1's DGXMM),
3. **across both** (one model split over both GPUs).

Measured on this hardware on 2026-11-14 (manual runs, not DGXMM):

| Setup | Decode tok/s |
|---|---|
| Qwen3.8-27B NVFP4, vLLM, one Spark | 11.6 |
| Qwen3.8-27B NVFP4, vLLM TP=2 across both (NCCL over RDMA, `NET/IB/0` + `NET/IB/1`) | 20.3 (68.6 aggregate @ 4 concurrent) |
| GLM-4.7 358B UD-Q4_K_XL (205 GB, too big for one box), llama.cpp RPC layer split | 9.5 |

## Owner requirements (2026-11-14)

1. DGXMM shows which model runs on dgx-1, on dgx-2, or on both together.
2. Deploying asks where to run it: a single DGX (dgx-1 or dgx-2) or both together as a cluster, with the
   cluster suggested when the model is too big for one DGX.
3. One LiteLLM (on dgx-1) serves every model, wherever it runs.
4. Every other feature stays the same, for cluster deployments too.
5. The DGXMM service runs on dgx-1 only.

## Goals

1. **Nodes.** A static list of nodes in `config.yaml`. With no `nodes:` key, DGXMM behaves exactly as today (one
   implicit local node). One DGXMM instance (on dgx-1) controls every node; dgx-2 runs no DGXMM. (R5)
2. **Placement per deployment.** `placement = {"kind": "single", "nodes": [id]}` or
   `{"kind": "cluster", "nodes": [head, worker...]}`. The old `node` field maps to a single placement. (R2)
3. **Placement options.** Before deploying, DGXMM returns every placement with its per-node memory need, whether
   it fits, and one recommendation: the single node with the most free memory if the model fits on one;
   otherwise the cluster; otherwise none (insufficient memory). The wizard asks the user and pre-selects the
   recommendation. (R2)
4. **Per-node memory.** Fit checks, budgets, the watchdog and the planner work per node. A cluster deployment
   reserves its per-rank share on every node it uses.
5. **Engines in a cluster.** vLLM: tensor parallel over NCCL/RDMA. llama.cpp: RPC layer split. (R2)
6. **Visibility.** The Deployments list, detail page and Dashboard show `dgx-1`, `dgx-2`, or
   `dgx-1 + dgx-2 (cluster)`; the Dashboard has one memory bar per node and a cluster deployment appears on both. (R1)
7. **One LiteLLM.** Every deployment registers in the dgx-1 LiteLLM; `api_base` is the head node's cluster IP
   (remote or cluster) or the current address (local single). Model names stay globally unique. (R3)
8. **Feature parity.** On-demand, schedules, sleep/wake, switch, priorities, watchdog, planner, decider and
   insights work for remote and cluster deployments. Where an engine cannot do something across nodes, DGXMM
   uses the nearest equivalent and the UI says so (for example, if vLLM `/sleep` does not work multi-node, a
   cluster "sleep" is a stop that frees both nodes; decided by a hardware check in the plan). (R4)

   **Hardware check (Task 14, 2026-11-15): vLLM sleep does not work across nodes.** Two GB10 nodes, one
   vLLM 0.26.1 container per node, TP=2 over NCCL/RDMA (`--nnodes 2`, `--enable-sleep-mode`,
   `VLLM_SERVER_DEV_MODE=1`), Qwen3.8-27B-NVFP4, `--gpu-memory-utilization 0.60`. MemAvailable (GB, node 1 /
   node 2), second run: loaded 36.5 / 42.5; after `POST /sleep?level=1` 89.2 / 95.7 (about 52 GB freed on each
   node; vLLM: 10.98 GiB backed up to CPU, 56 GiB discarded per rank); after `POST /wake_up` 19.0 / 24.3 (about
   17.5 GB *less* than before the sleep, on each node). The first run gave the same figures. Completions before
   the sleep were correct ("What is 17+25?" -> 42, largest planet -> Jupiter); after the wake-up the same requests
   produced garbage (the model reports a garbled question, invents unrelated tasks), also after
   `/reset_prefix_cache`. Decision: `supports_sleep` is false for every cluster
   (`cluster.SLEEP_ACROSS_NODES = False`). A cluster "sleep" stops every rank and frees every node; the API labels
   it `sleep_label: "sleep (stops all nodes)"` and `sleep_capable` is false; cluster ranks never get
   `--enable-sleep-mode` or the dev endpoints. llama.cpp has no sleep, so its clusters behave the same.
   Everything else works for clusters as for single deployments: on-demand mode (idle reap stops all ranks; a
   request starts all ranks and waits for the head's health; the gateway proxies to the head), schedules (a sleep
   schedule stops every rank, its wake starts them again), switch single <-> cluster, reconfigure (a restart of
   every rank), the watchdog and the planner (decisions about one node count only the cluster's share there;
   stopping it frees every node), and the decider (each deployment carries its placement and label, never ranks).
9. **Weights.** A deploy on a node whose HF cache lacks the artifact is refused with `weights_missing` and an
   action that copies the snapshot over the cluster link.

## Non-goals

- More than two nodes in the first cut (the design does not preclude it; tests use two).
- The decider choosing placement on its own (it may explain the recommendation; the user picks).
- Running DGXMM on more than one node, or controller failover.

## Design

**Remote Docker.** Each node has a `docker_url`. Local: `unix:///var/run/docker.sock` (today). Remote:
`ssh://<user>@<cluster_ip>` through docker-py's `use_ssh_client=True`. The key is a compose secret
(`/run/secrets/node_ssh_key`, 0600), `known_hosts` is pinned in config (no trust-on-first-use at runtime), and the
image gains `openssh-client`. One `DockerController` per node; every write keeps today's `dgxmm.managed` label guard.

**Remote memory.** `NodeProbe` keeps one long-lived, no-network, read-only probe container per remote node
(`/proc` mounted read-only at `/host/proc`) and reads `meminfo` through Docker exec every
`remote_probe_interval_s` (default 5 s). A probe older than `3 x interval` marks the node **stale**: no new
starts are admitted there (fail closed), running deployments are left alone.

**Data model.** `deployments.node VARCHAR(32)`, backfilled to the local node id (migration 41). Ports are unique
per node, not globally. Container names stay `dgxmm-<name>`; the unique deployment name stays global, so
LiteLLM names cannot collide across nodes.

**Cluster.** vLLM multiprocessing multi-node (`--nnodes/--node-rank/--master-addr/--master-port`,
`--headless` on workers), host networking, `/dev/infiniband`, `IPC_LOCK`, `memlock=-1`, `shm_size`, NCCL env
from node config (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `VLLM_HOST_IP`). The head serves the
API. Any rank exiting fails the whole deployment and stops the others. **llama.cpp:** `ggml-rpc-server`
containers on workers bound to the cluster IP, `llama-server --rpc` on the head.

**Security.** Remote engine ports are bound to the cluster IP only (point-to-point link). Engines gain an
`--api-key` (closes the review finding "engines have no API key"); LiteLLM and health checks send it. RPC/NCCL
ports are on the cluster link only. The SSH key gives root-equivalent Docker on dgx-2; documented in
`docs/security/overview.md` next to the existing Docker-socket note.

## Failure handling

| Condition | Behaviour |
|---|---|
| Remote node unreachable at deploy | `node_unreachable` 503, nothing created |
| Remote node unreachable while running | state unchanged; node shown `stale`; monitor retries with backoff; remote-transport errors never count toward the 5-errors-then-failed rule |
| DGXMM restarts | reconcile runs per node; remote managed containers matched by deployment-id label |
| Weights missing on node | `weights_missing` 409 with a `copy_weights` action |
| Cluster rank dies | whole deployment `failed`, other ranks stopped, reservations released only after every container is confirmed gone |
| Model too big for any single node | the wizard pre-selects the cluster; single options show "does not fit" |
| Model too big even for the cluster | no recommendation; deploy refused with `insufficient_memory` and the per-node shortfall |
