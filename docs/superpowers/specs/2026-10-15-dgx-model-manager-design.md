# DGX Model Manager (DGXMM) — Design Spec

- **Date:** 2026-10-15
- **Status:** Approved in brainstorming; awaiting written-spec review
- **Host:** DGX Spark `dgx-host` (NVIDIA GB10, aarch64, 121.7 GiB unified memory, 3.7 TB NVMe)
- **Repo:** `~/workspace/dgx-model-manager` on the DGX

## 1. Purpose

A web app, reachable over the network, for managing LLM serving on one DGX Spark. You use it to:

1. See which model weights are on disk (Hugging Face cache and GGUF directories).
2. See which models are deployed, and whether each one is ready to serve.
3. See memory use (host and per deployment) and serving metrics (tok/s, TTFT, ITL, KV-cache usage, queue depth, prefix-cache hit rate, spec-decode acceptance).
4. Deploy, stop and switch models, with **vLLM or llama.cpp** picked per deployment.
5. Check memory before every deploy so the box is never pushed into the kernel OOM killer.
6. See which combinations of models can run side by side in the memory currently free.
7. Browse models available to download from the Hugging Face Hub, with vLLM / llama.cpp support and a fit badge, and download them.
8. Get tuning advice for the best throughput and latency, based on rules and backed by measured benchmarks.

### Success criteria

- You open the UI, see what is on disk and what is serving, choose a model or a combination, and get a verdict like "fits, X GB headroom, expected ~Y tok/s". One confirmation then deploys it.
- No deploy started by DGXMM pushes `MemAvailable` below the configured safety margin. Deploys that would are refused, and the user gets a plan that would fit.
- During a switch, clients calling LiteLLM (`:4000`) under a stable alias see no change other than the downtime the swap itself causes.

### Current state found on the box (2026-10-15)

- vLLM containers using `vllm/vllm-openai:pinned-2026-08-27`:
  - `dmm-chat-inference` (Qwen3.8-27B-NVFP4 + DFlash, `gpu-memory-utilization 0.50`) exited with **137**, probably OOM.
  - `vllm-qwen` (Qwen3.6-35B-A3B-NVFP4, 0.55, sleep mode) is in state Created.
- Other services:
  - LiteLLM `:4000` with Postgres and Redis.
  - nginx gateway `:80`.
  - kokoro-tts, livekit, opensora-studio.
  - Ad-hoc Python experiment processes, one of which used ~21 GB.
- The Hugging Face cache holds ~15 models, ~470 GB in total.
- `nvidia-smi` reports GPU memory as **Not Supported** (unified memory). Memory accounting has to use `/proc/meminfo` and cgroups.
- The DGX runs Tailscale (`192.0.2.82`), and the tailnet includes other people's devices. Being on the network must not be enough to get access.
- An earlier "model-manager-v2" systemd unit was crash-looping on a deleted directory. The user is disabling it. DGXMM must not reuse its `dmm-` names.

## 2. Decisions

| Topic | Decision |
|---|---|
| Engines | vLLM and llama.cpp, chosen per deployment. Pluggable engine interface. |
| Access | LAN and tailnet. App-managed username/password with roles. |
| LiteLLM | DGXMM registers and deregisters models automatically through the LiteLLM admin API (DB-backed, no restart). |
| Not enough memory | Show an eviction/reduction plan and require confirmation. Never evict silently. |
| Tuning advice | Rules engine plus an on-demand benchmark runner. Measured numbers take priority over rules. |
| Model fit data | **llmfit** for catalog discovery, scoring and first tok/s estimates. DGXMM's own calculator is the authority for allowing or refusing a deploy. |
| Stack | FastAPI (Python 3.12) backend, React + TypeScript + Vite + Tailwind/shadcn frontend, SQLite. One Docker Compose service on the DGX. |
| Location | Code and runtime live on the DGX only. |

## 3. Architecture

```
Browser ──HTTP(S) LAN/tailnet──> nginx-gateway :80  location /dgxmm/ ──> dgxmm :8090
                                                                          │
┌─────────────────────────────────────────────────────────────────────────┴──┐
│ dgxmm (FastAPI + static React build)                                        │
│  auth       users, roles, sessions, CSRF, rate limiting                     │
│  inventory  HF cache scan + GGUF dirs → ModelArtifact                       │
│  catalog    HF Hub search + llmfit + engine arch support → fit badge        │
│  downloads  background hf_hub_download jobs w/ progress + resume            │
│  hostmem    /proc/meminfo, cgroup memory.current, NVML util/temp/power      │
│  memory     per-engine estimator + calibration store                        │
│  planner    fit check, eviction/reduction plans, co-location combos         │
│  engines/   vllm.py, llamacpp.py  (Engine interface)                        │
│  deployer   sole Docker writer; deployment state machine; locks             │
│  metrics    5 s scraper → ring buffer → SSE; 1 min rollups → SQLite         │
│  advisor    rules engine + benchmark runner                                 │
│  litellm    register/deregister via admin API                               │
│  watchdog   MemAvailable thresholds → alert / optional pre-emptive stop     │
│  reconcile  startup sync of DB ⇄ Docker                                     │
└──────┬────────────────┬──────────────────┬───────────────────┬──────────────┘
  docker.sock      /proc (ro)      ~/.cache/huggingface     LiteLLM :4000
  managed containers labeled dgxmm.managed=true, named dgxmm-<name>, network `gateway`
```

### 3.1 Engine interface

```python
class Engine(Protocol):
    name: Literal["vllm", "llamacpp"]
    def supports(self, artifact: ModelArtifact) -> SupportResult: ...
    def estimate_memory(self, artifact: ModelArtifact, cfg: EngineConfig) -> MemoryEstimate: ...
    def build_container_spec(self, dep: Deployment) -> ContainerSpec: ...
    async def health(self, dep: Deployment) -> HealthResult: ...      # /health + 1-token completion
    def parse_metrics(self, prom_text: str) -> EngineMetrics: ...
    def tuning_rules(self) -> list[Rule]: ...
    async def sleep(self, dep) -> None: ...   # optional; NotSupported if absent
    async def wake(self, dep) -> None: ...
```

Adding an engine later (e.g. SGLang) means adding one module that implements this interface. No other module changes.

### 3.2 Data model (SQLite)

- `users(id, username, pw_hash, role[admin|viewer], must_change_pw, created_at)`
- `sessions(id, user_id, csrf_token, expires_at, ip)`
- `artifacts(id, source[hf|gguf], repo_id, revision, path, format[safetensors|gguf], quant, size_bytes, config_json, scanned_at)`
- `deployments(id, name, engine, artifact_id, config_json, state, container_id, port, priority, litellm_model_ids, alias, created_by, created_at, updated_at)`
- `deployment_events(id, deployment_id, ts, from_state, to_state, message, log_tail)`
- `memory_calibration(engine, repo_id, config_hash, estimated_bytes, measured_bytes, ts)`
- `benchmarks(id, deployment_id, config_hash, concurrency, input_len, output_len, ttft_p50, ttft_p95, itl_p50, tps_total, tps_per_req, ts)`
- `metric_rollups(deployment_id|host, minute, json)`. Kept for 30 days.
- `downloads(id, repo_id, revision, files_json, bytes_done, bytes_total, state, error)`
- `audit_log(id, user_id, action, target, ts, detail)`

## 4. Memory model and OOM safety

### 4.1 Host budget

Recomputed every 2 s:

```
usable        = MemTotal − safety_margin (default 8 GB) − os_reserve (default 4 GB)
committed     = Σ reserved_bytes(managed deployments in starting|loading|ready|sleeping*)
external      = (MemTotal − MemAvailable) − Σ actual_bytes(managed deployments)
free_for_new  = usable − committed − max(external, 0)
```

- \* A sleeping vLLM deployment counts only its measured residual memory (level 1 sleep keeps nothing on the GPU, and its weights are offloaded to CPU memory, which is the **same pool** here). So sleeping deployments count at their measured cgroup `memory.current`.
- Page cache is reclaimable and is counted as free, because `MemAvailable` already does that. A deployment that is starting counts its full `reserved_bytes` from the moment it starts, because weights load through page cache.
- `actual_bytes` per container is read from the cgroup v2 `memory.current`.
- Processes not managed by DGXMM show in the UI as **external**, with the top consumers by RSS. They are never touched.

### 4.2 Per-deployment estimate

`reserved = weights + draft_weights + kv_cache + activations + runtime_fixed`

- **weights:** actual byte size of the safetensors/GGUF files on disk (handles mixed precision, NVFP4 scales and MoE). If there is no local copy, use the total file size from Hugging Face Hub metadata.
- **draft_weights:** file size of the speculative draft model (DFlash/EAGLE), if one is configured. MTP heads are already inside the weights.
- **kv_cache:** computed for each layer from `config.json`. GGUF uses the equivalent metadata keys.
  - Full attention: `2 × kv_heads × head_dim × bytes(kv_dtype) × tokens`.
  - Sliding-window layers: `tokens = min(context, window)`.
  - MLA: `(kv_lora_rank + qk_rope_head_dim) × bytes × tokens`.
  - Linear-attention / Gated-DeltaNet / Mamba layers: fixed state per sequence, no per-token cost.
  - vLLM: `tokens = max_model_len × max_num_seqs`, capped by what the memory share allows (vLLM fills whatever share it is given). The KV share is therefore something the user picks: the planner offers "full context × N sequences" or "K tokens of KV pool".
  - llama.cpp: `tokens = ctx_size` (shared across `parallel` slots).
- **activations:** vLLM ≈ `f(max_num_batched_tokens, hidden, intermediate)` plus a CUDA-graph allowance (default 1.5 GB). llama.cpp ≈ compute buffer from `n_batch/n_ubatch`.
- **runtime_fixed:** vLLM 3.0 GB (CUDA context, PyTorch, workers). llama.cpp 0.5 GB. Both are configurable.

### 4.3 vLLM utilization conversion

vLLM's `--gpu-memory-utilization` is a fraction of **total** device memory. DGXMM never asks the user for it. It computes:

```
gpu_memory_utilization = reserved_bytes / MemTotal     (rounded up to 0.01)
```

The value is also checked against `free_for_new` before launch. This prevents the failure mode where two containers at 0.50 plus external processes add up to more than physical memory.

### 4.4 Calibration

- Once a deployment reaches `ready`, record its measured `memory.current`. For vLLM, also parse the KV-cache size it reports in its logs.
- Store `(estimated, measured)` in `memory_calibration`.
- Later estimates for the same `(engine, repo, config_hash)` use the measured value plus 5%. For the same repo with other configs, scale the estimate by the observed ratio.
- The UI shows confidence: `estimated`, `calibrated` or `measured`.
- Exit code 137 during startup or serving is recorded as `likely OOM`. That combination then gets an extra 10% margin until it next starts successfully.

### 4.5 Enforcement points

1. **Pre-flight:** `estimate > free_for_new` → refuse. Return a plan (§5.3).
2. **During startup:** while starting or loading, if `MemAvailable < safety_margin` → stop that container and mark it `failed: aborted (memory)`.
3. **Watchdog (running):**
   - `MemAvailable < 1.5 × margin` → warning banner and event.
   - `< margin` → critical alert. If `auto_protect=true` (default **false**), stop the managed deployment with the lowest `priority`.
4. **Global memory lock:** pre-flight and container start happen under one lock, so two concurrent deploys can't both claim the same free memory.

### 4.6 Running models side by side

- The planner looks at downloaded models, each with its preferred config, or presets: "default", "long-context" and "high-concurrency".
- It enumerates combinations of up to 4 deployments whose `Σ reserved ≤ usable − external`.
- Ranking prefers more headroom first, then more models, then models the user has marked as favourites.
- It returns the top 10. Example: `Qwen3.6-35B-A3B-NVFP4 @128K×8 + Qwen3-8B @32K×4 → 71 GB, 13 GB spare`.

## 5. Engines, deploy and switch

### 5.1 vLLM

- **Image:** configurable, default `vllm/vllm-openai:pinned-2026-08-27`.
- **Typed config:**
  - `model`, `served_model_name`, `max_model_len`, `kv_cache_dtype`
  - `max_num_seqs`, `max_num_batched_tokens`
  - `enable_prefix_caching`, `enable_chunked_prefill`
  - `tool_call_parser`, `reasoning_parser`, `enable_auto_tool_choice`
  - `speculative_config{method, model, num_speculative_tokens}`
  - `enable_sleep_mode`, `trust_remote_code`, `safetensors_load_strategy`
  - `extra_args`
- **`extra_args`:** only flags on an allowlist. Each value is checked against a per-flag regex. The container command is passed as a list, never through a shell.
- **Mounts:**
  - HF cache → `/root/.cache/huggingface`
  - `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.triton` → their usual paths, so compile caches survive restarts.

### 5.2 llama.cpp

- **Image:** configurable, default `ghcr.io/ggml-org/llama.cpp:server-cuda`. **Verify during planning** that an arm64 + CUDA 13 build exists. If not, DGXMM ships a `docker/llamacpp.Dockerfile` that builds `llama-server` from a pinned commit for `sm_121`.
- **Typed config:**
  - `gguf_path` (must be under an allowlisted root)
  - `n_gpu_layers` (default all)
  - `ctx_size`, `parallel`
  - `cache_type_k`, `cache_type_v`
  - `flash_attn`, `n_batch`, `n_ubatch`
  - `metrics` (always on)
  - `extra_args` (allowlisted)
- **GGUF roots:** the HF cache plus configurable directories, by default `~/models/gguf`.

### 5.3 Deployment state machine

```
planned → [downloading] → starting → loading → ready ⇄ sleeping
               ↘ failed                  ↘ failed       ready|sleeping → stopping → stopped
```

- `ready` requires both: `/health` returns 200 **and** a 1-token completion succeeds.
- Startup timeout is configurable, default 15 min. When it runs out, or the container exits, the deployment goes to `failed` and `deployment_events` captures the exit code and the last 200 log lines.
- One `asyncio.Lock` per deployment. Actions on a deployment that is busy return 409.
- Ports are allocated from the pool 8100–8199. Containers join the `gateway` network and are labeled with `dgxmm.managed=true`, `dgxmm.deployment_id=<id>`.

### 5.4 Not-enough-memory plan

The planner returns a ranked list of options. Each option has its effect on memory and a short explanation:

- Sleep or stop specific managed deployments (lowest priority first).
- Reduce context, or switch the KV dtype to fp8.
- Reduce `max_num_seqs`.
- Use a smaller quantization that is on disk or downloadable.

The user picks one and confirms. External processes are listed as "cannot be freed by DGXMM".

### 5.5 Switch flow (A → B)

1. Plan B and run pre-flight. If A has to go, show the plan and wait for confirmation.
2. Sleep A if it is vLLM with sleep mode, otherwise stop it.
3. Start B and wait for `ready`.
4. LiteLLM: register B, including the shared alias if one is configured. Then deregister A.
5. If B fails: wake or restart A, restore A's LiteLLM entries, and report the failure. The UI streams each step over SSE.

### 5.6 Adopting existing containers

"Adopt" works on a container that is not managed by DGXMM (e.g. `vllm-qwen`):

1. Read its image, command, mounts and network.
2. Turn those into a typed config. Flags it doesn't recognise go into `extra_args`, if they are on the allowlist; otherwise adoption is refused.
3. Create a managed deployment. On the next start it is recreated as `dgxmm-<name>`.

The original container is left alone until the user removes it.

## 6. LiteLLM integration

- Uses the admin API at `http://litellm-proxy:4000` (`/model/new`, `/model/delete`, `/model/info`). Model entries persist in Postgres.
- Each entry DGXMM creates carries `model_info.dgxmm_id=<deployment_id>`. DGXMM only deletes entries with that marker. Routes defined in YAML are never touched, and LiteLLM is never restarted.
- `litellm_params`:
  - `model: openai/<served_name>`
  - `api_base: http://dgxmm-<name>:<port>/v1`
  - `api_key: none`
- Optional alias: a stable public name, e.g. `default-chat`, pointed at whichever deployment currently has it.
- The master key comes from the Docker secret `litellm_master_key`.
- If LiteLLM can't be reached, the deploy still succeeds and registration is retried with exponential backoff (max 10 min). The UI shows `LiteLLM: pending`.
- DGXMM reads per-model request and error counts from LiteLLM for the metrics view.

## 7. Inventory, catalog and downloads

- **Inventory:**
  - Scans `~/.cache/huggingface/hub` (`scan_cache_dir`) and the GGUF roots on startup, then every 5 min, plus a manual refresh.
  - Each artifact records size, format, quant (detected from `config.json` `quantization_config` / name / GGUF file type), architecture, and whether vLLM / llama.cpp support it.
  - Datasets and non-LLM models (e.g. whisper, BERT) are listed under "Other".
- **Catalog:**
  - Sources:
    - Hugging Face Hub search, filtered by `pipeline_tag=text-generation` or GGUF library.
    - `llmfit --json --memory <free_for_new>G --max-context <ctx> recommend|search`.
  - Each entry shows:
    - Downloads, likes and size.
    - Support: vLLM supports the architecture if it is in the pinned image's model registry (extracted once per image tag by running the image and listing the registry). llama.cpp supports it if a GGUF exists.
    - llmfit fit, quality, speed and context scores, and `estimated_tps`.
    - A **DGXMM fit badge**: `fits now`, `fits after evicting …` or `too big`.
    - Related variants: NVFP4 / FP8 / GGUF versions of the same base model.
  - llmfit results are cached for 10 min. If llmfit fails, the catalog falls back to HF search only.
  - When llmfit's and DGXMM's memory figures differ by more than 20%, a warning is shown.
- **Downloads:**
  - Background job using `huggingface_hub` with resume, a progress bar and cancel.
  - Before starting, checks that free disk minus the repo size is at least 50 GB.
  - Gated repos use the HF token from the Docker secret `hf_token`.
  - Admin only.

## 8. Metrics

- **vLLM** `/metrics`: running/waiting requests, `kv_cache_usage_perc`, prompt/generation token counters (→ tok/s), TTFT/ITL/E2E histograms (→ p50/p95), prefix-cache hit rate, spec-decode acceptance.
- **llama.cpp** `/metrics` + `/slots`: equivalent counters, slot states.
- **Host:** memory breakdown (§4.1), GPU util/temp/power from NVML, disk free.
- **LiteLLM:** per-model requests and errors.
- **Pipeline:**
  - Scrape every 5 s into an in-memory ring buffer (last 1 h), pushed to the UI over SSE at `/api/stream/metrics`.
  - 1 min rollups stored in SQLite for 30 days.

## 9. Advisor

### 9.1 Rules

Each rule is a pure function `(artifact, config, recent_metrics, benchmarks) → Advice?`. `Advice{title, rationale, expected_effect, confidence[rule|estimated|measured], patch: EngineConfig delta}`.

Initial set:

1. BF16/FP16 weights and an NVFP4 variant exists → suggest NVFP4 (~3.5× less weight memory, ~2× decode on Blackwell).
2. `kv_cache_dtype=auto` and context > 32K → fp8 KV.
3. A compatible DFlash/EAGLE/MTP draft exists and concurrency is usually ≤ 2 → enable spec decode.
4. Spec-decode acceptance < 50% → reduce `num_speculative_tokens` or disable.
5. `num_requests_waiting > 0` for > 50% of the last 15 min and KV usage < 70% → raise `max_num_seqs`.
6. KV usage > 95% with preemptions → lower context or seqs, or use fp8 KV.
7. Reserved KV much larger than the peak KV used over 24 h → shrink the reservation so another model fits.
8. Prefix-cache hit rate < 2% over 24 h → note it isn't helping (low-priority info).
9. MoE with few active parameters (≤ 5B) → raise `max_num_seqs` / batched tokens.
10. Single user, and a GGUF Q4_K_M/Q5_K_M exists → llama.cpp may give more tok/s for one stream. Suggest benchmarking both.
11. `max_num_batched_tokens` < 8192 with long prompts → raise it for TTFT.
12. Chunked prefill off with mixed long and short prompts → enable it.
13. llama.cpp: `n_gpu_layers` below all layers → offload all (unified memory, no penalty).
14. llama.cpp: flash attention off → enable it.
15. llama.cpp: `cache_type_k/v=f16` with a large `ctx_size` → q8_0 KV.

### 9.2 Benchmark runner

- Admin-triggered, one benchmark at a time globally.
- Load: random token prompts, by default input 1024 and output 256, at concurrency [1, 4, 8, 16], 60 s per level or 32 requests, whichever comes first.
- Uses async `httpx` against the deployment's OpenAI-compatible endpoint, not through LiteLLM.
- Records TTFT p50/p95, ITL p50, total and per-request tok/s. The UI shows curves of tok/s against concurrency.
- When a measured benchmark exists for the config being compared, advice cites it instead of the rule's estimate.
- **Apply** on any advice opens the switch flow with `patch` already filled in. Changes always need confirmation.

## 10. Security

- **Passwords and sessions:**
  - argon2id hashes. Server-side sessions with a 12 h sliding expiry.
  - Cookie is `HttpOnly`, `SameSite=Strict`, and `Secure` when the request came in over HTTPS (`X-Forwarded-Proto`).
  - CSRF token required on every state-changing request.
  - Login rate limit: 5/min per IP and per username. Lockout notices are written to the audit log.
- **First run:** creates `admin` with a random password printed once to the logs, with `must_change_pw=true`.
- **Roles:** `viewer` can read everything except users and secrets. `admin` can do all actions. The server checks the role on every route.
- **Docker:** only the deployer module holds the Docker client.
  - Stop/remove/start only act on containers with `dgxmm.managed=true`.
  - Inputs are validated:
    - repo IDs match `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`.
    - paths are resolved and must sit under the allowlisted roots.
    - `extra_args` are allowlisted.
  - Containers never run privileged, and only the needed mounts are added.
- **Secrets:** `./secrets/litellm_master_key` and `./secrets/hf_token` (mode 600) are mounted as Docker secrets. They never appear in logs or API responses, or in the SQLite database.
- **Network:** published only through nginx `location /dgxmm/` on the existing gateway. The container port is not published on the host. Optional `tailscale serve` setup for HTTPS is documented.
- **Audit:** every deploy, stop, switch, download, user change and login is recorded in `audit_log`.

## 11. Error handling

- **Startup reconciliation:**
  - DB says `ready` but there is no container → `failed: container missing`.
  - Managed container with no DB row → adopt it as `unknown` and surface it to the user.
  - Ports and LiteLLM entries are re-synced.
- **Engine crash:** record the exit code and log tail. 137 is labelled `likely OOM` and feeds calibration.
- **LiteLLM unavailable:** retry with backoff, show status in the UI. Deploys don't fail because of it.
- **HF Hub or llmfit unavailable:** the catalog degrades with a visible banner. Inventory and deploys are not affected.
- **Download failure:** resumable, with the error shown. Partial files go in the HF cache's `.incomplete` files.
- **SSE disconnect:** the client reconnects and gets the last N samples again.

## 12. Testing

- **Unit tests (pytest):**
  - Memory estimator against reference fixtures (stored `config.json` / GGUF headers): Qwen3-8B BF16, Qwen3.8-27B-NVFP4 + DFlash draft, Qwen3.6-35B-A3B-NVFP4 (hybrid linear attention, MoE), Gemma-4-31B (sliding window), a GGUF Q4_K_M. Each fixture has expected byte ranges.
  - Host budget maths, eviction planner, co-location enumerator, config validation and allowlists, rules engine, utilization conversion.
- **Integration tests:** fake Docker client and fake engine HTTP servers (serving canned Prometheus text and health) exercise the deploy/switch/rollback state machine, reconciliation and LiteLLM retry.
- **Security tests:** role checks on every route, CSRF enforcement, path traversal and arg injection rejected, label guard prevents touching unmanaged containers.
- **On-box validation script** (`scripts/validate_on_box.sh`, run manually on the DGX): deploys Qwen3-0.6B with each engine, compares estimate with measured memory, runs a short benchmark, then tears down.
- **UI:** Playwright smoke tests covering login, forced password change, dashboard, deploy wizard with a fit preview, a switch with a confirmation plan, and catalog search.

## 13. Packaging and operations

- `docker compose up -d` in `~/workspace/dgx-model-manager`.
- One multi-stage image: `node:22` builds the frontend, then `python:3.12-slim` (arm64) runs `uvicorn`. `llmfit` is installed with pip into the image.
- **Mounts:**
  - `/var/run/docker.sock`
  - Host `/proc` → `/host/proc:ro` (read-only; used only for `meminfo` and `<pid>/status` + `<pid>/cmdline` to list the top external memory consumers)
  - `/sys/fs/cgroup:ro`
  - HF cache (rw)
  - GGUF roots
  - `./data` (SQLite)
  - `./secrets`
- **Devices:** the NVIDIA runtime, for NVML.
- **Network:** joins `gateway`. A provided `deploy/nginx-dgxmm.conf` snippet is included into the existing nginx-gateway config.
- **Config:** `config.yaml`, covering safety margin, OS reserve, port pool, images, GGUF roots, startup timeout, `auto_protect`, scrape intervals and the LiteLLM URL. Any setting can be overridden by an env var.

## 14. Milestones

Each milestone ends with tests passing and an on-box check.

1. **M1: core and safety.** Repo scaffold, config, auth and users, inventory (HF cache), hostmem, memory estimator (vLLM), host budget, vLLM engine, deploy/stop with pre-flight, reconciliation, dashboard and inventory UI.
2. **M2: switching.** Eviction/reduction planner, switch flow with rollback, sleep/wake, LiteLLM integration, watchdog, adopt-existing, audit log.
3. **M3: llama.cpp.** Engine, GGUF inventory and estimator, image verification or local build.
4. **M4: catalog and downloads.** HF search, llmfit integration, support detection, fit badges, downloads, co-location planner UI.
5. **M5: metrics history and advice.** Scraper and SSE charts, rollups, rules engine, benchmark runner, Apply flow, calibration display.

## 15. Out of scope

Multiple hosts or multi-node tensor parallelism; SGLang, TensorRT-LLM and Ollama engines (the interface allows them later); quotas and billing; fine-tuning; public-internet exposure; SSO/OIDC.
