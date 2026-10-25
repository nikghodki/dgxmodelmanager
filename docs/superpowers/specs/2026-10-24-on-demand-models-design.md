# On-demand models — design

Date: 2026-10-24 · Status: agreed in conversation; awaiting written-design review.

## Intent

Today a DGXMM deployment is either running and holding its memory, or stopped and unreachable. An operator runs several models on one DGX Spark, which has 128 GB of unified memory, and wants them to come and go with traffic:

- a model starts when someone asks for it;
- it gives its memory back when nobody is using it;
- it never takes the box down with an out-of-memory crash.

Clients such as chat apps and agents on other machines call models by name through LiteLLM. They must not have to change anything, and a request should never fail just because its model was asleep.

## Goals

1. **Per-deployment mode: `on_demand` or `always_on`.**
   - Chosen in the deploy wizard and changeable later on the deployment page.
   - All on-demand models are equal: no priorities between them.
2. **Start on first request.** A request for an on-demand model that is down starts its deployment. The request waits on its own open connection, then gets the normal answer.
3. **Idle sleep.**
   - An on-demand model with no request in flight and none arriving for its idle timeout is put to sleep. "Sleep" means the deployment is stopped and its memory freed.
   - The idle timeout defaults to **5 minutes** and can be changed per deployment.
4. **Making room.** If the requested model doesn't fit in free memory:
   - On-demand models that have been idle for their timeout are put to sleep to make room, and the request proceeds.
   - If the memory is held by models that are serving, or were used within their idle timeout, the request is **queued**. It proceeds as soon as enough of them have gone idle for their timeout and been put to sleep.
   - Always-on models are never put to sleep for this.
5. **Queued notice.** The caller learns its message is queued, on the same request:
   - **Streaming request** (`"stream": true`): DGXMM sends a first visible text chunk right away, then streams the real answer into the same response once the model is serving:

     > ⏳ Your message is queued — waiting for memory to free up for `<model>`. It will start automatically.
   - **Non-streaming request:** the connection is held open. The final answer is returned normally. A one-line queued notice is prepended to the message content only if the request actually waited in the queue (not if it only waited for a start).
6. **Queue limit.**
   - A request that has waited **15 minutes** without its model serving gets a clear error on the same connection: an OpenAI-style error body, HTTP 503 with `Retry-After`, "model `<name>` is busy; try again later".
   - For a streaming request that already sent the queued chunk, the error is a final SSE error event.
7. **Never OOM.** Every start goes through the existing fit check, plan and startup gate. Nothing is started that the memory budget says doesn't fit, and nothing outside the rules above is ever stopped.
8. **Transparent to clients.** The LiteLLM model name stays the same in both modes, so clients change nothing.

## Non-goals

- Smart prediction or pre-warming. Starts and stops follow traffic only.
- Priorities or preemption among on-demand models, and cutting off in-flight requests. A model is only put to sleep when it has nothing in flight.
- Routing a request to a different model than the one asked for.
- The built-in decision model: it stays an always-on system deployment, because clients don't call it through LiteLLM.
- Non-OpenAI endpoints beyond what vLLM and llama.cpp already serve: chat/completions, completions, embeddings, models.

## Anti-goals (failure even if it "works")

- **An OOM, or a start that pushes other deployments into failure.**
- **A request silently dropped or answered by the wrong model.**
- **A model stopped while it is serving.**
- **Flapping:** models repeatedly starting and stopping under steady light traffic.
- **Existing always-on deployments changing behaviour** without an administrator switching them.

## Constraints

- **DGX Spark (GB10) unified memory, 128 GB, shared by everything.**
  - vLLM's own sleep mode keeps the weights in the same shared memory at level 1, and has crashed with CUDA graphs on the pinned image. So "sleep" here is a DGXMM stop and start.
  - Restart takes about 1–3 minutes for a 27B model: load, compile and warm-up. That is acceptable, and is why requests wait on the open connection.
- **Request path:** client → (an edge reverse proxy) → nginx → LiteLLM → model. Every hop must tolerate ≥ 15 minutes plus generation time:
  - **LiteLLM:** `timeout: 600` today, which must be raised. Per-model `timeout`/`stream_timeout` on the DGXMM-registered entries is preferred over a global change.
  - **nginx gateway** (`/v1/` → LiteLLM): it lives in a config file outside the repo, so document it.
  - **Edge reverse proxy (e.g. Caddy on another host):** configured outside the repo, so document it.
  - **The DGXMM gateway** itself.
- **Proxy requirements:** the existing DGXMM principles apply. Rules first, auditable, and fail safe; unlike the advisory decision features, this one does act, but only within the rules above. The proxy must add negligible latency when the model is ready, and must stream byte-for-byte, including SSE, without buffering.
- **The repo is public:** no hostnames or other environment-specific values in code, docs or fixtures.

## Decided "how" (proposals marked "Mine")

- **DGXMM on-demand gateway.**
  - For on-demand deployments, the LiteLLM entry's `api_base` points at a DGXMM-internal endpoint, e.g. `http://dgxmm:<port>/ondemand/<deployment-id>/v1`, instead of the engine container.
  - The entry stays registered whether the deployment is up or down.
  - The gateway counts in-flight requests and last-request time per deployment, starts or waits as needed, then proxies to the engine container.
  - Always-on deployments keep today's direct registration.
  - (Mine) The gateway listens only on the Docker network, not on a host port. Requests must carry the LiteLLM-to-engine key DGXMM already uses (`api_key` on the entry); the gateway rejects requests without it.
- **Idle detection** uses the gateway's own counters as the source of truth. (Mine) The engine's `num_requests_running`/`waiting` metrics are used as a cross-check, so requests that bypass the gateway, such as direct calls to the engine port, still count as activity and prevent a stop.
- **Stop and start** use the existing deployer `stop()`/`start()`/startup gate/fit planner, with a system actor `ondemand` in events and the audit log.
- **Defaults:**
  - New deployments default to `on_demand` (changed later: the API defaults to `always_on`, and the deploy wizard preselects on demand only when LiteLLM is enabled and synced).
  - (Mine) Existing deployments are migrated as `always_on`, so nothing changes for them until switched.
  - Idle timeout 5 min; queue limit 15 min. Both are global settings with per-deployment overrides.
- **(Mine) Queue order** is first come, first served across all on-demand models.
  - A queued request for model B waits until enough idle on-demand memory frees.
  - Requests for a model that's already starting join its wait rather than the queue.
- **(Mine) Anti-flap:** a model started for a request stays up for at least its idle timeout after its last request. A model put to sleep to make room is only stopped if it has nothing in flight at the moment of stopping. That is checked inside the deployer lock.
- **UI:**
  - **Wizard:** a mode selector, "Always on" / "On demand (start on first request, sleep after N min idle)".
  - **Deployment page:** the mode and timeout, editable.
  - **Deployments list and dashboard:** badges such as "Sleeping (on demand)", "Starting for a request", "Ready", "N queued".
  - **Events:** "Started for a request", "Put to sleep after 5 min idle", "Put to sleep to make room for X", "Queued request timed out".

## Left to the builder

- **Exact gateway placement:** inside the existing FastAPI app, or a separate small ASGI app or port in the same container, plus the streaming proxy implementation (httpx streaming). It must handle client disconnects: a caller that hangs up while queued is removed from the queue.
- **How the queued notice is produced as a valid SSE chunk** for chat/completions and completions. It must be a well-formed `chat.completion.chunk` with the deployment's served model name.
- **Data model:**
  - `deployments.mode`, `idle_timeout_s` and queue settings, via additive migrations;
  - in-memory queue and counters;
  - what survives a DGXMM restart. A queued request's connection is lost on restart anyway, so no persistence is needed.
- **The make-room algorithm:** which idle models to stop when several are idle (e.g. least recently used first), and with how much margin.
- **LiteLLM registration changes, plus timeout and retry settings.** LiteLLM must not retry or fail over a held request: `num_retries: 0` for these entries, and confirm LiteLLM doesn't treat a long first byte as a failure and put the deployment on cooldown.
- **Health checks.** DGXMM's own probes, LiteLLM health checks and the watchdog must not count as traffic, and must not wake a sleeping model.
- **Tests:** concurrency, the 15-minute timeout (with an injectable clock), streaming pass-through fidelity, disconnects, and memory-pressure scenarios with fake engines. Plus a live check with a small model and a large production deployment.
- **Docs:** user guide (modes), configuration, architecture (gateway, data flow diagram), and troubleshooting, including the timeouts to raise on proxies outside DGXMM. Also the CHANGELOG.

## Success looks like

- **Cold start through the real path:** a client calls an on-demand model that has been asleep, through the real edge proxy → nginx → LiteLLM path, and gets its answer, slowly the first time, with no client change.
- **Queued request:** with two on-demand models that can't both fit, a request for the second while the first is busy shows the queued notice when streaming. It's answered automatically about 5 minutes after the first goes idle.
- **Idle sleep:** an idle on-demand model is stopped after 5 minutes and its memory returns to the budget.
- **No collateral damage:** no OOM, no failed deployments, and always-on models are untouched.

## Calls made after the written-design review (mine)
- **Traffic definition:** only inference requests count as traffic and can wake a model: `POST` to `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`, plus the engine-specific equivalents.
  - `GET /v1/models` is answered by the gateway from the deployment record, without waking the model.
  - Health or probe paths (`/health`, `/ping`, `/metrics`, any `GET`) never count and never wake. This covers LiteLLM health checks, which only run when someone calls LiteLLM's `/health`.
  - DGXMM's own probes go straight to the engine container, never through the gateway, so they are unaffected.
- **Mode switch:** switching mode never drops requests. DGXMM re-registers the LiteLLM entry, adding the new `api_base` before removing the old one.
  - In-flight requests finish on the path they started on: the gateway keeps proxying its open requests, and direct requests are unaffected.
  - **On-demand → always-on:** starts the deployment if it is stopped, through the normal fit check. If it doesn't fit, the switch is refused with the usual plan options.
  - **Always-on → on-demand:** the idle timer starts from the switch time.
