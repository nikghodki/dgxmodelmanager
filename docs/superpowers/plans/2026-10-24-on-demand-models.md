# On-demand models Implementation Plan

> **Note:** this records the original plan. The code and the published docs are authoritative where they differ.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deployments in mode `on_demand` start on the first inference request, are stopped after an idle timeout, make room by stopping idle on-demand models, and queue requests (≤ 15 min) when memory is held by busy models — all transparently behind the same LiteLLM model name.

**Architecture:** A per-deployment controller (`dgxmm/ondemand/controller.py`) owns in-flight counters, a FIFO wait queue and an idle reaper, and drives the existing deployer (`restart`/`stop`, fit check, startup gate). A gateway router (`dgxmm/ondemand/gateway.py`, mounted at `/ondemand/{dep_id}/v1/...`) authenticates LiteLLM with a generated key, asks the controller for a slot, then streams the request to the engine container with httpx. LiteLLM sync registers on-demand deployments permanently with the gateway URL.

**Tech Stack:** Python 3.12, FastAPI, httpx (async streaming), SQLAlchemy/SQLite (additive migration), React + TypeScript + vitest, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-24-on-demand-models-design.md`

## Global Constraints
- Modes: `on_demand` | `always_on`. New deployments default `on_demand`; existing rows migrate to `always_on`. The System decider is always `always_on` and never managed by the controller.
- Defaults: idle timeout **300 s** (`ondemand_idle_timeout_s`), queue limit **900 s** (`ondemand_queue_timeout_s`); per-deployment `idle_timeout_s` override (60–86400 s, NULL = default).
- Only `POST` to `chat/completions`, `completions`, `embeddings` counts as traffic and may wake a model; `GET models` is answered from the record; any other path/method never wakes and is proxied only when the model is ready (else 503).
- Never stop a deployment with in-flight gateway requests or with engine `num_requests_running + num_requests_waiting > 0`; never stop always-on or system deployments; every start goes through the existing fit check / startup gate.
- Queue: FIFO across all on-demand deployments; requests for a deployment already starting join its wait (not the queue).
- Queued notice (exact copy): `⏳ Your message is queued — waiting for memory to free up for {model}. It will start automatically.\n\n`
- Timeout error: HTTP 503, header `Retry-After: 60`, body `{"error": {"message": "model {model} is busy; try again later", "type": "server_error", "code": "ondemand_queue_timeout"}}`; for a stream that already sent the notice, a final SSE `data: {"error": {...}}` then `data: [DONE]`.
- Events/audit actor: `ondemand`. Event texts: `Started for a request`, `Put to sleep after {n} min idle`, `Put to sleep to make room for {name}`, `Queued request timed out`.
- Repo is public: no hostnames/IPs/usernames of the deployment environment in code, docs, fixtures. Commits as `nikghodki <92228606+nikghodki@users.noreply.github.com>`, no trailers. All work happens on the DGX; no files on the laptop.

## Review Focus
1. Client hangs up while waiting or queued → its slot/queue entry is released, no model is started only for it if nobody else waits (Task 3: `test_disconnect_while_queued_releases_entry`).
2. Burst of N concurrent first requests for a sleeping model → exactly one start, all N served (Task 2: `test_concurrent_first_requests_start_once`).
3. DGXMM restarts while an on-demand model is ready and idle → reaper resumes from "now" (no immediate stop), counters reset safely (Task 2: `test_reaper_after_restart_waits_full_timeout`).
4. Engine crashes / start fails while requests wait → all waiters get a clear 503 with the failure reason, deployment is not retried in a loop (Task 2: `test_start_failure_fails_waiters_once`).
5. Streaming pass-through fidelity: SSE chunks forwarded unbuffered and byte-identical, `[DONE]` preserved, upstream 4xx/5xx passed through unchanged (Task 3: `test_stream_passthrough_byte_identical`, `test_upstream_error_passthrough`).

---

### Task 1: Mode, settings and API

**Files:**
- Modify: `dgxmm/db.py` (Deployment: `mode: Mapped[str] = mapped_column(String(16), default="always_on")`, `idle_timeout_s: Mapped[int | None]`)
- Modify: `dgxmm/migrations.py` (append `Migration(40, "deployments: on-demand mode", _ondemand_mode)`: add columns if missing; set `mode='always_on'` where NULL)
- Modify: `dgxmm/config.py` (`ondemand_idle_timeout_s: int = Field(300, ge=60, le=86400)`, `ondemand_queue_timeout_s: int = Field(900, ge=60, le=3600)`, `ondemand_key_file: str = "/run/secrets/dgxmm_ondemand_key"`)
- Modify: `dgxmm/deployer/service.py` (`DeployRequest.mode: Literal["on_demand","always_on"] = "on_demand"`, `idle_timeout_s: int | None = Field(None, ge=60, le=86400)`; persisted on deploy; `view()` gains `mode`, `idle_timeout_s`, `ondemand` (filled by controller, Task 2: `{"inflight", "queued", "last_request_at", "phase"}` with phase ∈ `idle|starting_for_request|serving|sleeping|queued`); system deployments forced `always_on`)
- Modify: `dgxmm/api/deployments.py` (`PATCH /{dep_id}/mode` admin+CSRF+audit `{mode, idle_timeout_s?}` → calls `ctx.ondemand.set_mode(dep_id, mode, idle_timeout_s, user_id)`)
- Modify: `install.sh` (generate `secrets/dgxmm_ondemand_key` 0600 like other secrets; mounted like the others)
- Test: `tests/test_ondemand_mode.py`

**Interfaces:**
- Produces: `Deployment.mode`, `Deployment.idle_timeout_s`; `Settings.ondemand_idle_timeout_s`, `Settings.ondemand_queue_timeout_s`, `Settings.ondemand_key_file`; `load_ondemand_key(settings) -> str`.

- [ ] **Step 1: Write failing tests** — `test_new_deploy_defaults_on_demand`, `test_existing_rows_migrate_always_on` (main-era fixture DB from `tests/fixtures/db/main-7c055b7.schema.sql` + migration 40 → every row `always_on`, idempotent twice), `test_system_deployment_forced_always_on`, `test_patch_mode_admin_only_csrf_audited` (viewer 403; admin 200; audit `ondemand_mode`), `test_idle_timeout_bounds` (59 → 422, 86401 → 422), `test_installer_generates_ondemand_key` (installer test harness: file exists, mode 600, not overwritten on rerun).
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_ondemand_mode.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** the columns, migration 40, settings, DeployRequest fields, PATCH route (until Task 2 exists, route calls a method stubbed to persist mode — Task 2 replaces it), installer key.
- [ ] **Step 4: Run** the test file and full `pytest -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ondemand): deployment mode, settings, migration and API`.

### Task 2: Controller (slots, queue, make-room, reaper)

**Files:**
- Create: `dgxmm/ondemand/__init__.py`, `dgxmm/ondemand/controller.py`
- Modify: `dgxmm/context.py` (construct `ctx.ondemand = OnDemandController(ctx)`; register reaper as a background task; deployer removal/state hooks)
- Test: `tests/test_ondemand_controller.py` (fake deployer/budget, injectable clock)

**Interfaces:**
- Produces:
  - `class OnDemandController(ctx, clock=time.monotonic)`
  - `async def acquire(self, dep_id: int, *, on_queued: Callable[[], Awaitable[None]] | None = None) -> Slot` — returns when the deployment is `ready`; raises `OnDemandTimeout(model)` after `queue_timeout`, `OnDemandUnavailable(code, message)` on start failure / not on-demand / removed. `on_queued` is awaited once, the first time the request has to wait for memory (not for a start that fits).
  - `class Slot` — async context manager; `__aexit__` decrements in-flight and stamps `last_request_at`.
  - `async def set_mode(self, dep_id: int, mode: str, idle_timeout_s: int | None, user_id: int | None) -> dict` — persists; on `always_on` starts if stopped (fit-checked; raises deployer `InsufficientMemory` unchanged); triggers LiteLLM sync.
  - `def status(self, dep_id: int) -> dict` — `{"inflight": int, "queued": int, "last_request_at": float | None, "phase": str}` (used by `view()`).
  - `async def reap_once(self) -> list[int]` — stops eligible idle deployments; returns their ids.
  - `async def run(self)` — loop calling `reap_once()` every 15 s.
- Consumes: deployer `restart(dep_id, user_id=None)`, `stop(dep_id, reason=...)`, `view()`, `InsufficientMemory`; `ctx.budget` free-for-new and fit (`deployer` fit check on restart); engine metrics via `ctx.metrics` (latest `num_requests_running`/`num_requests_waiting` for the deployment; missing metrics = treat as busy only if sample < 60 s old shows activity).

Algorithm (not determined by tests alone):
```
acquire(dep):
  if dep ready: return Slot
  if a start for dep is in progress: await its readiness (same deadline)
  loop until deadline:
    try deployer.restart(dep, actor=ondemand)  # fit check inside
        → mark "starting_for_request", await ready, return Slot
    except InsufficientMemory(need):
        victims = idle on-demand ready deployments (inflight==0, engine idle,
                  now-last_request >= their idle timeout), oldest last_request first,
                  until freed >= need
        if enough: stop them (event "Put to sleep to make room for X"), retry
        else: enqueue FIFO (call on_queued once), wait for a reaper/stop/state event, retry
  raise OnDemandTimeout
```
Starts are serialized per deployment (one asyncio.Lock per dep); queue wakeups happen on any deployment stopping or the reaper freeing memory; the head of the FIFO is retried first.

- [ ] **Step 1: Write failing tests** — `test_ready_deployment_returns_slot_immediately`, `test_stopped_fits_starts_and_waits` (event `Started for a request`), `test_concurrent_first_requests_start_once`, `test_make_room_stops_idle_lru_first` (two idle, need one's memory → stops the older; event text exact), `test_busy_holder_queues_then_serves_after_idle` (clock advanced 300 s after holder's last request → holder stopped, queued request served; `on_queued` called once), `test_queue_timeout_raises_after_900s`, `test_never_stops_inflight_or_engine_busy` (engine metrics running=1 → not stopped), `test_never_stops_always_on_or_system`, `test_reaper_stops_after_idle_timeout_and_respects_override`, `test_reaper_after_restart_waits_full_timeout`, `test_start_failure_fails_waiters_once`, `test_fifo_order_across_models`, `test_set_mode_always_on_starts_or_refuses`, `test_removed_deployment_releases_waiters`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_ondemand_controller.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** `controller.py` per Interfaces + algorithm; wire into context; `view()` reads `status()`.
- [ ] **Step 4: Run** file + full suite — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ondemand): controller with start-on-request, make-room, queue and idle reaper`.

### Task 3: Gateway proxy

**Files:**
- Create: `dgxmm/ondemand/gateway.py` (APIRouter, prefix `/ondemand`)
- Modify: `dgxmm/app.py` (include router; exclude it from session auth/CSRF middleware — it uses its own key)
- Test: `tests/test_ondemand_gateway.py` (fake engine via `httpx.MockTransport` or an in-process ASGI app; fake controller)

**Interfaces:**
- Consumes: `OnDemandController.acquire`, `Slot`, `OnDemandTimeout`, `OnDemandUnavailable`; deployer `base_url(name, port)`; `load_ondemand_key`.
- Produces: routes `GET /ondemand/{dep_id}/v1/models`, `POST /ondemand/{dep_id}/v1/{chat/completions|completions|embeddings}`, `{any} /ondemand/{dep_id}/v1/{path:path}` (proxied only if ready, else 503 `ondemand_not_ready`, never wakes).

Behaviour: `Authorization: Bearer <ondemand key>` required (constant-time compare; else 401). Body cap 16 MiB. For `"stream": true`: return a `StreamingResponse` immediately; if `on_queued` fires, yield one SSE chunk `{"id": "ondemand-queued", "object": "chat.completion.chunk", "model": <served>, "choices": [{"index": 0, "delta": {"role": "assistant", "content": <notice>}, "finish_reason": null}]}` (for `completions`: `{"object": "text_completion", "choices": [{"index": 0, "text": <notice>}]}`), then stream upstream bytes unchanged. Non-stream: await, proxy, and if queued prepend the notice to `choices[0].message.content` (chat) / `choices[0].text` (completions); embeddings never modified. Client disconnect cancels the wait and releases the queue entry/slot. Upstream connect/read timeouts: connect 10 s, read none (generation can be long), pool limits per deployment.

- [ ] **Step 1: Write failing tests** — `test_rejects_missing_or_wrong_key`, `test_models_answered_without_waking`, `test_get_health_never_wakes_returns_503_when_down`, `test_stream_passthrough_byte_identical`, `test_stream_queued_notice_then_answer` (exact notice copy as first content), `test_nonstream_queued_notice_prepended`, `test_embeddings_never_modified`, `test_timeout_503_with_retry_after_and_body`, `test_stream_timeout_after_notice_sends_error_event_and_done`, `test_upstream_error_passthrough`, `test_disconnect_while_queued_releases_entry`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_ondemand_gateway.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** `gateway.py`; mount in `app.py`.
- [ ] **Step 4: Run** file + full suite — Expected: PASS.
- [ ] **Step 5: Commit** `feat(ondemand): gateway that wakes, queues and streams to the engine`.

### Task 4: LiteLLM registration

**Files:**
- Modify: `dgxmm/litellm.py` (`desired()`: on-demand deployments in any state except `failed`/removed register with `api_base = <dgxmm internal base>/ondemand/{id}/v1`, `api_key = <ondemand key>`, `timeout = 1200`, `stream_timeout = 1200`, `num_retries = 0`; always-on unchanged; mode switch = add new then remove old in the same sync; `add()` accepts optional litellm params)
- Modify: `dgxmm/config.py` (`dgxmm_internal_url: str = "http://dgxmm:8090"` — the URL LiteLLM uses to reach DGXMM on the Docker network; include any root path DGXMM is served under)
- Test: `tests/test_litellm_ondemand.py`

**Interfaces:**
- Consumes: `Deployment.mode`, `load_ondemand_key`, `Settings.dgxmm_internal_url`.

- [ ] **Step 1: Write failing tests** — `test_ondemand_registered_while_stopped_with_gateway_base`, `test_always_on_unchanged_only_ready`, `test_mode_switch_adds_before_removes`, `test_entry_params_timeout_and_no_retries`, `test_key_never_logged` (caplog contains no key).
- [ ] **Step 2: Run** — Expected: FAIL. **Step 3: Implement. Step 4: Run** file + full suite — PASS.
- [ ] **Step 5: Commit** `feat(ondemand): register on-demand models with LiteLLM through the gateway`.

### Task 5: UI

**Files:**
- Modify: `frontend/src/pages/DeployWizard.tsx` (mode radio "On demand — start on the first request, sleep after N min idle" (default) / "Always on"; idle minutes input 1–1440 shown for on-demand)
- Modify: `frontend/src/pages/DeploymentDetail.tsx` (mode + idle timeout editor → `PATCH /deployments/{id}/mode`; admin only)
- Modify: `frontend/src/components/app/deployments-table.tsx`, dashboard deployment cards (badges from `view.ondemand.phase`: `Sleeping (on demand)`, `Starting for a request`, `Ready`, `{n} queued`)
- Modify: `frontend/src/lib/types.ts` (`mode`, `idle_timeout_s`, `ondemand`)
- Test: `frontend/src/pages/DeployWizard.ondemand.test.tsx`, `frontend/src/components/app/deployments-table.test.tsx` (extend)

- [ ] **Step 1: Write failing tests** — wizard defaults to on-demand and sends `mode`/`idle_timeout_s`; always-on hides idle input; detail editor PATCHes and is hidden for viewers; table shows each phase badge text exactly; system deployment shows no mode editor.
- [ ] **Step 2–4:** run vitest (FAIL → implement → PASS), `npm run build`.
- [ ] **Step 5: Commit** `feat(ui): on-demand mode in the wizard, deployment page and lists`.

### Task 6: Docs, proxies, changelog

**Files:**
- Create: `docs/user-guide/on-demand.md` (+ nav entry in `mkdocs.yml`)
- Modify: `docs/configuration/*.md` (new settings), `docs/architecture/` (gateway page with a Mermaid sequence: client → proxy → LiteLLM → DGXMM gateway → controller → deployer → engine), `docs/operations/troubleshooting.md` (long waits: raise read timeouts on every proxy hop — nginx `proxy_read_timeout 1200s; proxy_send_timeout 1200s; proxy_buffering off;` for `/v1/`, Caddy `reverse_proxy { transport http { read_timeout 20m } flush_interval -1 }`, LiteLLM per-entry timeouts set automatically), `deploy/nginx-dgxmm.conf` example comments, `CHANGELOG.md`.
- [ ] **Step 1:** write docs; **Step 2:** `mkdocs build --strict` (docker python:3.12 + requirements-docs.txt) — Expected: 0 warnings; docs-link and hygiene tests pass. **Step 3: Commit** `docs: on-demand models`.

### Task 7: Live verification on the DGX (no code)
- Apply proxy timeouts on the live hops (the nginx gateway's `/v1/` location; the edge reverse proxy) with backups; restart only those proxies.
- Deploy via `./install.sh` from the branch (main checkout stays on main until merge — use the worktree image tag or merge first per controller decision); create a tiny on-demand deployment and verify: cold start through LiteLLM, idle stop after timeout (set 60 s for the test), make-room with two models, queued notice in a streaming request, 15-min timeout via a lowered test setting, disconnect while queued, no OOM, always-on deployments untouched. Record results.
