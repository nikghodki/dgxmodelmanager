# DGXMM decision model POC — design

Date: 2026-10-16 · Status: **shipped** in the 2026-10-17 release (see [CHANGELOG.md](../../../CHANGELOG.md)). This is
the original design, kept for history; the user and operator docs describe what shipped:
[usage](../../usage.md#decision-model), [configuration](../../configuration.md), [architecture](../../architecture.md),
[operations](../../operations.md), [security](../../security.md), [evaluation](../../decider-evaluation.md).

## Differences from this design (as shipped)
- **Watchdog:** the decider is the *first* watchdog victim, not exempt: it is stopped before any user deployment
  under memory pressure, and whenever its measured memory runs away past its reservation, even with auto-protect off.
- **Thresholds:** calibrated on real data — parser 0.8 (not 0.9), workload 0.75 (not 0.7); plan ranking stays at
  0.6 and is the weakest feature (the safety filter always runs first).
- **Models:** `vllm-sr/d3-nano` is supported next to d3-mini (smaller, faster, less accurate), selectable on the
  Decision model page; both pinned to a revision with checked-in hashes.
- **Downloads:** the page's Download uses the pinned, verified, resumable snapshot downloader (the installer's and
  the CLI's), not the generic Hugging Face download.
- **System deployment actions:** no Switch or Reconfigure; it starts by enabling the decision model; removing it
  disables the decision model. Plans never touch it; when a model does not fit, the plan chooser says how much memory
  the decision model holds and where to turn it off.
- **Supervisor:** failed restarts count toward the hourly limit; after giving up, the page shows why and offers Retry.
- **Hardening and lifecycle** added after this design: start-up queue, key delivery via tmpfs and rotation, internal
  network, read-only container, integrity check (exit 3), retention of decisions and diagnoses, rate limits.

## Goal
Make DGXMM "smart" with a small local decision model (`vllm-sr/d3-mini`, 4.5B, Apache-2.0, pinned snapshot `97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e`) that answers typed questions with calibrated probabilities via `POST /v1/systemone`. Five features:
1. **Failure diagnosis** — explain why a deployment failed + a fix.
2. **Config suggestions** — tool-call / reasoning parser (and related flags) in the deploy wizard.
3. **Plan recommendation** — mark the best safe plan when a model doesn't fit.
4. **Artifact understanding** — model kind (llm / draft / embedding / reranker / vision-llm / speech / other) for models no rule recognises.
5. **Workload-aware advice** — classify each ready deployment's workload and rank/filter tuning advice.

Spike evidence (70-item golden set, `~/spike-decider/`): d3-mini 0.90 overall (diagnosis 0.96, parser 0.92 with option descriptions, plan 0.90, kind 1.00), calibrated (≥0.8 confidence → 98% accurate on 69%), injection-robust (max 0.013 on injected answer), ~75 ms p50 on GB10, ~12–13 GiB memory.

## Principles (binding)
- **Rules first, model second, abstain when unsure.** Every feature has a deterministic path; the model is consulted only where rules have no answer (or to rank among rule-allowed options) and its answer is used only above a per-feature confidence threshold. Below threshold → today's behaviour.
- **Advisory only.** The model never executes anything. Any change goes through the existing confirmed flows (wizard, Reconfigure/SwitchDialog with rollback, plan chooser). The model never decides `check_fit`, budget, state machine, watchdog stops, deletion, redaction or LiteLLM naming.
- **Plans:** a deterministic safety filter runs first (covers shortfall, never stops/sleeps an active deployment, a dependency (draft in use), a pinned/system deployment, keeps user-stated requirements); the model ranks only plans that pass. If none pass → no recommendation.
- **Fail open.** If the decider is disabled, absent, unhealthy, slow (timeout) or errors, every feature silently falls back to rules / no suggestion. No request path blocks on the model beyond its timeout (default 2 s; diagnosis 10 s background).
- **Untrusted inputs** (logs, model cards, chat templates, config.json) go into the `state` as quoted data, length-capped (logs: last ~120 lines / 6 KiB after redaction via `memory/redact.py redact_text`), never into question instructions. Redact before sending.
- **Everything is logged** in a `decisions` table (feature, subject, options, probabilities, choice, confidence, source rules|model|none, latency, model revision, created_at, user feedback accepted/rejected/null) and visible to admins.
- Commits: author/committer `nikghodki <92228606+nikghodki@users.noreply.github.com>`, **no trailers**, and no references to any other identity in files.

## Component 1 — "decision" engine type
- New engine `decision` in `dgxmm/engines/decision.py` registered in `engines/registry.py`. Deployable like other engines but special:
  - Image: `dgxmm-decider:<version>` built locally from `deploy/decider/Dockerfile`: `FROM vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8` + `pip install transformers==5.17.0 flash-linear-attention==0.5.2` (build-time, not start-up), entrypoint runs the model snapshot's `d3_server.py --model <snapshot> --device cuda:0 --batch-size 8 --host 0.0.0.0 --port 8000`. Bearer key via env `DECISION_API_KEY` from a generated secret (`secrets/decider_api_key`, 600). `HF_HUB_OFFLINE=1`; HF cache mounted read-only; snapshot revision pinned and MANIFEST check kept (the server does it).
  - Supported models: allowlist with `vllm-sr/d3-mini` (revision pinned); config fields minimal (batch size). Estimate: fixed reservation `13 GiB` (spike peak 12.6 GiB) + margin; health = `GET /health` then a tiny `/v1/systemone` probe.
  - Marked **system** (`deployments.system` / flag in config): excluded from plan stop/sleep candidates and LiteLLM registration, and shown with a "System" badge. Watchdog: it is advisory and fails open, so it is the *first* watchdog victim (before any user deployment) and is stopped when its measured memory exceeds its reservation + `decider_overuse_margin_gib`; the supervisor restarts it with backoff (production change, see docs/security.md).
  - Installer/compose: build the image on demand (`./install.sh` builds it only when the decider is enabled, or a `make decider-image` / API-triggered build via docker SDK `images.build`). POC: build via docker SDK from the app on first enable, with progress, or via a script; choose the simpler robust path and document.
- Only one decider deployment at a time.

## Component 2 — Decider service
- `dgxmm/decider/client.py`: async client for `/v1/systemone` (httpx, bearer), typed questions (`choice` with options + per-option descriptions, `noul`, `score`), returns `{choice, probabilities, confidence=max prob}`; timeouts; small LRU cache keyed by (feature, input hash, model revision); circuit breaker (after N failures back off 60 s).
- `dgxmm/decider/service.py`: `Decider` with `available()`, `decide(feature, state, question, options, threshold)` → `Decision(source, choice, prob, probabilities, latency)`; writes the `decisions` row; never raises to callers (returns `source="none"` on any failure).
- Settings (`app_settings`): `decider.enabled` (bool, default false), `decider.deployment_id`, per-feature thresholds (`diagnosis 0.8`, `parser 0.8`, `plan 0.6`, `kind 0.8`, `workload 0.75`; calibrated on real data, see docs/decider-evaluation.md — originally parser 0.9 / workload 0.7), per-feature on/off.
- API (`/api/decider`, admin unless noted): `GET /status` (viewer: enabled, healthy, model, revision, latency p50, counts), `PUT /settings`, `POST /enable` (creates/starts the decider deployment if memory fits — uses normal preview/fit/plan flow; returns plan options if it doesn't fit), `POST /disable` (stops it), `GET /decisions?feature=&limit=` (admin), `POST /decisions/{id}/feedback {accepted: bool}`.

## Feature 1 — Failure diagnosis
- Trigger: deployer state hook on `→ failed` (and on a failed adopt/reconcile with a log tail). Background task; result stored in a new `diagnoses` table or in `decisions` with subject = deployment id + event id, and pushed via SSE (`diagnosis` topic).
- Taxonomy (12): `gpu_oom`, `kv_cache_too_small`, `unsupported_architecture`, `trust_remote_code_required`, `auth_gated`, `bad_parser_flag`, `tokenizer_or_template`, `port_in_use`, `offline_missing_files`, `driver_mismatch`, `disk_full`, `other` — each with a one-line description (used as option descriptions) and a fix hint + optional config patch (e.g. `trust_remote_code: true`; `max_model_len` halved; `kv_cache_dtype: fp8`; remove/replace `tool_call_parser`; "add an HF token in Settings" link; "free port"/retry).
- Rules first: high-precision regexes (port in use, 401/403 gated, "trust_remote_code", "not supported" architectures, "No space left on device", CUDA OOM, "cache blocks", parser errors, driver/CUDA version). If exactly one rule fires → source rules. Else ask the model with the redacted log tail + exit code + engine + model ref; accept if prob ≥ threshold.
- UI: on DeploymentDetail (Overview top + Events), a "Diagnosis" card: cause label, confidence (badge with %), source (Rules / Model), explanation, fix hint, "Apply fix" (opens Reconfigure with the patch prefilled) when a patch exists, and 👍/👎 feedback. Below threshold: "Couldn't diagnose confidently — see the log" with the log tail.

## Feature 2 — Config suggestions (parsers)
- `POST /api/decider/suggest-config {model_ref, engine}` (admin). Rules: lookup table of known families (qwen3 → hermes / qwen3; qwen3-coder → qwen3_coder; llama3.x → llama3_json; mistral → mistral; deepseek-v3/r1 → deepseek_v3 / deepseek_r1; gpt-oss → openai / openai_gptoss; granite → granite; gemma → hermes? etc. — verify the parser names available in the pinned vLLM image) keyed by architecture + name + chat template markers. Unknown → ask the model with model name, architecture, chat-template excerpt (from the snapshot's `chat_template.jinja` / tokenizer_config), with per-option format descriptions (spike showed this lifts accuracy 0.79 → 0.92). Two questions: tool-call parser (options from the pinned image's parser list + "none") and reasoning parser (options + "none").
- Wizard: when a model is chosen, call suggest-config; show a "Suggested" chip next to Tool-call parser / Reasoning parser with confidence + source; auto-fill only if source=rules or prob ≥ threshold("parser") (0.8) (and the field is empty/untouched); otherwise show "Suggested: X (72%) — Apply". Never overwrite a user-typed value.

## Feature 3 — Plan recommendation
- In `deployer.plans_for` (or a wrapper used by preview): after `build_plans`, run the deterministic safety filter (above) → `allowed` set; mark disallowed plans with `unsafe_reason` (kept visible but not recommendable). If ≥2 allowed: ask the model (choice among allowed plans; state = shortfall, current config, each deployment's state, idle minutes from `metric_rollups`, priority, dependencies; options described by title+detail+frees/needs). If prob ≥ threshold → `recommended=true`, `recommendation={confidence, source, reason}` on that PlanOption; reason is a deterministic template from the plan's facts (e.g. "frees 48 GiB; chat-inference idle 3 days"). If exactly 1 allowed → recommend it with source rules. Latency: preview path; timeout 2 s, cached by inputs.
- UI plan chooser: "Recommended" badge + reason + confidence on that option; "Unsafe: <reason>" note on filtered ones (still selectable? NO — unsafe plans remain selectable only if they were selectable before (user choice), but never recommended; keep current behaviour, just annotate).

## Feature 4 — Artifact understanding
- Inventory: after `classify_kind`, if the rule result is uncertain (falls to "other" or architecture unknown), ask the model with config.json fields (architectures, model_type, task hints, file list summary) → kind among {llm, draft, embedding, reranker, vision-llm, speech, decision, other}; cache by (path, mtime, revision) in a table; apply if prob ≥ threshold (`kind_source="model"`, `kind_confidence`). Background (after scan), never in the request path.
- UI Models page: Kind badge shows a small "AI" marker + tooltip (confidence) when model-derived.

## Feature 5 — Workload-aware advice
- For each ready deployment with ≥ 30 min of rollups: compute a numeric summary (mean/peak running & waiting, prompt vs generation token rates, prompt length proxy, prefix-hit rate, spec accept, request rate, idle fraction). Rules for clear cases (idle >90% → `idle`); else ask the model: workload ∈ {single_user_chat, agent_tool_use, batch_throughput, long_context_rag, idle, mixed}. Cached 15 min per deployment.
- Advisor: given the workload, rank/filter rules' advice (e.g. idle → suggest sleep/stop-to-free memory first; batch → raise_seqs/moe_batch first; long_context → fp8_kv/prefix caching first; agent → spec decoding / parser correctness first) and add a "Workload: X (78%)" badge on the Advice tab; advice items get `relevance` ordering. Rules keep producing the advice set; the model only labels the workload.

## Admin UI — "Decision model" page (Integrations group)
- Status (model, revision, healthy, memory, p50 latency, decisions today, acceptance rate), Enable/Disable (deploys/stops the decider via normal fit/plan flow; shows plans if it doesn't fit), per-feature toggles + thresholds, recent decisions table (feature, subject, choice, confidence, source, latency, feedback), link to docs.

## Testing
- Backend: fake decider (httpx MockTransport) for all features; rule-first paths; threshold/abstain; fail-open on timeout/error/circuit open; safety filter properties (never recommends a plan that stops an active/dependency/pinned/system deployment); injection-style inputs are placed only in state; redaction applied; decisions logged; system decider excluded from plans/LiteLLM and stopped first by the watchdog.
- Real model: integration check on the DGX against the running decider with the spike golden set subset (diagnosis + parser + kind) — record accuracy & latency in the report (not a CI test).
- Frontend: vitest for diagnosis card, wizard suggestions (auto-fill rules, no overwrite), plan badge, kind marker, workload badge, decision page.

## Out of scope (POC)
LiteLLM request routing via the decider; natural-language command palette; training/fine-tuning on DGXMM data; pplx-flash alternative (same API; later).
