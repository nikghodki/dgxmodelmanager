# DGXMM decision model — advanced features (11) — design

Date: 2026-10-17 · Branch `feature/decider-advanced` (from main 7c055b7) · Status: design, approved in conversation; awaiting written-spec review.

## Goal
Extend the shipped decision model (vllm-sr/d3-mini or d3-nano, `POST /v1/systemone`, typed choice questions with calibrated probabilities) with eleven more advisory features:

1. Health triage of running deployments
2. Deploy-time config advice
3. Pre-download compatibility check
4. Draft-model pairing for speculative decoding
5. Engine choice (vLLM vs llama.cpp)
6. Natural-language commands in the ⌘K palette
7. Usage-pattern sleep/wake suggestions with user-owned schedules
8. Benchmark/calibration reading outlier labels
9. Threshold tuning from feedback (rules only)
10. LiteLLM routing advice (advisory only — no live routing)
11. `trust_remote_code` risk triage

Success: every feature works end to end against the live decider on real data from the DGX, with measured accuracy/coverage recorded in `docs/decider-evaluation.md`, passes review and the security review, and ships to main like the decision-model release.

## User decisions (binding)
- One spec for all eleven.
- Routing is **advisory only**: DGXMM suggests LiteLLM aliases; it never changes LiteLLM or sits on the request path.
- Sleep/wake: **suggest + optional schedule** — an accepted suggestion becomes a schedule the user owns (visible, editable, pausable) that DGXMM runs; it never sleeps a deployment with live traffic.
- ⌘K commands reach **navigation, lifecycle (start/stop/sleep/wake/restart), opening the deploy wizard / reconfigure dialog prefilled, and "free N GiB" (plan chooser)** — always through the normal confirmation dialog. Remove deployment, delete weights and user administration are not reachable by typed command.

## Principles (unchanged from the shipped decision model; binding)
- Rules first, model second, abstain below a per-feature threshold; every feature has a deterministic path and works (reduced) with the decider disabled.
- Advisory only. The model never executes anything. The only automation is a user-accepted schedule, which runs through the normal deployer paths (fit check, startup gate, protections) and is audited.
- Fail open: decider disabled / unhealthy / slow / erroring → rules or no suggestion; no request path blocks beyond its deadline.
- Untrusted text (logs, model cards, configs, chat templates, source files, typed commands) goes only into `state` as quoted, length-capped, redacted blocks; options are enumerated by code.
- Model answers are validated against the offered option keys (existing `parse_answer`); explanations shown to users are deterministic templates, never model text.
- Every model/rules decision is logged in `decisions`; per-feature on/off and threshold; retention applies.
- Commits: `nikghodki <92228606+nikghodki@users.noreply.github.com>`, no trailers; no forbidden names, box identifiers or real IPs (hygiene test).

## Architecture

### Feature registry
`dgxmm/decider/features.py` gains a declarative `FeatureSpec` registry:
`FeatureSpec(name, group, label, description, default_threshold, background: bool, needs_model: bool, viewer_visible: bool)`.
- The existing five (diagnosis, parser_tool/parser_reasoning → group parser, plan, kind, workload) are registered with unchanged behaviour and thresholds.
- Settings (`decider.features.<name>.enabled|threshold`), the Decision model page toggles/thresholds, `/api/decider/status.features`, rate-limit buckets and the evaluation harness are driven from the registry, so adding a feature never repeats wiring.
- New feature names/groups: `health`, `config`, `compat`, `draft`, `engine`, `command`, `schedule`, `reading`, `tuning` (rules only, `needs_model=False`), `routing`, `remote_code`.
- Defaults: every new feature is enabled when the decision model is enabled; rules-only paths (`tuning`, rule hits of `health`, `reading`, `compat`, `draft`, `engine`, `remote_code`) also run with the decider disabled. Initial thresholds 0.8 (`command` 0.7, see §6); each is recalibrated on its real corpus during implementation and recorded in `docs/decider-evaluation.md` and `protocol.py`.

### Insights feed
For proactive findings (health, schedule proposals, reading flags, tuning proposals, routing advice):
- Table `insights(id, kind, feature, subject, severity info|warning|critical, title, body, confidence NULL, source rules|model, decision_id NULL, action JSON NULL, status open|dismissed|snoozed|applied|expired, snooze_until NULL, dedupe_key, created_at, updated_at, expires_at NULL)`; one open insight per `dedupe_key` (updated in place, not duplicated). Retention: closed insights 30 days; open ones expire per kind (health 1 h after the condition clears, proposals 7 days).
- `action` is a descriptor the UI turns into an existing dialog (`{"type":"reconfigure","deployment_id":..,"patch":{..}}`, `{"type":"schedule","deployment_id":..,"proposal":{..}}`, `{"type":"threshold","feature":..,"value":..}`, `{"type":"copy","text":..}`). The server never executes an action from an insight; applying goes through the existing endpoints with their confirmations.
- API: `GET /api/insights?status=&subject=&kind=` (viewer), `POST /api/insights/{id}/dismiss|snooze {hours}|applied` (admin, CSRF). SSE topic `insights` (`{id}` only; viewer-visible; bodies are templated and redacted).
- UI: Insights page (Operations group), top-bar count badge, an "Insights" card on Deployment detail Overview.

### Background engine
`dgxmm/decider/insight_engine.py`: one scheduler running jobs with a per-cycle budget of decider calls and lowest client priority (below interactive features; uses the existing back-pressure limiter with a background lane). A job is skipped if its previous run is still active; all jobs run their CPU work off the event loop.

| job | cadence | feature |
|---|---|---|
| health check | every 5 min | health |
| traffic profiles | nightly | schedule |
| schedule proposals | nightly after profiles | schedule |
| schedule runner | every minute | schedule (no model) |
| routing advice | daily and on deployment change | routing |
| threshold tuning | weekly | tuning |
| reading labels | on each new reading | reading |

## The features

### 1. Health triage (`health`)
- Inputs per ready deployment since the last check: redacted, capped log lines; metric summary from live samples/rollups (KV-cache usage p50/p95, preemptions, running/waiting, prompt/generation throughput vs the deployment's own 24 h baseline, error/5xx counts, request rate).
- Rules (high precision): sustained KV usage ≥ 0.95 with preemptions → `kv_pressure`; preemptions/min above a limit → `preemption_storm`; throughput < 50 % of baseline at similar load → `throughput_drop`; error lines/5xx rising → `error_creep`; clients sending prompts over the context limit / malformed tool calls → `client_misuse`; none of these → `healthy` (rules only when all metrics are well inside limits).
- Model asked only when rules are ambiguous; labels as above.
- Output: an insight (warning) with templated explanation and suggested remedies as reconfigure actions (e.g. `kv_cache_dtype: fp8`, lower `max_num_seqs`, raise `gpu_memory_utilization` when memory allows per the calculator, enable prefix caching). Clears when healthy for 15 min.

### 2. Config advice (`config`)
- Wizard: user picks a goal (`chat`, `agent_tool_use`, `batch_throughput`, `long_context`; default from the parser suggestion/model kind).
- Rules generate candidate configs from the memory calculator: combinations of `max_model_len` (native, 128k, 64k, 32k, 16k, 8k ≤ native), `kv_cache_dtype` (auto, fp8), `max_num_seqs` (8, 32, 128, 256), prefix caching on/off; keep only those that fit the budget, then a Pareto cut to ≤ 8 options, each described by its trade-off (context vs concurrency vs memory headroom).
- Model ranks the candidates for the goal; ≥ threshold → "Suggested config" card with a one-click fill of untouched fields (never overwrites typed values); below → show the top three with their trade-offs.
- `POST /api/decider/config-advice {model_ref, engine, goal}` (admin) → `{options:[{key, config, summary, required_bytes}], recommended, source, confidence}`.

### 3. Compatibility check (`compat`)
- Catalog detail (and wizard for not-yet-downloaded refs): fetch only `config.json`, `tokenizer_config.json` / `chat_template.jinja`, `generation_config.json` and the first 8 KiB of the model card through the existing `HfHubClient` (HF token setting; 64 KiB cap per file; 5 s timeout; in memory only, never written to the cache).
- Rules: architecture vs the pinned vLLM image's supported-architecture list; GGUF availability for llama.cpp; `auto_map` → needs `trust_remote_code`; gated → needs HF token; no chat template → chat completions need a template; vision/audio sub-configs.
- Model only where rules can't decide (unknown/new architecture), choosing per engine among `runs`, `runs_with_flags`, `unsupported`, `unknown` with reason keys reused from diagnosis causes.
- `GET /api/catalog/compat?repo=` (viewer) → `{vllm:{verdict, reasons, flags}, llamacpp:{...}, source, confidence}`; cached by (repo, sha). UI: compat line on catalog detail and download dialog.

### 4. Draft pairing (`draft`)
- Rules: candidates = on-disk models of kind `draft`/`llm` with identical tokenizer (hash of tokenizer.json/vocab) or same family + vocab size, smaller than the target; method compatibility (eagle/eagle3/dflash need matching architectures per the method's rules).
- Model ranks compatible candidates (state: target and candidate names, sizes, families, method).
- `POST /api/decider/draft-suggest {target_ref, method}` (admin) → ranked list; wizard speculative-decoding field shows "Suggested draft: X".

### 5. Engine choice (`engine`)
- Rules: GGUF-only → llama.cpp; safetensors-only → vLLM; architecture unsupported by the pinned vLLM image but GGUF present → llama.cpp.
- Both formats available → model chooses `vllm`/`llamacpp` given goal, model size, quantisations available, free memory and engine benchmarks on this box.
- `POST /api/decider/engine-suggest {model_ref, goal}` (admin) → `{engine, source, confidence, reason}`; wizard chip next to the engine selector.

### 6. Natural-language commands (`command`)
- ⌘K gains an "Ask" mode: when the typed text doesn't match a palette entry, "Interpret: <text>" appears.
- Two questions: (a) action among an enumerated set {navigate, start, stop, sleep, wake, restart, deploy (open wizard), reconfigure (open dialog), free_memory}; (b) target among enumerated entities — rules pre-filter deployments/models/pages by fuzzy match to ≤ 20 options. Parameters (GiB to free, context length for deploy) are extracted by rules (regex on numbers + units), never by the model.
- ≥ threshold → one interpretation; below → top three interpretations; nothing ≥ 0.3 → "Couldn't interpret".
- Selecting an interpretation opens the normal confirmation/dialog; nothing runs on interpretation. Viewers get navigation interpretations only.
- `POST /api/decider/command {text}` (viewer; ≤ 300 chars; rate-limited) → `{interpretations:[{action, target, params, confidence, label}], source}`.

### 7. Sleep/wake schedules (`schedule`)
- `traffic_profiles(deployment_id, hour_of_week 0..167, requests_per_min_p50, p95, active_fraction, samples, updated_at)` computed nightly from `metric_rollups` (30-day retention already). Proposals need ≥ 7 days of data.
- Rules propose candidate windows: idle windows ≥ 2 h where `active_fraction` ≤ 0.05 in ≥ 90 % of observed weeks; wake 15 min before the first usually-active hour. The model decides `propose`/`skip` per candidate (state: profile summary, deployment priority, memory pressure history); ≥ threshold → proposal insight.
- Accepting a proposal (admin) creates a schedule: `deployment_schedules(id, deployment_id, action sleep|wake|stop|start, days (mon..sun bitmask), at_local HH:MM, timezone, enabled, created_by, created_at, last_run_at, last_result)`. Sleep uses engine sleep mode where supported, else stop; wake/start uses the normal start path (fit check, startup gate; never evicts — if it doesn't fit, skip and raise an insight).
- Runner: every minute; skips a sleep/stop if a request arrived in the last 10 min or requests are running/waiting; skips system deployments; one run per schedule per day; every run is a deployment event + audit entry with actor `schedule:<id>`.
- API: `GET/POST /api/deployments/{id}/schedules`, `PUT/DELETE /api/schedules/{id}` (admin; viewer read). UI: Schedules section on Deployment detail (list, edit, pause, delete).

### 8. Reading labels (`reading`)
- On each new benchmark/calibration reading: rules compare with prior readings for the same (model, config, engine) (median/MAD), host contention at the time (other tenants' memory change, concurrent startups), and first-run-after-start (cold cache).
- Labels `plausible`, `contended`, `cold_cache`, `broken`; model only where rules are inconclusive.
- Only `plausible` readings feed calibration and benchmark advice; the existing `quality` column maps plausible → verified, others → disputed; the two-consecutive-readings rule remains as the fallback when the feature is off.
- UI: label badge on benchmark rows; a reading-flag insight when a `broken` reading appears.

### 9. Threshold tuning (`tuning`, rules only)
- Weekly per feature with ≥ 20 feedback rows in the last 30 days: compute precision at candidate thresholds (0.5–0.95 step 0.05) from accepted/rejected model decisions, with Wilson 95 % lower bounds; propose the lowest threshold whose lower bound ≥ the feature's target (0.95; command 0.9), if it differs from the current one by ≥ 0.05.
- Insight with the evidence table; "Apply" sets the threshold through `PUT /api/decider/settings` (admin, audited).

### 10. Routing advice (`routing`)
- Traffic classes: `code`, `chat`, `tool_use`, `long_context`, `vision`, `batch`.
- Rules filter candidate deployments per class by capability (tool parser present, vision model, max_model_len ≥ 32k for long_context, ready or regularly used); the model ranks candidates per class using workload labels, benchmarks and memory footprint.
- Output: routing-advice insight + a section on the LiteLLM page listing class → deployment with confidence and a copyable LiteLLM `model_list` alias snippet (alias name `dgxmm-<class>` → the deployment's existing LiteLLM model name). DGXMM changes nothing in LiteLLM.
- `GET /api/decider/routing-advice` (admin).

### 11. Remote-code risk (`remote_code`)
- Trigger: whenever a deploy/reconfigure would enable `trust_remote_code` (wizard, reconfigure, diagnosis patch) for a model whose snapshot has `auto_map`.
- Static scan of the snapshot's `.py` files in a subprocess with a timeout and size caps, using Python's `ast` (never importing/executing): imports of network/process modules (`socket`, `requests`, `urllib`, `http`, `subprocess`, `os.system`/`popen`, `ctypes`), `eval`/`exec`/`compile`/`__import__`, base64/zlib + exec patterns, file writes outside the model dir, environment-variable reads of credentials, very long encoded literals.
- Model labels `routine`/`unusual` from the scan summary and excerpts. Warning shown if any rule finding exists OR the model says `unusual` ≥ threshold; a `routine` answer never removes a rule finding; with no findings and no model, the existing generic warning stays.
- `POST /api/decider/remote-code-check {model_ref}` (admin) → `{verdict, findings:[{file, line, kind}], source, confidence}`; the dialog lists findings with file:line.

## Data, migrations, retention
- New tables `insights`, `deployment_schedules`, `traffic_profiles`, `compat_cache(repo, sha, result JSON, updated_at)` via additive, idempotent `schema_version` migrations; an install from main 7c055b7 upgrades cleanly and rolls back (old code ignores new tables).
- Deleting a deployment deletes its schedules, profiles and open insights.
- Retention: insights (closed 30 days), compat cache 7 days, decisions/diagnoses as today.

## Security
- Typed commands are admin/viewer input but treated as untrusted text (quoted, capped at 300 chars); the model only selects among enumerated actions/targets; parameters come from rules; destructive actions are not in the action set.
- Compat fetch: only via the HF hub client to the configured endpoint, file allowlist, size/time caps, no execution, results cached without raw file contents beyond excerpts used for display.
- Remote-code scan never imports code; runs in a subprocess with CPU/memory/time limits; findings are informational and can only add warnings.
- Schedules run as an audited system actor, go through the same deployer paths and protections as users (fit, startup gate, system deployments excluded, never evict).
- All new routes: role-checked, CSRF on state changes, rate-limited from the registry, body-capped; insight bodies templated.
- `docs/security.md` gains sections for these paths; the security review covers ⌘K, the scheduler, compat fetching and the code scanner.

## Errors and fail-open
Each endpoint returns a result with `source: rules|model|none` and never 5xx because of the decider; background jobs log and continue; a failing job raises at most one deduplicated "feature degraded" insight.

## Testing and evaluation
- Unit/integration tests per feature: rules paths, fake-decider answers, abstain/threshold, fail-open, injection-shaped inputs only in `state`, logging, role/CSRF/rate-limit, migrations upgrade from a main-era DB.
- Real-data corpora on the DGX (committed fixtures follow the existing hygiene and provenance rules — public repos only, no box identifiers):
  - health: windows from real deployments incl. induced pressure (tiny KV budget, burst load) and healthy baselines;
  - config/engine/draft: wizard cases for on-disk and public models with judged best answers;
  - compat: the 140-repo corpus with known vLLM/llama.cpp outcomes;
  - command: ≥ 150 hand-written utterances incl. ambiguous and adversarial ones;
  - schedule: real traffic profiles plus synthetic weekly patterns;
  - reading: existing benchmark/calibration history incl. known contended readings;
  - remote_code: real `auto_map` repos plus synthetic malicious samples.
- Targets at chosen thresholds: precision ≥ 0.95 (command top-1 ≥ 0.9, top-3 ≥ 0.98); coverage reported. Results in `docs/decider-evaluation.md`.
- Live end-to-end on the DGX and a browser check (light/dark/mobile) of every new surface.

## Out of scope
Live request routing; any automatic action other than user-owned schedules; executing remote code; free-text generation by the model.
