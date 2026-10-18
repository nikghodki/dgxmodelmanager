# Changelog

All notable changes to DGXMM are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). DGXMM does not publish tagged releases yet
(the package version is `0.1.0`), so each section below is identified by its date. Upgrade notes for existing installs are in
[upgrade](docs/getting-started/upgrade.md#upgrading-an-existing-install).

## [2026-10-18] — Advanced decision-model features

Eleven more advisory features of the decision model (16 in all), an Insights feed with a background job engine,
deploy-time advice in the wizard and catalog, user-owned sleep/wake schedules and natural-language commands in
the ⌘K palette. Rules run first and every feature fails open, as before; nothing changes without an admin's
confirmation, except schedules an admin created.

### Added

- **Insights** (Operate → Insights, a count in the top bar and an Insights card on each deployment's Overview):
  findings from DGXMM's background checks, each with a templated explanation, its source (Rules or AI with the
  confidence) and an action that opens the normal dialog (Reconfigure, set a threshold, show a snippet, review a
  schedule, open a page). Admins can dismiss them or snooze them for 1 hour to 1 week. See
  [insights](docs/user-guide/insights.md).
- **Background job engine**: health checks every 5 minutes, reading labels, nightly traffic profiles and schedule
  proposals, daily routing advice, weekly threshold tuning and the schedule runner every minute. Model questions
  of these jobs have a per-run budget and queue behind interactive questions; a job that fails three times in a
  row raises one "degraded" insight. Their status is on the Decision model page (**Background jobs**).
- **Health triage** of ready deployments: KV-cache pressure, preemption storms, throughput drops, rising server
  errors and clients sending bad requests, with a suggested reconfigure fix where one fits the free memory.
- **Deploy-time advice** in the wizard: a goal (chat, agents and tool use, batch throughput, long context) with
  suggested configurations sized by the memory calculator (Fill sets only fields you have not changed), an engine
  suggestion next to the engine selector, a new **Speculative decoding** field with a suggested draft model from
  your disk, and a compatibility summary for models that are not downloaded yet.
- **Compatibility check** before a download (catalog details and the download dialog): whether vLLM and
  llama.cpp can run a Hugging Face repo here, and what it needs (trust_remote_code, an HF token, a chat template).
  It reads a few small files from the Hub (64 KiB each), never writes to the HF cache and caches results 7 days.
- **Remote-code check**: when trust_remote_code would be turned on for a model with custom code, the snapshot's
  Python files are scanned statically in an isolated, resource-limited process (never imported) and findings are
  listed with file and line. Findings only add warnings; a clean scan is shown as "no suspicious patterns found
  (static scan only)".
- **⌘K Ask**: text that matches no palette entry can be interpreted into navigation, start, stop, sleep, wake,
  restart, opening the deploy wizard or Reconfigure prefilled, or freeing N GiB with a plan. Every interpretation
  opens the normal confirmation; removing deployments, deleting weights and user administration are refused.
  Viewers get navigation only.
- **Sleep/wake schedules**: a **Schedules** card on the deployment page (add, edit, pause, delete), proposals from
  the traffic of the last 30 days (at least 7 days needed), and **Pause all schedules** on the Decision model
  page. A schedule never sleeps or stops a deployment that is serving requests, never stops anything else to make
  room, runs each time at most once a day and is audited as `schedule:<id>`. Schedules keep running when the
  schedule feature or the decision model is switched off; only Pause all (or pausing a schedule) stops them.
- **Reading labels** for benchmark levels and start-up memory measurements (plausible, contended, cold cache,
  broken): only plausible readings feed calibration and benchmark advice, and a broken one raises an insight.
  Badges on the Benchmarks tab.
- **Threshold tuning** (rules only): once a week, a feature with enough rated answers in the last 30 days gets a
  proposal for the lowest threshold whose precision (Wilson 95 % lower bound) reaches the target, with the
  evidence table. **Set threshold** applies it.
- **LiteLLM routing advice**: per traffic class (code, chat, tool use, long context, vision, batch) a suggested
  deployment and a copyable `router_settings.model_group_alias` snippet (`dgxmm-<class>` aliases) on the LiteLLM
  page and as an insight. DGXMM never changes LiteLLM itself.
- The Decision model page lists all 16 features with a switch and threshold each (threshold tuning, which asks
  no model, shows "Rules only — no threshold"), and the background jobs with their last result and next run.
- The installer detects the host's time zone and writes `DGXMM_TZ` to `.env`; daily and weekly jobs and schedules
  use it.
- New tables (`insights`, `deployment_schedules`, `traffic_profiles`, `compat_cache`) and the settings
  `decider_feedback_retention_days` (180) and `schedules.paused_all`.
- `SECURITY.md` (private vulnerability reporting, supported versions, scope), issue forms for bug reports and
  feature requests, and a pull request checklist.
- CI: a shellcheck job (pinned `koalaman/shellcheck:v0.11.0`) for the installer, its test stubs and the scripts;
  README badge for the tests workflow.
- Documentation site at <https://nikghodki.github.io/dgxmodelmanager/> (MkDocs Material, built from `docs/` and
  published by `.github/workflows/pages.yml`; pull requests build it with `mkdocs build --strict`). The pages were
  reorganised into getting started, user guide, architecture (with new request-flow, decision pipeline, insight
  lifecycle and schedule-runner diagrams), configuration, operations, security and evaluation; the decision model
  page's **Docs** link opens the site.

### Changed

- The default llama.cpp image is pinned by digest (`ghcr.io/ggml-org/llama.cpp@sha256:e4cfff28…`, build
  b11669), like the vLLM image. DGXMM's own base images (`node:22-bookworm-slim`,
  `python:3.12-slim-bookworm`) and the bundled LiteLLM's Postgres (`postgres:16.15-trixie`) are pinned by
  digest as well. Existing `config.yaml` files keep their image settings (see
  [upgrading](docs/getting-started/upgrade.md#upgrading-an-existing-install)).
- Who stopped a deployment is now recorded as a structured field of the stop event. A deployment that a schedule
  stopped before this upgrade is therefore not woken by that schedule afterwards: start it once by hand.
- Routing advice refreshes once deployment changes and the LiteLLM registrations they cause have settled (15 s,
  at most 60 s after the first change), so a burst of changes is one run.
- Chart legends use the muted text colour, so they are readable in the light theme too.
- Insight texts keep their structure: bulleted lists and the threshold-tuning evidence table are shown as a list
  and a table, other paragraphs keep their line breaks. Counts and time windows are pluralised ("1 request",
  "the last minute", "the same 3 windows").
- The code-scan summary counts the files with findings and the files scanned separately ("2 findings in 1 file
  (14 files scanned)").
- CI actions moved to their Node 24 releases and to `ubuntu-24.04` runners.
- README: requirements (network, disk, ports), supported hardware, dry run, security reporting and
  contributing. Operations: upgrading by hand, and moving a clone from before the history rewrites.

### Fixed

- Two DGXMM processes opening a new database at the same moment (the app and an admin command, or the
  test suite on a busy machine) could fail with "database is locked" while switching it to WAL; the
  switch is now retried for up to 5 seconds.
- `install.sh` is clean under shellcheck 0.9 as well (SC2015), the version on Ubuntu 24.04 runners.
- A new deployment could get the id of a removed one (SQLite reuses the highest row id) and show its
  benchmark runs, reading labels, rollups and insights. Deployment ids are now never reused: on the first start
  after the upgrade a one-time migration rebuilds the `deployments` table and starts the id sequence above every
  id still referenced. Benchmark results stay per model and configuration, as before.
- Health triage: a flood of client HTTP 4xx answers on a sleep-mode vLLM (which logs an error echo per
  rejected request) is reported as clients sending bad requests only, no longer also as rising server errors.

### Security

- Remote-code scan: rules for code objects built by hand, object-graph walks, asyncio / multiprocessing / fork
  process APIs, native code loaded at run time, code fetched from the Hub or from other repositories at run time,
  loaders given a file path, aliased and from-imported calls, `trust_remote_code=True` with a literal repo id,
  `torch.hub.load` and pip installs, and more unpickling paths (`pandas.read_pickle`, `np.load` with
  `allow_pickle`). The decision model can
  only add a warning, never report code as safe. One scan runs at a time; failed scans are remembered briefly. The
  scan is static only and has blind spots, listed in [security](docs/security.md#deploy-time-advice-hub-fetches-and-the-code-scanner).
- ⌘K: the rules' matching is bounded and runs off the event loop; a timed-out interpretation answers with the
  rules result instead of starting over. Unsafe free-memory plans must be acknowledged against the reason shown;
  if the reason changed meanwhile, the request is refused with the new one.
- Decision log: rules-only answers to viewers are not logged; rows with feedback are exempt from the row cap and
  kept 180 days (`decider_feedback_retention_days`, never fewer than `decider_retention_days`), so viewers'
  requests cannot push rated rows out of the log.
- Hub access: viewers' compatibility checks, catalog search and catalog details (`GET /api/catalog/search`,
  `GET /api/catalog/details`) read the Hub anonymously, never with the configured HF token, so a viewer neither
  sees the token owner's private repos nor probes gated access with it; private repos are never cached; Hub redirects go only to Hub
  hosts over https (at most 3); every Hub API call has a timeout (20 s per connect/read), on its own thread pool.
- Insight actions that would change the model or turn on trust_remote_code are rejected by the server and the
  client; deleting a deployment's schedules with it is audited; viewers do not see who created a schedule.
  `docs/security.md` covers the job engine, insights, the schedule runner, ⌘K and the new rate limits.

### Upgrade notes

- `./install.sh upgrade` migrates the database once at the first start: the new tables, then the rebuild of the
  `deployments` table (one transaction, foreign keys checked before and after). Rolling back with
  `git checkout <previous commit> && ./install.sh` keeps working on the migrated database (older code ignores
  the new tables). See [upgrading an existing install](docs/getting-started/upgrade.md#upgrading-an-existing-install).
- The background jobs start with the upgrade. With the decision model disabled they run their rules only and
  ask nothing; schedules exist only once an admin creates one.
- Re-run `./install.sh` once so `.env` gets `DGXMM_TZ`; without it, DGXMM uses `TZ`, `/etc/localtime`, else UTC.
- The repository history was rewritten again on 2026-10-18 (see
  [the history rewrite](docs/getting-started/upgrade.md#clones-from-before-the-2026-10-18-history-rewrite)).

## [2026-10-17] — Decision model

An optional local decision model that helps with failure diagnosis, parser choice, plan selection, model
kinds and workload-aware advice, plus a start-up queue, finer memory plans and fixes for weight deletion,
download cancellation and out-of-memory reporting.

### Added

- **Decision model** ("decider", off by default): a small local model (`vllm-sr/d3-mini`, Apache-2.0,
  pinned revision `97dd054d`; `vllm-sr/d3-nano` selectable) that answers typed questions with
  calibrated probabilities. Rules first, the model second, abstain below a per-feature threshold,
  advisory only, fail open: when it is off, unhealthy, slow or wrong, every feature behaves as with its
  rules alone. See [usage](docs/user-guide/decision-model.md).
- **Failure diagnosis**: when a deployment fails, rules (14 causes, such as GPU out of memory, KV cache
  too small, context longer than the model supports, gated repo, bad parser flag, port in use, disk full,
  and the decider's own integrity and key refusals) or the model explain why, with a fix hint and, where
  possible, **Apply fix…**, which prefills Reconfigure. Diagnosis card on the deployment page.
- **Parser suggestions** in the deploy wizard: tool-call and reasoning parser from rules for known model
  families and chat templates, else the model; filled in automatically only for a strong rule or a
  confident model answer, never over a value you typed.
- **Plan safety and recommendations**: a deterministic safety filter marks plans that would stop or
  sleep an active, recently used, depended-on, higher-priority or system deployment, or lower a value you
  set, as **Unsafe** with the reason; one safe plan is **Recommended** by rules, or by the model when
  several are safe.
- **Model kinds**: a new kind `decision` (classifiers, guard models, decision heads); models no rule
  recognises get a model-derived kind, marked **AI** in the Models table.
- **Workload-aware advice**: ready deployments with 30 minutes or more of metrics get a workload label
  (single-user chat, agent / tool use, batch throughput, long-context RAG, idle, mixed) that orders the
  tuning advice.
- **Decision model page** (Integrations > Decision model, admins): status, enable and disable through
  the normal memory check (with plans when it does not fit), in-app image build with live progress, model
  choice (d3-mini or d3-nano, 13 or 9 GiB), per-feature switches and thresholds, the decision log with
  feedback, key rotation, and a banner when a running decider needs a restart after an upgrade. Its
  **Download** uses the installer's verified, resumable snapshot downloader (pinned files only, each
  hash-checked before it is linked), with progress on the Model card and the Downloads page; any download
  of a decision model, including `POST /api/downloads`, takes that path. A **Decision model** status chip
  in the top bar (a read-only status popup for viewers).
- The decider runs as a *system* deployment of the new `decision` engine: fixed reservation, never a
  plan stop/sleep candidate (the plan chooser notes how much memory it holds and where to disable it),
  never registered in LiteLLM (no endpoint, snippets or registration badge for it), **System** badge,
  stopped by the memory watchdog before any user deployment and restarted by a supervisor (probes every
  30 s, backoff from 30 s to 15 minutes, at most 5 attempts an hour, failed ones included, never under
  memory pressure; when it gives up the page shows why and offers **Retry**). It is managed on the
  Decision model page: no Switch or Reconfigure, Start only while the decision model is enabled, and
  Remove also disables it. Its image `dgxmm-decider:3` is built locally from `deploy/decider`.
- **`./install.sh --with-decider`** (or `DGXMM_WITH_DECIDER=1`, remembered in `.env`): checks memory and
  disk, builds the decider image, downloads and verifies the pinned snapshot, enables the decider and
  waits until it is healthy (`DGXMM_DECIDER_TIMEOUT`, default 900 s). When only remembered in `.env`,
  decider problems are warnings that never fail an upgrade, and a decider an admin disabled stays
  disabled. `uninstall --purge` also removes this install's decider container, image and internal
  network; `--purge-decider-model` also deletes the model snapshots.
- **Admin CLI**: `docker exec dgxmm python -m dgxmm.admin decider status|build|download|enable|disable`
  and `rotate-decider-key`, over a new admin unix socket (`data/run/admin.sock`) with short-lived, scoped
  tokens; audited with `"via": "cli"`.
- **Per-install labels**: `DGXMM_INSTALL_ID` (generated once in `.env`) labels the containers DGXMM
  creates and the images it builds (`dgxmm.install`), so uninstall touches only this install's.
- **Start-up queue**: at most `max_concurrent_startups` engines (default 1) start or load at once,
  because vLLM's start-up memory profile fails when another engine starts or stops at the same moment on
  unified memory. Others wait in the new state **Queued** (position and reason shown) with their memory
  reserved; the fit is checked again at their turn, and they fail after `startup_queue_timeout_s`
  (30 minutes). A wake-up waits for the slot too: the request answers `202` after `wake_sync_wait_s`
  (50 s) and the wake-up continues in the background.
- **Per-deployment plans**: one sleep or stop plan per deployment that could make room, plus minimal
  combinations of up to three, least disruptive first (not busy, lower priority, longest idle).
- **Versioned database migrations** (`schema_version`), run once each in a transaction at start.
- API: `/api/decider/status|settings|enable|disable|models/download|image|image/build|decisions|
  decisions/{id}/feedback|suggest-config|rotate-key`, `GET /api/deployments/{id}/diagnosis`,
  `GET /api/deployments/{id}/workload`; plan options gain `recommended`, `recommendation` and
  `unsafe_reason`; inventory artifacts gain `kind_source` and `kind_confidence`; advice items gain
  `relevance` and `workload_fit`; deployment views gain `system` and `startup_queue`.
- Documentation: the decision model in [usage](docs/user-guide/decision-model.md), its settings in
  [configuration](docs/configuration/index.md#decision-model-decider), a [threat model](docs/security.md) and
  an [evaluation on real data](docs/decider-evaluation.md) (real failure logs, 140 real model repos, real
  traffic, performance under load), with the corpora as regression fixtures under `tests/fixtures/`.

### Changed

- Plan ids are `stop:<ids>` / `sleep:<ids>` instead of `stop` / `sleep` (API clients that matched them
  need updating), and there can be several sleep and stop plans.
- The log tail stored with a failed deployment keeps the last 200 lines plus the distinct error lines
  above them, so a root cause printed hundreds of lines earlier is no longer lost. Every stored tail is
  redacted, and tails stored by older versions are redacted when read.
- Removing a deployment also drops its metric rollups, cached workload label and diagnoses.
- GGUF quantization suffixes (`Q8_0`, `Q4_K_M`, `UD-Q4_K_XL`) are recognised in model names, so variant
  plans and the llama.cpp advice apply to them.

### Fixed

- Deleting a model also removes the files of the shared Hugging Face blob store (`hub/blobs`,
  huggingface_hub 2 and later) that only it uses, keeps those another model uses, and reports the space
  really freed.
- Cancelling a download stops the file in flight at once and removes its partial file; a cancelled or
  failed download is listed as incomplete instead of complete.
- Exit code 137 is reported as out of memory (and raises the next estimate) only when Docker reports an
  OOM kill or the log has the kernel OOM killer's line; a plain SIGKILL is reported as such.
- `docker stop` of DGXMM no longer ends in SIGKILL (exit 137): open event streams end on SIGTERM and it
  exits well within the stop timeout.
- A failed container start no longer leaves a container in state "Created" behind.
- When a host process holds a deployment's port, the start retries on the next free port (up to 4
  times).
- Redaction no longer masks ordinary exception messages such as `KeyError: 'model'` (parser errors were
  unreadable).
- Decision and classifier models were classified as vision-llm, Qwen3-Embedding and EAGLE heads as llm.
- Two DGXMM processes starting at once (the app and an admin command) no longer fail with "database is
  locked".
- The Catalog page fits a 390 px phone; text contrast is at least 4.5:1 in both themes.

### Security

- The decider container runs as uid 10001 with a read-only root filesystem, no capabilities,
  `no-new-privileges`, private IPC, a pids limit (512), a memory limit (16 GiB, host side) and `/tmp` as
  a 0.75 GiB tmpfs; it has no published port and joins only the internal network `dgxmm-decider` (no
  egress). Its only mounts are its own model files, read-only, never the whole HF cache.
- The decider runs only a private copy of its hash-pinned files, verified on its tmpfs and imported from
  there under `python3 -I`, so files planted next to the snapshot are never imported; it refuses to
  start (exit 3) on any mismatch. Its image uses a digest-pinned base and hash-pinned wheels.
- The decider's bearer key (`data/secrets/decider_api_key`, mode 600) is delivered as a 0400 file on a
  tmpfs after the container starts, never in the environment, the command line or `docker inspect`;
  it can be rotated from the UI, the API or the CLI.
- Untrusted text (engine logs, chat templates, model names, `config.json`) reaches the decider only
  quoted, capped and redacted; answers are validated (offered options only, finite probabilities,
  64 KiB reply cap) and are advisory: every change still needs an admin's confirmation.
  `trust_remote_code: true` is suggested only when a rule matched, with a warning.
- Untrusted log text is bounded (2 KiB per line, 512 KiB per diagnosis) before any rule runs, every
  regex over it is bounded, and matching runs off the event loop (an echoed architecture name could
  otherwise freeze the app).
- Decider API: per-user rate limits (`429 rate_limited`), a 64 KiB body limit, strict request schemas,
  every change audited. `POST /api/deployments/preview` is admin-only.
- Redaction also masks PEM private keys (also a key body whose first line a tail cut off), and the live
  log view of a deployment is redacted.
- The decider's network is never a trusted proxy, its image build log goes to admins only over SSE, and
  the decision log keeps 30 days / 50 000 rows (diagnoses: 20 per deployment).
- CI actions are pinned by commit SHA with a read-only token.

### Upgrade notes

- `./install.sh upgrade` migrates `data/dgxmm.sqlite3` in place at start (new tables and columns, then
  versioned migrations). Rolling back to the previous release keeps working; disable and remove the
  decision model first if you enabled it. See
  [upgrading an existing install](docs/getting-started/upgrade.md#upgrading-an-existing-install).
- The decision model stays off until you enable it on the Decision model page or with
  `./install.sh --with-decider`. It needs about 13 GiB of free memory (9 GiB for d3-nano), about 21 GB
  of Docker disk for its image's base and 8.5 GiB in the HF cache for d3-mini.
- Engine start-ups are now queued one at a time. Set `max_concurrent_startups: 0` in `config.yaml` for
  the old behaviour (not recommended on unified memory).
- Backups: the decider's settings, the decision log and diagnoses are in `data/dgxmm.sqlite3`, its key
  in `data/secrets/decider_api_key`; the model snapshot and image are re-created with
  `decider download` / `decider build` (see [backup and restore](docs/operations/backup.md)).

## [2026-10-16]

Weights management, the single LiteLLM endpoint and the processes view.

### Added

- **Delete downloaded weights** from **Models → On disk**: a whole Hugging Face repo (every revision
  and its lock directory) or a GGUF file with all its split shards. The dialog previews the space
  freed, the host paths and the deployments that use the model, and asks you to type the model's name.
- **Clean up incomplete downloads**, per row or all at once (**Clean up incomplete (N · size)**).
  Complete models are never touched; downloads written in the last 10 minutes are skipped.
- Root-owned files left in the HF cache by engine containers are removed by a short-lived helper
  container (no network, read-only root filesystem, one bind mount, minimal capabilities, label
  `dgxmm.helper=true`). New config keys `helper_image` and `helper_timeout_s`.
- API: `GET /api/inventory/delete-preview`, `DELETE /api/inventory?ref=`,
  `GET /api/inventory/cleanup-incomplete/preview`, `POST /api/inventory/cleanup-incomplete`.
- **LiteLLM page** (Integrations > LiteLLM, admins): internal URL, public endpoint URL, write-only
  master key, on/off switch, **Test connection**, **Sync now**, the models LiteLLM serves and
  **virtual keys** (create with models, budget, rate limits and expiry; edit; block; revoke).
- Runtime settings stored in DGXMM's database (new `app_settings` table). Values saved on the LiteLLM
  page win over `config.yaml` and apply without a restart.
- API: `GET/PUT /api/integrations/litellm` (with `?probe=false` for an immediate answer),
  `GET /api/integrations/litellm/status`, `POST …/test`, `POST …/sync`, `GET …/models` and
  `GET/POST/PATCH/DELETE …/keys`.
- **API access** card on a deployment's Overview tab (endpoint, model name, `curl` and Python
  snippets) and a **LiteLLM** tab (model name, registration status, **Re-sync**, keys with access,
  create a key for this deployment). API: `GET /api/deployments/{id}/litellm`.
- Rename a deployment's model name with `PATCH /api/deployments/{id}` (`{"alias": …,
  "rewrite_keys": true}`); virtual keys limited to the old name can be rewritten to the new one.
- New config key `litellm_public_url`, the base URL clients use (shown in the UI as `<url>/v1`).
- **Bundled LiteLLM**: `./install.sh --with-litellm` (or `DGXMM_WITH_LITELLM=1`) runs LiteLLM and
  Postgres in the compose profile `litellm`, with its own master key
  (`secrets/litellm_bundled_master_key`), salt key and database password, and its data in
  `data/litellm-db`. On a fresh install DGXMM is wired to it and `litellm_public_url` is set. New
  installer option `--litellm-port`.
- **Processes page** (Operate > Processes): every process using memory, by container or by process,
  with scope filter (all, external, DGXMM-managed), search and auto-refresh. The dashboard's external
  memory card shows container, user, command and the RAM / GPU split. API: `GET /api/host/processes`.
- Host `/etc/passwd` is mounted read-only (new config key `host_passwd`) to show host user names.

### Changed

- **One LiteLLM registration per deployment**, under its model name (the alias, or the served name).
  Older versions registered an aliased deployment twice; the first sync after upgrading removes the
  extra served-name entry. Only entries DGXMM created (`model_info.dgxmm_id`) are changed.
- **Model names are unique.** Deploy, start, wake and rename are refused with
  `409 model_name_taken` when another deployment holds the name; the UI offers a suggestion.
- Turning the LiteLLM integration off, or pointing it at another URL, first removes DGXMM's models
  from the LiteLLM being left.
- Deployment views report the LiteLLM registration error per deployment (`litellm_error`).
- Data tables keep their row actions pinned to the right edge; the Models table hides Architecture and
  Context on narrower screens.

### Security

- The LiteLLM master key is write-only, never logged or audited, and bound to the URL it was entered
  for; the secret-file key is only used with the default `litellm_url`. LiteLLM error texts are
  scrubbed of keys.
- The bundled LiteLLM never shares a master key with an external LiteLLM. Its secrets are never
  printed, and the installer redacts keys, bearer tokens and URL credentials in the LiteLLM and
  database log lines it shows on failure.
- Process command lines are masked before they leave the probe, failing closed on values that cannot
  be parsed; process environments are never read. Fixed cases where wrapper options (for example
  `docker exec -u …`, `sshpass`) or nested shells could leak a secret.
- Weight deletion uses path-based in-use checks and closes races with concurrent deploys, restarts and
  downloads (`409 model_being_deleted`).

### Fixed

- Delete preview reports downloads written in the last few minutes; a single **Clean up** is blocked
  with the reason.
- Process names padded with NUL bytes are displayed without trailing empty arguments.
- The LiteLLM page shows its settings immediately and fills in the LiteLLM version when a slow probe
  finishes; the LiteLLM model list is cached for 15 seconds.
- A newly created virtual key stays visible across refetch errors until the dialog is closed.

## 2026-10-16 - UI revamp, installer and documentation

The redesigned UI, the installer and the first documentation set.

### Added

- **New UI** built on shadcn/ui and Tailwind: light, dark and system themes; grouped sidebar, top bar
  with a memory chip and mobile tab bar; command palette (<kbd>⌘K</kbd> / <kbd>Ctrl K</kbd>);
  responsive data tables that become cards on phones; theme-aware charts with a table view.
- Redesigned pages: sign-in and change password, dashboard (memory hero, KPI tiles, deployments
  table), deployment detail with tabs (overview, performance, advice, benchmarks, events) and a logs
  sheet, deploy wizard with memory check and plans, switch and reconfigure dialogs, models, catalog with
  a details sheet, downloads with speed and ETA, planner, users and audit log. Routes are lazy-loaded.
- **One-command installer** `./install.sh` (`install`, `upgrade`, `status`, `logs`, `uninstall
  [--purge]`, `--dry-run`) with host checks, generated `.env` and `config.yaml`, and a portable
  `docker-compose.yaml`. `config.example.yaml` documents every key, with host-portable defaults.
- `/` redirects to `/dgxmm/`.
- Admin command line: `python -m dgxmm.admin reset-password <user>` and `list-users`.
- `trusted_proxies` setting (`DGXMM_TRUSTED_PROXIES` in `.env`), and an installer hint for reverse
  proxies on loopback installs.
- User documentation: README, usage, configuration, operations and architecture guides, and
  CONTRIBUTING.

### Changed

- The default vLLM image is the public multi-arch `vllm/vllm-openai`, pinned by digest.

### Security

- `X-Forwarded-For` is trusted only from configured proxies (default: loopback), so client IPs in rate
  limiting and logs cannot be spoofed.
- The user list is admin-only.

### Fixed

- Re-planning after a `409` in the deploy flow, stale previews, accessibility labels, mobile layouts and
  the command palette confirmation.
- First-run robustness of the installer: the admin password is captured only from a container
  created in this run, busy ports are detected reliably (including IPv6 listeners), and failed image
  pulls give a hint.

## 2026-10-15 - Initial release

The first version: deployments, memory planning, LiteLLM registration and the web UI.

### Added

- FastAPI backend in a single container with SQLite storage, argon2id authentication, sessions, CSRF
  and login rate limiting, `admin` and `viewer` roles and an audit log.
- Memory estimator for vLLM and llama.cpp, host memory probes (`/proc`, cgroup v2, NVML) and a memory
  budget with a safety margin, GB10 unified-memory accounting and a watchdog with optional
  `auto_protect`.
- vLLM and llama.cpp engines with a flag allowlist, a Docker guard (`dgxmm.managed=true`), the
  deployment lifecycle (deploy, stop, start, sleep, wake, remove) and startup reconciliation.
- Fit plans, switching with rollback, LiteLLM registration, adoption of unmanaged vLLM containers.
- Model inventory for the HF cache and GGUF directories; Hugging Face catalog with llmfit scores,
  downloads with a disk-space guard, and the co-location planner.
- Live and 30-day metrics, calibration from measured memory, tuning advice and benchmarks, and the
  end-to-end validation script `scripts/validate_on_box.sh`.
- React frontend and Docker packaging.
