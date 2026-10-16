# Changelog

All notable changes to DGXMM are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). DGXMM does not publish tagged releases yet
(the package version is `0.1.0`), so each section below is identified by its date and the commits it
covers. Upgrade notes for existing installs are in
[operations](docs/operations.md#upgrading-an-existing-install).

## [Unreleased] - 2026-10-16

Weights management, the single LiteLLM endpoint and the processes view (commits `d1b2c2d`…`5c25e6f`).

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

Commits `69e026a`…`fef8cf7`.

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

Commits `37e1708`…`00bc53a`.

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
