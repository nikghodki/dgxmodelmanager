# DGXMM: weight management, LiteLLM integration, process visibility — design

Date: 2026-10-16 · Branch: `feature/litellm-weights-processes` (from main 9ebca65)

## User requests (verbatim intent)
1. Option to delete downloaded weights from the DGX.
2. LiteLLM should expose all models under a single LiteLLM endpoint, selected by the `model` attribute — no separate endpoint per model.
3. LiteLLM deeply integrated in the app: user can set the endpoint hostname, configure virtual keys, etc. for given deployments.
4. If incomplete model weights are found, an option to clean them up from the DGX.
5. The running-processes view must show exactly what is being run.

The user asked not to be asked questions; decisions below are rulings.

## Global constraints
- Backend FastAPI (`dgxmm/`), SQLite via SQLAlchemy (`Base.metadata.create_all` — new tables are created automatically; adding columns to existing tables needs a tiny idempotent migration helper because there is no migration framework).
- Frontend React + shadcn/ui design system already in place (`frontend/src/components/app/*`, DataTable, ConfirmDialog with `requireText`, ResponsiveDialog, useActionMutation, StatusBadge, PageHeader, EmptyState/ErrorState, ChartCard). Follow existing patterns; light/dark; mobile (<768px) layouts; status never colour-only; icon-only buttons have aria-label; no window.confirm/prompt/alert.
- All mutating endpoints: `require_admin`, CSRF (existing deps), audit log entry (`dgxmm/audit.py`). Viewers may read non-sensitive data only.
- Never return secrets (LiteLLM master key, full virtual keys except once at creation) to the client after the fact; never log them.
- Tests: pytest for backend (currently 286 pass), vitest for frontend (116 pass). TDD: failing test first.
- Run npm only in docker `node:22-bookworm-slim`. Repo + env rules: /tmp/dgxmm-f3/context.md.

## 1 + 4. Weights deletion and incomplete cleanup

**What can be deleted:** any inventory artifact (`/inventory` item):
- HF cache model (`ref` = `org/name`): delete the whole repo directory `<hf_cache>/hub/models--org--name` (blobs, snapshots, refs) — all revisions. Also remove a matching `.locks/models--org--name` directory if present.
- GGUF (`ref` = absolute container path to a .gguf): delete that file; if it is a split GGUF (`*-00001-of-0000N.gguf`) delete all its sibling shards; delete a now-empty parent directory only if it is not a configured gguf root.

**Safety rules (server-enforced, tested):**
- Refuse (409 `model_in_use`) if any deployment with this `model_ref` is in a non-terminal state (anything except `stopped`, `failed`, `exited`, `removed`/absent). Response lists the blocking deployments (id, name, state). Stopped/failed deployments that reference it are returned as `warnings` in a dry-run preview; deletion proceeds if the user confirms (those deployments will fail to start; UI says so).
- Refuse (409 `download_active`) if a download for the same repo is queued/running.
- Path containment: the resolved target must be inside the configured `hf_cache.local`/`hub` or a configured `gguf_roots[*].local`; refuse symlink escapes (`Path.resolve()` then `is_relative_to`). Never delete a configured root itself.
- Two-step API: `GET /inventory/delete-preview?ref=` → `{ref, paths (host paths), bytes, deployments_blocking, deployments_affected, download_active}`; `DELETE /inventory?ref=` (body or query) performs it. Admin only, audited (`model_delete` with ref + bytes).
- Ownership: some HF cache files may be owned by root (seen on this box), so in-container `rm` as uid 1000 can fail. Deletion runs through a short-lived helper container started via the Docker API that DGXMM already has: image = the DGXMM image itself (`dgxmm:latest`, or settings-configurable), `user: "0:0"`, bind-mount ONLY the target's parent host directory, command `rm -rf -- <target-inside-mount>` (or `rm -f` for files), label `dgxmm.helper=true`, auto-remove, network none, 120 s timeout. Host paths come from the configured host/local mappings (hf_cache.host ↔ local, gguf_roots[i].host ↔ local). Fallback when the helper can't run: in-process `shutil.rmtree` and report any permission errors with a hint (`sudo chown -R <uid>:<gid> <path>`).
- After deletion: trigger an inventory refresh and return the freed bytes.

**Incomplete cleanup:** incomplete artifacts are those with `complete == false` (interrupted downloads, `.incomplete` blobs, missing snapshot files/shards). `POST /inventory/cleanup-incomplete` with `{refs?: string[]}` (default: all incomplete) deletes each eligible one with the same rules (skips ones with active downloads or blocking deployments and reports them). `GET /inventory/cleanup-incomplete/preview` returns the list with sizes. Admin only, audited.

**UI (Models page):**
- Row action "Delete…" (admin) → ResponsiveDialog/ConfirmDialog showing the preview (size, paths, affected deployments warning, blocking reasons disabling the confirm), typed confirmation of the short model name.
- Incomplete rows: a "Clean up" action instead of Deploy; header button "Clean up incomplete (N · X GiB)" when N > 0 → dialog listing them with checkboxes (all checked), skips explained.
- Toasts with freed space; table refreshes.

## 2 + 3. LiteLLM — single endpoint and deep integration

**Single endpoint model:** every ready DGXMM deployment is registered in LiteLLM exactly ONCE, under its *model name* = `alias` if set else `served_name`. (Today it is registered twice: served_name and alias.) All models are reached through one OpenAI-compatible endpoint: `<litellm public base URL>/v1` with `"model": "<model name>"`. The UI presents that endpoint + model name as the primary way to call a deployment; per-container direct endpoints move to a collapsed "Direct engine endpoints (advanced)" section.
- Migration: on sync, models DGXMM previously registered under a second name (`model_info.dgxmm_id` match but name ≠ canonical) are deleted from LiteLLM.
- Changing a deployment's alias (new `PATCH /deployments/{id}` `{alias}` or via the LiteLLM tab) re-registers under the new name. Alias rules: `^[A-Za-z0-9._:/-]{1,64}$`, unique among deployments.

**Runtime settings store:** new table `app_settings(key TEXT PK, value TEXT, updated_at)`; a small `SettingsStore` service with typed accessors and an in-memory cache. LiteLLM settings:
- `litellm.enabled` (bool, default true if a key is available)
- `litellm.url` (internal URL DGXMM uses; default from config `litellm_url`)
- `litellm.public_url` (what clients use, e.g. `https://llm.example.com` or `http://<dgx-ip>:4000`; default empty → UI shows a hint and falls back to the internal URL)
- `litellm.master_key` (write-only; stored in the DB; if unset, fall back to the secrets file `litellm_key_file`). API returns only `master_key_set: bool` and `master_key_source: "settings" | "secret_file" | "none"`.
- Precedence: settings table > config.yaml/env. The LiteLLM client reads settings at call time (no restart needed).

**API (`/integrations/litellm`, admin unless noted):**
- `GET /integrations/litellm` (viewer-readable minus secrets): `{enabled, url, public_url, endpoint: public_url||url + "/v1", master_key_set, master_key_source, status: {reachable, version?, error?, checked_at}}`.
- `PUT /integrations/litellm` `{enabled?, url?, public_url?, master_key?}` (empty string master_key = clear → fallback). Validates URLs (http/https). Audited (never the key).
- `POST /integrations/litellm/test` → tries `/health/liveliness` (or `/health/readiness`) and an authenticated `/model/info`; returns reachable/auth_ok/version/error.
- `POST /integrations/litellm/sync` → immediate resync; returns registered/removed counts.
- `GET /integrations/litellm/models` → models known to LiteLLM, each flagged `managed_by_dgxmm` (+ deployment id/name/state) or external.
- Virtual keys (proxied to LiteLLM admin API with the master key; LiteLLM must have a DB — if LiteLLM reports no DB, return 409 `litellm_no_db` with a helpful message):
  - `GET /integrations/litellm/keys` → list (key_alias, token hash/prefix, models, spend, max_budget, budget_duration, rpm_limit, tpm_limit, expires, created_at, blocked). Never the full key.
  - `POST /integrations/litellm/keys` `{key_alias, models: string[] (model names; empty = all models), max_budget?, budget_duration?, rpm_limit?, tpm_limit?, duration? (e.g. "30d"), metadata?}` → returns the full key ONCE.
  - `PATCH /integrations/litellm/keys/{token}` (models, limits, alias, blocked).
  - `DELETE /integrations/litellm/keys/{token}`.
  - All audited (`litellm_key_create/update/delete`, alias + models, never the key).
  - Keys created by DGXMM carry `metadata.created_by = "dgxmm"` and the DGXMM username.
- Per deployment: `GET /deployments/{id}/litellm` → `{model_name, endpoint, registered, status, keys: [keys whose models include model_name or are unrestricted]}`.

**Bundled LiteLLM for fresh installs:** `./install.sh --with-litellm` (or `DGXMM_WITH_LITELLM=1`) enables a compose profile `litellm` with services `dgxmm-litellm` (pinned LiteLLM image by digest/version) and `dgxmm-litellm-db` (postgres:16, volume under `data/litellm-db`), generated `LITELLM_MASTER_KEY` + `LITELLM_SALT_KEY` in `secrets/` (600), `STORE_MODEL_IN_DB=True`, a minimal litellm config file, port `${DGXMM_LITELLM_PORT:-4000}` bound to `${DGXMM_BIND}`. DGXMM's config/settings then point `litellm_url` at `http://dgxmm-litellm:4000` and the master key at the generated secret; the installer prints the single endpoint URL. Idempotent like the rest; `uninstall --purge` removes them; existing external LiteLLM setups are untouched when the flag isn't given.

**UI:**
- New admin page **LiteLLM** (nav group "Integrations", icon `Network` or `KeyRound`; also in ⌘K): sections (tabs or cards): *Connection* (form: internal URL, public endpoint URL with live preview `…/v1`, master key field "••••• set (from secret file)" + replace/clear, enabled switch, Test connection button with result, Sync now), *Models* (table of LiteLLM models: name, managed deployment link/state or "external", api_base host hidden for viewers), *Virtual keys* (DataTable + Create key dialog: alias, model multi-select from DGXMM model names + "All models", budget + duration, rpm/tpm, expiry; created-key dialog shows the key once with CopyButton and a curl example; Edit; Revoke with ConfirmDialog).
- **Deployment detail**: Overview "API access" card replaces the endpoints card: Endpoint (single LiteLLM `/v1` URL, copy), Model (name, copy), example `curl` + Python (OpenAI SDK) snippets with copy, LiteLLM status badge; "Direct engine endpoints (advanced)" collapsible with the old served name / gateway / host endpoints. New **LiteLLM** tab: model name (editable alias), registration status + re-sync, keys with access (list) and "Create key for this deployment" (pre-selects the model).
- Deployments table "LiteLLM" column shows the model name + status.
- Deploy wizard: rename "LiteLLM alias" field to "Model name (LiteLLM)" with help "Clients call `<endpoint>` with model: <name>. Defaults to the served name."

## 5. Processes — show exactly what runs

**Backend:** extend the host probe to return, for each process: `pid, ppid, user (from /proc/<pid>/status Uid → name via host /etc/passwd if mounted, else uid), command (argv from /proc/<pid>/cmdline joined with shell-style quoting, redacted), exe name, container: {id, name, image} | null (resolve cgroup docker id → name/image via the Docker API with a 30 s cache), managed: {deployment_id, name} | null, rss, gpu`. Redaction (unit-tested): values of args whose flag name contains `key|token|secret|password|passwd|auth|credential` (`--api-key X`, `--api-key=X`, `-p X` only when flag matches), `KEY=VALUE` tokens whose key matches, URL userinfo `scheme://user:pass@` → `scheme://***@`, bearer tokens (`Bearer xxx`), HF tokens `hf_[A-Za-z0-9]{20,}`, `sk-[A-Za-z0-9-_]{16,}` → `***`. Commands are truncated at 4 KiB.
- `GET /host/processes?scope=all|external|managed&limit=` (viewer-readable; redacted for everyone) → processes sorted by rss+gpu desc, default limit 50, with group-by-container summaries (`containers: [{id,name,image, rss, gpu, process_count, managed}]`). Keep `external_top` in `/host` backward compatible but enrich each item with `command`, `container`, `user`.
- Kernel threads / zombie / zero-memory processes are excluded; processes inside managed engine containers are flagged `managed`.

**UI:**
- New **Processes** page (Operate group, ⌘K): toggle *By container* / *By process*; DataTable: Command (monospace, wraps, full redacted command in a details row), Container (name + image badge, or "host"), User, RAM, GPU, PID; filter by scope (All / External / DGXMM-managed) and search. Auto-refresh every 5 s (pause toggle).
- Dashboard card "Largest external memory users" → shows container name (or "host") + the command (truncated, title = full redacted command) + RAM/GPU split, and "View all processes" link.

## Testing focus (beyond per-feature tests)
- Deleting a model used by a running deployment is impossible via API (409) and UI (button disabled with reason).
- Path traversal / symlink escape in delete is refused.
- Master key never appears in any API response, log line or audit row (test greps outputs).
- Full virtual key appears only in the create response.
- Redaction tests with realistic vLLM/litellm/python command lines.
- Alias change re-registers once and removes the old name.
