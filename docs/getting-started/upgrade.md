---
description: "Upgrade DGXMM, move a clone from before the history rewrite and what changes in an existing install."
---

# Upgrade

```bash
./install.sh upgrade
```

This runs `git pull --ff-only` and then the *updated* installer, which rebuilds the image and recreates
the `dgxmm` container. Your `.env`, `config.yaml`, `data/` and `secrets/` are kept. Engine containers keep
serving during the restart. DGXMM re-attaches to them on startup (see
[reconcile](../architecture/background-loops.md)). Settings you passed as flags are passed on
to the new installer.

The same by hand, from your checkout:

```bash
cd dgxmodelmanager            # your checkout
umask 022                     # the image build needs others-readable files (the installer checks)
git pull --ff-only
./install.sh
```

`.env`, `config.yaml`, `secrets/` and `data/` are untracked (gitignored), so `git` never changes them.
Take a [backup](../operations/backup.md) first if the release notes mention database migrations.

If the pull fails because you edited tracked files (for example `docker-compose.yaml`), move your
changes into `docker-compose.override.yaml` (see [configuration](../configuration/models-and-engines.md#model-directories)),
run `git checkout -- docker-compose.yaml`, and upgrade again. New keys in `config.example.yaml` are not
added to your `config.yaml`. They take their defaults until you add them.

## Clones from before the 2026-10-18 history rewrite

The repository's history was rewritten on 2026-10-17 and again on 2026-10-18 (to drop old screenshots
from every commit), so a clone made before 2026-10-18 no longer shares history with `origin/main`: `git pull --ff-only` and `./install.sh upgrade` stop with "Not possible to
fast-forward" (or "refusing to merge unrelated histories"). Move the clone to the new history once:

```bash
cd dgxmodelmanager            # your checkout
git status --short            # tracked files you changed show up here; reset discards them, so save them first
umask 022
git fetch origin
git reset --hard origin/main
./install.sh
```

`git reset --hard` only rewrites tracked files. Your `.env`, `config.yaml`, `secrets/`, `data/` (the
database and the bundled LiteLLM's `data/litellm-db`) and `docker-compose.override.yaml` are untracked,
so they are kept, and the installer picks them up as before. Running engine containers are not affected.
Later upgrades work with `./install.sh upgrade` again.

**With the decision model.** The decider container is not touched by an upgrade, so it keeps answering
from the image and key it was started with. When a release changes the decider image (its tag in
`images.decision`, currently `dgxmm-decider:3`), DGXMM logs "the running decider predates this DGXMM
build" at start and the Decision model page shows a banner with the reasons. With
`DGXMM_WITH_DECIDER=1` in `.env` the installer builds the new image during the upgrade; otherwise build
it on the page (**Build image…**) or with `docker exec dgxmm python -m dgxmm.admin decider build`. Then
press **Restart with a new key** (or run `docker exec dgxmm python -m dgxmm.admin rotate-decider-key`):
the old container is stopped and a new one starts from the new image with a new key. Until then the old
decider keeps working.

## Upgrading an existing install

What changes when an older install is upgraded with `./install.sh upgrade` (see the
[changelog](../changelog.md) for each release). Nothing needs to be done by hand unless a point below
applies to you.

- **Advanced decision-model release (2026-10-18).**
  - *Migrations run once at the first start*: the tables `insights`, `deployment_schedules`,
    `traffic_profiles` and `compat_cache` and the column `deployment_events.actor` are added, then
    migrations 3 (one live insight per dedupe key), 20 (one traffic-profile row per deployment and hour)
    and 30. Migration 30 rebuilds the `deployments` table so that ids are never reused: in one
    transaction, with foreign keys switched off for that step only and `PRAGMA foreign_key_check`
    compared before and after; rows, events, diagnoses and other child rows are kept, and the id sequence
    starts above every id still referenced. It is quick on a typical database. If it
    fails, it is rolled back, the error is logged and DGXMM does not start (the next start retries).
  - *Time zone*: re-running the installer writes `DGXMM_TZ` (detected from `timedatectl`,
    `/etc/timezone` or the `/etc/localtime` link) to `.env`. Without it DGXMM uses `TZ`, the container's
    `/etc/localtime`, else UTC; an `/etc/localtime` that is a copy rather than a link is not recognised by
    the installer, so set `DGXMM_TZ` yourself (an IANA name such as `Europe/Berlin`).
  - *New background jobs start* right after the upgrade (health checks, reading labels, nightly profiles
    and proposals, daily routing advice, weekly tuning, the schedule runner). With the decision model
    disabled they run their rules only and ask nothing; their findings appear under **Insights**. No
    schedule exists until an admin creates one, so nothing is slept or stopped by the upgrade.
  - *Stops recorded before the upgrade* have no `actor`, so a schedule never wakes a deployment that was
    stopped before the upgrade.
  - *Rolling back* (`git checkout <previous commit> && ./install.sh`) works on the migrated database:
    older code ignores the new tables and column and keeps ids unique.
- **Decision model release.** The decision model is off after the upgrade: nothing new runs and no
  extra memory is used until you enable it (see [decision model](../operations/decision-model.md)). The new tables
  (`decisions`, `diagnoses`, `artifact_kinds`, `cli_tokens`, `schema_version`) and columns
  (`deployments.system`, `deployments.oom_killed`) are added at start. The rule-based parts of the new
  features (failure diagnosis, parser suggestions, plan safety marks, model kinds, workload labels) work
  right away. The installer generates `DGXMM_INSTALL_ID` in `.env`, and only containers and images
  created from then on carry the install label, so `uninstall --purge` reports older engine containers
  and keeps them. Engine starts are now queued one at a time (`max_concurrent_startups: 1`).
- **A decider started by an older build** keeps running unchanged. If the Decision model page (or
  `GET /api/decider/status` → `upgrade`) says it needs a restart, see [upgrade](#) and
  [security](../security.md#upgrading-a-decider-started-by-an-older-build). A decider network created
  before install labels existed is reported and kept by `uninstall --purge`
  (`docker network rm dgxmm-decider` removes it by hand).
- **Database.** The new `app_settings` table, which holds the settings saved on the LiteLLM page, is
  created automatically when DGXMM starts. There is no migration step. Until you save something on
  the LiteLLM page, DGXMM keeps using `litellm_url` and `secrets/litellm_master_key` exactly as before.
- **One LiteLLM entry per deployment.** Older versions registered a deployment with an alias twice:
  under its served name and under the alias. The first sync after the upgrade deletes the served-name
  entry and keeps the alias. Only entries DGXMM created (those with `model_info.dgxmm_id`) are changed;
  models you added to LiteLLM yourself are never touched. Clients that called the served name of an
  aliased deployment must switch to the model name shown on the deployment's **API access** card.
- **Model names must be unique.** If two of your deployments share a model name (for example two
  deployments of the same model without aliases), deploy, start, wake and rename are refused with
  `409 model_name_taken`, naming the deployment that holds the name. If both are already ready, the
  oldest keeps the registration and the other shows LiteLLM status `error`. To resolve it, give one of
  them a different **Model name (LiteLLM)** on its LiteLLM tab (the error suggests
  `<served name>-<deployment name>`), or with
  `PATCH /api/deployments/{id}` and `{"alias": "<new name>"}`. Stopped deployments without an alias do
  not hold their served name.
- **New read-only mount.** `docker-compose.yaml` now mounts the host's `/etc/passwd` read-only at
  `/host/etc/passwd`, only to show host user names on the Processes page (no other account file is
  mounted). The upgrade recreates the container with it. If you replaced the volumes in a
  `docker-compose.override.yaml`, add the mount there too, or accept numeric uids.
- **Bundled LiteLLM later.** To opt in on an existing install, run `./install.sh --with-litellm` (it is
  remembered in `.env`). Your `config.yaml` and `secrets/litellm_master_key` are kept, so DGXMM keeps
  registering with the LiteLLM it already uses. The bundled LiteLLM always gets its own key in
  `secrets/litellm_bundled_master_key`, so the keys of the two LiteLLMs stay separate. To move DGXMM to
  the bundled one, open **Integrations > LiteLLM**, set the **Internal URL** to
  `http://dgxmm-litellm:4000`, enter the bundled master key and save. DGXMM first removes its models
  from the LiteLLM it is leaving, then registers them with the bundled one. Virtual keys of the old
  LiteLLM are not copied. If another LiteLLM already publishes port 4000, use `--litellm-port`.
- **Database migrations.** Startup upgrades `data/dgxmm.sqlite3` in place, in this order: new tables
  are created, new nullable columns and new non-unique indexes are added to existing tables, then the
  versioned migrations run, each once, in its own transaction, recorded in the `schema_version` table
  (`SELECT * FROM schema_version`). A failed migration is rolled back completely, the error is logged
  and DGXMM does not start; the next start retries it. Migrations so far: 1 (baseline), 2 (one
  diagnosis per failure event: a unique index on `diagnoses (deployment_id, event_id)`; duplicates, if
  any, are removed and the newest is kept), 3 (one live insight per dedupe key), 20 (one traffic-profile
  row per deployment and hour) and 30 (the `deployments` rebuild with `AUTOINCREMENT`, see above). They were tested on databases of the previous release and
  of this one. **Rolling back** to an older checkout (`git checkout <old> && ./install.sh`) keeps
  working: migrations only add tables, columns and indexes, and older versions ignore them
  (`schema_version`, `cli_tokens`, `decisions`, `diagnoses`, `artifact_kinds`); the previous release
  was checked to read and write a migrated database. A database that a newer DGXMM migrated is reported
  in the log ("written by a newer DGXMM") and left unchanged. Before rolling back to a release without
  the decision model, disable and remove it (`docker exec dgxmm python -m dgxmm.admin decider disable
  --remove`, or `POST /api/decider/disable` with `{"remove": true}`): older versions do not know the
  system deployment. Take a [backup](../operations/backup.md) before upgrading.
- **Decision model later.** Run `./install.sh --with-decider` (remembered in `.env`), or enable it in the
  UI (**Integrations > Decision model**). See [decision model](../operations/decision-model.md).
- **Pinned llama.cpp image.** The default `images.llamacpp` is now pinned by digest
  (`ghcr.io/ggml-org/llama.cpp@sha256:…`, build b11669) instead of the moving `server-cuda` tag. A
  `config.yaml` written by an earlier install still names `ghcr.io/ggml-org/llama.cpp:server-cuda` and
  keeps using it. To use the pin, copy the `images.llamacpp` line from `config.example.yaml` into your
  `config.yaml` (or delete yours to take the default), then `docker restart dgxmm`. Running llama.cpp
  containers keep the image they were created from; engine containers created after that use the pin. The bundled LiteLLM's Postgres is pinned too
  (`postgres:16.15-trixie`, the same major version, so `data/litellm-db` is used as is).
- **New `config.yaml` keys** take their defaults: `litellm_public_url`, `helper_image`,
  `helper_timeout_s`, `host_passwd`, `trusted_proxies`, and in the decision-model release the start-up
  queue (`max_concurrent_startups`, `startup_queue_timeout_s`, `wake_queue_timeout_s`,
  `wake_sync_wait_s`), `images.decision` and the `decider_*` keys (see
  [configuration](../configuration/index.md#decision-model-decider)). Set **Public endpoint URL** on the LiteLLM
  page, or `litellm_public_url`, so the UI shows the endpoint your clients use.

## Next steps

- [Changelog](../changelog.md)
- [Backup and restore](../operations/backup.md)
- [Uninstall](uninstall.md)
