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

- **Multi-node deployments, 0.2.0 (2026-11-16).** Without a `nodes:` key in `config.yaml` DGXMM keeps managing
  the one host it runs on; the points below still apply to an existing single-node install.
  - *Migrations 41–44 run once at the first start*: 41 adds `deployments.node` (existing deployments get
    `local`); 42 adds `deployments.engine_api_key` (existing rows stay empty until their next start); 43 adds
    `deployments.placement` and `deployments.ranks` and backfills them (`{"kind": "single", "nodes": [<node>]}`
    and `[]`); 44 adds `downloads.kind` and `downloads.node` (empty for existing downloads, which are Hub
    downloads). Take a [backup](../operations/backup.md) first.
  - *Engine API keys.* Every engine started from now on gets its own API key, and LiteLLM and DGXMM's own checks
    send it. Running engines keep serving without a key, but once a deployment restarts (Stop/Start, Restart,
    Reconfigure, a wake from a stop) its direct engine URLs (`127.0.0.1:<port>`, the container name on the
    DGXMM network) answer `401` without that key, which DGXMM never shows. Send clients through LiteLLM.
  - *vLLM Sleep now stops the model.* The new setting `vllm_sleep` defaults to `stop`: on DGX Spark's unified
    memory vLLM's level-1 sleep frees nothing and fails. Sleep on a single-node vLLM deployment then stops it
    (after a confirmation), and Wake starts it again: a cold start that takes as long as a normal start, not a
    fast wake. Sleep schedules therefore stop and start their deployments. Engines started from now on get
    neither `--enable-sleep-mode` nor vLLM's development routes. On a host with separate GPU memory, set
    `vllm_sleep: level1` in `config.yaml` to keep vLLM's own sleep, then `docker restart dgxmm`.
  - *Engine health probe, on by default.* Every `engine_health_interval_s` (30 s) DGXMM sends `GET /health` to
    each ready deployment's engine; after `engine_health_failures` (3) failures in a row the deployment is
    failed, its containers removed and its LiteLLM entry dropped. To turn it off, set
    `engine_health_interval_s: 0` in `config.yaml` and `docker restart dgxmm`. See
    [health checks](../user-guide/deployments.md#health-checks).
  - *Stored memory estimates.* The estimator now splits a tensor-parallel deployment's KV cache only as far as
    vLLM does (MLA and single-KV-head caches are needed in full on every rank) and models DeepSeek-V4's KV pool.
    A deployment keeps the estimate stored at its last start, with the old even KV split, until it is prepared
    again: its next Start, Restart, Reconfigure or wake from a stop.
  - *llama.cpp RPC image.* Only llama.cpp cluster deployments need it (`images.llamacpp_rpc`, default
    `dgxmm-llamacpp-rpc:61dbf49-u1`). It is built locally, not pulled: on the node running DGXMM run
    `docker build -t dgxmm-llamacpp-rpc:61dbf49-u1 deploy/llamacpp-rpc`, then load it on every remote node with
    `docker save dgxmm-llamacpp-rpc:61dbf49-u1 | ssh dgxuser@192.0.2.2 docker load`. See
    [configuration](../configuration/index.md#nodes).
  - *New `config.yaml` keys* take their defaults: `nodes`, `node_ssh_key_file`, `remote_probe_interval_s`,
    `remote_dev_endpoints`, `cluster_port_range`, `images.llamacpp_rpc`, `images.probe`, `vllm_sleep`,
    `engine_health_interval_s` and `engine_health_failures`.
  - *Rolling back* to a release before multi-node support: first stop and remove every deployment on a remote
    node and every cluster deployment, and let weight copies finish or cancel them (older versions know only the
    local node and would try to run them there). Then go back to the previous image: retag it if you kept it
    (`docker tag <previous image> dgxmm:latest` and `docker compose up -d dgxmm`), or
    `git checkout <previous commit> && ./install.sh`. Finally restart every running engine (Stop, then Start):
    engines started by 0.2.0 require their engine key, which older versions neither send nor give LiteLLM.
    Older versions ignore the new columns.
- **On-demand models (0.2.0).** Migration 40 adds `deployments.mode` and `idle_timeout_s` at the
  first start; every existing deployment becomes *always on*, so nothing changes until you switch one.
  New deployments are always on unless you pick on demand (the deploy wizard preselects it when LiteLLM
  is enabled and synced; the API needs `"mode": "on_demand"`). `./install.sh upgrade` generates `secrets/dgxmm_ondemand_key` and
  mounts it into the container. If LiteLLM reaches DGXMM through a reverse proxy rather than on the Docker
  network, or proxies in front of LiteLLM have short read timeouts, read
  [on-demand troubleshooting](../operations/troubleshooting.md#requests-to-an-on-demand-model-fail-after-about-60-seconds)
  before you switch a deployment to on demand.
- **GPU memory attribution (2026-10-21).** `docker-compose.yaml` now runs the DGXMM container with
  `pid: host`; `./install.sh upgrade` recreates it and adds `DGXMM_HOST_PID=1` to `.env` (set `0` there
  and re-run to opt out). Afterwards a ready deployment's **Measured** memory
  includes its GPU memory and *external* drops by the part that was really the deployment's. Nothing else
  changes, and no migration runs.
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
  row per deployment and hour), 30 (the `deployments` rebuild with `AUTOINCREMENT`, see above), 40
  (on-demand mode), 41 (deployment node), 42 (engine API key), 43 (placement and ranks, backfilled) and 44
  (download kind and target node), see above. They were tested on databases of the previous release and
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
