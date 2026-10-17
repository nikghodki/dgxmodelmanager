# Operations

How to install, upgrade, back up, monitor and troubleshoot DGXMM. All commands run in the DGXMM
checkout on the host, as the user that installed it.

- [What the installer does](#what-the-installer-does)
- [Upgrade](#upgrade)
- [Upgrading an existing install](#upgrading-an-existing-install)
- [Changing settings](#changing-settings)
- [Decision model](#decision-model)
- [Backup and restore](#backup-and-restore)
- [Uninstall](#uninstall)
- [Logs](#logs)
- [Health](#health)
- [Freeing disk space](#freeing-disk-space)
- [Admin command line](#admin-command-line)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Security hardening](#security-hardening)

## What the installer does

`./install.sh` (or `./install.sh install`) is non-interactive and idempotent. You can run it again at
any time, and it is also the upgrade path. `--dry-run` (`-n`) prints what it would do and changes
nothing. In order:

1. **Preflight.** The installer:
   - reports the CPU architecture (it warns on anything other than aarch64/x86_64);
   - refuses to run as root unless `--allow-root` is given;
   - checks that `docker` is installed and the daemon is reachable (with a hint if the socket gives
     "permission denied"), that `docker compose` v2 is installed, and that Docker has an `nvidia`
     runtime;
   - checks that `nvidia-smi -L` shows a GPU;
   - stops if another DGXMM container exists that was created from a different checkout (only one
     instance per host);
   - checks that the UI port is free, unless it is used by this DGXMM;
   - warns when less than 50 GiB is free on the HF cache disk;
   - with `--with-decider`: requires at least 16 GiB of `MemAvailable` (skipped when the decider already
     runs), 10 GiB free on the HF cache disk unless the model snapshot is already there, and warns when
     the Docker disk has less than 30 GiB free and the decider image still has to be built.
2. **Settings.** Works out the port, bind address, network, model paths, your uid/gid and the Docker
   socket's group id. Order of precedence: existing `.env`, then flags, then environment, then defaults.
   Paths and values are validated.
3. **Configuration.** Writes `.env` if missing (or appends keys an older `.env` lacks). Renders
   `config.yaml` from `config.example.yaml` if missing. Creates `data/` and `secrets/` (mode 700). Writes
   `secrets/hf_token` and `secrets/litellm_master_key` (mode 600; empty unless `HF_TOKEN` or
   `LITELLM_MASTER_KEY` is set, or the LiteLLM key could be read from a running `litellm-proxy`
   container). Warns if `litellm-proxy` is not on the DGXMM network. With `--with-litellm`, also
   generates the bundled LiteLLM's secrets and, in a fresh `config.yaml`, points DGXMM at it and sets
   `litellm_public_url` (see [bundled LiteLLM](configuration.md#bundled-litellm)). Existing files are
   always kept.
4. **Host directories.** Creates the HF cache, the models directory, `~/.cache/vllm`,
   `~/.cache/flashinfer` and `~/.triton` if missing.
5. **Docker.** Creates the Docker network (labelled `dgxmm.created=true`) if it does not exist. With
   `DGXMM_PULL_ENGINES=1`, pulls the engine images from `config.yaml`.
6. **Build and start.** `docker compose up -d --build` builds the image (frontend and backend) and
   (re)creates the `dgxmm` container. Running engine containers are not touched.
7. **Health wait.** Waits up to `DGXMM_HEALTH_TIMEOUT` seconds (default 300) for the container's health
   check. If the container is unhealthy or exits, the installer shows the last 40 log lines (with the
   password redacted) and fails.
8. **Summary.** Prints the URLs: every LAN IPv4 address, `<hostname>.local` and `<hostname>`, or a
   loopback URL for `127.0.0.1` binds. On the very first start it also prints the generated `admin`
   password once and saves it to `secrets/admin_password`. On a loopback install without
   `DGXMM_TRUSTED_PROXIES`, it prints a reverse-proxy hint with the Docker network's gateway address.
9. **Bundled LiteLLM** (only with `--with-litellm`). Waits up to `DGXMM_LITELLM_HEALTH_TIMEOUT` seconds
   for LiteLLM's `/health/liveliness`, then prints the single endpoint (`http://<host>:<port>/v1`) and
   how to create API keys. If LiteLLM or its database does not start, it shows their redacted logs and
   exits non-zero; DGXMM itself is already running at that point.
10. **Decision model** (only with `--with-decider`). See [decision model](#decision-model): builds the
    decider image, downloads and verifies the pinned model snapshot, enables the decider and waits up to
    `DGXMM_DECIDER_TIMEOUT` seconds (default 900) until it is healthy, then prints its status. Each step
    is skipped when already done. On failure it says which step failed; DGXMM itself keeps running.

Other commands: `./install.sh status`, `logs [-f]`, `upgrade` and `uninstall [--purge]`. The `Makefile`
has the same targets (`make install`, `make status`, …).

## Upgrade

```bash
./install.sh upgrade
```

This runs `git pull --ff-only` and then the *updated* installer, which rebuilds the image and recreates
the `dgxmm` container. Your `.env`, `config.yaml`, `data/` and `secrets/` are kept. Engine containers keep
serving during the restart. DGXMM re-attaches to them on startup (see
[reconcile](architecture.md#startup-and-background-loops)). Settings you passed as flags are passed on
to the new installer.

If the pull fails because you edited tracked files (for example `docker-compose.yaml`), move your
changes into `docker-compose.override.yaml` (see [configuration](configuration.md#model-directories)),
run `git checkout -- docker-compose.yaml`, and upgrade again. New keys in `config.example.yaml` are not
added to your `config.yaml`. They take their defaults until you add them.

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
[changelog](../CHANGELOG.md) for each release). Nothing needs to be done by hand unless a point below
applies to you.

- **Decision model release.** The decision model is off after the upgrade: nothing new runs and no
  extra memory is used until you enable it (see [decision model](#decision-model)). The new tables
  (`decisions`, `diagnoses`, `artifact_kinds`, `cli_tokens`, `schema_version`) and columns
  (`deployments.system`, `deployments.oom_killed`) are added at start. The rule-based parts of the new
  features (failure diagnosis, parser suggestions, plan safety marks, model kinds, workload labels) work
  right away. The installer generates `DGXMM_INSTALL_ID` in `.env`, and only containers and images
  created from then on carry the install label, so `uninstall --purge` reports older engine containers
  and keeps them. Engine starts are now queued one at a time (`max_concurrent_startups: 1`).
- **A decider started by an older build** keeps running unchanged. If the Decision model page (or
  `GET /api/decider/status` → `upgrade`) says it needs a restart, see [upgrade](#upgrade) and
  [security](security.md#upgrading-a-decider-started-by-an-older-build). A decider network created
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
  and DGXMM does not start; the next start retries it. Migrations so far: 1 (baseline) and 2 (one
  diagnosis per failure event: a unique index on `diagnoses (deployment_id, event_id)`; duplicates, if
  any, are removed and the newest is kept). They were tested on databases of the previous release and
  of this one. **Rolling back** to an older checkout (`git checkout <old> && ./install.sh`) keeps
  working: migrations only add tables, columns and indexes, and older versions ignore them
  (`schema_version`, `cli_tokens`, `decisions`, `diagnoses`, `artifact_kinds`); the previous release
  was checked to read and write a migrated database. A database that a newer DGXMM migrated is reported
  in the log ("written by a newer DGXMM") and left unchanged. Before rolling back to a release without
  the decision model, disable and remove it (`docker exec dgxmm python -m dgxmm.admin decider disable
  --remove`, or `POST /api/decider/disable` with `{"remove": true}`): older versions do not know the
  system deployment. Take a [backup](#backup-and-restore) before upgrading.
- **Decision model later.** Run `./install.sh --with-decider` (remembered in `.env`), or enable it in the
  UI (**Integrations > Decision model**). See [decision model](#decision-model).
- **New `config.yaml` keys** take their defaults: `litellm_public_url`, `helper_image`,
  `helper_timeout_s`, `host_passwd`, `trusted_proxies`, and in the decision-model release the start-up
  queue (`max_concurrent_startups`, `startup_queue_timeout_s`, `wake_queue_timeout_s`,
  `wake_sync_wait_s`), `images.decision` and the `decider_*` keys (see
  [configuration](configuration.md#decision-model-decider)). Set **Public endpoint URL** on the LiteLLM
  page, or `litellm_public_url`, so the UI shows the endpoint your clients use.

## Changing settings

- **`.env` values** (port, bind address, network, model paths): edit `.env`, then `./install.sh`. If you
  change paths or the network, update `config.yaml` to match (see
  [configuration](configuration.md#env-and-installer-variables)).
- **`config.yaml`**: edit it, then `docker restart dgxmm`. The file is mounted into the container and
  read at start; `./install.sh` alone does not restart a container whose image and `.env` are
  unchanged.
- **Secrets**: edit the file in `secrets/`, then `docker restart dgxmm` (see
  [secrets](configuration.md#secrets) for which files are read without a restart).
- **Trusted proxies**: set `DGXMM_TRUSTED_PROXIES` in `.env`, then `./install.sh` (or set
  `trusted_proxies` in `config.yaml`, then `docker restart dgxmm`).
- **LiteLLM connection** (URLs, master key, on/off): change it on the LiteLLM page. It applies
  immediately and wins over `config.yaml` (see
  [configuration](configuration.md#runtime-settings-and-precedence)). Turning the integration off or
  changing the internal URL first removes DGXMM's registrations (models with `model_info.dgxmm_id`) from
  the LiteLLM being left, with that LiteLLM's key; the audit log entry records how many were removed. If
  it was unreachable, the change is saved anyway with a warning: remove the leftover DGXMM models in that
  LiteLLM, or re-enable the integration with that URL so the next save/sync cleans them up.

## Decision model

The decision model ("decider", `vllm-sr/d3-mini` at a pinned revision) is an optional local model that
DGXMM consults for failure diagnosis, parser suggestions, plan recommendations, model kinds and
workload labels. Every feature falls back to its rules when the decider is off or unhealthy. It runs
as a *system* deployment: about 13 GiB of GPU memory, never stopped by plans, never registered in LiteLLM,
and stopped by the memory watchdog before any user deployment (when it uses more than its reservation, or as the
first `auto_protect` victim; the supervisor restarts it later with backoff).

**Install with the installer:**

```bash
./install.sh --with-decider          # or DGXMM_WITH_DECIDER=1 ./install.sh; remembered in .env
```

After DGXMM is healthy the installer runs, inside the `dgxmm` container:

1. `decider build`: builds the decider image (`images.decision`, default `dgxmm-decider:3`) from
   `deploy/decider` with the same code as the UI's **Build image** button. The first build pulls the
   pinned vLLM base (about 21 GB) and takes a few minutes. It is skipped when the image exists.
2. `decider download`: downloads the pinned snapshot into the HF cache (8.5 GiB) with progress,
   resumable after an interruption (re-run the installer). Only files with a known SHA-256 are fetched:
   code, config, tokenizer and `MODEL_MANIFEST.json` from the hash list in `deploy/decider`, then the
   files that manifest lists (weights, licence, card; `assets/` is skipped). Anything else the Hub lists
   is ignored, and unsafe names (absolute, `..`) are refused. Each file is hashed while it downloads and
   only placed in the cache when it matches; a mismatch is discarded, downloaded once more, and if it
   still does not match the step fails with nothing linked. It checks free disk space first, and only
   one download runs at a time. A complete, verified snapshot needs no network. The HF token is never
   sent (the repository is public).
3. `decider enable --wait`: enables the decider through the running app, with the normal memory check.
   If it does not fit, the installer lists the plans that would make room and changes nothing: those
   plans stop or sleep your deployments, so pick one in the UI. Then it waits until the decider is
   healthy.

The installer builds and downloads before enabling (rather than letting the app do it on enable), so
progress shows in the terminal, failures stop the install at the step that failed, and the slow parts
can be retried on their own. Enabling from the UI still offers the in-app image build.

**Admin CLI.** The same commands work by hand:

```bash
docker exec dgxmm python -m dgxmm.admin decider status        # --json, or --check (exit 1 unless healthy)
docker exec dgxmm python -m dgxmm.admin decider build         # --force rebuilds
docker exec dgxmm python -m dgxmm.admin decider download      # --verify-only checks without downloading
docker exec dgxmm python -m dgxmm.admin decider download --model vllm-sr/d3-nano   # the other supported model
docker exec dgxmm python -m dgxmm.admin decider enable --wait # --timeout SECONDS (default 900)
docker exec dgxmm python -m dgxmm.admin decider disable       # --remove also removes the deployment
```

`build` and `download` work on their own. `status`, `enable` and `disable` call the app's API over its
admin unix socket inside the container (`data/run/admin.sock`, mode 0600 in a 0700 directory, served by
a second listener without proxy-header handling) with a short-lived admin token that the command creates
in the database and deletes when it finishes. The token is never printed. It is refused on the TCP port
(whatever `X-Forwarded-For` says) and on any request with forwarding headers, and accepted only on the
decider routes (and the inventory refresh), for at most the wait time plus 10 minutes (so
`--timeout` is at most 10200 s), and only while its user is an admin. It works before the first admin
login (pending password change). The audit log shows the oldest admin as the actor (`--user NAME` picks
another admin), marks these actions `"via": "cli"`, and records each token's issue and revocation.

**Remembered vs. requested.** With `DGXMM_WITH_DECIDER=1` only remembered in `.env`, an install or
upgrade never fails because of the decider: low memory, a failed build, download or enable are warnings
(DGXMM itself is upgraded; the app's memory check decides at enable), and a decider an admin disabled
stays disabled. Passing `--with-decider` on the command line means "set it up now": the memory and disk
checks are fatal, failures exit non-zero, and a disabled decider is enabled again.

**Turning it off:** disable it on the Decision model page (or `decider disable`). The installer leaves it
off; set `DGXMM_WITH_DECIDER=0` in `.env` to stop it trying altogether.

`./install.sh status` shows the decider's state and, with `DGXMM_WITH_DECIDER=1`, exits non-zero while it
is not healthy. The UI's top bar shows a **Decision model** chip (Healthy, Building, Starting, Stopping,
Sleeping, Stopped, Disabled or Error) that opens the Decision model page.

**Install labels.** The installer generates `DGXMM_INSTALL_ID` once (kept in `.env`); DGXMM labels the
containers it creates and the images it builds `dgxmm.install=<id>`, so uninstall touches only this
install's decider container and image. Containers and images from before the label are reported and
kept (remove them by hand).

**Data retention.** The decision log (`decisions`) keeps 30 days and at most the newest 50 000 rows
(`decider_retention_days`, `decider_max_rows` in `config.yaml`); stored diagnoses (`diagnoses`) keep the latest
20 per deployment and are deleted with the deployment. The metrics loop prunes both once a minute (with the
metrics rollups, which keep 30 days). Feedback on a pruned decision answers 404.

## Backup and restore

What to keep:

| Path | Contains |
|---|---|
| `data/` | `dgxmm.sqlite3`: users, deployments and their configs, events, calibration, benchmarks, 30-day metrics, audit log, runtime settings (including the decision model's on/off switch, deployment, per-feature switches and thresholds: `app_settings` keys `decider.*`), the decision log (`decisions`, with feedback), failure diagnoses, model-kind cache and `schema_version`. Also cached vLLM architecture lists. |
| `data/run/admin.sock` | The admin CLI's unix socket while DGXMM runs. Not data: `tar` skips it ("socket ignored") and DGXMM recreates it at start. |
| `data/secrets/decider_api_key` | The decision model's API key (mode 600; part of `data/`). If it is lost, a new one is generated and the decider has to be restarted (disable, then enable) to use it. |
| `data/litellm-db/` | Bundled LiteLLM only: its Postgres database (virtual keys, budgets, spend, registered models) |
| `secrets/` | HF token, LiteLLM keys (incl. the bundled LiteLLM's master/salt key and DB password), first admin password |
| `config.yaml`, `.env` | your configuration |
| `docker-compose.override.yaml` | only if you created one |

Model files (HF cache, GGUF directories) are not part of a DGXMM backup. Back them up separately or
download them again. For the decision model: `docker exec dgxmm python -m dgxmm.admin decider download`
(or `./install.sh` with `DGXMM_WITH_DECIDER=1`) fetches and verifies its snapshot again. The decider
image is rebuilt with `decider build`.

**Backup (brief stop):**

```bash
docker stop dgxmm
tar czf dgxmm-backup-$(date +%F).tgz data secrets config.yaml .env
docker start dgxmm
```

With the bundled LiteLLM (`DGXMM_WITH_LITELLM=1`), `data/` also holds a live Postgres
(`data/litellm-db`). Copying a running database's files gives a corrupt backup, so stop the bundled
containers too, and start the database first when you are done:

```bash
docker compose --profile litellm stop dgxmm dgxmm-litellm dgxmm-litellm-db
tar czf dgxmm-backup-$(date +%F).tgz data secrets config.yaml .env
docker compose --profile litellm up -d
```

Or keep LiteLLM running and take a logical dump of its database. Back it up together with `secrets/`:
`litellm_salt_key` must match, or the stored credentials cannot be decrypted.

```bash
docker compose --profile litellm exec -T dgxmm-litellm-db pg_dump -U litellm -d litellm -Fc > litellm-$(date +%F).dump
```

Stopping `dgxmm` does not stop the engines. To back up the database without stopping DGXMM, use
SQLite's online backup:

```bash
docker exec dgxmm python -c "import sqlite3; s = sqlite3.connect('/app/data/dgxmm.sqlite3'); d = sqlite3.connect('/app/data/dgxmm-backup.sqlite3'); s.backup(d); d.close()"
# then copy data/dgxmm-backup.sqlite3 somewhere safe, together with secrets/, config.yaml and .env
```

**Restore** (same or new host):

```bash
git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager
tar xzf /path/to/dgxmm-backup-YYYY-MM-DD.tgz        # data/, secrets/, config.yaml, .env
./install.sh
```

With the bundled LiteLLM, the tarball's `data/litellm-db` and `secrets/` come back as they were and
`./install.sh` starts the database before LiteLLM. To restore a `pg_dump` instead: run `./install.sh`
first (this creates an empty database), then stop LiteLLM, load the dump and start it again:

```bash
docker compose --profile litellm stop dgxmm-litellm
docker compose --profile litellm exec -T dgxmm-litellm-db pg_restore -U litellm -d litellm --clean --if-exists < litellm-YYYY-MM-DD.dump
docker compose --profile litellm start dgxmm-litellm
```

Keep the original `secrets/litellm_salt_key` (and `litellm_bundled_master_key`) when you restore a dump.

On a different host, check `.env` first: `DGXMM_UID`/`DGXMM_GID`, `DOCKER_GID` (`stat -c %g
/var/run/docker.sock`) and the paths must fit the new machine. The installer warns when the uid or
Docker group do not match. If you restored from the online backup, rename `dgxmm-backup.sqlite3` to
`dgxmm.sqlite3` and make sure no stale `dgxmm.sqlite3-wal` file sits next to it. Deployments whose
containers do not exist on the new host are marked `failed` at startup. Start them from the UI.

The decision model's settings and decision log come back with `dgxmm.sqlite3`. If the decider was
enabled, DGXMM restarts its deployment like any other (on a new host: download its snapshot and build
its image first, see above, then **Enable** on the Decision model page or `decider enable --wait`). This
procedure (online backup, restore into a new data directory, start) was checked: thresholds, feature
switches, the decision log and the decider key came back unchanged.

## Uninstall

```bash
./install.sh uninstall           # remove the dgxmm container and dgxmm:latest image
./install.sh uninstall --purge   # also remove state and managed engines
```

| | `uninstall` | `uninstall --purge` |
|---|---|---|
| `dgxmm` container and image | removed | removed |
| Decision model container (this install's, label `dgxmm.install`) | stopped and removed; the database still has it enabled, so a reinstall brings it back | removed |
| Decision model image (labels `dgxmm.built=true` and this install's `dgxmm.install`) | kept | removed (other images, e.g. from before install labels, are reported and kept) |
| Decision model snapshots (`<HF cache>/hub/models--vllm-sr--d3-mini`, `models--vllm-sr--d3-nano`, and the shared-store blobs only they use) | kept | kept; removed with `--purge-decider-model` |
| Engine containers (`dgxmm.managed=true`) | left running (a reinstall picks them up again) | removed if labelled for this install (`dgxmm.install`); others (from before install labels, or another install) are reported and kept |
| Docker network | kept | removed if the installer created it |
| Decision model network (`decider_network`, default `dgxmm-decider`) | kept | removed if this install created it (labels `dgxmm.purpose=decider` and this install's `dgxmm.install`); others are reported and kept |
| Bundled LiteLLM containers (`dgxmm-litellm`, `dgxmm-litellm-db`) | removed | removed |
| `data/` (incl. the LiteLLM database `data/litellm-db`), `secrets/`, `config.yaml`, `.env` | kept | deleted |
| LiteLLM / `postgres:16` images | kept | kept (`docker image rm` them by hand) |
| HF cache, models directory | kept | **kept** |

Uninstall refuses to touch a `dgxmm` container that belongs to another checkout. Add `--dry-run` to
see what would happen.

## Logs

```bash
./install.sh logs        # last 200 lines of the dgxmm container (DGXMM_LOG_LINES=1000 for more)
./install.sh logs -f     # follow
docker logs dgxmm-<deployment-name>     # an engine container
docker logs dgxmm-decider               # the decision model (while its container exists)
```

In the UI, a deployment's **Logs** button shows its engine log (admins). The **Events** tab keeps the
last log lines of every failure, even after the container has been removed. The DGXMM log covers
startup reconciliation, memory watchdog changes, LiteLLM sync failures, calibration measurements, the
decision model's supervisor (probes, restarts, "the running decider predates this DGXMM build"),
decider retention passes and errors. The log is managed by Docker, and DGXMM writes no log files.
Decision-model questions and answers are not logged there: they are in the decision log on the Decision
model page (`GET /api/decider/decisions`).

## Health

```bash
./install.sh status
```

This shows the container state and health, the checkout it belongs to, whether `.env`, `config.yaml`
and each secret are present, the running engine containers and the URLs. It exits 0 only when DGXMM is
running and healthy, so you can use it in monitoring scripts.

The container health check (every 10 s) requests `http://127.0.0.1:8090/dgxmm/` inside the
container. From the host:

```bash
docker inspect --format '{{.State.Health.Status}}' dgxmm
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/dgxmm/     # 200
```

The container restarts automatically (`restart: unless-stopped`), also after a reboot.

Delete helpers only exist while a deletion runs. This should normally list nothing:

```bash
docker ps -a --filter label=dgxmm.helper=true
```

## Freeing disk space

Delete models you no longer need from **Models → On disk** (**Delete…**), and remove interrupted
downloads with **Clean up** or **Clean up incomplete** (see [usage](usage.md#delete-weights)). Both are
admin-only and audited.

- **Models in use** cannot be deleted: stop the deployments the dialog names first. A model whose
  download is running cannot be deleted either: cancel it on the Downloads page.
- **Stopped or failed deployments** that use a deleted model stay in the list but fail to start with
  "model not found" until you download the model again, reconfigure them to another model, or remove
  them.
- **Root-owned files.** vLLM containers run as root and can leave files in the HF cache that your user
  cannot delete. DGXMM removes them through a short-lived helper container that runs as root with
  only the model's parent directory mounted and no network (see
  [architecture](architecture.md#deleting-weights)). The helper uses DGXMM's own image and is never
  pulled. If it cannot run (for example `helper_image` names a missing image), DGXMM deletes what it can
  itself and fails with `delete_failed` and a hint such as:

  ```bash
  sudo chown -R <uid>:<gid> /home/<user>/.cache/huggingface/hub/models--org--name
  ```

  The uid and gid are `DGXMM_UID` and `DGXMM_GID` from `.env`. Run it, then delete again.
- **Shared files.** When two revisions of a repo share a file, deleting one GGUF only removes its link
  and frees nothing; the dialog says so. Delete the whole repo to free the space.
- **Clean up** skips models with a partial file written in the last 10 minutes, so a download that is
  still being written by another tool is not removed under it.

Deleting files outside DGXMM (for example with `rm` or `huggingface-cli delete-cache`) is fine too.
The next rescan (or **Rescan**) updates the list.

## Admin command line

Every admin command runs inside the `dgxmm` container, as `docker exec dgxmm python -m dgxmm.admin
<command>` (`--help` on any command lists its options):

| Command | What it does |
|---|---|
| `reset-password USER` | Sets a new random password (printed once), signs the user out everywhere and requires a change at next sign-in. See [forgot the admin password](#forgot-the-admin-password). |
| `list-users` | Lists user names and roles. |
| `reset-calibration REPO [--engine vllm\|llamacpp]` | Forgets the learned memory calibration of a model, so estimates start from the formula again. |
| `decider status [--json] [--check]` | The decision model's state; `--check` exits 1 unless it is enabled and healthy. |
| `decider build [--force]` | Builds the decider image (skipped when it exists). |
| `decider download [--model vllm-sr/d3-nano] [--verify-only]` | Downloads the pinned model snapshot into the HF cache and verifies every file; `--verify-only` only checks it. |
| `decider enable [--wait] [--timeout SECONDS]` | Enables the decider through the running app (normal memory check); `--wait` waits until it is healthy (default 900 s). |
| `decider disable [--remove]` | Disables it and stops its deployment; `--remove` also deletes the deployment. |
| `decider admin-disabled` | Exits 0 when an admin explicitly disabled the decider (used by the installer). |
| `rotate-decider-key [--wait SECONDS]` | New bearer key for the decider; a running decider is restarted with it. Exit 0 rotated, 1 failed, 2 not picked up by DGXMM, 3 still in progress (run it again to keep waiting). Default wait 90 s. |

`decider status|enable|disable` act as the oldest admin (`decider --user NAME …` picks another one) over
the admin unix socket; see [decision model](#decision-model). Every command that changes something is
written to the audit log.

## Troubleshooting

### `permission denied` on the Docker socket

Your user is not in the `docker` group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker            # or log out and back in
./install.sh
```

Do not run the installer with `sudo`. DGXMM's files must belong to your user.

### No `nvidia` runtime

The installer stops with "Docker has no nvidia runtime registered". Install the NVIDIA Container
Toolkit (package `nvidia-container-toolkit`), then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}'    # must list "nvidia"
```

If the installer says `nvidia-smi` cannot see a GPU, fix the NVIDIA driver first.

### Port already in use

The installer stops if another process uses the UI port. On a first install, pick another port:

```bash
./install.sh --port 8091
```

If `.env` already exists, set `DGXMM_PORT=8091` in `.env` and run `./install.sh`. Engine containers use
host ports 8100–8199 (`port_range` in `config.yaml`) and skip ports other containers have published.

### A model does not fit

The memory check shows the shortfall and ways to make it fit (see
[usage](usage.md#deploy-a-model)). Check also:

- **External processes** (Dashboard): notebooks, other containers or desktop apps use memory that DGXMM
  will not hand out. Stop what you do not need.
- **Sleeping deployments** still hold their weights. Stop them to free everything.
- **Estimates** that look too high: run the model once. After it becomes ready, DGXMM measures it and
  later estimates for that configuration are based on the measurement.
- On a box dedicated to DGXMM you can lower `os_reserve_gib` (see
  [tuning](configuration.md#tuning-the-memory-budget)).

### Engine startup timeout

A deployment fails with "startup timed out after 900s". Open its **Events** tab for the log tail:

- **First start of a vLLM model**: kernel compilation and CUDA graph capture can take a long time. The
  results are cached in `~/.cache/vllm`, `~/.cache/flashinfer` and `~/.triton`, so the next start is
  faster. For very large models raise `startup_timeout_s` in `config.yaml`.
- **Image pull**: the first deployment with an engine may have to pull its image. Pre-pull with
  `DGXMM_PULL_ENGINES=1 ./install.sh` or `docker pull <image>`.
- "aborted: available memory fell below the safety margin during startup": the model needed more than
  estimated, or something else grew. Try a smaller context or fp8 KV cache.
- **Out of memory** ("out of memory: Docker reports the container was OOM-killed", or "…the kernel OOM
  killer killed a process"): the kernel killed the engine. DGXMM adds 10 % to the next estimate for that
  configuration automatically, and the diagnosis is "Out of memory".
- **Exit code 137 without that** ("killed by SIGKILL"): something sent SIGKILL, such as `docker kill`, or
  a `docker stop` that timed out. It is not treated as out of memory and does not change estimates.

### A deployment stays `queued`

Only `max_concurrent_startups` (default 1) engines start at a time, and the deployment page says which
deployment it is waiting for. That start-up ends within `startup_timeout_s` (900 s) by becoming ready or
failing; then the next queued deployment starts. A queued deployment fails by itself after
`startup_queue_timeout_s` (30 minutes), or when it no longer fits at its turn. To give up earlier, press
**Stop** on it (that only leaves the queue). A wake-up that waits (*Waiting to wake up*) gives up after
`wake_queue_timeout_s` (300 s). After a DGXMM restart, queued deployments re-enter the queue.

### Gated model: 401/403 from Hugging Face

The catalog or a download reports a gated repo ("accept the licence on huggingface.co and add an HF
token"):

1. Sign in on huggingface.co and accept the model's licence.
2. Create a read token and give it to DGXMM:

   ```bash
   HF_TOKEN=hf_xxx ./install.sh     # if secrets/hf_token is empty
   # or replace an existing token:
   umask 077; printf '%s' 'hf_xxx' > secrets/hf_token; docker restart dgxmm
   ```

### LiteLLM status stays `pending`

`pending` means the last sync failed. DGXMM retries, backing off up to 10 minutes. The reason is on
the deployment's LiteLLM tab and in the Deployments table tooltip; the log has "LiteLLM sync failed"
(`./install.sh logs`). Start with **Test connection** on the LiteLLM page. Common causes:

- **LiteLLM unreachable** ("could not reach LiteLLM at …", "timed out", or "is not a LiteLLM
  endpoint"): the internal URL (default `http://litellm-proxy:4000`) must resolve from the DGXMM
  network. Fix it with `docker network connect <DGXMM_NETWORK> litellm-proxy`, or set an internal URL
  DGXMM can reach. "Not a LiteLLM endpoint" means something else answered on that URL.
- **Master key rejected** ("LiteLLM rejected the master key"): enter LiteLLM's master key on the
  LiteLLM page, or put it in `secrets/litellm_master_key`.
- **LiteLLM cannot store models**: set `store_model_in_db: true` in LiteLLM's `general_settings` (it
  needs its database).
- **LiteLLM cannot reach the engine** (registration works but requests fail): LiteLLM must be on the
  same Docker network, because engines are registered as `http://dgxmm-<name>:<port>/v1`.

### LiteLLM status is `disabled`, or the page says the master key needs re-entering

`disabled` means the integration is turned off on the LiteLLM page, or no master key applies. A master
key only applies to the URL it belongs to: a key entered on the page to the internal URL in effect
when it was saved, the `secrets/litellm_master_key` file to the default `litellm_url`. After you change
the internal URL, the page shows "the master key was entered for a different LiteLLM URL" (or "the
master key from the secret file is only used with the default LiteLLM URL"). Enter the master key for
the new URL, or set the URL back. This is deliberate: a stored key is never sent to a server it was not
entered for.

### Virtual keys: "LiteLLM is running without a database"

LiteLLM keeps virtual keys in its database. Configure `DATABASE_URL` (and `store_model_in_db: true`) for
your LiteLLM, or use the [bundled LiteLLM](configuration.md#bundled-litellm), which has one. The
keys sections stay unavailable until LiteLLM reports a connected database.

### "The model name … is already used by deployment …" (`model_name_taken`)

Two deployments cannot share one LiteLLM model name, because LiteLLM would then load-balance between
two different models. Deploy, start, wake and rename are refused with the name of the deployment that
holds it. Give one of them a different **Model name (LiteLLM)**: the error offers a suggestion
(`<served name>-<deployment name>`), and on a deployment's page **Set a model name** opens its LiteLLM
tab. Stopped deployments without an alias do not hold their served name.

If a deployment shows the LiteLLM status `error`, two ready deployments ended up with the same name
anyway. The oldest keeps the registration; rename the other on its LiteLLM tab.

### Clients cannot call a served name any more after an upgrade

Older versions registered an aliased deployment under both its served name and its alias. Now each
deployment is registered once, under its model name, and the first sync after the upgrade removes the
extra served-name entries. Point those clients at the model name shown on the deployment's API access
card.

### Processes: masked or numeric values

- **`***` in commands** is intentional: passwords, tokens and keys are masked, and so are values DGXMM
  cannot parse reliably. Masking is best-effort: it cannot recognise every way a program may take a
  secret on its command line, so treat the Processes page as visible to every DGXMM user and do not pass
  secrets as command-line arguments. Use files or environment variables, which DGXMM never reads.
- **Numeric uids** for host processes mean the host's `/etc/passwd` is not mounted: the mount is in
  `docker-compose.yaml`, so run `./install.sh` to recreate the container. Processes in containers always
  show their in-container uid (`uid 1000 (in container)`), because container uids are not host
  accounts.
- **Containers shown by id only** appear for a few seconds after DGXMM starts, until the container list
  is first read.

### Download fails with "owned by root"

An engine container created a folder in the HF cache as root. Run the `sudo chown -R <uid>:<gid> …`
command shown in the error (the uid:gid are `DGXMM_UID:DGXMM_GID` from `.env`), then retry.

### Catalog says "llmfit unavailable"

llmfit only adds scores. Search, fit badges and downloads keep working. Check it with
`docker exec dgxmm llmfit --version`.

### Decision model not healthy

Start with the chip in the top bar or `docker exec dgxmm python -m dgxmm.admin decider status`. While
it is not healthy every feature uses its rules, so nothing else is affected.

- **Enable answers `image_missing` or `image_outdated`**: the image `dgxmm-decider:3` is not built, or
  was built before the selected model was supported. Build it on the Decision model page or with
  `decider build` (`--force` to rebuild).
- **"Make room for the decision model"** (`insufficient_memory`): it needs 13 GiB (d3-mini) or 9 GiB
  (d3-nano). Pick one of the plans, stop something yourself, or switch to d3-nano.
- **Starting for several minutes**: the first start compiles GPU kernels (2–4 minutes for d3-mini). It
  becomes healthy only after its warm-up questions.
- **Failed with exit code 2** ("snapshot missing"): the model files are not in the HF cache. Run
  `docker exec dgxmm python -m dgxmm.admin decider download` (add `--model vllm-sr/d3-nano` for nano)
  or press **Download** on the Model card, then **Start**.
- **Failed with exit code 3** (diagnosis *Decision model files fail the integrity check*): a file of the
  pinned snapshot no longer matches its hash, for example because something wrote into the HF cache.
  The decider refuses to run it. `decider download --verify-only` lists the files that differ;
  `decider download` fetches them again and verifies them; then **Start**.
- **Failed with exit code 4** (*Decision model key not delivered*): the decider got no key file, or
  found a key in its environment (a `DECISION_API_KEY` override, which it refuses). Remove any such
  override, check the DGXMM log for why the key could not be written, then **Disable** and **Enable** so
  DGXMM starts it again with a fresh key file. **Rotate key** writes a new key if
  `data/secrets/decider_api_key` is unreadable.
- **Paused (rules only)**: the circuit breaker opened after three failed questions in a row; it retries
  after 60 seconds or as soon as a health probe succeeds.
- **Auto-restart "Gave up — …"**: 5 restart attempts in an hour failed or crashed again (the reason
  follows the dash, e.g. a missing image or a docker error). Read its Events tab and
  `docker logs dgxmm-decider`, fix the cause, then press **Retry** (or **Disable** and **Enable**).
- **Stopped by the watchdog** (Events: "watchdog: …"): memory ran low, or the decider used more than
  its reservation. The supervisor starts it again after its backoff once memory has recovered.

### `429 rate_limited` from the API

Some routes have per-user limits (see [API limits](configuration.md#api-limits)), for example 10 changes a
minute on the Decision model page or 30 parser suggestions a minute. The response has a `Retry-After`
header; wait that many seconds. Plan previews are never refused for this: over the limit they come back
without a model recommendation. Limits are per DGXMM process and reset when it restarts.

### Forgot the admin password

Reset it from the host. This works while DGXMM is running:

```bash
docker exec dgxmm python -m dgxmm.admin reset-password admin
```

It prints a new random password once, signs that user out everywhere, and requires a password change
at the next sign-in. The reset is written to the audit log. Use any username in place of `admin`. To see
which users exist:

```bash
docker exec dgxmm python -m dgxmm.admin list-users
```

`secrets/admin_password` is not updated, and only ever holds the first-run password. Another admin can
also reset a password from **Users → Reset password** in the UI.

### Two DGXMM instances

The installer refuses to start a second instance from another checkout, because both would adopt the
same engine containers. Manage the existing one from its own directory (the error names it), or
uninstall it first.

## Known limitations

- **Command-line masking is best-effort.** The Processes page and the dashboard's external memory card
  mask passwords, tokens and keys, and mask a whole value when it cannot be parsed reliably (fail
  closed). Some rare wrapper or quoting forms may still not be recognised, so a secret passed as a
  command-line argument can appear unmasked. Do not pass secrets as arguments; use files or
  environment variables, which DGXMM never reads.
- **Deletion only knows about DGXMM's deployments.** A model is protected from deletion when a DGXMM
  deployment uses it. Containers or processes that DGXMM does not manage (a vLLM started by hand, a
  notebook) and use the same HF cache or GGUF directory are not detected. Check the
  [Processes](usage.md#processes) page, or `docker ps`, before you delete a model.
- **Clearing the master key does not deregister models.** Only turning the integration off or changing
  the internal URL removes DGXMM's models from LiteLLM. Clearing a master key stored on the LiteLLM
  page (or removing the secret file) leaves the existing registrations in LiteLLM; turn the
  integration off first if you want them removed.
- **Virtual-key limits cannot be cleared.** LiteLLM's key update cannot remove a budget or rate limit once it is
  set, so **Edit** can only change it. Set a higher value, or create a new key
  and revoke the old one.
- **Per-process GPU memory on GB10.** Per-process GPU figures come from NVML. When NVML reports no value
  for a process, DGXMM shows `0`, so the GPU column on the Processes page may read 0 for a process that
  does use the GPU. On the GB10's unified memory, the RAM column and the deployment's measured memory
  are the more reliable figures.

## Security hardening

See also the threat model in [security.md](security.md).

- **Docker socket = root.** Anyone with the DGXMM admin role can start containers on the host. Only
  make trusted people admins, and give everyone else the `viewer` role.
- **Do not expose DGXMM directly to the internet.** For access from outside your LAN, use a VPN, or
  install with `DGXMM_BIND=127.0.0.1` behind an HTTPS reverse proxy (see
  [configuration](configuration.md#reverse-proxy)) and set `secure_cookies: always`.
- **Firewalls:** ports published by Docker bypass host firewall rules such as `ufw`. Restrict access
  with `DGXMM_BIND`, not only with the firewall.
- **Client IPs behind a proxy:** DGXMM only trusts `X-Forwarded-For` from IPs/subnets configured in
  `trusted_proxies` (default: `127.0.0.1,::1`, loopback-only). Behind a reverse proxy, set this to the
  proxy's IP or Docker network subnet (for example `DGXMM_TRUSTED_PROXIES=172.23.0.0/16` in `.env`
  for a proxy container on the `gateway` network) so that client IPs in logs and rate limiting are
  correct. `*` trusts every client and is not recommended. Have
  the proxy overwrite the header (`proxy_set_header X-Forwarded-For $remote_addr;` in nginx) rather than
  append to it. The per-username login limit applies either way.
- **Engine endpoints have no authentication.** Keep `publish_bind: 127.0.0.1` and give other machines
  access through LiteLLM with virtual keys (one per app or person, limited to the models they need).
- **Process command lines are visible to every user**, masked. Do not put secrets on command lines of
  processes on the box (see [Processes](#processes-masked-or-numeric-values)).
- **The LiteLLM master key** entered on the LiteLLM page is stored in `data/dgxmm.sqlite3`. Protect
  `data/` like `secrets/`.
- **Delete `secrets/admin_password`** after you have changed the first password. It is not needed
  afterwards.
- **Protect backups**: they contain password hashes, session data and your HF and LiteLLM keys.
- **Pin images**: the vLLM image is pinned by digest. Pin the llama.cpp image too if you want
  reproducible deployments (see [engine images](configuration.md#engine-images)).
