# Operations

How to install, upgrade, back up, monitor and troubleshoot DGXMM. All commands run in the DGXMM
checkout on the host, as the user that installed it.

- [What the installer does](#what-the-installer-does)
- [Upgrade](#upgrade)
- [Changing settings](#changing-settings)
- [Backup and restore](#backup-and-restore)
- [Uninstall](#uninstall)
- [Logs](#logs)
- [Health](#health)
- [Troubleshooting](#troubleshooting)
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
   - warns when less than 50 GiB is free on the HF cache disk.
2. **Settings.** Works out the port, bind address, network, model paths, your uid/gid and the Docker
   socket's group id. Order of precedence: existing `.env`, then flags, then environment, then defaults.
   Paths and values are validated.
3. **Configuration.** Writes `.env` if missing (or appends keys an older `.env` lacks). Renders
   `config.yaml` from `config.example.yaml` if missing. Creates `data/` and `secrets/` (mode 700). Writes
   `secrets/hf_token` and `secrets/litellm_master_key` (mode 600; empty unless `HF_TOKEN` or
   `LITELLM_MASTER_KEY` is set, or the LiteLLM key could be read from a running `litellm-proxy`
   container). Warns if `litellm-proxy` is not on the DGXMM network. Existing files are always kept.
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
   password once and saves it to `secrets/admin_password`.

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

## Changing settings

- **`.env` values** (port, bind address, network, model paths): edit `.env`, then `./install.sh`. If you
  change paths or the network, update `config.yaml` to match (see
  [configuration](configuration.md#env-and-installer-variables)).
- **`config.yaml`**: edit it, then `./install.sh` (or `docker restart dgxmm`, because the file is
  mounted).
- **Secrets**: edit the file in `secrets/`, then `docker restart dgxmm`.

## Backup and restore

What to keep:

| Path | Contains |
|---|---|
| `data/` | `dgxmm.sqlite3`: users, deployments and their configs, events, calibration, benchmarks, 30-day metrics, audit log. Also cached vLLM architecture lists. |
| `secrets/` | HF token, LiteLLM key, first admin password |
| `config.yaml`, `.env` | your configuration |
| `docker-compose.override.yaml` | only if you created one |

Model files (HF cache, GGUF directories) are not part of a DGXMM backup. Back them up separately or
download them again.

**Backup (brief stop):**

```bash
docker stop dgxmm
tar czf dgxmm-backup-$(date +%F).tgz data secrets config.yaml .env
docker start dgxmm
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

On a different host, check `.env` first: `DGXMM_UID`/`DGXMM_GID`, `DOCKER_GID` (`stat -c %g
/var/run/docker.sock`) and the paths must fit the new machine. The installer warns when the uid or
Docker group do not match. If you restored from the online backup, rename `dgxmm-backup.sqlite3` to
`dgxmm.sqlite3` and make sure no stale `dgxmm.sqlite3-wal` file sits next to it. Deployments whose
containers do not exist on the new host are marked `failed` at startup. Start them from the UI.

## Uninstall

```bash
./install.sh uninstall           # remove the dgxmm container and dgxmm:latest image
./install.sh uninstall --purge   # also remove state and managed engines
```

| | `uninstall` | `uninstall --purge` |
|---|---|---|
| `dgxmm` container and image | removed | removed |
| Engine containers (`dgxmm.managed=true`) | left running (a reinstall picks them up again) | removed |
| Docker network | kept | removed if the installer created it |
| `data/`, `secrets/`, `config.yaml`, `.env` | kept | deleted |
| HF cache, models directory | kept | **kept** |

Uninstall refuses to touch a `dgxmm` container that belongs to another checkout. Add `--dry-run` to
see what would happen.

## Logs

```bash
./install.sh logs        # last 200 lines of the dgxmm container (DGXMM_LOG_LINES=1000 for more)
./install.sh logs -f     # follow
docker logs dgxmm-<deployment-name>     # an engine container
```

In the UI, a deployment's **Logs** button shows its engine log (admins). The **Events** tab keeps the
last log lines of every failure, even after the container has been removed. The DGXMM log covers
startup reconciliation, memory watchdog changes, LiteLLM sync failures, calibration measurements and
errors. The log is managed by Docker, and DGXMM writes no log files.

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
- **Exit code 137** ("likely out of memory"): the kernel killed the engine. DGXMM adds 10 % to the next
  estimate for that configuration automatically.

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

`pending` means the last sync failed. DGXMM retries, backing off up to 10 minutes. Look for
"LiteLLM sync failed" in `./install.sh logs`. Common causes:

- **Wrong or missing key**: `secrets/litellm_master_key` must hold LiteLLM's master key. Restart with
  `docker restart dgxmm` after changing it. An empty file shows `disabled`.
- **LiteLLM cannot store models**: set `store_model_in_db: true` in LiteLLM's `general_settings` (it
  needs its database).
- **DGXMM cannot reach LiteLLM**: `litellm_url` (default `http://litellm-proxy:4000`) must resolve from
  the DGXMM network. Fix it with `docker network connect <DGXMM_NETWORK> litellm-proxy`, or point
  `litellm_url` at an address DGXMM can reach.
- **LiteLLM cannot reach the engine** (registration works but requests fail): LiteLLM must be on the
  same Docker network, because engines are registered as `http://dgxmm-<name>:<port>/v1`.

### Download fails with "owned by root"

An engine container created a folder in the HF cache as root. Run the `sudo chown -R <uid>:<gid> …`
command shown in the error (the uid:gid are `DGXMM_UID:DGXMM_GID` from `.env`), then retry.

### Catalog says "llmfit unavailable"

llmfit only adds scores. Search, fit badges and downloads keep working. Check it with
`docker exec dgxmm llmfit --version`.

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

## Security hardening

- **Docker socket = root.** Anyone with the DGXMM admin role can start containers on the host. Only
  make trusted people admins, and give everyone else the `viewer` role.
- **Do not expose DGXMM directly to the internet.** For access from outside your LAN, use a VPN, or
  install with `DGXMM_BIND=127.0.0.1` behind an HTTPS reverse proxy (see
  [configuration](configuration.md#reverse-proxy)) and set `secure_cookies: always`.
- **Firewalls:** ports published by Docker bypass host firewall rules such as `ufw`. Restrict access
  with `DGXMM_BIND`, not only with the firewall.
- **Client IPs behind a proxy:** DGXMM only trusts `X-Forwarded-For` from IPs/subnets configured in
  `trusted_proxies` (default: `127.0.0.1,::1`, loopback-only). Behind a reverse proxy, set this to the
  proxy's IP or docker network subnet (e.g., `DGXMM_TRUSTED_PROXIES=172.23.0.0/16` in `.env` for a
  dockerized proxy on the `gateway` network) to ensure correct client IPs in logs and rate limiting. Have
  the proxy overwrite the header (`proxy_set_header X-Forwarded-For $remote_addr;` in nginx) rather than
  append to it. The per-username login limit applies either way.
- **Engine endpoints have no authentication.** Keep `publish_bind: 127.0.0.1` and give other machines
  access through LiteLLM with its own API keys.
- **Delete `secrets/admin_password`** after you have changed the first password. It is not needed
  afterwards.
- **Protect backups**: they contain password hashes, session data and your HF and LiteLLM keys.
- **Pin images**: the vLLM image is pinned by digest. Pin the llama.cpp image too if you want
  reproducible deployments (see [engine images](configuration.md#engine-images)).
