# Configuration

DGXMM reads its settings from three places:

| File | Written by | Used by | Purpose |
|---|---|---|---|
| `.env` | `install.sh` (once) | Docker Compose | Host-specific values: user and group ids, port, bind address, model paths, Docker network |
| `config.yaml` | `install.sh` (once, from `config.example.yaml`) | the DGXMM app | Everything DGXMM itself does: memory gate, engines, paths, LiteLLM, polling |
| `secrets/*` | `install.sh` | Compose secrets → `/run/secrets/` | Hugging Face token, LiteLLM master key, first admin password |

The installer never overwrites `.env`, `config.yaml` or a non-empty secret file. After you edit one,
apply the change with `./install.sh`. Running it again is safe. It rebuilds and recreates the `dgxmm`
container. Engine containers keep running.

- [config.yaml reference](#configyaml-reference)
- [Environment variable overrides](#environment-variable-overrides)
- [.env and installer variables](#env-and-installer-variables)
- [Secrets](#secrets)
- [Model directories](#model-directories)
- [Engine images](#engine-images)
- [Networking](#networking)
- [Reverse proxy](#reverse-proxy)
- [LiteLLM integration](#litellm-integration)
- [Hugging Face token](#hugging-face-token)
- [Tuning the memory budget](#tuning-the-memory-budget)

## config.yaml reference

Every key is optional. Anything left out takes the default below (from `dgxmm/config.py`).
`config.example.yaml` lists the same keys with comments. In host paths, a leading `~` means the home
directory of the host user DGXMM runs as (`DGXMM_HOST_HOME` in `.env`), not the home directory inside
the container.

### Memory gate

| Key | Type | Default | Meaning |
|---|---|---|---|
| `safety_margin_gib` | float ≥ 0 | `8` | Memory that is never given to models. It is also the watchdog's critical threshold and the startup-abort threshold. |
| `os_reserve_gib` | float ≥ 0 | `4` | Kept back for the OS, desktop and other processes. |
| `auto_protect` | bool | `false` | `true`: when memory turns critical, the watchdog stops the lowest-priority deployment by itself. `false`: it only warns. |
| `vllm_runtime_fixed_gib` | float | `3.0` | Fixed vLLM overhead added to every vLLM estimate. |
| `vllm_cuda_graph_gib` | float | `1.5` | CUDA graph memory added to every vLLM estimate. |
| `llamacpp_runtime_fixed_gib` | float | `0.5` | Fixed llama.cpp overhead added to every llama.cpp estimate. |

### Engines

| Key | Type | Default | Meaning |
|---|---|---|---|
| `images.vllm` | string | `vllm/vllm-openai@sha256:22d3b121…352c8` | vLLM image (public multi-arch image, pinned by digest). |
| `images.llamacpp` | string | `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama.cpp server image. |
| `docker_network` | string | `dgxmm` | Docker network that engine containers join. **Must equal `DGXMM_NETWORK` in `.env`.** |
| `port_range` | `[low, high]` | `[8100, 8199]` | Host ports handed to engine containers (1024–65535, ascending). Ports used by other containers are skipped. |
| `publish_bind` | string or null | `127.0.0.1` | Host address that engine ports are published on. Keep it on loopback: clients should go through LiteLLM or the Docker network. |
| `engine_base_host` | string or null | `null` | `null`: DGXMM (and LiteLLM) reach engines as `dgxmm-<name>:<port>` on `docker_network`. Set a hostname or IP to use the published ports on that host instead. |
| `startup_timeout_s` | int | `900` | Seconds an engine may take to become ready before DGXMM fails it. |
| `vllm_cache_mounts` | list of `{host, target, mode}` | `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.triton` → `/root/...` | Host directories mounted into vLLM containers so compiled kernels survive restarts. `mode` defaults to `rw`. |
| `llmfit_bin` | string | `llmfit` | llmfit executable used for catalog scores. |

### Models and storage

| Key | Type | Default | Meaning |
|---|---|---|---|
| `hf_cache.host` | path | `~/.cache/huggingface` | Hugging Face cache on the host (mounted into engines). |
| `hf_cache.local` | path | `/data/hf` | The same directory inside the `dgxmm` container (set by `docker-compose.yaml`). |
| `gguf_roots` | list of `{host, local}` | `[{host: ~/models, local: /data/gguf/0}]` | Directories scanned for `.gguf` files. See [Model directories](#model-directories). |
| `min_free_disk_gib` | float | `50` | Downloads are refused when they would leave less than this free on the HF cache disk. |

### LiteLLM and secrets

| Key | Type | Default | Meaning |
|---|---|---|---|
| `litellm_url` | URL | `http://litellm-proxy:4000` | LiteLLM proxy that DGXMM registers models with. |
| `litellm_key_file` | path | `/run/secrets/litellm_master_key` | LiteLLM master key file. Empty or missing turns registration off. |
| `hf_token_file` | path | `/run/secrets/hf_token` | Hugging Face token file. Empty or missing means anonymous Hub access. |

### Web UI and sessions

| Key | Type | Default | Meaning |
|---|---|---|---|
| `base_path` | string | `/dgxmm` | URL prefix of the UI and API. The frontend is built for `/dgxmm`, so leave this alone. |
| `session_ttl_s` | int | `43200` (12 h) | Login session lifetime. It is extended on every request. |
| `secure_cookies` | `auto` / `always` / `never` | `auto` | `auto`: the cookie is `Secure` when the request is HTTPS (directly or via `X-Forwarded-Proto`). Use `always` behind an HTTPS proxy. |
| `trusted_proxies` | comma-separated IPs/CIDRs | `127.0.0.1,::1` | IPs/subnets trusted to set `X-Forwarded-For` (for correct client IPs in logs and rate limiting). Default is loopback-only. Set to your proxy's IP or docker network subnet (e.g., `<proxy-ip>` or `<network-subnet>`). |

### Polling intervals

| Key | Type | Default | Meaning |
|---|---|---|---|
| `monitor_poll_s` | float | `3` | How often a starting deployment is checked. |
| `budget_interval_s` | float | `2` | Memory budget refresh. It also drives the live UI, the watchdog and the liveness checks. |
| `scrape_interval_s` | float | `5` | Engine Prometheus metrics scrape. |
| `inventory_interval_s` | float | `300` | Rescan of the HF cache and GGUF roots. The **Rescan** button on Models runs one immediately. |

### Paths inside the container

These are set by the image. You rarely need to change them.

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `/app/data` | SQLite database and caches (`./data` on the host). |
| `static_dir` | `/app/static` | Built frontend. |
| `host_proc` | `/host/proc` | Host `/proc` (read-only mount). |
| `host_cgroup` | `/host/sys/fs/cgroup` | Host cgroup tree (read-only mount). |

## Environment variable overrides

Any key can be set with `DGXMM_<KEY_IN_UPPER_CASE>`, which wins over `config.yaml`. The value is
parsed as YAML, so lists and objects work. An object replaces the whole key, and its missing fields
take their defaults:

```bash
DGXMM_AUTO_PROTECT=true
DGXMM_PORT_RANGE="[8200, 8299]"
DGXMM_IMAGES="{vllm: vllm/vllm-openai:latest}"     # llamacpp keeps its default
```

Two more variables are not config keys: `DGXMM_CONFIG` (path of the YAML file, default `config.yaml`;
`/app/config.yaml` in the container) and `DGXMM_HOST_HOME` (what `~` means in host paths).

The container only receives the variables listed in `docker-compose.yaml`. To pass an override, add a
`docker-compose.override.yaml` next to it, which Compose merges automatically:

```yaml
services:
  dgxmm:
    environment:
      DGXMM_AUTO_PROTECT: "true"
```

Then run `./install.sh`. Editing `config.yaml` is usually simpler.

## .env and installer variables

`./install.sh` writes `.env` on first install from flags, environment variables and detected values.
Once `.env` exists, its values win: a different flag or variable only produces a warning. Keys missing
from an older `.env` are appended.

| `.env` key | Installer flag / variable | Default | Meaning |
|---|---|---|---|
| `DGXMM_UID`, `DGXMM_GID` | (detected) | `id -u`, `id -g` | User the container runs as. It owns `data/` and files DGXMM writes to the model caches. |
| `DOCKER_GID` | (detected) | group of `/var/run/docker.sock` | Added to the container so it can use the Docker socket. |
| `DGXMM_PORT` | `--port` / `DGXMM_PORT` | `8090` | Host port of the UI. |
| `DGXMM_BIND` | `--bind` / `DGXMM_BIND` | `0.0.0.0` | Host IPv4 address to bind. Use `127.0.0.1` behind a reverse proxy. |
| `DGXMM_NETWORK` | `--network` / `DGXMM_NETWORK` | `dgxmm` | Docker network for DGXMM and engines. It is created (labelled `dgxmm.created=true`) if it does not exist. |
| `DGXMM_HF_CACHE` | `--hf-cache` / `DGXMM_HF_CACHE` | `$HOME/.cache/huggingface` | HF cache, mounted at `/data/hf`. |
| `DGXMM_MODELS_DIR` | `--models-dir` / `DGXMM_MODELS_DIR` | `$HOME/models` | GGUF directory, mounted at `/data/gguf/0`. |
| `DGXMM_HOST_HOME` | `DGXMM_HOST_HOME` | `$HOME` | Home directory that `~` in `config.yaml` expands to. |
| `DGXMM_WITH_LITELLM` | `--with-litellm` / `DGXMM_WITH_LITELLM=1` | `0` (key not written) | Run the [bundled LiteLLM](#bundled-litellm). Written only when enabled; set `0` to turn it off. |
| `DGXMM_LITELLM_PORT` | `--litellm-port` / `DGXMM_LITELLM_PORT` | `4000` | Host port of the bundled LiteLLM, bound to `DGXMM_BIND`. |
| `COMPOSE_PROFILES` | (written with `DGXMM_WITH_LITELLM=1`) | `litellm` | Makes a plain `docker compose up -d` start the bundled LiteLLM too. A stray shell value is never adopted. |

Paths must be absolute and must not contain whitespace, a backtick, or any of `:` `"` `'` `\` `$` `#`.
Binds must be IPv4.

Variables that only affect one installer run:

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Written to `secrets/hf_token` when that file is missing or empty. |
| `LITELLM_URL` | Written as `litellm_url` when `config.yaml` is generated. |
| `LITELLM_MASTER_KEY` | Written to `secrets/litellm_master_key` when that file is missing or empty. |
| `DGXMM_PULL_ENGINES=1` | Pull the engine images listed in `config.yaml` before starting. |
| `DGXMM_HEALTH_TIMEOUT` | Seconds to wait for the container to become healthy (default 300). |
| `DGXMM_LITELLM_HEALTH_TIMEOUT` | Same for the bundled LiteLLM (default 300). `DGXMM_LITELLM_MAX_RESTARTS` (default 3) fails sooner on a crash loop. |
| `DGXMM_LOG_LINES` | Lines shown by `./install.sh logs` (default 200). |

When the installer generates `config.yaml`, it puts your HF cache, models directory, home directory
and network into the copy of `config.example.yaml`. If you change `DGXMM_HF_CACHE`, `DGXMM_MODELS_DIR`
or `DGXMM_NETWORK` in `.env` later, update `hf_cache.host`, `gguf_roots[0].host` and `docker_network` in
`config.yaml` to match. Engine containers mount the *host* paths from `config.yaml`. The installer warns
if the networks differ.

## Secrets

`secrets/` has mode 700 and every file in it has mode 600. Compose mounts the first two files at
`/run/secrets/`.

| File | Meaning |
|---|---|
| `secrets/hf_token` | Hugging Face token (optional; needed for gated repos). |
| `secrets/litellm_master_key` | Master key of the LiteLLM DGXMM registers models with (optional). |
| `secrets/litellm_bundled_master_key` | Bundled LiteLLM only: its own master key. Mounted only into `dgxmm-litellm`. |
| `secrets/litellm_salt_key` | Bundled LiteLLM only: encrypts credentials in its database. Never change it once set. |
| `secrets/litellm_db_password` | Bundled LiteLLM only: Postgres password. |
| `secrets/admin_password` | The first-run `admin` password, saved by the installer. DGXMM never reads it, and it goes stale once you change the password. |

Secrets are read when DGXMM starts. After you change one, restart DGXMM with `docker restart dgxmm`.

## Model directories

- **Hugging Face models (vLLM)** live in the HF cache (`DGXMM_HF_CACHE`). DGXMM lists complete snapshots
  and downloads into that cache. Models you fetch with `huggingface-cli download` show up too, after the
  next rescan.
- **GGUF files (llama.cpp)** are found under every `gguf_roots` entry, and also inside the HF cache.

To add another GGUF directory, for example `/data/more-gguf`:

1. Add it to `config.yaml`. `local` must be a new path inside the container:

   ```yaml
   gguf_roots:
     - host: /home/me/models        # keep the existing entry
       local: /data/gguf/0
     - host: /data/more-gguf
       local: /data/gguf/1
   ```

2. Mount it into the `dgxmm` container with a `docker-compose.override.yaml` in the checkout:

   ```yaml
   services:
     dgxmm:
       volumes:
         - /data/more-gguf:/data/gguf/1
   ```

3. Run `./install.sh`. llama.cpp containers mount each root read-only at `/models/<index>`.

`docker-compose.override.yaml` is not tracked by git. To keep it out of `git status`, add it to
`.git/info/exclude`.

## Engine images

The default vLLM image is pinned by digest, so upgrades of DGXMM do not silently change it. The
llama.cpp image uses the moving `server-cuda` tag. To pin or change either, set `images` in
`config.yaml`:

```yaml
images:
  vllm: vllm/vllm-openai:<tag>             # any tag or digest published for your CPU architecture
  llamacpp: ghcr.io/ggml-org/llama.cpp@sha256:<digest>
```

Images are pulled on first use, and the first vLLM image is pulled in the background when DGXMM first
starts. Pre-pull with `DGXMM_PULL_ENGINES=1 ./install.sh` or `docker pull <image>`. DGXMM caches the
list of model architectures each vLLM image supports in `data/`, keyed by image reference, so a new
image is inspected once.

## Networking

- **UI port**: `DGXMM_BIND:DGXMM_PORT` on the host (default `0.0.0.0:8090`) maps to port 8090 in the
  container. Change it in `.env` and re-run `./install.sh`.
- **URL path**: the UI and API are served under `/dgxmm/` (`/` redirects there). The prefix is built
  into the frontend, so a reverse proxy must forward `/dgxmm/` unchanged. Do not strip the prefix.
  Serving DGXMM under another path is not supported.
- **Docker network**: DGXMM and every engine join `DGXMM_NETWORK` (`docker_network`). Containers on the
  same network (LiteLLM, your proxy, other apps) can reach an engine at `http://dgxmm-<name>:<port>/v1`.
- **Engine ports** are also published at `publish_bind:<port>` (default `127.0.0.1`, ports 8100–8199),
  so tools on the host can call `http://127.0.0.1:<port>/v1`.

## Reverse proxy

For HTTPS or internet-facing access, bind DGXMM to loopback and put a proxy in front:

```bash
DGXMM_BIND=127.0.0.1 ./install.sh
```

Set `secure_cookies: always` in `config.yaml` when the proxy terminates HTTPS. The proxy must disable
response buffering for the event stream, allow long-lived connections, and pass `X-Forwarded-Proto`.

**Important**: Configure `trusted_proxies` to match your proxy's IP or network:
- **Proxy container on the DGXMM network**: set to the proxy container's IP or the docker network subnet (e.g., `DGXMM_TRUSTED_PROXIES=<network-subnet>`)
- **Proxy on the host** (docker port publishing): set to the DGXMM network's gateway IP (find with `docker network inspect <network> -f '{{(index .IPAM.Config 0).Gateway}}'`).


**nginx in a Docker container on the DGXMM network**: the shipped `deploy/nginx-dgxmm.conf`. It
resolves `dgxmm` through Docker's DNS, so nginx still starts while DGXMM is down. Install with
`DGXMM_NETWORK=<your proxy's network>` (or `docker network connect <network> <nginx-container>`) and include
it in your `server { }` block:

```nginx
    # DGX Model Manager. Variable upstream + Docker DNS so nginx still starts when dgxmm is down.
    location /dgxmm/ {
        resolver 127.0.0.11 valid=10s ipv6=off;
        set $dgxmm_upstream http://dgxmm:8090;
        proxy_pass $dgxmm_upstream;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;          # SSE
        proxy_read_timeout 1h;
        client_max_body_size 1m;
    }
```

**nginx on the host**: use the same block with `proxy_pass http://127.0.0.1:8090;` and without the
`resolver` and `set` lines. Set `trusted_proxies` to the gateway IP.

**Caddy on the host** (it gets certificates automatically):

```caddy
dgx.example.com {
    redir / /dgxmm/
    handle /dgxmm/* {
        reverse_proxy 127.0.0.1:8090 {
            flush_interval -1
        }
    }
}
```

Use `handle`, not `handle_path`: the `/dgxmm` prefix must reach DGXMM.

## LiteLLM integration

DGXMM can register every `ready` deployment with a [LiteLLM](https://docs.litellm.ai/) proxy and remove
it again when the deployment stops. Each deployment is registered under its served model name and,
if set, under its alias, with `model: openai/<served name>` and
`api_base: http://dgxmm-<name>:<port>/v1`.

Requirements:

1. LiteLLM stores models in its database: `general_settings: { store_model_in_db: true }` with a
   configured database.
2. DGXMM has the master key in `secrets/litellm_master_key`. If you install with no key file and a
   container named `litellm-proxy` is running, the installer reads the key from that container.
   Otherwise pass `LITELLM_MASTER_KEY=…` or write the file yourself.
3. DGXMM can reach `litellm_url` (default `http://litellm-proxy:4000`, so a container named
   `litellm-proxy` on the DGXMM network).
4. LiteLLM can reach the engines: put it on the same Docker network
   (`docker network connect dgxmm litellm-proxy`), or install DGXMM with `DGXMM_NETWORK=<LiteLLM's
   network>`.

Example for a fresh install next to an existing LiteLLM on network `ai`:

```bash
DGXMM_NETWORK=ai LITELLM_URL=http://litellm-proxy:4000 ./install.sh
```

Each deployment's LiteLLM status is `synced`, `pending` (DGXMM retries; see
[troubleshooting](operations.md#litellm-status-stays-pending)), `disabled` (no key) or `none` (not
ready). The **LiteLLM alias** field in the deploy wizard sets a stable name such as `default-chat`.
Switching models moves the alias to the new deployment, so clients keep working.

## Bundled LiteLLM

For a fresh box without a LiteLLM, `./install.sh --with-litellm` (or `DGXMM_WITH_LITELLM=1`) enables the
compose profile `litellm`, which has two more services:

- `dgxmm-litellm`: LiteLLM pinned to a released, multi-arch image. It uses
  `deploy/litellm/config.yaml` (`store_model_in_db: true`, so DGXMM's registrations persist) and
  publishes `${DGXMM_BIND}:${DGXMM_LITELLM_PORT:-4000}`.
- `dgxmm-litellm-db`: `postgres:16` with its data in `data/litellm-db`. It runs as `DGXMM_UID`, so the
  files belong to you. It has no published port. Containers on the shared Docker network can reach
  it, so it is password-protected (`secrets/litellm_db_password`).

Neither has a fixed container name: they are called `<compose project>-dgxmm-litellm-1` and so on.
DGXMM and engines reach LiteLLM as `http://dgxmm-litellm:4000` on `DGXMM_NETWORK`.

Keys. The bundled LiteLLM always gets its **own** master key, `secrets/litellm_bundled_master_key`,
generated once and never overwritten. It is never the key of another LiteLLM.

- Fresh install: `config.yaml` is generated with `litellm_url: http://dgxmm-litellm:4000`. If
  `secrets/litellm_master_key` is empty, the installer writes the same key there too, so registration
  works with nothing to configure.
- Existing install that uses another LiteLLM: `config.yaml` and `secrets/litellm_master_key` are kept
  and the installer prints a note. To switch DGXMM to the bundled LiteLLM, set `litellm_url` to
  `http://dgxmm-litellm:4000` and the master key from `secrets/litellm_bundled_master_key` on DGXMM's
  LiteLLM page (Integrations > LiteLLM). If a container named `litellm-proxy` runs, the installer
  warns that a second LiteLLM will run and needs a free port.

At the end the installer prints the single endpoint, `http://<host>:<port>/v1`. Clients pass the DGXMM
model name as `model`. Create virtual keys on DGXMM's LiteLLM page, or call `POST /key/generate` on
the endpoint with the bundled master key. `./install.sh status` lists both containers.

If the database or LiteLLM fails to start, the installer still finishes DGXMM's setup (and shows the
first-run admin password). It then prints the redacted LiteLLM and database logs and exits non-zero.

**Turning it off:** set `DGXMM_WITH_LITELLM=0` in `.env` and run `./install.sh`. It stops and removes
the two containers and leaves `litellm` out of `COMPOSE_PROFILES` for that run. `data/litellm-db` and
the secrets are kept; remove `litellm` from `COMPOSE_PROFILES` in `.env` too, so a manual
`docker compose up -d` does not start it again. `uninstall --purge` deletes `data/litellm-db` but leaves
the images. Remove them with `docker image rm postgres:16 ghcr.io/berriai/litellm:<tag>` (see
`docker-compose.yaml` for the exact tag).

For backups of the LiteLLM database (virtual keys, budgets, spend), see
[backup and restore](operations.md#backup-and-restore).

## Hugging Face token

Gated repos (Llama, Gemma, …) need a token from an account that has accepted the model's licence on
huggingface.co. DGXMM uses it for catalog details and downloads. Engines run offline from the cache, so
they never need it.

```bash
HF_TOKEN=hf_xxx ./install.sh                  # when secrets/hf_token is missing or empty
# or, to replace an existing token:
umask 077; printf '%s' 'hf_xxx' > secrets/hf_token; docker restart dgxmm
```

## Tuning the memory budget

- **`safety_margin_gib`** (8) is the most important knob. It is the memory DGXMM always leaves free,
  and the level below which the watchdog goes critical and starting deployments are aborted. Lower it
  only if the box runs nothing but DGXMM.
- **`os_reserve_gib`** (4) adds to the margin for the OS and desktop. On a headless box you can lower it.
  Memory used by other processes is measured and counted separately as "external" either way.
- **Engine overheads** (`vllm_runtime_fixed_gib`, `vllm_cuda_graph_gib`, `llamacpp_runtime_fixed_gib`)
  apply to every estimate. Calibration already corrects per model from measurements, so change these
  only if every model is off in the same direction.
- **`auto_protect: true`** lets the watchdog stop the lowest-priority deployment when `MemAvailable`
  drops below the margin. Give important deployments a higher **priority** (0–100) in the deploy
  wizard.
- Per deployment, the biggest levers are context length, concurrent sequences, the KV pool size
  (`kv_tokens`), and the KV cache dtype (`fp8` halves KV memory).
