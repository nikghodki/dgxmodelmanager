---
description: "Install DGXMM on a DGX Spark with one command: flags, a dry run and what the installer does."
---

# Install

Check the [requirements](requirements.md) first. Then, as a normal user in the `docker` group:

```bash
git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager && ./install.sh
```

The installer checks the host, writes `.env` and `config.yaml` for your paths, creates `data/`,
`secrets/` and the Docker network, then builds and starts the `dgxmm` container and waits for its
health check. It is non-interactive and safe to run again. At the end it prints something like:

```text
==> DGXMM is running. Open:
    http://192.0.2.10:8090/dgxmm/
    http://dgx-host.local:8090/dgxmm/  (mDNS)
    http://dgx-host:8090/dgxmm/
    admin user: admin   password: <generated>
    (shown once; also saved to secrets/admin_password. You must change it at first login.)
    manage: ./install.sh status | logs | upgrade | uninstall
```

## Dry run

To see first what it would do (host checks, files, ports) without changing anything:

```bash
./install.sh --dry-run
```

## Optional components

=== "Bundled LiteLLM"

    A LiteLLM proxy with its own Postgres that serves every deployment on one endpoint, already wired
    to DGXMM:

    ```bash
    ./install.sh --with-litellm
    ```

    The installer then waits for LiteLLM and adds:

    ```text
    ==> LiteLLM is running. Single OpenAI-compatible endpoint for all DGXMM models:
        http://192.0.2.10:4000/v1
        ...
        from containers on network dgxmm: http://dgxmm-litellm:4000/v1
        every ready DGXMM deployment is one model on this endpoint: pass "model": "<model name>"
    ```

=== "Decision model"

    The optional local decision model: build its image, download and verify the model, enable it and
    wait until it is healthy. You can also enable it later in the UI under
    **Integrations > Decision model**.

    ```bash
    ./install.sh --with-decider
    ```

    !!! warning "Memory"
        The decision model reserves about 13 GiB of memory while it runs (9 GiB with d3-nano). See
        [running the decision model](../operations/decision-model.md).

=== "Both"

    ```bash
    ./install.sh --with-litellm --with-decider
    ```

!!! note "The first start pulls the vLLM image"
    The first start runs a short one-off vLLM container to read which model architectures vLLM
    supports, so the vLLM image (several GB) is pulled in the background. Set `DGXMM_PULL_ENGINES=1` to
    pull both engine images during install instead.

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
   `litellm_public_url` (see [bundled LiteLLM](../configuration/litellm.md#bundled-litellm)). Existing files are
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
10. **Decision model** (only with `--with-decider`). See [decision model](../operations/decision-model.md): builds the
    decider image, downloads and verifies the pinned model snapshot, enables the decider and waits up to
    `DGXMM_DECIDER_TIMEOUT` seconds (default 900) until it is healthy, then prints its status. Each step
    is skipped when already done. On failure it says which step failed; DGXMM itself keeps running.

Other commands: `./install.sh status`, `logs [-f]`, `upgrade` and `uninstall [--purge]`. The `Makefile`
has the same targets (`make install`, `make status`, …).

## Common install options

Options are flags or environment variables; `./install.sh --help` lists all of them.

| Goal | Command |
|---|---|
| Another UI port | `./install.sh --port 8091` |
| Bind to loopback, behind a reverse proxy | `DGXMM_BIND=127.0.0.1 ./install.sh` |
| Join an existing proxy's Docker network | `DGXMM_BIND=127.0.0.1 DGXMM_NETWORK=gateway ./install.sh` |
| Trust a reverse proxy's `X-Forwarded-For` | `DGXMM_TRUSTED_PROXIES=172.18.0.0/16 ./install.sh` |
| Where models live | `./install.sh --hf-cache /data/hf --models-dir /data/gguf` |
| Token for gated Hugging Face models | `HF_TOKEN=hf_xxx ./install.sh` |
| Bundled LiteLLM (one `/v1` endpoint) | `./install.sh --with-litellm` (`--litellm-port 4001` for another port) |
| Local decision model (diagnosis, suggestions, insights, ~13 GiB GPU memory) | `./install.sh --with-decider` (see [running the decision model](../operations/decision-model.md)) |
| Time zone of daily jobs and schedules | `DGXMM_TZ=Europe/Berlin ./install.sh` (detected from the host by default) |
| Remove it again with its model files | `./install.sh uninstall --purge --purge-decider-model` |
| Register with an existing LiteLLM | `LITELLM_URL=http://litellm-proxy:4000 LITELLM_MASTER_KEY=sk-xxx ./install.sh` |
| See what would happen | `./install.sh --dry-run` |

Settings are written once to `.env`. After that, `.env` wins over flags: to change a setting, edit
`.env` and re-run `./install.sh`. The LiteLLM connection can also be changed at runtime on the LiteLLM
page. For reverse proxies, extra model directories and every `config.yaml` key, see
[configuration](../configuration/index.md).

## Manage the install

```bash
./install.sh status              # container health, URLs, running engines
./install.sh logs [-f]           # DGXMM container log
./install.sh upgrade             # git pull --ff-only, then rebuild and restart (your data is kept)
./install.sh uninstall           # remove the container and image; keeps data/, secrets/, config
```

Running engine containers keep serving while DGXMM restarts. Model files (HF cache, models directory)
are never deleted by the installer. See [upgrade](upgrade.md) and [uninstall](uninstall.md).

## Next steps

- [First login](first-login.md)
- [Configuration](../configuration/index.md)
- [Networking and reverse proxy](../configuration/networking.md)
