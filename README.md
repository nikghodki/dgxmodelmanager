# DGX Model Manager (DGXMM)

Web UI + API for running LLMs on an NVIDIA DGX Spark (GB10, 128 GB unified memory) or similar GPU host:
see what is on disk, deploy vLLM or llama.cpp containers behind an out-of-memory gate, switch models with
automatic rollback, keep LiteLLM in sync, watch tok/s, TTFT and KV usage, plan which models fit together,
browse/download from Hugging Face (with llmfit scores) and get benchmark-backed tuning advice.

Design: `docs/superpowers/specs/2026-10-15-dgx-model-manager-design.md`. Plan: `docs/superpowers/plans/`.

## Install

Needs: Linux (arm64 or x86_64) with an NVIDIA GPU, Docker Engine with the compose v2 plugin, and the
NVIDIA Container Toolkit (`nvidia` runtime). Run as a normal user in the `docker` group (not root).

    git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager && ./install.sh

The installer checks the host, writes `.env` (compose settings) and `config.yaml` (rendered from
`config.example.yaml` with your paths), creates `data/`, `secrets/` and the docker network, builds and starts
the container, waits for the health check and prints the URLs plus the one-time `admin` password (also saved
in `secrets/admin_password`). Open `http://<host>:8090/` (redirects to `/dgxmm/`).

Re-running `./install.sh` upgrades in place; existing `.env`, `config.yaml`, `data/` and `secrets/` are never
overwritten. Other commands: `./install.sh upgrade` (git pull + install), `status`, `logs [-f]`,
`uninstall [--purge]`, and `--dry-run` with any of them. Settings (env vars or flags, see `./install.sh --help`):

| Variable | Default | |
|---|---|---|
| `DGXMM_PORT` | `8090` | host port of the UI |
| `DGXMM_BIND` | `0.0.0.0` | `127.0.0.1` when a reverse proxy fronts DGXMM (example: `deploy/nginx-dgxmm.conf`) |
| `DGXMM_NETWORK` | `dgxmm` | docker network shared with engine containers (and LiteLLM / your proxy) |
| `DGXMM_HF_CACHE` | `~/.cache/huggingface` | Hugging Face cache |
| `DGXMM_MODELS_DIR` | `~/models` | GGUF files |
| `HF_TOKEN` | | written to `secrets/hf_token` (gated models) |
| `LITELLM_URL`, `LITELLM_MASTER_KEY` | | LiteLLM registration (needs `store_model_in_db: true`); the key is read from a running `litellm-proxy` container when not given |
| `DGXMM_PULL_ENGINES=1` | off | pre-pull the vLLM and llama.cpp images |

Behind an existing reverse proxy on its own network, e.g.:
`DGXMM_BIND=127.0.0.1 DGXMM_NETWORK=gateway ./install.sh`.

Only one DGXMM may run per host: it adopts every container labelled `dgxmm.managed=true`, so the installer
refuses to start a second instance from another checkout.

## How memory safety works

- Budget = MemTotal − 8 GiB safety margin − 4 GiB OS reserve − committed reservations − external usage.
- Each deployment's reservation is computed from `config.json` / GGUF headers (weights + KV + state +
  activations + runtime), then corrected by what was measured on this box.
- Per-deployment usage = cgroup anon+kernel memory + NVML per-process GPU memory (CUDA allocations on GB10
  are not charged to the container cgroup). Anything not attributed to a managed container counts as
  "external" and is never touched.
- vLLM's `--gpu-memory-utilization` is always derived as reservation / MemTotal.
- If something doesn't fit, the UI offers plans (fp8 KV, shorter context, fewer sequences, smaller quant,
  sleep or stop lower-priority deployments). Nothing is evicted without confirmation; the watchdog only stops
  a model on its own when `auto_protect: true`.

## Security

- Username/password login (argon2id), HTTP-only SameSite=Strict cookies, CSRF tokens, login rate limiting,
  roles `admin` / `viewer`, audit log of every action.
- DGXMM only stops/removes containers labelled `dgxmm.managed=true`; flags passed to engines are allowlisted.
- The Docker socket is mounted, which is root-equivalent on the host: keep the UI on the LAN/tailnet.
- Engine ports are published on 127.0.0.1 only; clients go through LiteLLM (`:4000`).

## Develop

    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest -q
    docker run --rm -u "$(id -u):$(id -g)" -v $PWD/frontend:/app -w /app -e HOME=/tmp node:22-bookworm-slim sh -c "npm ci && npx vitest run && npm run build"
    scripts/validate_on_box.sh                   # end-to-end on the DGX (deploys small models)

## Troubleshooting

- *Download fails with "owned by root"*: an engine container created that HF cache folder as root;
  run the `sudo chown -R <uid>:<gid> …` shown in the error (the uid:gid DGXMM runs as, DGXMM_UID:DGXMM_GID in `.env`).
- *LiteLLM status stays `pending`*: check `store_model_in_db: true` and the key in `secrets/litellm_master_key`.
- *Catalog says "llmfit unavailable"*: `docker exec dgxmm llmfit --version`; everything else keeps working.
- *Deployment failed with exit 137*: the kernel OOM-killed it; DGXMM adds 10% to the next estimate for that config.
