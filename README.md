# DGX Model Manager (DGXMM)

Web UI + API for running LLMs on the DGX Spark (`dgx-host`, GB10, 128 GB unified memory):
see what is on disk, deploy vLLM or llama.cpp containers behind an out-of-memory gate, switch models with
automatic rollback, keep LiteLLM in sync, watch tok/s, TTFT and KV usage, plan which models fit together,
browse/download from Hugging Face (with llmfit scores) and get benchmark-backed tuning advice.

Design: `docs/superpowers/specs/2026-10-15-dgx-model-manager-design.md`. Plan: `docs/superpowers/plans/`.

## Run

    cp config.example.yaml config.yaml          # adjust if needed
    mkdir -p data secrets && chmod 700 secrets
    touch secrets/hf_token                      # optional: paste an HF token for gated repos
    # LiteLLM master key (never printed):
    docker exec litellm-proxy sh -c 'grep -E "^\s*master_key:" /app/config.yaml | sed -E "s/.*master_key:\s*//"' > secrets/litellm_master_key
    chmod 600 secrets/*
    docker compose up -d --build
    docker compose logs dgxmm | grep "first run"   # one-time admin password

Open `http://<dgx>/dgxmm/` (via nginx-gateway) or `http://127.0.0.1:8090/dgxmm/` on the box.
LiteLLM must have `store_model_in_db: true` for automatic registration.

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
    docker run --rm -u 1000:1000 -v $PWD/frontend:/app -w /app -e HOME=/tmp node:22-bookworm-slim sh -c "npm ci && npx vitest run && npm run build"
    scripts/validate_on_box.sh                   # end-to-end on the DGX (deploys small models)

## Troubleshooting

- *Download fails with "owned by root"*: an engine container created that HF cache folder as root;
  run the `sudo chown -R 1000:1000 …` shown in the error.
- *LiteLLM status stays `pending`*: check `store_model_in_db: true` and the key in `secrets/litellm_master_key`.
- *Catalog says "llmfit unavailable"*: `docker exec dgxmm llmfit --version`; everything else keeps working.
- *Deployment failed with exit 137*: the kernel OOM-killed it; DGXMM adds 10% to the next estimate for that config.
