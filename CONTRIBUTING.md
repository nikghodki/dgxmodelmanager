# Contributing

Thanks for helping improve DGXMM. This page covers the development setup. Read
[docs/architecture.md](docs/architecture.md) first for how the pieces fit together.

## Backend (Python 3.12+)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                 # backend + installer tests (or: make test)
```

The tests need no GPU, no Docker daemon and no network. Docker, NVML, engines and the Hub are faked
(`tests/fakes.py`). The installer tests run `install.sh` in a temporary copy of the repo against stub
commands (`tests/fixtures/install_stubs/`). `make lint` runs shellcheck on the installer in a container.

To try backend changes on a real box, rebuild and restart the container with `./install.sh`. Do not
start a second backend from source on a host where DGXMM is installed: every instance adopts the
containers labelled `dgxmm.managed=true`.

## Frontend (Node 22)

```bash
cd frontend
npm ci
npx vitest run        # unit and component tests
npm run build         # type-check + production build into frontend/dist
```

Without Node on the host, run the same commands in a container from the repo root:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/frontend:/app" -w /app -e HOME=/tmp \
  node:22-bookworm-slim sh -c "npm ci && npx vitest run && npm run build"
```

**Dev server against a running backend.** Vite serves the UI with hot reload at
`http://127.0.0.1:5173/dgxmm/` and forwards `/dgxmm/api` to `http://127.0.0.1:8090`. That is the
installed DGXMM on the same host, so sign in with a normal DGXMM account:

```bash
cd frontend && npm run dev
# or, without Node on the host:
docker run --rm -it --network host -u "$(id -u):$(id -g)" -v "$PWD/frontend:/app" -w /app -e HOME=/tmp \
  node:22-bookworm-slim sh -c "npm ci && npm run dev -- --host 127.0.0.1"
```

To work from a laptop, forward the port (`ssh -L 5173:127.0.0.1:5173 <dgx-host>`).

## End-to-end check

`scripts/validate_on_box.sh` drives a running DGXMM through its API, and it **deploys and removes small
models**, so warn other users of the box first:

```bash
DGXMM_USER=admin DGXMM_PASSWORD='…' scripts/validate_on_box.sh
```

## Code layout

```text
dgxmm/            FastAPI backend (module map: docs/architecture.md#backend-module-map)
  api/            HTTP routers, one per area
  deployer/       deployment lifecycle, states, reconcile, adopt
  engines/        vLLM and llama.cpp plug-ins
  memory/         budget, estimates, calibration, host probes
  planner/ catalog/ inventory/ metrics/ advisor/ auth/
frontend/src/
  pages/          one component per route
  components/ui/  shadcn/ui primitives
  components/app/ app building blocks (shell, tables, memory bar, plan chooser, palette)
  lib/ hooks/     pure helpers and hooks (with *.test.ts next to them)
  api.ts auth.tsx stream.ts types.ts
tests/            pytest suite
install.sh        installer (see ./install.sh --help)
docs/             user documentation; docs/superpowers/ holds design history
deploy/           example reverse-proxy config
```

Conventions:

- Only `dgxmm/docker_ctl.py` talks to Docker, and every write must keep the `dgxmm.managed=true` guard.
- New engine flags go into the allowlist in `dgxmm/engines/argguard.py`, never into free-form strings.
- Memory estimates must stay conservative. Calibration may raise them, never lower them.
- In the UI, every status shows an icon and text (never colour alone), and every icon-only button has an
  `aria-label`. Use the shared dialogs, not `alert()`/`confirm()`.
- The repo-root `.gitignore` matches `lib/`, so new files under `frontend/src/lib/` need
  `git add -f`.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope): …`, `fix(ui): …`,
  `docs: …`, `test: …`, `style: …`, `chore: …`.
- Keep `.venv/bin/pytest -q`, `npx vitest run` and `npm run build` green. Add tests with every
  behaviour change.
- Update the docs in `docs/` when you change behaviour, configuration keys or installer options.
- Never commit `.env`, `config.yaml`, `data/` or `secrets/` (they are gitignored).
