# Contributing

Thanks for helping improve DGXMM. This page covers the development setup. Read
[the architecture overview](docs/architecture/index.md) first for how the pieces fit together.

## Backend (Python 3.12+)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                 # backend + installer tests (or: make test)
```

The tests need no GPU, no Docker daemon and no network. Docker, NVML, engines, the decision model and
the Hub are faked (`tests/fakes.py`). The installer tests run `install.sh` in a temporary copy of the
repo against stub commands (`tests/fixtures/install_stubs/`). The suite also includes:

- `tests/test_repo_hygiene.py`: tracked files must not contain the development machine's user name,
  host name, private or Tailscale IP addresses, or names on its forbidden list (some are stored only as
  hashes). Use the placeholders it names (`dgxuser`, `dgx-host`, the documentation ranges
  `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`). Fixtures may name only public Hugging Face
  repositories and need a provenance note.
- `tests/test_public_repo.py`: what a public repository needs: engine, base and CI images pinned by
  digest, local state (`.env`, `config.yaml`, `data/`, `secrets/`) gitignored, no tracked file over 1 MB,
  the security policy, the issue forms and pull request template, no dependabot version updates.
- `tests/test_docs_links.py`: every relative link and `#anchor` in the README, CHANGELOG, CONTRIBUTING,
  SECURITY and the pages under `docs/` resolves, links to the documentation site point at existing pages and
  headings, and the Decision model page's **Docs** link points at its page on the site.
- `tests/test_docs_site.py`: every page in `mkdocs.yml`'s nav exists, every page under `docs/` is in the nav,
  and the pinned documentation requirements carry hashes.
- `tests/test_frontend_hygiene.py`: no compiled `.js` next to the TypeScript sources.
- `tests/test_real_corpus.py` and `tests/test_decider_ops_corpora.py`: the decision-model rules against the
  real-data corpora in `tests/fixtures/` (see [docs/decider-evaluation.md](docs/decider-evaluation.md)).
- `tests/test_advanced_docs.py`: the overview table of `docs/decider-evaluation.md` has one row per feature
  group with the registry's threshold, and `dgxmm/decider/protocol.py` notes every threshold.
- `tests/test_advanced_routes.py`, `tests/test_advanced_jobs.py`, `tests/test_advanced_event_loop.py`: every
  route of the newer decision-model features in one anonymous / viewer / admin / CSRF matrix and
  rate-limited; the background jobs' budgets and that they cannot starve interactive questions; no blocking
  database or file work on the event loop in their async code.

Run one file with `.venv/bin/pytest -q tests/test_docs_links.py`.

**Pylint.** CI (`.github/workflows/pylint.yml`, Python 3.12) runs `pylint --errors-only dgxmm`, configured
in `pyproject.toml` (`[tool.pylint]`). Locally: `.venv/bin/pip install pylint && .venv/bin/pylint --errors-only dgxmm`.


**CI Tests.** The `.github/workflows/tests.yml` workflow runs on every push and pull request to main, with
three jobs: `backend` (pytest on Python 3.12; the whole suite, so the installer, docs-link, hygiene and
public-repo tests too), `frontend` (vitest, then the production build, on Node 22) and `shellcheck` (pinned
`koalaman/shellcheck:v0.11.0`) on every shell script. Backend tests install dev dependencies (`pip install -e '.[dev]'`) for pytest-asyncio. Tests
that need Docker or GPU are skipped when those aren't available. `tests/test_public_repo.py` checks what
the public repository needs: images pinned by digest, local state gitignored, the security policy and
the issue and pull request templates. To lint the workflows themselves:
`docker run --rm -v "$PWD:/repo:ro" -w /repo rhysd/actionlint:latest`.

**Shell scripts.** `make lint` runs shellcheck on the installer and its test stubs in a container. To
check every shell script:

```bash
docker run --rm -v "$PWD:/mnt" -w /mnt koalaman/shellcheck:v0.11.0 \
  install.sh tests/fixtures/install_stubs/* deploy/decider/entrypoint.sh scripts/*.sh
```

To try backend changes on a real box, rebuild and restart the container with `./install.sh`. Do not
start a second backend from source on a host where DGXMM is installed: every instance adopts the
containers labelled `dgxmm.managed=true`.

## Frontend (Node 22)

```bash
cd frontend
npm ci
npx vitest run        # unit and component tests (including theme contrast checks)
npx tsc --noEmit      # type-check only
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

## Documentation site

The site at <https://nikghodki.github.io/dgxmodelmanager/> is built from `docs/` with MkDocs Material
(`mkdocs.yml`, pinned with hashes in `requirements-docs.txt`) and published by
`.github/workflows/pages.yml` on every push to `main` that touches the documentation. Pull requests build it
with `mkdocs build --strict` (the `docs` job of the tests workflow), which fails on any warning, broken link
or missing `#anchor`.

```bash
.venv/bin/pip install --require-hashes -r requirements-docs.txt
.venv/bin/mkdocs serve           # http://127.0.0.1:8000/dgxmodelmanager/, reloads on save
.venv/bin/mkdocs build --strict  # what CI runs
```

- A new page goes into `docs/<section>/` and into `nav:` in `mkdocs.yml`, with a front-matter
  `description:` and a "Next steps" list where it helps.
- Headings get GitHub's ids on the site too, so `page.md#heading` links work in both places.
- `CHANGELOG.md`, `CONTRIBUTING.md` and `SECURITY.md` stay at the repository root and are included by
  `docs/overrides/hooks/root_files.py`, which rewrites their `docs/...` links for the site.
- Diagrams are Mermaid code blocks; Mermaid itself is served from the site
  (`docs/overrides/hooks/mermaid.py`), so pages load nothing from a CDN.
- Screenshots come in pairs (`name-light.png#only-light`, `name-dark.png#only-dark`), and the site shows the
  one that matches the reader's theme.

## End-to-end check

`scripts/validate_on_box.sh` drives a running DGXMM through its API, and it **deploys and removes small
models**, so warn other users of the box first:

```bash
DGXMM_USER=admin DGXMM_PASSWORD='…' scripts/validate_on_box.sh
```

## Adding a decision-model feature

Every feature follows the same pattern; the existing ones in `dgxmm/decider/` are the examples.

1. **Register it** in `dgxmm/decider/registry.py` with one `register(FeatureSpec(name, group, label,
   description, default_threshold, background=…, needs_model=…, viewer_visible=…))`. That gives it a switch
   and threshold on the Decision model page, the settings keys `decider.feature.<name>.*`, a rate-limit bucket
   `decider_<name>` and the client priority. Add its threshold note to `dgxmm/decider/protocol.py` and a row
   to the overview table of `docs/decider-evaluation.md` (`tests/test_advanced_docs.py` checks both).
2. **Rules first.** Write the deterministic rules and make the feature work with the decision model off.
   Ask the model only where the rules have no answer, through `Decider.decide` with a constant question, an
   option set built by code and a `state` in which untrusted text is quoted, capped and redacted
   (`quoted_block` and `safe_decide` in `dgxmm/decider/common.py`). Use the answer only at or above the threshold; explanations shown to users
   are templates, never model text. Log rules decisions too (`record_rules` / `arecord_rules`).
3. **Background work** goes through the insight engine: `ctx.insight_engine.register_job(name, feature,
   schedule, fn, budget_calls=…)` in `wire_features` (`dgxmm/decider/features.py`) or `wire_operations`.
   Findings are insights (`ctx.insights.upsert(..., dedupe_key=…, action=…)`) with an action descriptor
   that opens an existing dialog; the server never executes it.
4. **Routes** depend on `limited(feature_bucket(name), admin=…)` (`dgxmm/api/ratelimit.py`), which checks the
   role and the rate limit; state changes need the CSRF header like every other route. Keep blocking work off
   the event loop (`asyncio.to_thread`). Add them to the matrix in
   `tests/test_advanced_routes.py`.
5. **Evaluate it.** Put a corpus in `tests/fixtures/decider_corpus/<name>/` (public repos only, with a
   provenance note), add `eval_case(case)` to the feature's module, and run
   `.venv/bin/python tools/eval_feature.py <name> default --rules-only`, then against a decider with
   `--base http://<decider>:<port> --key-file data/secrets/decider_api_key --rate 1`. Choose the threshold
   from the Wilson 95 % lower bound and record the method, the numbers and the shortfalls in
   `docs/decider-evaluation.md`.
6. Document it in `docs/user-guide/decision-model.md`, `docs/configuration/decision-model.md` (any fixed
   values) and `docs/security.md` (new inputs, routes and what an answer can change).

## Code layout

```text
dgxmm/            FastAPI backend (module map: docs/architecture/index.md#backend-module-map)
  api/            HTTP routers, one per area
  deployer/       deployment lifecycle, states, reconcile, adopt
  engines/        vLLM, llama.cpp and decision-model plug-ins
  decider/        decision model: client, supervisor, image, snapshot, key, feature registry, insights,
                  job engine, and one module per feature
  memory/         budget, estimates, calibration, host probes
  planner/ catalog/ inventory/ metrics/ advisor/ auth/
frontend/src/
  pages/          one component per route
  components/ui/  shadcn/ui primitives
  components/app/ app building blocks (shell, tables, memory bar, plan chooser, palette)
  lib/ hooks/     pure helpers and hooks (with *.test.ts next to them)
  api.ts auth.tsx stream.ts types.ts
tests/            pytest suite (fixtures/ holds the real-data corpora)
install.sh        installer (see ./install.sh --help)
docs/             the documentation site's pages (mkdocs.yml); docs/superpowers/ holds design history
deploy/           example nginx config, bundled LiteLLM config (deploy/litellm/), decider image (deploy/decider/)
scripts/          end-to-end check, decider image and evaluation tools (scripts/decider_eval/)
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

Report bugs and ideas with the [issue forms](https://github.com/nikghodki/dgxmodelmanager/issues/new/choose),
and security problems privately as described in [SECURITY.md](SECURITY.md).

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope): …`, `fix(ui): …`,
  `docs: …`, `test: …`, `style: …`, `chore: …`.
- Keep `.venv/bin/pytest -q`, `npx vitest run` and `npm run build` green, and shellcheck clean. Add
  tests with every behaviour change.
- Decision-model features must stay advisory and fail open: rules first, the model only above its
  threshold, and no change without an admin's confirmation (see [docs/security.md](docs/security.md)).
- Update the docs in `docs/` when you change behaviour, configuration keys or installer options.
- Add user-visible changes to the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md).
- Screenshots in `docs/images/` must come from a demo instance with made-up data, never from a real box
  (they would show deployment, container and repository names and command lines).
- Never commit `.env`, `config.yaml`, `data/` or `secrets/` (they are gitignored).
