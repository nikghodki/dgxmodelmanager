# DGXMM documentation sources

This directory is the source of the documentation site,
**<https://nikghodki.github.io/dgxmodelmanager/>**, built with MkDocs Material (`mkdocs.yml` at the
repository root). Read the documentation there; the Markdown files here render on GitHub too.

| Directory | Site section |
|---|---|
| [`index.md`](index.md) | Home |
| [`getting-started/`](getting-started/features.md) | Features, requirements, install, first login, upgrade, uninstall |
| [`user-guide/`](user-guide/dashboard.md) | The UI page by page |
| [`architecture/`](architecture/index.md) | How DGXMM works inside, with diagrams |
| [`configuration/`](configuration/index.md) | Every setting with its default |
| [`operations.md`](operations.md), [`operations/`](operations/troubleshooting.md) | Backups, logs, admin command line, troubleshooting |
| [`security.md`](security.md), [`security/`](security/overview.md) | Threat model, application security model, hardening, security policy |
| [`decider-evaluation.md`](decider-evaluation.md) | Decision-model evaluation on real data |
| [`contributing/`](contributing/index.md), [`changelog.md`](changelog.md) | Include `CONTRIBUTING.md` and `CHANGELOG.md` from the repository root |
| [`images/`](images/) | Screenshots of the UI against made-up demo data (regenerate: see below) |
| `overrides/` | Theme overrides and build hooks (not pages) |
| `superpowers/` | Design history: the original specifications and plans (not published on the site) |

Build the site locally:

```bash
python3 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements-docs.txt
.venv/bin/mkdocs serve        # http://127.0.0.1:8000/dgxmodelmanager/
.venv/bin/mkdocs build --strict
```

## Screenshots

The images in `images/` are generated, not taken from a real box. One command regenerates all of them (Docker
only; nothing is started on the GPUs and no network is used during the capture):

```bash
tools/screenshots/run.sh                       # every image, light and dark, into docs/images/
tools/screenshots/run.sh dashboard deployments  # only these (names as in tools/screenshots/shots.py)
```

It copies the built UI out of the `dgxmm:<version>` image (`DGXMM_IMAGE=` to use another), serves it in a pinned
Playwright Chromium and answers every API request from `tools/screenshots/demo_data.py`: two nodes (dgx-1,
dgx-2) with made-up deployments, documentation addresses (192.0.2.0/24) and the user `admin`. The capture runs
with the network off, and the PNGs are then optimised losslessly with oxipng. `tools/screenshots/shots.py` lists
each image: its page, viewport, theme pair and the clicks before the capture. A UI change that needs data the
demo does not have shows up as *not in the demo data* in the output; add it to `demo_data.py`, never data from a
real installation.
