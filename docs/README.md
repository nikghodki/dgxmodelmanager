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
| [`images/`](images/) | Screenshots of a demo instance with made-up data |
| `overrides/` | Theme overrides and build hooks (not pages) |
| `superpowers/` | Design history: the original specifications and plans (not published on the site) |

Build the site locally:

```bash
python3 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements-docs.txt
.venv/bin/mkdocs serve        # http://127.0.0.1:8000/dgxmodelmanager/
.venv/bin/mkdocs build --strict
```
