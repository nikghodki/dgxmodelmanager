# DGXMM documentation

| Document | For | Contents |
|---|---|---|
| [Project README](../README.md) | everyone | What DGXMM is, features, quick start, calling a model, common install options |
| [Usage](usage.md) | users | The UI page by page: dashboard, deploying, deployment detail (API access, LiteLLM tab), models on disk (delete, clean up), catalog, planner, processes, LiteLLM and virtual keys, users, calling models |
| [Configuration](configuration.md) | operators | `config.yaml` reference, environment overrides, `.env` and installer variables, secrets, model directories, engine images, networking, reverse proxy and trusted proxies, LiteLLM integration and runtime settings, bundled LiteLLM, HF token, memory tuning |
| [Operations](operations.md) | operators | What the installer does, upgrade, upgrading an existing install, backup and restore, uninstall, logs, health, freeing disk space, troubleshooting, password reset, known limitations, security hardening |
| [Architecture](architecture.md) | developers | Components, module map, deploy flow, memory model, state machine, rollback, watchdog, LiteLLM sync, weight deletion, process view, background loops, data model, HTTP API, frontend, security model |
| [Changelog](../CHANGELOG.md) | everyone | What changed in each release, with upgrade notes |
| [Contributing](../CONTRIBUTING.md) | developers | Development setup, tests, code layout, commit style |

`docs/superpowers/` contains the original design specifications and implementation plans (design
history). They describe how DGXMM was planned. Where they differ from the code, the documents above and
the code are current.

Screenshots used in these documents are in [`images/`](images/).
