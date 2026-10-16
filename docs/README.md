# DGXMM documentation

| Document | For | Contents |
|---|---|---|
| [Project README](../README.md) | everyone | What DGXMM is, quick start, common install options |
| [Usage](usage.md) | users | The UI page by page: dashboard, deploying, deployment detail, catalog, planner, users, calling models |
| [Configuration](configuration.md) | operators | `config.yaml` reference, environment overrides, `.env`, secrets, model directories, engine images, networking, reverse proxy, LiteLLM, HF token, memory tuning |
| [Operations](operations.md) | operators | What the installer does, upgrade, backup and restore, uninstall, logs, health, troubleshooting, password reset, security hardening |
| [Architecture](architecture.md) | developers | Components, module map, deploy flow, memory model, state machine, rollback, watchdog, background loops, data model, frontend, security model |
| [Contributing](../CONTRIBUTING.md) | developers | Development setup, tests, code layout, commit style |

`docs/superpowers/` contains the original design specifications and implementation plans (design
history). They describe how DGXMM was planned. Where they differ from the code, the documents above and
the code are current.

Screenshots used in these documents are in [`images/`](images/).
