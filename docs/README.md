# DGXMM documentation

| Document | For | Contents |
|---|---|---|
| [Project README](../README.md) | everyone | What DGXMM is, features, quick start, calling a model, common install options |
| [Usage](usage.md) | users | The UI page by page: dashboard, deploying, the start-up queue, deployment detail (API access, LiteLLM tab), models on disk (delete, clean up), catalog, planner, processes, LiteLLM and virtual keys, the decision model, users, calling models |
| [Configuration](configuration.md) | operators | `config.yaml` reference (including the decision model and the start-up queue), environment overrides, `.env` and installer variables, secrets, model directories, engine images, networking, reverse proxy and trusted proxies, LiteLLM integration and runtime settings, bundled LiteLLM, HF token, memory tuning, decision-model runtime settings, API limits |
| [Operations](operations.md) | operators | What the installer does, upgrade, upgrading an existing install, the decision model, backup and restore, uninstall, logs, health, freeing disk space, admin command line, troubleshooting, password reset, known limitations, security hardening |
| [Architecture](architecture.md) | developers | Components, module map, deploy flow, start-up queue, memory model, state machine, rollback, watchdog, decision model, LiteLLM sync, downloads, weight deletion, process view, background loops, data model, HTTP API, frontend, security model |
| [Security](security.md) | operators, developers | Threat model: assets, trust boundaries, the decision model's container, network, bearer key, image supply chain, API boundary, prompt injection, data at rest, residual risks, how the controls were verified |
| [Decision-model evaluation](decider-evaluation.md) | developers | The decision model's features on real data (failure logs, model repos, traffic), how the thresholds were chosen, performance, capacity, d3-mini vs d3-nano |
| [Changelog](../CHANGELOG.md) | everyone | What changed in each release, with upgrade notes |
| [Contributing](../CONTRIBUTING.md) | developers | Development setup, tests, code layout, commit style |

`docs/superpowers/` contains the original design specifications and implementation plans (design
history). They describe how DGXMM was planned. Where they differ from the code, the documents above and
the code are current.

Screenshots used in these documents are in [`images/`](images/).
