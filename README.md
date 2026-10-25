# DGX Model Manager (DGXMM)

[![Tests](https://github.com/nikghodki/dgxmodelmanager/actions/workflows/tests.yml/badge.svg)](https://github.com/nikghodki/dgxmodelmanager/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-nikghodki.github.io%2Fdgxmodelmanager-1d4ed8.svg)](https://nikghodki.github.io/dgxmodelmanager/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DGXMM is a self-hosted web app for serving local LLMs on an NVIDIA DGX Spark (GB10, 128 GB unified
memory) or a similar single-GPU Linux box (see [supported hardware](https://nikghodki.github.io/dgxmodelmanager/getting-started/requirements/#supported-hardware)). It deploys the models on your disk as vLLM or llama.cpp
containers, checks before every start that a model fits next to everything already running, and
serves all of them through one OpenAI-compatible LiteLLM endpoint with virtual keys. An optional small
local decision model adds advice (why a start failed, which config, engine, parser or draft model to use,
whether a deployment is healthy, when it could sleep), always after DGXMM's own rules and never without
your confirmation. It runs as a single container and installs with one command.

**Documentation: <https://nikghodki.github.io/dgxmodelmanager/>**

![DGXMM dashboard: free memory, unified memory bar, KPI tiles and deployments (demo data)](docs/images/dashboard-light.png)

## Highlights

- **Memory-safe deploys:** every deploy is previewed against a unified-memory budget; when a model does not
  fit, DGXMM offers plans and does nothing until you confirm one.
- **vLLM and llama.cpp**, with switch and reconfigure that roll back automatically, and engines that start
  one at a time.
- **One LiteLLM endpoint** for every deployment, with virtual keys (your LiteLLM or a bundled one).
- **On-demand models:** a model can start on its first request and stop again after a few idle minutes,
  making room or queueing when memory is short. It works through LiteLLM on DGXMM's Docker network, so new
  deployments are always on unless you choose it (the deploy wizard preselects it when LiteLLM is set up).
  See [on-demand models](https://nikghodki.github.io/dgxmodelmanager/user-guide/on-demand/).
- **An optional decision model** for 16 advisory features, **insights** from background checks,
  **schedules** proposed from traffic, and **⌘K Ask** in the command palette.
- **Observability and administration:** live metrics, tuning advice, a benchmark, the process list,
  `admin` and `viewer` roles and an audit log.

See [all features](https://nikghodki.github.io/dgxmodelmanager/getting-started/features/) and the
[screenshots](https://nikghodki.github.io/dgxmodelmanager/getting-started/features/#screenshots).

## Quick start

On Linux with an NVIDIA GPU, Docker with the Compose plugin and the NVIDIA runtime
([requirements](https://nikghodki.github.io/dgxmodelmanager/getting-started/requirements/)):

```bash
git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager && ./install.sh
```

Open the URL it prints and sign in as `admin` with the printed password. Add `--with-litellm` for a
bundled LiteLLM endpoint, `--with-decider` for the decision model, or `--dry-run` to see first what it
would do. Next: [install options](https://nikghodki.github.io/dgxmodelmanager/getting-started/install/),
[first login](https://nikghodki.github.io/dgxmodelmanager/getting-started/first-login/) and
[upgrade](https://nikghodki.github.io/dgxmodelmanager/getting-started/upgrade/).

> **Cloned before 2026-10-18?** The repository history was rewritten, so `./install.sh upgrade` refuses
> to fast-forward an older clone. Move it once with `git fetch && git reset --hard origin/main`, then run
> `./install.sh` (your `.env`, `config.yaml`, `secrets/` and `data/` stay as they are; see
> [clones from before the history rewrite](https://nikghodki.github.io/dgxmodelmanager/getting-started/upgrade/#clones-from-before-the-2026-10-18-history-rewrite)).

## Documentation

| Section | What it covers |
|---|---|
| [Getting started](https://nikghodki.github.io/dgxmodelmanager/getting-started/features/) | Features, requirements and supported hardware, install, first login, upgrade, uninstall |
| [User guide](https://nikghodki.github.io/dgxmodelmanager/user-guide/dashboard/) | The UI page by page: dashboard, models and catalog, deploying, deployments, insights, schedules, ⌘K, the decision model, LiteLLM, processes, users, settings |
| [Architecture](https://nikghodki.github.io/dgxmodelmanager/architecture/) | Components, request and data flow, memory budget, deployer, decision model, job engine, schedule runner, data model, frontend |
| [Configuration](https://nikghodki.github.io/dgxmodelmanager/configuration/) | Every setting with its default |
| [Operations](https://nikghodki.github.io/dgxmodelmanager/operations/) | Backups, logs, admin command line, troubleshooting |
| [Security](https://nikghodki.github.io/dgxmodelmanager/security/overview/) | Threat model, hardening, security policy |
| [Evaluation](https://nikghodki.github.io/dgxmodelmanager/decider-evaluation/) | The decision model's 16 features measured on real data |

The site is built from [`docs/`](docs/) (see [`docs/README.md`](docs/README.md)).

## Security

DGXMM mounts the Docker socket so it can start engine containers: **access to it is equivalent to root
on the host**, so only give the `admin` role to people you would trust with root, and keep the UI off
the public internet. See the [security overview](https://nikghodki.github.io/dgxmodelmanager/security/overview/).

**Reporting a vulnerability:** please do not open a public issue. Use GitHub's private vulnerability
reporting as described in [SECURITY.md](SECURITY.md), which also lists what is in scope.

## Contributing

Bug reports and feature requests are welcome as
[issues](https://github.com/nikghodki/dgxmodelmanager/issues/new/choose): the forms ask for what is needed
to reproduce a problem (version, host and GPU, `./install.sh status`, logs). The development setup, the
test suites and the code layout are in [CONTRIBUTING.md](CONTRIBUTING.md); the changelog is in
[CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
