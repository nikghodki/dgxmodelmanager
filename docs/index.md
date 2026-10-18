---
description: "DGXMM is a self-hosted web app that serves local LLMs on an NVIDIA DGX Spark, with a memory check before every start and one OpenAI-compatible endpoint."
hide:
  - navigation
---

# DGX Model Manager

DGXMM is a self-hosted web app for serving local LLMs on an NVIDIA DGX Spark (GB10, 128 GB unified
memory) or a similar single-GPU Linux box. It deploys the models on your disk as vLLM or llama.cpp
containers, checks before every start that a model fits next to everything already running, and serves
all of them through one OpenAI-compatible LiteLLM endpoint with virtual keys. It runs as a single
container and installs with one command.

![DGXMM dashboard: free memory, the unified-memory bar, KPI tiles and deployments (demo data)](images/dashboard-light.png#only-light)
![DGXMM dashboard: free memory, the unified-memory bar, KPI tiles and deployments (demo data)](images/dashboard-dark.png#only-dark)

## Key capabilities

<div class="grid cards" markdown>

-   :material-memory:{ .lg .middle } **Memory-safe deploys**

    ---

    Every deploy is previewed against a unified-memory budget with a safety margin. When a model does
    not fit, DGXMM offers plans and does nothing until you confirm one.

    [:octicons-arrow-right-24: Deploying a model](user-guide/deploying.md)

-   :material-engine:{ .lg .middle } **vLLM and llama.cpp**

    ---

    Safetensors models on vLLM, GGUF files on llama.cpp. Switch or reconfigure with automatic rollback;
    engines start one at a time.

    [:octicons-arrow-right-24: Deployments](user-guide/deployments.md)

-   :material-api:{ .lg .middle } **One endpoint, virtual keys**

    ---

    Each ready deployment is one model on a LiteLLM `/v1` endpoint. Virtual keys with model
    restrictions, budgets and expiry; an existing LiteLLM or a bundled one.

    [:octicons-arrow-right-24: LiteLLM](user-guide/litellm.md)

-   :material-lightbulb-on-outline:{ .lg .middle } **Advisory decision model**

    ---

    An optional small local model for 16 features: failure diagnosis, suggested configs, health triage,
    routing advice and more. Rules first, never without your confirmation.

    [:octicons-arrow-right-24: Decision model](user-guide/decision-model.md)

-   :material-bell-badge-outline:{ .lg .middle } **Insights and schedules**

    ---

    Findings from background checks with an action that opens the normal dialog, and sleep/wake
    schedules proposed from 30 days of traffic.

    [:octicons-arrow-right-24: Insights](user-guide/insights.md)

-   :material-chart-line:{ .lg .middle } **Observability**

    ---

    Live throughput, latency, KV-cache usage and queue depth, 30 days of history, tuning advice, a
    built-in benchmark and every process using memory on the box.

    [:octicons-arrow-right-24: Dashboard](user-guide/dashboard.md)

</div>

## Quick start

On a DGX Spark with Docker, the Compose plugin and the NVIDIA runtime (see the
[requirements](getting-started/requirements.md)):

```bash
git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager && ./install.sh
```

Open the URL it prints, sign in as `admin` with the printed password and choose a new one. Add
`--with-litellm` for a bundled LiteLLM endpoint and `--with-decider` for the decision model, or run
`./install.sh --dry-run` first to see what it would do.

## Where to next

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Getting started**

    ---

    Requirements, install options, first login, upgrade and uninstall.

    [:octicons-arrow-right-24: Install](getting-started/install.md)

-   :material-book-open-variant:{ .lg .middle } **User guide**

    ---

    The UI page by page, from the dashboard to calling your models.

    [:octicons-arrow-right-24: User guide](user-guide/dashboard.md)

-   :material-sitemap-outline:{ .lg .middle } **Architecture**

    ---

    Components, the memory budget, the deployer, the decision model and the data model.

    [:octicons-arrow-right-24: Architecture](architecture/index.md)

-   :material-tune-variant:{ .lg .middle } **Configuration**

    ---

    Every setting with its default, networking, reverse proxies and LiteLLM.

    [:octicons-arrow-right-24: Configuration](configuration/index.md)

-   :material-wrench-outline:{ .lg .middle } **Operations**

    ---

    Backups, logs, the admin command line and troubleshooting.

    [:octicons-arrow-right-24: Operations](operations.md)

-   :material-shield-lock-outline:{ .lg .middle } **Security**

    ---

    The threat model, hardening and how to report a vulnerability.

    [:octicons-arrow-right-24: Security](security/overview.md)

</div>

DGXMM is open source under the MIT License:
[github.com/nikghodki/dgxmodelmanager](https://github.com/nikghodki/dgxmodelmanager). The screenshots on
this site show a demo instance with made-up data.
