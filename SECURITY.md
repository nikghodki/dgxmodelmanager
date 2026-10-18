# Security policy

## Reporting a vulnerability

Please do not report security problems in public issues, pull requests or discussions.

Report them privately through GitHub's private vulnerability reporting:
[open a draft security advisory](https://github.com/nikghodki/dgxmodelmanager/security/advisories/new)
(repository **Security** tab, **Report a vulnerability**).

Please include:

- the DGXMM commit (`git rev-parse --short HEAD`) and how it is installed (bind address, reverse proxy,
  bundled LiteLLM, decision model on or off);
- the steps to reproduce, and what an attacker needs first (network access, a `viewer` or `admin`
  account, access to the host);
- the impact you expect, and a proof of concept if you have one.

Leave out real secrets: passwords, keys, tokens and the contents of `secrets/` and `.env`.

DGXMM has a single maintainer, so reports are handled on a best-effort basis. Fixes are made on `main`
and listed in the [changelog](CHANGELOG.md), and the advisory is published once a fix is available.

## Supported versions

Only the latest commit on `main` is supported. There are no release branches and no backports: to get a
fix, upgrade with `./install.sh upgrade` (see
[upgrading an existing install](docs/getting-started/upgrade.md#upgrading-an-existing-install)).

## Scope

In scope: the code in this repository, which includes:

- the DGXMM backend and web UI;
- the installer (`install.sh`) and `docker-compose.yaml`;
- the decision-model image and launcher (`deploy/decider/`);
- the bundled LiteLLM wiring (`deploy/litellm/`);
- the example nginx configuration (`deploy/nginx-dgxmm.conf`).

Examples: authentication or role bypass, CSRF, a way for a `viewer` to change state or read admin data,
escaping the `dgxmm.managed=true` container guard, a secret leaked into logs, the UI or the API, injection
through model metadata, logs or the decision model, a way to make an insight action, a typed ⌘K command or a
schedule do more than its dialog shows, the compatibility check fetching anything but its allow-listed Hub
files, code in a model repository that the remote-code scan executes, and integrity problems in the pinned
images and model snapshot. Code the scan merely fails to flag is a known limitation (see its
[blind spots](docs/security.md#deploy-time-advice-hub-fetches-and-the-code-scanner)), not a vulnerability.

Out of scope:

- Vulnerabilities in vLLM, llama.cpp, LiteLLM, Postgres, Docker or the NVIDIA stack themselves. Report
  those to their projects. Do tell us if DGXMM uses them in an unsafe way.
- Behaviour the documentation already names as a design decision or residual risk: the Docker socket is
  root-equivalent, so an `admin` is trusted like root; engine ports on `127.0.0.1` have no
  authentication; secrets passed as command-line arguments are masked only on a best-effort basis. See
  [residual risks](docs/security.md#residual-risks) and
  [known limitations](docs/operations/known-limitations.md).
- Installs exposed to the internet without a reverse proxy and TLS, against the
  [security hardening](docs/security/hardening.md) advice.
- Denial of service that needs an `admin` account, or that needs volumes far beyond the documented
  limits.

## How DGXMM is secured

The threat model, with assets, trust boundaries, mitigations and residual risks, is in
[docs/security.md](docs/security.md). Deployment advice is in
[security hardening](docs/security/hardening.md).
