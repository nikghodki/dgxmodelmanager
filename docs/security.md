# Security and threat model

This document describes what DGXMM protects, where the trust boundaries are, what each mitigation does and what
risk remains. It covers the whole application and, in more detail, the decision model ("decider",
`vllm-sr/d3-mini`). Implementation details of authentication, sessions and the container guard are in
[architecture: security model](architecture.md#security-model); deployment advice is in
[operations: security hardening](operations.md#security-hardening).

Contents:
- [Assets](#assets)
- [Actors and trust boundaries](#actors-and-trust-boundaries)
- [DGXMM application](#dgxmm-application)
  - [Untrusted text and the event loop](#untrusted-text-and-the-event-loop)
  - [Live updates (SSE topics)](#live-updates-sse-topics)
- [Decision model (decider)](#decision-model-decider)
  - [Container](#container)
  - [Network](#network)
  - [Bearer key](#bearer-key)
  - [Image supply chain](#image-supply-chain)
  - [API boundary](#api-boundary)
  - [Untrusted input and prompt injection](#untrusted-input-and-prompt-injection)
  - [Data at rest](#data-at-rest)
- [Residual risks](#residual-risks)
- [Verifying the controls](#verifying-the-controls)

## Assets

| Asset | Where | Why it matters |
|---|---|---|
| Host control | Docker socket mounted into `dgxmm` | Starting containers is root-equivalent on the host |
| User accounts and sessions | `data/dgxmm.sqlite3` (`users`, `sessions`) | Admin sessions can deploy, stop and delete |
| Secrets | `secrets/` (HF token, LiteLLM keys), `app_settings` (LiteLLM master key), `data/secrets/decider_api_key` | Access to gated models, the LiteLLM proxy, the decider |
| Model weights and caches | HF cache, GGUF roots | Large, slow to replace; their code (remote code, `d3_server.py`) runs in containers |
| Running deployments | engine containers | Users depend on them; memory is shared (GB10 unified memory) |
| Decision log, diagnoses | `decisions`, `diagnoses`, `artifact_kinds` tables | Must not leak secrets that appear in engine logs |

## Actors and trust boundaries

| Actor | Trust |
|---|---|
| DGXMM admin | Trusted with root (the Docker socket). Every change needs this role. |
| DGXMM viewer | Read-only; must not change anything or see secrets, the live container logs, users or host paths. Viewers do see the **redacted log tails** stored with failed deployments (Events tab) and the redacted excerpt of a diagnosis: the same redaction as the admin log view (see [residual risks](#residual-risks)). |
| Anonymous network client | Can reach the login page only. |
| Other containers on the shared engine network (`docker_network`, e.g. LiteLLM, nginx, TTS) | Not trusted with DGXMM's secrets. They can reach engine ports on that network. |
| Other local users and processes on the host | Not trusted. They can reach ports published on the host. |
| Model repositories (config.json, chat templates, model cards, custom code) and everything an engine writes to its log | **Untrusted input.** A repository author controls most of it. |
| PyPI / container registries | Trusted only through pinned digests and hashes. |
| vLLM engine containers | Run as root with the **whole HF cache mounted read-write** (downloads need it) and may run a repository's `trust_remote_code` code, so they are treated as able to write anything in the HF cache — including the decider's snapshot directories. |

The HF cache is therefore a trust boundary: nothing in it is trusted as code by path. The decider runs only the
private, hash-verified copies of its pinned files (see [container](#container): Integrity); the weights it reads
from the cache are checked against the pinned manifest but the large shards only by size.

Boundaries: browser ↔ DGXMM API (authentication, CSRF, roles); DGXMM ↔ Docker daemon (container guard);
DGXMM ↔ engine containers (engine network); DGXMM ↔ decider (internal network + bearer key); untrusted
text ↔ decider prompt (quoted state blocks, redaction, closed option sets); decider answer ↔ any change
(advisory only, admin confirmation).

## DGXMM application

Summary (details in [architecture](architecture.md#security-model)):
- argon2id passwords (≥ 12 characters), login rate limit (5/min per IP and per username), forced change of
  generated passwords;
- HTTP-only `SameSite=Strict` session cookie (only its SHA-256 stored), CSRF header on every non-GET request;
- `viewer` / `admin` roles checked on every route; the audit log records every change (and failures of the
  important ones);
- the container guard: DGXMM only creates, stops or removes containers labelled `dgxmm.managed=true`;
- strict engine config schemas and an allowlist for extra engine flags;
- secrets (LiteLLM master key, virtual keys, the decider key) are never returned, logged or audited; process
  command lines are masked before they leave the server, and so is live engine output: the admin-only container
  log view (`GET /api/deployments/{id}/logs`) is redacted line by line (`memory/redact.py: redact_log`, which also
  masks multi-line PEM private keys, and runs of three or more base64 key lines whose BEGIN line a tail cut off) —
  engines can echo tokens, `Authorization` headers and URL credentials.

### Untrusted text and the event loop

DGXMM is one asyncio process: anything slow on its event loop stalls the API, SSE, the budget loop and the
watchdog's memory protection. Engine logs, chat templates, repository names and config.json values are attacker
text (a repository author controls them), so:
- every log line is cut at 2 KiB (dropping the trailing partial word, so no secret fragment escapes redaction) and
  a diagnosis reads at most the last 512 KiB, before any rule runs (`deployer/logtail.py: cap_line`,
  `decider/diagnosis.py: bounded`);
- every regex applied to such text uses bounded quantifiers (an echoed 44 KB architecture name made the old
  `Initializing a .* engine .* with config:` rule backtrack for 42 s);
- rule matching, patch derivation, log excerpts, the stored log tail, parser-suggestion rules and the injection
  check run in worker threads (`asyncio.to_thread`), never on the event loop.
`tests/test_regex_safety.py` runs every regex of the modules that read untrusted text (diagnosis, parser
suggestions, quoting, kinds, workload, log tails, redaction) against adversarial 200 KB inputs (repeated literals,
long single lines, 1 MB logs) in a child process with a deadline: each must finish in under 0.5 s.

### Live updates (SSE topics)

`GET /api/stream` (any signed-in user) pushes events. Topics by role (`api/stream.py`):

| Topic | Viewer | Admin | Carries |
|---|---|---|---|
| `budget`, `metrics`, `alert` | yes | yes | memory budget, live metrics, watchdog alerts (also `GET /api/host`) |
| `deployment`, `deployment_removed` | yes | yes | deployment views (as `GET /api/deployments`) |
| `inventory`, `download`, `benchmark` | yes | yes | what `GET /api/inventory`, `/api/downloads`, `/deployments/{id}/benchmarks` return |
| `diagnosis` | yes | yes | only the id of a deployment with a new diagnosis |
| `decider` | no | yes | decider image build progress (docker build output, the image's package list) |

A viewer therefore receives nothing over SSE that it cannot read over REST. The number of SSE subscribers is not
limited (each is one in-memory queue of at most 256 events).

## Decision model (decider)

The decider is a small classifier (`vllm-sr/d3-mini`, pinned snapshot `97dd054d…`) served by the snapshot's own
`d3_server.py` in a container DGXMM builds and starts (`dgxmm/engines/decision.py`, `deploy/decider/`). Five
features ask it typed questions (failure cause, tool-call / reasoning parser, plan ranking, model kind, workload
class). It is **advisory only**: it never runs anything, and every change it informs goes through an existing
flow that an admin confirms.

### Container

The container runs code shipped with the model snapshot, so it is locked down
(`DecisionEngine.container_spec`, `DockerController.run`):

| Control | Setting |
|---|---|
| User | `10001:10001` (the image's `decider` user; `decider_user`) — d3_server needs no root |
| Root filesystem | read-only; the only writable paths are tmpfs: `/tmp` (`decider_tmp_gib`, `nosuid,nodev`; exec is needed because Triton/TorchInductor load the kernels they compile there) and `/run/dgxmm-decider` (64 KiB, `noexec`, mode 0700, owned by the decider user) |
| Capabilities | all dropped (`cap_drop: ALL`), `no-new-privileges` |
| IPC | private (not `--ipc host` like the vLLM engines) |
| Processes | `pids_limit` = `decider_pids_limit` (threads count) |
| Memory | cgroup limit `decider_memory_limit_gib`, swap disabled (see below) |
| Mounts | read-only bind mounts only: the selected model's repo directory of the HF cache (never the whole cache, which holds the host's HF token) and, for a model downloaded into the shared content-addressed store (d3-nano, DGXMM's own downloads), one bind per store file its snapshot links to, at the same path (`shared_blob_mounts`: never the store directory, its `.refs`/`.lock` entries or another repo's files). No Docker socket, no other host path, no Triton cache shared with vLLM |
| Integrity | the launcher copies exactly the files of the image's pinned hash list (the snapshot's code, manifest, config, tokenizer and readout files) into a fresh directory on the container's own `/tmp` tmpfs, checks every *copy* against its pinned hash and refuses to start (exit 3) on any mismatch or missing file. It runs under `python3 -I` (no script directory, current directory, user site or `PYTHON*` variables on the import path) and imports only from that copy, which it makes read-only; the weight / tokenizer data files that the pinned `MODEL_MANIFEST.json` lists are linked into it (data suffixes only, never code). So a file planted next to the snapshot's code (`json.py`, `numpy/`, a `.pth`) is never imported, and nothing can be swapped between the check and the import. d3_server then verifies the weights against the pinned manifest (sizes for the large shards, see [residual risks](#residual-risks)) |
| Restart | `restart: no`; DGXMM recreates the container (a new one gets a fresh tmpfs and key file) |

**Memory: reservation and limit.** Two different numbers bound the decider:
- **Budget reservation (13 GiB, fixed).** DGXMM plans memory against the host's MemAvailable, which on GB10
  unified memory includes GPU allocations. The 12.6 GiB peak of the initial study was the MemAvailable drop of a d3-mini server
  under load; with the hardened container the drop measured 12.45 GiB from start to ready and after 26 diagnosis
  questions, *including* the 0.23 GiB the decider writes to its `/tmp` tmpfs (CUDA JIT cache 195 MiB, Triton 39 MiB;
  tmpfs pages count as used memory). `/tmp` is therefore capped at `decider_tmp_gib: 0.75`, so even a full `/tmp`
  (12.45 − 0.23 + 0.75 ≈ 12.97 GiB) stays inside the reservation.
- **cgroup limit (`decider_memory_limit_gib: 16`).** GPU allocations (`cudaMalloc`) are *not* charged to the
  container's memory cgroup on GB10, so this limit caps host-side memory: the process (2.1 GiB anonymous), tmpfs,
  and the page cache of the ~9 GB of weights it reads at start. That page cache makes the cgroup peak vary (9.3–12.7
  GiB measured, against a ~3 GiB steady state); 16 GiB is the highest peak plus ~26 %. At the limit the kernel first
  reclaims page cache, then kills the decider (DGXMM reports a failed deployment) before it can exhaust host memory.
  `decider_pids_limit: 512` is about 5x the measured peak of 82–98 processes and threads.
- **Early exits.** If the container exits before its key is delivered (entrypoint exit 2 = snapshot missing, 3 =
  snapshot integrity check failed, 4 = no key), DGXMM records that exit code and the redacted log tail as the
  deployment's failure (state `failed`, visible in Events and to the diagnosis) and then removes the container.
- **No user-namespace remapping.** DGXMM does not configure Docker's `userns-remap`; uid 10001 in the container is
  uid 10001 on the host. Everything it can write is tmpfs, and its only bind mount is read-only.

### Network

- The decider **publishes no host port** (`publish_bind` is ignored for it; its view has no `host_endpoint`).
- It joins only `decider_network` (default `dgxmm-decider`), an **internal** Docker network: no route out of the
  host (the d3 snapshot's code cannot phone home) and no route in from other networks. DGXMM creates it as
  internal on the first decider start (and refuses to use a non-internal network of that name) and attaches its
  own container to it, also again at every start (an upgraded DGXMM container rejoins automatically).
- DGXMM reaches the decider by container name on that network (`http://dgxmm-decider:<port>`), even when
  `engine_base_host` is set for the other engines.
- Who can reach it: DGXMM, and any other container an administrator explicitly attaches to `dgxmm-decider`.
  Not the host's other containers (LiteLLM, nginx, …) on the engine network, not the internet, not the LAN. The
  host itself can still open connections to the container's bridge IP (Docker does not filter host → container
  traffic on bridge networks), which is one reason for the key.
- With `decider_network: ''` the decider shares `docker_network`: every container on it can then reach its
  port, and only the bearer key protects it.
- **Decider → DGXMM.** The reverse direction exists too: DGXMM's own container is on `dgxmm-decider`, so the
  decider (which runs the snapshot's code) can open connections to DGXMM's TCP listener on that network. It gets no
  more than any network client: the login page, and nothing without a session. It is never a trusted proxy,
  whatever `trusted_proxies` says: the TCP listener applies `X-Forwarded-For` / `-Proto` through
  `dgxmm/proxy_headers.py`, which drops every forwarding header from a peer in the decider network's subnets (read
  from Docker at start and every 30 s), so the decider cannot spoof a client address (e.g. to dodge the per-IP
  login limit) even with a broad `trusted_proxies: 172.16.0.0/12`. CLI tokens are only accepted on the admin unix
  socket, which it cannot reach. With `decider_network: ''` this guard does not apply (keep `trusted_proxies`
  narrow then).

### Bearer key

`/v1/systemone` and `/v1/models` need `Authorization: Bearer <key>` (constant-time comparison in d3_server;
`/health` stays open for probes).
- The key is generated once (`secrets.token_urlsafe(32)`) in `data/secrets/decider_api_key` (mode 600, directory
  700) and read at use time.
- It is **never in the container's environment, command line or `docker inspect`**: right after the container
  starts, DGXMM writes it through `docker exec` stdin into `/run/dgxmm-decider/api_key` (mode 0400, tmpfs; written
  to a temporary name and renamed). `put_archive` cannot be used: it cannot write into a tmpfs and refuses a
  read-only root filesystem. If the write fails, the container is removed at once.
- The launcher (`deploy/decider/launch.py`) waits for the file, reads it, **deletes it** and passes the key to
  d3_server's `serve(api_key=…)` directly, so it is not in `/proc/*/environ` or any child process (Triton /
  Inductor compile workers). It refuses to start without a key, or when `DECISION_API_KEY` is set in the
  environment.
- It is never returned by the API, written to the audit log or logged.
- **Rotation**: `POST /api/decider/rotate-key` (admin; "Rotate key" on the Decision model page) or
  `docker exec dgxmm python -m dgxmm.admin rotate-decider-key [--wait 90]`. DGXMM stops a running decider (the old
  key dies with its process), writes a new key (atomic rename), and starts the decider again with the new key file.
  The features fall back to rules for the few minutes the decider takes to load. While a rotation runs, Start and
  enable of the decider answer 409 `rotation_in_progress`. The CLI hands the request to the running app through a
  mode-600 request file; the app writes an "in progress" marker, then the result. Exit 0 rotated, 1 failed, 2 not
  picked up, 3 still in progress; re-running the CLI then waits for the same rotation instead of starting another.
  Both are audited (`decider_rotate_key`, with `via: api|cli`).

### Upgrading a decider started by an older build

A decider container keeps running across a DGXMM upgrade, so one started by an older build (for example from an
older image tag, with its key in the environment, a published port, root, a shared network or a writable root
filesystem) stays as it was until it is restarted. `GET /api/decider/status` reports `upgrade: {needed, reasons}` by inspecting the running container
(image tag, key in the environment, user, published ports, network, read-only root), DGXMM logs a warning at
start, and the Decision model page shows the reasons. To upgrade:
1. build the new image (Decision model page → **Build image…**, or `POST /api/decider/image/build`);
2. restart the decider with a new key: **Restart with a new key** (or `POST /api/decider/rotate-key`, or
   `docker exec dgxmm python -m dgxmm.admin rotate-decider-key`). It stops the old container, writes a new key and
   starts a hardened container from the new image; the old key stops working.

### Image supply chain

`deploy/decider/Dockerfile`:
- the base image is pinned **by digest** (the same vLLM build as the vLLM engine);
- the only Python additions are the four wheels in `deploy/decider/requirements.lock` (transformers 5.17.0,
  flash-linear-attention 0.5.2, fla-core 0.5.2, tokenizers 0.23.3), resolved for the base (linux/arm64,
  CPython 3.12) and installed with `--require-hashes --no-deps --only-binary=:all:`: a changed file on the index
  fails the build, pip cannot pull in anything unlisted, and no sdist (code) runs at install time;
- `pip check` must not report any conflict the base did not already have;
- the full Python package list (an SBOM of the image's Python environment) is printed in the build log (`SBOM`
  lines) and kept at `/usr/local/share/dgxmm-decider/sbom-pip.txt`;
- `scripts/decider_image_check.sh` builds the image twice from scratch and compares the package sets;
  `scripts/decider_lock.sh` regenerates the lock (review the diff);
- the image is built locally and never pushed. Its tag (`images.decision`, default `dgxmm-decider:3`) changes
  whenever the image's contract with the app changes, so an older image is reported missing and rebuilt.

### API boundary

| Route | Viewer | Admin | Rate limit (per user) | Audited as |
|---|---|---|---|---|
| `GET /api/decider/status`, `GET /api/decider/image` | yes | yes | — | — |
| `GET /api/deployments/{id}/diagnosis`, `…/workload` | yes | yes | — | — |
| `PUT /api/decider/settings` | 403 | yes | 10/min shared "admin" bucket | `decider_settings` |
| `POST /api/decider/enable` | 403 | yes | admin bucket | `decider_enable` / `decider_enable_failed` |
| `POST /api/decider/disable` | 403 | yes | admin bucket | `decider_disable` |
| `POST /api/decider/image/build` | 403 | yes | admin bucket | `decider_image_build` |
| `POST /api/decider/models/download` | 403 | yes | admin bucket | `decider_model_download` |
| `POST /api/decider/rotate-key` | 403 | yes | admin bucket | `decider_rotate_key` |
| `GET /api/decider/decisions` | 403 | yes | 120/min | — (read) |
| `POST /api/decider/decisions/{id}/feedback` | 403 | yes | 60/min | `decider_feedback` |
| `POST /api/decider/suggest-config` | 403 | yes | 30/min, 120/min for all users | — (each answer is a `decisions` row) |
| `POST /api/deployments/preview` | 403 | yes | plan ranking by the model: 30/min, 120/min for all users (over it: no model ranking, never 429) | — (each ranking is a `decisions` row) |

`POST /api/deployments/preview` (which runs the plan recommendation) is admin-only like every deployment change;
its only callers, the deploy wizard and the switch / reconfigure dialogs, are admin-only too. Each preview that
does not fit asks the decider at most once (cached by its inputs, 2 s deadline), and only while the admin's
`decider_plan` bucket allows it: over the limit the preview still answers, with the plans' safety annotations but
no model ranking. A rules answer (exactly one allowed plan) is recorded once per cache period, not once per
preview. The admin-only live log view is covered under [DGXMM application](#dgxmm-application).

Anonymous requests get 401; state-changing requests without the session's CSRF header get 403; over the limit
→ 429 `rate_limited` with `Retry-After`. The rate limiter runs after the role check, so a viewer never consumes an
admin's budget. Bodies under `/api/decider` are limited to 64 KiB before parsing (413 `request_too_large`, also
for chunked bodies); request models reject unknown fields, `model_ref` is ≤ 1024 characters, `actions` ≤ 16,
`features` ≤ 16 entries with names matching `[a-z][a-z_]*`. `tests/test_decider_api_security.py` checks the full
anonymous / viewer / admin matrix for every route.

### Untrusted input and prompt injection

What reaches the decider is assembled by DGXMM, never by the untrusted source:
- **Fixed questions and option sets.** The question text is a constant per feature; untrusted text only goes
  into the `state`. The answer must be one of the offered option keys (anything else is discarded).
- **Quoted, capped, redacted state.** Untrusted text (log tail and error lines, model name, chat-template
  excerpt, config.json excerpt, file names) is wrapped in labelled `<<<` … `>>>` blocks marked "untrusted",
  length-capped (logs: last 120 lines / 6 KiB; template 2.5 KB; config 1.5 KB), and redacted
  (`memory/redact.py`: tokens of known shapes, `Authorization`/`Cookie`/`X-…-Key` headers, `KEY=value` pairs,
  URL credentials, PEM private-key blocks). Any run of three or more `<` or `>` inside the text is spaced out,
  so quoted text can never close its block and pose as trusted state.
- **Validated answers.** The decider's replies are untrusted too (`decider/client.py: capped_request`,
  `parse_answer`): of every reply — answers, `/health`, the authenticated probe and the start-up warm-up questions
  — at most 64 KiB is read (streamed, so a larger reply is cut off, not buffered); warm-up answers are validated
  like real ones, and the probe and `/health` bodies must be JSON objects of the expected shape; NaN / Infinity are refused by the JSON parser, every
  probability must be a number in [0, 1] for an option that was offered, their sum must be within 0.02 of 1
  (renormalised), and the choice must be the most probable option; its probability is the confidence. A question
  has at most 128 options. Anything else is a `bad_reply` error: the feature falls back to rules and the failure
  counts toward the circuit breaker. Decision, diagnosis and kind rows stored before this check are served with
  non-finite numbers as `null`.
- **Rules first.** Most answers come from deterministic rules; the model is asked only where they have no answer,
  and its answer is used only above a per-feature confidence threshold.
- **Advisory only, closed actions.** What an answer can do:
  - *diagnosis*: show a cause and a fix hint; "Apply fix…" only *prefills* the Reconfigure dialog with a patch
    validated against the engine schema, which the admin reviews and confirms (with rollback). The patch that
    weakens security — `trust_remote_code: true` — is offered only when a rule matched the transformers error
    phrase ("contains custom code which must be executed …") in the log, never on the model's word. That phrase
    can itself be echoed from attacker-shaped text (a repository's own config or code), so the rule is not proof;
    the patch only prefills the dialog, which shows `trust_remote_code: default → true` under an explicit
    "Security-sensitive change" warning (the engine will run the repository's code; the message that suggested
    it can come from the repository itself) and needs the admin's confirmation; "Apply fix…" is never a primary
    action;
  - *parser suggestion*: prefill an empty wizard field (auto-filled only at or above the parser threshold, default
    0.8, and only when no rule answered); the user still deploys, and a wrong parser only makes that deployment
    fail to start (diagnosed as `bad_parser_flag`);
  - *plan recommendation*: mark one of the plans that already passed the deterministic safety filter (never one
    that stops/sleeps an active, depended-on, higher-priority or system deployment); the admin chooses and applies;
  - *kind*, *workload*: labels and ordering only.
  `tests/test_decider_injection.py` drives every feature with an adversarial decider that always answers the most
  disruptive option with certainty and checks that no deployment, container or config changes.
- **Measured resilience** of the real d3-mini against injected text: see
  [verifying the controls](#verifying-the-controls).

### Data at rest

- `decisions` rows hold the feature, subject, choice, probabilities, latency, model revision, the **SHA-256 of
  the input** (never the state text) and feedback.
- `diagnoses` rows hold the cause, source, confidence, a fixed summary (plus the matched rule line, redacted, at
  most 200 characters), the fix hint, the validated patch and a **redacted** log excerpt (last 120 lines / 6 KiB).
- `artifact_kinds` holds kind, source and confidence per artifact.
- **Retention**: `decisions` keeps 30 days and at most 50 000 rows (`decider_retention_days`, `decider_max_rows`),
  `diagnoses` the latest 20 per deployment (deleted with the deployment), pruned once a minute by the metrics loop,
  so neither table grows without bound however often previews, suggestions or failures happen.
- `tests/test_decider_injection.py` plants HF, OpenAI, AWS, GitHub, bearer, database-URL, password, API-key and
  PEM secrets in a failure log and checks that none reaches the decider state, these tables or their API views.

## Residual risks

- **Docker socket.** DGXMM (and so every admin) is root-equivalent on the host. The decider hardening limits what
  the *decider's* code can do, not what DGXMM can do.
- **GPU memory is outside the cgroup limit.** A runaway decider could allocate GPU (unified) memory beyond its
  13 GiB reservation, which the kernel limit does not cap. The memory watchdog stops the decider when its
  measured memory exceeds the reservation by more than `decider_overuse_margin_gib` (two samples in a row), and
  picks it before any user deployment when `auto_protect` must free memory; the supervisor restarts it only after
  its backoff and never under memory pressure. On GB10 part of a container's GPU memory is not attributed to it
  (neither its cgroup nor NVML report it), so the overuse check sees only what is attributable. **`auto_protect`
  defaults to `false`**: with the defaults, unattributed decider GPU memory is covered by neither check, and
  memory turning critical only raises the alert. Set `auto_protect: true` in `config.yaml` (and restart DGXMM)
  to have the watchdog stop the decider — before any user deployment — when memory turns critical. If stopping
  the decider fails, it is skipped as a victim until its state changes, and the user-deployment order applies.
- **Host → decider.** Local processes on the host can connect to the decider's bridge IP; the bearer key is the
  control there. Anyone who can read `data/secrets/` (host root, the DGXMM user, backups) can use the decider.
- **The model's code runs in the container.** `d3_server.py` and its runtime are hash-pinned, but a flaw in them
  (or in torch/transformers) could be exploited by a crafted request from a holder of the key. Mitigated by the
  container controls and the internal network (no exfiltration path).
- **Redaction is best effort.** A secret in an unusual format inside an engine log may survive into the redacted
  log tail or diagnosis excerpt (both visible to viewers on the deployment page) — the same exposure as the log itself, which admins can
  read anyway. Do not put secrets in engine arguments.
- **Injection can still move answers between options.** It cannot add options or actions, but it can (rarely)
  make the model prefer a wrong cause, parser or allowed plan; the admin sees the source ("AI") and confidence
  and confirms every change. Thresholds and the feedback loop limit the impact.
- **Decider weights are size-checked, not hashed, at start.** Whoever can write the HF cache (the host user, a
  vLLM container) can replace a weight shard of the same size. That changes the decider's answers (it is advisory:
  every change still needs an admin), not the code it runs; `decider download` re-hashes every file.
- **Base image and wheels are trusted at their pinned digests/hashes**: pinning prevents silent change, not a
  compromise that was already present when the pins were taken.
- **No user-namespace remapping** (see [container](#container)): a container escape would land as uid 10001 on
  the host, not as root, but with that uid's host access.
- **Rate limits are per process and in memory**: they reset on restart and are not shared between instances
  (DGXMM runs as a single instance).

## Verifying the controls

Automated (backend suite): `tests/test_decider_hardening.py` (container spec, docker kwargs, key delivery, internal
network), `tests/test_decider_api_security.py` (access matrix, rate limits, size limits, audit, rotation),
`tests/test_decider_injection.py` (secrets, quoting, adversarial decider), `tests/test_decider_supply_chain.py`
(digest, hash-pinned lock, unprivileged image), `tests/test_decider_launch.py` (private verified copy, planted
files never imported, `python3 -I`), `tests/test_regex_safety.py` (adversarial regex timing, matching off the event
loop), `tests/test_decider_output_validation.py` (answer validation, reply size cap, legacy non-finite rows),
`tests/test_preview_security.py` (preview role and plan-ranking rate limit), `tests/test_watchdog_decider.py`
(runaway decider stopped first, restart backoff), `tests/test_decider_retention.py`, `tests/test_proxy_guard.py`
(decider network never a proxy), `tests/test_stream_topics.py` (SSE topics by role).

Manual, on the DGX (throwaway containers; the recorded results follow below):
- `scripts/decider_image_check.sh` — two from-scratch builds, same Python package set;
- `scripts/decider_spec_check.py` — inside a throwaway container with the Docker socket: the real
  `DecisionEngine` spec started by `DockerController.run`; `docker inspect` shows no key and the hardening flags;
  the engine's health check passes; rotation refuses the old key and accepts the new one;
- `scripts/decider_injection_eval.py` — injection carriers against the real d3-mini.

Measured on the DGX (GB10), 2026-10-16, with throwaway containers on a throwaway internal network:

| Check | Result |
|---|---|
| Inside the decider | uid 10001; CapEff/CapBnd 0; NoNewPrivs 1; seccomp on; `/` and the model mount read-only; `/tmp` and `/run/dgxmm-decider` tmpfs; no Docker socket; the key file is gone after start |
| Key exposure | key in `docker inspect`: no; in `/proc/1/environ`: no; in any process environment: no |
| Ports / egress | no published port; outbound connections (1.1.1.1:443, huggingface.co) fail |
| Reachability | DGXMM-side peer on the internal network: `/health` 200, `/v1/systemone` without key 401, with key 200; container on the shared `gateway` network: unreachable by name and by IP; host → bridge IP: reachable (key required) |
| App path (`DecisionEngine` + `DockerController.run` from a container) | started and key delivered in 0.5 s; healthy (engine health check incl. authenticated probe) after 236 s; inspect shows the hardening flags and only the read-only model bind |
| Rotation | old key → 401, new key → 200 on the restarted container; new key not in inspect |
| Limits | cgroup peak 9.3–12.7 GiB, 82–97 pids, no OOM kill (runs with 14 GiB / 1024, 12 GiB / 512 and 14 GiB / 512); a 64-request burst (16 concurrent) all 200 in 46 s |
| Memory budget | MemAvailable drop 12.45 GiB (start → ready → 26 questions), `/tmp` use 234 MiB |
| Image | two `--no-cache` builds: identical 267-package set; a lock with one altered hash fails the build (`THESE PACKAGES DO NOT MATCH THE HASHES`) |

Prompt injection against the real d3-mini (`scripts/decider_injection_eval.py`, 298 injected questions built with
DGXMM's own state builders; "flipped" = the injected answer is the attacker's target and the clean one was not;
"accepted" = the flip is also at/above the feature's threshold):

| Feature | Questions | Flipped | Flipped and accepted | Strongest carrier |
|---|---|---|---|---|
| Diagnosis (6 carriers in the engine log) | 156 | 15 (9.6 %) | 2 (1.3 %) | an injected exception line (9/26 flipped, 1 accepted) |
| Parser (template comment, repo name, break-out, flood) | 96 | 13 (13.5 %) | 1 (1.0 %) at the default 0.8 (0 at 0.9, the threshold when measured) | template comment / flood (5/24 each) |
| Kind (config.json fields, file names) | 40 | 0 | 0 | — |
| Plan (repo name, config value) | 6 | 0 | 0 | — |

Both accepted flips made the model answer `trust_remote_code_required`, for which DGXMM offers no patch on the
model's word; quoted break-out attempts (`>>>` …) and forged JSON answers moved nothing above threshold. The
evaluation ran when the parser threshold was 0.9; the real-data evaluation (docs/decider-evaluation.md) later set
the default to 0.8. Re-scored at 0.8, one parser flip qualifies for auto-fill: a comment in a chat template moved a
reasoning-parser answer from `none` (0.66) to `deepseek_r1` (0.83). It can only prefill an empty field of the
wizard for a model no parser rule recognises (the rules answered every real repo of the evaluation corpus), and the
admin still deploys it. In addition, a model answer is never auto-filled when the chat template or the repo name
contains a run of `<<<` / `>>>` or answer-like text naming a parser option (`config_suggest.injection_shaped`;
that carrier, and every other carrier of the evaluation, is shown as a suggestion only); no real template of the
evaluation corpus trips it. Raising the parser threshold to 0.9 also rules the case out, at the cost of fewer
suggestions.
Accuracy under the strongest
diagnosis carrier fell from 24/26 to 17/26 (the UI then shows "AI" as the source, with the confidence).
