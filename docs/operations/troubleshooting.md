---
description: "Fixes for common problems: Docker, ports, memory, start-up, LiteLLM, downloads, the decision model, schedules and ⌘K."
---

# Troubleshooting

## `permission denied` on the Docker socket

Your user is not in the `docker` group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker            # or log out and back in
./install.sh
```

Do not run the installer with `sudo`. DGXMM's files must belong to your user.

## No `nvidia` runtime

The installer stops with "Docker has no nvidia runtime registered". Install the NVIDIA Container
Toolkit (package `nvidia-container-toolkit`), then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}'    # must list "nvidia"
```

If the installer says `nvidia-smi` cannot see a GPU, fix the NVIDIA driver first.

## Port already in use

The installer stops if another process uses the UI port. On a first install, pick another port:

```bash
./install.sh --port 8091
```

If `.env` already exists, set `DGXMM_PORT=8091` in `.env` and run `./install.sh`. Engine containers use
host ports 8100–8199 (`port_range` in `config.yaml`) and skip ports other containers have published.

## A model does not fit

The memory check shows the shortfall and ways to make it fit (see
[usage](../user-guide/deploying.md)). Check also:

- **External processes** (Dashboard): notebooks, other containers or desktop apps use memory that DGXMM
  will not hand out. Stop what you do not need. If *external* looks several GiB too high and a ready
  deployment's **Measured** memory is far below its GPU use, see
  [GPU memory not counted](#gpu-memory-of-a-deployment-is-not-counted).
- **Sleeping deployments** still hold their weights. Stop them to free everything.
- **Estimates** that look too high: run the model once. After it becomes ready, DGXMM measures it and
  later estimates for that configuration are based on the measurement.
- On a box dedicated to DGXMM you can lower `os_reserve_gib` (see
  [tuning](../configuration/models-and-engines.md#tuning-the-memory-budget)).

## Engine startup timeout

A deployment fails with "startup timed out after 900s". Open its **Events** tab for the log tail:

- **First start of a vLLM model**: kernel compilation and CUDA graph capture can take a long time. The
  results are cached in `~/.cache/vllm`, `~/.cache/flashinfer` and `~/.triton`, so the next start is
  faster. For very large models raise `startup_timeout_s` in `config.yaml`.
- **Image pull**: the first deployment with an engine may have to pull its image. Pre-pull with
  `DGXMM_PULL_ENGINES=1 ./install.sh` or `docker pull <image>`.
- "aborted: available memory fell below the safety margin during startup": the model needed more than
  estimated, or something else grew. Try a smaller context or fp8 KV cache.
- **Out of memory** ("out of memory: Docker reports the container was OOM-killed", or "…the kernel OOM
  killer killed a process"): the kernel killed the engine. DGXMM adds 10 % to the next estimate for that
  configuration automatically, and the diagnosis is "Out of memory".
- **Exit code 137 without that** ("killed by SIGKILL"): something sent SIGKILL, such as `docker kill`, or
  a `docker stop` that timed out. It is not treated as out of memory and does not change estimates.

## A deployment stays `queued`

Only `max_concurrent_startups` (default 1) engines start at a time, and the deployment page says which
deployment it is waiting for. That start-up ends within `startup_timeout_s` (900 s) by becoming ready or
failing; then the next queued deployment starts. A queued deployment fails by itself after
`startup_queue_timeout_s` (30 minutes), or when it no longer fits at its turn. To give up earlier, press
**Stop** on it (that only leaves the queue). A wake-up that waits (*Waiting to wake up*) gives up after
`wake_queue_timeout_s` (300 s). After a DGXMM restart, queued deployments re-enter the queue.

## Gated model: 401/403 from Hugging Face

The catalog or a download reports a gated repo ("accept the licence on huggingface.co and add an HF
token"):

1. Sign in on huggingface.co and accept the model's licence.
2. Create a read token and give it to DGXMM:

   ```bash
   HF_TOKEN=hf_xxx ./install.sh     # if secrets/hf_token is empty
   # or replace an existing token:
   umask 077; printf '%s' 'hf_xxx' > secrets/hf_token; docker restart dgxmm
   ```

## LiteLLM status stays `pending`

`pending` means the last sync failed. DGXMM retries, backing off up to 10 minutes. The reason is on
the deployment's LiteLLM tab and in the Deployments table tooltip; the log has "LiteLLM sync failed"
(`./install.sh logs`). Start with **Test connection** on the LiteLLM page. Common causes:

- **LiteLLM unreachable** ("could not reach LiteLLM at …", "timed out", or "is not a LiteLLM
  endpoint"): the internal URL (default `http://litellm-proxy:4000`) must resolve from the DGXMM
  network. Fix it with `docker network connect <DGXMM_NETWORK> litellm-proxy`, or set an internal URL
  DGXMM can reach. "Not a LiteLLM endpoint" means something else answered on that URL.
- **Master key rejected** ("LiteLLM rejected the master key"): enter LiteLLM's master key on the
  LiteLLM page, or put it in `secrets/litellm_master_key`.
- **LiteLLM cannot store models**: set `store_model_in_db: true` in LiteLLM's `general_settings` (it
  needs its database).
- **LiteLLM cannot reach the engine** (registration works but requests fail): LiteLLM must be on the
  same Docker network, because engines are registered as `http://dgxmm-<name>:<port>/v1`.

## LiteLLM status is `disabled`, or the page says the master key needs re-entering

`disabled` means the integration is turned off on the LiteLLM page, or no master key applies. A master
key only applies to the URL it belongs to: a key entered on the page to the internal URL in effect
when it was saved, the `secrets/litellm_master_key` file to the default `litellm_url`. After you change
the internal URL, the page shows "the master key was entered for a different LiteLLM URL" (or "the
master key from the secret file is only used with the default LiteLLM URL"). Enter the master key for
the new URL, or set the URL back. This is deliberate: a stored key is never sent to a server it was not
entered for.

## Virtual keys: "LiteLLM is running without a database"

LiteLLM keeps virtual keys in its database. Configure `DATABASE_URL` (and `store_model_in_db: true`) for
your LiteLLM, or use the [bundled LiteLLM](../configuration/litellm.md#bundled-litellm), which has one. The
keys sections stay unavailable until LiteLLM reports a connected database.

## "The model name … is already used by deployment …" (`model_name_taken`)

Two deployments cannot share one LiteLLM model name, because LiteLLM would then load-balance between
two different models. Deploy, start, wake and rename are refused with the name of the deployment that
holds it. Give one of them a different **Model name (LiteLLM)**: the error offers a suggestion
(`<served name>-<deployment name>`), and on a deployment's page **Set a model name** opens its LiteLLM
tab. Stopped deployments without an alias do not hold their served name.

If a deployment shows the LiteLLM status `error`, two ready deployments ended up with the same name
anyway. The oldest keeps the registration; rename the other on its LiteLLM tab.

## Clients cannot call a served name any more after an upgrade

Older versions registered an aliased deployment under both its served name and its alias. Now each
deployment is registered once, under its model name, and the first sync after the upgrade removes the
extra served-name entries. Point those clients at the model name shown on the deployment's API access
card.

## Processes: masked or numeric values

- **`***` in commands** is intentional: passwords, tokens and keys are masked, and so are values DGXMM
  cannot parse reliably. Masking is best-effort: it cannot recognise every way a program may take a
  secret on its command line, so treat the Processes page as visible to every DGXMM user and do not pass
  secrets as command-line arguments. Use files or environment variables, which DGXMM never reads.
- **Numeric uids** for host processes mean the host's `/etc/passwd` is not mounted: the mount is in
  `docker-compose.yaml`, so run `./install.sh` to recreate the container. Processes in containers always
  show their in-container uid (`uid 1000 (in container)`), because container uids are not host
  accounts.
- **Containers shown by id only** appear for a few seconds after DGXMM starts, until the container list
  is first read.

## GPU memory of a deployment is not counted

A ready deployment's **Measured** memory is only a few GiB (its host memory) while `nvidia-smi
--query-compute-apps=pid,used_memory --format=csv` on the host shows its engine using much more, and the
log has "DGXMM runs in its own PID namespace". NVML only lists processes of the caller's PID namespace:
the DGXMM container needs `pid: host`, which `docker-compose.yaml` sets unless `.env` has `DGXMM_HOST_PID=0`.
Run `./install.sh` to recreate the container. With an older driver that reports no per-process GPU memory, this is expected; the budget then
credits the rest of the reservation (see [memory budget](../architecture/memory.md#budget)).

## Download fails with "owned by root"

An engine container created a folder in the HF cache as root. Run the `sudo chown -R <uid>:<gid> …`
command shown in the error (the uid:gid are `DGXMM_UID:DGXMM_GID` from `.env`), then retry.

## Catalog says "llmfit unavailable"

llmfit only adds scores. Search, fit badges and downloads keep working. Check it with
`docker exec dgxmm llmfit --version`.

## Decision model not healthy

Start with the chip in the top bar or `docker exec dgxmm python -m dgxmm.admin decider status`. While
it is not healthy every feature uses its rules, so nothing else is affected.

- **Enable answers `image_missing` or `image_outdated`**: the image `dgxmm-decider:3` is not built, or
  was built before the selected model was supported. Build it on the Decision model page or with
  `decider build` (`--force` to rebuild).
- **"Make room for the decision model"** (`insufficient_memory`): it needs 13 GiB (d3-mini) or 9 GiB
  (d3-nano). Pick one of the plans, stop something yourself, or switch to d3-nano.
- **Starting for several minutes**: the first start compiles GPU kernels (2–4 minutes for d3-mini). It
  becomes healthy only after its warm-up questions.
- **Failed with exit code 2** ("snapshot missing"): the model files are not in the HF cache. Run
  `docker exec dgxmm python -m dgxmm.admin decider download` (add `--model vllm-sr/d3-nano` for nano)
  or press **Download** on the Model card, then **Start**.
- **Failed with exit code 3** (diagnosis *Decision model files fail the integrity check*): a file of the
  pinned snapshot no longer matches its hash, for example because something wrote into the HF cache.
  The decider refuses to run it. `decider download --verify-only` lists the files that differ;
  `decider download` fetches them again and verifies them; then **Start**.
- **Failed with exit code 4** (*Decision model key not delivered*): the decider got no key file, or
  found a key in its environment (a `DECISION_API_KEY` override, which it refuses). Remove any such
  override, check the DGXMM log for why the key could not be written, then **Disable** and **Enable** so
  DGXMM starts it again with a fresh key file. **Rotate key** writes a new key if
  `data/secrets/decider_api_key` is unreadable.
- **Paused (rules only)**: the circuit breaker opened after three failed questions in a row; it retries
  after 60 seconds or as soon as a health probe succeeds.
- **Auto-restart "Gave up — …"**: 5 restart attempts in an hour failed or crashed again (the reason
  follows the dash, e.g. a missing image or a docker error). Read its Events tab and
  `docker logs dgxmm-decider`, fix the cause, then press **Retry** (or **Disable** and **Enable**).
- **Stopped by the watchdog** (Events: "watchdog: …"): memory ran low, or the decider used more than
  its reservation. The supervisor starts it again after its backoff once memory has recovered.

## Insights say a job is degraded

*"<Feature> is degraded"* means a background job failed three times in a row. DGXMM keeps retrying it
with a growing delay (up to an hour) and resolves the insight at the next success. Look at the
**Background jobs** card on the Decision model page (last result) and at the DGXMM log
(`./install.sh logs | grep "insight engine"`). A decision model that is down does not make jobs fail:
they then answer by rules. Switching the feature off stops the job.

## A schedule did not run

Check the schedule's line on the deployment's **Schedules** card (next run and last result) and the
deployment's Events tab:

- **All schedules are paused** (Decision model page, Schedules card) or the schedule itself is paused.
- **Deferred / "requests are running or waiting"**: a sleep or stop waits for 10 quiet minutes and a
  fresh metrics sample with nothing running; it is retried for 60 minutes, then skipped until the next
  day. "No metrics" / "no fresh metrics" means DGXMM could not tell, so it did not act.
- **"did not fit in free memory"** (a *Schedule could not …* insight): a wake-up or start never evicts
  anything. Free memory, or move the schedule.
- **Not started after a stop**: a wake-up only starts a deployment that a schedule stopped; one you
  stopped yourself, or one stopped before the 2026-10-18 upgrade, stays stopped.
- **Sleep schedules paused after a failure**: a failed scheduled sleep or wake-up pauses that deployment's
  sleep / wake schedules and raises an insight that offers the same times as stop / start.
- **Wrong hour**: schedules use their own time zone; daily jobs use the box's (`DGXMM_TZ`). After a
  missed slot (DGXMM down) a run is caught up only within 15 minutes.

## ⌘K says "Couldn't interpret that"

- "*Couldn't interpret that. Try …*": no interpretation reached 30 %. Name the action and the target,
  for example "sleep qwen3-8b-chat" or "open the insights page".
- "*… no page, deployment or model it names*": the target matched nothing you can see. Viewers only get
  pages.
- "*… nothing it names can be stopped right now*" (or slept, woken, …): the target exists but is not in
  a state for that action.
- A refusal note: removing deployments, deleting weights and user administration are not available as
  typed commands, by design.
- Without the decision model (or when it is busy) only the rules interpret, which understand fewer
  paraphrases.

## Compatibility check says "gated" or "unknown"

- **Gated** (*needs an HF token*): the repo is gated and the check had no access. Admin checks use the
  [HF token](../configuration/models-and-engines.md#hugging-face-token); accept the model's licence on huggingface.co with
  that account. Viewers' checks never use the token, so they always see gated repos as gated.
- **Unknown**: the architecture is not in the pinned vLLM image's list (vLLM may still serve it through
  its Transformers backend; only a start tells), the Hub did not answer within its time limits, or the
  repo is private. Unknown results are not cached; the next check asks again.

## `429 rate_limited` from the API

Some routes have per-user limits (see [API limits](../configuration/decision-model.md#api-limits)), for example 10 changes a
minute on the Decision model page or 30 parser suggestions a minute. The response has a `Retry-After`
header; wait that many seconds. Plan previews are never refused for this: over the limit they come back
without a model recommendation. Limits are per DGXMM process and reset when it restarts.

## Forgot the admin password

Reset it from the host. This works while DGXMM is running:

```bash
docker exec dgxmm python -m dgxmm.admin reset-password admin
```

It prints a new random password once, signs that user out everywhere, and requires a password change
at the next sign-in. The reset is written to the audit log. Use any username in place of `admin`. To see
which users exist:

```bash
docker exec dgxmm python -m dgxmm.admin list-users
```

`secrets/admin_password` is not updated, and only ever holds the first-run password. Another admin can
also reset a password from **Users → Reset password** in the UI.

## Two DGXMM instances

The installer refuses to start a second instance from another checkout, because both would adopt the
same engine containers. Manage the existing one from its own directory (the error names it), or
uninstall it first.

## Next steps

- [Known limitations](known-limitations.md)
- [Logs and monitoring](monitoring.md)
