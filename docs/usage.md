# Using DGXMM

This guide follows the UI from signing in to calling your models. It assumes DGXMM is installed (see the
[README](../README.md#quick-start)).

- [Sign in and roles](#sign-in-and-roles)
- [Dashboard](#dashboard)
- [Deploy a model](#deploy-a-model)
- [Deployment detail](#deployment-detail)
- [Models on disk](#models-on-disk)
- [Catalog and downloads](#catalog-and-downloads)
- [Planner](#planner)
- [Users and audit log](#users-and-audit-log)
- [Command palette and keyboard](#command-palette-and-keyboard)
- [Theme and mobile](#theme-and-mobile)
- [Calling your models](#calling-your-models)

## Sign in and roles

Open `http://<host>:8090/dgxmm/` (or your proxy URL) and sign in. On first install the user is
`admin`, and the installer printed its password. It is also saved in `secrets/admin_password`. You must
pick a new password (at least 12 characters) before you can do anything else. The same applies to users
an admin creates or resets.

| Role | Can |
|---|---|
| **viewer** | See the dashboard, deployments, models, catalog, downloads and planner, and run memory checks. |
| **admin** | Everything: deploy, start, stop, sleep, switch, reconfigure, remove, download, benchmark, view container logs, manage users, read the audit log. |

After 5 failed sign-ins within a minute (per IP or per username), further attempts are refused for a
minute. Sessions last 12 hours from the last activity. Forgot the admin password? See
[operations](operations.md#forgot-the-admin-password).

## Dashboard

![Dashboard in the dark theme](images/dashboard-dark.png)

- **Free for new deployments** is the headline number: memory a new model may use right now, after the
  safety margin and OS reserve, what running deployments have reserved, and what other processes use
  (see [memory model](architecture.md#memory-model)). The same figure is in the top bar on every page.
- **Unified memory** bar: one segment per DGXMM deployment, external processes, safety margin + OS
  reserve, and free memory. It also shows GPU utilisation, temperature and free disk. **Table** shows the same numbers as
  a table.
- **KPI tiles**: total generation throughput, the worst time to first token (p50) among ready
  deployments, the highest KV-cache usage, and the request queue.
- **Deployments** table with state, memory (measured / reserved), tok/s, KV usage and LiteLLM status.
  Row actions are in the `⋯` menu.
- **Memory over the last hour** chart, and the **largest external memory users**: processes DGXMM does
  not manage. DGXMM never touches them.
- **Containers not managed by DGXMM** (admins): vLLM containers started by hand. **Adopt settings**
  copies one's model and flags into a new, stopped DGXMM deployment. The original container is left
  alone: stop it before you start the adopted deployment.

A banner appears when memory runs low: *warning* below 1.5 × the safety margin, *critical* below it.

## Deploy a model

Open **Deploy model** (Dashboard, Deployments, or the command palette). It needs the admin role.

![Deploy wizard with memory check and plans](images/deploy-dark.png)

1. **Model**: pick a complete, downloaded model. Only models an engine can serve are listed. The line
   below shows the architecture, quantization, weight size and maximum context.

   ![Choosing a model](images/deploy-picker-dark.png)

2. **Engine and name**: vLLM for safetensors, llama.cpp for GGUF. The **deployment name** becomes the
   container name `dgxmm-<name>` (lowercase letters, digits and `-`). **Priority** (0–100) decides
   eviction order: lower priority is suggested for sleeping or stopping first. **LiteLLM alias** is an
   optional extra model name in LiteLLM, such as `default-chat`.
3. **Sizing and features**: context length, concurrent sequences, KV cache dtype, the shared KV pool
   size, prefix caching, sleep mode and so on. Advanced options include tool-call and reasoning parsers,
   the served model name, and allowlisted extra flags.

The **Memory check** panel updates as you type. It shows the estimate split into weights, KV cache,
activations and runtime, the KV pool in tokens, the vLLM memory share it will be started with, and a
confidence label (`low`, `estimated`, `calibrated` from earlier runs of this model, or `measured` for
this exact configuration).

**If it does not fit**, the panel says by how much and lists **Ways to make it fit**:

- a smaller KV cache dtype (fp8 for vLLM, q8_0 for llama.cpp);
- a shorter context or fewer concurrent sequences;
- a smaller quantization of the same model that is already on disk;
- putting lower-priority deployments to sleep, or stopping them.

Each option shows what it would need and whether that is enough ("fits" or "still short"). Choosing a
config change edits the form. Choosing sleep or stop arms those actions, and the button becomes
**Apply plan and deploy**. Nothing is stopped until you press it. If memory changed in the meantime,
DGXMM refuses, undoes any actions and shows fresh plans.

After **Deploy** you land on the deployment's page. It moves through *Starting* → *Loading* → *Ready*.
The first vLLM start of a model can take several minutes, because it compiles kernels (cached for later
starts) and may pull the engine image.

## Deployment detail

Header actions depend on the state (admins only):

| Action | When | What happens |
|---|---|---|
| **Sleep** | ready, vLLM with sleep mode | Frees the KV cache. Weights stay loaded for a fast wake-up. LiteLLM deregisters it while asleep. |
| **Wake** | sleeping | Takes the memory back (only if it is free). |
| **Start** | stopped or failed | Starts again with the stored configuration, after a fresh memory check. |
| **Reconfigure** | ready, sleeping, stopped, failed | Change the configuration with a live memory check. Applying it restarts the deployment. If the new configuration does not come up, the old one is restored. |
| **Logs** | any | The last lines of the engine container log. |
| **Switch model…** (`⋯` menu) | ready or sleeping | Replace this deployment with another model. DGXMM sleeps or stops this one as needed, moves the LiteLLM alias, and rolls back if the new model does not become ready. |
| **Stop** (`⋯`) | running states | Stops and removes the engine container. The deployment and its settings are kept. |
| **Remove** (`⋯`) | stopped or failed | Deletes the deployment record. |

Stop and Remove ask for confirmation.

Tabs:

- **Overview**: endpoints (served model name; the URL on the Docker network; the URL on the host;
  LiteLLM status and alias, with copy buttons), memory (reserved, measured, estimate parts and
  confidence), and the full configuration.
- **Performance**: throughput, time to first token, KV-cache usage, queue, inter-token latency and
  speculative-decoding acceptance. View it live or over 1 h, 24 h, 7 d or 30 d.
- **Advice**: suggestions from the measured metrics and benchmarks. Examples: an fp8 KV cache, an
  NVFP4 variant that is already on disk, speculative decoding with a local draft model, more concurrent
  sequences, a smaller KV pool, chunked prefill, and llama.cpp offload, flash-attention and KV-cache
  settings. **Apply** opens Reconfigure with the change filled in.
- **Benchmarks**: **Run benchmark** load-tests the ready deployment at concurrency 1, 4, 8 and 16
  (1024-token prompts, 256-token outputs, at most 60 s and 32 requests per level). Results (total and
  per-request tok/s, TTFT p50/p95, ITL p50) appear as each level finishes and are kept per
  configuration, so you can compare settings. Only one benchmark runs at a time.
- **Events**: every state change with its message. Failures include the container's last log lines.

## Models on disk

**On disk** lists the models DGXMM found in the Hugging Face cache and the GGUF directories: format,
architecture, quantization, size, which engines can serve them, and whether the download is complete.
Filter by kind. **Deploy** opens the wizard with the model selected. **Rescan** (admins) rescans right
away; otherwise DGXMM rescans every 5 minutes. To add directories, see
[configuration](configuration.md#model-directories).

## Catalog and downloads

![Catalog with fit badges](images/catalog-light.png)

**Catalog** searches Hugging Face. Turn on **GGUF (llama.cpp)** to search for GGUF repos, and set the
**Context** the fit badges should assume. Each result shows downloads, approximate size, a **fit**
badge based on memory free right now (*Fits now*, *Fits if you evict* running deployments, *Too big*),
whether vLLM supports the architecture (✓, ✗, or ? when unknown) or llama.cpp applies, and llmfit's
assessment (a ⚠ means llmfit's memory figure differs from DGXMM's by over 20 %). Badges show when a
model is already on disk, gated, or a known quantization.

Open a result for details: the DGXMM memory estimate at your context length, then **Download
safetensors** or a single file from the list of **GGUF files**. Other quantizations of the same model are
listed too. Downloads are refused if they would leave less than `min_free_disk_gib` (50 GiB) free.
Gated models need an [HF token](configuration.md#hugging-face-token).

**Downloads** shows progress, and lets admins cancel. A download interrupted by a DGXMM restart is
marked failed. Start it again: files that already finished are kept.

## Planner

**Planner** answers "what can I run together?". For a context length per model, it lists the downloaded
models with the memory their default configuration needs. It then lists the best combinations that fit
the capacity: usable memory minus what external processes use, as if no deployment were running.
Combinations with more models rank first, then those with more favourites, then those with more
headroom. Tick **Must include** to only see combinations that contain a model. Admins can star
**favourites**.

## Users and audit log

**Users** (admins): add users with a role and a temporary password, change roles, reset passwords
(users must change the temporary password at next sign-in, and their sessions end), and delete users.
You cannot delete yourself or remove the last admin.

**Audit log** (admins): who did what and when. It covers sign-ins (including failed ones), deploys,
stops, switches, reconfigures, downloads, benchmarks, user changes and watchdog actions. Filter by action.

## Command palette and keyboard

Press <kbd>⌘</kbd> <kbd>K</kbd> (macOS) or <kbd>Ctrl</kbd> <kbd>K</kbd> (elsewhere), or click
**Search or run…** in the top bar. Type to filter, use the arrow keys, and press <kbd>Enter</kbd> to run:

- go to any page;
- **Deploy a model**, or **Deploy &lt;model&gt;** for each deployable model on disk (admins);
- **Open**, **Sleep**, **Wake** or **Stop** a deployment (Stop asks for confirmation);
- switch the theme (light, dark, system), change your password, or log out.

## Theme and mobile

The sun/moon button in the top bar switches between light and dark. The default follows your system
setting, and the choice is remembered per browser. The sidebar can be collapsed with the button at its
bottom.

On a phone the layout adapts: tables become cards, dialogs become bottom sheets, and a tab bar gives
quick access to Home, Deploys, Models and More.

<img src="images/dashboard-mobile.png" alt="Dashboard on a phone" width="300">

## Calling your models

Every deployment serves the OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`) under its **served model name** (shown on the Overview tab). The engines themselves have
no authentication, which is why their ports are only published on the host's loopback interface.

**From the DGX itself**, use the "On the DGX" endpoint:

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**From another container on the DGXMM Docker network**, use `http://dgxmm-<deployment name>:<port>/v1`.

**From other machines, through LiteLLM** (recommended): every ready deployment is registered under its
served name and its alias. LiteLLM handles keys, rate limits and logging:

```bash
curl http://<litellm-host>:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "default-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://<litellm-host>:4000/v1", api_key="<litellm key>")
reply = client.chat.completions.create(model="default-chat",
                                       messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

Aliases make model changes invisible to clients: deploy with alias `default-chat`, and use **Switch
model…** later. The alias moves to the new deployment once it is ready.
