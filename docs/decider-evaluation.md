# Decision-model evaluation on real data

This page describes how DGXMM's decision-model features (failure diagnosis, parser suggestions, model kind,
workload-aware advice) were checked on real data from a DGX Spark (GB10), and how their default confidence
thresholds were chosen. The decision model is `vllm-sr/d3-mini` at revision `97dd054`. vLLM is the pinned image
`vllm/vllm-openai@sha256:22d3b121…` (vLLM 0.26.1rc1.dev1219). Measured 2026-10-16.

Every feature works the same way. Deterministic rules answer first. The model is asked only when no rule
answers, and its answer is used only at or above the feature's threshold. Below the threshold DGXMM behaves as it
did before the model existed. The numbers below come from four corpora: real failure logs, real model repos,
traffic on a real test deployment, and a hold-out repo set kept separate from rule tuning.

Latency, throughput, memory, behaviour under load and the d3-mini / d3-nano trade-off are in
[Performance and capacity](#performance-and-capacity) at the end.

Some numbers refer to the *initial study*: a feasibility study run before the decision model was built
into DGXMM, with a hand-labelled "golden set" of 70 questions (diagnosis, parser, plan, kind) plus 5
prompt-injection items and 24 parser items with option descriptions. The golden set is not part of the
repository. "First build" means the first version of these features, before the fixes below.

## Corpora

| corpus | what | where | size |
|---|---|---|---|
| failures | start-up failures induced on the DGX: each is one throwaway container from the pinned image, started with the command line DGXMM's `VllmEngine.container_spec` builds plus one fault. One more failure went through the live DGXMM API. The fixture is the log text the deployer stores (`dgxmm/deployer/logtail.py`), after its redaction (`logtail.stored_tail`: whole PEM blocks masked, condensed, then `redact_log`), plus the true label. | `tests/fixtures/real_failures/*.txt` | 16 logs, 12 of the 13 causes |
| repos | `config.json`, `tokenizer_config.json`, chat templates and the file list of real Hugging Face repos (never weights), plus the public repos already in the DGX's HF cache (file lists limited to files published on the Hub). Only public Hub repos are named; see the provenance note in `repos.txt`. | `tests/fixtures/decider_corpus/models/*.json`, list in `repos.txt` | 140 repos, of which 31 are hold-out |
| labels | The true kind and the accepted vLLM tool-call and reasoning parsers per repo, each with its source: the parser registries and implementations in the pinned image (`vllm/tool_parsers/__init__.py`, `vllm/reasoning/__init__.py`, `vllm/parser/*.py`), the vLLM docs (`features/tool_calling.md`, `reasoning_outputs.md`), model cards, and the repo's own chat template. | `tests/fixtures/decider_corpus/labels.json` | 95 tool, 94 reasoning, 140 kind |
| workload | DGXMM's own minute rollups of a throwaway test deployment: Qwen3-0.6B, 16k context, `max_num_seqs` 8, ≤ 10 GiB. It was driven by `scripts/decider_eval/traffic.py` for 18 minutes per profile, run one after another, and each window was summarised with DGXMM's `summarize`. | `tests/fixtures/workload/*.json` (raw rollups and phase log in `raw/`) | 24 windows of 10 min (4 per profile) + 4 production-like 60-min windows |

### Failures induced

The disk-full case used a tiny tmpfs over the compile caches and `/tmp`; no real disk was filled. The OOM case
used a 2 GiB cgroup memory limit, so the kernel OOM-killed only that container. Two causes were not produced:

- **Driver / CUDA mismatch:** this cannot be produced safely on the only GPU host.
- **GPU OOM from CUDA itself:** on unified memory this would put the user's workloads at risk. The cgroup kill
  stands in for it.

| fixture | true cause | how |
|---|---|---|
| unsupported-arch | unsupported_architecture | the NLI classifier on disk (`BartForSequenceClassification`) served as an LLM |
| trust-remote-code | trust_remote_code_required | `internlm/internlm2-chat-1_8b`, config files only, without `--trust-remote-code` (the code ran only in the container) |
| gated-no-token | auth_gated | `meta-llama/Llama-3.2-1B-Instruct`, online, no token |
| offline-missing | offline_missing_files | a repo that is not downloaded, with `HF_HUB_OFFLINE=1` |
| kv-too-small, kv-none-left, kv-small-budget | kv_cache_too_small | budgets of 1.9 to 4 GiB with contexts of 4k to 32k |
| ctx-above-model | context_too_long (new cause) | `max_model_len` 131072, above the model's 40960 |
| port-in-container | port_in_use | another process listening in the same network namespace |
| port-published | port_in_use | **through the live DGXMM API:** a host process held the port DGXMM allocated |
| broken-tokenizer | tokenizer_or_template | a truncated `tokenizer.json`. With no tokenizer files at all, vLLM still starts. |
| disk-full | disk_full | `ld: final link failed: No space left on device`, 487 lines above the end of the log |
| cgroup-oom | gpu_oom | killed with exit code 137 while loading, with no error in the log |
| bad-tool-parser, bad-reasoning-parser, auto-tool-no-parser | bad_parser_flag | an unknown parser name, or auto tool choice without a parser |

## Bugs the real data exposed (all fixed, each with a test)

1. **The 200-line log tail lost root causes.** The disk-full case printed its root cause about 480 lines above
   the end of the log, so DGXMM's tail never contained it. The deployer now reads up to 5,000 lines and stores the
   last 200 verbatim. Above them it adds the distinct error lines from the part that was cut, at most 40
   (`dgxmm/deployer/logtail.py`). vLLM repeats whole tracebacks at ERROR level; those copies are recognised as
   traceback frames, not error lines.
2. **Redaction hid the parser error.** vLLM reports an unknown parser as `KeyError: 'invalid tool call parser: …'`,
   and redaction treated `KeyError` as a secret key name, so the line became `KeyError: '***'` in the UI and in
   the model's input. Exception class names are no longer sensitive names; token-shaped values are still masked.
3. **No cause for a context longer than the model allows.** The new cause `context_too_long` comes with a rule,
   a patch (`max_model_len` set to the model's maximum) and a UI label.
4. **The reasoning-parser error was not matched.** The rule now also matches the newer message
   "`Reasoning parser 'x' not found`", and tokenizer-loading tracebacks match `tokenizer_or_template`.
5. **Port conflicts repeated on retry.** DGXMM runs in a container, so it cannot see host processes that hold a
   port. A port taken by one made every retry fail on the same port, while the fix hint said "start again". The
   deployer now moves to the next free port when docker reports `address already in use` / `port is already
   allocated`, up to 4 ports.
6. **GGUF variants were not matched to their base model.** `Qwen3-0.6B-Q8_0.gguf` kept its `-q8_0` suffix in
   `base_model_name`, so the try-llama.cpp advice and variant plans never fired for Q8_0 / Q4_0 / Q4_K_M /
   UD-Q4_K_XL files. The live deployment showed no advice for this reason.
7. **Decision models were labelled "vision-llm", Qwen3-Embedding "llm", EAGLE heads "llm".** A new kind,
   `decision`, and a strong-evidence rule tier in the scanner fix this (`dgxmm/inventory/kinds.py`). The evidence
   it uses:
   - decision-head configs (`decision_config.json` / `decider_config.json`) mark d3 and pplx-decider (and any
     other checkpoint that ships such a file);
   - sentence-transformers pooling marks embeddings;
   - `*_LogitScore` modules or a "rerank" name mark rerankers;
   - sequence classifiers are rerankers with 1 label and `decision` otherwise (NLI, sentiment, prompt-injection guards);
   - draft-head configs mark drafts;
   - `model_index.json` marks diffusion pipelines.

   These rules override the architecture-name rule, also for `*ForCausalLM` checkpoints that are not chat models.
   Consequence: a `*ForCausalLM` checkpoint with `decider_config.json` (a fine-tuned base with a readout) is no
   longer offered for a **new** vLLM chat deployment (the wizard lists `llm` only and the deploy API answers
   `engine_unsupported`). A deployment created before the reclassification keeps working: restart, reconfigure
   and rollback judge an existing deployment's own model with the architecture-only kind
   (`Deployer.prepare(existing_ref=…)`, `tests/test_deployer.py`).
8. **Parser rule gaps** (tuning set):
   - Qwen3-Next-Instruct was given `qwen3` from its name, although it has no reasoning (this was a wrong strong
     answer);
   - formats with no rule at all: LFM2, Granite 3, MiniCPM5, Phi-4-mini, Llama 4 "param1=", Cohere, Kimi
     K2.5 / Nemotron 3 / DeepSeek-V3.x thinking, and `<think>` + `enable_thinking` templates (Granite 4.2,
     SmolLM3, MiniCPM5);
   - name fallbacks for Mistral repos that have no HF chat template.
9. **Saturated batch traffic was not labelled.** 24 clients against `max_num_seqs` 8 averaged 7.96 running with a
   permanent queue, which missed the fixed "≥ 8 running" cut. Saturation is now judged against the engine's own
   sequence limit.

## Results

"Rules" means DGXMM without the model. "Model" means the decider asked directly on every item, even where a rule
answers, so it is measured on its own. "Pipeline" means rules first, then the model at or above its threshold.
Precision is the fraction of answers that are correct. Coverage is the fraction of items that get an answer.

### Rules: before (first build) → after

| feature | n | coverage before → after | precision before → after |
|---|---|---|---|
| diagnosis | 16 | 0.75 → 0.94 (the "before" figure already uses the new log tail; with the old 200-line tail disk-full is missed too, 0.69) | 1.00 → 1.00 |
| tool parser, tuning set | 69 | 0.91 → 1.00 (auto-apply 0.83 → 0.90) | 1.00 → 1.00 |
| tool parser, **hold-out** | 26 | 0.88 → 1.00 (auto-apply 0.85 → 0.96) | 1.00 → 1.00 |
| reasoning parser, tuning set | 69 | 0.83 → 1.00 (auto-apply 0.72 → 0.87) | 0.98 → 1.00 |
| reasoning parser, **hold-out** | 25 | 0.92 → 1.00 (auto-apply 0.84 → 0.92) | 1.00 → 1.00 |
| kind | 140 | 0.91 → 0.98 | 0.953 → 1.00 |

The hold-out repos were labelled from family documentation **before** the rules were run on them, after the rules
had been fixed on the tuning set, so they measure generalisation. On them the fixed rules were right every time
and answered all of them.

### Model on its own (d3-mini)

| feature | n | accuracy | precision / coverage at the default threshold | latency p50 (GPU shared with a 55 GB training job) |
|---|---|---|---|---|
| diagnosis | 16 | 1.00 (16/16) | 1.00 / 0.94 at 0.8 | 1.3 s |
| tool parser | 95 | 0.905 | 1.00 / 0.35 at 0.8 | 0.6 s (all parser and kind calls) |
| reasoning parser | 94 | 0.894 | 1.00 / 0.15 at 0.8 | |
| kind | 138 (2 repos have no config) | 0.964 | 0.99 / 0.92 at 0.8 | |
| workload (new state) | 24 | 0.958 (23/24) | 1.00 / 0.79 at 0.75 | 0.3 s |
| workload (old state) | 24 | 0.625 | 0.67 / 0.50 at 0.75 | |

Accuracy-vs-coverage curves (model answers with p ≥ t; precision / coverage):

| t | 0.3 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| tool parser | 0.92/0.93 | 0.95/0.67 | 0.95/0.61 | 0.96/0.48 | 0.95/0.43 | **1.00/0.35** | 1.00/0.28 | 1.00/0.11 |
| reasoning parser | 0.92/0.94 | 0.94/0.87 | 1.00/0.67 | 1.00/0.49 | 1.00/0.32 | **1.00/0.15** | 1.00/0.04 | 1.00/0.02 |
| kind | 0.96/1.00 | 0.97/0.99 | 0.98/0.96 | 0.98/0.94 | 0.98/0.93 | **0.99/0.92** | 1.00/0.87 | 1.00/0.83 |
| diagnosis | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | **1.00/0.94** | 1.00/0.88 | 1.00/0.88 |
| workload, new state | 0.96/1.00 | 1.00/0.79 | 1.00/0.79 | 1.00/0.79 | **1.00/0.79** | 1.00/0.79 | 1.00/0.63 | 1.00/0.58 |
| workload, old state | 0.63/1.00 | 0.60/0.83 | 0.73/0.63 | 0.73/0.63 | 0.67/0.50 | 1.00/0.33 | 1.00/0.21 | 1.00/0.17 |

The model's characteristic parser errors are all low-confidence:
- "has a Qwen-style template, so `qwen3`" for non-thinking Qwen models (0.50–0.58);
- `granite4` for Granite 4.2's XML format (0.77–0.78);
- `glm45` for GLM-4-0414 (0.20–0.33).

The model's confident kind error (d3-mini as vision-llm at 0.82) is now answered by the rules, so the model is
never asked about it.

## Thresholds

The defaults are in `dgxmm/decider/protocol.py`, with the reasoning in comments.

Targets: precision ≥ 95 % for answers that are applied automatically (parser auto-fill, kind), and ≥ 90 % for
answers that are only shown (diagnosis card, workload badge and advice order).

Two caveats apply to every number in this section:
- **The thresholds were chosen on the same data they are reported on.** There is no separate calibration split;
  the hold-out repos test the rules, not the thresholds.
- **The model is measured on every item, but in production it only sees what the rules leave unanswered.** With
  rule coverage at 1.00 on the parser corpus, the model answers no corpus item at all in the pipeline. The
  model-only curves estimate its behaviour on future models no rule recognises, which may be harder than this
  corpus.

- **parser 0.9 → 0.8.** Over both parser questions, the model was correct on all 47 answers at ≥ 0.8. That point estimate
  meets the 95 % target, but the Wilson 95 % lower bound is 0.92, which does **not** yet, so the threshold is to be
  re-checked against the feedback log. At 0.75 precision is 0.95, and at 0.9 only 12 % of model answers qualify.
  Answers between 0.5 and the threshold are still shown as "Suggested … Apply" and never auto-filled. Their
  precision is 0.95 (tool) and 0.94 (reasoning), which meets the 90 % display target. `SUGGEST_MIN` stays 0.5.
- **diagnosis 0.8 (unchanged).** Precision was 1.00 at every threshold, but the real set is small (16 logs). The
  initial study's golden set showed answers at ≥ 0.8 were 98 % accurate, so 0.8 stays. The Wilson lower bound of 15/15 is
  only 0.80, below the 90 % display target.
- **kind 0.8 (unchanged).** Precision is 0.99 at 0.8 (126/127, Wilson lower bound 0.96), which meets the 95 %
  target even at the lower bound, and coverage is 0.92.
- **workload 0.7 → 0.75.** With the new state every model answer at ≥ 0.4 on the 24 pure windows was right, and the correct non-mixed labels all scored ≥ 0.82; "mixed" was right but at 0.40–0.46, so it is not shown, which loses nothing because "mixed" does not reorder advice. Four 60-minute windows like the production feature's, each spanning 3–5 profiles (truth "mixed"), drew `batch_throughput` from the new state at 0.71, 0.60, 0.57 and 0.45. 0.75 keeps every correct pure label and rejects all four; at 0.7 one wrong label would show. 19/19 correct at ≥ 0.75 has a Wilson lower bound of 0.83, below the 90 % target: the windows are few and overlapping. The old state put those windows at 0.78–0.81, all wrong, and above its 0.7 threshold.
- **plan 0.6 (unchanged).** No real corpus exists beyond live checks on the DGX, so this is the value from the initial study.
  Plans are filtered for safety before the model ranks them.

## Workload: what changed and why

The first version sent the model a bare key/value dump and short option descriptions. On the real windows it
scored 0.625 on the 24 windows: it labelled RAG and mixed traffic as batch (0.50–0.78) and chat at only 0.32–0.42. The new state does three things:

- It writes the summary out in sentences.
- It adds estimates per request: request rate, prompt tokens and output tokens. These come from Little's law
  (requests in the system = rate × latency), because the engines export no per-request counters DGXMM keeps. On
  the chat phase the estimates were 1.8 requests per minute and 257 tokens out, against 1.83 and 263 counted by
  the client.
- It describes each option the way that profile actually looked in the metrics.

The new state scored 0.958. A last addition was the per-minute range and std/mean of the number of running requests; it fixed "mixed", which before it was labelled batch at 0.44–0.49.

### On the live app (first build) during the run

The monitor polled `/workload` and `/advice` every 5 minutes. Advice fired and was reordered live: `raise_seqs`
and `batched_tokens_small` were boosted to relevance 2.66 under the model's `batch_throughput` label (0.73).

From the batch phase onward, though, the live label stayed `batch_throughput` at 0.73–0.81. That included the RAG,
mixed and idle phases. The 1-hour window was dominated by the batch minutes, the old state is weak, and the 0.7
threshold let the label through. The live summary also reported "74 minutes" for a 1-hour window, which exposed a
second bug: each 5-second live sample was counted as a minute. Live samples are now averaged into minute rows.

`try_llamacpp` never fired live because of the GGUF suffix bug. Replaying the advisor with the fixed code on the
same real samples (`scripts/decider_eval/advice_replay.py`) produces it during the chat and idle phases. It is
demoted under batch / agent labels (relevance 0.59–0.63), while `batched_tokens_small` is boosted (2.67–2.73).

## Data and redaction

- Fixtures name only public Hugging Face repositories. File lists are limited to what the Hub publishes, so there
  are no local caches or results files in them. `repos.txt` carries the provenance and licence note for the
  config and chat-template excerpts.
- Failure logs are stored the way DGXMM now stores them: condensed, passed through `redact_text`, with host paths
  replaced. DGXMM now redacts every stored failure `log_tail`, including secrets that were in the cut-off part of
  the log. Event views also redact rows written by older versions when they are read. Diagnosis excerpts were
  already redacted.

## Limitations

- **Small sample sizes.** The failure corpus is 16 logs, and two causes could not be produced. The diagnosis model
  numbers are therefore indicative only, and the threshold was kept at the value from the initial study.
- **Parser labels are sets.** Several parsers parse the same format: `glm45` and `glm47` are one class, and
  `qwen3_coder` and `qwen3_xml` too. The Granite 4.1 `<tool_call>` JSON format is accepted by both `hermes` and
  `granite4`.
- **Ambiguous labels were left out.** The tool label of the R1 distills and the reasoning labels of Granite 3.3
  and LFM2.5 were excluded.
- **Tuning bias.** The rule fixes were made on the tuning set. The hold-out set (31 repos) is the honest measure,
  but it is small.
- **Workload windows are short and come from one deployment.**
  - Windows are 10 minutes, cut from 18-minute phases of one deployment, and the windows of a phase overlap,
    so they are not independent samples.
  - Production windows are 30–60 minutes and real hours are often mixed.
  - The test model is Qwen3-0.6B: in the agent profile it called tools in only 28 of 347 turns. The traffic shape
    is agent-like (long shared prefixes, short outputs back to back), but tool-call counts are not a metric DGXMM
    has.
- **Model labels on the shared GPU.** The model was evaluated on the live decider while a 55 GB training job
  shared the GPU, which affects latency, not answers.
- **No real-world accuracy over time yet.** Accuracy in daily use can only come from the feedback buttons and the
  `decisions` table. Thresholds may need retuning after weeks of use.

## Reproducing

All commands run on the DGX from the repository root. `induce_failures.py` starts throwaway vLLM containers on the
GPU: run it when nothing else needs the memory, and never on a box with deployments you care about.

```sh
.venv/bin/python scripts/decider_eval/induce_failures.py \
    --out tests/fixtures/real_failures --work ~/eval/work
python -I scripts/decider_eval/fetch_corpus.py hub tests/fixtures/decider_corpus/repos.txt ~/eval/corpus
python -I scripts/decider_eval/fetch_corpus.py local ~/.cache/huggingface/hub ~/eval/ondisk
python -I scripts/decider_eval/export_corpus.py ~/eval/corpus tests/fixtures/decider_corpus/models
python scripts/decider_eval/traffic.py --base http://127.0.0.1:<port>/v1 --model <served name> \
    --prompts ~/eval/prompts --docs ~/eval/docs --plan chat:18,agent:18,batch:18,rag:18,mixed:18,idle:18 \
    --log ~/eval/phases.jsonl
python scripts/decider_eval/workload_fixtures.py --db data/dgxmm.sqlite3 --deployment <id> \
    --phases ~/eval/phases.jsonl --out tests/fixtures/workload --window 10 --step 2
python scripts/decider_eval/evaluate.py --key-file data/secrets/decider_api_key --out ~/eval/results
python scripts/decider_eval/advice_replay.py --snapshot ~/eval/advice_snapshots.jsonl --hf ~/.cache/huggingface
```

`tests/test_real_corpus.py` runs the rules over all four corpora in CI. Rules must never be wrong on real data, and
answers that are auto-applied must be right 100 % of the time.

## Performance and capacity

### How it was measured

- `scripts/decider_loadtest.py` sends payloads built by DGXMM's own feature code: diagnosis states from real vLLM
  failure logs (`diagnosis.build_state`, a third of them padded to the production caps with other real logs, up to
  8.4 KB), parser states from the real chat templates in the HF cache (`config_suggest.state_for`, tool and reasoning
  questions with their option descriptions), kind states from real `config.json` files (`kind.build_state`), and plan
  and workload states (`plans.build_state`, `workload.build_state`). 80 payloads, state p50 1.4 KB.
  1. *per feature*: one client, every payload once;
  2. *capacity*: 1/2/4/8 clients sending back to back, straight to the server, mixed features;
  3. *bursts*: 16 simultaneous feature calls through `SystemOneClient` exactly as the app makes them (feature
     deadlines and priorities), once with the default back-pressure and once with it switched off (64 in flight,
     30 s queue), then one more request to see what the burst left behind on the server.
- The golden set of the initial study (70 items + 5 injection + 24 parser items with option descriptions; not in
  the repository) over HTTP, scored per feature.
- The live decider (d3-mini, deployment `decider`) was tested at low volume (≤ 200 requests per run). Throughout,
  another tenant's training job kept the GPU at 82–95 % utilisation and used ~55 GB, so every live number below is
  "GPU busy". The initial study measured the same model on an idle GPU (75 ms p50).

### d3-mini on a busy GPU (live)

Per feature, one client (`load_live2.json`):

| feature | n | state chars (p50) | p50 ms | p95 ms | max ms | app deadline |
|---|---|---|---|---|---|---|
| plan | 3 | 386 | 284 | 304 | 304 | 2 s |
| workload | 3 | 338 | 244 | 250 | 250 | 2 s (background) |
| kind | 16 | 852 | 333 | 527 | 558 | 2 s (background) |
| diagnosis | 26 | 1312 | 448 | 2140 | 2174 | 10 s (background) |
| parser (reasoning) | 16 | 2672 | 921 | 1035 | 1045 | 4 s (2 s when measured) |
| parser (tool) | 16 | 2672 | 1363 | 1448 | 1466 | 4 s (2 s when measured) |

Golden set over HTTP: p50 240 ms, p95 347 ms (short states); parser items with option descriptions p50 356 ms.

Capacity (mixed features, back to back, no client-side limit):

| clients | requests | p50 ms | p95 ms | p99 ms | throughput |
|---|---|---|---|---|---|
| 1 | 4 | 566 | 2119 | 2119 | 1.25 req/s |
| 2 | 8 | 1460 | 3479 | 3479 | 1.26 req/s |
| 4 | 16 | 2334 | 4819 | 4921 | 1.43 req/s |
| 8 | 32 | 4697 | 10309 | 10872 | 1.35 req/s |

d3_server answers **one request at a time** (a lock around the model): throughput is flat at ~1.3 req/s on a busy
GPU and latency grows linearly with the number of waiting clients. A request whose client gave up (timeout) still
runs to the end on the server.

Memory of the live decider process: 10.6 GB (NVML) before and after the load runs, unchanged under load;
MemAvailable did not move (34.1 → 34.6 GiB during the first run). The fixed 13 GiB reservation holds.

### Back-pressure (client side)

`SystemOneClient` keeps at most `decider_max_in_flight` (2) requests at the server. Others wait in DGXMM,
interactive questions (plan, parser) ahead of background ones (diagnosis, kind, workload), and are answered by rules
at once (`busy`, not an error, never counted against the circuit breaker) when

- the wait would exceed `decider_queue_timeout_s` (1 s) for interactive / half the deadline for background
  questions, or
- the caller's remaining deadline is shorter than the server's measured time per request (moving average).

Bursts of 16 simultaneous calls on the live decider (busy GPU):

| burst | back-pressure | answered | answered by rules (shed) | timed out | slowest call vs deadline | next request after the burst |
|---|---|---|---|---|---|---|
| mixed (all five features) | on (2 in flight) | 4 | 10 | 2 | within deadline | **569 ms** |
| mixed | off | 5 | 0 | 11 | within deadline | 6163 ms |
| interactive (parser, plan) | on | 2 | 12 | 2 | within deadline | **2488 ms** |
| interactive | off | 2 | 0 | 14 | within deadline | 12560 ms |

- No call exceeded its feature deadline in any run (worst: deadline + 4 ms). Request paths never block on the
  model beyond their timeout, with or without load.
- Without back-pressure, a burst leaves up to ~16 requests of work on the serial server: the next caller waited
  6–12.5 s. With it, the server never holds more than two requests, and shed calls fall back to rules within ≤ 1 s
  instead of waiting out their 2 s deadline.
- Under a busy GPU the two parser questions of one suggestion take 1.2 s + 0.8 s back to back on the serial server,
  so the second one regularly misses the 2 s deadline (it then shows no model suggestion; rules still apply).
  Sending both questions in one request was slower (2.5 s) and changed 2 of 16 answers, so it is not used.
  On an idle GPU both answer in ~0.2 s. The parser feature therefore has a 4 s deadline per question
  (`config_suggest.PARSER_TIMEOUT_S`; the wizard fetches suggestions in the background); every other feature keeps
  its deadline, and interactive questions still wait at most 1 s for a slot. The measurements above were taken with
  the earlier 2 s.

### Behaviour when the GPU is busy

- Latency rises ~3–10× (75 ms idle in the initial study → 240–450 ms typical, up to 2.2 s for long diagnosis states), with
  no errors (0 non-200 replies in all runs).
- Throughput stays ~1.3 req/s: the server is serial, so contention adds latency, not failures.
- Interactive features mostly fall back to rules during bursts (by design); background features (diagnosis 10 s)
  still get model answers.
- The circuit breaker did not open in any run: shed requests and timeouts after a long queue wait are not counted,
  only timeouts of requests that had at least half their deadline.

### d3-mini or d3-nano

`vllm-sr/d3-nano` (2.2B, revision `aeb70d4…`, same d3 runtime code as d3-mini: identical hashes for every `.py`)
is selectable on the Decision model page. Both were run on the golden set over HTTP; d3-nano from a throwaway
container of an image built before the container hardening (same model code; the release image is
`dgxmm-decider:3`), d3-mini on the live decider.

| | d3-mini (4.5B, default) | d3-nano (2.2B) |
|---|---|---|
| golden overall (70 items) | **0.900** | 0.814 |
| diagnosis (26) | **0.962** | 0.923 |
| parser, plain options (24) | 0.792 | **0.917** |
| parser, with option descriptions (24, what DGXMM sends) | 0.917 | **0.958** |
| plan (10) | **0.900** | 0.400 |
| kind (10) | **1.000** | 0.700 |
| injection (5, answer stays the true cause) | 5/5 | 5/5 |
| accuracy at confidence ≥ 0.8 (coverage) | 0.98 (69 %) | 0.91 (67 %) |
| golden p50 / p95 latency, GPU at 2–3 % | 102 / 158 ms | 55 / 72 ms¹ |
| golden p50 latency, GPU at 95 % | 240 ms | 87 ms |
| capacity (8 clients), GPU at 95 % | 1.35 req/s | 4.0 req/s |
| capacity (8 clients), GPU mostly idle | 3.1 req/s | 7.6 req/s¹ |
| memory | 12.6 GiB peak (initial study); NVML 10.6 GB | 8.2 GiB released at stop after load; NVML 6.0 GiB |
| reservation | 13 GiB | 9 GiB |
| start-up to ready (health + probe + warm-up) | 2–4 min | 82–90 s |

¹ first run; GPU utilisation not recorded, the other tenant's job had just ended.

**Decision: d3-mini stays the default.** d3-nano is clearly weaker on plan ranking (0.40) and on artifact kinds
(0.70; it calls drafts and speech models "llm") and less accurate overall, while it is better on parsers with option
descriptions. With DGXMM's thresholds the weaker answers would mostly be abstentions, but plan and kind are exactly
the features where a wrong confident answer costs most. Choose d3-nano when memory is tight (it reserves 4 GiB less)
or the GPU is shared and latency matters (2–3× faster, 3× the throughput).

### Lifecycle

- Start-up: the decider is marked ready only after `/health` = ready, an authenticated probe, and three warm-up
  questions shaped like the features' (a 60-line log with 12 described causes, a 40-line template with 16 parsers, a
  short plan question).
- The supervisor probes the ready decider every 30 s, closes an open breaker early on a prompt (< 1 s) probe,
  restarts it after a crash or 3 failed probes in a row (backoff 30 s doubling to 15 min, at most 5 restarts per
  hour, never while the watchdog reports memory pressure or the restart does not fit), and shows all of it in
  `/api/decider/status` and on the Decision model page.
- Engine start-ups are serialised (`max_concurrent_startups` = 1): a deployment started while the decider loads
  waits in state `queued` instead of breaking either start-up's memory profile.

Real checks on the DGX (throwaway deployments, worktree code, real Docker):

- **Start-up gate.** Two Qwen3-0.6B deployments requested at the same moment: with the gate the second was
  `queued` ("waiting for <the first deployment> to finish starting …"), started when the first became ready (242 s), and both
  served a completion. Without the gate (`max_concurrent_startups: 0`) both started together and both failed with
  vLLM's "Error in memory profiling. Initial free memory 41.3 GiB, current free memory 92.9 GiB" — in that run the
  memory change came mostly from another tenant's job ending, which the gate cannot prevent (it serialises DGXMM's
  own start-ups only).
- **Supervisor.** The worktree deployer deployed d3-nano as a decision deployment (ready after 90 s including the
  warm-up; first probe 32 ms; a real diagnosis question answered `port_in_use` 0.97 in 164 ms). `docker kill` →
  liveness marked it failed → the supervisor restarted it (audited `decider_auto_restart`), a second tick inside the
  backoff did nothing, it was ready again after 87 s.
- **Shared HF blob store.** huggingface_hub ≥ 2 (also DGXMM's own downloads) stores file contents in
  `hub/blobs/<xx>/<hash>` and a repo's blobs link into it; with only the repo mounted the decider refused to start
  ("tokenizer.json: FAILED open or read"). The decider gets exactly the store files its pinned snapshot links to,
  each bound read-only at its own path (d3-nano: 3 files, 4.1 GiB; d3-mini: none), never the store directory, its
  `.refs` or other repos' files; the snapshot's hash check passes with these mounts.
