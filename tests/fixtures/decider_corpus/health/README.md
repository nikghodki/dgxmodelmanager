# health/

Health-triage windows (`windows.jsonl`, feature 1). Each case is one check window as the health job sees it: the
5-second metric samples (as `dgxmm/metrics/store.py` computes them from the engine's `/metrics`; times relative to
the window), minute rows standing in for the 24 h baseline, the window's log lines that are not 2xx answers (plus the
count `ok_requests` of the 2xx ones; `eval_case` adds them back), the server errors of the previous window, and the
true label (a list when two labels describe the window equally well).

Provenance: a throwaway vLLM deployment of the public `Qwen/Qwen3-0.6B` (pinned DGXMM vLLM image, 1024-token
context, a 2048-token KV pool, 32 sequences) on a DGX Spark, driven through load phases of five minutes each (idle,
light chat, moderate, busy, KV pressure, a preemption storm, over-long prompts, malformed tool definitions, and
moderate / busy load while a GPU-burn container ran on the same GPU); each phase gives its full window and its two
halves. Client addresses are replaced by 192.0.2.10. Truth: the induced condition (under GPU contention:
`throughput_drop` when the per-request speed fell below 75 % of the baseline).

Synthetic (marked in `source`): recorded healthy windows with real-format vLLM server-error lines injected (vLLM
cannot be made to fail on demand), and recorded misuse windows with part of their 4xx lines dropped.
