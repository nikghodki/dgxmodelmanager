# reading/

Reading-label cases (`readings.jsonl`, feature 8): the facts DGXMM's reading rules gather for one benchmark level or
startup memory measurement (`dgxmm/decider/reading.py` `facts_benchmark` / `facts_calibration`), and the true
label.

- `history-*`: the benchmark and engine-log calibration history of a DGX Spark (public `Qwen/Qwen3-0.6B` with vLLM
  and `unsloth/Qwen3-0.6B-GGUF` with llama.cpp), with facts computed by DGXMM's own code over a copy of that box's
  database (host contention from its host rollups and deployment events). Truth: levels with no successful request
  are `broken`; benchmark 15 (a 4 s time to first token at concurrency 1 on the first run of a fresh deployment) is
  `cold_cache`; the rest `plausible`.
- `induced-*`: DGXMM's `BenchmarkRunner` against a throwaway vLLM deployment of `Qwen/Qwen3-0.6B`, run after a
  restart (cold), quietly, while a second throwaway vLLM started on the box, and while a GPU-burn container ran.
  Contention facts come from the host's used memory over each level's window and the second deployment's state
  changes. Truth: `contended` (or `cold_cache` after a restart) when the level was more than 10 % slower than the
  quiet median at its concurrency, else `plausible`.
