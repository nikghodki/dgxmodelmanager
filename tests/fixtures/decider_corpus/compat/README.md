# compat corpus

Provenance / licence: excerpts (config keys, file names, model-card starts, code-scan summaries) of the PUBLIC Hugging Face repositories named in the cases, fetched anonymously from huggingface.co for interoperability testing only; each remains under its repository's own licence. No weights. Built with scripts/decider_eval/build_deploy_corpora.py; remove a repo here on request.

Labels: the pinned vLLM image's supported-architecture list in `../vllm_archs.json` (`ModelRegistry.get_supported_archs()` of the pinned image), auto_map -> runs_with_flags; architectures the image's transformers implements with attention-backend support (`../transformers_backend.json`, so vLLM's Transformers backend can serve them) accept runs / runs_with_flags / unknown; other unlisted architectures and GGUF-only repos -> unsupported; llama.cpp: GGUF files in the repo.

Size: chat templates are cut to their first 1500 characters and per-layer tables in `quantization_config` to their first 64 entries (the compat question only asks whether a template exists, and its state is capped at 1500 characters, so every question is unchanged); this keeps the file under the 1 MB cap of `tests/test_public_repo.py`.
