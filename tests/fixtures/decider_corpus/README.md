# decider_corpus/

Provenance and licence are in [repos.txt](repos.txt): excerpts (config keys, file names, chat templates) of the
public Hugging Face repositories listed there, for interoperability testing only, each under its repository's own
licence. `labels.json` holds this project's expected answers.

Corpora of the advanced features live in sub-directories named after the feature (`<feature>/`, cases as
`*.jsonl` / `*.json` with `name` and `label`), each with its own provenance note; `tools/eval_feature.py <feature>
default` evaluates them.

`vllm_archs.json` is the supported-architecture list of the pinned vLLM image (`ModelRegistry.get_supported_archs()`,
read by DGXMM's `VllmArchRegistry`); the compat corpus labels and evaluation use it.

`transformers_backend.json` records, for the architectures the pinned vLLM image does not list natively, whether the
image's `transformers` implements them with attention-backend support (so vLLM's Transformers backend can serve
them); read with `getattr(transformers, arch)._supports_attention_backend` in the pinned image. The compat labels use
it.
