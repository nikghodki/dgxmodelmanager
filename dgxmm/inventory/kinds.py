"""Deterministic model-kind rules for Hugging Face snapshots (config.json + file list; no weights read).

Tier 1 (`classify_kind`): the architecture name alone -> llm / draft / other.
Tier 2 (`strong_kind`): high-precision evidence that overrides tier 1, also for *ForCausalLM checkpoints that are
not chat models: decision-model configs, sentence-transformers pooling, reranker heads, draft-head configs,
sequence classifiers, speech configs, diffusion pipelines. Each rule was checked on the real corpus in
tests/fixtures/decider_corpus (docs/decider-evaluation.md): it never contradicted a label there.
What neither tier decides stays `other` and may be classified by the decision model (dgxmm/decider/kind.py).
"""
from __future__ import annotations

import re

_LLM_SUFFIX = re.compile(r"(CausalLM|ConditionalGeneration|LMHeadModel)$")
_NON_LLM_PREFIX = ("Whisper", "Bart", "MBart", "T5", "MT5", "Pegasus", "Marian", "Speech", "Wav2Vec",
                   "Seamless", "Blip", "Clip", "Siglip", "Encodec")
_DRAFT_HINTS = ("Draft", "Eagle", "DFlash", "Medusa", "Speculator")
_SPEECH_ARCH = re.compile(r"^(Whisper|Wav2Vec|Hubert|Speech|Seamless|Encodec|Kokoro|StyleTTS|Bark|Vits|Parler)", re.I)
_DRAFT_NAME = re.compile(r"eagle|medusa|dflash|speculator|hydra", re.I)
# Decision models ship their decision head / readout configuration next to config.json:
# vllm-sr d3 and perplexity pplx-decider ship decision_config.json; other deciders use decider_config.json.
# Such checkpoints are often plain *ForCausalLM (a fine-tuned base with a readout): they become `decision`, so
# they are no longer offered for NEW vLLM chat deployments; existing deployments of them keep restarting
# (Deployer.prepare judges existing deployments with the architecture-only kind).
DECISION_FILES = frozenset({"decision_config.json", "decider_config.json"})


def classify_kind(arch: str | None) -> str:
    if not arch:
        return "other"
    if any(h in arch for h in _DRAFT_HINTS):
        return "draft"
    if _LLM_SUFFIX.search(arch) and not arch.startswith(_NON_LLM_PREFIX):
        return "llm"
    return "other"


def num_labels(cfg: dict) -> int | None:
    if isinstance(cfg.get("num_labels"), int):
        return cfg["num_labels"]
    if isinstance(cfg.get("id2label"), dict):
        return len(cfg["id2label"])
    return None


def _repo_name(ref: str | None) -> str:
    return (ref or "").rstrip("/").split("/")[-1].lower()


def strong_kind(ref: str | None, arch: str | None, config: dict | None, files: list[str]) -> tuple[str, str] | None:
    """(kind, reason) when the snapshot carries unambiguous evidence; None = no strong evidence."""
    cfg = config or {}
    arch = arch or ""
    names = {f.split("/")[-1] for f in files}
    tops = {f.split("/")[0] for f in files if "/" in f}
    name = _repo_name(ref)
    if names & DECISION_FILES:
        return "decision", "decision-model head configuration (" + ", ".join(sorted(names & DECISION_FILES)) + ")"
    if "model_index.json" in names and "config.json" not in files:
        return "other", "diffusers pipeline (model_index.json)"
    if any(t.endswith("_LogitScore") or t.endswith("_CrossEncoder") for t in tops) or "rerank" in name:
        return "reranker", "reranker (score head / name)"
    if arch.endswith("ForSequenceClassification"):
        n = num_labels(cfg)
        if n == 1:
            return "reranker", "sequence classifier with a single relevance score"
        return "decision", f"sequence classifier with {n or 2} labels"
    if any(t.endswith("_Pooling") for t in tops):
        return "embedding", "sentence-transformers pooling config"
    if "sentence_bert_config.json" in names and not _LLM_SUFFIX.search(arch):
        return "embedding", "sentence-transformers model"
    layers = cfg.get("num_hidden_layers")
    if "medusa_num_heads" in cfg or "draft_vocab_size" in cfg or "dflash_config" in cfg or (
            _DRAFT_NAME.search(name) and isinstance(layers, int) and layers <= 2):
        return "draft", "speculative-decoding draft head (config)"
    if _SPEECH_ARCH.search(arch) or "num_mel_bins" in cfg or "istftnet" in cfg:
        return "speech", "speech model architecture / audio config"
    return None


def artifact_kind(arch: str | None, config: dict | None, files: list[str], ref: str | None = None) -> str:
    """The kind the inventory reports: strong evidence first, else the architecture-name rule."""
    hit = strong_kind(ref, arch, config, files)
    return hit[0] if hit else classify_kind(arch)
