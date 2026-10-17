"""Feature 4 — artifact understanding: what kind of model is an artifact the inventory rules call `other`?

Only artifacts that the inventory rules (dgxmm/inventory/kinds.py) leave as `other` (or with no known architecture)
are considered. The scanner's strong-evidence tier already recognises clear cases (decision-model heads,
sentence-transformers embeddings, rerankers, sequence classifiers, draft heads, speech models); the rest go to the
decision model in a background task after the inventory scan (never in a request).
Results are cached in `artifact_kinds` by (ref, mtime); `kind` is overridden only when the model's probability is
at least threshold("kind"). Artifacts expose `kind_source` ("rules" | "model") and `kind_confidence`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from dgxmm.db import ArtifactKindRow
from dgxmm.decider.common import finite, quoted_block, safe_decide, safe_record_rules, threshold_of
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option
from dgxmm.inventory.kinds import strong_kind

log = logging.getLogger(__name__)
KINDS: list[tuple[str, str]] = [
    ("llm", "A text-generation language model"),
    ("draft", "A speculative-decoding draft model (EAGLE, MTP, DFlash, ...)"),
    ("embedding", "A text embedding model"),
    ("reranker", "A cross-encoder / reranker that scores query-document pairs"),
    ("vision-llm", "A language model that also takes images (or video)"),
    ("speech", "Speech recognition or speech synthesis"),
    ("decision", "A decision / classification model: decider, router, guard or intent classifier, NLI or sentiment "
                 "classifier (scores given options instead of writing free text)"),
    ("other", "Something else (diffusion / image pipeline, time-series model, OCR pipeline, ...)"),
]
OPTIONS = [Option(k, k, d) for k, d in KINDS]
QUESTION = "What kind of artifact is this model repository, judging from its config?"
CONFIG_KEYS = ("architectures", "model_type", "num_labels", "id2label", "num_hidden_layers", "hidden_size",
               "vocab_size", "draft_vocab_size", "vision_config", "audio_config", "text_config", "image_token_id",
               "num_mel_bins", "encoder_layers", "decoder_layers", "is_encoder_decoder", "pooling", "task",
               "task_specific_params", "istftnet", "plbert", "dflash_config", "num_nextn_predict_layers")
INTERVAL_S = 60.0


def needs_kind(artifact) -> bool:
    return artifact.kind == "other" or not artifact.architecture


def rule_kind(ref: str, architecture: str | None, config: dict | None, files: list[str]) -> tuple[str, str] | None:
    """(kind, reason) for clear cases the base inventory rules leave as `other`; None = uncertain. The same
    high-precision evidence the scanner uses (dgxmm/inventory/kinds.py), applied again to cached / older rows."""
    return strong_kind(ref, architecture, config, files)


def _excerpt(config: dict | None) -> str:
    cfg = config or {}
    small = {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
    if not small:  # unknown layout: the top-level keys say a lot
        small = {"keys": sorted(cfg)[:40]}
    return json.dumps(small, default=str)[:1500]


def build_state(artifact) -> str:
    files = list(artifact.files or [])
    summary = ", ".join(sorted(f for f in files if not f.endswith((".safetensors", ".bin", ".gguf"))))[:800]
    weights = sum(1 for f in files if f.endswith((".safetensors", ".bin", ".gguf")))
    return "\n".join([
        f"Source: {artifact.source}; format: {artifact.format}; weight files: {weights}",
        quoted_block("REPOSITORY", artifact.ref, max_chars=300),
        quoted_block("CONFIG.JSON (excerpt)", _excerpt(artifact.config), max_chars=1500, keep="head"),
        quoted_block("OTHER FILES", summary or "(none)", max_chars=800, keep="head"),
    ])


def artifact_mtime(artifact) -> float:
    for p in (os.path.join(artifact.local_path, "config.json"), artifact.local_path):
        try:
            return os.stat(p).st_mtime
        except (OSError, TypeError):
            continue
    return 0.0


def apply(artifact, kind: str | None, source: str, confidence: float | None, threshold: float,
          llm_ok: bool = True) -> None:
    """Override the artifact's kind from a cached result: rules always; a model answer only at/above the CURRENT
    threshold (rows keep the model's probability, so a changed threshold is honoured on the next scan), and never
    to `llm` unless an engine supports the architecture (`llm_ok`)."""
    if kind is None:
        return
    if source == "model" and (confidence is None or confidence < threshold or (kind == "llm" and not llm_ok)):
        return
    if kind == artifact.kind:
        if source == "model":
            artifact.kind_source, artifact.kind_confidence = "model", confidence
        return
    if source == "rules":
        artifact.kind, artifact.kind_source, artifact.kind_confidence = kind, "rules", None
    elif source == "model":
        artifact.kind, artifact.kind_source, artifact.kind_confidence = kind, "model", confidence


def architecture_of(artifact) -> str | None:
    if artifact.architecture:
        return artifact.architecture
    archs = (artifact.config or {}).get("architectures")
    return archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else None


class KindService:
    def __init__(self, ctx, interval_s: float = INTERVAL_S):
        self.ctx, self.interval_s = ctx, interval_s

    @property
    def decider(self):
        return decider_of(self.ctx)

    def _threshold(self) -> float:
        return threshold_of(self.decider, "kind", 1.01)  # unknown threshold: no model override

    def llm_ok(self, artifact) -> bool:
        """A model may only call something an llm if vLLM's supported-architecture list contains it."""
        engine = (self.ctx.engines or {}).get("vllm")
        source = getattr(engine, "arch_source", None)
        try:
            supported = source() if source else None
        except Exception:
            supported = None
        arch = architecture_of(artifact)
        return artifact.source == "hf" and bool(arch) and supported is not None and arch in supported

    def _rows(self) -> dict:
        with self.ctx.db.session() as s:
            return {r.ref: (r.mtime, r.kind, r.source, finite(r.confidence)) for r in s.query(ArtifactKindRow)}

    def apply_cached(self, artifacts: list) -> None:
        """Inventory post-scan hook (sync, runs inside refresh): apply cached kinds before items are published."""
        try:
            rows = self._rows()
        except Exception:
            log.warning("could not read artifact kind cache", exc_info=True)
            return
        thr = self._threshold()
        for a in artifacts:
            r = rows.get(a.ref)
            if r is not None and needs_kind(a) and abs(r[0] - artifact_mtime(a)) < 1e-6:
                apply(a, r[1], r[2], r[3], thr, self.llm_ok(a))

    def _store(self, ref: str, mtime: float, kind: str | None, source: str, conf: float | None, did: int | None):
        with self.ctx.db.session() as s:
            row = s.get(ArtifactKindRow, ref)
            if row is None:
                s.add(ArtifactKindRow(ref=ref, mtime=mtime, kind=kind, source=source, confidence=conf, decision_id=did))
            else:
                row.mtime, row.kind, row.source, row.confidence, row.decision_id = mtime, kind, source, conf, did

    async def classify_pending(self) -> int:
        """Classify every uncached `other` artifact; returns how many were (re)classified. Never raises."""
        inv = self.ctx.inventory
        if inv is None:
            return 0
        try:
            cache = {ref: r[0] for ref, r in (await asyncio.to_thread(self._rows)).items()}
        except Exception:
            log.warning("could not read artifact kind cache", exc_info=True)
            return 0
        done = 0
        for a in inv.list():
            if not needs_kind(a) or a.kind_source == "model":
                continue
            mtime = await asyncio.to_thread(artifact_mtime, a)
            if a.ref in cache and abs(cache[a.ref] - mtime) < 1e-6:
                continue
            try:
                hit = rule_kind(a.ref, a.architecture, a.config, list(a.files or []))
                subject = f"artifact:{a.ref}"[:512]
                if hit is not None:
                    d = safe_record_rules(self.decider, feature="kind", subject=subject, choice=hit[0], reason=hit[1])
                    await asyncio.to_thread(self._store, a.ref, mtime, hit[0], "rules", None, d.decision_id)
                    apply(a, hit[0], "rules", None, self._threshold())
                    done += 1
                    continue
                if a.config is None:
                    continue
                d = await safe_decide(self.decider, feature="kind", subject=subject, state=build_state(a),
                                      question=QUESTION, options=OPTIONS, timeout_s=2.0)
                if not d.probabilities and d.source != "model":
                    continue  # unavailable / timed out: not an answer, ask again next round
                top = d.choice or max(d.probabilities, key=d.probabilities.get)
                conf = d.confidence if d.confidence is not None else d.probabilities.get(top)
                # store the model's answer with its probability; whether it applies is decided at apply time
                await asyncio.to_thread(self._store, a.ref, mtime, top, "model", conf, d.decision_id)
                apply(a, top, "model", conf, self._threshold(), self.llm_ok(a))
                done += 1
            except Exception:
                log.warning("kind classification of %s failed", a.ref, exc_info=True)
        if done:
            self.ctx.events.publish("inventory", {"reason": "kind"})
        return done

    async def run(self) -> None:
        while True:
            await self.classify_pending()
            await asyncio.sleep(self.interval_s)
