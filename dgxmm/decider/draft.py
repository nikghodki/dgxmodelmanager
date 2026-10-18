"""Advanced feature 4 — draft-model pairing for speculative decoding: which on-disk model drafts for a target?

Rules decide which downloaded models are compatible (``candidates``):
- ``draft_model`` (a small LLM drafts tokens): kind ``llm`` / ``draft``, safetensors, complete, smaller than the
  target, not a draft head, the same vocabulary size, and the SAME tokenizer — an identical tokenizer file (sha256
  of ``tokenizer.json``, else ``tokenizer.model`` / ``vocab.json``), or, when a tokenizer file is missing on either
  side, the same model family (``model_type``). Known, different tokenizer files never qualify. Token ids must mean
  the same thing in both models (equal vocabulary sizes are required until the pinned vLLM is shown to accept a
  padded vocabulary of another size), so nothing else qualifies.
- ``eagle`` / ``eagle3`` / ``dflash`` (a head trained on the target's hidden states): a head of that method
  (architecture name) with the target's hidden size and vocabulary size, made for this target (its config names the
  target, its repo name carries the target's base name, or — DFlash — its ``num_target_layers`` equals the target's
  layer count).
- ``mtp`` / ``ngram`` need no draft model.
One compatible candidate is the rules' suggestion; with several the decision model ranks them (names only in the
quoted state, options are the repo ids). Below the threshold, or with the decider disabled, the candidates are
listed in rules order (identical tokenizer first, then a draft size closest to IDEAL_RATIO of the target) without a
suggestion.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

from dgxmm.config import GIB
from dgxmm.decider.common import arecord_rules, quoted_block, safe_decide, threshold_of
from dgxmm.decider.compat import rules_on
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option
from dgxmm.planner.plans import base_model_name

log = logging.getLogger(__name__)
FEATURE = "draft"
METHODS = ("draft_model", "eagle", "eagle3", "dflash", "mtp", "ngram")
NO_DRAFT = ("mtp", "ngram")
HEAD_METHODS = ("eagle", "eagle3", "dflash")
IDEAL_RATIO = 0.08  # draft size / target size the rules order prefers (fast, still a good guesser)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")
TOKENIZER_READ_MAX = 64 * 1024**2
TIMEOUT_S = 4.0
NEVER = 1.01
QUESTION = "Which draft model will speed up the target model the most with this speculative-decoding method?"
REASON_TEXT = {"same_tokenizer": "identical tokenizer file",
               "same_family_vocab": "same model family and vocabulary size",
               "matching_head": "a draft head built for this target (hidden size, vocabulary and target match)"}
_fp_cache: dict[tuple[str, int, float], str] = {}


def tokenizer_fingerprint(artifact) -> str | None:
    """sha256 of the first tokenizer file present (cached by path, size and mtime); None without one."""
    for name in TOKENIZER_FILES:
        if name not in (artifact.files or []):
            continue
        path = os.path.join(artifact.local_path, name)
        try:
            st = os.stat(path)
            if st.st_size > TOKENIZER_READ_MAX:
                return None
            key = (path, st.st_size, st.st_mtime)
            if key not in _fp_cache:
                h = hashlib.sha256(name.encode() + b"\0")
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                _fp_cache[key] = h.hexdigest()
            return _fp_cache[key]
        except OSError:
            return None
    return None


def _cfg(artifact) -> dict:
    cfg = artifact.config or {}
    t = cfg.get("text_config")
    return {**cfg, **t} if isinstance(t, dict) else dict(cfg)


def family(artifact) -> str | None:
    mt = _cfg(artifact).get("model_type")
    return mt.lower() if isinstance(mt, str) else None


def vocab(artifact) -> int | None:
    v = _cfg(artifact).get("vocab_size")
    if isinstance(v, int):
        return v
    return artifact.arch.vocab_size if artifact.arch is not None else None


def hidden(artifact) -> int | None:
    v = _cfg(artifact).get("hidden_size")
    if isinstance(v, int):
        return v
    return artifact.arch.hidden_size if artifact.arch is not None else None


def layers(artifact) -> int | None:
    v = _cfg(artifact).get("num_hidden_layers")
    if isinstance(v, int):
        return v
    return artifact.arch.num_layers if artifact.arch is not None else None


def head_method(artifact) -> str | None:
    """eagle / eagle3 / dflash / medusa for a draft head, from its architecture name; None for a full model."""
    arch = artifact.architecture or ""
    if "dflash" in arch.lower():
        return "dflash"
    if "eagle3" in arch.lower():
        return "eagle3"
    if "eagle" in arch.lower():
        return "eagle"
    if "medusa" in arch.lower() or "speculator" in arch.lower():
        return "medusa"
    return None


def _target_hints(cfg: dict) -> list[str]:
    out = []
    sc = cfg.get("speculators_config")
    if isinstance(sc, dict) and isinstance(sc.get("verifier"), dict):
        out.append(sc["verifier"].get("name_or_path"))
    for k in ("target_model", "target_model_name_or_path", "base_model_name_or_path", "verifier_name_or_path"):
        out.append(cfg.get(k))
    return [h for h in out if isinstance(h, str) and h]


def _names_target(head, target) -> bool:
    base = base_model_name(target.ref)
    hints = _target_hints(head.config or {})
    if hints:
        return any(base_model_name(h) == base for h in hints)
    name = head.ref.rstrip("/").split("/")[-1].lower()
    return bool(base) and base in name


def compatible(target, cand, method: str, target_fp: str | None, fp_of=tokenizer_fingerprint) -> str | None:
    """Why `cand` can draft for `target` with `method` (a REASON_TEXT key), or None. `target_fp`: the target's
    tokenizer fingerprint (``fp_of(target)``)."""
    if cand.ref == target.ref or not cand.complete or cand.source != "hf" or cand.format != "safetensors":
        return None
    if method == "draft_model":
        if cand.kind not in ("llm", "draft") or head_method(cand) is not None:
            return None
        if not cand.weights_bytes or cand.weights_bytes >= target.weights_bytes:
            return None
        fam, v = family(cand), vocab(cand)
        if v is None or v != vocab(target):  # token ids must index the same vocabulary (sizes included)
            return None
        fp = fp_of(cand)
        if fp is not None and fp == target_fp:
            return "same_tokenizer"
        if fp is not None and target_fp is not None:  # both tokenizers known and different
            return None
        if fam and fam == family(target):
            return "same_family_vocab"
        return None
    if method in HEAD_METHODS:
        if head_method(cand) != method:
            return None
        if hidden(cand) is None or hidden(cand) != hidden(target) or vocab(cand) != vocab(target):
            return None
        ntl = _cfg(cand).get("num_target_layers")
        if _names_target(cand, target) or (method == "dflash" and isinstance(ntl, int) and ntl == layers(target)):
            return "matching_head"
    return None


def _order_key(target, a, why: str):
    ratio = a.weights_bytes / target.weights_bytes if target.weights_bytes else 1.0
    return (why != "same_tokenizer", abs(math.log(max(ratio, 1e-6)) - math.log(IDEAL_RATIO)), a.ref)


def candidates(target, method: str, artifacts, fp_of=tokenizer_fingerprint) -> list[tuple[object, str]]:
    """Compatible on-disk drafts for `target`, in rules order."""
    if method in NO_DRAFT:
        return []
    fp = fp_of(target) if method == "draft_model" else None
    out = [(a, why) for a in artifacts if (why := compatible(target, a, method, fp, fp_of))]
    return sorted(out, key=lambda x: _order_key(target, x[0], x[1]))


def _describe(target, a, why: str) -> str:
    ratio = a.weights_bytes / target.weights_bytes if target.weights_bytes else 0
    return (f"{a.weights_bytes / GIB:.1f} GiB ({ratio:.0%} of the target), family {family(a) or 'unknown'}, "
            f"{REASON_TEXT[why]}")


def build_state(target, cands, method: str) -> str:
    lines = [f"Speculative-decoding method: {method}",
             "A draft speeds generation up only when it runs much faster than the target: drafts of about 2-15 % of "
             "the target's size work best; a larger draft guesses a little better but gives less speed-up, and one "
             "close to the target's size gives none.",
             quoted_block("TARGET MODEL", target.ref, max_chars=300),
             f"Target: {target.weights_bytes / GIB:.1f} GiB, family {family(target) or 'unknown'}, "
             f"{layers(target) or '?'} layers"]
    for i, (a, why) in enumerate(cands, 1):
        lines.append(quoted_block(f"CANDIDATE {i}", a.ref, max_chars=300))
        lines.append(f"Candidate {i}: {_describe(target, a, why)}")
    return "\n".join(lines)


def _cand_out(target, a, why: str, p: float | None = None) -> dict:
    return {"ref": a.ref, "weights_bytes": a.weights_bytes, "reason": why, "summary": _describe(target, a, why),
            "p": p}


def _empty(method: str, **kw) -> dict:
    return {"method": method, "candidates": [], "suggested": None, "source": "none", "confidence": None,
            "decision_id": None, "reason": None, "disabled": False, **kw}


async def suggest(ctx, target_ref: str, method: str) -> dict:
    """``{method, candidates: [{ref, weights_bytes, reason, summary, p}], suggested, source, confidence,
    decision_id, reason, disabled}``. KeyError for a target that is not on disk."""
    decider = decider_of(ctx)
    if not rules_on(decider, FEATURE):
        return _empty(method, disabled=True)
    target = ctx.inventory.get(target_ref)
    if target is None:
        raise KeyError(target_ref)
    if method in NO_DRAFT:
        return _empty(method, reason="no_draft_needed")
    cands = await asyncio.to_thread(candidates, target, method, list(ctx.inventory.list()))
    subject = f"model:{target_ref}:{method}"[:256]
    if not cands:
        return _empty(method, reason="no_compatible_draft")
    if len(cands) == 1:
        a, why = cands[0]
        d = await arecord_rules(decider, feature=FEATURE, subject=subject, choice=a.ref, reason=why)
        return _empty(method, candidates=[_cand_out(target, a, why, 1.0)], suggested=a.ref, source="rules",
                      confidence=1.0, decision_id=d.decision_id)
    options = [Option(a.ref, a.ref, _describe(target, a, why)) for a, why in cands]
    state = await asyncio.to_thread(build_state, target, cands, method)
    d = await safe_decide(decider, feature=FEATURE, subject=subject, state=state, question=QUESTION, options=options,
                          threshold=threshold_of(decider, FEATURE, NEVER), timeout_s=TIMEOUT_S)
    probs = d.probabilities or {}
    listed = [_cand_out(target, a, why, probs.get(a.ref)) for a, why in cands]
    if d.source == "model" and d.choice is not None:
        listed.sort(key=lambda c: c["ref"] != d.choice)
        return _empty(method, candidates=listed, suggested=d.choice, source="model", confidence=d.confidence,
                      decision_id=d.decision_id)
    return _empty(method, candidates=listed, confidence=d.confidence, decision_id=d.decision_id)


# ---------------- evaluation adapter ----------------
def _fake_artifact(d: dict):
    from dgxmm.inventory.archspec import parse_hf_config
    cfg = d.get("config") or {}
    try:
        arch = parse_hf_config(cfg)
    except Exception:
        arch = None
    archs = cfg.get("architectures") or (cfg.get("text_config") or {}).get("architectures") or [None]
    return SimpleNamespace(ref=d["ref"], config=cfg, arch=arch, architecture=archs[0], files=d.get("files") or [],
                           local_path="/nonexistent", complete=True, source="hf", format="safetensors",
                           kind=d.get("kind", "llm"), weights_bytes=int(d["weights_bytes"]),
                           tokenizer_fp=d.get("tokenizer_sha256"))


def eval_case(case: dict):
    """A corpus case ({target, candidates, method} with configs, sizes and tokenizer hashes) -> the question DGXMM
    asks; the rules answer when exactly one candidate is compatible."""
    target = _fake_artifact(case["target"])
    pool = [_fake_artifact(c) for c in case["candidates"]]
    method = case["method"]
    cands = candidates(target, method, pool, fp_of=lambda a: a.tokenizer_fp)
    rules = cands[0][0].ref if len(cands) == 1 else None
    return SimpleNamespace(rules=rules, state=build_state(target, cands, method), question=QUESTION,
                           options=[Option(a.ref, a.ref, _describe(target, a, why)) for a, why in cands])
