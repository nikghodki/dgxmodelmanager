"""Advanced feature 3 — pre-download compatibility check: will a Hugging Face repo run here, before it is downloaded?

Only a handful of small files are read, through the existing ``HfHubClient`` (``fetch_small``: allow-listed names,
64 KiB per file, model card 8 KiB, 5 s, held in memory and never written to the HF cache): ``config.json``,
``tokenizer_config.json`` / ``chat_template.jinja``, ``generation_config.json`` and the model card.

Rules first (per engine):
- vLLM: the architecture against the pinned image's supported-architecture list (``ctx.archs``); ``auto_map`` ->
  needs ``trust_remote_code``; gated -> needs an HF token (without access the repo is reported as gated, not as an
  error); no chat template -> chat completions need one; vision / audio sub-configs are reported.
- llama.cpp: GGUF files in the repo.
The decision model is asked only where the rules cannot decide: an architecture the pinned image does not list
natively (vLLM may still serve it through its Transformers backend) or an architecture list that is not loaded yet.
It chooses among ``VERDICTS``; below the threshold the verdict stays ``unknown``.

Results (verdicts, reason keys and flags; never file contents) are cached in ``compat_cache`` by (repo, sha) for 7
days. Answers that depend on missing access or a missing decider (gated without a token, architecture list not
loaded, model abstained, Hub errors) are not cached, so they are checked again next time.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import delete

from dgxmm.catalog.hub import CARD_CAP, HubGated
from dgxmm.db import CompatCacheRow, utcnow
from dgxmm.decider.common import arecord_rules, quoted_block, safe_decide, threshold_of
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
FEATURE = "compat"
CACHE_DAYS = 7
TIMEOUT_S = 4.0       # model question (the catalog sheet loads it in the background)
GATHER_TIMEOUT_S = 25.0  # metadata + at most five files of FETCH_TIMEOUT_S each
# Hub calls of the check run on their own small pool: a stalled Hub fills these threads (each call has its own
# timeout too), never asyncio's default pool that every database to_thread uses.
HUB_THREADS = 2
_HUB_POOL = concurrent.futures.ThreadPoolExecutor(HUB_THREADS, thread_name_prefix="compat-hub")
NEVER = 1.01
VERDICTS: dict[str, str] = {
    "runs": "vLLM serves this model as it is",
    "runs_with_flags": "vLLM serves it with extra settings (trust_remote_code, a Hugging Face token)",
    "unsupported": "vLLM cannot serve this architecture",
    "unknown": "There is not enough information to tell",
}
OPTIONS = [Option(k, k, d) for k, d in VERDICTS.items()]
QUESTION = "Can the pinned vLLM image serve the model in this Hugging Face repository?"
# Reason keys shown by the UI (frontend/src/lib/compat.ts); the first four are failure-diagnosis cause keys.
REASONS = ("unsupported_architecture", "trust_remote_code_required", "auth_gated", "tokenizer_or_template",
           "supported_architecture", "unknown_architecture", "registry_unavailable", "no_config", "no_safetensors",
           "vision", "audio", "gguf_available", "no_gguf")
RUN_FLAGS = ("trust_remote_code", "hf_token")  # flags without which vLLM does not start
CONFIG_KEYS = ("architectures", "model_type", "auto_map", "torch_dtype", "dtype", "quantization_config",
               "max_position_embeddings", "num_hidden_layers", "hidden_size", "vocab_size", "text_config",
               "vision_config", "audio_config", "is_encoder_decoder", "num_labels")


@dataclass
class RepoFacts:
    repo: str
    sha: str
    files: list[str]
    config: dict | None = None
    tokenizer_config: dict | None = None
    chat_template: str | None = None
    generation_config: dict | None = None
    card: str | None = None
    gated: object = False
    accessible: bool = True
    fetch_errors: list[str] = field(default_factory=list)
    weight_files: int | None = None  # corpus fixtures list weight files by count only

    @property
    def has_weights(self) -> bool:
        """Weights vLLM loads (safetensors / PyTorch), not only GGUF."""
        return bool(self.weight_files) or any(f.endswith((".safetensors", ".bin", ".pt", ".pth")) for f in self.files)

    @property
    def architecture(self) -> str | None:
        cfg = self.config or {}
        archs = cfg.get("architectures")
        if not archs and isinstance(cfg.get("text_config"), dict):
            archs = cfg["text_config"].get("architectures")
        return archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else None

    @property
    def gguf_files(self) -> list[str]:
        return [f for f in self.files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]

    @property
    def has_template(self) -> bool:
        tc = self.tokenizer_config or {}
        return bool(self.chat_template or tc.get("chat_template") or "chat_template.json" in self.files
                    or "chat_template.jinja" in self.files)


@dataclass
class CompatRules:
    vllm: dict
    llamacpp: dict
    ask_model: bool  # the vLLM verdict is for the decision model


def _verdict(verdict: str, reasons: list[str], flags: list[str]) -> dict:
    return {"verdict": verdict, "reasons": list(dict.fromkeys(reasons)), "flags": list(dict.fromkeys(flags))}


def rules(f: RepoFacts, archs: set[str] | None) -> CompatRules:
    gguf = f.gguf_files
    llamacpp = _verdict("runs", ["gguf_available"], []) if gguf else _verdict("unsupported", ["no_gguf"], [])
    if f.gated and not f.accessible:
        return CompatRules(_verdict("unknown", ["auth_gated"], ["hf_token"]), llamacpp, False)
    arch = f.architecture
    if f.config is None or not arch:
        verdict = "unsupported" if gguf and "config.json" not in f.files else "unknown"
        return CompatRules(_verdict(verdict, ["no_config"], []), llamacpp, False)
    extra, flags = [], []
    cfg = f.config
    if cfg.get("auto_map") or (isinstance(cfg.get("text_config"), dict) and cfg["text_config"].get("auto_map")):
        extra.append("trust_remote_code_required")
        flags.append("trust_remote_code")
    if f.gated:
        extra.append("auth_gated")
        flags.append("hf_token")
    if not f.has_template:
        extra.append("tokenizer_or_template")
        flags.append("chat_template")
    if isinstance(cfg.get("vision_config"), dict):
        extra.append("vision")
    if isinstance(cfg.get("audio_config"), dict):
        extra.append("audio")
    if archs is None:
        return CompatRules(_verdict("unknown", ["registry_unavailable", *extra], flags), llamacpp, True)
    if arch not in archs:
        return CompatRules(_verdict("unknown", ["unknown_architecture", *extra], flags), llamacpp, True)
    if gguf and not f.has_weights:  # a GGUF repo with a config.json for metadata: nothing vLLM can load
        return CompatRules(_verdict("unsupported", ["no_safetensors", *extra], flags), llamacpp, False)
    verdict = "runs_with_flags" if any(x in flags for x in RUN_FLAGS) else "runs"
    return CompatRules(_verdict(verdict, ["supported_architecture", *extra], flags), llamacpp, False)


def from_model(r: CompatRules, choice: str) -> dict:
    """The vLLM verdict for a model answer: the rules' flags still apply (a "runs" answer for a repo with auto_map
    is "runs_with_flags")."""
    flags = r.vllm["flags"]
    if choice == "runs" and any(x in flags for x in RUN_FLAGS):
        choice = "runs_with_flags"
    reasons = [x for x in r.vllm["reasons"] if x != "unknown_architecture"]
    if choice == "unsupported":
        reasons.insert(0, "unsupported_architecture")
    elif "registry_unavailable" not in reasons:
        reasons.insert(0, "unknown_architecture")
    return _verdict(choice, reasons, flags)


# ---------------- reading the repo ----------------
def _json(data: bytes | None) -> dict | None:
    if data is None:
        return None
    try:
        v = json.loads(data.decode("utf-8", "replace"))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def valid_sha(sha) -> str | None:
    """The Hub's commit sha when it is one (40 hex); anything else is not put in a URL or the cache key."""
    return sha if isinstance(sha, str) and SHA_RE.fullmatch(sha) else None


def gather(hub, repo: str, meta, anonymous: bool = False) -> RepoFacts:
    """Blocking (run in a thread): the allow-listed files of a repo whose metadata (`meta`) was read already.
    `anonymous`: without the HF token (checks a viewer asked for)."""
    rev = valid_sha(meta.sha)
    f = RepoFacts(repo=repo, sha=rev or "main", files=list(meta.files or []), gated=meta.gated or False)
    kw = {"anonymous": True} if anonymous else {}
    try:
        for name, attr in (("config.json", "config"), ("tokenizer_config.json", "tokenizer_config"),
                           ("generation_config.json", "generation_config")):
            if name in f.files:
                try:
                    got = hub.fetch_small(repo, name, revision=rev, **kw)
                except (TimeoutError, OSError, ValueError) as e:
                    f.fetch_errors.append(f"{name}: {type(e).__name__}")
                    continue
                setattr(f, attr, _json(got.data))
        for name, attr, cap, cut in (("chat_template.jinja", "chat_template", None, False),
                                     ("README.md", "card", CARD_CAP, True)):
            if name in f.files:
                try:
                    got = hub.fetch_small(repo, name, revision=rev, truncate=cut, **({"cap": cap} if cap else {}),
                                          **kw)
                except (TimeoutError, OSError, ValueError) as e:
                    f.fetch_errors.append(f"{name}: {type(e).__name__}")
                    continue
                if got.data is not None:
                    setattr(f, attr, got.data.decode("utf-8", "replace"))
    except HubGated:
        f.accessible = False
        f.gated = f.gated or True
        f.config = f.tokenizer_config = f.generation_config = None
        f.chat_template = f.card = None
    return f


def _excerpt(cfg: dict | None) -> str:
    c = cfg or {}
    small = {k: c[k] for k in CONFIG_KEYS if k in c}
    for sub in ("text_config", "vision_config", "audio_config"):
        if isinstance(small.get(sub), dict):
            small[sub] = {k: v for k, v in small[sub].items() if k in CONFIG_KEYS}
    return json.dumps(small or {"keys": sorted(c)[:40]}, default=str)[:1500]


def build_state(f: RepoFacts, archs: set[str] | None) -> str:
    if archs is None:
        listed = "The pinned vLLM image's list of supported architectures is not available."
    else:
        listed = (f"The pinned vLLM image implements {len(archs)} architectures natively; this architecture is NOT "
                  "among them (vLLM can serve some others through its generic Transformers backend).")
    weights = sorted({p.rsplit(".", 1)[-1] for p in f.files if p.endswith((".safetensors", ".bin", ".gguf", ".pt"))})
    return "\n".join([
        listed,
        f"Custom code (auto_map): {'yes' if (f.config or {}).get('auto_map') else 'no'}; weight formats: "
        f"{', '.join(weights) or 'none'}; chat template: {'yes' if f.has_template else 'no'}",
        quoted_block("REPOSITORY", f.repo, max_chars=300),
        quoted_block("CONFIG.JSON (excerpt)", _excerpt(f.config), max_chars=1500, keep="head"),
        quoted_block("MODEL CARD (start)", f.card or "(none)", max_chars=1500, keep="head"),
    ])


# ---------------- cache ----------------
def cache_get(db, repo: str, sha: str) -> dict | None:
    with db.session() as s:
        row = s.get(CompatCacheRow, (repo, sha))
        if row is None:
            return None
        from dgxmm.db import as_aware
        if as_aware(row.updated_at) < utcnow() - timedelta(days=CACHE_DAYS):
            return None
        return dict(row.result)


def cache_put(db, repo: str, sha: str, result: dict) -> None:
    keep = {k: result[k] for k in ("repo", "sha", "architecture", "gated", "vllm", "llamacpp", "source",
                                   "confidence")}
    with db.session() as s:
        row = s.get(CompatCacheRow, (repo, sha))
        if row is None:
            s.add(CompatCacheRow(repo=repo, sha=sha, result=keep))
        else:
            row.result, row.updated_at = keep, utcnow()


def prune_cache(db) -> int:
    """Retention: cached results older than 7 days."""
    with db.session() as s:
        return s.execute(delete(CompatCacheRow).where(
            CompatCacheRow.updated_at < utcnow() - timedelta(days=CACHE_DAYS))).rowcount


# ---------------- the check ----------------
def rules_on(decider, feature: str) -> bool:
    """The feature's toggle (its rules run with the decision model disabled); True when the decider has none."""
    fn = getattr(decider, "rules_enabled", None)
    try:
        return True if fn is None else bool(fn(feature))
    except Exception:
        return True


def _empty(repo: str, **kw) -> dict:
    unknown = _verdict("unknown", [], [])
    return {"repo": repo, "sha": None, "architecture": None, "gated": False, "vllm": unknown, "llamacpp": unknown,
            "source": "none", "confidence": None, "decision_id": None, "cached": False, "error": None,
            "disabled": False, **kw}


async def check(ctx, repo: str, anonymous: bool = False) -> dict:
    """``{repo, sha, architecture, gated, vllm: {verdict, reasons, flags}, llamacpp: {...}, source, confidence,
    decision_id, cached, error, disabled}``; never raises (Hub or decider trouble -> verdicts ``unknown``).
    `anonymous`: fetch without the HF token (a viewer's check); gated repos are then reported as gated, and
    rules-only answers are not logged (viewers cannot fill the decision log)."""
    decider = decider_of(ctx)
    if not rules_on(decider, FEATURE):
        return _empty(repo, disabled=True)
    hub = getattr(ctx, "hub", None)
    if hub is None:
        return _empty(repo, error="hub_unavailable")
    loop = asyncio.get_running_loop()
    try:
        meta = await asyncio.wait_for(loop.run_in_executor(
            _HUB_POOL, functools.partial(hub.repo_meta, repo, anonymous=anonymous)), GATHER_TIMEOUT_S)
        sha = valid_sha(meta.sha)
        if sha and (hit := await asyncio.to_thread(cache_get, ctx.db, repo, sha)) is not None:
            return {**_empty(repo), **hit, "cached": True}
        facts = await asyncio.wait_for(loop.run_in_executor(_HUB_POOL, gather, hub, repo, meta, anonymous),
                                       GATHER_TIMEOUT_S)
    except Exception as e:  # unknown repo, offline, slow Hub: no verdict, never a 5xx
        log.info("compat check of %s: hub unavailable (%s)", repo, type(e).__name__)
        return _empty(repo, error="hub_unavailable")
    archs_src = getattr(ctx, "archs", None)
    archs = archs_src.get() if archs_src is not None else None
    r = rules(facts, archs)
    subject = f"repo:{repo}"[:256]
    out = {**_empty(repo), "sha": valid_sha(facts.sha), "architecture": facts.architecture,
           "gated": bool(facts.gated), "llamacpp": r.llamacpp}
    record = not anonymous
    if record:
        await arecord_rules(decider, feature=FEATURE, subject=subject, choice=r.llamacpp["verdict"],
                            reason="llama.cpp: " + ", ".join(r.llamacpp["reasons"]))
    if not r.ask_model:
        d = await arecord_rules(decider, feature=FEATURE, subject=subject, choice=r.vllm["verdict"],
                                reason="vLLM: " + ", ".join(r.vllm["reasons"])) if record else None
        out.update(vllm=r.vllm, source="rules", confidence=1.0, decision_id=d.decision_id if d else None)
    else:
        state = await asyncio.to_thread(build_state, facts, archs)
        d = await safe_decide(decider, feature=FEATURE, subject=subject, state=state, question=QUESTION,
                              options=OPTIONS, threshold=threshold_of(decider, FEATURE, NEVER), timeout_s=TIMEOUT_S)
        if d.source == "model" and d.choice is not None:
            out.update(vllm=from_model(r, d.choice), source="model", confidence=d.confidence,
                       decision_id=d.decision_id)
        else:
            out.update(vllm=r.vllm, source="none", confidence=d.confidence, decision_id=d.decision_id)
    if facts.fetch_errors:
        out["error"] = "partial"
    # gated and private repos are never cached: what a token can read must not be served to a check made without it
    cacheable = (facts.accessible and not facts.gated and not getattr(meta, "private", False) and archs is not None
                 and not facts.fetch_errors
                 and out["source"] != "none" and valid_sha(facts.sha) is not None)
    if cacheable:
        try:
            await asyncio.to_thread(cache_put, ctx.db, repo, facts.sha, out)
        except Exception:
            log.warning("could not cache the compatibility of %s", repo, exc_info=True)
    return out


# ---------------- evaluation adapter (tools/eval_feature.py compat default) ----------------
_EVAL_ARCHS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "decider_corpus" / "vllm_archs.json"


def eval_case(case: dict, archs: set[str] | None = None):
    """A corpus case (a repo's config, file list and chat template, as in tests/fixtures/decider_corpus/models) ->
    the vLLM question DGXMM asks; the rules answer where the rules decide (an ``EvalCase`` of tools/eval_feature.py)."""
    if archs is None:
        archs = set(json.loads(_EVAL_ARCHS.read_text()))
    tmpl = case.get("template")
    f = RepoFacts(repo=case.get("ref") or case["name"], sha="eval", files=list(case.get("files") or []),
                  config=case.get("config"), tokenizer_config={"chat_template": tmpl} if tmpl else {},
                  card=case.get("card"), gated=case.get("gated", False), accessible=case.get("accessible", True),
                  weight_files=case.get("weight_files"))
    r = rules(f, archs)
    return SimpleNamespace(rules=None if r.ask_model else r.vllm["verdict"], state=build_state(f, archs),
                    question=QUESTION, options=OPTIONS)
