#!/usr/bin/env python3
"""Build the real-data corpora of the deploy-time decision-model features (compat, config, draft, engine,
remote_code) under tests/fixtures/decider_corpus/<feature>/ (docs/decider-evaluation.md).

    .venv/bin/python scripts/decider_eval/build_deploy_corpora.py [--hf-cache DIR] [compat] [config] [draft] [engine]
                                                                  [remote_code]

Only PUBLIC Hugging Face repos are fetched, anonymously, through the Hub API and ``resolve/`` (config.json,
tokenizer_config.json, the model card start, file lists, safetensors parameter counts, tokenizer.json for its
sha256, and — for remote_code — the repo's ``*.py`` files into a temporary directory that is scanned and deleted).
Models in the local HF cache are read from disk (config, sizes, tokenizer hashes). Text that would fail the repo
hygiene test (tests/test_repo_hygiene.py) is dropped. Labels:
- compat: the pinned vLLM image's supported-architecture list (tests/fixtures/decider_corpus/vllm_archs.json)
  — listed -> ``runs`` (``runs_with_flags`` with auto_map), not listed -> ``unsupported``; llama.cpp: GGUF files.
- config / draft / engine: the judging rubrics below (``judge_config``, ``judge_draft``, ``judge_engine``).
- remote_code: ``routine`` for the real repos (published model code of the listed organisations) except
  REMOTE_UNUSUAL (repos whose code does more than define a model), the synthetic samples by construction
  (``unusual`` unless the name starts with ``routine_``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dgxmm.catalog.hub import DTYPE_BYTES  # noqa: E402
from dgxmm.config import GIB  # noqa: E402
from tests.test_repo_hygiene import forbidden_hits, private_hosts  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "decider_corpus"
HUB = "https://huggingface.co"
http = httpx.Client(timeout=30, follow_redirects=True)


def clean(text: str | None) -> bool:
    return text is None or (not forbidden_hits(text) and not private_hosts(text))


def get_json(repo: str, name: str, rev: str = "main") -> dict | None:
    r = http.get(f"{HUB}/{repo}/resolve/{rev}/{name}")
    if r.status_code != 200:
        return None
    try:
        v = r.json()
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def get_text(repo: str, name: str, cap: int) -> str | None:
    r = http.get(f"{HUB}/{repo}/resolve/main/{name}", headers={"Range": f"bytes=0-{cap - 1}"})
    return r.content[:cap].decode("utf-8", "replace") if r.status_code in (200, 206) else None


def info(repo: str) -> dict | None:
    r = http.get(f"{HUB}/api/models/{repo}", params={"expand[]": ["safetensors", "siblings", "gated", "sha"]})
    return r.json() if r.status_code == 200 else None


def weights_of(meta: dict) -> int | None:
    st = (meta or {}).get("safetensors") or {}
    params = st.get("parameters") or {}
    return int(sum(n * DTYPE_BYTES.get(d, 2) for d, n in params.items())) or None


def tok_sha(repo: str) -> str | None:
    r = http.get(f"{HUB}/{repo}/resolve/main/tokenizer.json")
    if r.status_code != 200:
        return None
    return hashlib.sha256(b"tokenizer.json\0" + r.content).hexdigest()


def write(feature: str, cases: list[dict], note: str) -> None:
    d = CORPUS / feature
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cases.jsonl", "w") as f:
        for c in cases:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    (d / "README.md").write_text(note)
    print(f"{feature}: {len(cases)} cases")


PROVENANCE = ("Provenance / licence: excerpts (config keys, file names, model-card starts, code-scan summaries) of the "
              "PUBLIC Hugging Face repositories named in the cases, fetched anonymously from huggingface.co for "
              "interoperability testing only; each remains under its repository's own licence. No weights. Built "
              "with scripts/decider_eval/build_deploy_corpora.py; remove a repo here on request.\n")


# ---------------- local HF cache ----------------
def local_models(hf_cache: Path) -> list[dict]:
    """Public repos in the local HF cache: config, weight bytes, tokenizer sha256 (as dgxmm/decider/draft.py)."""
    out = []
    for d in sorted((hf_cache / "hub").glob("models--*")):
        ref = d.name[len("models--"):].replace("--", "/")
        if not clean(ref) or http.get(f"{HUB}/api/models/{ref}").status_code != 200:
            continue  # only public repos are named in fixtures
        snaps = sorted((d / "snapshots").glob("*"))
        if not snaps:
            continue
        snap = snaps[-1]
        try:
            cfg = json.loads((snap / "config.json").read_text())
        except (OSError, ValueError):
            continue
        weights = sum(os.path.getsize(p) for p in snap.glob("*.safetensors"))
        fp = None
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json"):
            if (snap / name).exists():
                fp = hashlib.sha256(name.encode() + b"\0" + (snap / name).read_bytes()).hexdigest()
                break
        out.append({"ref": ref, "config": cfg, "weights_bytes": weights, "tokenizer_sha256": fp,
                    "files": sorted(p.name for p in snap.iterdir()), "snapshot": str(snap)})
    return out


# ---------------- compat ----------------
SLIM_ITEMS = 64  # quantization_config children longer than this are cut (per-layer tables run to megabytes)
TEMPLATE_CHARS = 1500  # compat only asks whether a chat template exists; the start is kept as evidence


def slim_config(cfg):
    """Cut per-layer tables in quantization_config to their first SLIM_ITEMS entries, keeping the fixture small
    (tests/test_public_repo.py caps tracked files at 1 MB). The state DGXMM builds is capped at 1500 characters, so
    the question asked is unchanged."""
    q = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    if not isinstance(q, dict):
        return cfg
    slim = {}
    for k, v in q.items():
        if isinstance(v, dict) and len(v) > SLIM_ITEMS:
            v = dict(list(v.items())[:SLIM_ITEMS])
        elif isinstance(v, list) and len(v) > SLIM_ITEMS:
            v = v[:SLIM_ITEMS]
        slim[k] = v
    return {**cfg, "quantization_config": slim}


def build_compat(archs: set[str]) -> list[dict]:
    """Labels: listed -> runs (runs_with_flags with auto_map); not listed but implemented by the pinned image's
    transformers with attention-backend support (transformers_backend.json: vLLM's Transformers backend can serve
    it) -> any of runs / runs_with_flags / unknown; otherwise, or GGUF only -> unsupported."""
    tfb = json.loads((CORPUS / "transformers_backend.json").read_text())["architectures"]
    cases = []

    def label(cfg, files, weight_files=None):
        arch = (cfg.get("architectures") or (cfg.get("text_config") or {}).get("architectures") or [None])[0] \
            if cfg else None
        if not cfg or not arch:
            return "unsupported" if any(f.endswith(".gguf") for f in files) else "unknown"
        has_weights = bool(weight_files) or any(f.endswith((".safetensors", ".bin", ".pt", ".pth")) for f in files)
        if not has_weights and any(f.endswith(".gguf") for f in files):
            return "unsupported"  # GGUF only: nothing vLLM loads
        if arch in archs:
            return "runs_with_flags" if cfg.get("auto_map") else "runs"
        if tfb.get(arch, {}).get("attn_backend"):
            return ["runs", "runs_with_flags", "unknown"]
        return "unsupported"
    for p in sorted((CORPUS / "models").glob("*.json")):
        m = json.loads(p.read_text())
        cases.append({"name": m["ref"], "ref": m["ref"], "config": m.get("config"), "files": m.get("files") or [],
                      "weight_files": m.get("weight_files"), "template": (m.get("template") or None) and
                      m["template"][:TEMPLATE_CHARS],
                      "label": label(m.get("config"), m.get("files") or [], m.get("weight_files")),
                      "source": "models/"})
    seen = {c["ref"] for c in cases}
    # newer / unusual architectures: the most downloaded public text-generation repos the image does not list
    r = http.get(f"{HUB}/api/models", params={"pipeline_tag": "text-generation", "sort": "downloads", "limit": 1000,
                                              "expand[]": ["config", "gated"]})
    extra = 0
    for m in r.json():
        ref = m["id"]
        cfg0 = m.get("config") or {}
        arch = (cfg0.get("architectures") or [None])[0]
        if extra >= 45 or ref in seen or m.get("gated") or not arch or arch in archs or not clean(ref):
            continue
        cfg = slim_config(get_json(ref, "config.json"))
        meta = info(ref)
        if not cfg or not meta:
            continue
        files = [s["rfilename"] for s in meta.get("siblings") or []]
        tc = get_json(ref, "tokenizer_config.json") if "tokenizer_config.json" in files else None
        tmpl = tc.get("chat_template") if isinstance(tc, dict) and isinstance(tc.get("chat_template"), str) else None
        card = get_text(ref, "README.md", 8192) if "README.md" in files else None
        card = card[:1500] if card and clean(card) else None
        if not clean(json.dumps(cfg)) or not clean(tmpl):
            continue
        cases.append({"name": ref, "ref": ref, "config": cfg, "files": files, "card": card,
                      "template": tmpl and tmpl[:TEMPLATE_CHARS], "label": label(cfg, files),
                      "source": "hub-top-downloads"})
        seen.add(ref)
        extra += 1
    # GGUF repos: llama.cpp runs, vLLM has no config.json
    r = http.get(f"{HUB}/api/models", params={"library": "gguf", "sort": "downloads", "limit": 30})
    for m in r.json()[:12]:
        ref = m["id"]
        meta = info(ref)
        if not meta or not clean(ref) or ref in seen:
            continue
        files = [s["rfilename"] for s in meta.get("siblings") or []]
        cfg = get_json(ref, "config.json") if "config.json" in files else None
        cases.append({"name": ref, "ref": ref, "config": cfg, "files": files, "template": None,
                      "label": label(cfg, files), "llamacpp_label": "runs", "source": "hub-gguf"})
    return cases


# ---------------- config ----------------
GOALS = ("chat", "agent_tool_use", "batch_throughput", "long_context")


def judge_config(goal: str, opts) -> list[str]:
    """Judged best options (a set): long_context -> the longest context; batch_throughput -> the most concurrency;
    agent_tool_use -> at least 32k context with prefix caching, the most concurrency among those; chat -> 16k-64k
    context, full-precision KV where offered, 8-128 concurrent, the most context among those."""
    if not opts:
        return []
    if goal == "long_context":
        m = max(o.context for o in opts)
        return [o.key for o in opts if o.context == m]
    if goal == "batch_throughput":
        m = max(o.seqs for o in opts)
        return [o.key for o in opts if o.seqs == m]
    if goal == "agent_tool_use":
        ok = [o for o in opts if o.context >= 32768 and o.prefix]
        if not ok:
            m = max(o.context for o in opts)
            return [o.key for o in opts if o.context == m]
        m = max(o.seqs for o in ok)
        return [o.key for o in ok if o.seqs == m]
    ok = [o for o in opts if 16384 <= o.context <= 65536 and 8 <= o.seqs <= 128]
    if any(o.quality for o in ok):
        ok = [o for o in ok if o.quality]
    if not ok:
        return [o.key for o in opts if o.context == min(opts, key=lambda o: abs(math.log2(o.context / 32768))).context]
    m = max(o.context for o in ok)
    return [o.key for o in ok if o.context == m]


def _native(cfg: dict) -> int:
    from dgxmm.inventory.archspec import parse_hf_config
    try:
        return int(parse_hf_config(cfg).max_context or 0)
    except Exception:
        return 0


CONFIG_PUBLIC = ["Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "unsloth/Llama-3.3-70B-Instruct", "unsloth/Llama-3.1-8B-Instruct",
                 "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-30B-A3B", "unsloth/gemma-3-27b-it"]


def build_config(local: list[dict]) -> list[dict]:
    from dgxmm.decider.config import eval_options
    from dgxmm.inventory.kinds import artifact_kind
    pool = [m for m in local if m["weights_bytes"] > 0 and artifact_kind(
        (m["config"].get("architectures") or [None])[0], m["config"], m["files"], m["ref"]) == "llm"
        and _native(m["config"]) >= 8192]
    for ref in CONFIG_PUBLIC:
        meta, cfg = info(ref), get_json(ref, "config.json")
        if meta and cfg and weights_of(meta):
            pool.append({"ref": ref, "config": cfg, "weights_bytes": weights_of(meta)})
    cases = []
    for m in pool:
        for free_gib in (40, 90):
            if m["weights_bytes"] > (free_gib - 6) * GIB:
                continue
            for goal in GOALS:
                case = {"name": f"{m['ref']}@{free_gib}GiB:{goal}", "ref": m["ref"], "goal": goal,
                        "config": m["config"], "weights_bytes": m["weights_bytes"], "free_bytes": free_gib * GIB}
                try:
                    _, opts = eval_options(case)
                except Exception as e:  # config the estimator cannot read
                    print(f"config: skip {m['ref']}: {e}")
                    break
                label = judge_config(goal, opts)
                if label:
                    cases.append({**case, "label": label})
    return cases


# ---------------- draft ----------------
DRAFT_FAMILIES = {
    "qwen3": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B",
              "Qwen/Qwen3-32B"],
    "qwen2.5": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct"],
    "llama3": ["unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct", "unsloth/Llama-3.1-8B-Instruct",
               "unsloth/Llama-3.3-70B-Instruct"],
    "gemma3": ["unsloth/gemma-3-1b-it", "unsloth/gemma-3-4b-it", "unsloth/gemma-3-12b-it", "unsloth/gemma-3-27b-it"],
}


def judge_draft(target: dict, cands: list[dict]) -> list[str]:
    """Judged best drafts: candidates at 1-15 % of the target's size (fast, still accurate guessers); else the
    smallest."""
    ok = [c["ref"] for c in cands if 0.01 <= c["weights_bytes"] / target["weights_bytes"] <= 0.15]
    return ok or [min(cands, key=lambda c: c["weights_bytes"])["ref"]]


def build_draft(local: list[dict]) -> list[dict]:
    from dgxmm.decider.draft import eval_case
    cases = []
    from dgxmm.inventory.kinds import artifact_kind
    pool_local = []
    for m in local:
        arch = (m["config"].get("architectures") or [None])[0]
        kind = artifact_kind(arch, m["config"], m["files"], m["ref"])
        if m["weights_bytes"] > 0 and kind in ("llm", "draft"):
            pool_local.append({**{k: m[k] for k in ("ref", "config", "weights_bytes", "tokenizer_sha256", "files")},
                               "kind": kind})
    # on disk: every target x method with a compatible draft (rules cases, the check runs on models on disk)
    for t in pool_local:
        for method in ("draft_model", "dflash"):
            case = {"name": f"disk:{t['ref']}:{method}", "method": method, "target": t,
                    "candidates": [c for c in pool_local if c["ref"] != t["ref"]], "label": [], "source": "local"}
            ec = eval_case(case)
            if ec.options:
                case["label"] = [o.key for o in ec.options] if len(ec.options) == 1 else judge_draft(
                    t, [c for c in case["candidates"] if c["ref"] in {o.key for o in ec.options}])
                cases.append(case)
    # public families: each target with every smaller family member as a candidate
    for fam, refs in DRAFT_FAMILIES.items():
        ms = []
        for ref in refs:
            meta, cfg = info(ref), get_json(ref, "config.json")
            if meta and cfg and weights_of(meta):
                ms.append({"ref": ref, "config": cfg, "weights_bytes": weights_of(meta),
                           "tokenizer_sha256": tok_sha(ref), "files": ["config.json", "tokenizer.json"]})
        for t in ms:
            others = [c for c in ms if c["ref"] != t["ref"]]
            case = {"name": f"{fam}:{t['ref']}", "method": "draft_model", "target": t, "candidates": others,
                    "source": "hub"}
            ec = eval_case(case)
            if not ec.options:
                continue
            case["label"] = judge_draft(t, [c for c in others if c["ref"] in {o.key for o in ec.options}])
            cases.append(case)
    return cases


# ---------------- engine ----------------
ENGINE_PAIRS = [("Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF"), ("Qwen/Qwen3-4B", "unsloth/Qwen3-4B-GGUF"),
                ("Qwen/Qwen3-8B", "unsloth/Qwen3-8B-GGUF"), ("Qwen/Qwen3-14B", "unsloth/Qwen3-14B-GGUF"),
                ("Qwen/Qwen3-32B", "unsloth/Qwen3-32B-GGUF"),
                ("unsloth/Llama-3.1-8B-Instruct", "unsloth/Llama-3.1-8B-Instruct-GGUF"),
                ("unsloth/Llama-3.3-70B-Instruct", "unsloth/Llama-3.3-70B-Instruct-GGUF"),
                ("unsloth/gemma-3-27b-it", "unsloth/gemma-3-27b-it-GGUF"),
                ("mistralai/Mistral-Small-24B-Instruct-2501", "unsloth/Mistral-Small-24B-Instruct-2501-GGUF"),
                ("Qwen/Qwen3-30B-A3B", "unsloth/Qwen3-30B-A3B-GGUF")]


def judge_engine(goal: str, st_bytes: int, gguf_bytes: int, free: int) -> str:
    """vLLM when its weights fit with room for a KV cache (engine.need_bytes), since it serves concurrent and long
    requests far better (and chat gets sleep mode, prefix caching, faster prefill); otherwise llama.cpp, whose
    quantised GGUF fits — or, when neither fits, needs the least memory freed."""
    from dgxmm.decider.engine import need_bytes
    return "vllm" if need_bytes("safetensors", st_bytes) <= free else "llamacpp"


def build_engine(archs: set[str]) -> list[dict]:
    cases = []
    for st_ref, gg_ref in ENGINE_PAIRS:
        meta, cfg, gmeta = info(st_ref), get_json(st_ref, "config.json"), info(gg_ref)
        if not (meta and cfg and gmeta and weights_of(meta)):
            continue
        ggufs = [s for s in gmeta.get("siblings") or [] if s["rfilename"].endswith(".gguf") and "Q4_K_M" in
                 s["rfilename"] and "mmproj" not in s["rfilename"]]
        r = http.get(f"{HUB}/api/models/{gg_ref}/tree/main")
        sizes = {x["path"]: x.get("size") for x in r.json()} if r.status_code == 200 else {}
        if not ggufs or not sizes.get(ggufs[0]["rfilename"]):
            continue
        g = ggufs[0]["rfilename"]
        st_b, gg_b = weights_of(meta), int(sizes[g])
        arch = (cfg.get("architectures") or [None])[0]
        for free_gib in (24, 60, 100):
            for goal in GOALS:
                fmts = {"safetensors": {"ref": st_ref, "weights_bytes": st_b, "quant": "bf16"},
                        "gguf": {"ref": f"/models/{g}", "weights_bytes": gg_b, "quant": "Q4_K_M"}}
                cases.append({"name": f"{st_ref}@{free_gib}GiB:{goal}", "ref": st_ref, "goal": goal, "formats": fmts,
                              "architecture": arch, "vllm_supported": arch in archs, "free_bytes": free_gib * GIB,
                              "label": judge_engine(goal, st_b, gg_b, free_gib * GIB), "source": "hub"})
        cases.append({"name": f"{st_ref}:safetensors-only", "ref": st_ref, "goal": "chat",
                      "formats": {"safetensors": {"ref": st_ref, "weights_bytes": st_b, "quant": "bf16"}},
                      "architecture": arch, "vllm_supported": arch in archs, "free_bytes": 60 * GIB, "label": "vllm",
                      "source": "hub"})
        cases.append({"name": f"{gg_ref}:gguf-only", "ref": f"/models/{g}", "goal": "chat",
                      "formats": {"gguf": {"ref": f"/models/{g}", "weights_bytes": gg_b, "quant": "Q4_K_M"}},
                      "architecture": None, "vllm_supported": None, "free_bytes": 60 * GIB, "label": "llamacpp",
                      "source": "hub"})
    return cases


# ---------------- remote_code ----------------
def scan_dir(d: str) -> dict:
    out = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "dgxmm" / "decider" / "code_scan.py"), d, d],
                         capture_output=True, timeout=60, check=True)
    return json.loads(out.stdout)


# Real repos whose custom code does more than define a model: the d3 decision models ship their own HTTP server
# (http.server) and an urllib client — a serving runtime, legitimately, but not routine model code.
REMOTE_UNUSUAL = {"vllm-sr/d3-mini", "vllm-sr/d3-nano"}


def build_remote_code(local: list[dict]) -> list[dict]:
    cases = []
    repos = []
    for p in sorted((CORPUS / "models").glob("*.json")):
        m = json.loads(p.read_text())
        if (m.get("config") or {}).get("auto_map"):
            repos.append(m["ref"])
    for m in local:
        if m["config"].get("auto_map") and clean(m["ref"]):
            repos.append(m["ref"])
    for ref in dict.fromkeys(repos):
        meta = info(ref)
        if not meta:
            continue
        pys = [s["rfilename"] for s in meta.get("siblings") or [] if s["rfilename"].endswith(".py")]
        if not pys:
            continue
        tmp = tempfile.mkdtemp(prefix="adv-a1-scan-")
        try:
            for name in pys[:60]:
                r = http.get(f"{HUB}/{ref}/resolve/main/{name}")
                if r.status_code == 200 and len(r.content) < 2 * 1024**2:
                    dest = Path(tmp) / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(r.content)
            scan = scan_dir(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for f in scan["findings"]:
            if not clean(f.get("excerpt")):
                f["excerpt"] = ""
        if not clean(json.dumps(scan)):
            continue
        cases.append({"name": ref, "ref": ref, "scan": scan, "label": "unusual" if ref in REMOTE_UNUSUAL else "routine",
                      "source": "hub"})
    samples = CORPUS / "remote_code" / "samples"
    for d in sorted(samples.iterdir()):
        cases.append({"name": f"synthetic:{d.name}", "dir": f"samples/{d.name}",
                      "label": "routine" if d.name.startswith("routine_") else "unusual", "source": "synthetic"})
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("features", nargs="*", default=["compat", "config", "draft", "engine", "remote_code"])
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"))
    args = ap.parse_args()
    archs = set(json.loads((CORPUS / "vllm_archs.json").read_text()))
    local = local_models(Path(args.hf_cache))
    if "compat" in args.features:
        write("compat", build_compat(archs), "# compat corpus\n\n" + PROVENANCE + "\nLabels: the pinned vLLM image's "
              "supported-architecture list in `../vllm_archs.json` (`ModelRegistry.get_supported_archs()` of the pinned "
              "image), auto_map -> runs_with_flags; architectures the image's transformers implements with "
              "attention-backend support (`../transformers_backend.json`, so vLLM's Transformers backend can serve "
              "them) accept runs / runs_with_flags / unknown; other unlisted architectures and GGUF-only repos -> "
              "unsupported; llama.cpp: GGUF files in the repo.\n\nSize: chat templates are cut to their first 1500 "
              "characters and per-layer tables in `quantization_config` to their first 64 entries (the compat question "
              "only asks whether a template exists, and its state is capped at 1500 characters, so every question is "
              "unchanged); this keeps the file under the 1 MB cap of `tests/test_public_repo.py`.\n")
    if "config" in args.features:
        write("config", build_config(local), "# config corpus\n\n" + PROVENANCE + "\nLabels: the judging rubric "
              "`judge_config` in the build script. Models from the local HF cache are named only when public.\n")
    if "draft" in args.features:
        write("draft", build_draft(local), "# draft corpus\n\n" + PROVENANCE + "\nLabels: `judge_draft` in the "
              "build script; tokenizer hashes are sha256 of tokenizer.json.\n")
    if "engine" in args.features:
        write("engine", build_engine(archs), "# engine corpus\n\n" + PROVENANCE + "\nLabels: `judge_engine` in the "
              "build script.\n")
    if "remote_code" in args.features:
        write("remote_code", build_remote_code(local), "# remote_code corpus\n\n" + PROVENANCE + "\nReal repos: "
              "scans (findings with one-line excerpts, per-file import lists) of the repos' published `*.py` files, "
              "labelled `routine`, except the two d3 decision models (`unusual`: they ship an HTTP server and an urllib "
              "client). `samples/`: synthetic files written for this test — malicious patterns "
              "(network calls to documentation addresses, subprocess, exec of base64, writes to ~/.ssh, credential "
              "reads) labelled `unusual`, each starting with `raise ImportError(\"inert test sample\")`, and ordinary "
              "model code (`routine_*`). They are only ever parsed, never run.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
