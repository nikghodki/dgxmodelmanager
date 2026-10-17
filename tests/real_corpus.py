"""Loaders for the real-data corpora of the decision-model features (docs/decider-evaluation.md).

- tests/fixtures/real_failures/*.txt: start-up failures induced on the DGX (redacted log tail + true label).
- tests/fixtures/decider_corpus/models/*.json: config / file list / chat template of real HF repos (no weights).
- tests/fixtures/decider_corpus/labels.json: true kind and vLLM parsers per repo (with the source of each label).
- tests/fixtures/workload/*.json: metric summaries of real traffic on a test deployment + the true label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent / "fixtures"
FAILURES = ROOT / "real_failures"
CORPUS = ROOT / "decider_corpus"
WORKLOAD = ROOT / "workload"


@dataclass
class Failure:
    name: str
    label: str
    engine: str
    model_ref: str
    exit_code: int | None
    message: str
    log: str
    meta: dict = field(default_factory=dict)


def load_failures() -> list[Failure]:
    out = []
    for p in sorted(FAILURES.glob("*.txt")):
        head, _, log = p.read_text().partition("\n---\n")
        meta = {}
        for ln in head.splitlines():
            if ln.startswith("# ") and ":" in ln:
                k, v = ln[2:].split(":", 1)
                meta[k.strip()] = v.strip()
        code = meta.get("exit_code")
        out.append(Failure(p.stem, meta["label"], meta.get("engine", "vllm"), meta.get("model_ref", ""),
                           int(code) if code not in (None, "", "None") else None, meta.get("message", ""), log, meta))
    return out


@dataclass
class Repo:
    ref: str
    architecture: str | None
    model_type: str | None
    config: dict | None
    files: list[str]
    weight_files: int
    template: str | None


def load_repos() -> dict[str, Repo]:
    out = {}
    for p in sorted((CORPUS / "models").glob("*.json")):
        d = json.loads(p.read_text())
        out[d["ref"]] = Repo(d["ref"], d.get("architecture"), d.get("model_type"), d.get("config"),
                             d.get("files") or [], d.get("weight_files") or 0, d.get("template"))
    return out


def load_labels() -> dict[str, dict]:
    return json.loads((CORPUS / "labels.json").read_text())["repos"]


def load_workloads() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(WORKLOAD.glob("*.json"))]


def as_artifact(repo: Repo):
    """A ModelArtifact as the HF scanner would build it from this repo (weights are not needed by the rules)."""
    from dgxmm.inventory.artifact import ModelArtifact
    from dgxmm.inventory.hf_scan import artifact_kind
    files = list(repo.files) + [f"model-{i:05d}.safetensors" for i in range(repo.weight_files)]
    return ModelArtifact(ref=repo.ref, source="hf", local_path="", revision=None,
                         format="safetensors" if repo.weight_files else "other", weights_bytes=0, size_bytes=0,
                         complete=bool(repo.config), kind=artifact_kind(repo.architecture, repo.config, files, repo.ref),
                         arch=None, architecture=repo.architecture, quant=None, config=repo.config, files=files)


def model_info(repo: Repo):
    from dgxmm.decider.config_suggest import ModelInfo
    return ModelInfo(ref=repo.ref, architecture=repo.architecture, model_type=repo.model_type, template=repo.template)
