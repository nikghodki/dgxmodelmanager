from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dgxmm.inventory.archspec import ArchSpec

REPO_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"


@dataclass
class ModelArtifact:
    ref: str                       # HF repo id, or local GGUF path for source="gguf"
    source: Literal["hf", "gguf"]
    local_path: str                # snapshot dir or .gguf file (dgxmm container namespace)
    revision: str | None
    format: Literal["safetensors", "gguf", "other"]
    weights_bytes: int
    size_bytes: int
    complete: bool
    kind: Literal["llm", "draft", "other"]
    arch: ArchSpec | None
    architecture: str | None
    quant: str | None
    problems: list[str] = field(default_factory=list)
    config: dict[str, Any] | None = None
    files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        a = self.arch
        return {
            "ref": self.ref, "source": self.source, "revision": self.revision, "format": self.format,
            "weights_bytes": self.weights_bytes, "size_bytes": self.size_bytes, "complete": self.complete,
            "kind": self.kind, "architecture": self.architecture, "quant": self.quant, "problems": self.problems,
            "max_context": a.max_context if a else None, "is_moe": a.is_moe if a else None,
            "num_layers": a.num_layers if a else None, "confident": a.confident if a else None,
            "has_mtp": any("mtp" in f for f in self.files),
        }
