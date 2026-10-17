"""Shared decision types (binding interface for every decider feature)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Source = Literal["rules", "model", "none"]


@dataclass(frozen=True)
class Option:
    key: str            # stable id, e.g. "gpu_oom" or plan id or parser name
    label: str          # human text
    description: str = ""  # one-line meaning, sent to the model


@dataclass
class Decision:
    feature: str        # "diagnosis" | "parser_tool" | "parser_reasoning" | "plan" | "kind" | "workload"
    subject: str        # e.g. "deployment:12", "artifact:<ref>", "preview:<hash>"
    source: Source
    choice: str | None  # Option.key or None (abstained / unavailable)
    confidence: float | None  # top probability (0..1) when source=="model"; 1.0 for rules; None for none
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float | None = None
    decision_id: int | None = None   # row id in `decisions` (None if not logged)
    reason: str | None = None        # short deterministic explanation (rules / templates)
