"""Declarative registry of the decision-model features (spec "Feature registry").

One ``FeatureSpec`` per feature drives the settings (``decider.feature.<group>.enabled|threshold``), the toggles and
thresholds of the Decision model page (``GET /api/decider/status`` ``features`` + ``feature_specs``), the per-feature
rate-limit buckets (dgxmm/api/ratelimit.py), the client priority of model calls (``background``) and the defaults of
``Decider.threshold`` / ``feature_enabled``. Adding a feature is one ``register(FeatureSpec(...))`` call.

Groups: settings are per *group*. A group is the spec whose ``name == group``; other specs with that group are its
members (``parser_tool`` and ``parser_reasoning`` are logged as decisions of their own but share the ``parser``
toggle and threshold). Members must be registered after their group and take its threshold.

A registered feature without an implementation is harmless: nothing asks the decider for it, no job runs for it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NAME = re.compile(r"^[a-z][a-z_]{0,31}$")  # = the settings API's FeatureName
DEFAULT_RATE_LIMIT: tuple[int, float, int | None] = (30, 60.0, 120)  # per user / window s / all users
# Rate-limit buckets of dgxmm/api/ratelimit.py that are not a feature's: no feature may map onto one of them.
STATIC_BUCKETS = frozenset({"decider_admin", "decider_feedback", "decider_read", "insights_read", "insights_write"})


@dataclass(frozen=True)
class FeatureSpec:
    """A decision-model feature.

    - ``name``: decision feature as logged in ``decisions.feature`` (``[a-z][a-z_]{0,31}``).
    - ``group``: settings group (= ``name`` for a group; e.g. ``parser`` for ``parser_tool``).
    - ``label`` / ``description``: shown on the Decision model page (plain text).
    - ``default_threshold``: confidence a model answer needs (0..1) unless the admin set another.
    - ``background``: asked from background work (queues behind interactive questions in the decider client).
    - ``needs_model``: False for rules-only features (no decider calls at all, run with the decider disabled).
    - ``viewer_visible``: its results (insights) may be shown to viewers; False = admins only.
    - ``rate_limit``: (per user, window s, all users or None) of its API bucket (``feature_bucket(name)``).
    - ``bucket``: name of that bucket; default ``decider_<name>``.
    """
    name: str
    group: str
    label: str
    description: str
    default_threshold: float
    background: bool = False
    needs_model: bool = True
    viewer_visible: bool = True
    rate_limit: tuple[int, float, int | None] = DEFAULT_RATE_LIMIT
    bucket: str | None = field(default=None)

    def __post_init__(self) -> None:
        if not _NAME.match(self.name) or not _NAME.match(self.group):
            raise ValueError(f"invalid feature name {self.name!r} / group {self.group!r}")
        if not 0.0 <= float(self.default_threshold) <= 1.0:
            raise ValueError(f"threshold of {self.name} outside [0, 1]")
        if not self.label or not self.description:
            raise ValueError(f"feature {self.name} needs a label and a description")


_SPECS: dict[str, FeatureSpec] = {}
# Derived tables, updated in place by register() (other modules hold references to these very objects):
DEFAULT_THRESHOLDS: dict[str, float] = {}  # group -> default threshold (dgxmm.decider.protocol / service.FEATURES)
GROUP_OF: dict[str, str] = {}  # member name -> group, for members only (service.FEATURE_GROUP)


def register(spec: FeatureSpec) -> FeatureSpec:
    """Add a feature. Raises ValueError for a duplicate name or a member whose group is not registered yet."""
    if spec.name in _SPECS:
        raise ValueError(f"feature {spec.name} is already registered")
    if spec.group == spec.name and (spec.bucket or f"decider_{spec.name}") in STATIC_BUCKETS:
        raise ValueError(f"feature {spec.name}: its rate-limit bucket would be a static one")
    if spec.group != spec.name:
        head = _SPECS.get(spec.group)
        if head is None or head.group != head.name:
            raise ValueError(f"feature {spec.name}: group {spec.group} is not registered")
        GROUP_OF[spec.name] = spec.group
    else:
        DEFAULT_THRESHOLDS[spec.name] = float(spec.default_threshold)
    _SPECS[spec.name] = spec
    return spec


def _unregister(name: str) -> None:
    """Tests only."""
    spec = _SPECS.pop(name, None)
    if spec is not None:
        DEFAULT_THRESHOLDS.pop(name, None)
        GROUP_OF.pop(name, None)


def find(name: str) -> FeatureSpec | None:
    return _SPECS.get(name)


def get(name: str) -> FeatureSpec:
    """The spec of a feature or group; KeyError if unknown."""
    return _SPECS[name]


def all_features() -> list[FeatureSpec]:
    """Every registered spec (groups and members), in registration order."""
    return list(_SPECS.values())


def groups() -> list[FeatureSpec]:
    """The settings groups (one toggle + threshold each), in display order."""
    return [s for s in _SPECS.values() if s.group == s.name]


def group_of(name: str) -> str:
    """The settings group of a decision feature (unknown names are their own group)."""
    return GROUP_OF.get(name, name)


def is_background(name: str) -> bool:
    s = _SPECS.get(name) or _SPECS.get(group_of(name))
    return bool(s and s.background)


def feature_bucket(name: str) -> str:
    """The rate-limit bucket of a feature's API routes (members use their group's)."""
    s = _SPECS[group_of(name)]
    return s.bucket or f"decider_{s.name}"


def buckets() -> dict[str, tuple[int, float, int | None]]:
    return {feature_bucket(s.name): s.rate_limit for s in groups()}


def feature_info() -> list[dict]:
    """The groups as the Decision model page renders them (status ``feature_specs``)."""
    return [{"name": s.name, "label": s.label, "description": s.description,
             "default_threshold": s.default_threshold, "background": s.background, "needs_model": s.needs_model,
             "viewer_visible": s.viewer_visible,
             "members": [m.name for m in _SPECS.values() if m.group == s.name and m.name != s.name]}
            for s in groups()]


# ---------------- the shipped five (behaviour and thresholds unchanged; calibration notes in protocol.py) -------
register(FeatureSpec("diagnosis", "diagnosis", "Failure diagnosis",
                     "Explain why a deployment failed and suggest a fix.", 0.8, background=True))
register(FeatureSpec("parser", "parser", "Parser suggestions",
                     "Suggest tool-call and reasoning parsers in the deploy wizard.", 0.8, viewer_visible=False,
                     bucket="decider_suggest"))
register(FeatureSpec("parser_tool", "parser", "Tool-call parser", "Suggest the tool-call parser.", 0.8,
                     viewer_visible=False))
register(FeatureSpec("parser_reasoning", "parser", "Reasoning parser", "Suggest the reasoning parser.", 0.8,
                     viewer_visible=False))
register(FeatureSpec("plan", "plan", "Plan recommendation", "Mark the best safe plan when a model does not fit.",
                     0.6, viewer_visible=False, bucket="decider_plan"))
register(FeatureSpec("kind", "kind", "Model kind",
                     "Classify models no rule recognises (embedding, reranker, …).", 0.8, background=True))
register(FeatureSpec("workload", "workload", "Workload-aware advice",
                     "Label each deployment's workload and rank its advice.", 0.75, background=True))

# ---------------- the eleven advanced features (spec 2026-10-17; behaviour added by later lanes) ----------------
# Default thresholds below are the ones calibrated on each feature's real corpus: see the notes in protocol.py
# and docs/decider-evaluation.md (tests keep the three in step).
register(FeatureSpec("health", "health", "Health triage", "Health triage of running deployments.", 0.85,
                     background=True))
# Deploy-time features (lane A1): thresholds checked on their real corpora (protocol.py, docs/decider-evaluation.md).
register(FeatureSpec("config", "config", "Config advice",
                     "Suggest a sizing for the deployment's goal in the deploy wizard (from the memory calculator).",
                     0.8, viewer_visible=False))
register(FeatureSpec("compat", "compat", "Compatibility check",
                     "Say whether a Hugging Face repo runs on vLLM / llama.cpp here, before it is downloaded.", 0.8))
register(FeatureSpec("draft", "draft", "Draft pairing",
                     "Suggest an on-disk draft model for speculative decoding.", 0.8, viewer_visible=False))
register(FeatureSpec("engine", "engine", "Engine choice",
                     "Suggest vLLM or llama.cpp when a model is on disk in both formats.", 0.8,
                     viewer_visible=False))
register(FeatureSpec("command", "command", "Natural-language commands",
                     "Natural-language commands in the ⌘K palette.", 0.7))
register(FeatureSpec("schedule", "schedule", "Sleep/wake schedules",
                     "Usage-pattern sleep/wake suggestions with user-owned schedules.", 0.8, background=True,
                     rate_limit=(60, 60.0, 240)))  # also the bucket of the schedules API (page reads + edits)
register(FeatureSpec("reading", "reading", "Reading labels", "Benchmark/calibration reading outlier labels.", 0.95,
                     background=True))
register(FeatureSpec("tuning", "tuning", "Threshold tuning", "Threshold tuning from feedback (rules only).", 0.8,
                     background=True, needs_model=False, viewer_visible=False))
register(FeatureSpec("routing", "routing", "Routing advice",
                     "LiteLLM routing advice (advisory only — no live routing).", 0.8, background=True,
                     viewer_visible=False))
register(FeatureSpec("remote_code", "remote_code", "Remote-code risk",
                     "Scan a repo's custom code before trust_remote_code runs it, and flag unusual code.", 0.8,
                     viewer_visible=False))
