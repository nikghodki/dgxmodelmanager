"""Helpers shared by the smart features: quoting untrusted text into `state`, and a fail-open decide()."""
from __future__ import annotations

import asyncio
import logging
import math
import re

from dgxmm.decider.types import Decision, Option
from dgxmm.memory.redact import redact_text

log = logging.getLogger(__name__)
OPEN, CLOSE = "<<<", ">>>"
# The decider deployment holds (or is about to hold) a container: enable keeps or replaces it, status reports a
# pending model change. A key rotation also restarts a "stopping" one (it would never get the new key otherwise).
DECIDER_RUNNING = ("planned", "queued", "starting", "loading", "ready", "unknown", "sleeping")
DECIDER_ROTATED = (*DECIDER_RUNNING, "stopping")
_RUN = re.compile(r"<{3,}|>{3,}")  # any run of 3+ angle brackets (a plain replace() leaves "<<<<<" -> "< < <<<")


def quoted_block(header: str, text: str | None, *, max_chars: int = 6144, max_lines: int | None = None,
                 keep: str = "tail") -> str:
    """Redact `text` and wrap it as an untrusted, length-capped quoted block for a decider `state`.

    `keep="tail"` keeps the end (logs), `"head"` the beginning (files). The block delimiters inside the text are
    defused so quoted data cannot close the block and pose as trusted state."""
    raw = text or ""
    lines = raw.splitlines()
    truncated = False
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[-max_lines:] if keep == "tail" else lines[:max_lines]
        truncated = True
    body = redact_text("\n".join(lines))
    if len(body) > max_chars:
        body = body[-max_chars:] if keep == "tail" else body[:max_chars]
        truncated = True
    body = _RUN.sub(lambda m: " ".join(m.group(0)), body)
    note = "untrusted, truncated" if truncated else "untrusted"
    return f"{header} ({note}):\n{OPEN}\n{body}\n{CLOSE}"


def finite(v):
    """A stored number for the API: None unless it is a finite number (rows written before the decider's answers
    were validated may hold NaN / Infinity, which JSON responses refuse)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v if math.isfinite(v) else None


def finite_probs(p) -> dict:
    return {str(k): f for k, v in (p or {}).items() if (f := finite(v)) is not None} if isinstance(p, dict) else {}


def abstain(feature: str, subject: str) -> Decision:
    return Decision(feature=feature, subject=subject, source="none", choice=None, confidence=None)


def threshold_of(decider, feature: str, default: float) -> float:
    """decider.threshold(feature), or `default` if it errors."""
    try:
        return float(decider.threshold(feature))
    except Exception:
        log.warning("decider threshold for %s unavailable; using %s", feature, default, exc_info=True)
        return default


async def safe_decide(decider, *, feature: str, subject: str, state: str, question: str, options: list[Option],
                      threshold: float | None = None, timeout_s: float = 2.0) -> Decision:
    """Feature-enabled + availability check + decide() under ONE deadline of `timeout_s`, never raising:
    disabled, unavailable, slow or failing -> source="none"."""
    try:
        async with asyncio.timeout(timeout_s):
            if not decider.feature_enabled(feature) or not await decider.available():
                return abstain(feature, subject)
            d = await decider.decide(feature=feature, subject=subject, state=state, question=question,
                                     options=options, threshold=threshold, timeout_s=timeout_s)
    except Exception:  # TimeoutError included
        log.warning("decider %s failed for %s; falling back", feature, subject, exc_info=True)
        return abstain(feature, subject)
    if d is None or (d.choice is not None and d.choice not in {o.key for o in options}):
        return abstain(feature, subject)  # never trust a choice outside the offered options
    return d


def safe_record_rules(decider, *, feature: str, subject: str, choice: str, reason: str | None,
                      options: list[Option] | None = None) -> Decision:
    try:
        d = decider.record_rules(feature=feature, subject=subject, choice=choice, reason=reason, options=options)
        if d is not None:
            return d
    except Exception:
        log.warning("could not log rules decision %s for %s", feature, subject, exc_info=True)
    return Decision(feature=feature, subject=subject, source="rules", choice=choice, confidence=1.0, reason=reason)


def decider_view(ctx) -> dict | None:
    """The decider deployment: the one in settings, else any deployment of the decision engine (e.g. adopted)."""
    view = ctx.decider.deployment() if getattr(ctx, "decider", None) is not None else None
    if view is None and getattr(ctx, "deployer", None) is not None:
        view = next((v for v in ctx.deployer.list_views() if v["engine"] == "decision"), None)
    return view
