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


async def arecord_rules(decider, **kw) -> Decision:
    """``safe_record_rules`` in a worker thread (it writes the decision log): for code on the event loop."""
    import asyncio
    return await asyncio.to_thread(lambda: safe_record_rules(decider, **kw))


def decider_view(ctx) -> dict | None:
    """The decider deployment: the one in settings, else any deployment of the decision engine (e.g. adopted)."""
    view = ctx.decider.deployment() if getattr(ctx, "decider", None) is not None else None
    if view is None and getattr(ctx, "deployer", None) is not None:
        view = next((v for v in ctx.deployer.list_views() if v["engine"] == "decision"), None)
    return view


def _num(n) -> str:
    return f"{int(n):,}" if float(n).is_integer() else f"{n:g}"


def plural(n, word: str, many: str | None = None) -> str:
    """``n`` with the word in the right number: "1 minute", "3 minutes", "2 classes" (``many``)."""
    return f"{_num(n)} {word if n == 1 else (many or word + 's')}"


def last_span(n, unit: str, many: str | None = None) -> str:
    """A window phrase for insight text: "the last minute", "the last 5 minutes" (``n`` shown rounded)."""
    if n is None:
        return f"the last few {many or unit + 's'}"
    r = round(n)
    return f"the last {unit}" if r <= 1 else f"the last {plural(r, unit, many)}"


BUSY_KEYS = ("running", "waiting", "gen_tps", "prompt_tps")
GIB = 1024**3


def busy_minute(r: dict) -> bool:
    """A metrics sample or rollup minute with any request activity (running or waiting requests, or tokens
    processed). The one definition shared by the plan safety filter, workload labels, health triage and the
    schedule runner / profiles, so they always agree on what "idle" means."""
    return any((r.get(k) or 0) > 0 for k in BUSY_KEYS)


def percentile(vals: list[float], q: float, digits: int = 4) -> float | None:
    """The ``q`` quantile (0–1) of ``vals`` by linear interpolation between the two closest ranks (the
    "inclusive" method, numpy's default), rounded to ``digits``; None for no values."""
    if not vals:
        return None
    xs = sorted(vals)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), digits)


def gib(n) -> str:
    """Bytes as GiB text for insight and state text: "3.0 GiB" (None counts as 0)."""
    return f"{(n or 0) / GIB:.1f} GiB"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval (95 % by default) of a proportion k/n; (None, None) for n == 0. Used to judge
    thresholds from small samples (evaluation harness, threshold tuning)."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
