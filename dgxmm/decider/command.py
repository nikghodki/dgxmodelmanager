"""Feature 6 — natural-language commands in the ⌘K palette ("Ask" mode, spec §6).

``interpret(text)`` turns a typed command into at most three *interpretations* — an action from a fixed set, a
target from the app's own lists and parameters — that the palette offers; choosing one opens the normal
confirmation or dialog. Nothing is executed here or on interpretation alone.

1. Refusals (rules): commands that ask for something outside the action set — removing deployments, deleting
   weights, user administration — are answered with a templated refusal; they never reach the model, so a
   destructive request cannot be mapped onto a nearby action (e.g. "delete qwen" onto "stop qwen").
2. Parameters (rules only, never the model): GiB to free, context length, a model ref named verbatim.
3. Targets (rules): deployments, pages and downloaded models fuzzy-matched against the text (typo tolerant),
   at most ``MAX_TARGETS`` per action; only targets the action can apply to (start: stopped / failed deployments,
   sleep: ready ones with sleep mode, ...). System deployments are never lifecycle targets.
4. Two enumerated questions to the decision model, both feature ``command``: the action (among the actions the
   role may use and that have a target), then the target for each likely action. Rules answer either question
   when it is clear (one action verb; one strong match), the model when not; with the decision model
   unavailable the rules' answer stands, possibly split over several options.
5. Confidence of an interpretation = P(action) x P(target | action). At or above threshold("command") the best one
   is returned alone; otherwise the top three (each >= ``ALT_MIN``); below ``FLOOR`` nothing ("couldn't
   interpret").

Viewers get navigation interpretations only (open a page or a deployment). The typed text goes into ``state``
only, quoted and capped; labels are templates; every rules / model answer is logged in ``decisions`` (subject:
a hash of the text, never the text), except rules-only answers to viewers (``record=False``: viewers cannot fill
the decision log).

Cost: the rules' part (``prepare``: refusal, parameters, fuzzy scores, candidates) is CPU work, so ``interpret``
runs it in a worker thread, never on the event loop, and bounds it: at most ``MAX_WORDS`` words of the text and
``MAX_SCORED`` targets are scored, and comparisons that cannot reach ``MIN_MATCH`` are skipped on cheap upper
bounds. ``interpret(..., fallback=box)`` leaves the rules-only result in ``box["rules"]`` before the model is asked,
so a caller whose deadline passes answers with it instead of starting over.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from dgxmm.decider.common import quoted_block, safe_decide, safe_record_rules, threshold_of
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Option

log = logging.getLogger(__name__)
FEATURE = "command"
MAX_TEXT = 300
MAX_TARGETS = 20
FLOOR = 0.3      # best interpretation below this: "couldn't interpret"
ALT_MIN = 0.1    # alternatives (below threshold) are listed from this confidence
TIMEOUT_S = 3.0  # per model question
MAX_WORDS = 60     # words of the text the fuzzy matcher looks at
MAX_SCORED = 400   # targets scored (deployments and pages first, then models)
GIB = 1024**3
MAX_GIB = 4096
CTX_MIN, CTX_MAX = 512, 1_048_576

# (key, description for the model, target kind or None). The complete action set: nothing else is reachable.
ACTIONS: list[tuple[str, str, str | None]] = [
    ("navigate", "Open (go to, show) a page of the app or a deployment's page; nothing changes", "any"),
    ("start", "Start a stopped or failed deployment", "deployment"),
    ("stop", "Stop a running deployment and release its memory (it can be started again later)", "deployment"),
    ("sleep", "Put a running deployment to sleep (frees its KV cache, keeps the weights for a fast wake-up)",
     "deployment"),
    ("wake", "Wake a sleeping deployment up", "deployment"),
    ("restart", "Restart a deployment (stop it, then start it again)", "deployment"),
    ("deploy", "Deploy a downloaded model as a new deployment (opens the deploy wizard)", "model"),
    ("reconfigure", "Change a deployment's settings, e.g. its context length (opens the reconfigure dialog)",
     "deployment"),
    ("free_memory", "Free an amount of memory for something else (opens a choice of plans to sleep or stop "
                    "deployments)", None),
]
ACTION_KEYS = tuple(a[0] for a in ACTIONS)
VIEWER_ACTIONS = ("navigate",)
PAST = {"navigate": "opened", "start": "started", "stop": "stopped", "sleep": "put to sleep", "wake": "woken up",
        "restart": "restarted", "deploy": "deployed", "reconfigure": "reconfigured", "free_memory": "freed"}
ACTION_QUESTION = "Which action does the user's typed command ask for?"
TARGET_QUESTION = {"deployment": "Which deployment does the user's typed command refer to?",
                   "model": "Which downloaded model does the user's typed command refer to?",
                   "any": "Which page or deployment does the user want to open?"}
# Lifecycle actions -> the deployment states they apply to (frontend logic.ts deploymentActions).
STATES = {"start": ("stopped", "failed"), "stop": ("ready", "sleeping", "queued", "starting", "loading", "unknown"),
          "sleep": ("ready",), "wake": ("sleeping",), "restart": ("ready", "sleeping", "failed", "stopped"),
          "reconfigure": ("ready", "sleeping", "stopped", "failed")}
# Pages a typed command may open (no user administration: the Users page is not a target).
PAGES: list[tuple[str, str, str, bool]] = [  # path, label, keywords, admin only
    ("/", "Dashboard", "home overview start page main", False),
    ("/deployments", "Deployments", "deployments list running models servers", False),
    ("/planner", "Planner", "planner combinations what fits together capacity", False),
    ("/processes", "Processes", "processes gpu memory usage other tenants", False),
    ("/insights", "Insights", "insights findings health proposals advice notifications", False),
    ("/models", "On disk", "models on disk downloaded weights inventory local", False),
    ("/catalog", "Catalog", "catalog hugging face hub search browse find new models", False),
    ("/downloads", "Downloads", "downloads progress queue", False),
    ("/litellm", "LiteLLM", "litellm keys gateway proxy api endpoint openai", True),
    ("/decision-model", "Decision model", "decision model decider ai thresholds feedback", True),
    ("/audit", "Audit log", "audit log history who did what", True),
]

# ---------------- rules: refusals, verbs, parameters ----------------
_REFUSE = re.compile(
    r"\b(delete|deletes|deleting|deleted|remove|removes|removing|rm|uninstall|erase|wipe|purge|destroy|nuke|drop|"
    r"get\s+rid\s+of|trash|zap|scrap|obliterate|eliminate|throw\s+away|out\s+of\s+here)\b|"
    r"\b(users?|accounts?|passwords?|roles?|permissions?|privileges?|logins?)\b|"
    r"\bmake\s+\S+\s+(an?\s+)?admin\b|\b(promote|demote|grant|revoke|sudo|chmod)\b", re.I)
REFUSAL = ("Typed commands cannot remove deployments, delete downloaded weights or manage users. Use the "
           "deployment's page, the On disk page or the Users page for that.")
VERBS: dict[str, tuple[str, ...]] = {
    "navigate": (r"open", r"go\s+to", r"goto", r"show(\s+me)?", r"view", r"navigate(\s+to)?", r"take\s+me\s+to",
                 r"display", r"look\s+at", r"jump\s+to", r"switch\s+to\s+the\s+\w+\s+page", r"page"),
    "start": (r"start", r"launch", r"boot", r"bring\s+up", r"turn\s+on", r"power\s+on", r"fire\s+up", r"run"),
    "stop": (r"stop", r"shut\s*down", r"kill", r"turn\s+off", r"halt", r"terminate", r"power\s+off", r"unload",
             r"take\s+down", r"bring\s+down"),
    "sleep": (r"sleep", r"asleep", r"pause", r"suspend", r"hibernate", r"park", r"snooze"),
    "wake": (r"wake", r"wake\s+up", r"resume", r"unpause", r"awaken", r"unsleep"),
    "restart": (r"restart", r"reboot", r"relaunch", r"bounce", r"reload", r"cycle", r"re-start", r"fresh\s+start",
                r"start\s+over", r"start\s+(it\s+)?again", r"try\s+again"),
    "deploy": (r"deploy", r"serve", r"host", r"spin\s+up", r"new\s+deployment", r"set\s+up"),
    "reconfigure": (r"reconfigure", r"configure", r"change", r"set", r"adjust", r"tweak", r"modify", r"edit",
                    r"settings", r"increase", r"decrease", r"raise", r"lower", r"bump", r"resize"),
    "free_memory": (r"free(\s+up)?", r"make\s+(room|space)", r"release", r"reclaim", r"clear\s+(some\s+)?memory",
                    r"need\s+\d+(\.\d+)?\s*(gib|gb|g)\b"),
}
_VERB = {k: re.compile(r"(?<![\w-])(" + "|".join(v) + r")(?![\w-])", re.I) for k, v in VERBS.items()}
# Verbs that name a lifecycle action unmistakably ("run", the reconfigure and navigation words are weak): a command
# using one whose action has no target ("stop the decider", "choose stop for every deployment") is not
# re-read as something else.
_STRONG = {k: re.compile(r"(?<![\w-])(" + "|".join(x for x in v if x != r"run") + r")(?![\w-])", re.I)
           for k, v in VERBS.items() if k not in ("navigate", "reconfigure")}
_GIB = re.compile(r"(?<![\w.])(\d{1,4}(?:[.,]\d{1,2})?)\s*(gib|gb|g|gigs?|gigabytes?|tib|tb)(?![\w])", re.I)
_CTX = re.compile(r"(?<![\w.])(\d{1,4}(?:\.\d)?)\s*k(?:\s*(?:tokens?|ctx|context))?(?![\w])|"
                  r"(?<![\w.])(\d{3,7})\s*(?:tokens?|ctx)(?![\w])|"
                  r"(?:context|ctx)(?:\s+(?:length|len|size|window))?\s*(?:of|to|=|:)?\s*(\d{1,7})(k?)(?![\w])", re.I)
_REF = re.compile(r"(?<![\w.-])([A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,127})(?![\w.-])")
_WORD = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:MAX_TEXT]


def text_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()[:16]


# Lookalike letters folded before the refusal check (NFKC handles full-width forms and ligatures).
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_CONFUSABLE = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "ё": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ї": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ɡ": "g",
    # Greek
    "α": "a", "ο": "o", "ε": "e", "ρ": "p", "τ": "t", "ν": "v", "ι": "i", "κ": "k", "υ": "u", "χ": "x", "η": "n",
    "μ": "m", "β": "b", "δ": "d", "γ": "y", "ϲ": "c", "ς": "s", "σ": "s", "ω": "w",
})
# Blocked verbs also caught with one typo (insertion, deletion, substitution or swapped neighbours).
FUZZY_REFUSE = ("delete", "remove", "uninstall", "destroy", "obliterate", "eliminate")
# Real words one edit away from a blocked verb, never read as it ("remote" is not "remove").
FUZZY_ALLOW = frozenset({"remote", "remotes", "remoted", "deplete", "depleted", "depletes", "depleting", "devote",
                         "relete"})


def fold(text: str) -> str:
    """NFKC, no zero-width characters, case-folded, Cyrillic / Greek lookalikes as Latin letters."""
    return unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH).casefold().translate(_CONFUSABLE)


def _one_edit(a: str, b: str) -> bool:
    """Damerau-Levenshtein distance of a and b is at most 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]]
                                  and a[diff[1]] == b[diff[0]])
    if la > lb:
        a, b = b, a
    return any(a == b[:i] + b[i + 1:] for i in range(len(b)))


def refused(text: str, names: set[str] | frozenset[str] = frozenset()) -> bool:
    """The command asks for something outside the action set (removal, deleting weights, user administration),
    also spelled with lookalike letters, zero-width characters or one typo in a blocked verb. The typo match skips
    real words that merely look alike (FUZZY_ALLOW) and words of the app's own entity names (``names``: "wake
    remote-coder" names a deployment, it does not ask to remove anything)."""
    t = fold(text)
    if _REFUSE.search(t):
        return True
    return any(len(w) >= 5 and w not in FUZZY_ALLOW and w not in names and any(_one_edit(w, b) for b in FUZZY_REFUSE)
               for w in _WORD.findall(t))


def name_words(targets) -> frozenset[str]:
    """Every word of the entities' names (deployments, models, pages)."""
    return frozenset(w for t in targets for n in t.names for w in _WORD.findall(fold(n)))


def verbs(text: str) -> list[str]:
    """Actions whose verbs appear in the text (in ACTIONS order). "run"/"start" with a model ref -> deploy."""
    return [k for k in ACTION_KEYS if _VERB[k].search(text)]


def params(text: str) -> dict:
    """Parameters by rules: ``bytes`` (an amount of memory), ``context`` (tokens), ``model_ref`` (named
    verbatim)."""
    out: dict = {}
    m = _GIB.search(text)
    if m:
        n = float(m.group(1).replace(",", "."))
        if m.group(2).lower() in ("tb", "tib"):
            n *= 1024
        if 0 < n <= MAX_GIB:
            out["bytes"] = int(n * GIB)
    for m in _CTX.finditer(text):
        if m.group(1):
            n = int(float(m.group(1)) * 1024)
        elif m.group(2):
            n = int(m.group(2))
        else:
            n = int(m.group(3)) * (1024 if m.group(4) else 1)
        if CTX_MIN <= n <= CTX_MAX:
            out["context"] = n
            break
    m = _REF.search(text)
    if m:
        out["model_ref"] = m.group(1)
    return out


# ---------------- targets ----------------
@dataclass
class Target:
    kind: str           # page | deployment | model
    key: str            # page:/planner | deployment:3 | model:<ref>
    name: str
    names: list[str]    # what the user may call it
    detail: str = ""    # for the model: state, model ...
    state: str | None = None
    sleep_capable: bool = False
    system: bool = False
    engine: str | None = None
    path: str | None = None
    ref: str | None = None
    id: int | None = None
    score: float = 0.0

    def out(self) -> dict:
        d = {"kind": self.kind, "name": self.name}
        if self.kind == "page":
            d["path"] = self.path
        elif self.kind == "deployment":
            d.update(id=self.id, state=self.state, engine=self.engine)
        else:
            d["ref"] = self.ref
        return d


def _short(ref: str) -> str:
    return ref.rstrip("/").split("/")[-1].removesuffix(".gguf")


def build_targets(views: list[dict], artifacts: list[dict], *, admin: bool) -> list[Target]:
    """The app's own entities a command may name. ``artifacts``: [{ref, engines}] of deployable models."""
    out = [Target("page", f"page:{p}", label, [label, kw], detail="page", path=p)
           for p, label, kw, admin_only in PAGES if admin or not admin_only]
    for v in views:
        names = [v["name"], v.get("alias") or "", v.get("model_ref") or "", _short(v.get("model_ref") or ""),
                 v.get("served_name") or ""]
        out.append(Target("deployment", f"deployment:{v['id']}", v["name"], [n for n in names if n],
                          detail=f"{v['state']}{', sleep mode on' if v.get('sleep_capable') else ''}, model "
                                 f"{v.get('model_ref')}", state=v["state"], sleep_capable=bool(v.get("sleep_capable")),
                          system=bool(v.get("system")) or v.get("engine") == "decision", engine=v.get("engine"),
                          path=f"/deployments/{v['id']}", id=v["id"]))
    if admin:
        for a in artifacts:
            out.append(Target("model", f"model:{a['ref']}", a["ref"], [a["ref"], _short(a["ref"])],
                              detail=f"downloaded model for {', '.join(a.get('engines') or [])}", ref=a["ref"],
                              engine=(a.get("engines") or [None])[0]))
    return out


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


GENERIC = frozenset({"instruct", "model", "models", "chat", "base", "deployment", "deployments", "page", "gguf",
                     "the", "this", "that", "with", "from", "please", "and", "who", "did", "what", "new", "other",
                     "all", "data", "list", "find", "it", "for", "one", "use", "usage", "run", "running", "local"})


_SIZE = re.compile(r"^\d+(\.\d+)?b$")  # "8b", "12b": a model size names the deployment as well as its name does


def _name_words(name: str) -> list[str]:
    """Distinctive words of a name: alphabetic parts of 3+ letters and sizes ("Qwen3-Coder-30B" -> qwen, coder,
    30b)."""
    out = []
    for w in _WORD.findall(name.lower()):
        if _SIZE.match(w):
            out.append(w)
            continue
        for part in re.findall(r"[a-z]+|\d+b", w):
            if (len(part) >= 3 and part not in GENERIC) or _SIZE.match(part):
                out.append(part)
    return out


# Words for a deployment state: "wake the sleeping one", "the crashed one".
STATE_WORDS = {"sleeping": ("sleeping", "asleep", "dormant"), "stopped": ("stopped", "down", "off"),
               "failed": ("failed", "crashed", "broken", "dead", "erroring"), "ready": ("running", "ready", "up")}


def _ratio(m: difflib.SequenceMatcher, a: str, lb: int, lo: float) -> float:
    """Similarity of ``a`` and the matcher's second sequence (length ``lb``; indexed once per name, reused for every
    window), or 0.0 as soon as a cheap upper bound shows it is below ``lo``."""
    la = len(a)
    if not la or not lb or 2.0 * min(la, lb) / (la + lb) < lo:
        return 0.0
    m.set_seq1(a)
    if m.real_quick_ratio() < lo or m.quick_ratio() < lo:
        return 0.0
    r = m.ratio()
    return r if r >= lo else 0.0


def match_score(text: str, names: list[str]) -> float:
    """How well the text names an entity (0..1): the whole name (or its squashed form) in the text -> 1; a window
    of the text's words close to the squashed name -> its similarity (typos); a distinctive word of the name in
    the text -> 0.8 (0.8 x similarity for a typo of it). Scores below ``MIN_MATCH`` only ever mean "not named",
    so they are not computed exactly (0.0 or a lower value)."""
    low = text.lower()
    sq = _squash(low)
    words = _WORD.findall(low)[:MAX_WORDS]
    uniq = list(dict.fromkeys(words))
    best = 0.0
    m = difflib.SequenceMatcher(None, "", "")
    for name in names:
        n = name.lower().strip()
        ns = _squash(n)
        if len(ns) < 2:
            continue
        if len(ns) >= 3 and (re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", low) or
                             (len(ns) >= 5 and ns in sq)):
            return 1.0
        k = len(_WORD.findall(n)) or 1
        m.set_seq2(ns)
        wins = dict.fromkeys("".join(words[i:i + size]) for size in {max(1, k - 1), k, k + 1}
                             for i in range(0, max(len(words) - size + 1, 1)))
        for win in wins:
            if win:
                best = max(best, _ratio(m, win, len(ns), max(best, MIN_MATCH)))
        for w in _name_words(n):
            if len(w) >= 4:
                m.set_seq2(w)
            for tw in uniq:
                if tw == w:
                    best = max(best, 0.8 if not _SIZE.match(w) else 0.75)
                elif len(tw) >= 4 and len(w) >= 4:
                    if sorted(tw) == sorted(w):  # swapped letters ("qwne")
                        r = max(_ratio(m, tw, len(w), 0.9), 0.9)
                    else:  # 0.8 x r matters only above the best so far and MIN_MATCH
                        r = _ratio(m, tw, len(w), max(0.7, max(best, MIN_MATCH) / 0.8))
                    if r >= 0.7:
                        best = max(best, 0.8 * r)
    return best


MIN_MATCH = 0.7
NEAR_BEST = 0.2      # an action's candidates score at least the best match of any entity minus this
CLEAR_MATCH = 0.95   # a target the rules pick without asking: score >= this ...
CLEAR_GAP = 0.25     # ... and this much above the next one


def applies(action: str, t: Target) -> bool:
    if action == "navigate":
        return t.kind in ("page", "deployment")
    if action == "deploy":
        return t.kind == "model"
    if t.kind != "deployment" or t.system:
        return False
    if action == "sleep":
        return t.state in STATES["sleep"] and t.sleep_capable
    return t.state in STATES.get(action, ())


def _state_named(text: str, t: Target) -> bool:
    words = set(_WORD.findall(text.lower()))
    return t.kind == "deployment" and bool(words & set(STATE_WORDS.get(t.state or "", ())))


def scores(text: str, targets: list[Target]) -> dict[str, float]:
    """Match score of every entity (a deployment in a state the text names, e.g. "the sleeping one", scores 0.75);
    at most ``MAX_SCORED`` entities (pages and deployments first), the rest score 0."""
    order = sorted(range(len(targets)), key=lambda i: targets[i].kind == "model")[:MAX_SCORED]
    out = dict.fromkeys((t.key for t in targets), 0.0)
    for i in order:
        t = targets[i]
        out[t.key] = max(match_score(text, t.names), 0.75 if _state_named(text, t) else 0.0)
    return out


def candidates(action: str, text: str, targets: list[Target], p: dict, sc: dict[str, float] | None = None) -> list[Target]:
    """Targets of an action matched in the text, best first, at most MAX_TARGETS. A model ref named verbatim is
    the deploy target. An entity the action cannot apply to still counts as matched: "stop the decider" names the
    (system) decider, so weaker look-alikes are not offered instead."""
    if action == "free_memory":
        return []
    if action == "deploy" and p.get("model_ref"):
        exact = [t for t in targets if t.kind == "model" and t.ref and t.ref.lower() == p["model_ref"].lower()]
        if exact:
            return [_scored(exact[0], 1.0)]
    sc = scores(text, targets) if sc is None else sc
    kinds = {"navigate": ("page", "deployment"), "deploy": ("model",)}.get(action, ("deployment",))
    best_any = max((v for t in targets if t.kind in kinds for v in [sc[t.key]]), default=0.0)
    floor = max(MIN_MATCH, best_any - NEAR_BEST)
    out = [_scored(t, sc[t.key]) for t in targets if applies(action, t) and sc[t.key] >= floor]
    out.sort(key=lambda t: (-t.score, t.kind != "deployment", t.name))
    return out[:MAX_TARGETS]


def _scored(t: Target, s: float) -> Target:
    return Target(**{**t.__dict__, "score": round(s, 3)})


# ---------------- interpretation ----------------
@dataclass
class Interpretation:
    action: str
    target: Target | None
    params: dict
    confidence: float
    source: str
    label: str = ""

    def out(self) -> dict:
        return {"action": self.action, "target": self.target.out() if self.target else None, "params": self.params,
                "confidence": round(self.confidence, 4), "label": self.label, "source": self.source}


@dataclass
class Result:
    interpretations: list[Interpretation] = field(default_factory=list)
    source: str = "none"
    message: str | None = None
    refused: bool = False

    def out(self) -> dict:
        return {"interpretations": [i.out() for i in self.interpretations], "source": self.source,
                "message": self.message, "refused": self.refused}


def label_for(action: str, t: Target | None, p: dict) -> str:
    """Deterministic text of an interpretation (never model text)."""
    gib = f"{p['bytes'] / GIB:g} GiB" if p.get("bytes") else None
    ctx = f"{p['context']:,} tokens of context" if p.get("context") else None
    name = t.name if t else ""
    if action == "navigate":
        return f"Open {name}" if t and t.kind == "page" else f"Open deployment {name}"
    if action == "free_memory":
        return f"Free {gib} (choose a plan)" if gib else "Free memory (choose a plan)"
    if action == "deploy":
        return f"Deploy {name}" + (f" with {ctx}" if ctx else "") + " (opens the wizard)"
    if action == "reconfigure":
        return f"Reconfigure {name}" + (f": {ctx}" if ctx else "") + " (opens the dialog)"
    return f"{action.capitalize()} {name}"


def _params_for(action: str, t: Target | None, p: dict) -> dict:
    if action == "free_memory":
        return {"bytes": p["bytes"]} if p.get("bytes") else {}
    if action == "deploy":
        out = {"model_ref": t.ref} if t else {}
        if p.get("context"):
            out["context"] = p["context"]
        return out
    if action == "reconfigure" and p.get("context") and t is not None:
        key = "ctx_size" if t.engine == "llamacpp" else "max_model_len"
        return {"patch": {key: p["context"]}}
    return {}


def _state(text: str, admin: bool, mentioned: list[Target], p: dict) -> str:
    role = ("admin (may open pages and change deployments)" if admin else
            "viewer (may only open pages and deployments)")
    lines = [f"- {t.kind} {t.name}: {t.detail}" for t in mentioned[:MAX_TARGETS]]
    nums = []
    if p.get("bytes"):
        nums.append(f"{p['bytes'] / GIB:g} GiB of memory")
    if p.get("context"):
        nums.append(f"a context length of {p['context']:,} tokens")
    return "\n".join([
        f"The user is signed in as {role} and typed a command into the app's command palette.",
        quoted_block("COMMAND", text, max_chars=MAX_TEXT),
        quoted_block("THINGS THE COMMAND MAY MENTION (from the app's own lists)", "\n".join(lines) or "(none)",
                     max_chars=3000),
        f"Amounts found in the command: {', '.join(nums) or 'none'}.",
    ])


class CommandService:
    def __init__(self, ctx):
        self.ctx = ctx

    def snapshot(self, *, admin: bool) -> list[Target]:
        views = self.ctx.deployer.list_views() if getattr(self.ctx, "deployer", None) is not None else []
        arts = []
        inv = getattr(self.ctx, "inventory", None)
        if admin and inv is not None:
            for a in inv.list():
                engines = [n for n, e in (self.ctx.engines or {}).items() if e.supports(a)]
                if a.complete and a.kind == "llm" and engines:
                    arts.append({"ref": a.ref, "engines": engines})
        return build_targets(views, arts, admin=admin)

    async def interpret(self, text: str, *, admin: bool, targets: list[Target] | None = None,
                        fallback: dict | None = None) -> Result:
        if targets is None:
            targets = await asyncio.to_thread(self.snapshot, admin=admin)
        return await interpret(text, targets, decider_of(self.ctx), admin=admin, record=admin, fallback=fallback)


async def interpret(text: str, targets: list[Target], decider, *, admin: bool, record: bool = True,
                    fallback: dict | None = None) -> Result:
    """The interpretations the palette shows (see the module docstring). ``record=False``: rules-only answers are
    not logged (viewer requests). ``fallback``: receives ``"rules"``, the rules-only result, before the model is
    asked."""
    prep = await asyncio.to_thread(prepare, text, targets, admin=admin)
    if prep.early is not None:
        if prep.refusal and record:
            await asyncio.to_thread(safe_record_rules, decider, feature=FEATURE, subject=prep.subject,
                                    choice="refused", reason="outside the action set")
        if fallback is not None:
            fallback["rules"] = prep.early
        return prep.early
    from dgxmm.decider.protocol import NullDecider
    rules_only = choose(await _decide(prep, NullDecider(), admin=admin, record=False), 0.7)
    if fallback is not None:
        fallback["rules"] = rules_only
    ranked = await _decide(prep, decider, admin=admin, record=record)
    return choose(ranked, threshold_of(decider, FEATURE, 0.7))


def choose(ranked: list[Interpretation], thr: float) -> Result:
    """Threshold / top three / couldn't interpret over the ranked interpretations."""
    source = "model" if any(i.source == "model" for i in ranked) else "rules"
    if not ranked or ranked[0].confidence < FLOOR:
        return Result(source=source, message="Couldn't interpret that. Try the palette's search, or say what to do "
                                             "and to which deployment, page or model.")
    if ranked[0].confidence >= thr:
        return Result([ranked[0]], source=source)
    return Result([i for i in ranked if i.confidence >= ALT_MIN][:3], source=source)


@dataclass
class Prepared:
    """The rules' part of an interpretation (CPU work, run in a thread): an early result, or what the questions
    need."""
    text: str
    subject: str
    early: Result | None = None
    refusal: bool = False
    p: dict = field(default_factory=dict)
    found: list = field(default_factory=list)
    usable: list = field(default_factory=list)
    cands: dict = field(default_factory=dict)
    state: str = ""


def prepare(text: str, targets: list[Target], *, admin: bool) -> Prepared:
    """Refusal, parameters, fuzzy scores and candidates (no decider, no database): bounded CPU work."""
    text = normalise(text)
    subject = f"command:{text_hash(text)}"
    if not text:
        return Prepared(text, subject, early=Result(message="Type a command."))
    if refused(text, name_words(targets)):
        return Prepared(text, subject, early=Result(source="rules", message=REFUSAL, refused=True), refusal=True)
    p = params(text)
    allowed = ACTION_KEYS if admin else VIEWER_ACTIONS
    found = verbs(text)
    sc = scores(text, targets)
    cands = {a: candidates(a, text, targets, p, sc) for a in allowed}
    usable = [a for a in allowed if cands[a] or (a == "free_memory" and ("free_memory" in found or p.get("bytes")))]
    named = any(t.kind == "deployment" and sc[t.key] >= 0.9 for t in targets)
    strong = [a for a, rx in _STRONG.items() if rx.search(text)]
    if admin and found and not any(a in usable for a in found) and (named or strong):
        a = (strong or found)[0]
        return Prepared(text, subject, early=Result(source="rules", message=f"Couldn't interpret that: nothing it "
                                                    f"names can be {PAST.get(a, a)} right now."))
    if not usable:
        return Prepared(text, subject, early=Result(message="Couldn't interpret that: no page, deployment or model "
                                                            "it names."))
    mentioned = sorted({t.key: t for a in usable for t in cands[a]}.values(), key=lambda t: -t.score)
    return Prepared(text, subject, p=p, found=found, usable=usable, cands=cands,
                    state=_state(text, admin, mentioned, p))


async def rank(text: str, targets: list[Target], decider, *, admin: bool,
               record: bool = True) -> tuple[list[Interpretation], Result | None]:
    """Every candidate interpretation, most likely first, or an early result (empty text, refusal, nothing
    named)."""
    prep = await asyncio.to_thread(prepare, text, targets, admin=admin)
    if prep.early is not None:
        if prep.refusal and record:
            await asyncio.to_thread(safe_record_rules, decider, feature=FEATURE, subject=prep.subject,
                                    choice="refused", reason="outside the action set")
        return [], prep.early
    return await _decide(prep, decider, admin=admin, record=record), None


async def _decide(prep: Prepared, decider, *, admin: bool, record: bool) -> list[Interpretation]:
    """The two questions (action, then target) over a prepared command; rules answer when they can."""
    subject, state, usable, found, cands, p = prep.subject, prep.state, prep.usable, prep.found, prep.cands, prep.p
    sources: set[str] = set()

    # ---- the action ----
    said = [a for a in found if a in usable]
    if not admin:
        p_action, src = {"navigate": 1.0}, "rules"  # the only action a viewer has
    elif len(said) == 1:
        p_action, src = {said[0]: 1.0}, "rules"
    elif len(usable) == 1:
        p_action, src = {usable[0]: 1.0}, "rules"
    else:
        opts = [Option(k, k, d) for k, d, _ in ACTIONS if k in usable]
        d = await safe_decide(decider, feature=FEATURE, subject=subject, state=state, question=ACTION_QUESTION,
                              options=opts, threshold=0.0, timeout_s=TIMEOUT_S)
        if d.probabilities and d.choice is not None:
            p_action, src = {k: v for k, v in d.probabilities.items() if k in usable}, "model"
        else:  # decision model unavailable: the verbs said, else every usable action equally
            pool = said or usable
            p_action, src = {a: 1.0 / len(pool) for a in pool}, "rules"
    if src == "rules" and len(p_action) == 1 and record:
        a = next(iter(p_action))
        await asyncio.to_thread(safe_record_rules, decider, feature=FEATURE, subject=subject, choice=f"action:{a}",
                                reason="verb")
    sources.add(src)

    # ---- the target of each likely action ----
    out: list[Interpretation] = []
    for action, pa in sorted(p_action.items(), key=lambda kv: -kv[1])[:2]:
        if pa < ALT_MIN:
            continue
        if action == "free_memory":
            out.append(Interpretation(action, None, _params_for(action, None, p), pa, src))
            continue
        ts = cands[action]
        dist, tsrc = await _target_dist(decider, subject, state, action, ts, record=record)
        sources.add(tsrc)
        for t, pt in dist:
            out.append(Interpretation(action, t, _params_for(action, t, p), pa * pt,
                                      "model" if "model" in (src, tsrc) else "rules"))
    out.sort(key=lambda i: -i.confidence)
    for i in out:
        i.label = label_for(i.action, i.target, i.params)
    return out


async def _target_dist(decider, subject: str, state: str, action: str, ts: list[Target], *,
                       record: bool = True) -> tuple[list, str]:
    """(target, P(target | action)) best first, and the source."""
    if len(ts) == 1:
        return [(ts[0], 1.0)], "rules"
    if ts[0].score >= CLEAR_MATCH and ts[0].score - ts[1].score >= CLEAR_GAP:
        if record:
            await asyncio.to_thread(safe_record_rules, decider, feature=FEATURE, subject=subject,
                                    choice=f"target:{ts[0].key}"[:256], reason="clear match")
        return [(ts[0], 1.0)], "rules"
    kind = "any" if action == "navigate" else ts[0].kind
    opts = [Option(f"t{i + 1}", t.name, f"{t.kind} {t.name} ({t.detail})"[:300]) for i, t in enumerate(ts)]
    d = await safe_decide(decider, feature=FEATURE, subject=subject,
                          state=state + f"\nAction: {action}.", question=TARGET_QUESTION[kind], options=opts,
                          threshold=0.0, timeout_s=TIMEOUT_S)
    if d.probabilities and d.choice is not None:
        by = {o.key: t for o, t in zip(opts, ts)}
        return sorted(((by[k], v) for k, v in d.probabilities.items() if k in by), key=lambda kv: -kv[1]), "model"
    # unavailable: the fuzzy scores, normalised over the matches that score like the best one
    top = [t for t in ts if t.score >= ts[0].score - 0.05]
    return [(t, 1.0 / len(top)) for t in top], "rules"


# ---------------- evaluation adapter ----------------
def eval_case(case: dict):
    """The action question of a corpus case (the full two-question pipeline is scripts/decider_eval/command_eval.py).
    ``inventory``: {"deployments": [...views], "models": [{ref, engines}]}; ``admin`` (default true)."""
    from tools.eval_feature import EvalCase  # noqa: PLC0415
    admin = case.get("admin", True)
    inv = case.get("inventory") or {}
    targets = build_targets(inv.get("deployments", []), inv.get("models", []), admin=admin)
    text = normalise(case["text"])
    if refused(text, name_words(targets)):
        return EvalCase(rules="none", state="", question=ACTION_QUESTION, options=[])
    p = params(text)
    allowed = ACTION_KEYS if admin else VIEWER_ACTIONS
    found = verbs(text)
    usable = [a for a in allowed if candidates(a, text, targets, p) or
              (a == "free_memory" and ("free_memory" in found or p.get("bytes")))]
    said = [a for a in found if a in usable]
    mentioned = sorted({t.key: t for a in usable for t in candidates(a, text, targets, p)}.values(),
                       key=lambda t: -t.score)
    rules = said[0] if len(said) == 1 else (usable[0] if len(usable) == 1 else None)
    return EvalCase(rules=rules, state=_state(text, admin, mentioned, p), question=ACTION_QUESTION,
                    options=[Option(k, k, d) for k, d, _ in ACTIONS if k in usable])
