"""Control-plane availability: every regex DGXMM applies to untrusted text (container logs, chat templates, repo
names, config.json values, process command lines) finishes quickly on adversarial input, and the diagnosis never
matches on the event loop (docs/security.md, "Untrusted text and the event loop").

Each check runs in a forked child with a hard deadline, so a catastrophic backtrack fails the test instead of
hanging the suite."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import re
import time

import pytest

from dgxmm.decider import common, config_suggest, diagnosis, kind, workload
from dgxmm.deployer import logtail
from dgxmm.inventory import kinds
from dgxmm.memory import redact

LIMIT_S = 0.5
SIZE = 200 * 1024
MODULES = [diagnosis, config_suggest, common, kind, workload, logtail, kinds, redact]


def _patterns():
    out = []
    for mod in MODULES:
        for name, v in sorted(vars(mod).items()):
            items = v if isinstance(v, (list, tuple)) else [v]
            for i, item in enumerate(items):
                p = item[1] if isinstance(item, tuple) and len(item) == 2 else item
                if isinstance(p, re.Pattern):
                    out.append((f"{mod.__name__}.{name}" + (f"[{i}]" if p is not v else ""), p))
    return out


PATTERNS = _patterns()


def _fill(unit: str, size: int = SIZE) -> str:
    return (unit * (size // max(1, len(unit)) + 1))[:size]


GENERIC = ["a", " ", "\t", "<", ">", "-", "--a ", "=", "a=", "a:", ":", "/", ".", "_", "9", "'", '"', "'a',",
           "[", "(", "[a]", "{", "\\", "e", "Error", "ERROR ", "a b ", "x\n", "+ ", "AKIA", "hf_", "ey", "==",
           "A" * 64 + "\n", "Bearer ", "key=", "--key ", "(EngineCore pid=1) ", "   "]


def _adversarial(p: re.Pattern) -> list[str]:
    words = list(dict.fromkeys(re.findall(r"[A-Za-z_]{3,}", p.pattern)))[:24]
    out = [_fill(u) for u in GENERIC]
    # literal runs of the pattern ("tool call parser", "-----BEGIN ") repeated
    phrases = [ph for ph in re.findall(r"[^\\()\[\]{}|?*+^$]{6,}", p.pattern) if ph.strip()]
    out += [_fill(ph + " ") for ph in list(dict.fromkeys(phrases))[:24]]
    out += [_fill(w + " ") for w in words]
    if words:
        out.append(_fill(" ".join(words) + " "))
        # a line that starts with one literal of the pattern and repeats another (".* X .*" shapes backtrack here)
        out += [w + " a " + _fill(v + " ") for w in words[:16] for v in words[:16]]
    return out


def _run_with_deadline(fn, deadline_s: float):
    """fn() in a forked child; returns its result, or fails the test when the child misses the deadline."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def child():
        try:
            q.put(("ok", fn()))
        except BaseException as e:  # noqa: BLE001 - report anything to the parent
            q.put(("err", repr(e)))

    p = ctx.Process(target=child)
    p.start()
    p.join(deadline_s)
    if p.is_alive():
        p.kill()
        p.join()
        pytest.fail(f"did not finish within {deadline_s:.0f} s (catastrophic backtracking?)")
    status, value = q.get(timeout=5)
    assert status == "ok", value
    return value


@pytest.mark.parametrize("name,pattern", PATTERNS, ids=[n for n, _ in PATTERNS])
def test_every_module_regex_is_fast_on_adversarial_200k_input(name, pattern):
    inputs = _adversarial(pattern)

    def measure():
        slow = []
        for s in inputs:
            t = time.perf_counter()
            pattern.search(s)
            dt = time.perf_counter() - t
            if dt >= LIMIT_S:
                slow.append((s[:40], round(dt, 2)))
        return slow

    slow = _run_with_deadline(measure, LIMIT_S * len(inputs) + 10)
    assert slow == [], f"{name}: {slow}"


# The shapes the security review measured (cubic / quadratic before the fix), each one 200 KB line, and the same
# inside a 1 MB log of many long lines.
ECHO_ARCH = "Model architectures ['Initializing a " + _fill("engine ", SIZE) + "'] are not supported"
PARSER_REPEAT = _fill("tool call parser ", SIZE)
LONG_LOG = "\n".join([ECHO_ARCH[:SIZE], PARSER_REPEAT, _fill("Error: x " * 3, SIZE), _fill("a", SIZE),
                      _fill("--api-key=", SIZE)])
ADVERSARIAL_LOGS = [ECHO_ARCH, PARSER_REPEAT, LONG_LOG, _fill("Initializing a engine ", SIZE),
                    _fill("non-default args: ", SIZE), _fill("estimated maximum model length is 1", SIZE)]
CFG = {"max_model_len": 8192, "kv_cache_dtype": "auto", "ctx_size": 8192}


def _diagnosis_pipeline(text: str) -> None:
    diagnosis.rule_hits(text, 1, "vllm")
    diagnosis.rule_hits(text, 3, "decision")
    for cause in diagnosis.CAUSES:
        for engine in ("vllm", "llamacpp"):
            diagnosis.config_patch(cause, engine, CFG, text, ("hermes", "qwen3"))
    diagnosis.error_lines(text)
    diagnosis.log_excerpt(text)
    diagnosis.build_state(engine="vllm", model_ref="a/b", exit_code=1, message=text[:1000], hits=[], log_text=text)
    logtail.stored_tail(text)


@pytest.mark.parametrize("i", range(len(ADVERSARIAL_LOGS)))
def test_diagnosis_and_stored_tail_are_fast_on_adversarial_logs(i):
    text = ADVERSARIAL_LOGS[i]

    def measure():
        t = time.perf_counter()
        _diagnosis_pipeline(text)
        return time.perf_counter() - t

    # the whole pipeline (2 rule passes, 30 patch derivations, excerpt, state, condense + redact) on this input
    assert _run_with_deadline(measure, 30) < 10 * LIMIT_S


def test_rules_still_fire_on_long_lines_after_the_cap():
    """The per-line cap keeps the start of a line (where engines put the error), so rules still match."""
    line = "OSError: [Errno 28] No space left on device " + "x" * 50_000
    assert [c for c, _ in diagnosis.rule_hits(line)] == ["disk_full"]
    echo = "INFO Initializing a V1 LLM engine (v0.26.1) with config: model='m', " + "x" * 50_000
    assert diagnosis.rule_hits(echo + "\nOSError: [Errno 28] No space left on device") == [
        ("disk_full", "OSError: [Errno 28] No space left on device")]
    assert len(diagnosis.evidence("y" * 1_000_000)) <= diagnosis.MAX_LINE_CHARS
    assert len(diagnosis.evidence("\n".join(["z" * 100] * 100_000))) <= diagnosis.MAX_TEXT_CHARS


def test_stored_tail_caps_line_length():
    text = "\n".join(["Error: " + "q" * 100_000] * 300)
    out = logtail.stored_tail(text)
    assert max(len(ln) for ln in out.splitlines()) <= logtail.MAX_LINE_CHARS + 16


@pytest.mark.parametrize("text", [_fill("the answer is ", SIZE), _fill("parser ", SIZE), _fill("<<", SIZE),
                                  _fill("answer qwen", SIZE), _fill("tool function think ", SIZE)])
def test_parser_text_checks_are_fast(text):
    def measure():
        t = time.perf_counter()
        config_suggest.injection_shaped(text, text[:300])
        config_suggest.template_excerpt(text)
        kinds.classify_kind(text)
        kinds.strong_kind(text[:300], text, {"architectures": [text]}, ["config.json"])
        return time.perf_counter() - t
    assert _run_with_deadline(measure, 30) < 4 * LIMIT_S


async def test_diagnosis_matches_off_the_event_loop(stack, monkeypatch):
    from tests.fakes import FakeDecider
    from tests.test_decider_diagnosis import LOGS, fail_with, make_ctx
    on_loop: dict[str, bool] = {}

    def spy(name, fn):
        def wrapped(*a, **kw):
            try:
                asyncio.get_running_loop()
                on_loop[name] = True
            except RuntimeError:
                on_loop.setdefault(name, False)
            return fn(*a, **kw)
        monkeypatch.setattr(diagnosis, name, wrapped)

    for name in ("rule_hits", "config_patch", "log_excerpt"):
        spy(name, getattr(diagnosis, name))
    ctx = make_ctx(stack, FakeDecider())
    dep_id = await fail_with(stack, ctx, "Loading...\n" + LOGS["trust_remote_code_required"])
    assert ctx.diagnosis.latest(dep_id)["cause"] == "trust_remote_code_required"
    assert on_loop == {"rule_hits": False, "config_patch": False, "log_excerpt": False}


async def test_parser_suggestion_regexes_run_off_the_event_loop(monkeypatch):
    from dgxmm.decider.config_suggest import ModelInfo
    from tests.fakes import FakeDecider
    on_loop: dict[str, bool] = {}

    def spy(name):
        fn = getattr(config_suggest, name)

        def wrapped(*a, **kw):
            try:
                asyncio.get_running_loop()
                on_loop[name] = True
            except RuntimeError:
                on_loop.setdefault(name, False)
            return fn(*a, **kw)
        monkeypatch.setattr(config_suggest, name, wrapped)

    for name in ("rule_tool", "rule_reasoning", "state_for", "injection_shaped"):
        spy(name)
    fake = FakeDecider(answers={"parser_tool": ("hermes", 0.99), "parser_reasoning": ("qwen3", 0.99)})
    await config_suggest.suggest_config(fake, ModelInfo("org/plain-model", "FooForCausalLM", template=None), "vllm")
    assert on_loop == {"rule_tool": False, "rule_reasoning": False, "state_for": False, "injection_shaped": False}
