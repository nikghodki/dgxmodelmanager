"""Feature 2 — tool-call / reasoning parser suggestions for the deploy wizard.

Rules first (chat-template markers, then model-family name/architecture lookups), then the decision model with
per-option format descriptions, then no suggestion. Untrusted inputs (model name, chat template) only go into the
decider `state` as quoted, redacted, length-capped blocks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from dgxmm.decider.common import quoted_block, safe_decide, safe_record_rules, threshold_of
from dgxmm.decider.protocol import decider_of
from dgxmm.decider.types import Decision, Option
from dgxmm.engines.vllm import PARSER_RE

log = logging.getLogger(__name__)

# Parser registries of the pinned vLLM image
# vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8 (vLLM 0.26.1rc1.dev1219),
# listed with ToolParserManager.lazy_parsers / ReasoningParserManager.lazy_parsers on 2026-10-16.
# Descriptions are one-line format summaries sent to the decision model as option descriptions.
TOOL_PARSERS: dict[str, str] = {
    "apertus": "Apertus (Swiss AI) tool-call tokens",
    "cohere_command3": "Cohere Command A / R7B: <|START_ACTION|> JSON list of tool calls",
    "cohere_command4": "Cohere Command 4 action blocks",
    "deepseek_v3": "DeepSeek-V3 / R1-0528: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>name + ```json args",
    "deepseek_v31": "DeepSeek-V3.1: <｜tool▁call▁begin｜>name<｜tool▁sep｜>JSON args (no code fence)",
    "deepseek_v32": "DeepSeek-V3.2 DSML tool-call blocks",
    "deepseek_v4": "DeepSeek-V4 tool-call blocks",
    "dots": "dots.llm (rednote) tool calls",
    "ernie45": "Baidu ERNIE 4.5: <tool_call> JSON blocks",
    "functiongemma": "FunctionGemma: <start_function_call>call:name{args}<end_function_call>",
    "gemma4": "Gemma 4: <|tool_call>call:name{...}<tool_call|>",
    "gigachat3": "GigaChat 3 function-call blocks",
    "glm45": "GLM-4.5 / 4.6: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>",
    "glm47": "GLM-4.7: <tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value> (4.7 variant)",
    "granite": "IBM Granite 3.x: <|tool_call|> JSON list",
    "granite-20b-fc": "IBM granite-20b-functioncalling: <function_call> JSON",
    "granite4": "IBM Granite 4: <tool_call> JSON blocks",
    "hermes": "Hermes / Qwen2.5 / Qwen3 style: <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> JSON in XML tags",
    "hunyuan_a13b": "Tencent Hunyuan-A13B: <tool_calls> JSON list",
    "hy_v3": "Tencent Hunyuan V3 tool calls",
    "inkling": "Inkling tool-call format",
    "internlm": "InternLM2: <|action_start|><|plugin|> JSON",
    "jamba": "AI21 Jamba: <tool_calls> JSON list",
    "kimi_k2": "Kimi K2: <|tool_calls_section_begin|><|tool_call_begin|>id<|tool_call_argument_begin|>JSON<|tool_call_end|>",
    "kimi_k3": "Kimi K3 tool-call sections",
    "lfm2": "Liquid LFM2: <|tool_call_start|>[name(args)]<|tool_call_end|>",
    "ling3": "inclusionAI Ling 3 tool calls",
    "llama3_json": "Llama 3.1 / 3.2 / 3.3: bare JSON {\"name\": ..., \"parameters\": {...}} (optionally <|python_tag|>)",
    "llama4_json": "Llama 4 JSON tool calls (same format as llama3_json)",
    "llama4_pythonic": "Llama 4: pythonic list [func(arg=value), ...]",
    "longcat": "Meituan LongCat-Flash: <longcat_tool_call> JSON",
    "mimo": "Xiaomi MiMo: Qwen3-style XML <tool_call><function=..><parameter=..>",
    "minicpm5": "MiniCPM 5 XML tool calls",
    "minimax_m2": "MiniMax-M2: <minimax:tool_call><invoke name=..><parameter name=..>",
    "minimax_m3": "MiniMax-M3 invoke blocks",
    "mistral": "Mistral / Mixtral / Devstral / Magistral: [TOOL_CALLS] name[ARGS]{json} or [TOOL_CALLS][{...}]",
    "muse_glimmer": "Muse Glimmer tool calls",
    "olmo3": "OLMo 3: pythonic <function_calls> list",
    "openai": "gpt-oss (OpenAI harmony): <|start|>assistant to=functions.name<|channel|>commentary json<|message|>",
    "phi4_mini_json": "Phi-4-mini: functools[{...}] JSON list",
    "poolside_v1": "Poolside v1 tool calls",
    "pythonic": "Generic pythonic list of calls [func(arg=value)] (Llama 3.2 pythonic, ToolACE)",
    "qwen3_coder": "Qwen3-Coder / Qwen3.5: <tool_call><function=name><parameter=k>v</parameter></function></tool_call> XML",
    "qwen3_xml": "Qwen3 XML tool calls (same XML format as qwen3_coder)",
    "seed_oss": "ByteDance Seed-OSS: <seed:tool_call><function=..><parameter=..>",
    "step3": "StepFun Step-3: <|tool_calls_begin|> blocks",
    "step3p5": "StepFun Step-3.5 tool calls",
    "xlam": "Salesforce xLAM: JSON list of {name, arguments}",
}
REASONING_PARSERS: dict[str, str] = {
    "cohere_command3": "Cohere Command A reasoning: <|START_THINKING|>...<|END_THINKING|>",
    "cohere_command4": "Cohere Command 4 thinking blocks",
    "deepseek_r1": "DeepSeek-R1 and R1 distills: <think>...</think> (generation prompt opens <think>)",
    "deepseek_v3": "DeepSeek-V3.1+ hybrid thinking (<think> toggled by the chat template)",
    "deepseek_v4": "DeepSeek-V4 thinking",
    "ernie45": "ERNIE 4.5 thinking: <think>...</think>",
    "gemma4": "Gemma 4 thinking: <|think|> / <|channel>thought ... <channel|>",
    "glm45": "GLM-4.5 / 4.6 thinking: <think>...</think> (enable_thinking)",
    "glm47": "GLM-4.7 thinking: <think>...</think>",
    "granite": "IBM Granite 3.2+: 'Here is my thought process:' / 'Here is my response:'",
    "holo2": "Holo2: DeepSeek-V3 style thinking",
    "hunyuan_a13b": "Hunyuan-A13B: <think>...</think><answer>...</answer>",
    "hy_v3": "Hunyuan V3 thinking",
    "inkling": "Inkling thinking",
    "kimi_k2": "Kimi K2 Thinking: <think>...</think>",
    "kimi_k3": "Kimi K3 thinking",
    "ling3": "Ling 3 thinking",
    "mimo": "Xiaomi MiMo: <think>...</think>",
    "minimax_m2": "MiniMax-M2: <think>...</think> (kept in content)",
    "minimax_m2_append_think": "MiniMax-M2 with the opening <think> appended to the output",
    "minimax_m3": "MiniMax-M3 thinking",
    "mistral": "Magistral: [THINK]...[/THINK]",
    "muse_glimmer": "Muse Glimmer thinking",
    "nemotron_v3": "NVIDIA Nemotron 3 / Nano reasoning (<think>, /think /no_think)",
    "olmo3": "OLMo 3 Think: <think>...</think>",
    "openai_gptoss": "gpt-oss (harmony): analysis channel <|channel|>analysis ... final channel",
    "poolside_v1": "Poolside v1 thinking",
    "qwen3": "Qwen3 thinking models / Qwen3.5: <think>...</think> (enable_thinking)",
    "seed_oss": "Seed-OSS: <seed:think>...</seed:think>",
    "step3": "Step-3 thinking",
    "step3p5": "Step-3.5 thinking",
}
NONE = "none"
NONE_TOOL = "The model has no tool-call format (base model or no tools in its chat template)"
NONE_REASONING = "The model does not emit separate reasoning / thinking content"
NEVER = 1.01  # threshold used when the real one is unavailable: nothing auto-applies
SUGGEST_MIN = 0.5  # model answers below this are not shown at all; below threshold("parser") they are not auto-applied
TEMPLATE_READ_MAX = 1024 * 1024        # chat_template.jinja / chat_template.json
TOKENIZER_CONFIG_READ_MAX = 8 * 1024**2  # tokenizer_config.json can embed large added-token tables
TEMPLATE_EXCERPT_CHARS = 2500

# Deadline per parser question. Both run on the serial decider; on a busy GPU they take ~1.2 s + 0.8 s back to back
# (docs/decider-evaluation.md), and the wizard fetches suggestions in the background, so 4 s, not the default 2 s.
PARSER_TIMEOUT_S = 4.0
QUESTION_TOOL = "Which vLLM --tool-call-parser matches this model's tool-call format? (none = the model has no tool-call format)"
QUESTION_REASONING = ("Which vLLM --reasoning-parser matches how this model emits its reasoning? "
                      "(none = the model does not emit separate reasoning)")


def valid_parser(name: str | None) -> bool:
    return name is None or name == NONE or bool(re.fullmatch(PARSER_RE, name))


def tool_options() -> list[Option]:
    """Valid parser names only (granite-20b-fc has a hyphen the config model rejects)."""
    out = [Option(k, k, d) for k, d in TOOL_PARSERS.items() if valid_parser(k)]
    return out + [Option(NONE, "none", NONE_TOOL)]


def reasoning_options() -> list[Option]:
    out = [Option(k, k, d) for k, d in REASONING_PARSERS.items() if valid_parser(k)]
    return out + [Option(NONE, "none", NONE_REASONING)]


@dataclass
class ModelInfo:
    ref: str
    architecture: str | None = None
    model_type: str | None = None
    template: str | None = None  # chat template text (untrusted)


@dataclass
class RuleHit:
    value: str  # parser name or "none"
    reason: str
    strong: bool = True  # chat-template evidence; False = name/family lookup only (suggested, never auto-applied)


def _template_from(snapshot: Path, files: list[str]) -> str | None:
    def read(rel: str, cap: int = TEMPLATE_READ_MAX) -> str | None:
        try:
            p = snapshot / rel
            with open(p, "rb") as f:
                data = f.read(cap + 1)
            if len(data) > cap:
                return None  # truncated JSON / template would be misleading
            return data.decode("utf-8", "replace")
        except OSError:
            return None
    if "chat_template.jinja" in files:
        t = read("chat_template.jinja")
        if t:
            return t
    if "chat_template.json" in files:
        try:
            t = json.loads(read("chat_template.json") or "{}").get("chat_template")
            if isinstance(t, str):
                return t
        except (ValueError, AttributeError):
            pass
    if "tokenizer_config.json" in files:
        try:
            t = json.loads(read("tokenizer_config.json", TOKENIZER_CONFIG_READ_MAX) or "{}").get("chat_template")
        except ValueError:
            return None
        if isinstance(t, str):
            return t
        if isinstance(t, list):  # [{"name": "default", "template": ...}, {"name": "tool_use", ...}]
            return "\n".join(x.get("template", "") for x in t if isinstance(x, dict))
    return None


def model_info(artifact) -> ModelInfo:
    cfg = artifact.config or {}
    tmpl = None
    if artifact.source == "hf" and artifact.local_path:
        tmpl = _template_from(Path(artifact.local_path), list(artifact.files or []))
    mt = cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else None
    return ModelInfo(ref=artifact.ref, architecture=artifact.architecture, model_type=mt, template=tmpl)


def _name(info: ModelInfo) -> str:
    return info.ref.rstrip("/").split("/")[-1].lower() if info.ref else ""


def _has(t: str, *needles: str) -> bool:
    return all(n in t for n in needles)


# Name / family lookups: (substring of the repo name, parser). First match wins, so specific names come first.
_TOOL_NAMES = [("qwen3-coder", "qwen3_coder"), ("qwen3.5", "qwen3_coder"), ("qwen3_5", "qwen3_coder"),
               ("qwen3", "hermes"), ("qwen2.5", "hermes"), ("qwq", "hermes"), ("hermes", "hermes"),
               ("llama-4", "llama4_pythonic"), ("llama4", "llama4_pythonic"), ("llama-3.", "llama3_json"),
               ("llama3.", "llama3_json"), ("devstral", "mistral"), ("magistral", "mistral"), ("mistral", "mistral"),
               ("mixtral", "mistral"), ("gpt-oss", "openai"), ("glm-4.7", "glm47"), ("glm-4.6", "glm45"),
               ("glm-4.5", "glm45"), ("gemma-4", "gemma4"), ("functiongemma", "functiongemma"),
               ("deepseek-v3.1", "deepseek_v31"), ("deepseek-v3.2", "deepseek_v32"), ("kimi-k2", "kimi_k2"),
               ("granite-4", "granite4"), ("minimax-m2", "minimax_m2"), ("seed-oss", "seed_oss"),
               ("hunyuan-a13b", "hunyuan_a13b"), ("ernie-4.5", "ernie45"), ("phi-4-mini", "phi4_mini_json")]
_REASONING_NAMES = [("qwen3-coder", NONE), ("instruct-2507", NONE), ("qwen3.5", "qwen3"),
                    ("thinking-2507", "qwen3"), ("qwq", "deepseek_r1"), ("deepseek-r1", "deepseek_r1"),
                    ("deepseek-v3.1", "deepseek_v3"), ("deepseek-v3.2", "deepseek_v3"),
                    ("gpt-oss", "openai_gptoss"), ("glm-4.7", "glm47"), ("glm-4.6", "glm45"), ("glm-4.5", "glm45"),
                    ("gemma-4", "gemma4"), ("magistral", "mistral"), ("devstral", NONE), ("mistral-small", NONE),
                    ("minimax-m2", "minimax_m2"), ("kimi-k2-thinking", "kimi_k2"), ("seed-oss", "seed_oss"),
                    ("olmo-3", "olmo3")]
# Families that reason although their chat template carries no thinking marker (the model opens <think> itself, or
# the format lives in the tokenizer, not the template): for these "no marker in the template" is NOT evidence of
# "no reasoning". Measured on the real corpus (docs/decider-evaluation.md): e.g. DeepSeek-R1-0528's template has
# no <think>, Qwen3-Next-Instruct's has none and really does not reason.
_MARKERLESS_REASONING = ("deepseek-r1", "qwq", "magistral", "kimi-k2-thinking", "minimax-m2", "seed-oss", "olmo-3")
_THINK_MARKERS = ("<think>", "think>", "enable_thinking", "<|channel|>analysis", "<|think|>", "<|channel>thought",
                  "[THINK]", "reasoning_content", "<seed:think>", "START_THINKING")


def rule_tool(info: ModelInfo) -> RuleHit | None:
    t, n = info.template or "", _name(info)
    if t:
        if _has(t, "<function=", "<parameter="):
            if "<seed:tool_call>" in t:
                return RuleHit("seed_oss", "chat template uses <seed:tool_call> XML")
            return RuleHit("qwen3_coder", "chat template uses <function=…><parameter=…> XML tool calls")
        if "<|tool_calls_section_begin|>" in t:
            return RuleHit("kimi_k2", "chat template uses Kimi K2 tool-call sections")
        if _has(t, "<arg_key>", "<arg_value>"):
            return RuleHit("glm47" if "glm-4.7" in n else "glm45", "chat template uses GLM <arg_key>/<arg_value> tool calls")
        if "<|tool_call>" in t:
            return RuleHit("gemma4", "chat template uses Gemma 4 <|tool_call> blocks")
        if "<minimax:tool_call>" in t:
            return RuleHit("minimax_m2", "chat template uses <minimax:tool_call>")
        if "[TOOL_CALLS]" in t:
            return RuleHit("mistral", "chat template uses Mistral [TOOL_CALLS]")
        if "<|channel|>" in t and ("to=functions." in t or "to=\" + \"functions." in t or "<|call|>" in t):
            return RuleHit("openai", "chat template uses the gpt-oss harmony format")
        if "tool▁call" in t or ("tool_call_begin" in t and "deepseek" in n):
            return RuleHit("deepseek_v31" if "v3.1" in n else "deepseek_v3", "chat template uses DeepSeek tool-call tokens")
        if "<tool_call>" in t and ('"name"' in t or "'name'" in t or "tool_call.name" in t or "function-name" in t):
            return RuleHit("hermes", "chat template uses <tool_call> JSON (Hermes style)")
        if "<|tool_call_start|>" in t:
            return RuleHit("lfm2", "chat template uses LFM2 <|tool_call_start|>[call(...)] blocks")
        if "<|tool_call|>" in t:
            return RuleHit("granite", "chat template uses IBM Granite 3 <|tool_call|> JSON")
        if '<function name="' in t and '<param name="' in t:
            return RuleHit("minicpm5", 'chat template uses MiniCPM5 <function name="..."><param name="..."> XML')
        if "<|tool|>" in t and "<|/tool|>" in t:
            return RuleHit("phi4_mini_json", "chat template uses Phi-4-mini <|tool|> definitions")
        if "<|START_ACTION|>" in t:  # Command A generation 3 vs 4 is not visible in the template: suggest only
            return RuleHit("cohere_command3", "chat template uses Cohere <|START_ACTION|> blocks", strong=False)
        if "<|python_tag|>" in t or ('"parameters"' in t and "function call" in t.lower()):
            return RuleHit("llama3_json", "chat template uses Llama 3 JSON tool calls")
        if "func_name1(params_name1=" in t or "func_name1(param1=" in t:
            return RuleHit("llama4_pythonic" if ("llama-4" in n or "llama4" in n) else "pythonic",
                           "chat template uses pythonic [func(arg=value)] calls")
        if not re.search(r"tool|function", t, re.I):
            return RuleHit(NONE, "chat template has no tool / function support")
        if (info.model_type or "") in ("chatglm", "glm4"):
            return RuleHit(NONE, "GLM-4 / ChatGLM function format: no vLLM tool parser", strong=False)
    for k, v in _TOOL_NAMES:
        if k in n:
            return RuleHit(v, f"model family {k}", strong=False)
    return None


def rule_reasoning(info: ModelInfo) -> RuleHit | None:
    t, n = info.template or "", _name(info)
    for k, v in _REASONING_NAMES[:2]:  # explicit non-thinking variants beat template markers
        if k in n:
            return RuleHit(v, f"{k} models do not emit separate reasoning", strong=False)
    if t:
        if "<|channel|>analysis" in t or ("<|channel|>" in t and "analysis" in t):
            return RuleHit("openai_gptoss", "chat template uses the gpt-oss analysis channel")
        if "<|think|>" in t or "<|channel>thought" in t:
            return RuleHit("gemma4", "chat template uses Gemma 4 thinking markers")
        if "<seed:think>" in t:
            return RuleHit("seed_oss", "chat template uses <seed:think>")
        if "[THINK]" in t:
            return RuleHit("mistral", "chat template uses [THINK]")
        if "<|START_THINKING|>" in t:
            return RuleHit("cohere_command3", "chat template uses Cohere <|START_THINKING|>", strong=False)
        if "<think>" in t:
            mt = (info.model_type or "").lower()
            fam = mt + " " + (info.architecture or "").lower() + " " + n
            if "glm" in fam:
                return RuleHit("glm47" if "glm-4.7" in n else "glm45", "GLM chat template with <think>")
            if "deepseek-v3" in n or mt == "deepseek_v32":
                return RuleHit("deepseek_v3", "DeepSeek-V3.x hybrid-thinking template with <think>")
            if "deepseek-r1" in n or ("<｜Assistant｜><think>" in t):
                return RuleHit("deepseek_r1", "DeepSeek-R1 style template opens <think>")
            if mt.startswith("kimi"):
                return RuleHit("kimi_k2", "Kimi K2 chat template with <think>")
            if "nemotron" in fam:
                return RuleHit("nemotron_v3", "Nemotron 3 chat template with <think> (enable_thinking)")
            if "qwen3" in fam or "qwen3_5" in fam:
                return RuleHit("qwen3", "Qwen3 chat template with <think>")
            if "enable_thinking" in t:  # the qwen3 parser's contract: <think>…</think>, switched by enable_thinking
                return RuleHit("qwen3", "chat template switches <think> blocks with enable_thinking")
        if not any(m in t for m in _THINK_MARKERS) and not any(k in n for k in _MARKERLESS_REASONING):
            return RuleHit(NONE, "chat template has no thinking markers")
    for k, v in _REASONING_NAMES[2:]:
        if k in n:
            return RuleHit(v, f"model family {k}", strong=False)
    return None


def template_excerpt(template: str | None) -> str:
    """The lines that carry tool-call / thinking format markers (what the spike showed the model needs)."""
    if not template:
        return ""
    keep = [ln.strip() for ln in template.splitlines()
            if re.search(r"tool|function|think|reason|channel|call|\[|<\|", ln, re.I)]
    out = "\n".join(keep or template.splitlines()[:40])
    return out[:TEMPLATE_EXCERPT_CHARS]


def state_for(info: ModelInfo) -> str:
    return "\n".join([
        quoted_block("MODEL NAME", info.ref, max_chars=256),
        quoted_block("CONFIG.JSON ARCHITECTURE", f"{info.architecture or 'unknown'} (model_type {info.model_type or '?'})",
                     max_chars=256),
        quoted_block("CHAT TEMPLATE EXCERPT", template_excerpt(info.template) or "(no chat template found)",
                     max_chars=TEMPLATE_EXCERPT_CHARS, keep="head"),
    ])


_DELIMITER_RUN = re.compile(r"<{3,}|>{3,}")  # shaped like the quoted-block delimiters of the decider state
_ANSWER_WORDS = r"(?:answer|choose|choice|select|pick|respond|reply|parser|classifier)"


def injection_shaped(template: str | None, ref: str | None) -> bool:
    """The untrusted text the parser questions quote (chat template, repo name) contains a delimiter-like run of
    ``<<<`` / ``>>>`` or answer-like text naming a parser option ("the answer is deepseek_r1", "reasoning parser:
    qwen3"). The model's answer is then still shown, but never auto-filled (docs/security.md)."""
    text = f"{template or ''}\n{ref or ''}"
    if _DELIMITER_RUN.search(text):
        return True
    keys = sorted({o.key for o in tool_options() + reasoning_options()} - {NONE}, key=len, reverse=True)
    rx = re.compile(rf"\b{_ANSWER_WORDS}\b[^\n]{{0,40}}?\b(?:{'|'.join(map(re.escape, keys))})\b", re.I)
    return bool(rx.search(text))


def _value(v: str | None) -> str | None:
    return None if v in (None, NONE) else v


def suggestion_from_rules(hit: RuleHit) -> dict:
    return {"value": _value(hit.value), "source": "rules", "confidence": 1.0, "auto_apply": hit.strong,
            "alternatives": [], "reason": hit.reason}


def suggestion_from_model(d: Decision) -> dict | None:
    """source "model" = accepted at/above the parser threshold (auto-apply); an abstained answer whose top
    probability is at least SUGGEST_MIN is still shown as a suggestion, never auto-applied."""
    if d.source == "model" and d.choice is not None:
        choice, conf, auto = d.choice, d.confidence, True
    elif d.probabilities:
        choice = max(d.probabilities, key=d.probabilities.get)
        conf, auto = d.probabilities[choice], False
        if conf < SUGGEST_MIN:
            return None
    else:
        return None
    alts = sorted(((k, p) for k, p in d.probabilities.items() if k != choice), key=lambda kv: -kv[1])[:3]
    return {"value": _value(choice), "source": "model", "confidence": conf, "auto_apply": auto,
            "alternatives": [{"value": _value(k), "p": round(p, 4)} for k, p in alts if p > 0],
            "decision_id": d.decision_id}


async def _suggest_one(decider, feature: str, info: ModelInfo, hit: RuleHit | None, question: str,
                       options: list[Option], subject: str) -> dict | None:
    if hit is not None and valid_parser(hit.value):
        d = safe_record_rules(decider, feature=feature, subject=subject, choice=hit.value, reason=hit.reason)
        return {**suggestion_from_rules(hit), "decision_id": d.decision_id}
    # the real parser threshold: answers below it are logged as abstentions, not accepted decisions
    thr = threshold_of(decider, "parser", NEVER)
    state = await asyncio.to_thread(state_for, info)  # regexes over the untrusted template: off the event loop
    d = await safe_decide(decider, feature=feature, subject=subject, state=state, question=question,
                          options=options, threshold=thr, timeout_s=PARSER_TIMEOUT_S)
    sug = suggestion_from_model(d)
    if sug is not None and sug["auto_apply"] and await asyncio.to_thread(injection_shaped, info.template, info.ref):
        sug = {**sug, "auto_apply": False,
               "reason": "not auto-filled: the chat template or model name contains text shaped like an answer or a "
                         "delimiter"}
    return sug


async def suggest_config(decider, info: ModelInfo, engine: str) -> dict:
    """{tool_call_parser: Suggestion|null, reasoning_parser: Suggestion|null} (contract.md). Both questions run
    concurrently, so the worst case is one decider timeout (PARSER_TIMEOUT_S)."""
    if engine != "vllm":  # llama.cpp has no parser flags
        return {"tool_call_parser": None, "reasoning_parser": None}
    subject = f"model:{info.ref}"[:512]
    tool_hit, reas_hit = await asyncio.to_thread(lambda: (rule_tool(info), rule_reasoning(info)))
    tool, reas = await asyncio.gather(
        _suggest_one(decider, "parser_tool", info, tool_hit, QUESTION_TOOL, tool_options(), subject),
        _suggest_one(decider, "parser_reasoning", info, reas_hit, QUESTION_REASONING, reasoning_options(), subject))
    return {"tool_call_parser": tool, "reasoning_parser": reas}


async def suggest_for_artifact(ctx, artifact, engine: str) -> dict:
    info = await asyncio.to_thread(model_info, artifact)  # reads snapshot files
    return await suggest_config(decider_of(ctx), info, engine)
