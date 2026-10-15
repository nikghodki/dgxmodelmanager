"""Allowlisted extra CLI flags. DGXMM-managed flags (--model, --port, --host,
--gpu-memory-utilization, …) are deliberately absent so users cannot override them."""
from __future__ import annotations

import re

_JSON_OBJ = r"\{[^;&|`$<>\\]{0,2000}\}"
VLLM_EXTRA_FLAGS: dict[str, str | None] = {
    "--enforce-eager": None,
    "--disable-log-requests": None,
    "--disable-log-stats": None,
    "--async-scheduling": None,
    "--disable-custom-all-reduce": None,
    "--seed": r"\d{1,10}",
    "--max-logprobs": r"\d{1,4}",
    "--max-seq-len-to-capture": r"\d{1,7}",
    "--dtype": r"auto|bfloat16|float16",
    "--quantization": r"[a-z0-9_]{1,32}",
    "--load-format": r"[a-z0-9_]{1,32}",
    "--generation-config": r"auto|vllm",
    "--chat-template-content-format": r"auto|string|openai",
    "--limit-mm-per-prompt": _JSON_OBJ,
    "--compilation-config": _JSON_OBJ,
    "--hf-overrides": _JSON_OBJ,
    "--override-generation-config": _JSON_OBJ,
    "--mm-processor-kwargs": _JSON_OBJ,
    "--mamba-ssm-cache-dtype": r"auto|float32|bfloat16",
}
LLAMACPP_EXTRA_FLAGS: dict[str, str | None] = {
    "--threads": r"\d{1,3}",
    "--threads-batch": r"\d{1,3}",
    "--mlock": None,
    "--no-mmap": None,
    "--jinja": None,
    "--cont-batching": None,
    "--no-warmup": None,
    "--kv-unified": None,
    "--swa-full": None,
    "--seed": r"\d{1,10}",
    "--reasoning-format": r"[a-z_]{1,32}",
    "--reasoning-budget": r"-?\d{1,7}",
    "--chat-template": r"[A-Za-z0-9_.-]{1,64}",
    "--rope-scaling": r"none|linear|yarn",
    "--rope-freq-scale": r"\d+(\.\d+)?",
    "--defrag-thold": r"\d+(\.\d+)?",
    "--temp": r"\d+(\.\d+)?",
}


def validate_extra_args(args: list[str], allow: dict[str, str | None]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        flag, eq, val = str(args[i]).partition("=")
        if flag not in allow:
            raise ValueError(f"flag not allowed: {flag}")
        pattern = allow[flag]
        if pattern is None:
            if eq:
                raise ValueError(f"{flag} takes no value")
            out.append(flag)
            i += 1
            continue
        if eq:
            i += 1
        else:
            if i + 1 >= len(args):
                raise ValueError(f"{flag} needs a value")
            val = str(args[i + 1])
            i += 2
        if not re.fullmatch(pattern, val):
            raise ValueError(f"invalid value for {flag}")
        out += [flag, val]
    return out
