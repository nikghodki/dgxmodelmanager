"""GGUF header metadata → ArchSpec (no tensor data is read)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from gguf import GGUFReader, GGUFValueType

from dgxmm.inventory.archspec import ArchSpec, LayerGroup

FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
              19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
              26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
              36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


def _scalar(part) -> Any:
    v = part.tolist()
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def _value(field) -> Any:
    if not field.types:
        return None
    t0 = field.types[0]
    if t0 == GGUFValueType.ARRAY:
        if len(field.types) > 1 and field.types[1] == GGUFValueType.STRING:
            return len(field.data)  # token lists etc.: only the count is needed
        return [_scalar(field.parts[i]) for i in field.data]
    if t0 == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    return _scalar(field.parts[field.data[0]])


def read_gguf_meta(path: Path) -> dict[str, Any]:
    reader = GGUFReader(str(path), "r")
    return {k: _value(f) for k, f in reader.fields.items() if not k.startswith("GGUF.")}


def archspec_from_gguf(meta: dict[str, Any]) -> ArchSpec:
    a = str(meta.get("general.architecture", "unknown"))
    assumptions: list[str] = []

    def g(key: str, default: Any = None, required: bool = False) -> Any:
        v = meta.get(f"{a}.{key}")
        if v is None:
            if required:
                assumptions.append(key)
            return default
        return v

    n_layers = int(g("block_count", 32, True))
    hidden = int(g("embedding_length", 4096, True))
    heads = g("attention.head_count", max(hidden // 128, 1), True)
    heads = max(heads) if isinstance(heads, list) else int(heads)
    kv = g("attention.head_count_kv", heads)
    kv_list = [int(x) for x in kv] if isinstance(kv, list) else [int(kv)] * n_layers
    if len(kv_list) != n_layers:
        assumptions.append("head_count_kv_length")
        kv_list = (kv_list * n_layers)[:n_layers]
    key_len = int(g("attention.key_length") or hidden // heads)
    val_len = int(g("attention.value_length") or key_len)
    head_dim = (key_len + val_len) // 2
    window = g("attention.sliding_window")
    pattern = g("attention.sliding_window_pattern")
    fai = g("full_attention_interval")
    has_ssm = g("ssm.state_size") is not None
    kinds: list[tuple[str, int]] = []
    for i in range(n_layers):
        if kv_list[i] == 0 or (fai and (i + 1) % int(fai) != 0):
            kinds.append(("mamba" if has_ssm else "linear", 0))
        elif window and isinstance(pattern, list) and i < len(pattern):
            kinds.append(("sliding" if pattern[i] else "full", kv_list[i]))
        elif window and isinstance(pattern, int) and pattern > 0:
            kinds.append(("full" if (i + 1) % pattern == 0 else "sliding", kv_list[i]))
        else:
            kinds.append(("full", kv_list[i]))
    state = 0
    if any(k in ("mamba", "linear") for k, _ in kinds):
        d_inner = int(g("ssm.inner_size", 2 * hidden))
        d_state = int(g("ssm.state_size", 128))
        conv = int(g("ssm.conv_kernel", 4))
        groups = int(g("ssm.group_count", 1))
        state = d_inner * d_state + (conv - 1) * (d_inner + 2 * groups * d_state)
    groups_out = []
    order = {"full": 0, "sliding": 1, "linear": 3, "mamba": 4}
    for (kind, kvh), count in sorted(Counter(kinds).items(), key=lambda x: (order[x[0][0]], x[0][1])):
        if kind in ("mamba", "linear"):
            groups_out.append(LayerGroup(kind, count, state_elems=state))  # type: ignore[arg-type]
        else:
            groups_out.append(LayerGroup(kind, count, kv_heads=kvh, head_dim=head_dim,  # type: ignore[arg-type]
                                         window=int(window) if kind == "sliding" and window else None))
    experts = g("expert_count")
    used = g("expert_used_count")
    ff = g("feed_forward_length", 4 * hidden)
    ff = max(ff) if isinstance(ff, list) else int(ff)
    expert_ff = g("expert_feed_forward_length")
    is_moe = bool(experts and int(experts) > 1)
    inter = int(expert_ff) * int(used or 1) + int(g("expert_shared_feed_forward_length", 0) or 0) if is_moe and expert_ff else ff
    vocab = meta.get("tokenizer.ggml.tokens") or g("vocab_size")
    if not vocab:
        assumptions.append("vocab_size")
        vocab = 32000
    ftype = meta.get("general.file_type")
    return ArchSpec(architecture=f"gguf:{a}", model_type=a, num_layers=n_layers, hidden_size=hidden,
                    vocab_size=int(vocab), max_context=int(g("context_length", 4096, True)),
                    layer_groups=tuple(groups_out), intermediate_size=inter, is_moe=is_moe,
                    num_experts=int(experts) if experts else None, experts_per_tok=int(used) if used else None,
                    torch_dtype="bfloat16", quant=f"gguf:{FILE_TYPES.get(ftype, ftype)}",
                    assumptions=tuple(dict.fromkeys(assumptions)))
