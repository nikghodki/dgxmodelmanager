"""Turn a Hugging Face config.json into the numbers the memory estimator needs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

LayerKind = Literal["full", "sliding", "mla", "linear", "mamba"]
_KIND_ORDER = {"full": 0, "sliding": 1, "mla": 2, "linear": 3, "mamba": 4}


@dataclass(frozen=True)
class LayerGroup:
    kind: LayerKind
    count: int
    kv_heads: int = 0
    head_dim: int = 0
    window: int | None = None
    mla_dim: int = 0        # MLA: kv_lora_rank + qk_rope_head_dim (one latent per token)
    state_elems: int = 0    # linear/mamba: conv + recurrent state elements per layer per sequence


@dataclass(frozen=True)
class ArchSpec:
    architecture: str
    model_type: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    max_context: int
    layer_groups: tuple[LayerGroup, ...]
    intermediate_size: int
    is_moe: bool = False
    num_experts: int | None = None
    experts_per_tok: int | None = None
    torch_dtype: str = "bfloat16"
    quant: str = "bf16"
    assumptions: tuple[str, ...] = ()  # fields that were missing and defaulted

    @property
    def confident(self) -> bool:
        return not self.assumptions


def _merged(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg.get("text_config")
    if not isinstance(t, dict):
        return dict(cfg)
    m = {**cfg, **t}
    m["architectures"] = cfg.get("architectures") or t.get("architectures")
    m["quantization_config"] = cfg.get("quantization_config") or t.get("quantization_config")
    return m


def _norm_kind(raw: Any) -> LayerKind | None:
    s = str(raw).lower()
    if s in ("full_attention", "attention", "global", "global_attention", "full"):
        return "full"
    if s in ("sliding_attention", "local", "local_attention", "sliding", "chunked_attention"):
        return "sliding"
    if s in ("linear_attention", "linear", "deltanet", "gated_deltanet"):
        return "linear"
    if s.startswith("mamba") or s == "ssm":
        return "mamba"
    return None


def detect_quant(qc: dict | None, dtype: str) -> str:
    if not qc:
        return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(str(dtype), "bf16")
    method = str(qc.get("quant_method", "")).lower()
    algos = {str(qc.get("quant_algo", "")).upper()}
    for layer in (qc.get("quantized_layers") or {}).values():
        if isinstance(layer, dict):
            algos.add(str(layer.get("quant_algo", "")).upper())
    if any("NVFP4" in a for a in algos):
        return "nvfp4"
    if any("MXFP4" in a for a in algos) or "mxfp4" in method:
        return "mxfp4"
    bits: set[tuple[Any, Any]] = set()
    formats = {str(qc.get("format", "")).lower()}
    for g in (qc.get("config_groups") or {}).values():
        w = (g or {}).get("weights") or {}
        bits.add((w.get("num_bits"), w.get("type")))
        formats.add(str((g or {}).get("format", "")).lower())
    if any("nvfp4" in f for f in formats) or (4, "float") in bits:
        return "nvfp4"
    if any("FP8" in a for a in algos) or (8, "float") in bits or method == "fp8":
        return "fp8"
    if method in ("awq", "gptq", "bitsandbytes", "gguf", "hqq", "eetq"):
        return "bnb" if method == "bitsandbytes" else method
    if (4, "int") in bits:
        return "w4a16"
    if (8, "int") in bits:
        return "w8a8"
    return method or "unknown"


def parse_hf_config(cfg: dict[str, Any]) -> ArchSpec:
    c = _merged(cfg)
    assumptions: list[str] = []

    def need(key: str, default: Any) -> Any:
        v = c.get(key)
        if v is None:
            assumptions.append(key)
            return default
        return v

    n_layers = max(int(need("num_hidden_layers", 32)), 1)
    hidden = int(need("hidden_size", 4096))
    n_heads = int(need("num_attention_heads", max(hidden // 128, 1)))
    kv_heads = int(c.get("num_key_value_heads") or n_heads)
    head_dim = int(c.get("head_dim") or max(hidden // max(n_heads, 1), 1))
    vocab = int(need("vocab_size", 32000))
    max_ctx = c.get("max_position_embeddings") or c.get("max_sequence_length") or c.get("seq_length")
    if max_ctx is None:
        assumptions.append("max_position_embeddings")
        max_ctx = 4096
    window = c.get("sliding_window")

    raw_types = c.get("layer_types") or c.get("layers_block_type")
    kinds: list[LayerKind]
    if isinstance(raw_types, list) and raw_types:
        kinds = []
        for raw in raw_types:
            k = _norm_kind(raw)
            if k is None:
                assumptions.append(f"layer_type:{raw}")
                k = "full"
            kinds.append(k)
        if len(kinds) != n_layers:
            assumptions.append("layer_types_length")
            kinds = (kinds * (n_layers // len(kinds) + 1))[:n_layers]
    elif c.get("full_attention_interval") and c.get("linear_num_value_heads"):
        iv = int(c["full_attention_interval"])
        kinds = ["full" if (i + 1) % iv == 0 else "linear" for i in range(n_layers)]
    elif window and c.get("use_sliding_window", True) is not False:
        kinds = ["sliding"] * n_layers
    else:
        kinds = ["full"] * n_layers
    if c.get("kv_lora_rank"):
        kinds = ["mla" if k == "full" else k for k in kinds]
    if "sliding" in kinds and not window:
        assumptions.append("sliding_window")

    linear_state = 0
    if "linear" in kinds:
        nk, nv = int(c.get("linear_num_key_heads") or 16), int(c.get("linear_num_value_heads") or 32)
        dk, dv = int(c.get("linear_key_head_dim") or 128), int(c.get("linear_value_head_dim") or 128)
        kconv = int(c.get("linear_conv_kernel_dim") or 4)
        if not c.get("linear_num_value_heads"):
            assumptions.append("linear_num_value_heads")
        key_dim, value_dim = nk * dk, nv * dv
        linear_state = (kconv - 1) * (2 * key_dim + value_dim) + nv * dk * dv
    mamba_state = 0
    if "mamba" in kinds:
        nh = int(c.get("mamba_n_heads") or 0)
        dh = int(c.get("mamba_d_head") or 64)
        ds = int(c.get("mamba_d_state") or c.get("ssm_state_size") or 128)
        ng = int(c.get("mamba_n_groups") or c.get("n_groups") or 1)
        dconv = int(c.get("mamba_d_conv") or c.get("conv_kernel") or 4)
        if not nh:
            assumptions.append("mamba_n_heads")
            nh = max((2 * hidden) // dh, 1)
        mamba_state = nh * dh * ds + (dconv - 1) * (nh * dh + 2 * ng * ds)

    full_kv = int(c.get("num_global_key_value_heads") or kv_heads)
    full_hd = int(c.get("global_head_dim") or head_dim)
    mla_dim = int(c.get("kv_lora_rank") or 0) + int(c.get("qk_rope_head_dim") or 64) if c.get("kv_lora_rank") else 0
    built = []
    for kind, count in sorted(Counter(kinds).items(), key=lambda kv: _KIND_ORDER[kv[0]]):
        if kind == "full":
            built.append(LayerGroup("full", count, kv_heads=full_kv, head_dim=full_hd))
        elif kind == "sliding":
            built.append(LayerGroup("sliding", count, kv_heads=kv_heads, head_dim=head_dim,
                                    window=int(window) if window else None))
        elif kind == "mla":
            built.append(LayerGroup("mla", count, mla_dim=mla_dim))
        elif kind == "linear":
            built.append(LayerGroup("linear", count, state_elems=linear_state))
        else:
            built.append(LayerGroup("mamba", count, state_elems=mamba_state))

    num_experts = c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts")
    is_moe = bool(num_experts and int(num_experts) > 1)
    per_tok = int(c.get("num_experts_per_tok") or c.get("moe_topk") or 0) or None
    dense_inter = int(c.get("intermediate_size") or 4 * hidden)
    if is_moe:
        moe_inter = int(c.get("moe_intermediate_size") or dense_inter)
        shared = int(c.get("shared_expert_intermediate_size") or 0)
        inter = moe_inter * (per_tok or 1) + shared
    else:
        inter = dense_inter
    dtype = str(c.get("torch_dtype") or c.get("dtype") or "bfloat16")
    archs = c.get("architectures") or []
    return ArchSpec(
        architecture=str(archs[0]) if archs else "unknown",
        model_type=str(c.get("model_type") or "unknown"),
        num_layers=n_layers, hidden_size=hidden, vocab_size=vocab, max_context=int(max_ctx),
        layer_groups=tuple(built), intermediate_size=inter, is_moe=is_moe,
        num_experts=int(num_experts) if num_experts else None, experts_per_tok=per_tok,
        torch_dtype=dtype, quant=detect_quant(c.get("quantization_config"), dtype),
        assumptions=tuple(dict.fromkeys(assumptions)),
    )
