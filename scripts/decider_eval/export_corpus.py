#!/usr/bin/env python3
"""Turn downloaded repo metadata (config.json, tokenizer_config.json, chat_template.*, file list -- never weights)
into compact corpus fixtures for the parser / kind evaluation.

    python -I scripts/decider_eval/export_corpus.py <download dir> tests/fixtures/decider_corpus/models

Each fixture keeps exactly what DGXMM's rules and decider states read: repo id, architecture, model_type, the
config keys kind.CONFIG_KEYS lists, the non-weight file names (+ weight file count) and the chat template text.
Inputs are untrusted: they are only parsed as JSON / read as text, never executed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_KEYS = ("architectures", "model_type", "num_labels", "id2label", "num_hidden_layers", "hidden_size",
               "vocab_size", "draft_vocab_size", "vision_config", "audio_config", "text_config", "image_token_id",
               "num_mel_bins", "encoder_layers", "decoder_layers", "is_encoder_decoder", "pooling", "task",
               "task_specific_params", "istftnet", "plbert", "dflash_config", "num_nextn_predict_layers",
               "max_position_embeddings", "speculators_model_type", "transformer_layer_config", "medusa_num_heads",
               "auto_map", "sentence_transformers")
WEIGHT_EXT = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".onnx", ".msgpack", ".h5", ".ckpt", ".mlmodel")
MAX_TEMPLATE = 64 * 1024


def _small(v, depth=0):
    """Nested dicts reduced to their keys below the first level (vision_config etc. are signals by presence)."""
    if isinstance(v, dict):
        return {k: (_small(x, depth + 1) if depth == 0 else "...") for k, x in list(v.items())[:40]} if depth == 0 \
            else sorted(v)[:20]
    if isinstance(v, list):
        return v[:8]
    return v


def template_of(d: Path) -> str | None:
    if (d / "chat_template.jinja").is_file():
        return (d / "chat_template.jinja").read_text(errors="replace")
    if (d / "chat_template.json").is_file():
        try:
            t = json.loads((d / "chat_template.json").read_text()).get("chat_template")
            if isinstance(t, str):
                return t
        except (ValueError, AttributeError):
            pass
    if (d / "tokenizer_config.json").is_file():
        try:
            t = json.loads((d / "tokenizer_config.json").read_text()).get("chat_template")
        except ValueError:
            return None
        if isinstance(t, str):
            return t
        if isinstance(t, list):
            return "\n".join(x.get("template", "") for x in t if isinstance(x, dict))
    return None


def export(d: Path, ref: str) -> dict:
    cfg = {}
    if (d / "config.json").is_file():
        try:
            cfg = json.loads((d / "config.json").read_text())
        except ValueError:
            cfg = {}
    files = json.loads((d / "files.json").read_text()) if (d / "files.json").is_file() else sorted(
        str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    t = template_of(d)
    archs = cfg.get("architectures")
    return {
        "ref": ref,
        "architecture": archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else None,
        "model_type": cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else None,
        "config": {k: _small(cfg[k]) for k in CONFIG_KEYS if k in cfg} if cfg else None,
        "files": sorted(f for f in files if not f.lower().endswith(WEIGHT_EXT))[:200],
        "weight_files": sum(1 for f in files if f.lower().endswith(WEIGHT_EXT)),
        "template": t[:MAX_TEMPLATE] if t else None,
    }


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir() and "--" in p.name and not p.name.startswith("_")):
        ref = d.name.replace("--", "/", 1)
        (out / f"{d.name}.json").write_text(json.dumps(export(d, ref), indent=1, ensure_ascii=False) + "\n")
        n += 1
    print(f"exported {n} repos to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
