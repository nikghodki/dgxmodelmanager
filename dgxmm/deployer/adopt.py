"""Turn an existing hand-made vLLM container into a DGXMM deployment (spec §5.6)."""
from __future__ import annotations

import asyncio
import json
import re

from sqlalchemy import select

from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer.service import NAME_PATTERN, DeployError
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS

VALUE_FLAGS = {
    "--served-model-name": ("served_model_name", str), "--max-model-len": ("max_model_len", "int"),
    "--max-num-seqs": ("max_num_seqs", "int"), "--max-num-batched-tokens": ("max_num_batched_tokens", "int"),
    "--kv-cache-dtype": ("kv_cache_dtype", str), "--tool-call-parser": ("tool_call_parser", str),
    "--reasoning-parser": ("reasoning_parser", str), "--safetensors-load-strategy": ("safetensors_load_strategy", str),
}
BOOL_FLAGS = {
    "--enable-prefix-caching": ("enable_prefix_caching", True), "--no-enable-prefix-caching": ("enable_prefix_caching", False),
    "--enable-chunked-prefill": ("enable_chunked_prefill", True), "--no-enable-chunked-prefill": ("enable_chunked_prefill", False),
    "--enable-auto-tool-choice": ("enable_auto_tool_choice", True), "--enable-sleep-mode": ("enable_sleep_mode", True),
    "--trust-remote-code": ("trust_remote_code", True),
}
MANAGED = {"--host", "--port", "--gpu-memory-utilization", "--revision"}


class AdoptError(DeployError):
    pass


_DEC = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_BIN = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def human_int(flag: str, val: str) -> int:
    """Parse an integer the way vLLM's CLI does: 32k = 32000 (decimal), 32K = 32768 (binary)."""
    v = val.strip()
    if re.fullmatch(r"\d+", v):
        return int(v)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgtKMGT])", v)
    if m:
        num, suf = m.groups()
        if suf in _DEC:
            return int(float(num) * _DEC[suf])
        if "." not in num:
            return int(num) * _BIN[suf]
    raise AdoptError("invalid_value", f"cannot adopt: {flag} has an unparseable value {val!r}", 422)


def parse_vllm_command(cmd: list[str]) -> tuple[str, dict, list[str]]:
    toks = list(cmd)
    if toks[:2] == ["vllm", "serve"]:
        toks = toks[2:]
    model = None
    if toks and not toks[0].startswith("-"):
        model, toks = toks[0], toks[1:]
    cfg: dict = {"enable_sleep_mode": False}
    ignored, extra, unsupported = [], [], []
    i = 0
    while i < len(toks):
        flag, eq, val = toks[i].partition("=")
        takes_value = flag in VALUE_FLAGS or flag in MANAGED or flag in ("--model", "--speculative-config") \
            or (flag in VLLM_EXTRA_FLAGS and VLLM_EXTRA_FLAGS[flag] is not None)
        if takes_value and not eq:
            val = toks[i + 1] if i + 1 < len(toks) else ""
            i += 2
        else:
            i += 1
        if flag == "--model":
            model = val
        elif flag in MANAGED:
            ignored.append(flag)
        elif flag in VALUE_FLAGS:
            key, typ = VALUE_FLAGS[flag]
            cfg[key] = human_int(flag, val) if typ == "int" else typ(val)
        elif flag in BOOL_FLAGS:
            key, value = BOOL_FLAGS[flag]
            cfg[key] = value
        elif flag == "--speculative-config":
            try:
                spec = json.loads(val)
            except ValueError:
                spec = None
            if not isinstance(spec, dict):
                raise AdoptError("invalid_value", "cannot adopt: --speculative-config is not a JSON object", 422)
            cfg["speculative"] = spec
        elif flag in VLLM_EXTRA_FLAGS:
            extra += [flag] + ([val] if VLLM_EXTRA_FLAGS[flag] is not None else [])
        else:
            unsupported.append(flag)
    if unsupported:
        raise AdoptError("unsupported_flags", f"cannot adopt: unsupported flags {', '.join(unsupported)}", 422)
    if not model:
        raise AdoptError("no_model", "could not find --model in the container command", 422)
    if extra:
        cfg["extra_args"] = extra
    return model, cfg, ignored


def list_adoptable(deployer) -> list[dict]:
    with deployer.db.session() as s:
        adopted = {d.adopted_from: d.name for d in s.scalars(select(Deployment)) if d.adopted_from}
    out = []
    for c in deployer.docker.list_all():
        if c.managed or "vllm" not in c.image:
            continue
        reason = adopt_blocker(deployer, c.command)
        out.append({**c.as_dict(), "adopted_as": adopted.get(c.name), "adoptable": reason is None, "reason": reason})
    return sorted(out, key=lambda c: c["name"])


def adopt_blocker(deployer, command: list[str]) -> str | None:
    """Why this container cannot be adopted (None if it can): its command must parse and its model must be
    a downloaded LLM (excludes e.g. whisper / embedding servers that also use the vLLM image)."""
    try:
        ref, _, _ = parse_vllm_command(command)
    except DeployError as e:
        return e.message
    art = deployer.inventory.get(ref)
    if art is None or art.kind != "llm":
        return f"{ref} is not a downloaded LLM"
    return None


async def adopt(deployer, container: str, name: str, user_id: int | None) -> dict:
    if not re.fullmatch(NAME_PATTERN, name):
        raise AdoptError("invalid_name", "deployment name must be lowercase letters, digits and dashes", 422)
    info = await asyncio.to_thread(deployer.docker.get, container)
    if info is None:
        raise AdoptError("not_found", f"container {container} not found", 404)
    if info.managed:
        raise AdoptError("already_managed", f"{info.name} is already managed by dgxmm", 409)
    if "vllm" not in info.image:
        raise AdoptError("unsupported_image", f"only vLLM containers can be adopted (image {info.image})", 422)
    ref, cfg, _ = parse_vllm_command(info.command)
    prep = deployer.prepare("vllm", ref, cfg)
    with deployer.db.session() as s:
        if s.scalar(select(Deployment).where(Deployment.name == name)):
            raise AdoptError("name_in_use", f"a deployment named {name} already exists", 409)
        row = Deployment(name=name, engine="vllm", model_ref=ref, config=prep.cfg.model_dump(mode="json"),
                         state="stopped", reserved_bytes=prep.estimate.total, estimate=prep.estimate.as_dict(),
                         adopted_from=info.name, created_by=user_id)
        s.add(row)
        s.flush()
        s.add(DeploymentEvent(deployment_id=row.id, from_state="stopped", to_state="stopped",
                              message=f"adopted settings from container {info.name}; original container left untouched"))
        dep_id = row.id
    view = deployer.view(dep_id)
    deployer.events.publish("deployment", view)
    return view
