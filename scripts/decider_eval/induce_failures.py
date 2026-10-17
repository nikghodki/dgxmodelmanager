#!/usr/bin/env python3
"""Induce real vLLM start-up failures on this host and keep their log tails as diagnosis regression fixtures.

Evaluation tooling for docs/decider-evaluation.md, run by hand on the DGX (never by DGXMM or CI):

    flock ~/workspace/dgxmm-wt/.gpu-lock .venv/bin/python scripts/decider_eval/induce_failures.py \
        --out tests/fixtures/real_failures --work ~/dgxmm-eval-work [--case NAME ...]

Every case starts ONE throwaway container named ``prod-h1-<case>`` from the engine image DGXMM uses, with the
command line DGXMM's own VllmEngine.container_spec builds (plus the one change that provokes the failure), waits
for it to exit, captures ``docker logs --tail 200`` exactly like the deployer's ``_fail_exited`` and removes the
container. The container is NOT labelled ``dgxmm.managed``, so DGXMM never sees it. Downloaded repositories (only
config / tokenizer files, never weights) go to a private HF cache under ``--work``, never into the user's cache;
the user's cache is mounted read-only. The fixture is the log tail after DGXMM's ``redact_text``, with host paths
replaced, preceded by a small header (true label, exit code, status message).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import docker
from docker.types import DeviceRequest, Mount as DMount

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dgxmm.config import Settings  # noqa: E402
from dgxmm.deployer.logtail import SCAN_LINES, condense  # noqa: E402
from dgxmm.engines.base import LaunchContext  # noqa: E402
from dgxmm.engines.vllm import VllmEngine  # noqa: E402
from dgxmm.inventory.hf_scan import scan_hf_cache  # noqa: E402
from dgxmm.memory.redact import redact_text  # noqa: E402

IMAGE = "vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8"
HOST_HF = Path.home() / ".cache" / "huggingface"
PREFIX = "prod-h1-"
GIB = 1024 ** 3
TOTAL = 128 * GIB  # GB10 unified memory (gpu_memory_utilization is a fraction of it)


@dataclass
class Case:
    name: str
    label: str
    model: str                      # repo id (or a local path inside the container)
    cfg: dict = field(default_factory=dict)
    reserved_gib: float = 5.0       # -> --gpu-memory-utilization
    env: dict = field(default_factory=dict)
    private_cache: bool = False     # use the private HF cache under --work instead of the user's (read-only)
    download: list[str] = field(default_factory=list)  # files fetched into the private cache (no weights)
    extra_cmd: list[str] = field(default_factory=list)
    drop_args: list[str] = field(default_factory=list)  # flags removed from DGXMM's command (with their value)
    mem_limit: str | None = None
    tmpfs: dict = field(default_factory=dict)
    network_container: str | None = None  # share the network namespace of a helper container
    local_files: dict = field(default_factory=dict)  # container path -> host file (bind-mounted read-only)
    note: str = ""


QWEN = "Qwen/Qwen3-0.6B"
SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048, "enable_sleep_mode": False}
CASES = [
    Case("unsupported-arch", "unsupported_architecture", "valhalla/distilbart-mnli-12-1", dict(SMALL), 3,
         note="BartForSequenceClassification (an NLI classifier on disk) served as a generative model"),
    Case("trust-remote-code", "trust_remote_code_required", "internlm/internlm2-chat-1_8b", dict(SMALL), 3,
         private_cache=True,
         download=["config.json", "configuration_internlm2.py", "tokenizer_config.json", "generation_config.json"],
         note="model_type internlm2 needs the repository's configuration code; only config files downloaded"),
    Case("gated-no-token", "auth_gated", "meta-llama/Llama-3.2-1B-Instruct", dict(SMALL), 3, private_cache=True,
         env={"HF_HUB_OFFLINE": "0"}, note="gated repository, online, no Hugging Face token"),
    Case("offline-missing", "offline_missing_files", "Qwen/Qwen2.5-0.5B-Instruct", dict(SMALL), 3, private_cache=True,
         note="repository not downloaded, HF_HUB_OFFLINE=1 (DGXMM's default)"),
    Case("kv-too-small", "kv_cache_too_small", QWEN, {**SMALL, "max_model_len": 32768}, 3.2,
         note="max_model_len 32768 with a 3.2 GiB budget: weights fit, the KV cache for one full request does not"),
    Case("kv-none-left", "kv_cache_too_small", QWEN, {**SMALL, "max_model_len": 8192}, 1.9,
         note="1.9 GiB budget: almost nothing is left for KV cache blocks after the weights"),
    Case("ctx-above-model", "context_too_long", QWEN, {**SMALL, "max_model_len": 131072, "kv_cache_dtype": "fp8"}, 5,
         note="max_model_len 131072 above Qwen3-0.6B's max_position_embeddings 40960"),
    Case("port-in-container", "port_in_use", QWEN, dict(SMALL), 4, network_container="listener",
         note="another process in the same network namespace already listens on the server port"),
    Case("kv-small-budget", "kv_cache_too_small", "/models/qwen3-no-tokenizer", dict(SMALL), 4,
         local_files={"config.json": "config.json", "model.safetensors": "model.safetensors",
                      "generation_config.json": "generation_config.json"},
         note="a local checkpoint without tokenizer files and a 4 GiB budget: the engine failed earlier, on the "
              "KV cache (true cause)"),
    Case("broken-tokenizer", "tokenizer_or_template", "/models/qwen3-broken-tokenizer", dict(SMALL), 6,
         local_files={"config.json": "config.json", "model.safetensors": "model.safetensors",
                      "generation_config.json": "generation_config.json", "tokenizer_config.json":
                      "tokenizer_config.json", "vocab.json": "vocab.json", "merges.txt": "merges.txt",
                      "tokenizer.json": "@broken/tokenizer.json"},
         note="a local checkpoint whose tokenizer.json is truncated (an interrupted copy); without tokenizer files "
              "at all vLLM still starts (it falls back), so a corrupt file is the realistic failure"),
    Case("disk-full", "disk_full", QWEN, {**SMALL, "enable_sleep_mode": False}, 4,
         tmpfs={"/root/.cache/vllm": "size=64k", "/root/.triton": "size=64k", "/tmp": "size=256k"},
         note="tiny tmpfs for the compile caches and /tmp (simulates a full disk without filling a real one)"),
    Case("cgroup-oom", "gpu_oom", QWEN, dict(SMALL), 4, mem_limit="2g",
         note="container memory limit 2 GiB: the kernel OOM-kills the engine while loading (exit 137)"),
    Case("bad-tool-parser", "bad_parser_flag", QWEN,
         {**SMALL, "enable_auto_tool_choice": True, "tool_call_parser": "qwen3_tools"}, 3,
         note="a tool-call parser name that does not exist in this vLLM"),
    Case("bad-reasoning-parser", "bad_parser_flag", QWEN, {**SMALL, "reasoning_parser": "qwen35"}, 3,
         note="a reasoning parser name that does not exist in this vLLM"),
    Case("auto-tool-no-parser", "bad_parser_flag", QWEN, {**SMALL, "enable_auto_tool_choice": True}, 3,
         note="--enable-auto-tool-choice without --tool-call-parser"),
]


def host_sanitise(text: str) -> str:
    home = str(Path.home())
    text = text.replace(home, "/home/USER")
    return re.sub(r"/home/[A-Za-z0-9._-]+", "/home/USER", text)


def build_command(engine: VllmEngine, settings: Settings, case: Case, artifacts: dict) -> list[str]:
    art = artifacts.get(case.model)
    if art is None:  # not on disk: an artifact-shaped stand-in so DGXMM's own command builder can be used
        from dgxmm.inventory.artifact import ModelArtifact
        art = ModelArtifact(ref=case.model, source="hf", local_path="", revision=None, format="safetensors",
                            weights_bytes=0, size_bytes=0, complete=True, kind="llm", arch=None, architecture=None,
                            quant=None)
    cfg = engine.validate_config(case.cfg)
    spec = engine.container_spec(art, cfg, None, LaunchContext(0, PREFIX + case.name, 8000,
                                                               int(case.reserved_gib * GIB), TOTAL), settings)
    cmd = list(spec.command)
    if case.model.startswith("/"):  # local path: no --revision of the repo it was copied from
        cmd = [case.model if (i > 0 and cmd[i - 1] == "--model") else c for i, c in enumerate(cmd)]
        if "--revision" in cmd:
            i = cmd.index("--revision")
            del cmd[i:i + 2]
    for flag in case.drop_args:
        if flag in cmd:
            i = cmd.index(flag)
            del cmd[i:i + 2]
    return cmd + case.extra_cmd


def fetch(case: Case, cache: Path) -> None:
    from huggingface_hub import hf_hub_download
    for f in case.download:
        try:
            hf_hub_download(case.model, f, cache_dir=str(cache / "hub"))
        except Exception as e:  # e.g. optional file missing
            print(f"  [{case.name}] download {f}: {type(e).__name__}", file=sys.stderr)


def run_case(client, engine, settings, case: Case, artifacts: dict, work: Path, timeout_s: int) -> dict:
    name = PREFIX + case.name
    private = work / "hf" / case.name
    if case.private_cache:
        private.mkdir(parents=True, exist_ok=True)
        fetch(case, private)
    cmd = build_command(engine, settings, case, artifacts)
    env = {"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", **case.env}
    mounts = [DMount("/root/.cache/huggingface", str(private if case.private_cache else HOST_HF), type="bind",
                     read_only=not case.private_cache)]
    for d in ("vllm", "flashinfer"):
        if f"/root/.cache/{d}" not in case.tmpfs:
            p = work / "cache" / d
            p.mkdir(parents=True, exist_ok=True)
            mounts.append(DMount(f"/root/.cache/{d}", str(p), type="bind"))
    if case.local_files:
        snap = artifacts[QWEN].local_path
        for target, rel in case.local_files.items():
            if rel.startswith("@broken/"):  # first 4 KiB of the real file: a truncated copy
                src = Path(os.path.realpath(os.path.join(snap, rel.split("/", 1)[1])))
                dst = work / "broken" / case.name / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes()[:4096])
                host = str(dst)
            else:
                host = os.path.realpath(os.path.join(snap, rel))
            mounts.append(DMount(f"{case.model}/{target}", host, type="bind", read_only=True))
    helper = None
    kwargs = dict(image=IMAGE, command=cmd, name=name, detach=True, environment=env, mounts=mounts,
                  labels={"prod-h1.eval": "true"}, device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                  ipc_mode="host", tmpfs=case.tmpfs or None)
    if case.mem_limit:
        kwargs["mem_limit"] = case.mem_limit
        kwargs["memswap_limit"] = case.mem_limit
    if case.network_container:
        helper = client.containers.run(IMAGE, entrypoint=["python3", "-c",
                                       "import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,"
                                       "socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',8000));s.listen();time.sleep(900)"],
                                       name=PREFIX + case.name + "-listener", detach=True,
                                       labels={"prod-h1.eval": "true"})
        time.sleep(3)
        kwargs["network_mode"] = f"container:{helper.id}"
    t0 = time.time()
    c = client.containers.run(**kwargs)
    try:
        status, code = "running", None
        while time.time() - t0 < timeout_s:
            c.reload()
            if c.status in ("exited", "dead"):
                status, code = c.status, c.attrs["State"]["ExitCode"]
                oom = c.attrs["State"].get("OOMKilled")
                break
            time.sleep(2)
        else:
            oom = False
        full = c.logs(tail=SCAN_LINES).decode(errors="replace")
        logs = condense(full)  # exactly what the deployer stores with the failed event
    finally:
        c.remove(force=True)
        if helper is not None:
            helper.remove(force=True)
    hint = " (likely out of memory)" if code == 137 else ""
    message = f"container exited with code {code}{hint}" if status != "running" else \
        f"startup timed out after {timeout_s}s"
    return {"case": case.name, "label": case.label, "exit_code": code, "oom_killed": oom, "message": message,
            "seconds": round(time.time() - t0, 1), "log": host_sanitise(redact_text(logs)), "raw": full, "note": case.note,
            "model": case.model, "command": " ".join(cmd)}


def write_fixture(out: Path, r: dict) -> Path:
    p = out / f"{r['case']}.txt"
    header = [f"# label: {r['label']}", "# engine: vllm", f"# model_ref: {r['model']}",
              f"# exit_code: {r['exit_code']}", f"# message: {r['message']}", f"# induced: {r['note']}",
              "# source: induced on the DGX Spark (GB10) with scripts/decider_eval/induce_failures.py",
              "---"]
    p.write_text("\n".join(header) + "\n" + r["log"].rstrip("\n") + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--case", action="append")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild the fixtures of the selected cases from their saved raw logs (no containers)")
    args = ap.parse_args()
    if args.from_raw:
        meta = {}
        for ln in (args.work / "results.jsonl").read_text().splitlines():
            r = json.loads(ln)
            meta[r["case"]] = r
        for case in CASES:
            raw = args.work / "raw" / f"{case.name}.log"
            if (args.case and case.name not in args.case) or not raw.is_file() or case.name not in meta:
                continue
            r = {**meta[case.name], "log": host_sanitise(redact_text(condense(raw.read_text())))}
            print(f"   rebuilt {write_fixture(args.out, r)}")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()
    settings = Settings(images={"vllm": IMAGE}, hf_cache={"host": str(HOST_HF), "local": str(HOST_HF)})
    engine = VllmEngine(settings)
    artifacts = {a.ref: a for a in scan_hf_cache(HOST_HF / "hub")}
    for case in CASES:
        if args.case and case.name not in args.case:
            continue
        print(f"== {case.name} ({case.label})", flush=True)
        r = run_case(client, engine, settings, case, artifacts, args.work, args.timeout)
        p = write_fixture(args.out, r)
        print(f"   exit={r['exit_code']} oom={r['oom_killed']} {r['seconds']}s -> {p}", flush=True)
        raw = args.work / "raw"  # private: never committed (may contain unredacted text)
        raw.mkdir(mode=0o700, exist_ok=True)
        (raw / f"{case.name}.log").write_text(r["raw"])
        with open(args.work / "results.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("log", "raw")}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
