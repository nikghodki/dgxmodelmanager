"""Made-up demo data for the documentation screenshots: two DGX Spark nodes (dgx-1, dgx-2), public model names,
documentation addresses (192.0.2.0/24) and the generic user "admin". Nothing here comes from a real installation.

World.handle(method, path, query, body) answers the UI's API requests; anything not listed returns MISSING (404)
and capture.py prints it, so a UI change that needs new data shows up in the capture log."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

GIB = 1024 ** 3
MIB = 1024 ** 2
NOW = datetime(2026, 11, 12, 15, 20, tzinfo=timezone.utc)  # a Thursday
MISSING = object()
TOTAL = int(121.7 * GIB)
MARGIN, OS_RESERVE = 8 * GIB, 4 * GIB
ENDPOINT = "https://llm.example.com/v1"


def iso(dt: datetime) -> str:
    return dt.isoformat()


def ago(**kw) -> str:
    return iso(NOW - timedelta(**kw))


def g(x: float) -> int:
    return int(x * GIB)


NODES = [("local", "dgx-1", False), ("dgx-2", "dgx-2", True)]
LABEL = {n: label for n, label, _ in NODES}


def wave(i: int, base: float, amp: float, period: float = 9.0, phase: float = 0.0) -> float:
    """Deterministic, lively-looking series (no randomness)."""
    return base + amp * (0.6 * math.sin(i / period * 2 * math.pi + phase) + 0.4 * math.sin(i / 3.1 + phase * 2))


# --------------------------------------------------------------------------------------------------------------
# deployments
# --------------------------------------------------------------------------------------------------------------

def _estimate(weights, kv, act, runtime, confidence="estimated", draft=0.0):
    e = {"weights": g(weights), "draft_weights": g(draft), "kv_cache": g(kv), "state": 0, "activations": g(act),
         "runtime": g(runtime), "margin": 0, "confidence": confidence, "notes": []}
    e["total"] = sum(v for k, v in e.items() if isinstance(v, int))
    return e


def _vllm_cfg(ctx, seqs, **extra):
    c = {"max_model_len": ctx, "max_num_seqs": seqs, "max_num_batched_tokens": 8192, "kv_cache_dtype": "auto",
         "enable_prefix_caching": True, "enable_chunked_prefill": True, "enable_auto_tool_choice": False,
         "enable_sleep_mode": True, "trust_remote_code": False}
    c.update(extra)
    return c


def _dep(id, name, engine, model_ref, state, nodes, reserved, actual, *, alias=None, served=None, mode="always_on",
         config=None, estimate=None, litellm="synced", system=False, ondemand=None, port=None, created=None,
         priority=0, idle=None):
    cluster = len(nodes) > 1
    kind = "cluster" if cluster else "single"
    placement_label = " + ".join(LABEL[n] for n in nodes) + " (cluster)" if cluster else LABEL[nodes[0]]
    host = "192.0.2.11" if nodes[0] == "local" else "192.0.2.12"
    served = served or model_ref.split("/")[-1].lower()
    sleep_label = None
    if engine == "vllm":
        sleep_label = "sleep (stops all nodes)" if cluster else "sleep (stops the model)"
    d = {
        "id": id, "name": name, "engine": engine, "model_ref": model_ref, "state": state, "port": port,
        "priority": priority, "alias": alias, "reserved_bytes": g(reserved), "actual_bytes": g(actual) if actual else None,
        "estimate": estimate or {}, "config": config or {}, "litellm_status": litellm, "last_error": None,
        "exit_code": None, "container_id": None, "served_name": None if system else served,
        "adopted_from": None, "system": system, "mode": mode, "idle_timeout_s": idle, "pending_mode": None,
        "ondemand": ondemand, "startup_queue": None, "model_name": None if system else (alias or served),
        "litellm_error": None,
        "endpoint": f"http://dgxmm-{name}:{port}/v1" if port and not system else None,
        "host_endpoint": f"http://{host}:{port}/v1" if port and not system else None,
        "engine_key": False, "sleep_capable": False, "sleep_label": sleep_label,
        "placement": {"kind": kind, "nodes": list(nodes)}, "placement_label": placement_label,
        "rank_summary": [], "created_at": created or ago(days=12), "updated_at": ago(hours=3),
    }
    if cluster:
        d["rank_summary"] = [{"rank": i, "node": n, "node_label": LABEL[n], "role": "head" if i == 0 else "worker",
                              "status": "running" if state == "ready" else state} for i, n in enumerate(nodes)]
    return d


def deployments(scenario: str) -> list[dict]:
    deps = [
        _dep(1, "decider", "decision", "vllm-sr/d3-mini", "ready", ["local"], 13.0, 12.4, system=True, port=8100,
             config={"max_model_len": 4096, "max_num_seqs": 8},
             estimate=_estimate(9.0, 1.5, 0.3, 2.2, "measured")),
        _dep(3, "llama-3-1-8b", "vllm", "meta-llama/Llama-3.1-8B-Instruct", "ready", ["local"], 24.2, 23.5,
             alias="llama-3.1-8b-instruct", port=8102, priority=40,
             config=_vllm_cfg(16384, 64), estimate=_estimate(15.0, 4.0, 0.75, 4.5)),
        _dep(4, "qwen3-8-27b", "vllm", "Qwen/Qwen3.8-27B-FP8", "ready", ["dgx-2"], 38.6, 37.9, alias="default-chat",
             port=8101, priority=50,
             config=_vllm_cfg(32768, 32, enable_auto_tool_choice=True, tool_call_parser="hermes",
                              reasoning_parser="qwen3", kv_cache_dtype="fp8"),
             estimate=_estimate(27.4, 6.0, 0.7, 4.5)),
        _dep(5, "gemma-3-4b-gguf", "llamacpp", "gemma-3-4b-it-Q4_K_M.gguf", "ready",
             ["dgx-2"], 5.2, 4.3, served="gemma-3-4b-it-q4_k_m", port=8103, mode="on_demand",
             ondemand={"inflight": 1, "queued": 0, "last_request_at": NOW.timestamp() - 40, "phase": "serving"},
             config={"ctx_size": 16384, "parallel": 2, "n_gpu_layers": 999, "flash_attn": "on"},
             estimate=_estimate(2.3, 1.6, 0.3, 1.0)),
        _dep(6, "glm-4-7-flash", "vllm", "zai-org/GLM-4.7-Flash-FP8", "stopped", ["dgx-2"], 34.0, None,
             alias="glm-flash", mode="on_demand", idle=900, port=8104,
             ondemand={"inflight": 0, "queued": 0, "last_request_at": NOW.timestamp() - 5400, "phase": "sleeping"},
             config=_vllm_cfg(32768, 16), estimate=_estimate(29.0, 2.5, 0.5, 2.0)),
        _dep(7, "deepseek-v4-flash", "vllm", "deepseek-ai/DeepSeek-V4-Flash", "ready", ["local", "dgx-2"], 108.0, 105.2,
             alias="deepseek-v4-flash", port=8110, priority=60, created=ago(days=2),
             config=_vllm_cfg(65536, 16, kv_cache_dtype="fp8", enable_auto_tool_choice=True,
                              tool_call_parser="deepseek_v3", reasoning_parser="deepseek_r1"),
             estimate=_estimate(94.0, 8.0, 1.2, 4.8)),
        _dep(8, "qwen3-30b-a3b", "vllm", "Qwen/Qwen3-30B-A3B-Instruct-2507", "stopped", ["local"], 67.6, None,
             litellm="not_registered", port=None, config=_vllm_cfg(32768, 32), estimate=_estimate(56.8, 6.0, 0.3, 4.5)),
    ]
    if scenario == "before-cluster":  # the deploy wizard: DeepSeek-V4-Flash is the model being deployed
        deps = [d for d in deps if d["id"] != 7]
    return deps


EXTERNAL = {"local": 3.2, "dgx-2": 1.6}


def budget(node: str, deps: list[dict]) -> dict:
    used = []
    for d in deps:
        nodes = d["placement"]["nodes"]
        if node not in nodes or d["state"] in ("stopped", "failed"):
            continue
        share = 1 / len(nodes)
        used.append({"deployment_id": d["id"], "name": d["name"], "state": d["state"],
                     "reserved": int(d["reserved_bytes"] * share), "actual": int((d["actual_bytes"] or 0) * share)})
    committed = sum(u["reserved"] for u in used)
    actual = sum(u["actual"] for u in used)
    external = g(EXTERNAL[node])
    free = TOTAL - MARGIN - OS_RESERVE - committed - external
    return {"total": TOTAL, "available": TOTAL - actual - external - g(2.1), "margin": MARGIN, "os_reserve": OS_RESERVE,
            "usable": TOTAL - MARGIN - OS_RESERVE, "committed": committed, "managed_actual": actual,
            "external": external, "free_for_new": free, "deployments": used}


def live(dep_id: int, n: int = 60) -> list[dict]:
    base = {3: (720, 95, 0.97, 4, 22), 4: (310, 240, 0.52, 0, 8), 5: (66, 60, 0.18, 0, 1), 7: (54, 410, 0.41, 0, 4),
            1: (None, None, None, None, None)}[dep_id]
    out = []
    for i in range(n):
        ts = NOW.timestamp() - (n - 1 - i) * 5
        tps, ttft, kv, waiting, running = base
        if tps is None:
            out.append({"ts": ts})
            continue
        out.append({
            "ts": ts, "gen_tps": round(wave(i, tps, tps * 0.12, phase=dep_id), 1),
            "prompt_tps": round(wave(i, tps * 2.6, tps * 0.4, phase=dep_id + 1), 1),
            "ttft_p50": round(wave(i, ttft, ttft * 0.15, 13, dep_id) / 1000, 4),
            "ttft_p95": round(wave(i, ttft * 2.2, ttft * 0.3, 13, dep_id) / 1000, 4),
            "itl_p50": round(wave(i, 18 + dep_id, 2, 11, dep_id) / 1000, 4), "itl_p95": round(wave(i, 31 + dep_id, 4, 11) / 1000, 4),
            "kv_usage": max(0.0, min(0.99, wave(i, kv, 0.03, 17, dep_id))), "waiting": max(0, round(wave(i, waiting, 2, 7))),
            "running": max(1, round(wave(i, running, 2, 8, dep_id))), "prefix_hit_rate": 0.42,
            "spec_accept_rate": None, "preemptions": 0,
        })
    return out


def host_live(deps: list[dict], n: int = 60) -> list[dict]:
    b = budget("local", deps)
    out = []
    for i in range(n):
        ext = b["external"] + g(wave(i, 0.0, 0.5, 23))
        used = b["managed_actual"] + ext + g(2.1)
        out.append({"ts": NOW.timestamp() - (n - 1 - i) * 60, "used": used, "available": TOTAL - used,
                    "free_for_new": b["free_for_new"] - (ext - b["external"]), "external": ext,
                    "committed": b["committed"], "gpu_util": round(wave(i, 47, 15)), "temp_c": 54, "power_w": 61})
    return out


# --------------------------------------------------------------------------------------------------------------
# the request router: World methods decorated with @route(method, path regex)
# --------------------------------------------------------------------------------------------------------------

def route(method: str, pattern: str):
    def deco(fn):
        fn.route = (method, pattern)
        return fn
    return deco


class World:
    def __init__(self, scenario: str = "default"):
        self.scenario = scenario
        self.deps = deployments(scenario)
        self.routes = []
        for attr in dir(type(self)):
            fn = getattr(self, attr)
            spec = getattr(fn, "route", None)
            if spec:
                self.routes.append((spec[0], re.compile(spec[1] + "$"), fn))

    def handle(self, method, path, query, body):
        for m, rx, fn in self.routes:
            mt = rx.match(path)
            if m == method and mt:
                r = fn(query, body, *mt.groups())
                return r if isinstance(r, tuple) else (200, r)
        return 404, MISSING

    def dep(self, id) -> dict | None:
        return next((d for d in self.deps if d["id"] == int(id)), None)

    # ---- shell ----
    @route("GET", r"/auth/me")
    def me(self, q, b):
        return {"user": {"id": 1, "username": "admin", "role": "admin", "must_change_pw": False}, "csrf_token": "demo"}

    @route("GET", r"/host")
    def host(self, q, b):
        return {"budget": budget("local", self.deps), "external_top": EXTERNAL_TOP,
                "gpu": {"util": 47, "temp_c": 54, "power_w": 61}, "alert": {"level": "ok"},
                "disk": {"total": int(3.6 * 1024 * GIB), "free": int(2.4 * 1024 * GIB)},
                "ondemand": {"idle_timeout_s": 300, "queue_timeout_s": 120, "litellm_ready": True}}

    @route("GET", r"/host/metrics")
    def host_metrics(self, q, b):
        return host_live(self.deps)

    @route("GET", r"/nodes")
    def nodes(self, q, b):
        return [{"id": n, "label": label, "remote": remote, "reachable": True, "stale": False, "reason": None,
                 "budget": budget(n, self.deps)} for n, label, remote in NODES]

    @route("GET", r"/deployments")
    def deployments(self, q, b):
        return self.deps

    @route("GET", r"/deployments/(\d+)")
    def deployment(self, q, b, id):
        d = self.dep(id)
        return dict(d, events=EVENTS.get(d["id"], [])) if d else (404, {"detail": "not found"})

    @route("GET", r"/deployments/(\d+)/metrics")
    def metrics(self, q, b, id):
        d = self.dep(id)
        if not d or d["state"] != "ready":
            return []
        return live(int(id)) if q.get("window", "live") == "live" else live(int(id), 96)

    @route("GET", r"/containers/adoptable")
    def adoptable(self, q, b):
        return []

    @route("GET", r"/insights")
    def insights(self, q, b):
        rows = insights_for(self)
        if q.get("status"):
            rows = [r for r in rows if r["status"] in q["status"].split(",")]
        if q.get("subject"):
            rows = [r for r in rows if r["subject"] == q["subject"]]
        return rows

    @route("GET", r"/decider/status")
    def decider_status(self, q, b):
        return decider_status()


    # ---- deployment page ----
    @route("GET", r"/deployments/(\d+)/litellm")
    def dep_litellm(self, q, b, id):
        d = self.dep(id)
        return {"deployment_id": d["id"], "model_name": d["model_name"], "endpoint": ENDPOINT,
                "registered": d["litellm_status"] == "synced", "status": d["litellm_status"], "enabled": True,
                "error": None, "keys": [k for k in KEYS if not k["models"] or d["model_name"] in k["models"]],
                "keys_visible": True, "keys_error": None}

    @route("GET", r"/deployments/(\d+)/diagnosis")
    def diagnosis(self, q, b, id):
        return {"diagnosis": None}

    @route("GET", r"/deployments/(\d+)/schedules")
    def schedules(self, q, b, id):
        return {"deployment_id": int(id), "schedules": SCHEDULES.get(int(id), []),
                "allowed_actions": ["sleep", "wake", "stop", "start"], "timezone": "UTC", "paused_all": False}

    @route("GET", r"/schedules/settings")
    def schedule_settings(self, q, b):
        return {"paused_all": False, "timezone": "UTC"}

    @route("GET", r"/deployments/(\d+)/workload")
    def workload(self, q, b, id):
        return {"label": "chat", "source": "model", "confidence": 0.88, "summary": None, "reason": None, "decision_id": 41}

    @route("GET", r"/deployments/(\d+)/advice")
    def advice(self, q, b, id):
        return []

    @route("GET", r"/deployments/(\d+)/benchmarks")
    def benchmarks(self, q, b, id):
        return []

    @route("GET", r"/deployments/(\d+)/logs")
    def logs(self, q, b, id):
        return {"logs": ""}

    # ---- deploy wizard ----
    @route("GET", r"/inventory")
    def inventory(self, q, b):
        return INVENTORY

    @route("GET", r"/engines")
    def engines(self, q, b):
        return ENGINES

    @route("GET", r"/downloads")
    def downloads(self, q, b):
        return []

    @route("POST", r"/deployments/placement-options")
    def placement_options(self, q, b):
        need = wizard_need(b)
        free = {n: budget(n, self.deps)["free_for_new"] for n, _, _ in NODES}
        opts = []
        for n, label, _ in NODES:
            fits = need <= free[n]
            opts.append({"placement": {"kind": "single", "nodes": [n]}, "label": label, "per_node": {n: need},
                         "free": {n: free[n]}, "fits": fits, "recommended": False,
                         "reason": "fits" if fits else ("too_big_for_one_node" if need > TOTAL - MARGIN - OS_RESERVE - g(EXTERNAL[n]) - g(20) else "insufficient_memory")})
        share = need // 2  # the per-rank share: the model split over the two nodes
        both = all(share <= free[n] for n, _, _ in NODES)
        opts.append({"placement": {"kind": "cluster", "nodes": ["local", "dgx-2"]}, "label": "dgx-1 + dgx-2 (cluster)",
                     "per_node": {"local": share, "dgx-2": share}, "free": free, "fits": both, "recommended": False,
                     "reason": "fits" if both else "insufficient_memory"})
        best = next((o for o in opts if o["fits"]), None)
        if best:
            best["recommended"] = True
        return opts

    @route("POST", r"/deployments/preview")
    def preview(self, q, b):
        ref = b.get("model_ref", "")
        est = dict(WIZARD.get(ref, WIZARD["Qwen/Qwen3-8B"])["estimate"])
        pl = b.get("placement") or {"kind": "single", "nodes": ["local"]}
        node = pl["nodes"][0]
        free = budget(node, self.deps)["free_for_new"]
        need = est["total"] if pl["kind"] == "single" else est["total"] // 2
        fit = {"fits": need <= free, "required": need, "free_for_new": free, "headroom": free - need, "estimate": est}
        return {"estimate": est, "fit": fit, "budget": budget(node, self.deps), "served_name": ref.split("/")[-1].lower(),
                "config": b.get("config", {}), "gpu_memory_utilization": round(need / TOTAL, 2),
                "kv_pool_tokens": 65536, "plans": [] if fit["fits"] else PLANS,
                "llmfit": None, "notes": [] if fit["fits"] else PLAN_NOTES}

    @route("POST", r"/decider/engine-suggest")
    def engine_suggest(self, q, b):
        return {"engine": "vllm", "model_ref": b["model_ref"], "source": "rules", "confidence": None,
                "reason": "safetensors_only", "decision_id": None, "alternatives": {}, "disabled": False}

    @route("POST", r"/decider/config-advice")
    def config_advice(self, q, b):
        w = WIZARD.get(b["model_ref"])
        if not w or not w["options"]:
            return {"goal": b["goal"], "options": [], "recommended": None, "top": [], "source": "none", "confidence": None,
                    "decision_id": None, "free_bytes": None, "reason": None, "disabled": False}
        return {"goal": b["goal"], "options": w["options"], "recommended": w["options"][0]["key"],
                "top": [o["key"] for o in w["options"]], "source": "model", "confidence": 0.89, "decision_id": 52,
                "free_bytes": g(69.3), "reason": None, "disabled": False}

    @route("POST", r"/decider/suggest-config")
    def suggest_config(self, q, b):
        s = WIZARD.get(b["model_ref"], {}).get("parsers")
        return s

    @route("POST", r"/decider/draft-suggest")
    def draft_suggest(self, q, b):
        return {"method": b["method"], "candidates": [{"ref": "Qwen/Qwen3-0.6B", "weights_bytes": g(1.4),
                "reason": "same_tokenizer", "summary": "Same tokenizer and vocabulary", "p": None}],
                "suggested": "Qwen/Qwen3-0.6B", "source": "rules", "confidence": None, "decision_id": None,
                "reason": None, "disabled": False}

    @route("POST", r"/decider/remote-code-check")
    def remote_code(self, q, b):
        f = lambda line, kind: {"file": "modeling_demo.py", "line": line, "kind": kind, "detail": ""}
        return {"verdict": "unusual", "findings": [f(4, "process_import"), f(5, "network_import"), f(12, "process_call"),
                f(16, "credential_env"), f(18, "file_write_outside")], "source": "rules", "confidence": None,
                "model_label": None, "decision_id": None, "scanned_files": 2, "truncated": False, "error": None,
                "disabled": False, "remote_code": True}

    @route("GET", r"/integrations/litellm")
    def litellm(self, q, b):
        return LITELLM_SETTINGS

    @route("GET", r"/integrations/litellm/status")
    def litellm_status(self, q, b):
        return LITELLM_SETTINGS["status"]

    @route("GET", r"/integrations/litellm/models")
    def litellm_models(self, q, b):
        rows = [{"id": "ext-1", "model_name": "gpt-4o-mini", "model": "openai/gpt-4o-mini", "api_base": "https://api.openai.com/v1",
                 "managed_by_dgxmm": False, "deployment": None, "db_model": True}]
        for d in self.deps:
            if d["system"] or d["litellm_status"] != "synced":
                continue
            host = "192.0.2.11" if d["placement"]["nodes"][0] == "local" else "192.0.2.12"
            rows.append({"id": f"dgxmm-{d['id']}", "model_name": d["model_name"], "model": f"openai/{d['served_name']}",
                         "api_base": f"http://{host}:{d['port']}/v1", "managed_by_dgxmm": True,
                         "deployment": {"id": d["id"], "name": d["name"], "state": d["state"]}, "db_model": True})
        return rows

    @route("GET", r"/integrations/litellm/keys")
    def litellm_keys(self, q, b):
        return KEYS

    @route("GET", r"/decider/routing-advice")
    def routing_advice(self, q, b):
        return ROUTING

    # ---- other pages ----
    @route("GET", r"/host/processes")
    def processes(self, q, b):
        return processes()

    @route("GET", r"/catalog/search")
    def catalog_search(self, q, b):
        return {"results": CATALOG, "free_for_new": budget("local", self.deps)["free_for_new"], "banner": None}

    @route("GET", r"/catalog/details")
    def catalog_details(self, q, b):
        return CATALOG_DETAILS

    @route("GET", r"/catalog/compat")
    def catalog_compat(self, q, b):
        return {"repo": q.get("repo"), "sha": "c296ed0a908dbc97718d2b2ff8de659b2fedf052", "architecture": "Qwen3ForCausalLM",
                "gated": False, "vllm": {"verdict": "runs", "reasons": [], "flags": []},
                "llamacpp": {"verdict": "unsupported", "reasons": ["no GGUF files in this repo"], "flags": []},
                "source": "rules", "confidence": None, "decision_id": None, "cached": True, "error": None, "disabled": False}

    @route("GET", r"/decider/decisions")
    def decisions(self, q, b):
        return DECISIONS

    @route("POST", r"/decider/command")
    def command(self, q, b):
        return {"interpretations": [{"action": "sleep", "target": {"kind": "deployment", "name": "qwen3-8-27b", "id": 4,
                "state": "ready", "engine": "vllm"}, "confidence": 0.97, "label": "Sleep qwen3-8-27b", "source": "model",
                "params": {}}], "source": "model", "message": None, "refused": False}


EXTERNAL_TOP = [
    {"pid": 41210, "name": "python3", "rss": g(2.4), "gpu": g(0.0), "command": "python3 train_lora.py --base Qwen/Qwen3-0.6B --epochs 3 --wandb-key *** --out runs/lora-demo",
     "user": "dgxuser", "user_scope": "host", "container": None, "dgxmm_label": None},
    {"pid": 2231, "name": "jupyter-lab", "rss": g(0.8), "gpu": 0, "command": "/opt/conda/bin/python /opt/conda/bin/jupyter-lab --ip=0.0.0.0 --no-browser --ServerApp.token=***",
     "user": "uid 1000", "user_scope": "container", "container": {"id": "3f2a9c1b7d40", "name": "jupyter", "image": "quay.io/jupyter/pytorch-notebook:latest"}, "dgxmm_label": None},
]


# --------------------------------------------------------------------------------------------------------------
# page data
# --------------------------------------------------------------------------------------------------------------

VLLM_IMAGE = "vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8"
LLAMACPP_IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:e4cfff28678327b0ae7eda129e700e56a0fa55aa23db2f016256ea155c15c2b5"


def _art(ref, kind, weights, *, fmt="safetensors", quant="bf16", arch=None, ctx=None, moe=False, engines=("vllm",),
         rev=None, source="hf", remote=False, kind_source="rules", kind_conf=None):
    return {"ref": ref, "source": source, "revision": rev, "format": fmt, "weights_bytes": g(weights),
            "size_bytes": g(weights * 1.01), "complete": True, "kind": kind, "architecture": arch, "quant": quant,
            "problems": [], "max_context": ctx, "is_moe": moe, "num_layers": None, "confident": True, "has_mtp": False,
            "engines": list(engines), "has_remote_code": remote, "kind_source": kind_source, "kind_confidence": kind_conf}


INVENTORY = [
    _art("gemma-3-4b-it-Q4_K_M.gguf", "llm", 2.3, fmt="gguf", quant="gguf:Q4_K_M", arch="gemma3", ctx=131072,
         engines=("llamacpp",), source="gguf"),
    _art("BAAI/bge-m3", "other", 2.1, quant="fp32", arch="XLMRobertaModel", rev="5617a9f61b028005a4858fdac845db406aefb181",
         engines=(), kind_source="model", kind_conf=0.91),
    _art("deepseek-ai/DeepSeek-V4-Flash", "llm", 94.0, quant="fp8", arch="DeepseekV4ForCausalLM", ctx=131072, moe=True,
         rev="4d1c7e0a9b3f62d85e17c0b9a4f3e2d1c6b5a798"),
    _art("example-org/phi3-custom-code-demo", "llm", 7.1, arch="Phi3ForCausalLM", ctx=4096,
         rev="afbdf6d0c1e2a3b4c5d6e7f8091a2b3c4d5e6f70", remote=True),
    _art("meta-llama/Llama-3.1-8B-Instruct", "llm", 15.0, arch="LlamaForCausalLM", ctx=131072,
         rev="1f290537e221ffae2bf10289e227bfe7054cdbdf"),
    _art("Qwen/Qwen3-0.6B", "llm", 1.4, arch="Qwen3ForCausalLM", ctx=40960, rev="01e764fa5f0d8c4b27e9c03a6d1f2e3b4a5c6d7e"),
    _art("Qwen/Qwen3-30B-A3B-Instruct-2507", "llm", 56.8, arch="Qwen3MoeForCausalLM", ctx=262144, moe=True,
         rev="ffb539cf0e1d2c3b4a5968778695a4b3c2d1e0f9"),
    _art("Qwen/Qwen3-8B", "llm", 15.3, arch="Qwen3ForCausalLM", ctx=40960, rev="c296ed0a908dbc97718d2b2ff8de659b2fedf052"),
    _art("Qwen/Qwen3.8-27B-FP8", "llm", 27.4, quant="fp8", arch="Qwen3ForCausalLM", ctx=131072,
         rev="7b2e9d4c1a0f38e56d7c9b0a1e2f3d4c5b6a7980"),
    _art("vllm-sr/d3-mini", "other", 9.0, arch=None, rev="97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e", engines=()),
    _art("zai-org/GLM-4.7-Flash-FP8", "llm", 29.0, quant="fp8", arch="Glm4MoeForCausalLM", ctx=131072, moe=True,
         rev="3c8a1f0e9d2b47c6a5e1d0f9c8b7a6e5d4c3b2a1"),
]

ENGINES = [
    {"name": "vllm", "supports_sleep": True, "image": VLLM_IMAGE,
     "defaults": {"max_model_len": 32768, "max_num_seqs": 8, "kv_cache_dtype": "auto", "kv_tokens": None,
                  "max_num_batched_tokens": 8192, "enable_prefix_caching": True, "enable_chunked_prefill": True,
                  "enable_sleep_mode": True, "enable_auto_tool_choice": False, "tool_call_parser": None,
                  "reasoning_parser": None, "served_model_name": None, "trust_remote_code": False,
                  "safetensors_load_strategy": ""}},
    {"name": "llamacpp", "supports_sleep": False, "image": LLAMACPP_IMAGE,
     "defaults": {"ctx_size": 16384, "parallel": 2, "cache_type_k": "f16", "cache_type_v": "f16", "flash_attn": "auto",
                  "n_gpu_layers": 999}},
]


def _opt(key, summary, total, ctx, seqs, kv="fp8"):
    return {"key": key, "config": {"max_model_len": ctx, "max_num_seqs": seqs, "kv_cache_dtype": kv, "enable_prefix_caching": True},
            "summary": summary, "required_bytes": g(total), "headroom_bytes": g(69.3 - total), "context": ctx, "seqs": seqs,
            "kv_cache_dtype": kv, "prefix_caching": True, "kv_pool_tokens": 65536}


WIZARD = {
    "Qwen/Qwen3-8B": {
        "estimate": _estimate(15.3, 9.0, 0.65, 4.5),
        "options": [_opt("chat-40k", "40k context · 32 concurrent · fp8 KV · prefix caching — needs 29.4 GiB, leaves 39.9 GiB free", 29.4, 40960, 32),
                    _opt("chat-32k", "32k context · 16 concurrent · fp8 KV · prefix caching — needs 24.1 GiB", 24.1, 32768, 16)],
        "parsers": {
            "tool_call_parser": {"value": "hermes", "source": "rules", "confidence": None, "auto_apply": False,
                                 "alternatives": [], "decision_id": None, "reason": None},
            "reasoning_parser": {"value": "qwen3", "source": "rules", "confidence": None, "auto_apply": True,
                                 "alternatives": [], "decision_id": None, "reason": None}},
    },
    "deepseek-ai/DeepSeek-V4-Flash": {"estimate": _estimate(94.0, 8.0, 1.2, 4.8), "options": [], "parsers": None},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"estimate": _estimate(56.8, 6.0, 0.33, 4.5), "options": [], "parsers": None},
    "example-org/phi3-custom-code-demo": {"estimate": _estimate(7.1, 3.0, 0.4, 3.5), "options": [], "parsers": None},
}

# the plans offered when Qwen3-30B-A3B does not fit on dgx-1 (the plan-chooser screenshot)
PLANS = [
    {"id": "stop:3,7", "title": "Stop llama-3-1-8b, deepseek-v4-flash", "detail": "Frees about 78.2 GiB on dgx-1 (deepseek-v4-flash stops on both nodes). Can be restarted later.",
     "new_required": g(67.6), "frees": g(78.2), "fits": True, "actions": [{"type": "stop", "deployment_id": 3}, {"type": "stop", "deployment_id": 7}],
     "patch": None, "recommended": False, "recommendation": None,
     "unsafe_reason": "would stop deepseek-v4-flash, which has a higher priority (60) than this request (0)"},
    {"id": "patch:kv", "title": "Use a smaller KV-cache dtype", "detail": "Set kv_cache_dtype=fp8; quality impact is usually negligible.",
     "new_required": g(64.6), "frees": 0, "fits": False, "actions": [], "patch": {"kv_cache_dtype": "fp8"}, "recommended": False,
     "recommendation": None, "unsafe_reason": "does not free enough memory"},
    {"id": "stop:3", "title": "Stop llama-3-1-8b", "detail": "Frees about 24.2 GiB. Can be restarted later.",
     "new_required": g(67.6), "frees": g(24.2), "fits": False, "actions": [{"type": "stop", "deployment_id": 3}], "patch": None,
     "recommended": False, "recommendation": None, "unsafe_reason": "does not free enough memory"},
]
PLAN_NOTES = ["The decision model holds 13 GiB on dgx-1 — disable it on the Decision model page to free it"]


def wizard_need(body) -> int:
    return WIZARD.get(body.get("model_ref"), WIZARD["Qwen/Qwen3-8B"])["estimate"]["total"]


LITELLM_SETTINGS = {
    "enabled": True, "url": "http://litellm-proxy:4000", "public_url": "https://llm.example.com", "endpoint": ENDPOINT,
    "master_key_set": True, "master_key_source": "secret_file", "master_key_hint": None,
    "default_url": "http://litellm-proxy:4000",
    "status": {"reachable": True, "version": "1.105.1", "db": "connected", "error": None, "checked_at": ago(minutes=1)},
}


def _key(token, alias, models, budget, rpm=None, tpm=None, tail="D2PA"):
    return {"token": token, "key_name": f"sk-...{tail}", "key_alias": alias, "models": models, "spend": 0.0,
            "max_budget": budget, "budget_duration": "30d" if budget else None, "rpm_limit": rpm, "tpm_limit": tpm,
            "expires": None, "created_at": ago(days=9), "blocked": False, "metadata": None, "created_by_dgxmm": True}


KEYS = [
    _key("a" * 64, "team-chat", ["default-chat"], 50.0, rpm=120),
    _key("b" * 64, "ci-pipeline", [], None, tpm=200000, tail="0JKQ"),
    _key("c" * 64, "research-agents", ["deepseek-v4-flash", "default-chat"], 200.0, rpm=60, tail="7wXe"),
]


def _cls(cls, label, alias, dep, model_name, source, conf, reason, cands=0):
    ids = {"qwen3-8-27b": 4, "deepseek-v4-flash": 7, "llama-3-1-8b": 3}
    return {"class": cls, "label": label, "alias": alias, "deployment_id": ids.get(dep), "deployment": dep,
            "model_name": model_name, "model_ref": None, "source": source, "confidence": conf, "reason": reason,
            "routable": dep is not None, "note": None,
            "candidates": [{"deployment_id": i, "deployment": n, "model_name": n} for n, i in list(ids.items())[:cands]]}


ROUTING_CLASSES = [
    _cls("code", "Code generation and review", "dgxmm-code", "deepseek-v4-flash", "deepseek-v4-flash", "model", 0.87, "largest code-capable model", 3),
    _cls("chat", "General chat and assistants", "dgxmm-chat", "qwen3-8-27b", "default-chat", "model", 0.91, "chat workload", 3),
    _cls("tool_use", "Agents and tool calling", "dgxmm-tool-use", "qwen3-8-27b", "default-chat", "rules", None, "tool-call parser set", 2),
    _cls("long_context", "Long documents and RAG (32k tokens or more per request)", "dgxmm-long-context", "deepseek-v4-flash", "deepseek-v4-flash", "rules", None, "64k context", 0),
    _cls("vision", "Images in the prompt", "dgxmm-vision", None, None, "none", None, "no vision model deployed", 0),
    _cls("batch", "Batch jobs and many concurrent clients", "dgxmm-batch", "llama-3-1-8b", "llama-3.1-8b-instruct", "model", 0.95, "highest throughput", 2),
]
ROUTING_SNIPPET = """# Suggested by DGXMM (routing advice); DGXMM changes nothing in LiteLLM. Merge into your existing
# router_settings in LiteLLM's config.yaml (do not add a second router_settings key).
router_settings:
  model_group_alias:
    dgxmm-code: "deepseek-v4-flash"  # code -> deployment "deepseek-v4-flash"
    dgxmm-chat: "default-chat"  # chat -> deployment "qwen3-8-27b"
    dgxmm-tool-use: "default-chat"  # tool_use -> deployment "qwen3-8-27b"
    dgxmm-long-context: "deepseek-v4-flash"  # long_context -> deployment "deepseek-v4-flash"
    dgxmm-batch: "llama-3.1-8b-instruct"  # batch -> deployment "llama-3-1-8b"
"""
ROUTING = {"enabled": True, "classes": ROUTING_CLASSES, "snippet": ROUTING_SNIPPET, "computed_at": ago(minutes=1), "source": "model"}


def _proc(pid, name, rss, gpu, command, container=None, managed=None, user="root", scope="container"):
    return {"pid": pid, "ppid": 1, "uid": 0, "user": user, "user_scope": scope, "name": name, "exe": f"/usr/bin/{name}",
            "command": command, "command_truncated": False, "container": container, "managed": managed,
            "dgxmm_label": None, "rss": g(rss), "gpu": g(gpu)}


def processes() -> dict:
    c_llama = {"id": "9b1f0c2d3e4a", "name": "dgxmm-llama-3-1-8b", "image": VLLM_IMAGE}
    c_ds = {"id": "4e5f6a7b8c9d", "name": "dgxmm-deepseek-v4-flash-r0", "image": VLLM_IMAGE}
    c_dec = {"id": "1a2b3c4d5e6f", "name": "dgxmm-decider", "image": "dgxmm-decider:3"}
    c_jup = {"id": "3f2a9c1b7d40", "name": "jupyter", "image": "quay.io/jupyter/pytorch-notebook:latest"}
    procs = [
        _proc(3121, "python3", 2.8, 20.7, "python3 --model meta-llama/Llama-3.1-8B-Instruct --revision 1f290537e221ffae2bf10289e227bfe7054cdbdf --served-model-name llama-3.1-8b-instruct --host 0.0.0.0 --port 8102 --max-model-len 16384", c_llama, {"deployment_id": 3, "name": "llama-3-1-8b"}),
        _proc(5210, "python3", 3.9, 50.1, "python3 --model deepseek-ai/DeepSeek-V4-Flash --tensor-parallel-size 2 --distributed-executor-backend ray --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8110 --kv-cache-dtype fp8", c_ds, {"deployment_id": 7, "name": "deepseek-v4-flash"}),
        _proc(5233, "raylet", 0.6, 0.0, "/usr/local/lib/python3.12/dist-packages/ray/core/src/ray/raylet/raylet --node_ip_address=192.0.2.11", c_ds, {"deployment_id": 7, "name": "deepseek-v4-flash"}),
        _proc(2210, "python3", 1.5, 10.9, "python3 --port 8100 --batch-size 8", c_dec, {"deployment_id": 1, "name": "decider"}),
        _proc(41210, "python3", 2.4, 0.4, "python3 train_lora.py --base Qwen/Qwen3-0.6B --epochs 3 --wandb-key *** --out runs/lora-demo", None, None, "dgxuser", "host"),
        _proc(1022, "dockerd", 0.1, 0.0, "/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock", None, None, "root", "host"),
        _proc(2231, "python", 0.8, 0.0, "/opt/conda/bin/python /opt/conda/bin/jupyter-lab --ip=0.0.0.0 --no-browser --ServerApp.token=***", c_jup, None, "uid 1000"),
        _proc(2240, "python", 1.8, 0.0, "/opt/conda/bin/python -m ipykernel_launcher -f /tmp/jupyter-runtime/kernel-1.json", c_jup, None, "uid 1000"),
    ]
    groups = {}
    for p in procs:
        key = p["container"]["id"] if p["container"] else "host"
        gp = groups.setdefault(key, {**(p["container"] or {"id": "host", "name": None, "image": None}), "rss": 0, "gpu": 0,
                                     "process_count": 0, "managed": p["managed"], "dgxmm_label": None})
        gp["rss"] += p["rss"]; gp["gpu"] += p["gpu"]; gp["process_count"] += 1
    return {"scope": "all", "total": len(procs), "processes": procs, "containers": list(groups.values()),
            "host": {"rss": sum(p["rss"] for p in procs), "gpu": sum(p["gpu"] for p in procs), "process_count": len(procs)},
            "duration_ms": 9}


def _cat(repo, downloads, est, badge, quant=None, vllm=True, llamacpp=False, downloaded=False):
    return {"repo_id": repo, "downloads": downloads, "est_bytes": g(est) if est else None, "fit": {"badge": badge},
            "support": {"vllm": vllm, "llamacpp": llamacpp}, "quant": quant, "gated": False, "downloaded": downloaded,
            "llmfit": None}


CATALOG = [
    _cat("Qwen/Qwen3-8B", 9122623, 15.3, "fits_now", downloaded=True),
    _cat("Qwen/Qwen3-8B-AWQ", 2170448, 5.7, "fits_now", quant="AWQ"),
    _cat("Qwen/Qwen3-8B-Base", 758046, 15.3, "fits_now"),
    _cat("Qwen/Qwen3-8B-GGUF", 607951, None, "unknown", vllm=False, llamacpp=True),
    _cat("Qwen/Qwen3-8B-FP8", 511163, 8.8, "fits_now", quant="FP8"),
    _cat("nvidia/Qwen3-8B-NVFP4", 131217, 6.0, "fits_now", quant="NVFP4"),
    _cat("unsloth/Qwen3-8B-GGUF", 117465, None, "unknown", vllm=False, llamacpp=True),
    _cat("example-org/Qwen3-8B-demo-finetune", 20311, 15.3, "fits_now"),
    _cat("unsloth/Qwen3-8B", 18340, 15.3, "fits_now"),
    _cat("Qwen/Qwen3-8B-MLX-4bit", 9812, None, "unknown", vllm=False),
    _cat("example-org/Qwen3-8B-long-context", 4410, 15.3, "fits_after_evict"),
    _cat("example-org/Qwen3-8B-instruct-merge", 2304, 15.3, "fits_now"),
    _cat("example-org/Qwen3-8B-AWQ-int4", 1290, 5.7, "fits_now", quant="AWQ"),
]
CATALOG_DETAILS = {
    "repo_id": "Qwen/Qwen3-8B", "architecture": "Qwen3ForCausalLM", "quant": "bf16", "max_context": 40960, "gated": False,
    "estimate": {"total": g(29.4)}, "fit": {"badge": "fits_now"}, "llmfit": None, "downloaded": True,
    "support": {"vllm": True, "llamacpp": False}, "gguf_files": [],
    "variants": [{"repo_id": "Qwen/Qwen3-8B-AWQ", "quant": "AWQ", "est_bytes": g(5.7)},
                 {"repo_id": "Qwen/Qwen3-8B-GGUF", "quant": None, "est_bytes": None},
                 {"repo_id": "Qwen/Qwen3-8B-FP8", "quant": "FP8", "est_bytes": g(8.8)},
                 {"repo_id": "nvidia/Qwen3-8B-NVFP4", "quant": "NVFP4", "est_bytes": g(6.0)},
                 {"repo_id": "unsloth/Qwen3-8B-GGUF", "quant": None, "est_bytes": None},
                 {"repo_id": "unsloth/Qwen3-8B", "quant": None, "est_bytes": g(15.3)}],
}

DECISIONS = [
    {"id": 60 - i, "created_at": ago(minutes=3 + i * 7), "feature": f, "subject": s, "source": src, "choice": c,
     "confidence": conf, "latency_ms": lat, "feedback": fb}
    for i, (f, s, src, c, conf, lat, fb) in enumerate([
        ("health", "deployment:3", "rules", "kv_pressure", None, None, None),
        ("schedule", "deployment:4", "model", "propose", 0.89, 44, None),
        ("routing", "class:code", "model", "deepseek-v4-flash", 0.87, 39, "accepted"),
        ("config", "Qwen/Qwen3-8B", "model", "chat-40k", 0.89, 41, "accepted"),
        ("parser_reasoning", "Qwen/Qwen3-8B", "rules", "qwen3", None, None, "accepted"),
        ("command", "command", "model", "sleep", 0.97, 37, None),
        ("kind", "BAAI/bge-m3", "model", "embedding", 0.91, 35, None),
        ("workload", "deployment:4", "model", "chat", 0.88, 40, None),
    ])
]

SCHEDULES = {
    4: [
        {"id": 11, "deployment_id": 4, "action": "wake", "days": 1, "days_label": "Mon", "at_local": "07:45", "timezone": "UTC",
         "enabled": True, "created_by": 1, "created_at": ago(days=3), "updated_at": ago(days=3), "last_run_at": ago(days=3, hours=7, minutes=35),
         "last_result": "ok", "next_run_at": "2026-11-16T07:45:00+00:00", "deferred": False, "pending": False},
        {"id": 10, "deployment_id": 4, "action": "sleep", "days": 32, "days_label": "Sat", "at_local": "16:00", "timezone": "UTC",
         "enabled": True, "created_by": 1, "created_at": ago(days=3), "updated_at": ago(days=3), "last_run_at": None,
         "last_result": None, "next_run_at": "2026-11-14T16:00:00+00:00", "deferred": False, "pending": False},
    ],
}

EVENTS = {
    7: [{"ts": ago(days=2), "from": "planned", "to": "starting", "message": "starting 2 ranks: dgx-1 (head), dgx-2 (worker)", "log_tail": None},
        {"ts": ago(days=2, minutes=-6), "from": "starting", "to": "ready", "message": "ready after 6 min 12 s", "log_tail": None}],
}

KV_BODY = ("KV-cache usage was at or above 95 % in 100 % of the busy time over the last minute (p95 99 %) and 8 requests were "
           "preempted. Suggested: use an fp8 KV cache (about twice the tokens in the same memory); lower max_num_seqs from "
           "64 to 32 (fewer requests compete for the cache; the rest queue).")
SCHEDULE_BODY = ("qwen3-8-27b was idle in the same 4 windows every week for 3+ observed weeks. A schedule would stop it at 20:00 "
                 "(Mon–Thu) and start it again at 07:45 (Tue–Fri), UTC: about 47 hours a week, freeing about 38.6 GiB on dgx-2. "
                 "It never acts while requests arrive, and you can pause or delete it on the deployment's page.")
ROUTING_BODY = """Suggested LiteLLM aliases per traffic class (advisory: DGXMM does not change LiteLLM):

- dgxmm-code (code generation and review): deepseek-v4-flash on deployment deepseek-v4-flash (decision model, 87%)
- dgxmm-chat (general chat and assistants): default-chat on deployment qwen3-8-27b (decision model, 91%)
- dgxmm-tool-use (agents and tool calling): default-chat on deployment qwen3-8-27b (rules)
- dgxmm-long-context (long documents and RAG): deepseek-v4-flash on deployment deepseek-v4-flash (rules)
- dgxmm-batch (batch jobs and many concurrent clients): llama-3.1-8b-instruct on deployment llama-3-1-8b (decision model, 95%)

No deployment fits: dgxmm-vision.

Copy the snippet into LiteLLM's config to route these aliases (it holds only deployments LiteLLM serves now); the LiteLLM page lists the same advice."""
TUNING_ROWS = [("0.50", 102, 96), ("0.55", 99, 96), ("0.60 (proposed)", 96, 96), ("0.65", 96, 96), ("0.70", 84, 84),
               ("0.75", 69, 69), ("0.80 (current)", 54, 54), ("0.85", 39, 39), ("0.90", 24, 24), ("0.95", 10, 10)]


def _wilson(k, n, z=1.96):
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


TUNING_BODY = ("Lowering the Failure diagnosis threshold from 0.80 to 0.60 is supported by 102 judged model answers of the last "
               "30 days: at 0.60 the Wilson 95 % lower bound of their precision reaches the target of 95 %.\n\n"
               "threshold | answers | accepted | precision | Wilson 95 % lower bound\n--- | --- | --- | --- | ---\n"
               + "\n".join(f"{t} | {n} | {k} | {100 * k / n:.1f} % | {100 * _wilson(k, n):.1f} %" for t, n, k in TUNING_ROWS)
               + "\n\nApply sets the threshold through the Decision model settings (audited); nothing changes until then.")


def _insight(id, kind, feature, subject, severity, title, body, source, conf, action, minutes):
    return {"id": id, "kind": kind, "feature": feature, "subject": subject, "severity": severity, "title": title, "body": body,
            "confidence": conf, "source": source, "decision_id": None, "action": action, "status": "open",
            "snooze_until": None, "dedupe_key": f"{kind}:{subject}", "created_at": ago(minutes=minutes),
            "updated_at": ago(minutes=minutes), "expires_at": None}


def insights_for(world) -> list[dict]:
    return [
        _insight(21, "health", "health", "deployment:3", "warning", "KV cache pressure on llama-3-1-8b", KV_BODY, "rules", None,
                 {"type": "reconfigure", "deployment_id": 3, "patch": {"kv_cache_dtype": "fp8", "max_num_seqs": 32},
                  "label": "Reconfigure…"}, 0),
        _insight(20, "schedule", "schedule", "deployment:4", "info", "Sleep qwen3-8-27b on a schedule?", SCHEDULE_BODY, "model", 0.89,
                 {"type": "schedule", "deployment_id": 4, "proposal": {"sleep_action": "stop", "sleep_at": "20:00", "sleep_days": 15,
                  "wake_action": "start", "wake_at": "07:45", "wake_days": 30, "timezone": "UTC", "idle_hours_per_week": 47, "weeks": 3},
                  "label": "Review schedule"}, 0),
        _insight(19, "routing", "routing", "routing", "info", "Routing advice for 5 traffic classes", ROUTING_BODY, "model", 0.87,
                 {"type": "copy", "text": ROUTING_SNIPPET, "label": "Copy LiteLLM snippet"}, 1),
        _insight(18, "tuning", "tuning", "feature:diagnosis", "info", "Failure diagnosis: threshold 0.80 → 0.60 suggested", TUNING_BODY,
                 "rules", None, {"type": "threshold", "feature": "diagnosis", "value": 0.6, "label": "Set threshold to 0.60"}, 1),
    ]


FEATURES = [  # name, label, description, threshold (None: rules only), background, viewer_visible, members
    ("diagnosis", "Failure diagnosis", "Explain why a deployment failed and suggest a fix.", 0.8, True, True, ["diagnosis"]),
    ("parser", "Parser suggestions", "Suggest tool-call and reasoning parsers in the deploy wizard.", 0.8, False, False, ["parser", "parser_tool", "parser_reasoning"]),
    ("plan", "Plan recommendation", "Mark the best safe plan when a model does not fit.", 0.6, False, False, ["plan"]),
    ("kind", "Model kind", "Classify models no rule recognises (embedding, reranker, …).", 0.8, True, True, ["kind"]),
    ("workload", "Workload-aware advice", "Label each deployment's workload and rank its advice.", 0.75, True, True, ["workload"]),
    ("health", "Health triage", "Health triage of running deployments.", 0.85, True, True, ["health"]),
    ("config", "Config advice", "Suggest a sizing for the deployment's goal in the deploy wizard (from the memory calculator).", 0.8, False, False, ["config"]),
    ("compat", "Compatibility check", "Say whether a Hugging Face repo runs on vLLM / llama.cpp here, before it is downloaded.", 0.8, False, True, ["compat"]),
    ("draft", "Draft pairing", "Suggest an on-disk draft model for speculative decoding.", 0.8, False, False, ["draft"]),
    ("engine", "Engine choice", "Suggest vLLM or llama.cpp when a model is on disk in both formats.", 0.8, False, False, ["engine"]),
    ("command", "Natural-language commands", "Natural-language commands in the ⌘K palette.", 0.7, False, True, ["command"]),
    ("schedule", "Sleep/wake schedules", "Usage-pattern sleep/wake suggestions with user-owned schedules.", 0.8, True, True, ["schedule"]),
    ("reading", "Reading labels", "Benchmark/calibration reading outlier labels.", 0.95, True, True, ["reading"]),
    ("tuning", "Threshold tuning", "Threshold tuning from feedback (rules only).", 0.8, True, False, ["tuning"]),
    ("routing", "Routing advice", "LiteLLM routing advice (advisory only — no live routing).", 0.8, True, False, ["routing"]),
    ("remote_code", "Remote-code risk", "Scan a repo's custom code before trust_remote_code runs it, and flag unusual code.", 0.8, False, False, ["remote_code"]),
]


def _job(name, feature, schedule, minutes_ago, ms, result, next_in):
    return {"name": name, "feature": feature, "schedule": schedule, "enabled": True, "running": False,
            "last_run_at": ago(minutes=minutes_ago), "last_duration_ms": ms, "last_result": result,
            "next_run_at": iso(NOW + next_in), "failures": 0, "runs": 140, "skipped": 0, "budget_calls": 30,
            "calls_used": 4, "calls_refused": 0}


def decider_status() -> dict:
    return {
        "enabled": True, "healthy": True, "deployment_id": 1, "state": "ready", "model": "vllm-sr/d3-mini",
        "revision": "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e", "p50_ms": 41, "decisions_24h": 23, "accepted_rate": 0.94,
        "features": {f[0]: {"enabled": True, "threshold": f[3]} for f in FEATURES},
        "image": {"present": True, "building": False, "error": None}, "upgrade": {"needed": False, "reasons": []},
        "configured_model": "vllm-sr/d3-mini", "restart_required": False,
        "models": [
            {"repo": "vllm-sr/d3-mini", "revision": "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e", "served_name": "d3-mini",
             "params": "4.5B", "reservation_bytes": g(13.0), "default": True, "downloaded": True},
            {"repo": "vllm-sr/d3-nano", "revision": "aeb70d4dd53ed25b855c3b0a66394b8fe265db7f", "served_name": "d3-nano",
             "params": "2.2B", "reservation_bytes": g(9.0), "default": False, "downloaded": False}],
        "breaker": {"open": False, "failures": 0, "retry_in_s": 0},
        "load": {"max_in_flight": 2, "in_flight": 0, "queued": 0, "shed": 0, "service_ms": 41},
        "supervisor": {"last_action": None, "last_reason": None, "probe": {"ok": True, "latency_ms": 1, "at": NOW.timestamp() - 20},
                       "failed_probes": 0, "restarts_last_hour": 0, "max_restarts_per_hour": 3, "backoff_s": 0,
                       "gave_up": False, "pending_restart": None, "next_attempt_in_s": None},
        "feature_specs": [{"name": n, "label": l, "description": d, "default_threshold": t, "background": bg,
                           "needs_model": n != "tuning", "viewer_visible": vv, "members": m} for n, l, d, t, bg, vv, m in FEATURES],
        "jobs": [
            _job("health_check", "health", "every 300 s", 0, 48, "ok: 5 deployments checked, 1 with findings", timedelta(minutes=5)),
            _job("threshold_tuning", "tuning", "weekly sun 04:30", 1, 19, "ok: 15 features checked, 1 proposal", timedelta(days=3, hours=13)),
            _job("reading_labels", "reading", "every 900 s", 1, 32, "ok: 4 of 4 readings labelled (0 asked the model)", timedelta(minutes=13)),
            _job("routing_advice", "routing", "daily 05:15", 1, 26, "ok: 5 of 6 classes have a suggestion", timedelta(hours=13, minutes=55)),
            _job("traffic_profiles", "schedule", "daily 03:10", 1, 280, "ok: 5 profiles", timedelta(hours=11, minutes=50)),
            _job("schedule_proposals", "schedule", "daily 03:40", 0, 55, "ok: 1 proposed of 3 candidates", timedelta(hours=12, minutes=20)),
            _job("schedule_runner", "schedule", "every 60 s", 1, 6, "ok: nothing due", timedelta(seconds=40)),
        ],
    }
