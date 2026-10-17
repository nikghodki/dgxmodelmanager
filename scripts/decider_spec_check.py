"""Real app-path check of the hardened decider container (manual; run INSIDE a throwaway container that has the
docker socket, so DGXMM's own "join the internal network" step is exercised exactly as in production):

    python scripts/decider_spec_check.py IMAGE NETWORK KEYFILE HF_CACHE_HOST [MEM_LIMIT_GIB]

Builds the spec with DGXMM's DecisionEngine and starts it with DockerController.run (key delivered as a tmpfs file
over exec stdin), checks `docker inspect` (no key, no published port, hardening flags), waits for the engine's own
health check (open /health + authenticated probe), then rotates the key the way the app does (new key file, new
container) and checks the old key is refused and the new one works. Cleans up (container, network membership,
network, key file). Prints no secret.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

from dgxmm.config import Settings
from dgxmm.decider.secret import ensure_key, read_key, rotate_key
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.decision import DecisionEngine


def inspect_summary(ctl: DockerController, cid: str, key: str) -> dict:
    a = ctl.client.api.inspect_container(cid)
    hc = a["HostConfig"]
    return {
        "key_in_inspect": key in json.dumps(a), "env_names": sorted(e.split("=", 1)[0] for e in a["Config"]["Env"]),
        "user": a["Config"]["User"], "readonly_rootfs": hc["ReadonlyRootfs"], "cap_drop": hc["CapDrop"],
        "security_opt": hc["SecurityOpt"], "pids_limit": hc["PidsLimit"], "memory": hc["Memory"],
        "memory_swap": hc["MemorySwap"], "tmpfs": sorted(hc.get("Tmpfs") or {}), "ipc_mode": hc["IpcMode"],
        "port_bindings": hc.get("PortBindings") or {}, "published": a["NetworkSettings"].get("Ports"),
        "binds": hc.get("Binds"), "networks": sorted(a["NetworkSettings"]["Networks"]),
        "privileged": hc["Privileged"],
    }


async def wait_healthy(eng, http, base, timeout=600) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if await eng.health(http, base, "d3-mini"):
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        await asyncio.sleep(5)
    raise SystemExit("decider did not become healthy")


async def status_with(http, base, key) -> int:
    r = await http.post(f"{base}/v1/systemone", headers={"authorization": f"Bearer {key}"},
                        json={"state": "x", "questions": {"p": {"type": "noul", "instructions": "ok?"}}}, timeout=60)
    return r.status_code


async def burst(http, base, key) -> str:
    from dgxmm.decider import diagnosis as dg
    log = "\n".join(f"(EngineCore pid=1) ERROR line {i}: RuntimeError: something failed" for i in range(120))
    st = dg.build_state(engine="vllm", model_ref="org/m", exit_code=1, message="exited", hits=[], log_text=log)
    body = {"state": st, "questions": {"q": {"type": "choice", "instructions": dg.QUESTION,
                                             "criteria": {o.key: o.description for o in dg.OPTIONS}}}}

    async def one():
        r = await http.post(f"{base}/v1/systemone", json=body, headers={"authorization": f"Bearer {key}"}, timeout=300)
        return r.status_code
    t0, codes = time.monotonic(), []
    for _ in range(4):
        codes += await asyncio.gather(*[one() for _ in range(16)])
    return f"{len(codes)} requests, statuses {sorted(set(codes))}, {time.monotonic() - t0:.1f} s"


async def main(image, network, keyfile, hf_host, mem_gib="12"):
    s = Settings.model_validate({"hf_cache": {"host": hf_host, "local": "/data/hf"}, "decider_network": network,
                                 "decider_key_file": keyfile, "images": {"decision": image},
                                 "decider_memory_limit_gib": float(mem_gib)})
    eng, ctl = DecisionEngine(s), DockerController.from_env()
    key = ensure_key(s)
    spec = eng.container_spec(None, eng.validate_config({}), None, LaunchContext(999, "x", 8000, 0, 0), s)
    spec.name = "prod-h3-spec-decider"
    spec.labels["dgxmm.probe"] = "h3"
    cids = []
    try:
        t0 = time.monotonic()
        cid = ctl.run(spec)
        cids.append(cid)
        print("started + key delivered in", round(time.monotonic() - t0, 2), "s")
        print("inspect:", json.dumps(inspect_summary(ctl, cid, key)))
        base = f"http://{spec.name}:8000"
        async with httpx.AsyncClient() as http:
            print("healthy after", round(await wait_healthy(eng, http, base), 1), "s (engine.health: /health + "
                  "authenticated probe)")
            print("no key ->", (await http.post(f"{base}/v1/systemone", json={"questions": {}}, timeout=30)).status_code)
            print("current key ->", await status_with(http, base, key))
            print("burst (16 concurrent x 4 diagnosis questions):", await burst(http, base, key))
            # rotation as the app does it: stop, new key, start a new container with the new key file
            ctl.stop_and_remove(cid)
            rotate_key(s)
            new = read_key(s)
            spec2 = eng.container_spec(None, eng.validate_config({}), None, LaunchContext(999, "x", 8000, 0, 0), s)
            spec2.name, spec2.labels["dgxmm.probe"] = spec.name, "h3"
            cid2 = ctl.run(spec2)
            cids.append(cid2)
            print("rotated; key changed:", new != key, "; new key in inspect:", inspect_summary(ctl, cid2, new)["key_in_inspect"])
            print("healthy after rotation in", round(await wait_healthy(eng, http, base), 1), "s")
            print("old key ->", await status_with(http, base, key), "; new key ->", await status_with(http, base, new))
    finally:
        for c in cids:
            ctl.stop_and_remove(c)
        try:
            net = ctl.client.networks.get(network)
            me = ctl.own_container_id()
            if me:
                net.disconnect(me, force=True)
            net.remove()
            print("network removed")
        except Exception as e:
            print("network cleanup:", type(e).__name__)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
