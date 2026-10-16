"""Test doubles: Docker SDK client, host probe, engine HTTP endpoints."""
from __future__ import annotations

import itertools

import docker.errors
import httpx

from dgxmm.config import GIB
from dgxmm.memory.hostmem import HostMemory, HostSample

_ids = itertools.count(1)


class FakeContainer:
    def __init__(self, client, cid, name, image, command, labels, env, ports, networks, mounts, status="running"):
        self.client, self.id, self.name, self.image_name = client, cid, name, image
        self.command, self.labels, self.env, self.ports = command or [], labels or {}, env or {}, ports
        self.networks, self.mounts, self.status, self.exit_code = networks, mounts, status, None
        self.log_text, self.stopped = "", False

    @property
    def attrs(self):
        return {
            "State": {"Status": self.status, "ExitCode": self.exit_code if self.exit_code is not None else 0},
            "Config": {"Image": self.image_name, "Cmd": self.command, "Labels": self.labels,
                       "Env": [f"{k}={v}" for k, v in self.env.items()]},
            "Mounts": [{"Source": h, "Destination": t, "Mode": m} for h, t, m in self.mounts],
            "NetworkSettings": {"Networks": {n: {} for n in self.networks},
                                "Ports": {f"{p}/tcp": [{"HostIp": b, "HostPort": str(p)}] for p, b in self.ports}},
        }

    def reload(self):
        pass

    def stop(self, timeout=10):
        self.stopped = True
        self.status = "exited"
        if self.exit_code is None:
            self.exit_code = 0

    def remove(self, force=False):
        self.client._containers.pop(self.id, None)

    def logs(self, tail=200, **_):
        return self.log_text.encode()


class FakeHelperContainer:
    """A short-lived auto-removed helper (dgxmm.helper=true); never appears in containers.list()."""

    def __init__(self, client):
        self.client, self.id = client, f"{next(_ids):064x}"

    def wait(self, timeout=None, condition=None):
        self.client.helper_waits.append(timeout)
        if self.client.helper_wait_error:
            raise self.client.helper_wait_error
        return {"StatusCode": self.client.helper_exit_code}

    def remove(self, force=False):
        self.client.helper_killed += 1


class FakeImages:
    def __init__(self, client):
        self.client = client

    def get(self, name):
        self.client.images_checked.append(name)
        if name in self.client.missing_images:
            raise docker.errors.ImageNotFound(f"no such image: {name}")
        return object()


class FakeContainers:
    def __init__(self, client):
        self.client = client

    def list(self, all=False, filters=None):
        items = list(self.client._containers.values())
        if filters and "label" in filters:
            k, _, v = filters["label"].partition("=")
            items = [c for c in items if c.labels.get(k) == v]
        return items

    def get(self, key):
        # Support skip_gets to allow initial gets to succeed before failing
        if self.client.skip_gets > 0:
            self.client.skip_gets -= 1
            for c in self.client._containers.values():
                if key in (c.id, c.name):
                    return c
            raise docker.errors.NotFound("no such container")
        # Support fail_get_count for transient errors
        if self.client.fail_get_count > 0:
            self.client.fail_get_count -= 1
            raise docker.errors.APIError("transient error")
        if self.client.always_fail_get:
            raise docker.errors.APIError("persistent error")
        for c in self.client._containers.values():
            if key in (c.id, c.name):
                return c
        raise docker.errors.NotFound("no such container")

    def run(self, image, command=None, name=None, environment=None, labels=None, network=None, volumes=None,
            ports=None, entrypoint=None, remove=False, **kw):
        if (labels or {}).get("dgxmm.helper") == "true":
            call = dict(image=image, command=command, name=name, environment=environment, labels=labels,
                        network=network, volumes=volumes, ports=ports, entrypoint=entrypoint, remove=remove, **kw)
            self.client.helper_runs.append(call)
            if self.client.fail_helper:
                raise docker.errors.ImageNotFound("no such image")
            if self.client.helper_action:
                self.client.helper_action(call)
            return FakeHelperContainer(self.client)
        self.client.run_calls.append(dict(image=image, command=command, name=name, environment=environment,
                                          labels=labels, network=network, volumes=volumes, ports=ports,
                                          entrypoint=entrypoint, remove=remove, **kw))
        if remove:
            return self.client.oneshot_output.encode()
        if self.client.fail_run:
            raise docker.errors.APIError("boom")
        if any(c.name == name for c in self.client._containers.values()):
            raise docker.errors.APIError("Conflict. The container name is already in use")
        cid = f"{next(_ids):064x}"
        mounts = [(h, v["bind"], v.get("mode", "rw")) for h, v in (volumes or {}).items()]
        plist = [(int(k.split("/")[0]), v[0]) for k, v in (ports or {}).items()]
        c = FakeContainer(self.client, cid, name, image, command, labels, environment, plist,
                          [network] if network else [], mounts)
        if name in self.client.exit_on_start:
            c.status, c.exit_code = "exited", self.client.exit_on_start[name]
        self.client._containers[cid] = c
        return c


class FakeDockerClient:
    def __init__(self):
        self._containers: dict[str, FakeContainer] = {}
        self.containers = FakeContainers(self)
        self.images = FakeImages(self)
        self.images_checked: list[str] = []
        self.missing_images: set[str] = set()
        self.run_calls: list[dict] = []
        self.fail_run = False
        self.oneshot_output = ""
        self.fail_get_count = 0
        self.always_fail_get = False
        self.skip_gets = 0
        self.exit_on_start: dict[str, int] = {}
        self.helper_runs: list[dict] = []      # delete-helper containers (dgxmm.helper=true)
        self.helper_action = None              # callable(call) simulating the helper's effect
        self.fail_helper = False
        self.helper_exit_code = 0
        self.helper_wait_error: Exception | None = None
        self.helper_waits: list = []
        self.helper_killed = 0

    def add_unmanaged(self, name, image="vllm/vllm-openai:x", command=None, labels=None, ports=(), status="running",
                      mounts=(), networks=("gateway",), env=None):
        cid = f"{next(_ids):064x}"
        c = FakeContainer(self, cid, name, image, command, labels or {}, env or {}, list(ports), list(networks),
                          list(mounts), status=status)
        self._containers[cid] = c
        return c

    def container(self, cid) -> FakeContainer:
        return self._containers[cid]


class FakeProbe:
    def __init__(self, total_gib: float = 121, available_gib: float = 100):
        self.total, self.available = int(total_gib * GIB), int(available_gib * GIB)
        self.managed_actual: dict[str, int] = {}

    def meminfo(self) -> HostMemory:
        return HostMemory(total=self.total, available=self.available, free=0, cached=0, swap_total=0, swap_free=0)

    def sample(self, ids):
        return HostSample(mem=self.meminfo(), managed={c: self.managed_actual.get(c, 0) for c in ids},
                          external_top=[], gpu={})


class FakeEngines:
    """Answers engine HTTP calls by port: /health, /v1/completions, /metrics, /sleep, /wake_up."""

    def __init__(self):
        self.healthy: set[int] = set()
        self.metrics: dict[int, str] = {}
        self.sleeping: set[int] = set()
        self.fail_paths: set[str] = set()
        self.calls: list[tuple[str, int, str]] = []
        self.all_healthy = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        port, path = request.url.port, request.url.path
        self.calls.append((request.method, port, path))
        if path in self.fail_paths:
            return httpx.Response(500)
        if path == "/health":
            return httpx.Response(200 if (self.all_healthy or port in self.healthy) else 503)
        if path == "/v1/completions":
            return httpx.Response(200 if (self.all_healthy or port in self.healthy) else 503, json={"choices": [{"text": "x"}]})
        if path == "/metrics":
            return httpx.Response(200, text=self.metrics.get(port, ""))
        if path == "/sleep":
            self.sleeping.add(port)
            return httpx.Response(200)
        if path == "/wake_up":
            self.sleeping.discard(port)
            return httpx.Response(200)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def free_on_stop(deployer, probe) -> None:
    """Make FakeProbe behave like GB10 unified memory: stopping a deployment returns its memory
    (reservation, or measured memory if it was sleeping) to MemAvailable."""
    orig = deployer.stop

    async def stop(dep_id, *a, **kw):
        dep = deployer._get(dep_id)
        before = dep.state if dep else None
        actual = deployer._actual(dep_id) or 0
        out = await orig(dep_id, *a, **kw)
        if before in ("ready", "unknown", "starting", "loading"):
            probe.available += max(dep.reserved_bytes, actual)
        elif before == "sleeping":
            probe.available += actual
        return out
    deployer.stop = stop


class FakeLiteLLM:
    """In-memory LiteLLM proxy (v1.100 shapes): health, /model/*, /key/* admin API, master-key auth."""

    def __init__(self, master_key: str = "sk-test", version: str = "1.100.0"):
        self.master_key, self.version = master_key, version
        self.models = [{"model_name": "hand-written", "litellm_params": {"model": "openai/x", "api_base": "http://ext:8000/v1"},
                        "model_info": {"id": "keep-me", "db_model": True}}]
        self.keys: dict[str, dict] = {}
        self.down = False
        self.no_db = False
        self.readiness_version = False  # older LiteLLM builds put litellm_version in /health/readiness
        self.readiness_db: str | None = None  # override the reported db state
        self.impostor = False  # behave like some other HTTP server
        self.openapi_status, self.openapi_pad = 200, 0
        self.raw: dict[str, httpx.Response] = {}  # path -> canned response (after auth)
        self.fail_update_tokens: set[str] = set()  # /key/update fails (500) for these tokens
        self.auth: list[str | None] = []
        self.calls: list[tuple[str, str]] = []
        self.hosts: list[str] = []
        self.bodies: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def add_key(self, alias: str, models: list[str] | None = None, **extra) -> str:
        import hashlib
        import secrets
        key = "sk-" + secrets.token_urlsafe(16)
        token = hashlib.sha256(key.encode()).hexdigest()
        self.keys[token] = {"token": token, "key_name": f"sk-...{key[-4:]}", "key_alias": alias,
                            "models": models if models is not None else [], "spend": 0.0, "max_budget": None,
                            "budget_duration": None, "rpm_limit": None, "tpm_limit": None, "expires": None,
                            "created_at": "2026-10-16T12:00:00Z", "blocked": None, "metadata": {}, **extra}
        return token

    @staticmethod
    def _err(status: int, message: str, typ: str = "error") -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message, "type": typ, "param": "None", "code": str(status)}})

    def handler(self, req: httpx.Request) -> httpx.Response:
        import hashlib
        import json
        import secrets
        path = req.url.path
        self.calls.append((req.method, path))
        self.hosts.append(req.url.host)
        self.auth.append(req.headers.get("authorization"))
        if self.down:
            raise httpx.ConnectError("connection refused")
        if self.impostor:
            return httpx.Response(200, text="<html>hello</html>") if path == "/health/liveliness" else httpx.Response(404)
        if path == "/health/liveliness":
            return httpx.Response(200, json="I'm alive!")
        if path == "/health/readiness":
            body = {"status": "healthy", "db": self.readiness_db or ("Not connected" if self.no_db else "connected")}
            if self.readiness_version:
                body["litellm_version"] = self.version
            return httpx.Response(200, json=body)
        if path == "/openapi.json":
            return httpx.Response(self.openapi_status, json={"info": {"title": "LiteLLM API", "version": self.version},
                                                             "paths": {}, "pad": "x" * self.openapi_pad})
        if req.headers.get("authorization") != f"Bearer {self.master_key}":
            return self._err(401, "Authentication Error, Invalid proxy server token passed. Received API Key = sk-...wxyz, "
                                  "Key Hash (Token) =" + "ab" * 32, "token_not_found_in_db")
        if path in self.raw:
            return self.raw[path]
        body = json.loads(req.content) if req.content else {}
        if body:
            self.bodies.append(body)
        if path == "/model/info":
            return httpx.Response(200, json={"data": self.models})
        if path == "/model/new":
            self.models.append({**body, "model_info": {**body.get("model_info", {}), "id": secrets.token_hex(8)}})
            return httpx.Response(200, json={"ok": True})
        if path == "/model/delete":
            self.models = [m for m in self.models if m["model_info"]["id"] != body["id"]]
            return httpx.Response(200, json={"ok": True})
        if path.startswith("/key/") and self.no_db:
            return self._err(500, "No connected db.", "internal_server_error")
        if path == "/key/generate":
            key = "sk-" + secrets.token_urlsafe(16)
            token = hashlib.sha256(key.encode()).hexdigest()
            rec = {"token": token, "key_name": f"sk-...{key[-4:]}", "key_alias": body.get("key_alias"),
                   "models": body.get("models") or [], "spend": 0.0, "max_budget": body.get("max_budget"),
                   "budget_duration": body.get("budget_duration"), "rpm_limit": body.get("rpm_limit"),
                   "tpm_limit": body.get("tpm_limit"), "expires": "2026-11-15T12:00:00Z" if body.get("duration") else None,
                   "created_at": "2026-10-16T12:00:00Z", "blocked": None, "metadata": body.get("metadata") or {}}
            self.keys[token] = rec
            return httpx.Response(200, json={**rec, "key": key, "token_id": token})
        if path == "/key/list":
            size, page = int(req.url.params.get("size", 10)), int(req.url.params.get("page", 1))
            items = list(self.keys.values())
            chunk = items[(page - 1) * size: page * size]
            full = req.url.params.get("return_full_object") == "true"
            return httpx.Response(200, json={"keys": chunk if full else [k["token"] for k in chunk],
                                             "total_count": len(items), "current_page": page,
                                             "total_pages": max(1, -(-len(items) // size))})
        if path == "/key/info":
            rec = self.keys.get(req.url.params.get("key", ""))
            if rec is None:
                return self._err(404, "Key not found in database", "not_found_error")
            return httpx.Response(200, json={"key": rec["token"], "info": {k: v for k, v in rec.items() if k != "token"}})
        if path == "/key/update":
            rec = self.keys.get(body.get("key", ""))
            if rec is None:
                return self._err(404, "Key not found.", "not_found_error")
            if rec["token"] in self.fail_update_tokens:
                return self._err(500, "boom", "internal_server_error")
            for k, v in body.items():
                if k != "key" and v is not None:
                    rec[k] = v
            return httpx.Response(200, json={**rec, "key": rec["token"]})
        if path == "/key/delete":
            found = [t for t in body.get("keys", []) if t in self.keys]
            if not found:
                return self._err(404, "{'error': 'No keys found'}", "internal_server_error")
            for t in found:
                del self.keys[t]
            return httpx.Response(200, json={"deleted_keys": found})
        return httpx.Response(404, json={"detail": "Not Found"})
