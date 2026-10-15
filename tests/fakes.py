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
        for c in self.client._containers.values():
            if key in (c.id, c.name):
                return c
        raise docker.errors.NotFound("no such container")

    def run(self, image, command=None, name=None, environment=None, labels=None, network=None, volumes=None,
            ports=None, entrypoint=None, remove=False, **kw):
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
        self.client._containers[cid] = c
        return c


class FakeDockerClient:
    def __init__(self):
        self._containers: dict[str, FakeContainer] = {}
        self.containers = FakeContainers(self)
        self.run_calls: list[dict] = []
        self.fail_run = False
        self.oneshot_output = ""

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

    def handler(self, request: httpx.Request) -> httpx.Response:
        port, path = request.url.port, request.url.path
        self.calls.append((request.method, port, path))
        if path in self.fail_paths:
            return httpx.Response(500)
        if path == "/health":
            return httpx.Response(200 if port in self.healthy else 503)
        if path == "/v1/completions":
            return httpx.Response(200 if port in self.healthy else 503, json={"choices": [{"text": "x"}]})
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
