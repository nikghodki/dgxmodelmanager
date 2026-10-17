"""The only module that talks to the Docker daemon. Writes are limited to dgxmm-managed containers,
plus short-lived delete helpers (``dgxmm.helper=true``) that this process creates itself."""
from __future__ import annotations

import logging
import posixpath
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import docker
import docker.errors
from docker.types import DeviceRequest
from docker.types import Mount as DockerMount

from dgxmm.engines.base import MANAGED_LABEL, ContainerSpec

log = logging.getLogger(__name__)
INSTALL_LABEL = "dgxmm.install"  # per-install id (Settings.install_id), set by the deployer
HELPER_LABEL = "dgxmm.helper"
HELPER_MOUNT = "/x"  # where a helper sees the single host directory it may touch
CONTAINER_CACHE_TTL_S = 30.0
LOOKUP_TIMEOUT_S = 3


class SecretDeliveryFailed(RuntimeError):
    """A secret file could not be written into a freshly started container (which is then removed). If the container
    had already exited, ``exit_code`` and a redacted ``log_tail`` say why (e.g. the decider's integrity check)."""

    def __init__(self, message: str, exit_code: int | None = None, log_tail: str | None = None):
        super().__init__(message)
        self.exit_code, self.log_tail = exit_code, log_tail


# Exit codes of deploy/decider/entrypoint.sh / launch.py
EARLY_EXIT_REASONS = {2: "the pinned model snapshot is missing", 3: "snapshot integrity check failed",
                      4: "no API key was delivered"}


class HelperUnavailable(RuntimeError):
    """The delete helper cannot run (e.g. its image is missing); callers fall back to in-process deletion."""


@dataclass
class ContainerInfo:
    id: str
    name: str
    status: str
    exit_code: int | None
    labels: dict[str, str]
    image: str
    command: list[str]
    mounts: list[dict] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    host_ports: set[int] = field(default_factory=set)
    oom_killed: bool = False  # Docker's State.OOMKilled of an exited container (the kernel OOM-killed it)

    @property
    def managed(self) -> bool:
        return self.labels.get(MANAGED_LABEL) == "true"

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status, "exit_code": self.exit_code,
                "image": self.image, "command": self.command, "managed": self.managed,
                "host_ports": sorted(self.host_ports), "networks": self.networks}


def _info(c) -> ContainerInfo:
    a = c.attrs
    state, cfg = a.get("State", {}), a.get("Config", {})
    ports: set[int] = set()
    for bindings in (a.get("NetworkSettings", {}).get("Ports") or {}).values():
        for b in bindings or []:
            if b.get("HostPort"):
                ports.add(int(b["HostPort"]))
    status = state.get("Status", getattr(c, "status", "unknown"))
    return ContainerInfo(
        id=c.id, name=c.name.lstrip("/"), status=status,
        exit_code=state.get("ExitCode") if status in ("exited", "dead") else None,
        labels=dict(cfg.get("Labels") or {}), image=cfg.get("Image", ""), command=list(cfg.get("Cmd") or []),
        mounts=[{"source": m.get("Source"), "target": m.get("Destination"), "mode": m.get("Mode", "")}
                for m in a.get("Mounts") or []],
        networks=list((a.get("NetworkSettings", {}).get("Networks") or {}).keys()), host_ports=ports,
        oom_killed=status in ("exited", "dead") and state.get("OOMKilled") is True,
    )


class ContainerDirectory:
    """Read-only container id → {id, name, image, deployment_id}, from one sparse `docker ps` at most every `ttl`
    seconds (a miss may refresh early, but not more often than every `miss_refresh_s`). Never modifies containers.

    The Docker call runs outside the lock and at most one refresh runs at a time, so cache readers (the budget
    loop uses refresh=False) never wait on the daemon."""

    def __init__(self, client, ttl: float = CONTAINER_CACHE_TTL_S, miss_refresh_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self.client, self.ttl, self.miss_refresh_s, self.clock = client, ttl, miss_refresh_s, clock
        self._by_id: dict[str, dict] = {}
        self._at: float | None = None
        self._refreshing = False
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            if self._refreshing:
                return  # another thread is already asking Docker
            self._refreshing = True
            self._at = self.clock()  # also on failure: a down daemon is retried at the normal pace
        try:
            by_id = self._list()
        finally:
            with self._lock:
                self._refreshing = False
        if by_id is not None:
            with self._lock:
                self._by_id = by_id

    def _list(self) -> dict[str, dict] | None:
        try:
            items = self.client.containers.list(all=False, sparse=True)
        except Exception as e:  # docker.errors.APIError, connection errors, timeouts
            log.debug("container listing failed: %s", e)
            return None
        by_id: dict[str, dict] = {}
        for c in items:
            a = getattr(c, "attrs", None) or {}
            cfg = a.get("Config") or {}
            name = getattr(c, "name", None) or next(iter(a.get("Names") or []), None)
            labels = cfg.get("Labels") or a.get("Labels") or {}
            dep = labels.get("dgxmm.deployment_id") if labels.get(MANAGED_LABEL) == "true" else None
            by_id[c.id] = {"id": c.id[:12], "name": name.lstrip("/") if name else None,
                           "image": cfg.get("Image") or a.get("Image") or None,
                           "deployment_id": int(dep) if dep and str(dep).isdigit() else None}
        return by_id

    def lookup(self, cid: str | None, refresh: bool = True) -> dict | None:
        """refresh=False: cache only, never calls Docker (may be stale or empty)."""
        if not cid:
            return None
        with self._lock:
            now = self.clock()
            age = None if self._at is None else now - self._at
            stale = age is None or age >= self.ttl or (cid not in self._by_id and age >= self.miss_refresh_s)
        if refresh and stale:
            self.refresh()
        with self._lock:
            found = self._by_id.get(cid)
        return dict(found) if found else None


class DockerController:
    def __init__(self, client, lookup_client=None):
        self.client = client
        self.directory = ContainerDirectory(lookup_client or client)
        self._own_id: str | None = None

    def own_container_id(self) -> str | None:
        """Id of the container this process runs in (None when DGXMM runs on the host)."""
        if self._own_id is None:
            self._own_id = self._own_container_id("/proc/self/mountinfo", "/proc/self/cgroup")
        return self._own_id

    def lookup(self, cid: str | None, refresh: bool = True) -> dict | None:
        """Cached, read-only name/image lookup by full container id (for the processes view)."""
        return self.directory.lookup(cid, refresh=refresh)

    @classmethod
    def from_env(cls) -> "DockerController":
        # Lookups use their own client with a short timeout so a slow daemon cannot stall the processes view.
        return cls(docker.from_env(), lookup_client=docker.from_env(timeout=LOOKUP_TIMEOUT_S))

    def list_all(self) -> list[ContainerInfo]:
        return [_info(c) for c in self.client.containers.list(all=True)]

    def list_managed(self) -> list[ContainerInfo]:
        return [_info(c) for c in self.client.containers.list(all=True, filters={"label": f"{MANAGED_LABEL}=true"})]

    def get(self, id_or_name: str | None) -> ContainerInfo | None:
        if not id_or_name:
            return None
        try:
            c = self.client.containers.get(id_or_name)
            c.reload()
            return _info(c)
        except docker.errors.NotFound:
            return None

    def _guard(self, id_or_name: str):
        c = self.client.containers.get(id_or_name)
        if (c.labels or {}).get(MANAGED_LABEL) != "true":
            raise PermissionError(f"container {c.name} is not managed by dgxmm")
        return c

    def run(self, spec: ContainerSpec) -> str:
        if spec.labels.get(MANAGED_LABEL) != "true":
            raise PermissionError("refusing to create a container without the dgxmm.managed label")
        if spec.internal_network:
            install = spec.labels.get(INSTALL_LABEL)
            self.ensure_internal_network(spec.network, install_id=install)
        kwargs: dict = dict(image=spec.image, command=spec.command, name=spec.name, detach=True,
                            environment=spec.environment, labels=spec.labels, network=spec.network,
                            volumes={m.host: {"bind": m.target, "mode": m.mode} for m in spec.mounts},
                            restart_policy={"Name": "no"})
        if spec.entrypoint is not None:
            kwargs["entrypoint"] = spec.entrypoint
        if spec.publish_bind:
            kwargs["ports"] = {f"{spec.port}/tcp": (spec.publish_bind, spec.port)}
        if spec.gpus:
            kwargs["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
        if spec.ipc_host:
            kwargs["ipc_mode"] = "host"
        if spec.user:
            kwargs["user"] = spec.user
        if spec.read_only:
            kwargs["read_only"] = True
        if spec.tmpfs:
            kwargs["tmpfs"] = dict(spec.tmpfs)
        if spec.cap_drop:
            kwargs["cap_drop"] = list(spec.cap_drop)
        if spec.security_opt:
            kwargs["security_opt"] = list(spec.security_opt)
        if spec.pids_limit:
            kwargs["pids_limit"] = spec.pids_limit
        if spec.mem_limit:
            kwargs["mem_limit"] = spec.mem_limit
            kwargs["memswap_limit"] = spec.mem_limit  # no swap on top of the limit
        try:
            cid = self.client.containers.run(**kwargs).id
        except Exception:
            # docker-py creates, then starts: a failed start (device request, port in use, bad mount ...) leaves the
            # container in state "created". Never leave it behind.
            self.remove_leftover(spec.name, spec.labels.get("dgxmm.deployment_id"))
            raise
        if spec.secret_files:
            try:
                for path, data in spec.secret_files.items():
                    self.write_secret_file(cid, path, data)
            except Exception as e:
                exit_code, tail = self._exit_details(cid)
                try:
                    self.client.containers.get(cid).remove(force=True)
                except Exception:
                    log.warning("could not remove %s after a failed secret delivery", spec.name, exc_info=True)
                if exit_code is not None:
                    why = EARLY_EXIT_REASONS.get(exit_code)
                    raise SecretDeliveryFailed(f"{spec.name} exited with code {exit_code} before its key could be "
                                               f"delivered" + (f": {why}" if why else ""), exit_code, tail) from None
                raise SecretDeliveryFailed(f"could not deliver a secret file to {spec.name}: {e}") from None
        return cid

    def remove_leftover(self, name: str, deployment_id: int | str | None = None) -> bool:
        """Remove the dgxmm-managed container called ``name`` if one exists (any state); never an unmanaged one,
        and with ``deployment_id`` only if its dgxmm.deployment_id label matches. True if one was removed. Errors are
        logged, not raised (cleanup paths call it)."""
        try:
            if deployment_id is not None:
                c = self._guard(name)
                if (c.labels or {}).get("dgxmm.deployment_id") != str(deployment_id):
                    log.warning("not removing %s: it belongs to another deployment", name)
                    return False
            self.remove_by_name(name)
            return True
        except PermissionError:
            log.warning("not removing %s: it is not managed by dgxmm", name)
        except docker.errors.NotFound:
            pass
        except Exception:
            log.warning("could not remove leftover container %s", name, exc_info=True)
        return False

    def _exit_details(self, cid: str, tail: int = 120) -> tuple[int | None, str | None]:
        """(exit code, redacted log tail) if the container has already stopped, else (None, None)."""
        from dgxmm.memory.redact import redact_log
        try:
            c = self.client.containers.get(cid)
            c.reload()
            state = (c.attrs or {}).get("State") or {}
            if state.get("Status") not in ("exited", "dead"):
                return None, None
            logs = c.logs(tail=tail)
            text = logs.decode(errors="replace") if isinstance(logs, bytes) else str(logs or "")
            return int(state.get("ExitCode", -1)), redact_log(text)
        except Exception:
            log.warning("could not inspect %s after a failed secret delivery", cid[:12], exc_info=True)
            return None, None

    def write_secret_file(self, cid: str, path: str, data: bytes, timeout: float = 20.0) -> None:
        """Write `data` to `path` (mode 0400, atomically) inside the running container `cid`, through `docker exec`
        stdin: unlike `put_archive` this reaches a tmpfs and works with a read-only root filesystem, and the content
        never appears in the environment, a command line or `docker inspect`. Errors never contain the content."""
        if not path.startswith("/") or "\n" in path:
            raise ValueError("secret path must be absolute")
        api = self.client.api
        cmd = ["/bin/sh", "-c", 'umask 0377 && cat > "$1.part" && mv -f "$1.part" "$1"', "dgxmm-secret", path]
        exec_id = api.exec_create(cid, cmd, stdin=True, stdout=True, stderr=True)["Id"]
        sock = api.exec_start(exec_id, socket=True)
        raw = getattr(sock, "_sock", sock)
        try:
            raw.settimeout(timeout)
            raw.sendall(data)
            raw.shutdown(socket.SHUT_WR)
            while raw.recv(4096):  # drain until the exec closes its side
                pass
        finally:
            sock.close()
        deadline = time.monotonic() + timeout
        while True:
            info = api.exec_inspect(exec_id)
            if not info.get("Running"):
                break
            if time.monotonic() > deadline:
                raise SecretDeliveryFailed("timed out writing the secret file")
            time.sleep(0.05)
        if info.get("ExitCode") != 0:
            raise SecretDeliveryFailed(f"writing the secret file exited with {info.get('ExitCode')}")

    def ensure_internal_network(self, name: str, install_id: str | None = None) -> None:
        """`name` exists as an *internal* network (created if missing) and this DGXMM container is attached to it.
        Refuses a non-internal network of that name, and refuses when DGXMM is not in a container (it could never
        reach a container on an internal network). A network it creates carries dgxmm.install=<install_id> (when
        set), so `./install.sh uninstall --purge` removes only this install's."""
        try:
            net = self.client.networks.get(name)
        except docker.errors.NotFound:
            labels = {MANAGED_LABEL: "true", "dgxmm.purpose": "decider"}
            if install_id:
                labels[INSTALL_LABEL] = install_id
            net = self.client.networks.create(name, driver="bridge", internal=True, check_duplicate=True,
                                              labels=labels)
        if not (net.attrs or {}).get("Internal"):
            raise PermissionError(f"docker network {name} is not internal; refusing to attach the decider to it "
                                  "(remove it, or set decider_network to another name)")
        if not self._join(net):
            raise RuntimeError(f"DGXMM is not running in a container, so it cannot reach the decider on the internal "
                               f"network {name}; set decider_network: '' to use docker_network instead")

    def network_subnets(self, name: str) -> list[str]:
        """The IPAM subnets of docker network `name` ([] if it does not exist)."""
        if not name:
            return []
        try:
            net = self.client.networks.get(name)
        except docker.errors.NotFound:
            return []
        cfg = ((net.attrs or {}).get("IPAM") or {}).get("Config") or []
        return [c["Subnet"] for c in cfg if isinstance(c, dict) and c.get("Subnet")]

    def attach_self(self, name: str) -> bool:
        """Startup: join the decider network if it exists (never creates it). True if attached."""
        if not name:
            return False
        try:
            net = self.client.networks.get(name)
        except docker.errors.NotFound:
            return False
        if not (net.attrs or {}).get("Internal"):
            log.warning("docker network %s is not internal; not joining it", name)
            return False
        return self._join(net)

    def _join(self, net) -> bool:
        me = self.own_container_id()
        if not me:
            return False
        net.reload()
        members = (net.attrs or {}).get("Containers") or {}
        if any(cid.startswith(me) or me.startswith(cid) for cid in members):
            return True
        try:
            net.connect(me)
        except docker.errors.APIError as e:
            if "already exists" not in str(e):
                raise
        return True

    def stop_and_remove(self, cid: str | None, timeout: int = 30) -> None:
        if not cid:
            return
        try:
            c = self._guard(cid)
        except docker.errors.NotFound:
            return
        try:
            c.stop(timeout=timeout)
        except docker.errors.APIError:
            pass
        try:
            c.remove(force=True)
        except docker.errors.NotFound:
            pass

    def remove_by_name(self, name: str) -> None:
        try:
            c = self._guard(name)
        except docker.errors.NotFound:
            return
        try:
            c.remove(force=True)
        except docker.errors.NotFound:
            pass

    def logs(self, cid: str | None, tail: int = 200) -> str:
        if not cid:
            return ""
        try:
            return self.client.containers.get(cid).logs(tail=tail).decode(errors="replace")
        except docker.errors.NotFound:
            return ""

    def image_present(self, tag: str) -> bool:
        try:
            self.client.images.get(tag)
            return True
        except docker.errors.ImageNotFound:
            return False

    def image_labels(self, tag: str) -> dict | None:
        """Labels of a local image (None if it does not exist)."""
        try:
            img = self.client.images.get(tag)
        except docker.errors.ImageNotFound:
            return None
        return dict(getattr(img, "labels", None) or {})

    def build_image(self, fileobj, tag: str, labels: dict[str, str] | None = None):
        """Build ``tag`` from an in-memory tar build context; yields the daemon's decoded progress chunks. Local
        only: nothing is pushed. Always labelled ``dgxmm.built=true`` (plus ``labels``)."""
        return self.client.api.build(fileobj=fileobj, custom_context=True, tag=tag, rm=True, forcerm=True,
                                     decode=True, labels={**(labels or {}), "dgxmm.built": "true"})

    def run_oneshot(self, image: str, entrypoint: list[str], command: list[str]) -> str:
        out = self.client.containers.run(image, command=command, entrypoint=entrypoint, remove=True,
                                         stdout=True, stderr=False,
                                         labels={MANAGED_LABEL: "true", "dgxmm.oneshot": "true"})
        return out.decode(errors="replace") if isinstance(out, bytes) else str(out)

    @staticmethod
    def _own_container_id(mountinfo: str, cgroup: str) -> str | None:
        pats = [(mountinfo, re.compile(r"/containers/([0-9a-f]{64})/")),
                (cgroup, re.compile(r"(?:docker-|/docker/)([0-9a-f]{64})"))]
        for path, rx in pats:
            try:
                with open(path) as f:
                    m = rx.search(f.read())
            except OSError:
                continue
            if m:
                return m.group(1)
        return None

    def self_image(self, mountinfo: str = "/proc/self/mountinfo", cgroup: str = "/proc/self/cgroup") -> str | None:
        """Image of the container this process runs in: container id from /proc (mountinfo, then cgroup),
        else the hostname (docker's default hostname is the short id)."""
        for key in (self._own_container_id(mountinfo, cgroup), socket.gethostname()):
            if not key:
                continue
            try:
                c = self.client.containers.get(key)
                image = (c.attrs.get("Config") or {}).get("Image")
                if image:
                    return image
            except Exception:
                continue
        return None

    def run_helper(self, image: str, host_dir: str, command: list[str], timeout: float = 120) -> int | None:
        """Run ``command`` as root in a throwaway, network-less, capability-trimmed container that bind-mounts
        ONLY ``host_dir`` (at HELPER_MOUNT). Never pulls images. Returns the exit code, or None if unknown
        (timed out / already removed). Raises HelperUnavailable when the image is missing."""
        norm = posixpath.normpath(host_dir) if host_dir else ""
        if not host_dir or not posixpath.isabs(host_dir) or norm == "/" or ".." in host_dir.split("/"):
            raise ValueError(f"refusing to mount {host_dir!r} into a helper container")
        if not command:
            raise ValueError("helper needs a command")
        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            raise HelperUnavailable(f"helper image {image} is not present locally (set helper_image)")
        c = self.client.containers.run(
            image, entrypoint=[command[0]], command=list(command[1:]), detach=True, auto_remove=True,
            user="0:0", network_mode="none", labels={HELPER_LABEL: "true"},
            # Mount (not volumes=): a missing source is an error instead of an auto-created root-owned dir
            mounts=[DockerMount(target=HELPER_MOUNT, source=norm, type="bind", read_only=False)],
            cap_drop=["ALL"], cap_add=["DAC_OVERRIDE", "FOWNER"], read_only=True, pids_limit=64,
            security_opt=["no-new-privileges"],
        )
        try:
            res = c.wait(timeout=timeout, condition="removed")
        except docker.errors.NotFound:
            return None  # exited and auto-removed before we started waiting
        except Exception as e:  # timeout or daemon trouble: make sure it does not linger
            log.warning("delete helper did not finish: %s", e)
            try:
                c.remove(force=True)
            except Exception:
                pass
            return None
        return int((res or {}).get("StatusCode", -1))
