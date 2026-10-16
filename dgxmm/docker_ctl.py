"""The only module that talks to the Docker daemon. Writes are limited to dgxmm-managed containers,
plus short-lived delete helpers (``dgxmm.helper=true``) that this process creates itself."""
from __future__ import annotations

import logging
import posixpath
import re
import socket
from dataclasses import dataclass, field

import docker
import docker.errors
from docker.types import DeviceRequest
from docker.types import Mount as DockerMount

from dgxmm.engines.base import MANAGED_LABEL, ContainerSpec

log = logging.getLogger(__name__)
HELPER_LABEL = "dgxmm.helper"
HELPER_MOUNT = "/x"  # where a helper sees the single host directory it may touch


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
    )


class DockerController:
    def __init__(self, client):
        self.client = client

    @classmethod
    def from_env(cls) -> "DockerController":
        return cls(docker.from_env())

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
        return self.client.containers.run(**kwargs).id

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
        c.remove(force=True)

    def logs(self, cid: str | None, tail: int = 200) -> str:
        if not cid:
            return ""
        try:
            return self.client.containers.get(cid).logs(tail=tail).decode(errors="replace")
        except docker.errors.NotFound:
            return ""

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
