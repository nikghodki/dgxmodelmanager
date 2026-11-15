"""Node registry: Docker clients for local and remote nodes."""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable

import docker.errors
import requests
import urllib3.exceptions

from dgxmm.config import LOCAL_NODE_ID, NodeConfig, Settings
from dgxmm.docker_ctl import DockerController


SSH_CONFIG_ENV = "DGXMM_SSH_CONFIG"  # the image sets it (Dockerfile) for its ssh wrapper and for this module alike


def ssh_config_path(settings: Settings | None = None) -> Path:
    """The ssh client config NodeRegistry writes (per-node user, key, pinned known_hosts, StrictHostKeyChecking):
    $DGXMM_SSH_CONFIG, else <data_dir>/ssh/config. A DGXMM-owned path, never a ~/.ssh: OpenSSH reads ~/.ssh/config
    from the passwd home (not $HOME), so nothing may rely on it being found there. Every ssh DGXMM runs gets it
    explicitly: the weight copy passes -F, and docker-py's ssh:// transport (plain `ssh -l user -- host ...`) runs
    the image's wrapper deploy/ssh/dgxmm-ssh, first on PATH, which adds -F $DGXMM_SSH_CONFIG."""
    env = os.environ.get(SSH_CONFIG_ENV)
    if env:
        return Path(env)
    data_dir = settings.data_dir if settings is not None else Settings.model_fields["data_dir"].default
    return Path(data_dir) / "ssh" / "config"


def _mkdirs_private(d: Path) -> None:
    """Create ``d`` and any missing parents, each one DGXMM creates with mode 0700 (existing ones are left as is)."""
    missing = []
    while not d.exists():
        missing.append(d)
        d = d.parent
    for m in reversed(missing):
        m.mkdir(mode=0o700, exist_ok=True)
        m.chmod(0o700)  # mkdir's mode is filtered by the umask


class UnknownNode(KeyError):
    """Raised when a node ID is not found in the registry."""


class NodeUnreachable(RuntimeError):
    """Raised when a remote node cannot be reached."""


# Failures of the way to the node (SSH client, socket, HTTP-over-SSH), as opposed to the node's Docker daemon
# answering with an error (docker.errors.APIError) or the managed-label guard refusing (PermissionError).
_TRANSPORT_ERRORS = (ConnectionError, TimeoutError, socket.timeout, EOFError, subprocess.SubprocessError,
                     requests.exceptions.ConnectionError, requests.exceptions.Timeout, urllib3.exceptions.HTTPError)


def is_transport_error(e: BaseException) -> bool:
    """True if ``e`` means the node could not be reached (not that its Docker daemon refused the request)."""
    if isinstance(e, NodeUnreachable):
        return True
    if isinstance(e, (docker.errors.APIError, PermissionError)):
        return False  # APIError is an OSError (requests.HTTPError) but the daemon did answer
    if isinstance(e, _TRANSPORT_ERRORS) or isinstance(e, OSError):
        return True
    if isinstance(e, docker.errors.DockerException):  # docker-py wraps transport failures ("... API version: ...")
        seen: set[int] = set()
        cur = e.__cause__ or e.__context__
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if not isinstance(cur, docker.errors.DockerException) and is_transport_error(cur):
                return True
            cur = cur.__cause__ or cur.__context__
    return False


class RemoteDocker:
    """A remote node's DockerController in which every call that fails on the transport (SSH down, timeout, broken
    pipe) raises NodeUnreachable instead; other errors pass through unchanged. The one place where transport errors
    are mapped. ``controller`` is the controller or a zero-argument function returning it: then it is only built
    inside the first method call (a worker thread), since connecting over SSH blocks."""

    def __init__(self, controller: DockerController | Callable[[], DockerController], node_id: str):
        self._resolve = controller if callable(controller) else (lambda: controller)
        self.node_id = node_id

    def _mapped(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except NodeUnreachable:
            raise
        except Exception as e:
            if is_transport_error(e):
                raise NodeUnreachable(f"node {self.node_id} is unreachable: {e}") from e
            raise

    def __getattr__(self, name: str):
        if name.startswith("__") or not callable(getattr(DockerController, name, None)):
            return self._mapped(lambda: getattr(self._resolve(), name))  # e.g. client, directory

        def call(*args, **kwargs):
            return self._mapped(lambda: getattr(self._resolve(), name)(*args, **kwargs))
        call.__name__ = name
        return call


class NodeRegistry:
    """Registry of compute nodes with lazy Docker client construction."""

    def __init__(self, settings: Settings, client_factory: Callable[[NodeConfig], Any] | None = None):
        """Initialize the registry and write SSH config for remote nodes.
        
        Args:
            settings: Runtime settings with node configurations
            client_factory: Optional factory for creating Docker clients (for testing)
        """
        self.settings = settings
        self._client_factory = client_factory
        self._controllers: dict[str, DockerController] = {}
        self._node_map = {n.id: n for n in settings.all_nodes()}
        
        # Write SSH config if we have remote nodes
        remote_nodes = [n for n in settings.all_nodes() if self.is_remote(n.id)]
        if remote_nodes:
            self._write_ssh_config(remote_nodes)
    
    def _write_ssh_config(self, remote_nodes: list[NodeConfig]) -> None:
        """Write SSH config with pinned known_hosts for remote nodes."""
        config_path = ssh_config_path(self.settings)
        ssh_dir = config_path.parent
        _mkdirs_private(ssh_dir)
        config_lines = []
        
        for node in remote_nodes:
            # Remote nodes must have cluster_ip and known_hosts (validated in Settings)
            if not node.cluster_ip:
                raise ValueError(f"remote node {node.id} requires cluster_ip")
            if not node.known_hosts:
                raise ValueError(f"remote node {node.id} requires known_hosts")
            
            # Extract ssh_user from docker_url if not provided
            ssh_user = node.ssh_user
            if not ssh_user:
                # Parse ssh://user@host from docker_url
                if node.docker_url.startswith("ssh://"):
                    url_part = node.docker_url[6:]  # strip ssh://
                    if "@" in url_part:
                        ssh_user = url_part.split("@")[0]
                if not ssh_user:
                    raise ValueError(f"remote node {node.id} requires ssh_user (not found in docker_url)")
            
            # Write per-node known_hosts file
            known_hosts_path = ssh_dir / f"known_hosts_{node.id}"
            known_hosts_path.write_text(node.known_hosts + "\n")
            known_hosts_path.chmod(0o600)
            
            # Add Host block to config
            config_lines.extend([
                f"Host {node.cluster_ip}",
                f"  User {ssh_user}",
                f"  IdentityFile {self.settings.node_ssh_key_path}",
                f"  UserKnownHostsFile {known_hosts_path}",
                "  StrictHostKeyChecking yes",
                "  BatchMode yes",
                "  ConnectTimeout 10",  # a node that is down fails in 10 s, not after the kernel's TCP timeout
                "  ServerAliveInterval 15",  # a dead link ends the session after 3 x 15 s without an answer
                "  ServerAliveCountMax 3",
                "",
            ])
        
        if config_lines:
            config_path.write_text("\n".join(config_lines))
            config_path.chmod(0o600)
    
    def ids(self) -> list[str]:
        """Return all node IDs."""
        return list(self._node_map.keys())
    
    @property
    def local_id(self) -> str:
        """Return the local node ID."""
        return LOCAL_NODE_ID
    
    def config(self, node_id: str) -> NodeConfig:
        """Get the configuration for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            NodeConfig for the requested node
            
        Raises:
            UnknownNode: If node_id is not found
        """
        if node_id not in self._node_map:
            raise UnknownNode(f"node {node_id} not found")
        return self._node_map[node_id]
    
    def is_remote(self, node_id: str) -> bool:
        """Check if a node is remote (not local).
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if the node is remote, False if it is the local node
        """
        return node_id != LOCAL_NODE_ID
    
    def docker(self, node_id: str) -> DockerController:
        """Get the Docker controller for a node (lazy, cached).
        
        Args:
            node_id: Node identifier
            
        Returns:
            DockerController for the requested node
            
        Raises:
            UnknownNode: If node_id is not found
        """
        if node_id not in self._node_map:
            raise UnknownNode(f"node {node_id} not found")
        
        if node_id not in self._controllers:
            node = self._node_map[node_id]
            if self._client_factory:
                # Testing: use the provided factory
                client = self._client_factory(node)
                self._controllers[node_id] = DockerController(client, lookup_client=client)
            else:
                # Production: use DockerController.for_node
                self._controllers[node_id] = DockerController.for_node(node)
        
        return self._controllers[node_id]
