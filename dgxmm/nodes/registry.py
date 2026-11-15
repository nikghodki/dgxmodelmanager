"""Node registry: Docker clients for local and remote nodes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from dgxmm.config import LOCAL_NODE_ID, NodeConfig, Settings
from dgxmm.docker_ctl import DockerController


class UnknownNode(KeyError):
    """Raised when a node ID is not found in the registry."""


class NodeUnreachable(RuntimeError):
    """Raised when a remote node cannot be reached."""


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
        home = Path(os.environ.get("HOME", str(Path.home())))
        ssh_dir = home / ".ssh"
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        
        config_path = ssh_dir / "config"
        config_lines = []
        
        for node in remote_nodes:
            if not node.cluster_ip or not node.ssh_user or not node.known_hosts:
                continue
            
            # Write per-node known_hosts file
            known_hosts_path = ssh_dir / f"known_hosts_{node.id}"
            known_hosts_path.write_text(node.known_hosts + "\n")
            known_hosts_path.chmod(0o600)
            
            # Add Host block to config
            config_lines.extend([
                f"Host {node.cluster_ip}",
                f"  User {node.ssh_user}",
                f"  IdentityFile {self.settings.node_ssh_key_file}",
                f"  UserKnownHostsFile {known_hosts_path}",
                "  StrictHostKeyChecking yes",
                "  ServerAliveInterval 15",
                "  BatchMode yes",
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
