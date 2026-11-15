"""Weights presence check and copy to remote nodes (Task 6)."""
from __future__ import annotations

import re
from pathlib import Path

from dgxmm.config import NodeConfig
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact


def weights_present(probe, node_id: str, artifact: ModelArtifact) -> bool:
    """Check if weights are present on a remote node using NodeProbe.exec.
    
    Checks for config.json and all weight files listed in artifact.files under the snapshot directory.
    Uses a single exec call with shell test commands.
    """
    if artifact.source != "hf":
        # GGUF files are not synced between nodes
        return False
    
    # Compute repo_dir from ref
    repo_dir = "models--" + artifact.ref.replace("/", "--")
    
    # Build test commands for each file
    snapshot_base = f"/hf/hub/{repo_dir}/snapshots/{artifact.revision}"
    
    # Check config.json and all weight files
    files_to_check = ["config.json"] + [f for f in artifact.files if f and f != "config.json"]
    
    # Build a compound test command: test -f file1 && test -f file2 && ...
    test_parts = [f'test -f "{snapshot_base}/{f}"' for f in files_to_check]
    test_cmd = " && ".join(test_parts)
    
    try:
        # exec with a shell command to run compound test
        probe.exec(node_id, ["sh", "-c", test_cmd])
        return True
    except Exception:
        return False


def _validate_repo_dir(repo_dir: str) -> None:
    """Validate repo_dir to prevent path traversal and ensure valid format."""
    # Check for path traversal
    if ".." in repo_dir or repo_dir.startswith("/"):
        raise ValueError(f"invalid repo_dir: {repo_dir}")
    
    # Extract the ref from repo_dir (remove "models--" prefix and convert "--" back to "/")
    if not repo_dir.startswith("models--"):
        raise ValueError(f"repo_dir must start with 'models--': {repo_dir}")
    
    ref = repo_dir[8:].replace("--", "/", 1)  # Remove "models--" and replace first "--" with "/"
    
    # Validate ref matches REPO_RE pattern
    if not re.match(REPO_RE, ref):
        raise ValueError(f"invalid repo format: {ref}")


def copy_argv(local_hf: str, node: NodeConfig, repo_dir: str, revision: str) -> tuple[list[str], list[str]]:
    """Generate tar and ssh argv lists for copying weights to a remote node.
    
    The tar command reads from local_hf/hub and the ssh command extracts to node.hf_cache_host/hub.
    Only copies the requested snapshot dir, its blobs (symlink targets), and refs/ for that repo.
    
    Args:
        local_hf: Local HF cache path (inside DGXMM container)
        node: Target node configuration
        repo_dir: Repository directory name (e.g., "models--Qwen--Qwen3-0.6B")
        revision: Git revision/snapshot ID
    
    Returns:
        Tuple of (tar_argv, ssh_argv) to be piped together
    """
    # Validate repo_dir
    _validate_repo_dir(repo_dir)
    
    # Build paths
    local_hub = Path(local_hf) / "hub"
    remote_hub = Path(node.hf_cache_host) / "hub"
    
    # Snapshot path
    snapshot_rel = f"{repo_dir}/snapshots/{revision}"
    
    # Build tar command to create archive from local cache
    # Include: snapshot dir, refs/ for this repo
    # The snapshot dir contains symlinks to blobs - tar will follow them
    tar_argv = [
        "tar",
        "-C", str(local_hub),
        "-chf", "-",  # -c create, -h follow symlinks, -f - to stdout
        snapshot_rel,
        f"{repo_dir}/refs",
    ]
    
    # Build ssh command to extract on remote node
    # Use BatchMode=yes for non-interactive
    ssh_argv = [
        "ssh",
        "-o", "BatchMode=yes",
        f"{node.ssh_user}@{node.cluster_ip}",
        "tar",
        "-C", str(remote_hub),
        "-xf", "-",  # -x extract, -f - from stdin
    ]
    
    return tar_argv, ssh_argv
