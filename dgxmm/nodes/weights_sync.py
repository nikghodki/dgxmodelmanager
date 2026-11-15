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


async def copy_weights(db, settings, nodes, node_id: str, ref: str, revision: str, user_id: int | None = None,
                      events=None) -> dict:
    """Copy model weights to a remote node via tar | ssh tar pipeline.
    
    Creates a Download record with kind="copy" and spawns the copy process.
    Returns the job view dict.
    """
    import asyncio
    from dgxmm.db import Download
    from dgxmm.inventory.artifact import REPO_RE
    from sqlalchemy import select
    
    # Validate inputs
    if node_id == "local":
        raise ValueError("cannot copy to local node")
    if not nodes or node_id not in nodes.ids():
        raise ValueError(f"unknown node: {node_id}")
    if not re.match(REPO_RE, ref):
        raise ValueError(f"invalid ref: {ref}")
    
    # Compute repo_dir
    repo_dir = "models--" + ref.replace("/", "--")
    _validate_repo_dir(repo_dir)
    
    # Check if already copying
    with db.session() as s:
        existing = s.scalar(select(Download).where(
            Download.repo_id == ref,
            Download.kind == "copy",
            Download.state.in_(("queued", "running"))
        ))
        if existing:
            raise ValueError(f"{ref} is already being copied")
    
    # Create Download record
    with db.session() as s:
        row = Download(
            repo_id=ref,
            revision=revision,
            files=[],  # We don't track individual files for copies
            bytes_total=0,  # Unknown until tar runs
            bytes_done=0,
            state="queued",
            kind="copy",
            created_by=user_id
        )
        s.add(row)
        s.flush()
        job_id = row.id
    
    # Spawn the copy task
    asyncio.create_task(_run_copy(db, settings, nodes, job_id, node_id, ref, revision, repo_dir, events))
    
    # Return job view
    with db.session() as s:
        d = s.get(Download, job_id)
        return {
            "id": d.id,
            "repo_id": d.repo_id,
            "revision": d.revision,
            "files": d.files,
            "state": d.state,
            "bytes_done": d.bytes_done,
            "bytes_total": d.bytes_total,
            "error": d.error,
            "kind": d.kind,
            "created_at": d.created_at.isoformat()
        }


async def _run_copy(db, settings, nodes, job_id: int, node_id: str, ref: str, revision: str, repo_dir: str,
                   events=None):
    """Run the tar | ssh tar pipeline to copy weights."""
    import asyncio
    
    def update(state: str, error: str | None = None, bytes_done: int | None = None):
        with db.session() as s:
            d = s.get(Download, job_id)
            d.state = state
            if error is not None:
                d.error = error
            if bytes_done is not None:
                d.bytes_done = bytes_done
        if events:
            with db.session() as s:
                d = s.get(Download, job_id)
                events.publish("download", {
                    "id": d.id,
                    "repo_id": d.repo_id,
                    "revision": d.revision,
                    "files": d.files,
                    "state": d.state,
                    "bytes_done": d.bytes_done,
                    "bytes_total": d.bytes_total,
                    "error": d.error,
                    "kind": d.kind,
                    "created_at": d.created_at.isoformat()
                })
    
    try:
        update("running")
        
        node = nodes.config(node_id)
        tar_argv, ssh_argv = copy_argv(settings.hf_cache.local, node, repo_dir, revision)
        
        # Create the two subprocess pipes: tar -> ssh tar
        tar_proc = await asyncio.create_subprocess_exec(
            *tar_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        ssh_proc = await asyncio.create_subprocess_exec(
            *ssh_argv,
            stdin=tar_proc.stdout,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Close tar stdout in parent so ssh gets EOF when tar exits
        if tar_proc.stdout:
            tar_proc.stdout.close()
        
        # Wait for both processes
        tar_task = asyncio.create_task(tar_proc.wait())
        ssh_task = asyncio.create_task(ssh_proc.wait())
        
        tar_code = await tar_task
        ssh_code = await ssh_task
        
        # Collect stderr
        tar_stderr = (await tar_proc.stderr.read()).decode('utf-8', errors='replace') if tar_proc.stderr else ""
        ssh_stderr = (await ssh_proc.stderr.read()).decode('utf-8', errors='replace') if ssh_proc.stderr else ""
        
        if tar_code != 0:
            tail = tar_stderr.strip().split('\n')[-10:]  # Last 10 lines
            update("failed", f"tar failed with code {tar_code}: {' '.join(tail)}")
        elif ssh_code != 0:
            tail = ssh_stderr.strip().split('\n')[-10:]
            update("failed", f"ssh failed with code {ssh_code}: {' '.join(tail)}")
        else:
            update("done")
    
    except asyncio.CancelledError:
        # Kill both processes
        if 'tar_proc' in locals():
            try:
                tar_proc.kill()
                await tar_proc.wait()
            except Exception:
                pass
        if 'ssh_proc' in locals():
            try:
                ssh_proc.kill()
                await ssh_proc.wait()
            except Exception:
                pass
        update("cancelled")
        raise
    except Exception as e:
        update("failed", str(e))
        raise
