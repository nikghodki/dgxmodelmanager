---
description: "Protecting the host from memory exhaustion on GB10 unified memory systems."
---

# Host memory protection on GB10

On GB10 systems with unified memory, CUDA allocations are not charged to cgroups or RSS. This means:

- A container with `--memory 2g` can allocate 4 GiB of GPU memory without hitting its cgroup limit
- A process using 4.2 GiB of GPU memory may report only 0.6 GiB RSS
- Docker memory limits, the kernel OOM killer, and earlyoom cannot see GPU memory usage

**Impact:** When memory is exhausted by a process outside DGXMM (such as a benchmark or notebook), the host
can freeze (ping responds, but sshd becomes unresponsive) because the kernel's standard memory protection
mechanisms are blind to GPU allocations.

## Recommended protection

Deploy a GPU-aware OOM guard that monitors both CPU and GPU memory:

1. **Kill the largest GPU user when memory is critically low.** Query running processes with:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
   ```

2. **Set the threshold below DGXMM's reserved margins.** For example, trigger when `MemAvailable` falls
   below 6 GiB (below DGXMM's 8 GiB `safety_margin_gib` + 4 GiB `os_reserve_gib`), so DGXMM-placed models
   are never terminated.

3. **Combine with earlyoom for CPU-only protection:**
   - Configure with swap ignored (`--ignore-root-user` to protect system services)
   - Exclude critical services: sshd, systemd, dockerd, containerd
   - Set memory threshold to match the GPU guard (e.g., 6 GiB available)

**Example guard script skeleton:**
```bash
#!/bin/bash
# Check MemAvailable
AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
THRESHOLD=$((6 * 1024 * 1024))  # 6 GiB in KiB

if [ "$AVAIL" -lt "$THRESHOLD" ]; then
  # Find largest GPU process
  LARGEST=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | \
            sort -t, -k2 -rn | head -1 | cut -d, -f1)
  
  if [ -n "$LARGEST" ]; then
    logger "GPU OOM guard: killing process $LARGEST (MemAvailable: ${AVAIL}K)"
    kill -9 "$LARGEST"
  fi
fi
```

Run this guard periodically (e.g., every 5 seconds via cron or systemd timer).

## Why this is needed

DGXMM's own memory management (watchdog, `auto_protect`, memory gates) protects deployments DGXMM manages.
It cannot protect the host from:

- Processes started outside DGXMM (jupyter notebooks, manual vLLM instances, benchmarks)
- Rapid memory allocation spikes before the watchdog can react
- Memory exhaustion when DGXMM is stopped or before it starts

The host-level guard is the last line of defense, ensuring the system remains responsive even when non-DGXMM
workloads consume all memory.

## Verification

Test the guard by running a GPU memory hog outside DGXMM:
```python
import torch
# Allocate until near the threshold
tensors = []
while True:
    tensors.append(torch.zeros(100_000_000, device='cuda'))  # ~400 MB per tensor
```

The guard should terminate the process before the host freezes.

## Next steps

- [Memory budget](../architecture/memory.md)
- [Known limitations](known-limitations.md)
- [Monitoring](monitoring.md)
