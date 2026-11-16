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

1. **Kill a large GPU user when memory is critically low** (the sample below prefers processes outside
   DGXMM). Query running processes with:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
   ```

2. **Set the threshold below DGXMM's reserved margins.** For example, trigger when `MemAvailable` falls
   below 6 GiB (below DGXMM's 8 GiB `safety_margin_gib` + 4 GiB `os_reserve_gib`). DGXMM only places a model
   when it fits with those margins left over, so the guard fires only when something DGXMM does not account
   for (another process, or an engine using more than its estimate) has eaten into them. It does not protect
   DGXMM's models: when it fires, the largest GPU process is often a DGXMM engine.

3. **Combine with earlyoom for CPU-only protection:**
   - Configure with swap ignored (`--ignore-root-user` to protect system services)
   - Exclude critical services: sshd, systemd, dockerd, containerd
   - Set memory threshold to match the GPU guard (e.g., 6 GiB available)

**If the guard kills a DGXMM engine.** DGXMM notices it like any other engine crash: its liveness check sees
the container exit, or, when only a process inside the container died, the engine health probe fails
(`engine_health_interval_s`, see [health checks](../user-guide/deployments.md#health-checks)). Either way the
deployment is marked failed and DGXMM stops counting its memory; start it again once memory is free.

**Example guard script skeleton.** It prefers GPU processes outside DGXMM: processes in containers labelled
`dgxmm.managed=true` (DGXMM's engines) are only killed when no other GPU process is left. Drop the label check
to always kill the largest GPU process.
```bash
#!/bin/bash
# Check MemAvailable
AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
THRESHOLD=$((6 * 1024 * 1024))  # 6 GiB in KiB

dgxmm_managed() {  # is PID $1 in a container labelled dgxmm.managed=true?
  local cid
  cid=$(grep -oE '[0-9a-f]{64}' "/proc/$1/cgroup" 2>/dev/null | head -1)
  [ -n "$cid" ] && [ "$(docker inspect -f '{{index .Config.Labels "dgxmm.managed"}}' "$cid" 2>/dev/null)" = true ]
}

if [ "$AVAIL" -lt "$THRESHOLD" ]; then
  # GPU processes, largest first
  mapfile -t PIDS < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | \
                      sort -t, -k2 -rn | cut -d, -f1 | tr -d ' ')
  VICTIM=""
  for pid in "${PIDS[@]}"; do
    if ! dgxmm_managed "$pid"; then VICTIM=$pid; break; fi
  done
  VICTIM=${VICTIM:-${PIDS[0]}}  # only DGXMM engines left: the largest of them

  if [ -n "$VICTIM" ]; then
    logger "GPU OOM guard: killing process $VICTIM (MemAvailable: ${AVAIL}K)"
    kill -9 "$VICTIM"
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
