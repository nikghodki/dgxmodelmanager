---
description: "How the process list is collected and masked."
---

# Process view

`memory/hostmem.py` scans `/proc` for `GET /api/host/processes` and for the external processes in the
budget. Per process it reads `status` first and skips kernel threads, zombies and processes with no
RSS and no GPU memory, then reads `cgroup` (to find the Docker container id) and `cmdline`.
`/proc/<pid>/environ` is never opened. GPU memory comes from NVML, keyed by host PID; NVML only lists
processes of the caller's PID namespace, so the DGXMM container runs with `pid: host`. A process's GPU memory
counts for its deployment when it runs in a managed container (see [memory budget](memory.md#budget)).

- **User.** Host processes get their name from the host's `/etc/passwd` (re-read when it changes).
  Processes in containers show the uid as seen inside the container, because container uids are not
  host accounts.
- **Container.** The cgroup's container id is resolved to name and image through the container
  directory in `docker_ctl.py`: one read-only `docker ps` listing, cached for 30 seconds and refreshed by
  a background loop. A miss may refresh early, at most every 5 seconds; Docker errors keep the last
  map. The budget loop only reads the cache, so it never waits for Docker.
- **Managed.** A process is DGXMM-managed when its container is an active deployment's container. A
  container with DGXMM labels that is not an active deployment is reported as `dgxmm_label`, and its
  memory counts as external.
- **Command.** `memory/redact.py` masks passwords, tokens and keys in the argv before it leaves the
  probe. Values that cannot be parsed reliably are masked as a whole (fail closed), and input size is
  capped so masking cost stays bounded. The result is shell-quoted and cut at 4 KiB. Masking is
  best-effort and favours over-masking; `tests/test_redact.py` documents the cases it covers.

Viewers and admins get the same masked data. The scan runs in a worker thread and takes tens of
milliseconds on a DGX Spark.

## Next steps

- [Startup and background loops](background-loops.md)
