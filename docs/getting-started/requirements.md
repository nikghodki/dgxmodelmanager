---
description: "What DGXMM needs on the host (GPU driver, Docker, the NVIDIA runtime, disk, ports) and the hardware it supports."
---

# Requirements

## Host

- Linux on arm64 or x86_64 with an NVIDIA GPU and driver (`nvidia-smi` works). DGX OS works as is.
- Docker Engine with the Compose v2 plugin (`docker compose version`).
- The NVIDIA Container Toolkit, registered as the Docker `nvidia` runtime
  (`docker info --format '{{json .Runtimes}}'` lists `nvidia`).
- A normal user in the `docker` group (the installer refuses to run as root), with `git` and `bash`.
- Internet access while installing: the image build pulls its base images from Docker Hub and packages
  from PyPI and npm; engine images come from Docker Hub (vLLM) and GitHub's registry (llama.cpp), and
  models from Hugging Face.
- Disk space: about 25 GB on the Docker disk for the vLLM engine image (about 21 GB, pulled on first
  start) and DGXMM's own image, and 50 GiB or more free for models.
- Free ports: 8090 for the UI (`--port` to change), 4000 for the bundled LiteLLM if you use it, and
  8100–8199 on `127.0.0.1` for the engines.
- For the optional decision model: about 13 GiB of free memory while it runs (9 GiB with d3-nano), 8.5 GiB
  in the HF cache, and 30 GB free on the Docker disk to build its image (it reuses the vLLM image as its
  base).

!!! warning "The Docker socket is root"
    DGXMM mounts the Docker socket so it can start engine containers. Access to the Docker socket is
    equivalent to root on the host: only give DGXMM's `admin` role to people you would trust with root
    (see [security hardening](../security/hardening.md)).

## Supported hardware

- **NVIDIA DGX Spark** (GB10, arm64, 128 GB unified memory, DGX OS) is what DGXMM is built and tested on.
- **Other machines with unified CPU/GPU memory** and one NVIDIA GPU should work the same way, but are
  untested.
- **Machines with a discrete GPU** (x86_64 workstations, servers) install and run, but the memory check
  budgets host memory (`MemAvailable`), not the GPU's own memory.

!!! warning "Discrete GPUs: check what fits yourself"
    On a machine with a discrete GPU, DGXMM's fit decisions do not reflect VRAM, because the memory
    budget is host memory. Only the first GPU is reported, and engines see all GPUs.

## Next steps

- [Install](install.md)
- [Memory budget and fit planner](../architecture/memory.md)
