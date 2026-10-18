---
description: "Model directories, engine images, the Hugging Face token and tuning the memory budget."
---

# Models, engines and memory

## Model directories

- **Hugging Face models (vLLM)** live in the HF cache (`DGXMM_HF_CACHE`). DGXMM lists complete snapshots
  and downloads into that cache. Models you fetch with `huggingface-cli download` show up too, after the
  next rescan.
- **GGUF files (llama.cpp)** are found under every `gguf_roots` entry, and also inside the HF cache.

To add another GGUF directory, for example `/data/more-gguf`:

1. Add it to `config.yaml`. `local` must be a new path inside the container:

   ```yaml
   gguf_roots:
     - host: /home/me/models        # keep the existing entry
       local: /data/gguf/0
     - host: /data/more-gguf
       local: /data/gguf/1
   ```

2. Mount it into the `dgxmm` container with a `docker-compose.override.yaml` in the checkout:

   ```yaml
   services:
     dgxmm:
       volumes:
         - /data/more-gguf:/data/gguf/1
   ```

3. Run `./install.sh`. llama.cpp containers mount each root read-only at `/models/<index>`.

`docker-compose.override.yaml` is not tracked by git. To keep it out of `git status`, add it to
`.git/info/exclude`.

## Engine images

The default vLLM and llama.cpp images are pinned by digest (multi-arch manifest lists, arm64 and
amd64), so upgrades of DGXMM do not silently change them. A `config.yaml` written before the
llama.cpp pin keeps the moving `ghcr.io/ggml-org/llama.cpp:server-cuda` tag until you change it. To
pin or change either, set `images` in `config.yaml`. The manifest-list digest of a tag is the first
`Digest:` printed by `docker buildx imagetools inspect <image>:<tag>`:

```yaml
images:
  vllm: vllm/vllm-openai:<tag>             # any tag or digest published for your CPU architecture
  llamacpp: ghcr.io/ggml-org/llama.cpp@sha256:<digest>
```

Images are pulled on first use, and the first vLLM image is pulled in the background when DGXMM first
starts. Pre-pull with `DGXMM_PULL_ENGINES=1 ./install.sh` or `docker pull <image>`. DGXMM caches the
list of model architectures each vLLM image supports in `data/`, keyed by image reference, so a new
image is inspected once.

## Hugging Face token

Gated repos (Llama, Gemma, …) need a token from an account that has accepted the model's licence on
huggingface.co. DGXMM uses it for catalog details and downloads. Engines run offline from the cache, so
they never need it.

```bash
HF_TOKEN=hf_xxx ./install.sh                  # when secrets/hf_token is missing or empty
# or, to replace an existing token:
umask 077; printf '%s' 'hf_xxx' > secrets/hf_token; docker restart dgxmm
```

## Tuning the memory budget

- **`safety_margin_gib`** (8) is the most important knob. It is the memory DGXMM always leaves free,
  and the level below which the watchdog goes critical and starting deployments are aborted. Lower it
  only if the box runs nothing but DGXMM.
- **`os_reserve_gib`** (4) adds to the margin for the OS and desktop. On a headless box you can lower it.
  Memory used by other processes is measured and counted separately as "external" either way.
- **Engine overheads** (`vllm_runtime_fixed_gib`, `vllm_cuda_graph_gib`, `llamacpp_runtime_fixed_gib`)
  apply to every estimate. Calibration already corrects per model from measurements, so change these
  only if every model is off in the same direction.
- **`auto_protect: true`** lets the watchdog stop the lowest-priority deployment when `MemAvailable`
  drops below the margin. Give important deployments a higher **priority** (0–100) in the deploy
  wizard.
- Per deployment, the biggest levers are context length, concurrent sequences, the KV pool size
  (`kv_tokens`), and the KV cache dtype (`fp8` halves KV memory).

## Next steps

- [Memory budget and fit planner](../architecture/memory.md)
- [Networking and reverse proxy](networking.md)
