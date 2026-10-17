# DGXMM decision model ("decider"): a pinned d3 package (vllm-sr/d3-mini by default, or vllm-sr/d3-nano) served by
# the package's own d3_server.py. The container's D3_REPO/D3_REVISION (set by DGXMM) select the model.
# Built locally (never pushed) as dgxmm-decider:<version>; the app builds it on demand (POST /api/decider/image/build)
# or by hand:  docker build -t dgxmm-decider:3 deploy/decider
#
# Supply chain (docs/security.md):
# - the base is pinned by digest (the same torch/CUDA build as the vLLM engine image);
# - the only Python additions are the four wheels in requirements.lock, installed with --require-hashes --no-deps:
#   a changed file on the index fails the build, and pip cannot pull in anything that is not listed;
# - `pip check` must not report anything the base did not already report;
# - the full Python package list (the image's SBOM) is printed in the build log and kept in the image at
#   /usr/local/share/dgxmm-decider/sbom-pip.txt; scripts/decider_image_check.sh builds twice and compares it;
# - at every start launch.py copies the model snapshot's pinned code/config/tokenizer files into a tmpfs, checks the
#   copies against the pinned hashes and imports only from there (python3 -I).
FROM vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8

COPY requirements.lock /usr/local/share/dgxmm-decider/requirements.lock
RUN set -eu; \
    export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1; \
    python3 -m pip check > /tmp/pip-check.before 2>&1 || true; \
    python3 -m pip install --no-cache-dir --no-deps --require-hashes --only-binary=:all: \
        -r /usr/local/share/dgxmm-decider/requirements.lock; \
    python3 -m pip check > /tmp/pip-check.after 2>&1 || true; \
    if ! diff -u /tmp/pip-check.before /tmp/pip-check.after; then \
        echo "dgxmm-decider: the locked packages introduced new dependency conflicts (above)" >&2; exit 1; fi; \
    python3 -c "import transformers, fla; print('transformers', transformers.__version__, 'fla', fla.__version__)"; \
    python3 -m pip list --format=freeze 2>/dev/null | LC_ALL=C sort > /usr/local/share/dgxmm-decider/sbom-pip.txt; \
    echo "SBOM (python packages, $(wc -l < /usr/local/share/dgxmm-decider/sbom-pip.txt)):"; \
    sed 's/^/SBOM /' /usr/local/share/dgxmm-decider/sbom-pip.txt; \
    rm -f /tmp/pip-check.before /tmp/pip-check.after

# Unprivileged runtime user (d3_server needs no root: the GPU device nodes are world-accessible under the NVIDIA
# runtime and every writable path is a tmpfs; see docs/security.md).
RUN useradd --system --uid 10001 --user-group --no-create-home --home-dir /tmp/home --shell /usr/sbin/nologin decider

# Read-only root filesystem at run time: everything that wants to write (Python bytecode, Triton / TorchInductor
# kernel caches, CUDA's JIT cache, HOME) is pointed into /tmp, a tmpfs. The model is mounted read-only under HF_HOME.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/models/hf \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/pyc \
    PYTHONUNBUFFERED=1 \
    D3_REPO=vllm-sr/d3-mini \
    D3_REVISION=97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e

# sha256 of each pinned snapshot's code, manifest, config, tokenizer and readout files (d3-mini 97dd054…, d3-nano
# aeb70d4…): launch.py refuses to start if a copy differs. d3_server.py verifies weights against
# MODEL_MANIFEST.json itself, which is only meaningful because the verifier and the manifest are pinned here.
COPY d3-mini-97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e.sha256 d3-nano-aeb70d4dd53ed25b855c3b0a66394b8fe265db7f.sha256 /usr/local/share/dgxmm-decider/
COPY launch.py /usr/local/lib/dgxmm-decider/launch.py
COPY entrypoint.sh /usr/local/bin/dgxmm-decider
RUN chmod 0755 /usr/local/bin/dgxmm-decider && chmod 0644 /usr/local/lib/dgxmm-decider/launch.py \
           /usr/local/share/dgxmm-decider/*.sha256
# The models this image serves (<repo>@<revision>, as dgxmm/engines/decision.py DECIDER_MODELS); also its
# dgxmm.decider.models label, which DGXMM reads to refuse a model the image predates. The build fails unless each
# one has a well-formed pinned hash list covering the snapshot's code, manifest, config and tokenizer.
ARG DGXMM_DECIDER_MODELS="vllm-sr/d3-mini@97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e,vllm-sr/d3-nano@aeb70d4dd53ed25b855c3b0a66394b8fe265db7f"
RUN set -eu; for m in $(echo "$DGXMM_DECIDER_MODELS" | tr , ' '); do repo=${m%@*}; rev=${m#*@}; f="/usr/local/share/dgxmm-decider/${repo##*/}-$rev.sha256"; [ -s "$f" ] || { echo "dgxmm-decider: no pinned hash list $f for $m" >&2; exit 1; }; if grep -Evq '^[0-9a-f]{64}  [A-Za-z0-9_./-]+$' "$f"; then echo "dgxmm-decider: malformed hash list $f" >&2; exit 1; fi; for need in d3_server.py MODEL_MANIFEST.json config.json tokenizer.json; do grep -Eq "^[0-9a-f]{64}  $need\$" "$f" || { echo "dgxmm-decider: $f does not pin $need" >&2; exit 1; }; done; echo "pinned hashes for $m: $(wc -l < "$f") files"; done
LABEL org.opencontainers.image.title="dgxmm-decider" \
      dgxmm.decider.revision="97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e" \
      dgxmm.decider.models="${DGXMM_DECIDER_MODELS}" \
      dgxmm.decider.base="vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8"
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/dgxmm-decider"]
