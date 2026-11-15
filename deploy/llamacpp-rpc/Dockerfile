# DGXMM llama.cpp RPC image: llama.cpp with the CUDA and RPC backends, for llama.cpp deployments that span nodes.
# A worker rank runs `ggml-rpc-server --host <its cluster ip> -p <port>` (no -c tensor cache); the head rank runs
# `llama-server ... --rpc <ip:port,...> --tensor-split <a,b>`. Both binaries are on PATH and the image has no
# entrypoint: DGXMM passes the binary as the first word of the command.
# Non-root: DGXMM runs every rank with --user <DGXMM uid:gid> (the owner of the HF caches); without --user the image
# runs as the unprivileged uid 10001, never root.
# Built locally (never pushed) and tagged dgxmm-llamacpp-rpc:61dbf49-u1, then copied to the other node with
#   docker save dgxmm-llamacpp-rpc:61dbf49-u1 | ssh <other node> docker load
# Build by hand:  docker build -t dgxmm-llamacpp-rpc:61dbf49-u1 deploy/llamacpp-rpc
#
# Supply chain: both bases are pinned by digest (the arm64 manifests of nvidia/cuda:13.0.1-{devel,runtime}-ubuntu24.04)
# and the source is pinned to a full commit id, checked after the fetch.
# The build targets the DGX Spark (GB10, sm_121) only: CMAKE_CUDA_ARCHITECTURES=121, GGML_NATIVE off.
# GGML_BACKEND_DL=ON (as upstream's cuda.Dockerfile): the CPU, CUDA and RPC backends are shared objects next to the
# binaries in /app, loaded at start, so the binaries also start (e.g. --help) on a host without the NVIDIA driver.
# RDMA: the RPC transport auto-negotiates RDMA through libibverbs (GGML_RPC_RDMA=ON); the runtime stage carries
# libibverbs1 and the ibverbs providers so the ConnectX NIC is usable when the container gets the RDMA devices.
ARG LLAMACPP_COMMIT=61dbf49228c8a96b72e40c89d93cb7c003411f8c

FROM nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d AS build
ARG LLAMACPP_COMMIT
RUN set -eu; apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libibverbs-dev libgomp1; \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN set -eu; git init -q llama.cpp; cd llama.cpp; \
    git remote add origin https://github.com/ggml-org/llama.cpp; \
    git fetch -q --depth 1 origin "$LLAMACPP_COMMIT"; git checkout -q FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "$LLAMACPP_COMMIT"
WORKDIR /src/llama.cpp
RUN set -eu; \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DGGML_BACKEND_DL=ON -DGGML_RPC=ON -DGGML_RPC_RDMA=ON -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined; \
    cmake --build build --config Release -j"$(nproc)" --target llama-server ggml-rpc-server; \
    mkdir -p /out; \
    cp build/bin/llama-server build/bin/ggml-rpc-server /out/; \
    find build -name "*.so*" -exec cp -P {} /out/ \;

FROM nvidia/cuda@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4
ARG LLAMACPP_COMMIT
RUN set -eu; apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 libibverbs1 ibverbs-providers curl; \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /app/
# the backends are found next to the executable (/proc/self/exe resolves these links to /app)
RUN set -eu; echo /app > /etc/ld.so.conf.d/llama.cpp.conf; ldconfig; \
    ln -s /app/llama-server /usr/local/bin/llama-server; ln -s /app/ggml-rpc-server /usr/local/bin/ggml-rpc-server; \
    ls /app/libggml-rpc.so /app/libggml-cuda.so /app/libggml-cpu.so; \
    ggml-rpc-server --help > /dev/null; llama-server --help > /dev/null
RUN useradd --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin llama
USER 10001:10001
LABEL org.opencontainers.image.title="dgxmm-llamacpp-rpc" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.revision="${LLAMACPP_COMMIT}" \
      dgxmm.llamacpp.rpc="1"
ENTRYPOINT []
CMD ["llama-server", "--help"]
