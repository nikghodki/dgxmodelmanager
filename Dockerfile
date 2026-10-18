# Base images pinned by digest (multi-arch manifest lists). To update one: docker buildx imagetools inspect <image>:<tag>
FROM node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392 AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/data/hf HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY pyproject.toml ./
COPY dgxmm ./dgxmm
COPY deploy/decider ./deploy/decider
RUN pip install --no-cache-dir . && useradd -u 1000 -m dgxmm && mkdir -p /app/data && chown 1000:1000 /app/data
COPY --from=ui /ui/dist /app/static
USER 1000:1000
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/dgxmm/').status==200 else 1)"
CMD ["python", "-m", "dgxmm"]
