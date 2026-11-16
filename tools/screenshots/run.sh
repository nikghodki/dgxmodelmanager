#!/usr/bin/env bash
# Regenerate docs/images/ from the released UI against made-up demo data (see docs/README.md).
#   tools/screenshots/run.sh [shot-name ...]
# Needs only Docker. Uses the DGXMM image's built frontend (DGXMM_IMAGE, default dgxmm:<pyproject version>),
# a pinned Playwright image (no network: every request is answered by the capture script) and a pinned oxipng image.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/pyproject.toml" | head -1)
image=${DGXMM_IMAGE:-dgxmm:$version}
playwright=mcr.microsoft.com/playwright/python:v1.55.0-noble@sha256:640d578aae63cfb632461d1b0aecb01414e4e020864ac3dd45a868dc0eff3078
oxipng=ghcr.io/shssoichiro/oxipng:v9.1.5@sha256:e1623b0da9f14b63b242ffb78e4eef4ba0f09c9159880d238665af4654c75b4f
out=${OUT:-$root/docs/images}

work=$(mktemp -d)
cid=""
cleanup() {
  if [ -n "$cid" ]; then docker rm -f "$cid" >/dev/null 2>&1 || true; fi
  rm -rf "$work"
}
trap cleanup EXIT

cid=$(docker create "$image")
docker cp -q "$cid:/app/static" "$work/static"
docker rm "$cid" >/dev/null && cid=""
mkdir -p "$out"

# the Playwright Python package (the image ships the browsers only), installed once into the temp dir
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$work:/work" -v "$here:/tool:ro" "$playwright" \
  pip install -q --disable-pip-version-check --only-binary=:all: --target /work/py -r /tool/requirements.txt

only=()
[ $# -gt 0 ] && only=(--only "$@")
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/py \
  -v "$here:/tool:ro" -v "$work/static:/static:ro" -v "$work/py:/py:ro" -v "$out:/out" \
  "$playwright" python3 /tool/capture.py --static /static --out /out "${only[@]}"

# lossless: smaller files, same pixels
docker run --rm --network none --user "$(id -u):$(id -g)" -v "$out:/out" "$oxipng" -o max --strip safe -q --recursive /out
