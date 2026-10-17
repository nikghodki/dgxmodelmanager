#!/bin/sh
# Supply-chain check of the decider image: build deploy/decider twice from scratch (--no-cache, under throwaway tags)
# and compare the Python package sets (the SBOM the Dockerfile writes to /usr/local/share/dgxmm-decider/sbom-pip.txt
# and prints in the build log). Exit 0 = identical. Removes both throwaway images afterwards.
#   scripts/decider_image_check.sh [tag-prefix]
set -eu
DIR="$(cd "$(dirname "$0")/.." && pwd)/deploy/decider"
P="${1:-dgxmm-decider-check}"
out=$(mktemp -d)
trap 'docker image rm -f "$P:a" "$P:b" >/dev/null 2>&1 || true; rm -rf "$out"' EXIT
for t in a b; do
    docker build --no-cache --pull=false --progress=plain -t "$P:$t" "$DIR" > "$out/build-$t.log" 2>&1 \
        || { tail -30 "$out/build-$t.log"; exit 1; }
    docker run --rm --entrypoint cat "$P:$t" /usr/local/share/dgxmm-decider/sbom-pip.txt > "$out/sbom-$t.txt"
    grep -c '^#[0-9]* [0-9.]* SBOM ' "$out/build-$t.log" > "$out/n-$t" || true
    echo "build $t: $(wc -l < "$out/sbom-$t.txt") packages ($(cat "$out/n-$t") SBOM lines in the build log)"
done
grep -E 'Successfully installed' "$out/build-a.log" | sed 's/^#[0-9]* [0-9.]* //'
if diff -u "$out/sbom-a.txt" "$out/sbom-b.txt"; then
    echo "reproducible: both builds have the same $(wc -l < "$out/sbom-a.txt") Python packages"
    echo "sha256 of the package list: $(sha256sum < "$out/sbom-a.txt" | cut -c1-64)"
else
    echo "NOT reproducible: the package sets differ (above)" >&2
    exit 1
fi
