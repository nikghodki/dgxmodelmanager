#!/bin/sh
# Regenerate deploy/decider/requirements.lock: resolve the decider's Python additions inside the pinned base image
# (a throwaway container; nothing is installed) and print them hash-pinned. Review the diff before committing.
#   scripts/decider_lock.sh > deploy/decider/requirements.lock.new
set -eu
BASE=$(sed -n 's/^FROM \(.*\)$/\1/p' "$(dirname "$0")/../deploy/decider/Dockerfile" | head -1)
REQS="transformers==5.17.0 flash-linear-attention==0.5.2"
docker run --rm --entrypoint sh -e PIP_DISABLE_PIP_VERSION_CHECK=1 "$BASE" -c "
python3 -m pip install --dry-run --quiet --report /tmp/r.json $REQS >/dev/null 2>&1
python3 - <<'PY'
import json
r = json.load(open('/tmp/r.json'))
for i in r['install']:
    m, a = i['metadata'], i['download_info']
    h = a.get('archive_info', {}).get('hashes', {}).get('sha256')
    if not h:
        raise SystemExit(f\"no sha256 for {m['name']}\")
    print(f\"{m['name']}=={m['version']} \\\\\n    --hash=sha256:{h}\")
PY"
