#!/bin/sh
# Serve the pinned d3 snapshot from the read-only model mount. Extra arguments go to the launcher after the
# defaults (e.g. `--port 8123 --batch-size 4`; argparse: the last one wins).
#
# launch.py copies exactly the files of the image's pinned hash list into a private tmpfs directory, verifies the
# copies and imports only from there (docs/security.md): files added to the snapshot directory are never imported,
# and nothing can be swapped between the check and the import. `python3 -I` keeps the script directory, the current
# directory, the user site and PYTHON* variables off the import path; -B and the pycache prefix keep Python from
# reading or writing bytecode next to any source.
#
# The bearer key is never taken from the environment: DGXMM writes it (mode 0400) into the tmpfs file named by
# DECISION_API_KEY_FILE right after the container starts, and launch.py waits for it, reads it, deletes the file and
# hands it to d3_server's serve() directly, so it is not in `docker inspect`, /proc/*/environ or any child process.
set -eu
repo_dir="${HF_HOME:-/models/hf}/hub/models--$(echo "$D3_REPO" | sed 's#/#--#g')"
snap="$repo_dir/snapshots/$D3_REVISION"
# one pinned hash list per supported model: <model name>-<revision>.sha256 (D3_REPO/D3_REVISION select the model)
hashes="/usr/local/share/dgxmm-decider/${D3_REPO##*/}-$D3_REVISION.sha256"
if [ ! -f "$snap/d3_server.py" ]; then
    echo "dgxmm-decider: snapshot $D3_REPO@$D3_REVISION not found under $repo_dir (download it first)" >&2
    exit 2
fi
if [ ! -f "$hashes" ]; then
    echo "dgxmm-decider: no pinned hash list for revision $D3_REVISION" >&2
    exit 3
fi
if [ -n "${DECISION_API_KEY:-}" ]; then
    echo "dgxmm-decider: DECISION_API_KEY is set in the environment; refusing to start (deliver the key as the" \
         "file named by DECISION_API_KEY_FILE instead)" >&2
    exit 4
fi
mkdir -p "${HOME:-/tmp/home}"
cd /tmp
exec python3 -I -B -u -X pycache_prefix=/tmp/pyc /usr/local/lib/dgxmm-decider/launch.py --snapshot "$snap" \
    --hashes "$hashes" --work-dir /tmp --device cuda:0 --batch-size 8 --host 0.0.0.0 --port 8000 "$@"
