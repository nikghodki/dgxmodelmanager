#!/usr/bin/env bash
# End-to-end check on the DGX against a running dgxmm. Deploys small models; tell users first.
# Usage: DGXMM_USER=admin DGXMM_PASSWORD=... scripts/validate_on_box.sh
set -euo pipefail
B=${DGXMM_URL:-http://127.0.0.1:8090/dgxmm/api}
J=$(mktemp); trap 'rm -f "$J"' EXIT
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; exit 1; }
py() { python3 -c "import sys,json; d=json.load(sys.stdin); $1"; }

CSRF=$(curl -sf -c "$J" -H 'content-type: application/json' \
  -d "{\"username\":\"${DGXMM_USER:-admin}\",\"password\":\"${DGXMM_PASSWORD:?set DGXMM_PASSWORD}\"}" "$B/auth/login" | py 'print(d["csrf_token"])')
H=(-sf -b "$J" -H "X-CSRF-Token: $CSRF" -H 'content-type: application/json')
pass "login"

curl "${H[@]}" "$B/host" | py 'b=d["budget"]; assert b["total"]>100*2**30; print("free_for_new GiB", round(b["free_for_new"]/2**30,1))' && pass "host budget"
curl "${H[@]}" "$B/inventory" | py 'assert any(a["ref"]=="Qwen/Qwen3-0.6B" and a["complete"] for a in d)' && pass "inventory has Qwen3-0.6B"

wait_state() {  # id wanted timeout_s
  for _ in $(seq 1 "$3"); do
    s=$(curl "${H[@]}" "$B/deployments/$1" | py 'print(d["state"])')
    [ "$s" = "$2" ] && return 0
    [ "$s" = "failed" ] && return 1
    sleep 1
  done; return 1
}

run_one() {  # name engine ref config_json
  local id
  id=$(curl "${H[@]}" -d "{\"name\":\"$1\",\"engine\":\"$2\",\"model_ref\":\"$3\",\"config\":$4}" "$B/deployments" | py 'print(d["id"])')
  wait_state "$id" ready 600 || { curl "${H[@]}" "$B/deployments/$id" | py 'print(d["last_error"])'; fail "$1 ready"; }
  pass "$1 ready"
  sleep 20
  curl "${H[@]}" "$B/deployments/$id" | py 'r,a=d["reserved_bytes"],d["actual_bytes"] or 0; print(f"reserved {r/2**30:.2f} GiB measured {a/2**30:.2f} GiB ratio {a/max(r,1):.2f}"); assert a<=r*1.15, "measured exceeds reservation by >15%"' && pass "$1 memory within reservation"
  curl "${H[@]}" -d '{"levels":[1,4],"input_len":256,"output_len":64,"duration_s":20,"max_requests":8}' "$B/deployments/$id/benchmark" >/dev/null
  for _ in $(seq 1 120); do n=$(curl "${H[@]}" "$B/deployments/$id/benchmarks" | py 'print(len([b for b in d if b["current"]]))'); [ "$n" -ge 2 ] && break; sleep 1; done
  curl "${H[@]}" "$B/deployments/$id/benchmarks" | py '[print(f"  c={b[\"concurrency\"]} total={b[\"tps_total\"]:.1f} tok/s ttft={b[\"ttft_p50\"]}") for b in d if b["current"]]' && pass "$1 benchmark"
  curl "${H[@]}" "$B/deployments/$id/advice" | py 'print("  advice:", [a["id"] for a in d])' && pass "$1 advice"
  curl "${H[@]}" "$B/deployments/$id" | py 'print("  litellm:", d["litellm_status"])'
  curl "${H[@]}" -X POST "$B/deployments/$id/stop" >/dev/null && curl "${H[@]}" -X DELETE "$B/deployments/$id" >/dev/null && pass "$1 stopped and removed"
}

run_one val-vllm vllm Qwen/Qwen3-0.6B '{"max_model_len":8192,"max_num_seqs":4}'
G=$(curl "${H[@]}" "$B/inventory" | py 'g=[a["ref"] for a in d if a["source"]=="gguf" and a["complete"] and "0.6B" in a["ref"]]; print(g[0] if g else "")')
if [ -n "$G" ]; then run_one val-llama llamacpp "$G" '{"ctx_size":8192,"parallel":2}'; else echo "SKIP  no small GGUF on disk (Task 24 downloads one)"; fi
echo "ALL CHECKS PASSED"
