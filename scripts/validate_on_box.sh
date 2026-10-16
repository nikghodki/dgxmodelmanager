#!/usr/bin/env bash
# End-to-end check on the DGX against a running dgxmm. Deploys small models; tell users first.
# Usage: DGXMM_USER=admin DGXMM_PASSWORD=... scripts/validate_on_box.sh
set -euo pipefail
B=${DGXMM_URL:-http://127.0.0.1:8090/dgxmm/api}
J=$(mktemp); trap 'rm -f "$J"' EXIT
pass() { echo "PASS  $*"; }
CUR_ID=""
fail() {
  echo "FAIL  $*"
  if [ -n "$CUR_ID" ]; then  # show why, then never leave a test deployment running
    curl "${H[@]}" "$B/deployments/$CUR_ID" | py 'e=[x for x in d.get("events",[]) if x.get("log_tail")]; print("\n".join((e[0]["log_tail"] or "").splitlines()[-25:]) if e else "(no log tail)")' || true
    curl "${H[@]}" -X POST "$B/deployments/$CUR_ID/stop" >/dev/null 2>&1 || true
    curl "${H[@]}" -X DELETE "$B/deployments/$CUR_ID" >/dev/null 2>&1 || true
  fi
  exit 1
}
py() { python3 -c "import sys,json; d=json.load(sys.stdin); $1"; }

CSRF=$(curl -sf -c "$J" -H 'content-type: application/json' \
  -d "{\"username\":\"${DGXMM_USER:-admin}\",\"password\":\"${DGXMM_PASSWORD:?set DGXMM_PASSWORD}\"}" "$B/auth/login" | py 'print(d["csrf_token"])')
H=(-sf -b "$J" -H "X-CSRF-Token: $CSRF" -H 'content-type: application/json')
pass "login"

curl "${H[@]}" "$B/host" | py 'b=d["budget"]; assert b["total"]>100*2**30; print("free_for_new GiB", round(b["free_for_new"]/2**30,1))' || fail "host budget"
pass "host budget"
curl "${H[@]}" "$B/inventory" | py 'assert any(a["ref"]=="Qwen/Qwen3-0.6B" and a["complete"] for a in d)' || fail "inventory missing Qwen3-0.6B"
pass "inventory has Qwen3-0.6B"

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
  CUR_ID=$id
  wait_state "$id" ready 600 || { curl "${H[@]}" "$B/deployments/$id" | py 'print(d["last_error"])'; fail "$1 ready"; }
  pass "$1 ready"
  sleep 20
  curl "${H[@]}" "$B/deployments/$id" | py 'r,a=d["reserved_bytes"],d["actual_bytes"] or 0; print("reserved %.2f GiB measured %.2f GiB ratio %.2f (measured = cgroup + NVML; on GB10 CUDA memory is not attributed, so the gap is credited out of external while ready)" % (r/2**30, a/2**30, a/max(r,1))); assert a<=r*1.15, "measured exceeds reservation by >15%"' || fail "$1 memory check"
  curl "${H[@]}" "$B/host" | py 'b=d["budget"]; print("  host while ready: external %.1f GiB committed %.1f GiB free_for_new %.1f GiB" % (b["external"]/2**30, b["committed"]/2**30, b["free_for_new"]/2**30)); assert b["free_for_new"] == b["usable"]-b["committed"]-b["external"]' || fail "$1 budget identity"
  pass "$1 memory within reservation"
  # benchmark rows are kept per model; only look at rows created by THIS run (id > the newest existing one)
  export BMIN; BMIN=$(curl "${H[@]}" "$B/deployments/$id/benchmarks" | py 'print(max([b["id"] for b in d], default=0))')
  curl "${H[@]}" -d '{"levels":[1,4],"input_len":256,"output_len":64,"duration_s":20,"max_requests":8}' "$B/deployments/$id/benchmark" >/dev/null
  for _ in $(seq 1 120); do n=$(curl "${H[@]}" "$B/deployments/$id/benchmarks" | py 'import os; print(len([b for b in d if b["id"]>int(os.environ["BMIN"])]))'); [ "$n" -ge 2 ] && break; sleep 1; done
  [ "$n" -ge 2 ] || fail "$1 benchmark timeout (got $n rows, need 2)"
  curl "${H[@]}" "$B/deployments/$id/benchmarks" | py 'import os; c=[b for b in d if b["id"]>int(os.environ["BMIN"])]; assert len(c)>=2, "need 2+ new rows"; assert all(b["current"] for b in c), "new rows must match the current config"; assert all(b["tps_total"]>0 and b["requests"]>0 for b in c), "invalid benchmark data"; print("\n".join("  c=%d total=%.1f tok/s ttft=%.3f requests=%d" % (b["concurrency"], b["tps_total"], b.get("ttft_p50") or 0, b["requests"]) for b in c))' || fail "$1 benchmark verification"
  pass "$1 benchmark"
  curl "${H[@]}" "$B/deployments/$id/advice" | py 'print("  advice:", [a["id"] for a in d])' || fail "$1 advice fetch"
  pass "$1 advice"
  curl "${H[@]}" "$B/deployments/$id" | py 'print("  litellm:", d["litellm_status"])' || fail "$1 litellm status fetch"
  curl "${H[@]}" -X POST "$B/deployments/$id/stop" >/dev/null || fail "$1 stop"
  curl "${H[@]}" -X DELETE "$B/deployments/$id" >/dev/null || fail "$1 delete"
  CUR_ID=""
  pass "$1 stopped and removed"
}

run_one val-vllm vllm Qwen/Qwen3-0.6B '{"max_model_len":8192,"max_num_seqs":4}'
G=$(curl "${H[@]}" "$B/inventory" | py 'g=[a["ref"] for a in d if a["source"]=="gguf" and a["complete"] and "0.6B" in a["ref"]]; print(g[0] if g else "")')
if [ -n "$G" ]; then run_one val-llama llamacpp "$G" '{"ctx_size":8192,"parallel":2}'; else echo "SKIP  no small GGUF on disk (Task 24 downloads one)"; fi
echo "ALL CHECKS PASSED"
