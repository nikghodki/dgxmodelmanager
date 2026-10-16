#!/usr/bin/env bash
# DGXMM (DGX Model Manager) installer: one command to install, upgrade, inspect or remove DGXMM
# on a DGX Spark (or any Linux arm64/x86_64 host with an NVIDIA GPU, Docker and the NVIDIA
# Container Toolkit).
#
#   git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager && ./install.sh
#
# Re-running is safe: existing .env, config.yaml, data/ and secrets/ are always kept.
# Run ./install.sh --help for options.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONTAINER=dgxmm
IMAGE=dgxmm:latest
DOCKER_SOCK=/var/run/docker.sock
MIN_FREE_GIB=50
ENV_FILE="$REPO_DIR/.env"
CONFIG_FILE="$REPO_DIR/config.yaml"
EXAMPLE_CONFIG="$REPO_DIR/config.example.yaml"
SECRETS_DIR="$REPO_DIR/secrets"
DATA_DIR="$REPO_DIR/data"
ENV_KEYS=(DGXMM_UID DGXMM_GID DOCKER_GID DGXMM_PORT DGXMM_BIND DGXMM_HF_CACHE DGXMM_MODELS_DIR DGXMM_NETWORK DGXMM_HOST_HOME DGXMM_TRUSTED_PROXIES)
# Written to .env only when the bundled LiteLLM is enabled (--with-litellm / DGXMM_WITH_LITELLM=1).
LITELLM_ENV_KEYS=(DGXMM_WITH_LITELLM DGXMM_LITELLM_PORT COMPOSE_PROFILES)
LITELLM_SERVICE=dgxmm-litellm
LITELLM_DB_SERVICE=dgxmm-litellm-db
LITELLM_INTERNAL_URL=http://dgxmm-litellm:4000
LITELLM_DB_DIR="$DATA_DIR/litellm-db"

usage() {
  cat <<'EOF'
DGXMM installer

Usage: ./install.sh [COMMAND] [OPTIONS]

Commands:
  install      Install or upgrade in place (default). Idempotent: keeps .env, config.yaml,
               data/ and secrets/; rebuilds the image and restarts the dgxmm container.
  upgrade      git pull --ff-only (when this is a git checkout), then install.
  status       Show container state/health, URLs, managed engine containers.
  logs         Show the dgxmm container logs (-f to follow).
  uninstall    Stop and remove the dgxmm container and image (and the bundled LiteLLM
               containers, if any). Keeps data/ (incl. the LiteLLM database data/litellm-db),
               secrets/, config.yaml and .env, and leaves engine containers running.
               With --purge also removes those files, the engine containers DGXMM manages
               (label dgxmm.managed=true) and the docker network if install.sh created it.
               Model caches (HF cache, models dir) are never touched.

Options:
  --dry-run, -n        Print what would be done; change nothing.
  --port PORT          Host port for the UI            (env DGXMM_PORT, default 8090)
  --bind ADDR          Host IPv4 address to bind       (env DGXMM_BIND, default 0.0.0.0;
                       use 127.0.0.1 when a reverse proxy fronts DGXMM)
  --network NAME       Docker network for DGXMM+engines (env DGXMM_NETWORK, default dgxmm)
  --hf-cache DIR       Hugging Face cache on the host  (env DGXMM_HF_CACHE,
                       default $HOME/.cache/huggingface)
  --models-dir DIR     GGUF models directory           (env DGXMM_MODELS_DIR, default $HOME/models)
  --with-litellm       Also run a bundled LiteLLM proxy + Postgres (compose profile "litellm"):
                       one OpenAI-compatible endpoint http://<host>:<litellm port>/v1 for all
                       DGXMM deployments (env DGXMM_WITH_LITELLM=1). Remembered in .env.
  --litellm-port PORT  Host port for the bundled LiteLLM (env DGXMM_LITELLM_PORT, default 4000;
                       bound to the same address as the UI)
  --allow-root         Allow running as root (not recommended).
  --purge              With uninstall: also delete DGXMM state (see above).
  -f, --follow         With logs: follow the log.
  -h, --help           Show this help.

Environment (all optional; non-interactive, nothing is prompted):
  DGXMM_PORT, DGXMM_BIND, DGXMM_NETWORK, DGXMM_HF_CACHE, DGXMM_MODELS_DIR   as above
  HF_TOKEN             Hugging Face token, written to secrets/hf_token (gated models).
  DGXMM_WITH_LITELLM=1 Same as --with-litellm. The bundled LiteLLM gets its own random master
                       key in secrets/litellm_bundled_master_key (sk-...), plus
                       secrets/litellm_salt_key and secrets/litellm_db_password (mode 600, never
                       printed, never overwritten); its database lives in data/litellm-db.
                       On a fresh config.yaml, litellm_url is http://dgxmm-litellm:4000 and, when
                       secrets/litellm_master_key is empty, DGXMM gets the same key (zero touch).
                       An existing config.yaml or key (e.g. for an external LiteLLM) is kept.
                       To turn it off later, set DGXMM_WITH_LITELLM=0 in .env and re-run
                       ./install.sh: it stops the bundled containers (data/litellm-db is kept).
  DGXMM_LITELLM_PORT   as --litellm-port
  LITELLM_URL          LiteLLM proxy URL for model registration (default http://litellm-proxy:4000,
                       or http://dgxmm-litellm:4000 with --with-litellm).
  LITELLM_MASTER_KEY   Master key DGXMM uses, written to secrets/litellm_master_key. If unset and a
                       running container named litellm-proxy exists, the key is read from it
                       (unless DGXMM uses the bundled LiteLLM; then a set LITELLM_MASTER_KEY
                       also seeds secrets/litellm_bundled_master_key).
  DGXMM_TRUSTED_PROXIES  Comma-separated IPs/CIDRs trusted to set X-Forwarded-For (default
                       127.0.0.1,::1). Behind a reverse proxy, set to the proxy's IP or docker
                       network subnet.
  DGXMM_PULL_ENGINES=1 Pre-pull the vLLM and llama.cpp engine images (large; default off).
  DGXMM_HEALTH_TIMEOUT Seconds to wait for the container to become healthy (default 300).
  DGXMM_LITELLM_HEALTH_TIMEOUT  Same for the bundled LiteLLM (default 300; a crash loop of
                       DGXMM_LITELLM_MAX_RESTARTS=3 restarts fails sooner).

Settings are written once to .env; when .env exists its values win (edit .env to change them).

Examples:
  ./install.sh                                    # install / upgrade, reachable on the LAN
  DGXMM_BIND=127.0.0.1 DGXMM_NETWORK=gateway ./install.sh   # behind an existing reverse proxy
  HF_TOKEN=hf_xxx ./install.sh --port 8091
  ./install.sh --with-litellm                     # fresh box: DGXMM + its own LiteLLM endpoint
  ./install.sh status
  ./install.sh uninstall            # keeps your data; add --purge to delete it
EOF
}

# --- output helpers -----------------------------------------------------------------------------

info() { printf '==> %s\n' "$*"; }
ok()   { printf '    ok   %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*" >&2; }
die()  { printf 'ERROR %s\n' "$*" >&2; exit 1; }

# run CMD...: execute, or only describe it in --dry-run mode.
run() {
  if ((DRY_RUN)); then
    printf '    [dry-run] would run:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# run_q CMD...: like run, but discards the command's stdout (ids/names docker echoes back).
run_q() {
  if ((DRY_RUN)); then
    run "$@"
  else
    "$@" >/dev/null
  fi
}

# write_file PATH MODE CONTENT: create a non-secret file (content shown in --dry-run).
write_file() {
  local path=$1 mode=$2 content=$3
  if ((DRY_RUN)); then
    note "[dry-run] would write ${path#"$REPO_DIR"/} (mode $mode):"
    printf '%s\n' "$content" | sed 's/^/        | /'
    return
  fi
  printf '%s\n' "$content" > "$path"
  chmod "$mode" "$path"
  ok "wrote ${path#"$REPO_DIR"/}"
}

# write_secret PATH VALUE SOURCE: create a 600 secret file; the value is never printed.
write_secret() {
  local path=$1 value=$2 source=$3
  if ((DRY_RUN)); then
    note "[dry-run] would write ${path#"$REPO_DIR"/} (mode 600, $source)"
    return
  fi
  (umask 077 && printf '%s' "$value" > "$path")
  chmod 600 "$path"
  ok "wrote ${path#"$REPO_DIR"/} ($source)"
}

# --- argument parsing -----------------------------------------------------------------------------

CMD=install
DRY_RUN=0
PURGE=0
ALLOW_ROOT=0
FOLLOW=0
FLAG_PORT="" FLAG_BIND="" FLAG_NETWORK="" FLAG_HF="" FLAG_MODELS="" FLAG_WITH_LITELLM="" FLAG_LITELLM_PORT=""
PASSTHRU=()   # options re-applied when upgrade re-executes the updated installer

parse_args() {
  local cmd_seen=0
  while (($#)); do
    if [[ $1 == --*=* ]]; then
      set -- "${1%%=*}" "${1#*=}" "${@:2}"
    fi
    case $1 in
      install|upgrade|status|logs|uninstall)
        ((cmd_seen)) && { printf 'more than one command given: %s\n' "$1" >&2; exit 2; }
        CMD=$1; cmd_seen=1 ;;
      -n|--dry-run) DRY_RUN=1; PASSTHRU+=("$1") ;;
      --purge) PURGE=1 ;;
      --allow-root) ALLOW_ROOT=1; PASSTHRU+=("$1") ;;
      -f|--follow) FOLLOW=1 ;;
      --with-litellm) FLAG_WITH_LITELLM=1; PASSTHRU+=("$1") ;;
      --port|--bind|--network|--hf-cache|--models-dir|--litellm-port)
        (($# >= 2)) || { printf 'option %s needs a value\n' "$1" >&2; exit 2; }
        case $1 in
          --port) FLAG_PORT=$2 ;;
          --bind) FLAG_BIND=$2 ;;
          --network) FLAG_NETWORK=$2 ;;
          --hf-cache) FLAG_HF=$2 ;;
          --models-dir) FLAG_MODELS=$2 ;;
          --litellm-port) FLAG_LITELLM_PORT=$2 ;;
        esac
        PASSTHRU+=("$1" "$2"); shift ;;
      -h|--help) usage; exit 0 ;;
      *) printf 'unknown option or command: %s (see ./install.sh --help)\n' "$1" >&2; exit 2 ;;
    esac
    shift
  done
  if ((PURGE)) && [ "$CMD" != uninstall ]; then
    printf '%s\n' "--purge only applies to uninstall" >&2; exit 2
  fi
}

# --- settings ---------------------------------------------------------------------------------------

# env_file_get KEY: print KEY's value from .env (exit 1 when absent).
env_file_get() {
  [ -f "$ENV_FILE" ] || return 1
  local line value
  line=$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -n 1) || return 1
  value=${line#*=}
  value=${value%$'\r'}
  value=${value#\"}; value=${value%\"}
  value=${value#\'}; value=${value%\'}
  printf '%s' "$value"
}

# resolve KEY FLAG_VALUE DEFAULT: .env value if present (warning when a different value was
# requested), else flag, else environment, else default.
resolve() {
  local key=$1 flag=$2 default=$3 requested="" current
  if [ -n "$flag" ]; then
    requested=$flag
  elif [ -n "${!key:-}" ]; then
    requested=${!key}
  fi
  if current=$(env_file_get "$key"); then
    if [ -n "$requested" ] && [ "$requested" != "$current" ]; then
      warn "$key=$requested was requested but .env has $key=$current; keeping .env (edit .env to change it)"
    fi
    printf '%s' "$current"
  else
    printf '%s' "${requested:-$default}"
  fi
}

# bool01 VALUE: print 1 or 0 for a truthy/falsy setting (exit 1 when it is neither).
bool01() {
  case ${1,,} in
    1|true|yes|on) printf 1 ;;
    0|false|no|off|"") printf 0 ;;
    *) return 1 ;;
  esac
}

# strip_quotes VALUE: VALUE without one pair of surrounding single or double quotes.
strip_quotes() {
  local v=$1
  v=${v#\"}; v=${v%\"}; v=${v#\'}; v=${v%\'}
  printf '%s' "$v"
}

expand_tilde() {
  local p=$1
  # shellcheck disable=SC2088  # matching a literal leading "~" on purpose
  case $p in
    "~") p=$HOME ;;
    "~/"*) p=$HOME/${p#"~/"} ;;
  esac
  printf '%s' "${p%/}"
}

check_path() {  # check_path NAME VALUE
  case $2 in
    /*) ;;
    *) die "$1 must be an absolute path (got '$2')" ;;
  esac
  case $2 in
    *[[:space:]:\"\'\\\$\`\#]*) die "$1 must not contain spaces or any of : \" ' \\ \$ \` # (got '$2')" ;;
  esac
}

load_settings() {
  local def_uid def_gid def_dgid
  def_uid=$(id -u)
  def_gid=$(id -g)
  if ! def_dgid=$(stat -c %g "$DOCKER_SOCK" 2>/dev/null); then
    die "cannot stat $DOCKER_SOCK: DGXMM needs the local Docker socket (rootless Docker / remote DOCKER_HOST is not supported)"
  fi
  HOST_HOME=$(expand_tilde "$(resolve DGXMM_HOST_HOME "" "$HOME")")
  DGXMM_UID=$(resolve DGXMM_UID "" "$def_uid")
  DGXMM_GID=$(resolve DGXMM_GID "" "$def_gid")
  DOCKER_GID=$(resolve DOCKER_GID "" "$def_dgid")
  PORT=$(resolve DGXMM_PORT "$FLAG_PORT" 8090)
  BIND=$(resolve DGXMM_BIND "$FLAG_BIND" 0.0.0.0)
  NETWORK=$(resolve DGXMM_NETWORK "$FLAG_NETWORK" dgxmm)
  HF_CACHE=$(expand_tilde "$(resolve DGXMM_HF_CACHE "$FLAG_HF" "$HOST_HOME/.cache/huggingface")")
  MODELS_DIR=$(expand_tilde "$(resolve DGXMM_MODELS_DIR "$FLAG_MODELS" "$HOST_HOME/models")")
  TRUSTED_PROXIES=$(resolve DGXMM_TRUSTED_PROXIES "" "")
  WITH_LITELLM=$(resolve DGXMM_WITH_LITELLM "$FLAG_WITH_LITELLM" 0)
  WITH_LITELLM=$(bool01 "$WITH_LITELLM") || die "DGXMM_WITH_LITELLM must be 1 or 0 (got '$WITH_LITELLM')"
  LITELLM_PORT=$(resolve DGXMM_LITELLM_PORT "$FLAG_LITELLM_PORT" 4000)
  # COMPOSE_PROFILES: the .env value, else "litellm"; a stray shell value is never adopted.
  COMPOSE_PROFILES_VALUE=$(env_file_get COMPOSE_PROFILES || printf litellm)

  if ! [[ $PORT =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    die "DGXMM_PORT must be a port number 1-65535 (got '$PORT')"
  fi
  if ((WITH_LITELLM)); then
    if ! [[ $LITELLM_PORT =~ ^[0-9]+$ ]] || ((LITELLM_PORT < 1 || LITELLM_PORT > 65535)); then
      die "DGXMM_LITELLM_PORT must be a port number 1-65535 (got '$LITELLM_PORT')"
    fi
    ((LITELLM_PORT != PORT)) || die "DGXMM_LITELLM_PORT and DGXMM_PORT must differ (both $PORT)"
    ENV_KEYS+=("${LITELLM_ENV_KEYS[@]}")
  fi
  if ! [[ ${DGXMM_HEALTH_TIMEOUT:-300} =~ ^[0-9]+$ ]] || ((${DGXMM_HEALTH_TIMEOUT:-300} < 1)); then
    die "DGXMM_HEALTH_TIMEOUT must be a positive whole number of seconds (got '${DGXMM_HEALTH_TIMEOUT}')"
  fi
  [[ $BIND =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die "DGXMM_BIND must be an IPv4 address such as 0.0.0.0 or 127.0.0.1 (got '$BIND')"
  [[ $NETWORK =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "DGXMM_NETWORK is not a valid docker network name (got '$NETWORK')"
  local n
  for n in DGXMM_UID DGXMM_GID DOCKER_GID; do
    [[ ${!n} =~ ^[0-9]+$ ]] || die "$n must be numeric (got '${!n}')"
  done
  check_path DGXMM_HOST_HOME "$HOST_HOME"
  check_path DGXMM_HF_CACHE "$HF_CACHE"
  check_path DGXMM_MODELS_DIR "$MODELS_DIR"
  if [ "$DGXMM_UID" != "$def_uid" ]; then
    warn ".env has DGXMM_UID=$DGXMM_UID but you are uid $def_uid; files DGXMM writes will belong to uid $DGXMM_UID"
  fi
  if [ "$DOCKER_GID" != "$def_dgid" ]; then
    warn ".env has DOCKER_GID=$DOCKER_GID but $DOCKER_SOCK belongs to gid $def_dgid; DGXMM may not reach Docker (fix DOCKER_GID in .env)"
  fi
}

env_value() {  # env_value KEY -> effective value for .env
  case $1 in
    DGXMM_UID) printf '%s' "$DGXMM_UID" ;;
    DGXMM_GID) printf '%s' "$DGXMM_GID" ;;
    DOCKER_GID) printf '%s' "$DOCKER_GID" ;;
    DGXMM_PORT) printf '%s' "$PORT" ;;
    DGXMM_BIND) printf '%s' "$BIND" ;;
    DGXMM_HF_CACHE) printf '%s' "$HF_CACHE" ;;
    DGXMM_MODELS_DIR) printf '%s' "$MODELS_DIR" ;;
    DGXMM_NETWORK) printf '%s' "$NETWORK" ;;
    DGXMM_HOST_HOME) printf '%s' "$HOST_HOME" ;;
    DGXMM_TRUSTED_PROXIES) printf '%s' "$TRUSTED_PROXIES" ;;
    DGXMM_WITH_LITELLM) printf '%s' "$WITH_LITELLM" ;;
    DGXMM_LITELLM_PORT) printf '%s' "$LITELLM_PORT" ;;
    COMPOSE_PROFILES) printf '%s' "$COMPOSE_PROFILES_VALUE" ;;
  esac
}

export_settings() {
  # .env is authoritative: export the resolved values so stray shell variables cannot override
  # it during compose interpolation.
  local k
  for k in "${ENV_KEYS[@]}"; do
    export "$k=$(env_value "$k")"
  done
}

# --- docker helpers ------------------------------------------------------------------------------------

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed (or not on PATH). Install Docker Engine: https://docs.docker.com/engine/install/"
  local out
  if ! out=$(docker info --format '{{json .Runtimes}}' 2>&1); then
    if grep -qi 'permission denied' <<<"$out"; then
      die "cannot reach the Docker daemon: permission denied on $DOCKER_SOCK. Add your user to the docker group: sudo usermod -aG docker $(id -un), then log out and back in (or run: newgrp docker)"
    fi
    die "cannot reach the Docker daemon ($out). Is it running? Try: sudo systemctl start docker"
  fi
  DOCKER_RUNTIMES=$out
}

# container_info: "STATE|WORKDIR" of the dgxmm container (exit 1 if there is none).
container_info() {
  docker inspect --format '{{.State.Status}}|{{index .Config.Labels "com.docker.compose.project.working_dir"}}' "$CONTAINER" 2>/dev/null
}

container_health() {
  docker inspect --format '{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$CONTAINER" 2>/dev/null || printf 'missing/'
}

# check_ownership: refuse to touch a dgxmm container created from another checkout (or by hand).
OWN_STATE=""
check_ownership() {
  local info state workdir
  info=$(container_info) || { OWN_STATE=""; return 0; }
  state=${info%%|*}
  workdir=${info#*|}
  if [ "$workdir" != "$REPO_DIR" ]; then
    die "another DGXMM instance exists on this host: container '$CONTAINER' ($state) was created from ${workdir:-outside docker compose}, not $REPO_DIR. Two instances would both adopt the same engine containers. Manage that installation from its own checkout (cd ${workdir:-<its checkout>} && ./install.sh ...), or remove it first."
  fi
  OWN_STATE=$state
}

# litellm_ids [SERVICE...]: ids of this checkout's bundled LiteLLM containers (compose labels; they
# have no fixed container_name, so their names derive from the compose project).
litellm_ids() {
  local svc
  for svc in "${@:-$LITELLM_SERVICE}"; do
    docker ps -aq --filter "label=com.docker.compose.project.working_dir=$REPO_DIR" \
      --filter "label=com.docker.compose.service=$svc" 2>/dev/null || true
  done
}

# litellm_health: STATE/HEALTH of this checkout's dgxmm-litellm container ("missing/" if none).
litellm_health() {
  local id
  id=$(litellm_ids "$LITELLM_SERVICE" | head -n 1)
  [ -n "$id" ] || { printf 'missing/'; return 0; }
  docker inspect --format '{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$id" 2>/dev/null ||
    printf 'missing/'
}

external_litellm_running() {
  [ -n "$(docker ps -q --filter 'name=^litellm-proxy$' 2>/dev/null)" ]
}

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    # Capture first, then match in one awk: a pipeline into grep -q can die of SIGPIPE under
    # pipefail and make a busy port look free.
    local listeners
    listeners=$(ss -Hltn 2>/dev/null) || listeners=""
    awk -v port="$1" '{ n = split($4, a, ":"); if (a[n] == port) found = 1 } END { exit !found }' <<<"$listeners"
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
  fi
}

host_ipv4s() {
  if command -v ip >/dev/null 2>&1; then
    ip -4 -o addr show scope global 2>/dev/null |
      awk '$2 !~ /^(docker|br-|veth|virbr|cni|flannel|cali|vxlan|lxc|podman)/ { split($4, a, "/"); print a[1] }'
  elif command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -Ev '^(127\.|172\.1[7-9]\.|$)' || true
  fi
}

# --- preflight ----------------------------------------------------------------------------------------

check_not_root() {
  if [ "$(id -u)" = 0 ] && ((!ALLOW_ROOT)); then
    die "refusing to run as root: run ./install.sh as the user that should own DGXMM's files (that user needs Docker access: sudo usermod -aG docker <user>). Pass --allow-root to override."
  fi
}

preflight() {
  info "Preflight"
  local arch
  arch=$(uname -m)
  case $arch in
    aarch64|arm64|x86_64|amd64) ok "arch $arch" ;;
    *) warn "arch $arch is untested (supported: aarch64, x86_64)" ;;
  esac
  check_not_root
  require_docker
  ok "docker daemon reachable"
  docker compose version --short >/dev/null 2>&1 ||
    die "docker compose v2 is required ('docker compose version' failed). Install the Docker Compose plugin (package docker-compose-plugin): https://docs.docker.com/compose/install/linux/"
  ok "docker compose $(docker compose version --short 2>/dev/null)"
  grep -q '"nvidia"' <<<"$DOCKER_RUNTIMES" ||
    die "Docker has no nvidia runtime registered. Install the NVIDIA Container Toolkit (nvidia-container-toolkit), then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  ok "nvidia container runtime registered"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: install the NVIDIA driver so the GPU is visible"
  local gpus
  if ! gpus=$(nvidia-smi -L 2>&1) || ! grep -q '^GPU ' <<<"$gpus"; then
    die "nvidia-smi cannot see a GPU ($(head -n 1 <<<"$gpus")). Check the NVIDIA driver installation"
  fi
  ok "$(grep '^GPU ' <<<"$gpus" | head -n 1 | sed 's/ (UUID.*//')"
  check_ownership
  if [ "$OWN_STATE" = running ]; then
    ok "port $PORT (in use by this DGXMM; it will be restarted)"
  elif port_busy "$PORT"; then
    die "port $PORT is already in use on this host by another process. Pick a free one, e.g. DGXMM_PORT=8091 ./install.sh (or --port 8091)"
  else
    ok "port $PORT free"
  fi
  if ((WITH_LITELLM)); then
    if external_litellm_running; then
      warn "a LiteLLM container named litellm-proxy is already running; --with-litellm starts a second, separate LiteLLM ($LITELLM_SERVICE) on port $LITELLM_PORT. DGXMM registers models only with the LiteLLM that litellm_url in config.yaml points at."
    fi
    local lstate
    lstate=$(litellm_health)
    if [[ $lstate == running/* ]]; then
      ok "port $LITELLM_PORT (in use by this DGXMM's LiteLLM; it will be updated)"
    elif port_busy "$LITELLM_PORT"; then
      die "port $LITELLM_PORT (LiteLLM) is already in use on this host by another process. Pick a free one, e.g. DGXMM_LITELLM_PORT=4001 ./install.sh --with-litellm (or --litellm-port 4001)"
    else
      ok "port $LITELLM_PORT free (LiteLLM)"
    fi
  fi
  local probe=$HF_CACHE avail_kb gib
  while [ ! -e "$probe" ] && [ "$probe" != / ]; do probe=$(dirname "$probe"); done
  avail_kb=$(df -Pk "$probe" 2>/dev/null | awk 'NR == 2 { print $4 }')
  if [[ $avail_kb =~ ^[0-9]+$ ]]; then
    gib=$((avail_kb / 1048576))
    if ((gib < MIN_FREE_GIB)); then
      warn "only $gib GiB free at $probe; models need a lot of space (>= $MIN_FREE_GIB GiB recommended)"
    else
      ok "disk: $gib GiB free at $probe"
    fi
  fi
}

# --- install steps -----------------------------------------------------------------------------------

ensure_env_file() {
  local k missing=()
  if [ -f "$ENV_FILE" ]; then
    note "kept existing .env"
    for k in "${ENV_KEYS[@]}"; do
      env_file_get "$k" >/dev/null || missing+=("$k")
    done
    if ((${#missing[@]})); then
      local block
      block="# added by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      for k in "${missing[@]}"; do block+=$'\n'"$k=$(env_value "$k")"; done
      if ((DRY_RUN)); then
        note "[dry-run] would append missing keys to .env:"
        printf '%s\n' "$block" | sed 's/^/        | /'
      else
        printf '%s\n' "$block" >> "$ENV_FILE"
        ok "added missing keys to .env: ${missing[*]}"
      fi
    fi
    return
  fi
  local content
  content="# DGXMM docker compose settings, generated by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
# install.sh never overwrites this file. Edit it, then re-run ./install.sh to apply."
  for k in "${ENV_KEYS[@]}"; do content+=$'\n'"$k=$(env_value "$k")"; done
  write_file "$ENV_FILE" 644 "$content"
}

# effective_litellm_url: litellm_url for a freshly rendered config.yaml ("" = keep the example's).
effective_litellm_url() {
  if [ -n "${LITELLM_URL:-}" ]; then
    printf '%s' "$LITELLM_URL"
  elif ((WITH_LITELLM)); then
    printf '%s' "$LITELLM_INTERNAL_URL"
  fi
}

render_config() {
  R_HF=$HF_CACHE R_MODELS=$MODELS_DIR R_HOME=$HOST_HOME R_NET=$NETWORK R_LITELLM=$(effective_litellm_url) \
  awk '
    function repl(s, old, new,   i) {
      i = index(s, old)
      if (i == 0) return s
      return substr(s, 1, i - 1) new substr(s, i + length(old))
    }
    /host: ~\/\.cache\/huggingface( |$)/ { $0 = repl($0, "~/.cache/huggingface", ENVIRON["R_HF"]) }
    /host: ~\/models( |$)/               { $0 = repl($0, "~/models", ENVIRON["R_MODELS"]) }
    /host: ~\//                          { $0 = repl($0, "~/", ENVIRON["R_HOME"] "/") }
    /^docker_network: dgxmm( |$)/        { $0 = repl($0, "docker_network: dgxmm", "docker_network: " ENVIRON["R_NET"]) }
    /^litellm_url:/ && ENVIRON["R_LITELLM"] != "" { $0 = "litellm_url: " ENVIRON["R_LITELLM"] }
    { print }
  ' "$EXAMPLE_CONFIG"
}

# config_litellm_url < CONFIG: the litellm_url DGXMM will use (quotes stripped; DGXMM's default when unset).
CONFIG_LITELLM_URL=""
config_litellm_url() {
  local url
  url=$(awk '/^litellm_url:/ { print $2; exit }')
  url=$(strip_quotes "$url")
  printf '%s' "${url:-http://litellm-proxy:4000}"
}

ensure_config() {
  if [ -f "$CONFIG_FILE" ]; then
    note "kept existing config.yaml"
    local cfg_net
    cfg_net=$(awk '/^docker_network:/ { print $2; exit }' "$CONFIG_FILE")
    if [ -n "$cfg_net" ] && [ "$cfg_net" != "$NETWORK" ]; then
      warn "config.yaml has docker_network: $cfg_net but DGXMM_NETWORK is $NETWORK; engines would join a different network than DGXMM. Make them match."
    fi
    CONFIG_LITELLM_URL=$(config_litellm_url <"$CONFIG_FILE")
    if ((WITH_LITELLM)) && [ "$CONFIG_LITELLM_URL" != "$LITELLM_INTERNAL_URL" ]; then
      note "note: config.yaml keeps litellm_url: $CONFIG_LITELLM_URL, so DGXMM keeps registering models there, not with the bundled LiteLLM. To switch DGXMM to the bundled LiteLLM, set litellm_url to $LITELLM_INTERNAL_URL and the master key from secrets/litellm_bundled_master_key on DGXMM's LiteLLM page (Integrations > LiteLLM)."
    fi
    return
  fi
  [ -f "$EXAMPLE_CONFIG" ] || die "config.example.yaml is missing; is this a complete checkout?"
  local body
  body=$(render_config)
  CONFIG_LITELLM_URL=$(config_litellm_url <<<"$body")
  if ! grep -qF "docker_network: $NETWORK" <<<"$body" || ! grep -qF "host: $HF_CACHE" <<<"$body" ||
     grep -q 'host: ~' <<<"$body"; then
    die "could not render config.yaml from config.example.yaml (unexpected layout); copy it by hand"
  fi
  if ((DRY_RUN)); then
    note "[dry-run] would write config.yaml (mode 644) from config.example.yaml with:"
    grep -E '^[[:space:]]*(- )?host:|^docker_network:|^litellm_url:' <<<"$body" | sed 's/^/        | /'
    return
  fi
  write_file "$CONFIG_FILE" 644 "# Generated by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) from config.example.yaml.
# install.sh never overwrites this file; edit it and run ./install.sh (or docker compose up -d).
$body"
}

ensure_dir() {  # ensure_dir PATH [MODE]
  if [ -d "$1" ]; then
    if [ -n "${2:-}" ] && ((!DRY_RUN)); then chmod "$2" "$1"; fi
    return
  fi
  if [ -n "${2:-}" ]; then run mkdir -p -m "$2" "$1"; else run mkdir -p "$1"; fi
  if ((!DRY_RUN)); then ok "created $1"; fi
}

detect_litellm_key() {
  [ -n "$(docker ps -q --filter 'name=^litellm-proxy$' 2>/dev/null)" ] || return 1
  local raw
  # Same extraction as the README; the value is only ever written to the secret file.
  # shellcheck disable=SC2016  # single quotes are intentional: evaluated inside the container
  raw=$(docker exec litellm-proxy sh -c 'grep -E "^\s*master_key:" /app/config.yaml | head -n 1 | sed -E "s/.*master_key:\s*//"' 2>/dev/null) || return 1
  raw=${raw%$'\r'}
  raw=${raw%%[[:space:]]#*}
  raw=${raw#\"}; raw=${raw%\"}; raw=${raw#\'}; raw=${raw%\'}
  if [[ $raw == os.environ/* ]]; then
    raw=$(docker exec litellm-proxy printenv "${raw#os.environ/}" 2>/dev/null) || return 1
  fi
  [ -n "$raw" ] || return 1
  printf '%s' "$raw"
}

# random_token LENGTH: LENGTH random [A-Za-z0-9] characters (no pipe is closed early: safe under
# pipefail).
random_token() {
  local raw
  raw=$(head -c 4096 /dev/urandom | LC_ALL=C tr -dc 'A-Za-z0-9')
  ((${#raw} >= $1)) || die "could not read enough random bytes from /dev/urandom"
  printf '%s' "${raw:0:$1}"
}

# ensure_secret NAME VALUE SOURCE: create secrets/NAME (or fill it when it exists but is empty).
ensure_secret() {
  local path=$SECRETS_DIR/$1 value=$2 source=$3
  if [ -s "$path" ]; then
    note "kept existing secrets/$1"
  elif [ -n "$value" ]; then
    write_secret "$path" "$value" "$source"
  elif [ -e "$path" ]; then
    note "kept existing secrets/$1 (empty)"
  else
    write_secret "$path" "" "empty placeholder"
  fi
  if [ -e "$path" ] && ((!DRY_RUN)); then chmod 600 "$path"; fi
}

# ensure_bundled_secrets: the bundled LiteLLM's own master key, salt key and DB password. Existing
# non-empty files are never replaced. Sets BUNDLED_KEY to the bundled master key.
BUNDLED_KEY=""
ensure_bundled_secrets() {
  local path=$SECRETS_DIR/litellm_bundled_master_key source="generated"
  if [ -s "$path" ]; then
    BUNDLED_KEY=$(<"$path")
    ensure_secret litellm_bundled_master_key "" ""
  else
    if [ "$CONFIG_LITELLM_URL" = "$LITELLM_INTERNAL_URL" ] && [ -n "${LITELLM_MASTER_KEY:-}" ]; then
      BUNDLED_KEY=$LITELLM_MASTER_KEY; source="from LITELLM_MASTER_KEY"
    elif [ "$CONFIG_LITELLM_URL" = "$LITELLM_INTERNAL_URL" ] && [ -s "$SECRETS_DIR/litellm_master_key" ]; then
      # DGXMM is already configured for the bundled LiteLLM with this key.
      BUNDLED_KEY=$(<"$SECRETS_DIR/litellm_master_key"); source="same key as secrets/litellm_master_key"
    else
      BUNDLED_KEY="sk-$(random_token 48)"
    fi
    ensure_secret litellm_bundled_master_key "$BUNDLED_KEY" "$source"
  fi
  local name
  for name in litellm_salt_key litellm_db_password; do
    if [ -s "$SECRETS_DIR/$name" ]; then
      ensure_secret "$name" "" ""
    elif [ "$name" = litellm_salt_key ]; then
      ensure_secret "$name" "sk-$(random_token 48)" "generated"
    else
      ensure_secret "$name" "$(random_token 40)" "generated"
    fi
  done
}

ensure_secrets() {
  ensure_dir "$SECRETS_DIR" 700
  ensure_secret hf_token "${HF_TOKEN:-}" "from HF_TOKEN"
  local key="${LITELLM_MASTER_KEY:-}" source="from LITELLM_MASTER_KEY"
  if ((WITH_LITELLM)); then
    ensure_bundled_secrets
    if [ "$CONFIG_LITELLM_URL" = "$LITELLM_INTERNAL_URL" ]; then
      # DGXMM talks to the bundled LiteLLM: give it the same key (zero touch), never replacing one.
      if [ ! -s "$SECRETS_DIR/litellm_master_key" ]; then
        ensure_secret litellm_master_key "$BUNDLED_KEY" "same key as secrets/litellm_bundled_master_key"
      else
        ensure_secret litellm_master_key "" ""
        if ((!DRY_RUN)) && [ "$(<"$SECRETS_DIR/litellm_master_key")" != "$BUNDLED_KEY" ]; then
          warn "secrets/litellm_master_key differs from secrets/litellm_bundled_master_key, but config.yaml points DGXMM at the bundled LiteLLM: registration will fail to authenticate unless the bundled key is set on DGXMM's LiteLLM page (Integrations > LiteLLM) or copied into secrets/litellm_master_key"
        fi
      fi
      return 0
    fi
    # DGXMM keeps using another LiteLLM: its key (secrets/litellm_master_key) is handled as usual.
  fi
  if [ -z "$key" ] && [ ! -s "$SECRETS_DIR/litellm_master_key" ]; then
    if key=$(detect_litellm_key); then
      source="read from the running litellm-proxy container"
    else
      key=""
    fi
  fi
  ensure_secret litellm_master_key "$key" "$source"
  if [ ! -s "$SECRETS_DIR/litellm_master_key" ] && [ -z "$key" ]; then
    note "LiteLLM registration stays off until secrets/litellm_master_key holds the master key"
  fi
  if [ -n "$(docker ps -q --filter 'name=^litellm-proxy$' 2>/dev/null)" ] &&
     ! docker inspect --format '{{json .NetworkSettings.Networks}}' litellm-proxy 2>/dev/null | grep -q "\"$NETWORK\""; then
    warn "container litellm-proxy is not on network $NETWORK, so DGXMM cannot reach it as http://litellm-proxy:4000. Either: docker network connect $NETWORK litellm-proxy, or set litellm_url in config.yaml"
  fi
}

ensure_network() {
  if docker network inspect "$NETWORK" >/dev/null 2>&1; then
    ok "docker network $NETWORK exists"
  else
    run_q docker network create --label dgxmm.created=true "$NETWORK"
    if ((!DRY_RUN)); then ok "created docker network $NETWORK"; fi
  fi
}

engine_images() {
  awk '/^images:/ { f = 1; next } f && /^[^ #]/ { f = 0 } f && /^[[:space:]]+(vllm|llamacpp):/ { print $2 }' "$1"
}

pull_engines() {
  [ "${DGXMM_PULL_ENGINES:-0}" = 1 ] || return 0
  local cfg=$CONFIG_FILE img
  [ -f "$cfg" ] || cfg=$EXAMPLE_CONFIG
  info "Pulling engine images (this can take a while)"
  while read -r img; do
    if [ -n "$img" ]; then
      run docker pull "$img" ||
        die "could not pull $img. Check network access to the registry, or point images.* in config.yaml at an image that exists locally (docker images)"
    fi
  done < <(engine_images "$cfg")
}

# capture_admin_password: on first run, save the password DGXMM generated for `admin` to
# secrets/admin_password (600). Returns 0 and sets ADMIN_PW only when a new one was captured.
ADMIN_PW=""
RUN_START=""
CREATED_THIS_RUN=0
container_id() {
  docker inspect --format '{{.Id}}' "$CONTAINER" 2>/dev/null || true
}

# Only a container (re)created by this run, logging after this run started, can hold a fresh
# first-run password; anything else would be stale.
capture_admin_password() {
  [ -e "$SECRETS_DIR/admin_password" ] && return 1
  if ((!CREATED_THIS_RUN)) || [ -z "$RUN_START" ]; then return 1; fi
  local pw
  pw=$(docker logs --since "$RUN_START" "$CONTAINER" 2>&1 |
       sed -n "s/.*DGXMM first run: created user 'admin' with password \([^ ]*\).*/\1/p" | tail -n 1)
  [ -n "$pw" ] || return 1
  (umask 077 && printf '%s\n' "$pw" > "$SECRETS_DIR/admin_password")
  chmod 600 "$SECRETS_DIR/admin_password"
  ADMIN_PW=$pw
}

fail_with_logs() {
  printf '\n' >&2
  if capture_admin_password; then
    warn "the admin password was saved to secrets/admin_password"
  fi
  warn "last container log lines:"
  docker logs --tail 40 "$CONTAINER" 2>&1 | sed -E 's/(with password )[^ ]+/\1<redacted>/' | sed 's/^/    /' >&2 || true
  die "$1. Inspect with: ./install.sh logs"
}

wait_healthy() {
  local timeout=${DGXMM_HEALTH_TIMEOUT:-300} poll=${DGXMM_HEALTH_POLL:-3} start=$SECONDS state
  info "Waiting for DGXMM to become healthy (up to ${timeout}s)"
  while :; do
    state=$(container_health)
    case $state in
      running/healthy)
        if [ -t 1 ]; then printf '\n'; fi
        ok "healthy after $((SECONDS - start))s"
        return 0 ;;
      running/unhealthy) fail_with_logs "the dgxmm container is unhealthy" ;;
      running/*|created/*|restarting/*) ;;
      *) fail_with_logs "the dgxmm container is not running (state: ${state%/})" ;;
    esac
    if ((SECONDS - start >= timeout)); then
      fail_with_logs "DGXMM did not become healthy within ${timeout}s (state: $state)"
    fi
    if [ -t 1 ]; then printf '.'; fi
    sleep "$poll"
  done
}

# print_endpoints PORT PATH LOOPBACK_NOTE: the URLs a published port is reachable at.
print_endpoints() {
  local port=$1 path=$2 loopback_note=$3 ip
  case $BIND in
    0.0.0.0)
      while read -r ip; do
        if [ -n "$ip" ]; then note "http://$ip:$port$path"; fi
      done < <(host_ipv4s)
      local host
      host=$(hostname 2>/dev/null || echo localhost)
      case $host in
        *.*|localhost) ;;
        *) note "http://$host.local:$port$path  (mDNS)" ;;
      esac
      note "http://$host:$port$path"
      ;;
    127.0.0.1)
      note "http://127.0.0.1:$port$path  ($loopback_note)"
      ;;
    *) note "http://$BIND:$port$path" ;;
  esac
}

print_urls() {
  print_endpoints "$PORT" /dgxmm/ "loopback only: publish it through a reverse proxy, see deploy/nginx-dgxmm.conf"
}

print_litellm_endpoint() {
  print_endpoints "$LITELLM_PORT" /v1 "loopback only: publish it through a reverse proxy"
  note "from containers on network $NETWORK: $LITELLM_INTERNAL_URL/v1"
  note "every ready DGXMM deployment is one model on this endpoint: pass \"model\": \"<model name>\""
  note "API keys: create virtual keys on DGXMM's LiteLLM page (Integrations > LiteLLM), or call"
  note "  POST /key/generate on the endpoint with the master key from secrets/litellm_bundled_master_key"
}

# redact_secrets: filter that masks secrets in log lines (key files' values, sk-... tokens, bearer
# tokens and URL userinfo such as postgresql://user:password@host).
redact_secrets() {
  local args=() f v
  for f in litellm_bundled_master_key litellm_master_key litellm_salt_key litellm_db_password; do
    if [ -s "$SECRETS_DIR/$f" ]; then
      v=$(<"$SECRETS_DIR/$f")
      if [[ $v =~ ^[A-Za-z0-9_-]+$ ]]; then args+=(-e "s/$v/***/g"); fi
    fi
  done
  sed -E "${args[@]}" -e 's#://[^@/[:space:]]*@#://***@#g' -e 's/(sk-)[A-Za-z0-9_-]+/\1***/g' \
    -e 's/([Bb]earer )[^[:space:]"]+/\1***/g'
}

# show_litellm_logs: the last lines of the bundled LiteLLM + database logs (redacted), on stderr.
show_litellm_logs() {
  warn "last LiteLLM / database log lines:"
  docker compose --profile litellm logs --tail 30 "$LITELLM_DB_SERVICE" "$LITELLM_SERVICE" 2>&1 |
    redact_secrets | sed 's/^/    /' >&2 || true
}

litellm_restarts() {
  local id n
  id=$(litellm_ids "$LITELLM_SERVICE" | head -n 1)
  [ -n "$id" ] || { printf 0; return 0; }
  n=$(docker inspect --format '{{.RestartCount}}' "$id" 2>/dev/null) || n=0
  [[ $n =~ ^[0-9]+$ ]] || n=0
  printf '%s' "$n"
}

wait_litellm_healthy() {
  local timeout=${DGXMM_LITELLM_HEALTH_TIMEOUT:-300} poll=${DGXMM_HEALTH_POLL:-3} start=$SECONDS state fail
  local max_restarts=${DGXMM_LITELLM_MAX_RESTARTS:-3} restarts
  info "Waiting for LiteLLM to become healthy (/health/liveliness, up to ${timeout}s)"
  while :; do
    state=$(litellm_health)
    case $state in
      running/healthy)
        if [ -t 1 ]; then printf '\n'; fi
        ok "LiteLLM healthy after $((SECONDS - start))s"
        return 0 ;;
      running/unhealthy) fail="the $LITELLM_SERVICE container is unhealthy" ;;
      running/*|created/*|restarting/*) fail="" ;;
      *) fail="the $LITELLM_SERVICE container is not running (state: ${state%/})" ;;
    esac
    if [ -z "$fail" ]; then
      restarts=$(litellm_restarts)
      if ((restarts >= max_restarts)); then
        fail="the $LITELLM_SERVICE container restarted $restarts times (crash loop; state: ${state%/})"
      elif ((SECONDS - start >= timeout)); then
        fail="LiteLLM did not become healthy within ${timeout}s (state: $state)"
      fi
    fi
    if [ -n "$fail" ]; then
      warn "$fail"
      show_litellm_logs
      return 1
    fi
    if [ -t 1 ]; then printf '.'; fi
    sleep "$poll"
  done
}

proxy_hint() {
  # Loopback bind and no trusted proxies: a reverse proxy on this host reaches DGXMM from the docker
  # network gateway, so its X-Forwarded-For would be ignored until DGXMM_TRUSTED_PROXIES is set.
  case $BIND in 127.0.0.1|::1) ;; *) return 0 ;; esac
  [ -z "$TRUSTED_PROXIES" ] || return 0
  local gw
  gw=$(docker network inspect "$NETWORK" -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)
  printf '\n'
  info "Behind a reverse proxy? Set DGXMM_TRUSTED_PROXIES in .env to your proxy's address, then re-run ./install.sh"
  if [ -n "$gw" ]; then
    note "for a proxy on this host use the docker network gateway: DGXMM_TRUSTED_PROXIES=$gw"
  fi
}

# stop_disabled_litellm: DGXMM_WITH_LITELLM=0 but this checkout's bundled containers exist (the bundle
# was turned off in .env): stop and remove them (data/litellm-db and secrets are kept), and drop the
# litellm profile from COMPOSE_PROFILES so compose does not start them again.
stop_disabled_litellm() {
  local ids=()
  mapfile -t ids < <(litellm_ids "$LITELLM_SERVICE" "$LITELLM_DB_SERVICE" | grep . || true)
  if ((${#ids[@]})); then
    info "Stopping the bundled LiteLLM (DGXMM_WITH_LITELLM=0)"
    run_q docker stop -t 30 "${ids[@]}"
    run_q docker rm "${ids[@]}"
    note "kept data/litellm-db and its secrets; set DGXMM_WITH_LITELLM=1 in .env (or --with-litellm) to start it again"
  fi
  local profiles p kept=""
  profiles=$(env_file_get COMPOSE_PROFILES || printf '%s' "${COMPOSE_PROFILES:-}")
  case ",$profiles," in
    *,litellm,*)
      local list=()
      IFS=, read -r -a list <<<"$profiles"
      for p in "${list[@]}"; do
        if [ "$p" != litellm ]; then kept+=${kept:+,}$p; fi
      done
      export COMPOSE_PROFILES="$kept" ;;
  esac
}

cmd_install() {
  load_settings
  preflight
  info "Settings"
  note "port $PORT on $BIND, network $NETWORK, uid:gid $DGXMM_UID:$DGXMM_GID, docker gid $DOCKER_GID"
  note "HF cache $HF_CACHE, models $MODELS_DIR"

  info "Configuration"
  ensure_env_file
  ensure_config
  ensure_dir "$DATA_DIR"
  ensure_secrets
  if ((WITH_LITELLM)) && [ ! -f "$REPO_DIR/deploy/litellm/config.yaml" ]; then
    die "deploy/litellm/config.yaml is missing (needed by --with-litellm); is this a complete checkout?"
  fi

  info "Host directories"
  local d
  for d in "$HF_CACHE" "$MODELS_DIR" "$HOST_HOME/.cache/vllm" "$HOST_HOME/.cache/flashinfer" "$HOST_HOME/.triton"; do
    if [ -d "$d" ]; then ok "exists: $d"; else ensure_dir "$d"; fi
  done
  if ((WITH_LITELLM)); then
    # Created here (not by docker, which would make it root-owned): Postgres runs as DGXMM_UID.
    if [ -d "$LITELLM_DB_DIR" ]; then ok "exists: ${LITELLM_DB_DIR#"$REPO_DIR"/}"; else ensure_dir "$LITELLM_DB_DIR" 700; fi
  fi

  info "Docker"
  ensure_network
  pull_engines
  export_settings
  local compose=(docker compose)
  if ((WITH_LITELLM)); then
    compose+=(--profile litellm)
    case ",$COMPOSE_PROFILES_VALUE," in
      *,litellm,*) ;;
      *) warn ".env has COMPOSE_PROFILES=$COMPOSE_PROFILES_VALUE without litellm: a plain 'docker compose up -d' will not start the bundled LiteLLM (install.sh still does)" ;;
    esac
  fi
  if ((!WITH_LITELLM)); then stop_disabled_litellm; fi
  info "Building and starting DGXMM (${compose[*]} up -d --build)"
  RUN_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local id_before litellm_up_failed=0
  id_before=$(container_id)
  if ! run "${compose[@]}" up -d --build; then
    # With the bundled LiteLLM, a database that never gets healthy fails `up` although dgxmm itself
    # started: finish DGXMM's setup (first-run admin password) first, then report the failure.
    if ((WITH_LITELLM)) && [[ $(container_health) == running/* ]]; then
      litellm_up_failed=1
      warn "docker compose up reported an error, but the dgxmm container is running: finishing DGXMM's setup first; the bundled LiteLLM is checked at the end"
    else
      die "docker compose up failed (image build or container start); see the output above, then run ./install.sh logs"
    fi
  fi
  if [ "$(container_id)" != "$id_before" ]; then CREATED_THIS_RUN=1; fi

  if ((DRY_RUN)); then
    note "[dry-run] would wait for the healthcheck, then save a first-run admin password to secrets/admin_password"
    info "Dry run complete: nothing was changed. DGXMM would be reachable at:"
    print_urls
    if ((WITH_LITELLM)); then
      info "The bundled LiteLLM endpoint would be:"
      print_litellm_endpoint
    fi
    proxy_hint
    return 0
  fi

  wait_healthy
  printf '\n'
  info "DGXMM is running. Open:"
  print_urls
  if capture_admin_password; then
    note "admin user: admin   password: $ADMIN_PW"
    note "(shown once; also saved to secrets/admin_password. You must change it at first login.)"
  elif [ -e "$SECRETS_DIR/admin_password" ]; then
    note "admin password: see secrets/admin_password (if unchanged), or reset it (see docs/operations.md)"
  else
    note "admin password: see secrets/admin_password, or reset it (see docs/operations.md)"
  fi
  note "manage: ./install.sh status | logs | upgrade | uninstall"
  proxy_hint
  if ((WITH_LITELLM)); then
    printf '\n'
    local litellm_hint="DGXMM itself is running. Inspect: docker compose --profile litellm logs --tail 100 $LITELLM_DB_SERVICE $LITELLM_SERVICE, fix the cause, then re-run ./install.sh"
    if ((litellm_up_failed)); then
      warn "the bundled LiteLLM stack failed to start (docker compose up error above)"
      show_litellm_logs
      die "the bundled LiteLLM did not start; $litellm_hint"
    fi
    wait_litellm_healthy || die "the bundled LiteLLM did not start; $litellm_hint"
    info "LiteLLM is running. Single OpenAI-compatible endpoint for all DGXMM models:"
    print_litellm_endpoint
  fi
}

cmd_upgrade() {
  if command -v git >/dev/null 2>&1 && git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    info "Updating the checkout (git pull --ff-only)"
    if ((DRY_RUN)); then
      run git -C "$REPO_DIR" pull --ff-only
    else
      git -C "$REPO_DIR" pull --ff-only ||
        die "git pull --ff-only failed (local changes or a diverged branch?). Resolve it, then re-run ./install.sh upgrade"
      # Run the freshly pulled installer.
      exec bash "$REPO_DIR/install.sh" install "${PASSTHRU[@]}"
    fi
  else
    warn "not a git checkout (or git missing); skipping git pull"
  fi
  cmd_install
}

cmd_status() {
  require_docker
  local info state workdir health rc=0
  if ! info=$(container_info); then
    note "DGXMM is not installed on this host (no '$CONTAINER' container). Run ./install.sh"
    return 1
  fi
  state=${info%%|*}
  workdir=${info#*|}
  health=$(container_health)
  health=${health#*/}
  info "Container $CONTAINER: $state${health:+ ($health)}"
  note "checkout: ${workdir:-unknown}"
  [ "$workdir" = "$REPO_DIR" ] || warn "this container was not created from $REPO_DIR"
  [ "$state/$health" = running/healthy ] || rc=1
  local f
  for f in .env config.yaml; do
    if [ -f "$REPO_DIR/$f" ]; then note "$f: present"; else note "$f: missing"; fi
  done
  local with_litellm
  with_litellm=$(bool01 "$(env_file_get DGXMM_WITH_LITELLM || echo 0)") || with_litellm=0
  local secret_names=(hf_token litellm_master_key admin_password)
  if [ "$with_litellm" = 1 ]; then secret_names+=(litellm_bundled_master_key litellm_salt_key litellm_db_password); fi
  for f in "${secret_names[@]}"; do
    if [ -s "$SECRETS_DIR/$f" ]; then note "secrets/$f: set"; else note "secrets/$f: not set"; fi
  done
  local lids=() lid line
  mapfile -t lids < <(litellm_ids "$LITELLM_SERVICE" "$LITELLM_DB_SERVICE")
  if [ "$with_litellm" = 1 ] || ((${#lids[@]})); then
    info "Bundled LiteLLM containers (profile litellm)"
    if ((${#lids[@]})); then
      for lid in "${lids[@]}"; do
        [ -n "$lid" ] || continue
        line=$(docker inspect --format '{{.Name}}  {{.State.Status}}{{if .State.Health}} ({{.State.Health.Status}}){{end}}' "$lid" 2>/dev/null) || continue
        note "${line#/}"
      done
    else
      note "none (DGXMM_WITH_LITELLM=1 in .env: run ./install.sh to start them)"
    fi
    if [ "$with_litellm" = 1 ] && [ "$(litellm_health)" != running/healthy ]; then rc=1; fi
  fi
  info "Managed engine containers (dgxmm.managed=true)"
  local engines
  engines=$(docker ps --filter label=dgxmm.managed=true --format '{{.Names}}  {{.Status}}')
  if [ -n "$engines" ]; then
    local line
    while IFS= read -r line; do note "$line"; done <<<"$engines"
  else
    note "none running"
  fi
  PORT=$(env_file_get DGXMM_PORT || echo 8090)
  BIND=$(env_file_get DGXMM_BIND || echo 0.0.0.0)
  info "URLs"
  print_urls
  if [ "$with_litellm" = 1 ]; then
    LITELLM_PORT=$(env_file_get DGXMM_LITELLM_PORT || echo 4000)
    NETWORK=$(env_file_get DGXMM_NETWORK || echo dgxmm)
    info "LiteLLM endpoint"
    print_litellm_endpoint
  fi
  return "$rc"
}

cmd_logs() {
  require_docker
  container_info >/dev/null || die "DGXMM is not installed on this host (no '$CONTAINER' container)"
  if ((FOLLOW)); then
    docker logs --tail "${DGXMM_LOG_LINES:-200}" -f "$CONTAINER"
  else
    docker logs --tail "${DGXMM_LOG_LINES:-200}" "$CONTAINER"
  fi
}

cmd_uninstall() {
  check_not_root
  require_docker
  check_ownership
  info "Removing the DGXMM container and image"
  if [ -n "$OWN_STATE" ]; then
    if [ "$OWN_STATE" = running ]; then run_q docker stop -t 30 "$CONTAINER"; fi
    run_q docker rm "$CONTAINER"
  else
    note "no $CONTAINER container"
  fi
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    run_q docker image rm "$IMAGE"
  fi
  local lids=()
  mapfile -t lids < <(litellm_ids "$LITELLM_SERVICE" "$LITELLM_DB_SERVICE" | grep . || true)
  if ((${#lids[@]})); then
    info "Removing the bundled LiteLLM containers"
    run_q docker stop -t 30 "${lids[@]}"
    run_q docker rm "${lids[@]}"
  fi
  local network
  network=$(env_file_get DGXMM_NETWORK || echo dgxmm)
  if ((!PURGE)); then
    local n
    n=$(docker ps -q --filter label=dgxmm.managed=true | grep -c . || true)
    note "kept data/, secrets/, config.yaml and .env (re-run ./install.sh to reinstall)"
    if [ -d "$LITELLM_DB_DIR" ]; then note "kept the LiteLLM database in data/litellm-db (--purge removes it)"; fi
    if ((n > 0)); then
      note "$n engine container(s) labelled dgxmm.managed=true are still running; a reinstall re-adopts them (--purge removes them)"
    fi
    if ((DRY_RUN)); then info "Dry run complete: nothing was changed."; fi
    return 0
  fi
  info "Purging DGXMM state"
  local ids=()
  mapfile -t ids < <(docker ps -aq --filter label=dgxmm.managed=true)
  if ((${#ids[@]})); then
    run_q docker rm -f "${ids[@]}"
  fi
  if [ "$(docker network inspect --format '{{index .Labels "dgxmm.created"}}' "$network" 2>/dev/null || true)" = true ]; then
    run_q docker network rm "$network" || warn "could not remove network $network (still in use?)"
  fi
  if [ -d "$LITELLM_DB_DIR" ]; then
    run rm -rf "$LITELLM_DB_DIR" || die "could not remove data/litellm-db (files owned by another user?); remove it by hand"
  fi
  run rm -rf "$DATA_DIR" "$SECRETS_DIR" "$CONFIG_FILE" "$ENV_FILE"
  note "model caches were not touched"
  if ((DRY_RUN)); then info "Dry run complete: nothing was changed."; fi
  return 0
}

main() {
  parse_args "$@"
  cd "$REPO_DIR"
  if ((DRY_RUN)); then info "Dry run: no changes will be made"; fi
  case $CMD in
    install) cmd_install ;;
    upgrade) cmd_upgrade ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    uninstall) cmd_uninstall ;;
  esac
}

main "$@"
