#!/usr/bin/env bash
# File: <project>\moe_track\lib\llama_launcher.sh      (installed as  ~/.local/bin/llama  by  moe_track/12_upgrade_llama_launcher.sh)
# llama launcher v2 for the stuart-creator server.  Same job as the launcher it replaces; adds --profile, --pelias-tier, --mtp, --print.
#
#   llama --serve --stuart-creator [--bg] [--profile long|fast] [--pelias-tier] [--mtp [--mtp-n N] [--mtp-gguf FILE]] [--print] [-- extra llama-server flags]
#   llama --stop
#   llama --status
#   llama --version | --help
#
#   --bg            run in the background (pid in ~/.stuart-creator.pid, log in the logs folder). Without it the server runs in this window and Ctrl-C stops it.
#   --profile long  (default) context 131072, ubatch 2048.        --profile fast   context 65536, ubatch 2048 (frees ~0.66 GiB of KV cache for more hot experts)
#   --pelias-tier   use the Pelias-Tier build of llama-server (~/llama-pelias-tier, made by patches/09_build_pelias_tier.sh) with the hot-set file
#                   (logs/pelias_hot_set.txt, made by sim/export_hot_set.py). Default expert placement becomes --n-cpu-moe 40 and --fit off, as in fork/gauntlet.py's tier family.
#   --mtp           speculative decoding with the model's own MTP block: --spec-type draft-mtp --spec-draft-n-max 2, and GGML_CUDA_DISABLE_FUSION=1
#                   (llama.cpp issue #29168). --mtp-n N changes the draft length. --mtp-gguf FILE uses a separate draft model (-md FILE), e.g. the one made by
#                   mtp_align/11_merge_blk40_and_export.py; it implies --mtp.
#   --print         show the environment and the exact command and stop; starts nothing, checks nothing on the network.
#   -- FLAGS...     everything after the double dash is passed to llama-server unchanged (last, so it wins over the defaults).
#
# What it refuses to do: start when the port is already taken, start a second background server while one is recorded in the pid file, start without the model file
# or the server binary, start --pelias-tier without its build or its hot-set file (it would silently run as a plain control).
#
# Environment overrides (all optional):
#   LLAMA_BIN        path of llama-server (wins over everything, including the tier build)      LLAMA_MODEL   model GGUF
#   LLAMA_CREATOR    base folder (default <project>)                              LLAMA_LOGDIR  log folder      LLAMA_PIDFILE  pid file
#   LLAMA_PORT       default 8012      LLAMA_THREADS  default 7      LLAMA_PRIO  default 2      LLAMA_UB  default 2048     LLAMA_CTX  overrides the profile's context
#   LLAMA_LOAD_MODE  default none  (empty string = do not pass --load-mode at all)
#   EXPERT_FLAGS     expert placement flags, default "--n-cpu-moe 28" ("--n-cpu-moe 40" with --pelias-tier)
#   TIER_SRC         default ~/llama-pelias-tier      HOT_SET  default <logs>/pelias_hot_set.txt      TIER_BYPASS_N  default 32      TIER_MAX_MIB  default 6144
LAUNCHER_VERSION="2.0 (moe_track, 2026-10-20)"

CREATOR_DIR="${LLAMA_CREATOR:-<project>}"
BIN_PROD="<llama.cpp>/build-wsl/bin/llama-server"
MODEL="${LLAMA_MODEL:-$CREATOR_DIR/gguf/test-model-Q4_K_M.gguf}"
LOGDIR="${LLAMA_LOGDIR:-$CREATOR_DIR/logs}"
PIDFILE="${LLAMA_PIDFILE:-$HOME/.stuart-creator.pid}"
PORT="${LLAMA_PORT:-8012}"
UB="${LLAMA_UB:-2048}"
THREADS="${LLAMA_THREADS:-7}"
PRIO="${LLAMA_PRIO:-2}"
LOAD_MODE="${LLAMA_LOAD_MODE-none}"
TIER_SRC="${TIER_SRC:-$HOME/llama-pelias-tier}"
HOT_SET="${HOT_SET:-$LOGDIR/pelias_hot_set.txt}"
TIER_BYPASS_N="${TIER_BYPASS_N:-32}"
TIER_MAX_MIB="${TIER_MAX_MIB:-6144}"

die() { printf 'llama: %s\n' "$*" >&2; exit 2; }
is_int() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | grep -q ":${PORT} "
    else
        (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null          # no ss: a successful connect means something is listening
    fi
}
usage() {
    echo "usage: llama --serve --stuart-creator [--bg] [--profile long|fast] [--pelias-tier] [--mtp [--mtp-n N] [--mtp-gguf FILE]] [--print] [-- extra_flags]"
    echo "       llama --stop | --status | --version | --help"
}

serve=0; target=""; bg=0; stop=0; status=0; use_tier=0; use_mtp=0; do_print=0
profile=long; mtp_n=2; mtp_gguf=""; extra=()

while [ $# -gt 0 ]; do
    case "$1" in
        --serve) serve=1 ;;
        --stuart-creator) target=sc ;;
        --bg) bg=1 ;;
        --stop) stop=1 ;;
        --status) status=1 ;;
        --print) do_print=1 ;;
        --pelias-tier) use_tier=1 ;;
        --mtp) use_mtp=1 ;;
        --profile)
            [ $# -ge 2 ] || die "--profile needs fast or long"
            case "$2" in fast|long) profile="$2" ;; *) die "unknown profile '$2' (fast or long)" ;; esac
            shift ;;
        --mtp-n)
            [ $# -ge 2 ] || die "--mtp-n needs a number"
            is_int "$2" && [ "$2" -ge 1 ] && [ "$2" -le 16 ] || die "--mtp-n must be a whole number from 1 to 16"
            mtp_n="$2"; use_mtp=1; shift ;;
        --mtp-gguf)
            [ $# -ge 2 ] || die "--mtp-gguf needs a file"
            mtp_gguf="$2"; use_mtp=1; shift ;;
        --version) echo "llama launcher $LAUNCHER_VERSION"; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) usage >&2; die "unknown option '$1'" ;;
    esac
    shift
done

if [ "$status" -eq 1 ]; then
    running="$(pgrep -a llama-server 2>/dev/null | cut -c1-160)"
    if [ -n "$running" ]; then echo "$running"; else echo "no llama-server running"; fi
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "health: ok (port ${PORT})"; else echo "health: no answer on port ${PORT}"; fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
    exit 0
fi

if [ "$stop" -eq 1 ]; then
    if [ -f "$PIDFILE" ] && pid="$(cat "$PIDFILE" 2>/dev/null)" && is_int "$pid" && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 1; waited=$((waited + 1)); done
        if kill -0 "$pid" 2>/dev/null; then
            echo "pid $pid is still shutting down after 60 s; the pid file is kept (llama --status to look)"
            exit 1
        fi
        rm -f "$PIDFILE"
        echo "stopped (pid $pid)"
    else
        [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
        echo "no background server recorded (for one running in a window, press Ctrl-C there)"
    fi
    exit 0
fi

if [ "$serve" -ne 1 ] || [ "$target" != sc ]; then
    usage >&2
    exit 2
fi

# ---- build the command --------------------------------------------------------------------------------------------------------------------------------------
if [ "$profile" = fast ]; then CTX=65536; else CTX=131072; fi
CTX="${LLAMA_CTX:-$CTX}"
is_int "$CTX" && is_int "$UB" && is_int "$THREADS" && is_int "$PORT" || die "LLAMA_CTX, LLAMA_UB, LLAMA_THREADS and LLAMA_PORT must be whole numbers"

BIN="$BIN_PROD"
envs=()
if [ "$use_tier" -eq 1 ]; then
    BIN="$TIER_SRC/build-wsl/bin/llama-server"
    envs+=("PELIAS_HOT_SET=$HOT_SET" "PELIAS_TIER_BYPASS_N=$TIER_BYPASS_N" "PELIAS_TIER_MAX_MIB=$TIER_MAX_MIB")
fi
[ -n "${LLAMA_BIN:-}" ] && BIN="$LLAMA_BIN"

mtp_args=()
if [ "$use_mtp" -eq 1 ]; then
    envs+=("GGML_CUDA_DISABLE_FUSION=1")
    mtp_args=(--spec-type draft-mtp --spec-draft-n-max "$mtp_n")
    [ -n "$mtp_gguf" ] && mtp_args+=(-md "$mtp_gguf")
fi

if [ -n "${EXPERT_FLAGS:-}" ]; then
    read -r -a EXP <<< "$EXPERT_FLAGS"
elif [ "$use_tier" -eq 1 ]; then
    EXP=(--n-cpu-moe 40)
else
    EXP=(--n-cpu-moe 28)
fi
[ "$use_tier" -eq 1 ] && EXP+=(--fit off)

lm_args=()
[ -n "$LOAD_MODE" ] && lm_args=(--load-mode "$LOAD_MODE")

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/server_Q4_K_M_${STAMP}.log"
CMD=("$BIN" -m "$MODEL" --alias stuart-creator --host 127.0.0.1 --port "$PORT" --jinja -np 1 -c "$CTX" -ub "$UB" -t "$THREADS" --prio "$PRIO"
     "${lm_args[@]}" -n -1 --no-context-shift -ngl 99 "${EXP[@]}" --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
     --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":true}' --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
     "${mtp_args[@]}" "${extra[@]}")

if [ "$do_print" -eq 1 ]; then
    echo "profile=$profile ctx=$CTX ub=$UB tier=$use_tier mtp=$use_mtp bg=$bg"
    for e in "${envs[@]}"; do printf 'env  %s\n' "$e"; done
    printf 'cmd '; printf ' %q' "${CMD[@]}"; printf '\n'
    printf 'log  %s\n' "$LOG"
    exit 0
fi

# ---- checks before anything starts -------------------------------------------------------------------------------------------------------------------------
[ -x "$BIN" ]   || die "server binary not found or not executable: $BIN$([ "$use_tier" -eq 1 ] && echo '   (build it first: bash moe_track/patches/09_build_pelias_tier.sh build)')"
[ -f "$MODEL" ] || die "model file not found: $MODEL"
if [ "$use_tier" -eq 1 ]; then
    [ -f "$HOT_SET" ] || die "hot-set file not found: $HOT_SET   (make it first: python moe_track/sim/export_hot_set.py). Without it the tier build would run as a plain control."
fi
if [ -n "$mtp_gguf" ]; then
    [ -f "$mtp_gguf" ] || die "draft model not found: $mtp_gguf"
fi
if port_in_use; then
    echo "port $PORT is already in use, so a server is already running (llama --status)" >&2
    exit 1
fi
if [ -f "$PIDFILE" ] && opid="$(cat "$PIDFILE" 2>/dev/null)" && is_int "$opid" && kill -0 "$opid" 2>/dev/null; then
    echo "a background server is already recorded in $PIDFILE (pid $opid); stop it first: llama --stop" >&2
    exit 1
fi
mkdir -p "$LOGDIR" || die "cannot create the log folder $LOGDIR"
[ -e "$LOG" ] && die "log file $LOG already exists; try again in a second"

for e in "${envs[@]}"; do export "${e?}"; done

if [ "$bg" -eq 1 ]; then
    setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
    spid=$!
    echo "$spid" > "$PIDFILE"
    sleep 1
    if ! kill -0 "$spid" 2>/dev/null; then
        rm -f "$PIDFILE"
        echo "the server exited right away; last log lines ($LOG):" >&2
        tail -n 15 "$LOG" >&2
        exit 1
    fi
    echo "started in the background (pid $spid); profile=$profile ctx=$CTX ub=$UB tier=$use_tier mtp=$use_mtp; log: $LOG"
else
    "${CMD[@]}" 2>&1 | tee "$LOG"
fi
