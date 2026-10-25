#!/usr/bin/env bash
# File: <project>\moe_track\microbench\05_n_sweep.sh   (WSL: <project>/moe_track/microbench/05_n_sweep.sh)
#
# Step 2e / gate G0 data: measure short-prompt generation speed at extra --n-cpu-moe values, so the floor model is tested
# on points it was NOT fitted on. The model is fitted on N=40 (all experts on CPU) and N=28 (the current setting) and has
# to predict N=34 within 10%.
#
#   bash microbench/05_n_sweep.sh                 run N=34 (default), then score gate G0
#   NS="34 22" bash microbench/05_n_sweep.sh      run several extra N values (each takes about 4 minutes)
#   bash microbench/05_n_sweep.sh gate            only score G0 from the files already on disk
#
# Existing measurements are reused: the all-CPU point is the 2026-10-20 "experts on CPU, ctx 131072, ub 2048" run and the N=28
# point is the production baseline. Override with  POINT40=FILE.json  POINT28=FILE.json.
# One change per run: everything except --n-cpu-moe is the production configuration (launch parity).
# Only depth 0 (short prompt) is measured because G0 compares short-prompt generation speed.
# Each run stops the server and restarts it. The server is left running at the LAST tested N; restart your usual setting after.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"

SPEED_DIR="$LOGS/speed_server"
POINT40="${POINT40:-$SPEED_DIR/20261020_025525_q4km-kvq8_0-ctx131k-ub2048.json}"
POINT28="${POINT28:-$SPEED_DIR/20261020_032031_q4km-ncmoe28-ctx131k-ub2048.json}"
NS="${NS:-34}"
mkdir -p "$LOGS/floors"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

run_one() {
    local n="$1"
    local label="q4km-ncmoe${n}-sweep-ctx131k-ub2048"
    moe_log "N=$n: restarting the server with --n-cpu-moe $n (nothing else changed)"
    llama --stop || true
    sleep 2
    wait_for_port_free 60 || { moe_log "port $PORT is busy"; return 1; }
    EXPERT_FLAGS="--n-cpu-moe $n" llama --serve --stuart-creator --bg -- -c 131072 -ub 2048 -lv 4 || { moe_log "launcher failed"; return 1; }
    wait_for_server 420 || { moe_log "server did not come up; read $(latest_server_log)"; return 1; }
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S')  label $label"
        pgrep -a llama-server | cut -c1-2000
    } > "$LOGS/floors/sweep_launch_${label}.txt"
    local used; used="$(gpu_used_mib)"
    moe_log "GPU memory in use with this setting: ${used:-unknown} MiB of 12288"
    cd "$CREATOR" || return 1
    "$PY" stuart_server_probe.py check || { moe_log "check failed"; return 1; }
    "$PY" stuart_server_speed.py run --label "$label" --depths 0 --repeats 4 \
        --note "Q4_K_M, KV q8_0, ctx 131072, ub 2048, n-cpu-moe $n, G0 sweep point (depth 0 only)" || return 1
    local newest; newest="$(newest_json "$SPEED_DIR" "$label")"
    [ -n "$newest" ] || { moe_log "no result file for $label"; return 1; }
    echo "$n=$newest" >> "$LOGS/floors/g0_points.txt"
    moe_log "N=$n result: $newest"
}

gate() {
    local args=(--point "40=$POINT40" --point "28=$POINT28")
    local line
    if [ -f "$LOGS/floors/g0_points.txt" ]; then
        # the newest file per N wins
        while IFS= read -r line; do
            [ -n "$line" ] && args+=(--point "$line")
        done < <(awk -F= '{last[$1]=$0} END {for (k in last) print last[k]}' "$LOGS/floors/g0_points.txt" | sort)
    fi
    "$PY" "$HERE/06_verify_gate_g0.py" "${args[@]}"
}

case "${1:-run}" in
    gate) gate; exit $? ;;
    run) ;;
    *) sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

have_cmd llama || moe_die "the 'llama' launcher is not on PATH (RUNBOOK Step 7b creates ~/.local/bin/llama)"
{ [ -x "$PY" ] || have_cmd "$PY"; } || moe_die "venv python not found at $PY"
[ -f "$POINT40" ] || moe_die "all-CPU speed file not found: $POINT40 (set POINT40=...)"
[ -f "$POINT28" ] || moe_die "N=28 speed file not found: $POINT28 (set POINT28=...)"
for n in $NS; do
    case "$n" in 40|28) moe_log "N=$n is already a fitted point; skipping"; continue ;; esac
    [ "$n" -ge 1 ] 2>/dev/null && [ "$n" -le 39 ] 2>/dev/null || { moe_log "ignoring N='$n' (use 1 to 39)"; continue; }
    run_one "$n" || moe_die "N=$n failed; the sweep stops here"
done
gate
