#!/usr/bin/env bash
# File: <project>\moe_track\01_phase1_os_tuning.sh   (WSL: <project>/moe_track/01_phase1_os_tuning.sh)
#
# Phase 1: zero-code launch tuning on top of the working --n-cpu-moe 28 setting. ONE change per run, so a gain
# or loss can be pinned on one cause. Needs no sudo and writes nothing to /sys or to any Windows setting.
#
#   bash 01_phase1_os_tuning.sh audit                   read-only: THP, huge pages, .wslconfig, memlock, CPU, memory
#   bash 01_phase1_os_tuning.sh run lmnone              1b: add only  --load-mode none  (pinned anonymous load)
#   bash 01_phase1_os_tuning.sh run t8|t7|t6            1c: add only  -t N  (set KEEP_LMNONE=1 to keep --load-mode none if 1b won)
#   bash 01_phase1_os_tuning.sh run custom --label X -- <llama flags>    any other single change, labelled X
#   bash 01_phase1_os_tuning.sh gate NEW.json [BASE.json]   score a finished run with gate P1
#
# A "run" stops the server, restarts it with the production flags plus the one change, waits until it is healthy,
# prints the buffer lines from the load log (so a silent failure to pin memory is visible), runs the project's
# `check` and the speed test under its own label, then scores gate P1 against the baseline.
# Keep a change only if gate P1 says PASS. Do not run anything else heavy while a run is going.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

BASELINE_DEFAULT="$LOGS/speed_server/20261020_032031_q4km-ncmoe28-ctx131k-ub2048.json"
BASELINE="${BASELINE:-$BASELINE_DEFAULT}"
EXPERT_FLAGS="${EXPERT_FLAGS:---n-cpu-moe 28}"
mkdir -p "$LOGS/floors"

usage() { sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

# ------------------------------------------------------------------------------------------------ audit
cmd_audit() {
    local out="$LOGS/floors/phase1_audit.txt"
    {
        echo "=== Phase 1a audit (read-only) $(date '+%Y-%m-%d %H:%M:%S') ==="
        echo "--- transparent huge pages ---"
        for f in enabled defrag; do
            printf '%-8s ' "$f:"; cat "/sys/kernel/mm/transparent_hugepage/$f" 2>&1
        done
        grep -i -E 'hugepages|AnonHuge|Hugetlb' /proc/meminfo 2>/dev/null | head -8
        echo "--- .wslconfig (Windows side) ---"
        local found=0
        for f in /mnt/c/Users/*/.wslconfig; do
            [ -f "$f" ] && { found=1; echo "# $f"; cat "$f"; }
        done
        [ "$found" = 0 ] && echo "(no .wslconfig found: WSL defaults apply)"
        echo "--- locked-memory limit (pinned memory) ---"
        printf 'ulimit -l: '; ulimit -l
        echo "--- cpu ---"
        lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|Core|L3|MHz' || nproc
        echo "--- memory ---"
        free -m
        echo "--- servers running ---"
        pgrep -a llama-server | cut -c1-200 || echo "none"
        echo "--- gpu ---"
        nvidia-smi --query-gpu=name,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1
        echo
        echo "--- how to read this ---"
        echo "THP 'always': 2 MB pages are used for large anonymous allocations. 'madvise': only programs that ask; llama.cpp does not,"
        echo "so --load-mode none (which loads into ordinary anonymous memory) will use 4 KB pages. 'never': no huge pages."
        echo "If THP is not 'always', changing it is NOT done by this script. It needs root or a Windows-side .wslconfig kernel line"
        echo "(restarts WSL, affects everything, effect on the WSL2 kernel unverified). Test 1b/1c first: if they gain enough, skip it."
    } 2>&1 | tee "$out"
    moe_log "audit saved to $out"
}

# ------------------------------------------------------------------------------------------------ run
cmd_run() {
    [ $# -ge 1 ] || usage
    local preset="$1"; shift
    local label="" extras=()
    case "$preset" in
        lmnone) label="lmnone"; extras=(--load-mode none) ;;
        t6|t7|t8)
            label="$preset"; extras=(-t "${preset#t}")
            [ "${KEEP_LMNONE:-0}" = 1 ] && { label="lmnone-$preset"; extras+=(--load-mode none); } ;;
        custom)
            [ "${1:-}" = "--label" ] || moe_die "custom needs: --label NAME -- <flags>"
            label="$2"; shift 2
            [ "${1:-}" = "--" ] && shift
            extras=("$@") ;;
        *) usage ;;
    esac

    have_cmd() { command -v "$1" >/dev/null 2>&1; }
    have_cmd llama || moe_die "the 'llama' launcher is not on PATH (RUNBOOK Step 7b creates ~/.local/bin/llama)"
    { [ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1; } || moe_die "venv python not found at $PY"
    [ -f "$BASELINE" ] || moe_log "note: baseline file $BASELINE not found; gate P1 will be skipped"

    local full_label="q4km-ncmoe28-${label}-ctx131k-ub2048"
    moe_log "run '$label': extras = ${extras[*]}  (expert flags: $EXPERT_FLAGS)"

    local used; used="$(gpu_used_mib)"
    if [ -n "$used" ] && [ "$used" -gt 1500 ] 2>/dev/null; then
        moe_log "WARNING: ${used} MiB of GPU memory is in use before the server starts (another program?). Peak-VRAM numbers will include it."
    fi

    llama --stop || true
    sleep 2
    wait_for_port_free 60 || moe_die "something still listens on port $PORT; stop it first (llama --status)"

    EXPERT_FLAGS="$EXPERT_FLAGS" llama --serve --stuart-creator --bg -- -c 131072 -ub 2048 "${extras[@]}" -lv 4 \
        || moe_die "the launcher failed to start"
    moe_log "waiting for the model to load (about 2.5 minutes)"
    wait_for_server 420 || moe_die "server did not come up; read $(latest_server_log)"

    local srv_log; srv_log="$(latest_server_log)"
    local rec="$LOGS/floors/phase1_launch_${full_label}.txt"
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S')  label $full_label"
        echo "# command line of the running server (launch-parity record):"
        pgrep -a llama-server | cut -c1-2000
        echo "# server log: $srv_log"
        echo "# load-log lines about buffers, threads and listening (log wording is not guaranteed; read, do not parse):"
        grep -n -i -E 'CUDA_Host|CPU_Mapped|CPU_REPACK|buffer size|n_threads|n_ctx_slot|listening' "$srv_log" 2>/dev/null | head -40
    } > "$rec"
    cat "$rec"

    cd "$CREATOR" || moe_die "cannot enter $CREATOR"
    "$PY" stuart_server_probe.py check || moe_die "check failed; the speed numbers would mean nothing"
    "$PY" stuart_server_speed.py run --label "$full_label" \
        --note "Q4_K_M, KV q8_0, ctx 131072, ub 2048, n-cpu-moe 28, extras: ${extras[*]}"

    local newest; newest="$(newest_json "$LOGS/speed_server" "$full_label")"
    [ -n "$newest" ] || moe_die "no result file found for label $full_label"
    moe_log "result: $newest"
    if [ -f "$BASELINE" ]; then
        "$PY" "$HERE/lib/gates.py" p1 --baseline "$BASELINE" --candidate "$newest"
        local rc=$?
        [ $rc -eq 0 ] && moe_log "P1 PASS: keep '$label'. Next change goes on top of it." || moe_log "P1 HOLD: do not keep '$label'."
        return $rc
    fi
}

# ------------------------------------------------------------------------------------------------ gate
cmd_gate() {
    [ $# -ge 1 ] || usage
    local cand="$1" base="${2:-$BASELINE}"
    "$PY" "$HERE/lib/gates.py" p1 --baseline "$base" --candidate "$cand"
}

case "${1:-}" in
    audit) shift; cmd_audit "$@" ;;
    run)   shift; cmd_run "$@" ;;
    gate)  shift; cmd_gate "$@" ;;
    -h|--help|"") usage 0 ;;
    *) usage ;;
esac
