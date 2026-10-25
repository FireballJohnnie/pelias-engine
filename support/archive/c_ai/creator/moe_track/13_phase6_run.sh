#!/usr/bin/env bash
# File: <project>\moe_track\13_phase6_run.sh   (WSL: <project>/moe_track/13_phase6_run.sh)
# Phase 6 driver: make the hot set, measure the Pelias-Tier build against the production control, and check that the answers did not change.
# It only calls the tools that already exist in this folder; it contains no new logic of its own, and it never edits the production build.
#
#   bash 13_phase6_run.sh check                  what exists and what is missing (starts nothing, writes nothing)
#   bash 13_phase6_run.sh hotset [args]          = python sim/export_hot_set.py [args]      (needs the traces from Phase 3; default bank budget 5.0 GiB)
#   bash 13_phase6_run.sh speed [SPEC...]        stops the production server, starts the tier build, runs the same speed test and gates as every Phase 5 variant,
#                                                stops it and starts the production server again.   default SPEC:  tier:n=40
#                                                Examples:  speed tier:n=40,mtp=2      speed tier:n=40,bypass=16      speed tier:n=40 tier:n=40,cap=5632
#   bash 13_phase6_run.sh kl-base                reference logits from the PRODUCTION build with the same small batches (server must be stopped; ~4 GB of disk)
#   bash 13_phase6_run.sh kl-compare             same text through the TIER build with its hot bank active, then gate H6 (mean KL <= 0.005, top-1 >= 99%)
#   bash 13_phase6_run.sh report                 table of everything measured so far
#
# Why the KL check uses batches of 16: the bank is used only for batches smaller than PELIAS_TIER_BYPASS_N (32). A normal perplexity run uses batches of
# 512, would take the stock path, and would report a perfect score without testing anything. -b 16 -ub 16 keeps the check honest; both runs use it.
#
# Overrides:  HOT_SET=file   TIER_SRC=~/llama-pelias-tier   BYPASS=32   KL_LABEL_BASE=tier_ref16   KL_LABEL=tier16
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

MODE="${1:-check}"
[ "$#" -gt 0 ] && shift
TIER_SRC="${TIER_SRC:-$HOME/llama-pelias-tier}"
TIER_BIN="$TIER_SRC/build-wsl/bin"
HOT_SET="${HOT_SET:-$LOGS/pelias_hot_set.txt}"
BYPASS="${BYPASS:-32}"
KL_LABEL_BASE="${KL_LABEL_BASE:-tier_ref16}"
KL_LABEL="${KL_LABEL:-tier16}"
KL_DIR="$LOGS/moe_trace/kl"
SMALL_BATCH=(-b 16 -ub 16)

pyrun() { if [ -x "$PY" ]; then "$PY" "$@"; else python3 "$@"; fi; }
need_stopped() { if server_healthy; then moe_die "a llama-server is answering on port $PORT and holds the GPU. Stop it first:  llama --stop"; fi; }

case "$MODE" in
check)
    moe_log "Phase 6 status (nothing is started or written)"
    ok=1
    [ -f "$GGUF" ]                 && moe_log "model            : ok  $GGUF"                  || { moe_log "model            : MISSING $GGUF"; ok=0; }
    [ -x "$TIER_BIN/llama-server" ] && moe_log "tier build       : ok  $TIER_BIN/llama-server"  || { moe_log "tier build       : MISSING -> bash patches/09_build_pelias_tier.sh build"; ok=0; }
    [ -x "$TIER_BIN/llama-perplexity" ] && moe_log "tier perplexity  : ok" || moe_log "tier perplexity  : missing (needed only for kl-compare)"
    if [ -f "$HOT_SET" ]; then
        moe_log "hot set          : ok  $HOT_SET  ($(awk '/^layer/{l++; n+=NF-3} END{print l+0" layers, "n+0" experts"}' "$HOT_SET"))"
        [ -f "$HOT_SET.json" ] && pyrun - "$HOT_SET.json" <<'PY' 2>/dev/null || true
import json, sys
m = json.load(open(sys.argv[1]))
print(f"[moe_track] hot set bank     : {m.get('bank_gib_with_zero_slots')} GiB incl. zero slots, chosen from {m.get('chosen_from')}; "
      f"estimated byte hit rate in-sample {m.get('insample',{}).get('hit_rate_bytes')}, held-out {(m.get('heldout') or {}).get('hit_rate_bytes')}")
PY
    else
        moe_log "hot set          : MISSING $HOT_SET -> bash 13_phase6_run.sh hotset   (needs traces: python trace/05_collect_traces.py)"; ok=0
    fi
    base="$(ls "$LOGS"/speed_server/*ncmoe28*.json 2>/dev/null | head -1)"
    [ -n "$base" ] && moe_log "speed baseline   : ok  $base" || moe_log "speed baseline   : the N=28 file from 2026-10-20 was not found; speed will need --baseline"
    if server_healthy; then moe_log "server           : RUNNING on port $PORT ('speed' will stop it and start it again afterwards)"; else moe_log "server           : not running"; fi
    [ "$ok" -eq 1 ] && moe_log "ready: bash 13_phase6_run.sh speed" || moe_log "not ready: fix the MISSING items above first"
    ;;
hotset)
    exec "$(command -v python3)" "$HERE/sim/export_hot_set.py" "$@"
    ;;
speed)
    [ -x "$TIER_BIN/llama-server" ] || moe_die "tier build not found ($TIER_BIN). Build it first: bash patches/09_build_pelias_tier.sh build"
    [ -f "$HOT_SET" ] || moe_die "hot set not found ($HOT_SET). Make it first: bash 13_phase6_run.sh hotset"
    SPECS=("$@")
    [ "${#SPECS[@]}" -gt 0 ] || SPECS=("tier:n=40,bypass=$BYPASS")
    export TIER_SRC
    moe_log "variants: ${SPECS[*]}   (each: load 3-8 min + check + speed test ~6 min)"
    pyrun "$HERE/fork/gauntlet.py" run "${SPECS[@]}" --restore
    rc=$?
    moe_log "gauntlet finished with code $rc. Results: logs/forks/<variant>/verdict.json and the table below"
    pyrun "$HERE/fork/gauntlet.py" report
    moe_log "next: bash 13_phase6_run.sh kl-base   then   kl-compare"
    exit "$rc"
    ;;
kl-base)
    need_stopped
    moe_log "reference from the production build, batches of 16 (label $KL_LABEL_BASE)"
    pyrun "$HERE/sim/kl_gate.py" base --label "$KL_LABEL_BASE" --extra "${SMALL_BATCH[*]}"
    ;;
kl-compare)
    need_stopped
    [ -x "$TIER_BIN/llama-perplexity" ] || moe_die "$TIER_BIN/llama-perplexity not found; build it: cmake --build $TIER_SRC/build-wsl --target llama-perplexity -j 8"
    [ -f "$HOT_SET" ] || moe_die "hot set not found ($HOT_SET)"
    [ -f "$KL_DIR/$KL_LABEL_BASE.logits.bin" ] || moe_die "no reference yet. Run first: bash 13_phase6_run.sh kl-base"
    export PELIAS_HOT_SET="$HOT_SET" PELIAS_TIER_BYPASS_N="$BYPASS" PELIAS_TIER_VERBOSE=1 PELIAS_TIER_MAX_MIB="${PELIAS_TIER_MAX_MIB:-6144}"
    export PELIAS_TIER_STATS_FILE="$KL_DIR/$KL_LABEL.tier_stats.json"
    moe_log "tier build, experts in RAM (--n-cpu-moe 40) + VRAM hot bank, batches of 16 (label $KL_LABEL)"
    pyrun "$HERE/sim/kl_gate.py" compare --label "$KL_LABEL" --base "$KL_LABEL_BASE" --bin "$TIER_BIN" --extra "${SMALL_BATCH[*]} --n-cpu-moe 40 --fit off"
    rc=$?
    [ "$rc" -eq 0 ] || exit "$rc"
    LOG="$(ls -t "$KL_DIR/${KL_LABEL}_vs_${KL_LABEL_BASE}".*.log 2>/dev/null | head -1)"
    if [ -n "$LOG" ] && grep -q "pelias-tier: hot bank active" "$LOG"; then
        moe_log "confirmed: the hot bank was active during the comparison ($(grep -o 'hot bank active for [0-9]* layers, [0-9.]* MiB' "$LOG" | head -1))"
        if [ -f "$PELIAS_TIER_STATS_FILE" ]; then moe_log "dual graphs built: $(tr -d '\n ' < "$PELIAS_TIER_STATS_FILE" | head -c 300)"; fi
        pyrun "$HERE/lib/gates.py" h6 --kl "$KL_DIR/${KL_LABEL}_vs_${KL_LABEL_BASE}.json"
    else
        moe_log "WARNING: the log has no 'hot bank active' line, so this comparison tested the PLAIN build. Do not trust the score. Log: ${LOG:-<none>}"
        exit 3
    fi
    ;;
report)
    exec "$(command -v python3)" "$HERE/fork/gauntlet.py" report
    ;;
*)
    moe_die "unknown mode '$MODE' (check, hotset, speed, kl-base, kl-compare, report)"
    ;;
esac
