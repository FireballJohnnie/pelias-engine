#!/usr/bin/env bash
# File: <project>\moe_track\mtp_align\10a_dump_hidden.sh   (WSL: <project>/moe_track/mtp_align/10a_dump_hidden.sh)
# Stage 10a of the MTP-head alignment: runs your merged Stuart GGUF over the training samples and writes down the hidden state at every position.
# Needs the llama-server STOPPED (it uses the GPU). Builds a small helper (dump_hidden) next to this file with your production llama.cpp headers; no sudo, nothing installed.
#
#   python 10_train_blk40_mtp_lora.py export-samples      first (makes mtp_samples.ids)
#   bash 10a_dump_hidden.sh [check|build|run]             default: run (builds the helper if it is missing)
#
# Output: mtp_hidden.bin next to this file (about 4 KB per token). Refuses to overwrite an existing one; move the old one away first.
# Overrides: SAMPLES=file  OUT=file  THREADS=8  UBATCH=512  CTX=4096  NGL=99  NCPUMOE=28  LIMIT=N (only the first N samples; useful for a first short try)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"

MODE="${1:-run}"
SAMPLES="${SAMPLES:-$HERE/mtp_samples.ids}"
OUT="${OUT:-$HERE/mtp_hidden.bin}"
DUMPER="$HERE/dump_hidden"
THREADS="${THREADS:-8}"; UBATCH="${UBATCH:-512}"; CTX="${CTX:-4096}"; NGL="${NGL:-99}"; NCPUMOE="${NCPUMOE:-28}"

do_build() {
    command -v g++ >/dev/null 2>&1 || moe_die "g++ not found (sudo apt install build-essential is the usual fix; nothing here installs it for you)"
    [ -f "$LLAMA_PROD/include/llama.h" ] || moe_die "$LLAMA_PROD/include/llama.h not found (LLAMA_PROD=$LLAMA_PROD)"
    [ -f "$LLAMA_PROD_BIN/libllama.so" ] || moe_die "$LLAMA_PROD_BIN/libllama.so not found; this tool links against your production build"
    moe_log "building dump_hidden against $LLAMA_PROD (a few seconds)"
    g++ -O2 -std=c++17 "$HERE/dump_hidden.cpp" -I"$LLAMA_PROD/include" -I"$LLAMA_PROD/ggml/include" -L"$LLAMA_PROD_BIN" -lllama -lggml -lggml-base \
        -Wl,-rpath,"$LLAMA_PROD_BIN" -o "$DUMPER" || moe_die "the build failed (the error is above)"
    moe_log "built $DUMPER"
}

case "$MODE" in
check)
    [ -f "$GGUF" ]    && moe_log "model    : ok  $GGUF"      || moe_log "model    : MISSING $GGUF"
    [ -f "$SAMPLES" ] && moe_log "samples  : ok  $SAMPLES"   || moe_log "samples  : MISSING -> python 10_train_blk40_mtp_lora.py export-samples"
    [ -x "$DUMPER" ]  && moe_log "helper   : ok  $DUMPER"    || moe_log "helper   : not built yet (bash $0 build)"
    [ -e "$OUT" ]     && moe_log "output   : EXISTS $OUT (will not be overwritten)" || moe_log "output   : $OUT (free)"
    if server_healthy; then moe_log "server   : RUNNING on port $PORT - stop it first:  llama --stop"; else moe_log "server   : not running"; fi
    ;;
build)
    do_build
    ;;
run)
    [ -f "$GGUF" ]    || moe_die "model not found: $GGUF"
    [ -f "$SAMPLES" ] || moe_die "no sample file ($SAMPLES). Run: python 10_train_blk40_mtp_lora.py export-samples"
    [ -e "$OUT" ]     && moe_die "$OUT already exists. Move it away or set OUT=..."
    if server_healthy; then moe_die "a llama-server is answering on port $PORT and holds the GPU. Stop it first:  llama --stop"; fi
    [ -x "$DUMPER" ] || do_build
    EXTRA=()
    [ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
    moe_log "dumping hidden states (GPU layers $NGL, first $NCPUMOE layers' experts in RAM, batch $UBATCH) -> $OUT"
    "$DUMPER" -m "$GGUF" --in "$SAMPLES" --out "$OUT" -t "$THREADS" --ngl "$NGL" --n-cpu-moe "$NCPUMOE" --ubatch "$UBATCH" --ctx "$CTX" ${EXTRA[@]+"${EXTRA[@]}"}
    rc=$?
    [ "$rc" -eq 0 ] || moe_die "dump_hidden stopped with code $rc (see above)"
    moe_log "done: $OUT   next: python 10_train_blk40_mtp_lora.py train"
    ;;
*)
    moe_die "unknown mode '$MODE' (check, build, run)"
    ;;
esac
