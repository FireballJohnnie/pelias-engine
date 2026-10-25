#!/usr/bin/env bash
# File: <project>\moe_track\trace\build.sh   (WSL: <project>/moe_track/trace/build.sh)
# Compiles trace/moe_trace.cpp against the llama.cpp tree and build that production uses, so the tracer sees the same
# graph code, the same CUDA backend and the same libraries as llama-server. Compiles only; runs nothing.
#   bash trace/build.sh
#   LLAMA_PROD=/path/to/llama.cpp LLAMA_BUILD_BIN=/path/to/build/bin bash trace/build.sh
# After a llama.cpp update, build again: the tracer must match the server it is being compared with.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"
BIN_DIR="${LLAMA_BUILD_BIN:-$LLAMA_PROD_BIN}"
OUT="${MOE_TRACE_BIN:-$HERE/bin}"
mkdir -p "$OUT"

command -v g++ >/dev/null 2>&1 || moe_die "g++ not found"
[ -f "$LLAMA_PROD/include/llama.h" ]            || moe_die "no $LLAMA_PROD/include/llama.h (set LLAMA_PROD to the llama.cpp source folder)"
[ -f "$LLAMA_PROD/ggml/include/ggml-backend.h" ] || moe_die "no ggml headers under $LLAMA_PROD/ggml/include"
ls "$BIN_DIR"/libllama.so* >/dev/null 2>&1       || moe_die "no libllama.so in $BIN_DIR (set LLAMA_BUILD_BIN; the production build must be a shared-library build)"
ls "$BIN_DIR"/libggml.so* >/dev/null 2>&1        || moe_die "no libggml.so in $BIN_DIR"

if [ -d "$LLAMA_PROD/.git" ]; then
    have="$(git -C "$LLAMA_PROD" rev-parse HEAD 2>/dev/null)"
    [ -n "$have" ] && [ "$have" != "$LLAMA_COMMIT" ] && moe_log "NOTE: source tree is at ${have:0:9}, the pinned production commit is ${LLAMA_COMMIT:0:9}. The tracer matches the tree it is built from."
fi

moe_log "compiling against $LLAMA_PROD and libraries in $BIN_DIR"
g++ -O2 -std=c++17 -Wall -Wextra \
    -DMOE_TRACE_DEFAULT_BACKEND_DIR="\"$BIN_DIR\"" \
    "$HERE/moe_trace.cpp" \
    -I"$LLAMA_PROD/include" -I"$LLAMA_PROD/ggml/include" \
    -L"$BIN_DIR" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$BIN_DIR" \
    -o "$OUT/moe_trace" || moe_die "compile failed (llama.h API change? the error above names the call)"
chmod +x "$OUT/moe_trace" 2>/dev/null
moe_log "built $OUT/moe_trace"
