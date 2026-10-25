#!/usr/bin/env bash
# File: <project>\moe_track\env.sh   (WSL: <project>/moe_track/env.sh)
# Source this from the other moe_track scripts:   source <project>/moe_track/env.sh
# Sets paths and helper functions only. It starts nothing and writes nothing.
# Every value can be overridden by exporting the variable before sourcing.

export CREATOR="${CREATOR:-<project>}"
export MOE_TRACK="${MOE_TRACK:-$CREATOR/moe_track}"
export PY="${PY:-$HOME/stuart_creator_engine/venv/bin/python}"
export GGUF="${GGUF:-$CREATOR/gguf/test-model-Q4_K_M.gguf}"
export LOGS="${LOGS:-$CREATOR/logs}"
export LLAMA_PROD="${LLAMA_PROD:-<llama.cpp>}"
export LLAMA_PROD_BIN="${LLAMA_PROD_BIN:-$LLAMA_PROD/build-wsl/bin}"
export LLAMA_COMMIT="${LLAMA_COMMIT:-2ca15f5404760548c39e7b92bd43116a09414a1a}"   # pinned production build b11399
export PORT="${PORT:-8012}"
export HOST="${HOST:-127.0.0.1}"
export PATH="$HOME/.local/bin:$PATH"

# the sidecar checksum is read lazily; a missing file is not an error here
if [ -z "${GGUF_SHA256:-}" ] && [ -f "$GGUF.sha256" ]; then
    GGUF_SHA256="$(awk '{print $1; exit}' "$GGUF.sha256")"
fi
export GGUF_SHA256="${GGUF_SHA256:-}"

# Production launch flags, as baked into ~/.local/bin/llama plus the agreed extras (ctx 131072, ub 2048).
# Launch parity rule: any experiment launch must reproduce these and change only the one thing under test.
# shellcheck disable=SC2034
PROD_FLAGS=(
    --alias stuart-creator --host "$HOST" --port "$PORT" --jinja -np 1
    -c 131072 -ub 2048 -n -1 --no-context-shift -ngl 99
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
    --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":true}'
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
)
# shellcheck disable=SC2034
PROD_EXPERT_FLAGS=(--n-cpu-moe 28)        # the working placement as of 2026-10-20 (41.1 tok/s at short context)
# shellcheck disable=SC2034
ALL_CPU_EXPERT_FLAGS=(-ot ".ffn_.*_exps.=CPU")

moe_log()  { printf '[moe_track] %s\n' "$*"; }
moe_die()  { printf '[moe_track] ERROR: %s\n' "$*" >&2; exit 1; }

# server_healthy: true when /health answers 200
server_healthy() { curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; }

# wait_for_server [max_seconds]: poll /health; returns 1 on timeout
wait_for_server() {
    local max_wait="${1:-300}" waited=0
    until server_healthy; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$max_wait" ]; then
            moe_log "server did not become healthy within ${max_wait}s"
            return 1
        fi
    done
    return 0
}

# wait_for_port_free [max_seconds]: after a stop, wait until nothing listens on the port
wait_for_port_free() {
    local max_wait="${1:-60}" waited=0
    while ss -ltn 2>/dev/null | grep -q ":${PORT} "; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$max_wait" ]; then
            moe_log "port ${PORT} is still in use after ${max_wait}s"
            return 1
        fi
    done
    return 0
}

# latest_server_log: newest server log written by the llama launcher
latest_server_log() { ls -t "$LOGS"/server_Q4_K_M_*.log 2>/dev/null | head -1; }

# newest_json DIR PATTERN: newest result file whose name contains PATTERN
newest_json() { ls -t "$1"/*"$2"*.json 2>/dev/null | head -1; }

# gpu_used_mib: whole-card memory in use (includes the desktop); empty if nvidia-smi is unavailable
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
