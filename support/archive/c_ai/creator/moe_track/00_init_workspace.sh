#!/usr/bin/env bash
# File: <project>\moe_track\00_init_workspace.sh   (WSL: <project>/moe_track/00_init_workspace.sh)
# Step 0. Creates the folders the other scripts write into and checks what tools are present.
# It writes no files except empty folders, changes no setting, starts no server and needs no sudo.
#   bash <project>/moe_track/00_init_workspace.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

mkdir -p "$MOE_TRACK"/{lib,microbench,trace,sim,fork,patches,pelias_serve,mtp_align,tests}
mkdir -p "$LOGS"/{floors,moe_trace,fork_runs}
moe_log "folders ready under $MOE_TRACK and $LOGS"

have() { command -v "$1" >/dev/null 2>&1; }
row()  { printf '  %-14s %s\n' "$1" "$2"; }

echo "--- tools (a missing one only blocks the steps that need it) ---"
for t in bash python3 curl ss git cmake make g++ nvcc nvidia-smi llama; do
    if have "$t"; then row "$t" "$(command -v "$t")"; else row "$t" "NOT FOUND"; fi
done
if [ -x "$PY" ]; then row "venv python" "$PY ($("$PY" --version 2>&1))"; else row "venv python" "NOT FOUND at $PY"; fi
if [ -x "$PY" ]; then
    "$PY" - <<'EOF' 2>&1 | sed 's/^/  /'
import importlib
for m in ("numpy", "torch", "gguf", "safetensors"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:<14}{getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m:<14}missing ({type(e).__name__})")
EOF
fi

echo "--- files ---"
for f in "$GGUF" "$GGUF.sha256" "$LLAMA_PROD_BIN/llama-server" "$LLAMA_PROD_BIN/llama-bench" "$LLAMA_PROD_BIN/llama-perplexity"; do
    if [ -e "$f" ]; then row "found" "$f"; else row "MISSING" "$f"; fi
done

echo "--- cores and memory ---"
row "cpu threads" "$(nproc)"
free -m | sed 's/^/  /'

echo
moe_log "next: bash $MOE_TRACK/01_phase1_os_tuning.sh audit      (read-only)"
moe_log "full order of steps: $MOE_TRACK/README.md"
