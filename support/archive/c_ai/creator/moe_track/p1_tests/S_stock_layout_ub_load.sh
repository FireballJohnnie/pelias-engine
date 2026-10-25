#!/usr/bin/env bash
# S  Stock build (no hot bank), context 131072, KV q8_0: --n-cpu-moe 25/26/27, -ub 4096, --load-mode none, and generation threads (-t 5,6,7,8,10 with -tb 8).
# STATUS: COMPLETED 2026-10-22 12:04 to 12:54 (stamps 120417, 120758, 121055, 123225, 123535, 123833, 124326, 124824, 125013, 125203, 125345).
# Uses the llama launcher (llama --serve) and stuart_server_speed.py like the original runs. Each variant stops the server and starts a new one.
GROUP=S_stock_layout_ub_load; NO_TSV_HEADER=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
cd <project> || exit 1; TSV="$OUT/${GROUP}_$(date +%Y%m%d_%H%M%S).tsv"
stock_run() {   # LABEL NCMOE "SERVER FLAGS AFTER --"
  local label="$1" n="$2" fl="$3" U TOT F
  echo; echo "### ${GROUP} / ${label}   start $(date +%H:%M:%S)   n-cpu-moe $n   flags: $fl"
  llama --stop
  EXPERT_FLAGS="--n-cpu-moe $n" llama --serve --stuart-creator --bg -- -c 131072 $fl
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8012/health | grep -q ok && break; sleep 5; done
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  echo "    after load: $U / $TOT MiB"
  if curl -s -m 2 http://127.0.0.1:8012/health | grep -q ok && [ "$U" -lt $((TOT-512)) ]; then
    $PY stuart_server_probe.py check && $PY stuart_server_speed.py run --label "$label" --note "Q4_K_M, KV q8_0, ctx 131072, n-cpu-moe $n, extras: $fl"
    F=$(ls -t logs/speed_server/*_${label}.txt 2>/dev/null | head -n 1)
    printf '%s\t%s\t%s\n' "$label" "$(grep -E '^(short|4000|12000)' "$F" | awk '{printf "%s gen %s | ", $1, $6}')" "$F" >> "$TSV"
  else
    echo "    STOP: server not healthy or VRAM within 512 MiB of the card total"; printf '%s\tFAILED\t-\n' "$label" >> "$TSV"
  fi
}
stock_run q4km-ncmoe27-ctx131k-ub2048              27 "-ub 2048"
stock_run q4km-ncmoe26-ctx131k-ub2048              26 "-ub 2048"
stock_run q4km-ncmoe25-ctx131k-ub2048              25 "-ub 2048"
stock_run q4km-ncmoe27-ctx131k-ub4096              27 "-ub 4096"
stock_run q4km-ncmoe27-lmnone-ctx131k-ub2048       27 "-ub 2048 --load-mode none"
stock_run q4km-ncmoe25-lmnone-ctx131k-ub2048       25 "-ub 2048 --load-mode none"
for TT in 7 6 5 8 10; do
  stock_run q4km-ncmoe25-lmnone-t$TT-tb8-ctx131k-ub2048 25 "-ub 2048 --load-mode none -t $TT -tb 8"
done
llama --stop
echo; echo "=== ${GROUP} summary ($TSV)"; cat "$TSV"
