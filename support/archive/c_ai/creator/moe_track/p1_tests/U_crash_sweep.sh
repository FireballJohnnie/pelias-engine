#!/usr/bin/env bash
# U  Crash sweep: fresh server per prompt length, a prompt of N consecutive token ids, 2 tokens generated; counts "illegal memory" lines.
# STATUS: COMPLETED 2026-10-22 14:04 (prompts of 8 to 61 tokens at bypass 32 after the quantiser fix: no CUDA error). RECONSTRUCTED from the T2 block of the ledger; the original 14:04 script text is not in my records.
# Override with env vars:  BY=9 NLIST="9 13 34" SET=/path/to/hot_set CAP=7800 bash U_crash_sweep.sh     (defaults: bypass 32, default 6.1 GiB set, cap 6400)
GROUP=U_crash_sweep; NO_TSV_HEADER=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
BY="${BY:-32}"; SET="${SET:-$H/pelias_hot_set.txt}"; CAP="${CAP:-6400}"; NLIST="${NLIST:-8 9 10 12 13 16 20 31 32 34 40 61}"
guard() { pkill -TERM -f llama-server 2>/dev/null; sleep 4; pkill -KILL -f llama-server 2>/dev/null; sleep 2; }
BIN=$HOME/llama-pelias-tier/build-wsl/bin/llama-server
S="$OUT/U_sweep_by${BY}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$S"
RUN="env PELIAS_HOT_SET=$SET PELIAS_TIER_BYPASS_N=$BY PELIAS_TIER_MAX_MIB=$CAP PELIAS_TIER_VERBOSE=1"
FLAGS="-m $HOME/models/test-model-Q4_K_M.gguf --host 127.0.0.1 --port 8013 -np 1 -c 32768 -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-cpu-moe 40 --fit off -lv 4"
echo "bypass $BY, hot set $SET, cap $CAP MiB, lengths: $NLIST   (logs in $S)"
for N in $NLIST; do
  guard
  PELIAS_TIER_STATS_FILE=$S/stats_N$N.json $RUN $BIN $FLAGS > $S/server_N$N.log 2>&1 &
  P=$!
  for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:8013/health | grep -q ok && break; kill -0 $P 2>/dev/null || break; sleep 1; done
  HTTP=$(python3 -c "import json;print(json.dumps({'prompt':list(range(1000,1000+$N)),'n_predict':2,'temperature':0,'cache_prompt':False}))" | curl -s -m 180 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d @- http://127.0.0.1:8013/completion)
  kill -TERM $P 2>/dev/null; wait $P 2>/dev/null
  echo "tokens=$N http=$HTTP illegal=$(grep -c 'illegal memory' $S/server_N$N.log) dual=$(tr -d '\n ' < $S/stats_N$N.json 2>/dev/null | head -c 200)" | tee -a "$S/summary.txt"
done
guard
echo; echo "summary file: $S/summary.txt   (http=200 and illegal=0 means pass)"
