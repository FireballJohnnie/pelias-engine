#!/usr/bin/env bash
# V  Prompt-evaluation time for prompts of N identical tokens, 3 requests each, after one warm-up, for each bypass value (7.5 GiB set, -t 6 -tb 8 -ub 2048, default load).
# STATUS: COMPLETED 2026-10-22 about 13:3x with bypass 9 and 13 (that run, before the quantiser fix, crashed at N=13 with bypass 13). Original run_by function reproduced from the pasted command.
# Override:  BY_LIST="9 32" N_LIST="8 16 24 32 48" bash V_bypass_latency_by_length.sh    (default: BY_LIST="9 13", N_LIST="8 12 13 14 15 16 17 61")
GROUP=V_bypass_latency_by_length; NO_TSV_HEADER=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
BY_LIST="${BY_LIST:-9 13}"; N_LIST="${N_LIST:-8 12 13 14 15 16 17 61}"
BIN=$HOME/llama-pelias-tier/build-wsl/bin/llama-server
llama --stop
run_by() {
  BY=$1
  ( export PELIAS_HOT_SET=$H/pelias_hot_set_b7.5.txt PELIAS_TIER_BYPASS_N=$BY PELIAS_TIER_MAX_MIB=7800 PELIAS_TIER_VERBOSE=1; nohup $BIN -m $HOME/models/test-model-Q4_K_M.gguf --host 127.0.0.1 --port 8012 -np 1 -c 32768 -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --n-cpu-moe 40 --fit off -t 6 -tb 8 -ub 2048 > /dev/null 2>&1 & )
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8012/health | grep -q ok && break; sleep 5; done
  curl -s -m 120 http://127.0.0.1:8012/completion -d '{"prompt":[1000,1000,1000,1000],"n_predict":1,"cache_prompt":false}' > /dev/null
  for N in $N_LIST; do
    TOK=$(python3 -c "print(','.join(['1000']*$N))")
    for r in 1 2 3; do
      printf "bypass=%s N=%s: " $BY $N
      curl -s -m 120 http://127.0.0.1:8012/completion -d "{\"prompt\":[$TOK],\"n_predict\":1,\"cache_prompt\":false}" | python3 -c "import sys,json; t=json.load(sys.stdin)['timings']; print(t['prompt_n'],'tokens',round(t['prompt_ms']),'ms')" 2>&1 | tail -n 1
    done
  done
  pkill -f "$BIN"; sleep 8
}
for BY in $BY_LIST; do run_by $BY; done
llama --status
