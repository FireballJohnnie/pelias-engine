#!/usr/bin/env bash
# J  Sampling and output length (generation speed). Same base as the P1 cell: 7.5 GiB set, bypass 32, --load-mode none -ub 2048 -t 6 -tb 8.
# fork/gauntlet.py cannot pass sampling settings to the speed tool, so this script starts the server itself (as K does) and runs the speed tool against it.
# Launch 1, one server, four speed runs:   base (temp 1.0 top-p 0.95 top-k 20 min-p 0)  |  greedy (temp 0)  |  no truncation (top-k 0, top-p 1.0)  |  top-k 40 min-p 0.05
# Launch 2: server flag --ignore-eos (every request generates the full 256 tokens = fixed output length)
# Launch 3: server penalties --repeat-penalty 1.1 --presence-penalty 0.5 --frequency-penalty 0.3
# Launch 4: base again (control: launch-to-launch drift, which earlier ran 5 to 21%)
# Not scripted: thinking on/off. The speed tool fixes enable_thinking=false, and thinking changes what is generated, not the speed per token [ESTIMATE].
# The speed tool already leaves a request out of the rate if it generated fewer than its minimum tokens; the 'reps' columns (good/total) show how many were kept.
# Run: bash <project>/moe_track/p1_tests/J_sampling_output.sh      About 4 launches x 3 min [ESTIMATE].
GROUP=J_sampling_output; NO_TSV_HEADER=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
CREATOR=<project>
BIN=$HOME/llama-pelias-tier/build-wsl/bin/llama-server
URL=http://127.0.0.1:8012
TSV="$OUT/${GROUP}_$(date +%Y%m%d_%H%M%S).tsv"
printf 'variant\tgen_short\tgen_4k\tgen_12k\tprompt_12k\treps_short/4k/12k\tpeak_GiB\tbank_layers\tspeed_log\n' > "$TSV"
PID=""; STATS=""; SLOG=""

srv_start() {   # NAME "EXTRA SERVER FLAGS"
  local name="$1" extra="$2" stamp; stamp=$(date +%Y%m%d_%H%M%S)
  SLOG="$OUT/J_server_${name}_${stamp}.log"; STATS="$OUT/J_stats_${name}_${stamp}.json"
  echo; echo "### ${GROUP} / server launch ${name}   extra server flags: ${extra:-none}   start $(date +%H:%M:%S)"
  llama --stop >/dev/null 2>&1
  local CMD=("$BIN" -m "$HOME/models/test-model-Q4_K_M.gguf" --alias stuart-creator --host 127.0.0.1 --port 8012 --jinja -np 1 -c 32768 -n -1 --no-context-shift
             -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":true}'
             --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --n-cpu-moe 40 --fit off -lv 4 -t 6 -tb 8 --load-mode none -ub 2048 $extra)
  ( export PELIAS_HOT_SET=$H/pelias_hot_set_b7.5.txt PELIAS_TIER_BYPASS_N=32 PELIAS_TIER_MAX_MIB=7800 PELIAS_TIER_VERBOSE=1 PELIAS_TIER_STATS_FILE="$STATS"
    setsid nohup "${CMD[@]}" > "$SLOG" 2>&1 & echo $! > "$OUT/J_pid.txt" )
  PID=$(cat "$OUT/J_pid.txt")
  for i in $(seq 1 120); do curl -s -m 2 $URL/health | grep -q ok && break; kill -0 "$PID" 2>/dev/null || break; sleep 3; done
  if ! curl -s -m 2 $URL/health | grep -q ok; then
    echo "    FAILED: the server did not become healthy. Last log lines:"; tail -n 5 "$SLOG" | cut -c1-200
    kill -TERM "$PID" 2>/dev/null; return 1
  fi
  ( cd "$CREATOR" && $PY stuart_server_probe.py check --url $URL > /dev/null 2>&1; echo "    check exit $?" )
}

speed_run() {   # VARIANT "SAMPLING FLAGS FOR THE SPEED TOOL"
  local name="$1" samp="$2" rlog blk g0 g4 g12 p12 pk r0 r4 r12
  rlog="$OUT/J_speed_${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "    speed run ${name}: ${samp}"
  ( cd "$CREATOR" && $PY stuart_server_speed.py run --label "J-${name}" --depths 0,4000,12000 --gen 256 --repeats 3 --warmup 2 --url $URL $samp \
      --note "J sampling test ${name}; tier 7.5 GiB set, bypass 32, --load-mode none -ub 2048 -t 6 -tb 8" ) > "$rlog" 2>&1
  blk=$(grep -E '^(short|4000|12000) +[0-9]+ +[0-9.]+ +[0-9.]+ s +[0-9.]+ +\(' "$rlog" | tail -3)
  g0=$(echo "$blk" | awk '$1=="short"{print $6}'); g4=$(echo "$blk" | awk '$1=="4000"{print $6}'); g12=$(echo "$blk" | awk '$1=="12000"{print $6}'); p12=$(echo "$blk" | awk '$1=="12000"{print $3}')
  r0=$(echo "$blk" | awk '$1=="short"{print $8}'); r4=$(echo "$blk" | awk '$1=="4000"{print $8}'); r12=$(echo "$blk" | awk '$1=="12000"{print $8}')
  pk=$(grep 'highest seen' "$rlog" | tail -1 | sed -E 's/.*highest seen ([0-9.]+) GiB.*/\1/')
  if [ -z "$g12" ]; then echo "    FAILED: no speed result. See $rlog"; tail -n 4 "$rlog" | cut -c1-200
    printf '%s\tFAILED\t-\t-\t-\t-\t-\t-\t%s\n' "$name" "$rlog" >> "$TSV"
  else echo "    gen ${g0} / ${g4} / ${g12} tok/s (short/4k/12k)   prompt@12k ${p12} tok/s   reps ${r0} ${r4} ${r12}   peak ${pk} GiB"
    printf '%s\t%s\t%s\t%s\t%s\t%s %s %s\t%s\t%s\t%s\n' "$name" "$g0" "$g4" "$g12" "$p12" "$r0" "$r4" "$r12" "${pk:--}" "@BANK@" "$rlog" >> "$TSV"
  fi
}

srv_stop() {
  local bl
  kill -TERM "$PID" 2>/dev/null     # ONE SIGTERM only: a second one makes llama-server exit at once and skips the stats file
  for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$PID" 2>/dev/null; then echo "    (server still running after 30 s; sending a second signal)"; kill -TERM "$PID" 2>/dev/null; sleep 3; fi
  bl=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('layers_with_bank','?'))" "$STATS" 2>/dev/null || echo "no-stats")
  [ "$bl" = "40" ] || echo "    WARNING: layers_with_bank=${bl}; the bank may not have been active in this launch."
  sed -i "s/@BANK@/${bl}/" "$TSV"
}

if srv_start L1 ""; then
  speed_run base        "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0"
  speed_run greedy      "--temp 0 --top-p 1.0 --top-k 0 --min-p 0.0"
  speed_run no_truncate "--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0"
  speed_run topk40_minp "--temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.05"
  srv_stop
fi
if srv_start L2 "--ignore-eos"; then
  speed_run ignore_eos  "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0"
  srv_stop
fi
if srv_start L3 "--repeat-penalty 1.1 --presence-penalty 0.5 --frequency-penalty 0.3"; then
  speed_run penalties   "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0"
  srv_stop
fi
if srv_start L4 ""; then
  speed_run base_again  "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0"
  srv_stop
fi
echo; echo "=== ${GROUP}: summary (saved in $TSV)"
(column -t -s "$(printf '\t')" "$TSV" 2>/dev/null || cat "$TSV") | cut -c1-200
echo "The server on port 8012 is STOPPED. Start it again with:  llama --serve --stuart-creator --bg"
