#!/usr/bin/env bash
# K  Logging overhead. Run: bash <project>/moe_track/p1_tests/K_logging.sh
# fork/gauntlet.py cannot run this test: it checks the log for the line 'pelias-tier: hot bank active', which only prints at -lv 4 with PELIAS_TIER_VERBOSE=1,
# so at a lower log level it stops with "bank not active". This script therefore starts the server itself (same command as the gauntlet's 7.5 GiB, bypass 32,
# --load-mode none, -ub 2048, -t 6 -tb 8 run), speed-tests it with the same tool and settings, and checks the bank afterwards from the tier stats file.
# Variants: lv4 + PELIAS_TIER_VERBOSE=1 (gauntlet default), lv3 + verbose, lv3 + verbose off, lv1 + verbose off.   About 4 x 1.6 min.
GROUP=K_logging; NO_TSV_HEADER=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
CREATOR=<project>
BIN=$HOME/llama-pelias-tier/build-wsl/bin/llama-server
URL=http://127.0.0.1:8012
TSV="$OUT/${GROUP}_$(date +%Y%m%d_%H%M%S).tsv"
printf 'variant\tgen_short\tgen_4k\tgen_12k\tprompt_12k\tpeak_GiB\tbank_layers\tdual_builds\tserver_log\n' > "$TSV"

k_run() {   # NAME LOGLEVEL VERBOSE(1 = PELIAS_TIER_VERBOSE=1, 0 = not set)
  local name="$1" lv="$2" verb="$3" stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local slog="$OUT/K_server_${name}_${stamp}.log" rlog="$OUT/K_speed_${name}_${stamp}.log" stats="$OUT/K_stats_${name}_${stamp}.json"
  echo; echo "### ${GROUP} / ${name}   -lv $lv   PELIAS_TIER_VERBOSE=$([ "$verb" = 1 ] && echo 1 || echo unset)   start $(date +%H:%M:%S)"
  llama --stop >/dev/null 2>&1
  local CMD=("$BIN" -m "$HOME/models/test-model-Q4_K_M.gguf" --alias stuart-creator --host 127.0.0.1 --port 8012 --jinja -np 1 -c 32768 -n -1 --no-context-shift
             -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":true}'
             --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --n-cpu-moe 40 --fit off -lv "$lv" -t 6 -tb 8 --load-mode none -ub 2048)
  ( export PELIAS_HOT_SET=$H/pelias_hot_set_b7.5.txt PELIAS_TIER_BYPASS_N=32 PELIAS_TIER_MAX_MIB=7800 PELIAS_TIER_STATS_FILE="$stats"
    if [ "$verb" = 1 ]; then export PELIAS_TIER_VERBOSE=1; else unset PELIAS_TIER_VERBOSE; fi
    setsid nohup "${CMD[@]}" > "$slog" 2>&1 & echo $! > "$OUT/K_pid.txt" )
  local pid; pid=$(cat "$OUT/K_pid.txt")
  for i in $(seq 1 120); do curl -s -m 2 $URL/health | grep -q ok && break; kill -0 "$pid" 2>/dev/null || break; sleep 3; done
  if ! curl -s -m 2 $URL/health | grep -q ok; then
    echo "    FAILED: the server did not become healthy. Last log lines:"; tail -n 5 "$slog" | cut -c1-200
    printf '%s\tFAILED\t-\t-\t-\t-\t-\t-\t%s\n' "$name" "$slog" >> "$TSV"; kill -TERM "$pid" 2>/dev/null; return
  fi
  ( cd "$CREATOR" && $PY stuart_server_probe.py check --url $URL > /dev/null 2>&1; echo "    check exit $?" )
  ( cd "$CREATOR" && $PY stuart_server_speed.py run --label "K-${name}" --depths 0,4000,12000 --gen 256 --repeats 3 --warmup 2 --url $URL \
      --note "K logging test: -lv $lv, PELIAS_TIER_VERBOSE=$verb; tier 7.5 GiB set, bypass 32, --load-mode none -ub 2048 -t 6 -tb 8" ) > "$rlog" 2>&1
  # ONE SIGTERM only: llama-server forces an immediate exit on a second signal, which skips the stats file and ends in a CUDA teardown error.
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "    (server still running after 30 s; sending a second signal)"; kill -TERM "$pid" 2>/dev/null; sleep 3; fi
  local blk g0 g4 g12 p12 pk bl db
  blk=$(grep -E '^(short|4000|12000) +[0-9]+ +[0-9.]+ +[0-9.]+ s +[0-9.]+ +\(' "$rlog" | tail -3)
  g0=$(echo "$blk" | awk '$1=="short"{print $6}'); g4=$(echo "$blk" | awk '$1=="4000"{print $6}'); g12=$(echo "$blk" | awk '$1=="12000"{print $6}'); p12=$(echo "$blk" | awk '$1=="12000"{print $3}')
  pk=$(grep 'highest seen' "$rlog" | tail -1 | sed -E 's/.*highest seen ([0-9.]+) GiB.*/\1/')
  bl=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('layers_with_bank','?'))" "$stats" 2>/dev/null || echo "no-stats")
  db=$($PY -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('dual_layer_builds_by_n_tokens',{})).replace(' ',''))" "$stats" 2>/dev/null || echo "-")
  if [ -z "$g12" ]; then echo "    FAILED: no speed result. See $rlog"; tail -n 4 "$rlog" | cut -c1-200
  else
    echo "    gen ${g0} / ${g4} / ${g12} tok/s (short/4k/12k)   prompt@12k ${p12} tok/s   peak ${pk} GiB   bank layers ${bl}   dual builds ${db}"
    if [ "$bl" != "40" ]; then echo "    WARNING: the bank was not active (layers_with_bank=${bl}); this run measured the plain build."; fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${g0:-FAILED}" "${g4:--}" "${g12:--}" "${p12:--}" "${pk:--}" "$bl" "$db" "$slog" >> "$TSV"
}

k_run lv4_verbose1 4 1
k_run lv3_verbose1 3 1
k_run lv3_verbose0 3 0
k_run lv1_verbose0 1 0
echo; echo "=== ${GROUP}: summary (saved in $TSV)"
(column -t -s "$(printf '\t')" "$TSV" 2>/dev/null || cat "$TSV")
echo "The server on port 8012 is STOPPED. Start it again with:  llama --serve --stuart-creator --bg"
