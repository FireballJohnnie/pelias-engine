#!/usr/bin/env bash
# _lib.sh  shared helper for the P1 test scripts in this folder. Sourced by the letter scripts; not run on its own.
# Base configuration = the one that passed P1 on 2026-10-22: 7.5 GiB hot set, bypass 32, --load-mode none, -ub 2048, -t 6 -tb 8.
# Every variant goes through fork/gauntlet.py: one variant at a time, one thing changed.
# The gauntlet STOPS the llama-server on port 8012 and leaves it stopped when the last variant ends.
HERE_P1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT=<project>/moe_track
cd "$MT" || exit 1
source ./env.sh
export TIER_SRC=$HOME/llama-pelias-tier
H=<project>/logs
OUT="$HERE_P1/results"; mkdir -p "$OUT"
SET75="tier:n=40,bypass=32,cap=7800,hot=$H/pelias_hot_set_b7.5.txt"
SET70="tier:n=40,bypass=32,cap=7300,hot=$H/pelias_hot_set_b7.0.txt"
T=6; TB=8; UB=2048; LM="--load-mode none"
TSV="$OUT/${GROUP}.tsv"
[ -n "$NO_TSV_HEADER" ] || [ -s "$TSV" ] || printf 'variant\tgen_short\tgen_4k\tgen_12k\tprompt_12k\tpeak_GiB\tP1\tG3\tlog\n' > "$TSV"

base_flags() { echo "-t $T -tb $TB $LM -ub $UB"; }

# run_variant NAME SPEC "EXTRA FLAGS" "ENV=VAL ENV2=VAL" "gauntlet options"
run_variant() {
  local name="$1" spec="$2" flags="$3" envs="$4" gopts="$5"
  local log="$OUT/${GROUP}_${name}_$(date +%Y%m%d_%H%M%S).log"
  echo; echo "### ${GROUP} / ${name}   start $(date +%H:%M:%S)"
  echo "    spec : $spec"
  echo "    flags: $flags"
  if [ -n "$envs" ]; then echo "    env  : $envs"; fi
  env $envs $PY fork/gauntlet.py run "$spec" --extra "$flags" $gopts --keep-going > "$log" 2>&1
  local blk g0 g4 g12 p12 pk p1 g3
  blk=$(grep -E '^(short|4000|12000) +[0-9]+ +[0-9.]+ +[0-9.]+ s +[0-9.]+ +\(' "$log" | tail -3)
  g0=$(echo "$blk" | awk '$1=="short"{print $6}')
  g4=$(echo "$blk" | awk '$1=="4000"{print $6}')
  g12=$(echo "$blk" | awk '$1=="12000"{print $6}')
  p12=$(echo "$blk" | awk '$1=="12000"{print $3}')
  pk=$(grep 'highest seen' "$log" | tail -1 | sed -E 's/.*highest seen ([0-9.]+) GiB.*/\1/')
  p1=$(grep '\[gate P1\]' "$log" | tail -1 | awk '{print $3}')
  g3=$(grep '\[gate G3\]' "$log" | tail -1 | awk '{print $3}')
  if [ -z "$g12" ]; then
    echo "    FAILED, no speed result. Reason from the gauntlet log:"
    (grep -E "STOP:|FAILED|does not list|did not start|bank not active|this binary" "$log" | head -n 4; echo "    status table line:"; grep -E "^${name}|label" "$log" | tail -n 1) | cut -c1-220
    echo "    full log: $log"
    printf '%s\tFAILED\t-\t-\t-\t-\t-\t-\t%s\n' "$name" "$log" >> "$TSV"
  else
    echo "    gen ${g0} / ${g4} / ${g12} tok/s (short/4k/12k)   prompt@12k ${p12} tok/s   peak ${pk} GiB   P1 ${p1}   G3 ${g3}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$g0" "$g4" "$g12" "$p12" "$pk" "$p1" "$g3" "$log" >> "$TSV"
  fi
}

finish() {
  echo; echo "=== ${GROUP}: summary (saved in $TSV; full logs in $OUT)"
  (column -t -s "$(printf '\t')" "$TSV" 2>/dev/null || cat "$TSV")
  if type llama >/dev/null 2>&1; then llama --status; fi
  echo "The server on port 8012 is STOPPED. Start it again with:  llama --serve --stuart-creator --bg"
}
