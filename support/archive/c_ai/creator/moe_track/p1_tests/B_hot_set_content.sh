#!/usr/bin/env bash
# B  Hot-bank content: does the trace mix the hot set was built from change the speed? (7.5 GiB bank, bypass 32, --load-mode none -ub 2048 -t 6 -tb 8)
# Step 1 builds five 7.5 GiB hot sets from the finished trace shards with sim/export_hot_set.py: all (rebuilt control; a test on 2026-10-22 showed it differs slightly from the existing b7.5 set: 4198 vs 4170 experts), persona, agent, code, reasoning.
#        The sets go to results/B_hot_sets/ only. Nothing in logs/ is changed. Printed hit rates are SIMULATED from the traces (in-sample and held-out), not measured on the server.
# Step 2 runs the existing production b7.5 set and then each rebuilt set through the gauntlet (about 1.6 min per set, 6 runs) with the speed tool's normal prompts. Its filler text has no measured hit rate.
# NOT scripted here (needs a patch or new code, your decision): a live hit-rate counter in the tier, and static vs LRU banks.
# Run: bash <project>/moe_track/p1_tests/B_hot_set_content.sh
GROUP=B_hot_set_content
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
TR=<project>/logs/moe_trace; HS="$OUT/B_hot_sets"; mkdir -p "$HS"
HSTSV="$OUT/B_hot_sets.tsv"
[ -s "$HSTSV" ] || printf 'mix\tshards\tdecode_tokens_note\tsimulated_hit_in_sample\tsimulated_hit_held_out\tfile\n' > "$HSTSV"
ls "$TR"/*.pltrace >/dev/null 2>&1 || { echo "STOP: no trace shards in $TR"; exit 1; }

mk() {   # NAME DOMAIN...   -> $HS/hot_NAME_b7.5.txt ; prints the simulated hit rates
  local name="$1"; shift; local files=() d f out res ins held
  for d in "$@"; do for f in "$TR"/${d}_p*_s*.pltrace; do [ -e "${f%.pltrace}.done" ] && files+=("$f"); done; done
  out="$HS/hot_${name}_b7.5.txt"
  echo; echo "### build hot set: $name   (${#files[@]} finished shards: $*)"
  if [ "${#files[@]}" -eq 0 ]; then echo "    FAILED: no finished shards"; printf '%s\t0\t-\t-\t-\tFAILED\n' "$name" >> "$HSTSV"; return 1; fi
  res=$($PY sim/export_hot_set.py --budget-gib 7.5 --out "$out" "${files[@]}" 2>&1 | tail -n 4)
  echo "$res" | cut -c1-200
  ins=$(echo "$res" | sed -nE 's/.*in-sample ([0-9.]+).*/\1/p' | head -n 1); held=$(echo "$res" | sed -nE 's/.*held-out ([0-9.]+).*/\1/p' | head -n 1)
  [ -s "$out" ] || { echo "    FAILED: no hot-set file written"; printf '%s\t%s\t-\t-\t-\tFAILED\n' "$name" "${#files[@]}" >> "$HSTSV"; return 1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${#files[@]}" "$(sed -n 's/^# source: //p' "$out" | head -n 1)" "${ins:--}" "${held:--}" "$out" >> "$HSTSV"
}
mk all persona agent code reasoning
if diff <(grep -v '^#' "$HS/hot_all_b7.5.txt" 2>/dev/null) <(grep -v '^#' "$H/pelias_hot_set_b7.5.txt") >/dev/null 2>&1; then
  echo "    control: the rebuilt 'all' set is IDENTICAL to logs/pelias_hot_set_b7.5.txt (expert lists compared)"
else
  echo "    control: the rebuilt 'all' set DIFFERS from logs/pelias_hot_set_b7.5.txt (expert lists compared); the 'all' run below uses the rebuilt one"
fi
mk persona   persona
mk agent     agent
mk code      code
mk reasoning reasoning

run_variant hs_existing_b75 "$SET75" "$(base_flags)" "" ""   # the production 7.5 GiB set (4170 experts); the rebuilt sets use the current trace tool and size file (all = 4198 experts), so they are not byte-identical to it
for NAME in all persona agent code reasoning; do
  [ -s "$HS/hot_${NAME}_b7.5.txt" ] || { echo; echo "### skip $NAME (no hot-set file)"; continue; }
  run_variant "hs_$NAME" "tier:n=40,bypass=32,cap=7800,hot=$HS/hot_${NAME}_b7.5.txt" "$(base_flags)" "" ""
done
echo; echo "=== hot sets built (simulated hit rates): $HSTSV"; (column -t -s "$(printf '\t')" "$HSTSV" 2>/dev/null || cat "$HSTSV") | cut -c1-200
finish
