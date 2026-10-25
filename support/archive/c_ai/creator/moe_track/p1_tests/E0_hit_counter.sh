#!/usr/bin/env bash
# E0  Live hit-rate counter on the P1 reference cell (7.0 GiB bank, -ub 4096).  Author request 2026-10-22.
# What it does: makes a COPY of the live tier tree (~/llama-pelias-tier-iso/e0hits), applies moe_track/e0/e0_hit_counter.patch to the copy, builds it,
# then runs the gauntlet 3 times on the reference cell with PELIAS_TIER_HITS=1. The tier writes per-layer hit counts into its stats file at server exit;
# this script copies each run's stats file next to the result TSV and prints the hit rates.
# READ THIS BEFORE USING THE NUMBERS:
#   - The counting callback stops the scheduler after each of the 40 id nodes per token, so the SPEED columns of this run are NOT valid. Use them for nothing.
#   - The hit rate is measured on the gauntlet's own prompts (short / 4k / 12k generations), not on the trace domains the hot set was chosen from.
#   - The simulated held-out hit rate for the production 7.5 GiB set is 0.827; the 7.0 GiB set's simulated rate is in the project notes, not re-derived here.
# The live tree and its build are never touched. Nothing is deleted. Needs the server on port 8012 stopped (llama --stop).
# Run:  bash <project>/moe_track/p1_tests/E0_hit_counter.sh      (about 10 min build + 3 x about 3 min [ESTIMATE, from O and W timestamps])
GROUP=E0_hit_counter
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
UB4="-t $T -tb $TB $LM -b 4096 -ub 4096"
LIVE=$HOME/llama-pelias-tier; ISO=$HOME/llama-pelias-tier-iso/e0hits
CREATOR=<project>; PATCH=$CREATOR/moe_track/e0/e0_hit_counter.patch
FORKS=$CREATOR/logs/forks
[ -d "$LIVE/.git" ] || { echo "STOP: $LIVE not found"; exit 1; }
[ -f "$PATCH" ] || { echo "STOP: missing $PATCH"; exit 1; }
if [ ! -x "$ISO/build-wsl/bin/llama-server" ]; then
  if [ -e "$ISO" ]; then echo "STOP: $ISO exists without a finished build. mv $ISO $ISO.old and run again (this script deletes nothing)."; exit 1; fi
  mkdir -p "$(dirname "$ISO")" && mkdir "$ISO" || exit 1
  echo "copying the tier tree to $ISO"
  tar -C "$LIVE" --exclude=./build-wsl -cf - . | tar -C "$ISO" -xf - || { echo "FAILED: copy"; exit 1; }
  ( cd "$ISO" && git apply --check "$PATCH" ) || { echo "FAILED: the patch does not apply to the copy"; exit 1; }
  ( cd "$ISO" && git apply "$PATCH" ) || { echo "FAILED: applying the patch"; exit 1; }
  echo "patch applied; building (log: $OUT/E0_build_$(date +%Y%m%d_%H%M%S).log) start $(date +%H:%M:%S)"
  CCACHE_DISABLE=1 LLAMA_DIR="$ISO" BUILD_NAME=build-wsl JOBS="${JOBS:-8}" CUDA_ARCH=89 bash "$CREATOR/build_llama_cuda_v2.sh" > "$OUT/E0_build_$(date +%Y%m%d_%H%M%S).log" 2>&1
  [ -x "$ISO/build-wsl/bin/llama-server" ] || { echo "FAILED: no llama-server after the build; see the E0_build log in $OUT"; exit 1; }
  echo "built $(date +%H:%M:%S)"
fi
for R in 1 2 3; do
  STAMP=$(date +%s)
  run_variant e0hits_r$R "$SET70" "$UB4" "TIER_SRC=$ISO PELIAS_TIER_HITS=1" ""
  sleep 3
  F=$(find "$FORKS" -name tier_stats.json -newermt "@$STAMP" 2>/dev/null | head -1)
  if [ -n "$F" ]; then
    cp "$F" "$OUT/${GROUP}_r${R}_tier_stats.json"
    echo "    hit counts: $OUT/${GROUP}_r${R}_tier_stats.json"
    $PY - "$F" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
print("    hit_rate_1token %.4f   hit_rate_2to31tokens %s" % (d.get("hit_rate_1token",-1), d.get("hit_rate_2to31tokens","n/a")))
hl=d.get("hit_counts_by_layer",{})
if hl:
    r=sorted(((v[0]/v[1] if v[1] else -1, int(k)) for k,v in hl.items()))
    print("    layers counted %d; lowest 1-token hit rates: %s" % (len(hl), ", ".join("L%d %.3f"%(k,x) for x,k in r[:4])))
    print("    highest: %s" % ", ".join("L%d %.3f"%(k,x) for x,k in r[-4:]))
PYEOF
  else
    echo "    NO tier_stats.json newer than the start of this run (the server may not have exited cleanly)."
  fi
done
finish
echo "REMINDER: the speed columns above are invalid for this series (the counting callback slows decode). Only the hit counts are the result."
