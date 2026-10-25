#!/usr/bin/env bash
# E0  Per-layer timeline + per-token latency on the P1 reference cell (7.0 GiB bank, -ub 4096).  Author request 2026-10-22 ("both").
# What it does: makes a COPY of the live tier tree (~/llama-pelias-tier-iso/e0time), applies moe_track/e0/e0_timeline.patch (timer: host stamps around every
# scheduler split, CUDA events around every GPU graph call), builds it, then starts the server three times from the gauntlet's own launch command and
# streams one short and one 12k-context generation (300 tokens each) through moe_track/e0/token_latency.py:
#   A  timer OFF, CUDA graphs on   -> the undisturbed per-token latency (what the timer costs = B minus A)
#   B  timer ON,  CUDA graphs on   -> the realistic timeline
#   C  timer ON,  CUDA graphs off  -> cross-check (graphs off is slower, see M)
# Files: results/E0_timeline/<stamp>/{A_off,B_graphs_on,C_graphs_off}_tokens.tsv + _summary.txt, B_ and C_ .host.tsv / .gpu.tsv (the timeline), server logs.
# Send those files back (the .host.tsv / .gpu.tsv are the large ones, a few MB each [UNMEASURED]).
# The live tree and its build are never touched. Nothing is deleted. Stops the server on port 8012 (llama --stop) and leaves it stopped.
# Run:  bash <project>/moe_track/p1_tests/E0_timeline.sh     (about 10 min build + 3 x about 4 min [ESTIMATE])
GROUP=E0_timeline
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
UB4="-t $T -tb $TB $LM -b 4096 -ub 4096"
LIVE=$HOME/llama-pelias-tier; ISO=$HOME/llama-pelias-tier-iso/e0time
CREATOR=<project>; E0=$CREATOR/moe_track/e0
PATCH=$E0/e0_timeline.patch; CLIENT=$E0/token_latency.py; FORKS=$CREATOR/logs/forks
for f in "$PATCH" "$CLIENT"; do [ -f "$f" ] || { echo "STOP: missing $f"; exit 1; }; done
[ -d "$LIVE/.git" ] || { echo "STOP: $LIVE not found"; exit 1; }
if [ ! -x "$ISO/build-wsl/bin/llama-server" ]; then
  if [ -e "$ISO" ]; then echo "STOP: $ISO exists without a finished build. mv $ISO $ISO.old and run again (this script deletes nothing)."; exit 1; fi
  mkdir -p "$(dirname "$ISO")" && mkdir "$ISO" || exit 1
  echo "copying the tier tree to $ISO"
  tar -C "$LIVE" --exclude=./build-wsl -cf - . | tar -C "$ISO" -xf - || { echo "FAILED: copy"; exit 1; }
  ( cd "$ISO" && git apply --check "$PATCH" ) || { echo "FAILED: the patch does not apply to the copy"; exit 1; }
  ( cd "$ISO" && git apply "$PATCH" ) || { echo "FAILED: applying the patch"; exit 1; }
  echo "patch applied; building (log: $OUT/E0_timeline_build_$(date +%Y%m%d_%H%M%S).log) start $(date +%H:%M:%S)"
  CCACHE_DISABLE=1 LLAMA_DIR="$ISO" BUILD_NAME=build-wsl JOBS="${JOBS:-8}" CUDA_ARCH=89 bash "$CREATOR/build_llama_cuda_v2.sh" > "$OUT/E0_timeline_build_$(date +%Y%m%d_%H%M%S).log" 2>&1
  [ -x "$ISO/build-wsl/bin/llama-server" ] || { echo "FAILED: no llama-server after the build; the CUDA part was compiled for the first time here. Send the E0_timeline_build log from $OUT"; exit 1; }
  echo "built $(date +%H:%M:%S)"
fi
type llama >/dev/null 2>&1 && llama --stop
STAMP=$(date +%s)
echo "writing the gauntlet launch command for the reference cell (dry run, starts nothing)"
env TIER_SRC="$ISO" $PY fork/gauntlet.py run "$SET70" --extra "$UB4" --dry-run > "$OUT/E0_timeline_dryrun.txt" 2>&1
LJ=$(find "$FORKS" -name launch.json -newermt "@$STAMP" 2>/dev/null | head -1)
[ -n "$LJ" ] || { echo "FAILED: no launch.json written; see $OUT/E0_timeline_dryrun.txt"; exit 1; }
grep -q "$ISO" "$LJ" || { echo "STOP: launch.json does not point at $ISO/build-wsl; see $LJ"; exit 1; }
D="$OUT/E0_timeline/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$D"; cp "$LJ" "$D/launch.json"
run_cell() {   # TAG  ENV...
  local tag="$1"; shift
  echo; echo "### ${GROUP} / ${tag}   start $(date +%H:%M:%S)"
  local envargs=(); for e in "$@"; do envargs+=(--env "$e"); done
  $PY "$CLIENT" --launch-json "$LJ" --tag "$tag" --out "$D" "${envargs[@]}" --ctx 0,12000 --gen 300
  sleep 5
}
run_cell A_off
run_cell B_graphs_on  "PELIAS_TIMELINE=$D/B_graphs_on"
run_cell C_graphs_off "PELIAS_TIMELINE=$D/C_graphs_off" "GGML_CUDA_DISABLE_GRAPHS=1"
echo; echo "=== ${GROUP}: summaries"; for t in A_off B_graphs_on C_graphs_off; do echo "--- $t"; cat "$D/${t}_summary.txt" 2>/dev/null; done
echo; ls -l "$D"
echo "The server on port 8012 is STOPPED. Send me the folder: $D"
