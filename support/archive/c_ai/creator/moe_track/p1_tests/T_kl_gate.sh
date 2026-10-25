#!/usr/bin/env bash
# T  Gate H6 accuracy runs (KL divergence against the stock batch-8 reference tier_ref8), 16 chunks of 1024 tokens each.
# STATUS: COMPLETED 2026-10-21/22 (results in logs/moe_trace/kl/ and in the GitHub repo, raw/stage2-tier-cuda/h6/).
# The first four commands below are as recorded in the shell history; tier8_b85 and tier16_b75_by32 are RECONSTRUCTED from their result files (bank file and cap inferred from the names).
# Results are written under labels ending in _rerun so the existing results are not overwritten (KL_SUFFIX=... to change). Needs the server stopped; each run takes several minutes.
GROUP=T_kl_gate; NO_TSV_HEADER=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
SUF="${KL_SUFFIX:-_rerun}"; TB_DIR=$HOME/llama-pelias-tier/build-wsl/bin; KL=<project>/logs/moe_trace/kl
unset PELIAS_HOT_SET PELIAS_TIER_VERBOSE PELIAS_TIER_STATS_FILE PELIAS_TIER_BYPASS_N PELIAS_TIER_MAX_MIB
guard() { pkill -TERM -f llama-server 2>/dev/null; sleep 4; pkill -KILL -f llama-server 2>/dev/null; sleep 2; }
TSV="$OUT/${GROUP}_$(date +%Y%m%d_%H%M%S).tsv"; guard
[ -f "$KL/tier_ref8.base.json" ] || { echo "reference tier_ref8 not found in $KL. Create it first:  $PY sim/kl_gate.py base --label tier_ref8 --extra \"-b 8 -ub 8\""; exit 1; }
kl() {   # LABEL "ENV=VAL ..." "EXTRA FLAGS" tier|stock
  local label="$1${SUF}" envs="$2" extra="$3" which="$4" args
  args=(compare --label "$label" --base tier_ref8)
  [ "$which" = tier ] && args+=(--bin "$TB_DIR")
  [ -n "$extra" ] && args+=(--extra "$extra")
  echo; echo "### $label   env: $envs   extra: $extra   build: $which"
  env $envs $PY sim/kl_gate.py "${args[@]}" 2>&1 | tail -n 3
  $PY - "$KL/${label}_vs_tier_ref8.json" "$label" "$TSV" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    row = "%s\t%.6f\t%.6f\t%.3f\t%.6f / %.6f\t-\t-\t-\t%s\n" % (sys.argv[2], d["mean_kl"], d["mean_kl_stderr"], d["top1_agreement"] * 100, d["ppl_candidate"], d["ppl_reference"], sys.argv[1])
    print("    mean KL %.6f (se %.6f)  top-1 %.3f%%  ppl %.6f / %.6f" % (d["mean_kl"], d["mean_kl_stderr"], d["top1_agreement"] * 100, d["ppl_candidate"], d["ppl_reference"]))
except Exception as e:
    row = "%s\tFAILED\t-\t-\t-\t-\t-\t-\t-\n" % sys.argv[2]; print("    no result file:", e)
open(sys.argv[3], "a").write(row)
PY
}
TX="-b 8 -ub 8 --n-cpu-moe 40 --fit off"
kl tier8        "PELIAS_HOT_SET=$H/pelias_hot_set.txt PELIAS_TIER_BYPASS_N=9 PELIAS_TIER_MAX_MIB=6400 PELIAS_TIER_VERBOSE=1" "$TX" tier
kl tier8_rep    "PELIAS_HOT_SET=$H/pelias_hot_set.txt PELIAS_TIER_BYPASS_N=9 PELIAS_TIER_MAX_MIB=6400 PELIAS_TIER_VERBOSE=1" "$TX" tier
kl tier8_nobank ""                                                                                                          "$TX" tier
kl stock16      ""                                                                                                          "-b 16 -ub 16" stock
kl stock1024    ""                                                                                                          "" stock
kl tier8_b85    "PELIAS_HOT_SET=$H/pelias_hot_set_b8.5.txt PELIAS_TIER_BYPASS_N=9 PELIAS_TIER_MAX_MIB=8800 PELIAS_TIER_VERBOSE=1" "$TX" tier
kl tier16_b75_by32 "PELIAS_HOT_SET=$H/pelias_hot_set_b7.5.txt PELIAS_TIER_BYPASS_N=32 PELIAS_TIER_MAX_MIB=7800 PELIAS_TIER_VERBOSE=1" "-b 16 -ub 16 --n-cpu-moe 40 --fit off" tier
echo; echo "=== ${GROUP} summary (columns: label, mean KL, std. error, top-1 %, perplexity cand/ref)"; cat "$TSV"
