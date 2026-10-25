#!/usr/bin/env bash
# O  Patch isolation: does one patch change the speed? Each variant = a COPY of the tier tree with one patch reversed, built in its own folder, then the P1 base cell through the gauntlet.
# The live tree ~/llama-pelias-tier and its build are never touched. Copies live in ~/llama-pelias-tier-iso/<name> and are NOT deleted afterwards (you decide).
# Variants (default list: ctrl nodedup nodiag2):
#   ctrl     copy with nothing reversed. CONTROL: shows whether a rebuilt copy runs at the same speed as the live build. Read the others against it.
#   nodedup  mm_ids_dedup.patch reversed (the helper dedup; whether it is still needed is an open question)
#   nodiag2  mmq_dump_diag2.patch reversed (diagnostic step 2; shows whether the diagnostic code costs speed)
#   nodiag12 diag2, then mmq_dump_diag.patch reversed (both diagnostics)                                           optional
#   noquant  quantize_signed_id.patch reversed. EXPECTED TO CRASH at bypass 32 (it is the crash fix): negative control only.   optional
# NOT scripted: pin_cold_to_cpu, cuda_hole_fix and pelias_tier_v1 do NOT reverse cleanly on their own (probe 2026-10-22: later patches touch the same lines). Isolating them
# needs hand edits to the sources, which is your decision (and none of the loose patches is folded into pelias_tier_v1.patch without your OK).
# Each variant needs a full build in a new folder. Build time is UNMEASURED (earlier incremental tier builds took about 5 to 9 min [DERIVED from log stamps]); a fresh folder is a full compile.
# Run all defaults:   bash <project>/moe_track/p1_tests/O_patch_isolation.sh
# Run chosen ones:    bash <project>/moe_track/p1_tests/O_patch_isolation.sh ctrl nodedup
GROUP=O_patch_isolation
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
LIST="${*:-ctrl nodedup nodiag2}"
LIVE=$HOME/llama-pelias-tier; ISO_ROOT=$HOME/llama-pelias-tier-iso
CREATOR=<project>; BKP=/mnt/c/AI/cl-sndbx-bkp/stuart-creator/2026-10-22/t4_helper_fix
P_DEDUP=$BKP/mm_ids_dedup.patch; P_DIAG=$BKP/mmq_dump_diag.patch; P_DIAG2=$BKP/mmq_dump_diag2.patch; P_QUANT=$BKP/quantize_signed_id.patch
[ -d "$LIVE/.git" ] || { echo "STOP: $LIVE not found"; exit 1; }
for p in $P_DEDUP $P_DIAG $P_DIAG2 $P_QUANT $CREATOR/build_llama_cuda_v2.sh; do [ -f "$p" ] || { echo "STOP: missing $p"; exit 1; }; done
echo "source tree size without builds: $(du -sh --exclude=build-wsl --exclude='build*' "$LIVE" 2>/dev/null | cut -f1)   free disk: $(df -h "$HOME" | awk 'NR==2{print $4}')"

iso_make() {   # NAME PATCH...   (patches reversed in the order given) -> sets ISO ; returns 1 on failure
  local name="$1"; shift; local dst="$ISO_ROOT/$name" p stamp; stamp=$(date +%Y%m%d_%H%M%S); ISO="$dst"
  if [ -x "$dst/build-wsl/bin/llama-server" ]; then echo "    reusing the build already in $dst"; return 0; fi
  if [ -e "$dst" ]; then echo "    STOP: $dst exists without a finished build. Move it away (mv $dst $dst.old) and run again; this script deletes nothing."; return 1; fi
  mkdir -p "$ISO_ROOT" && mkdir "$dst" || return 1
  echo "    copying the tier tree to $dst"
  tar -C "$LIVE" --exclude=./build-wsl -cf - . | tar -C "$dst" -xf - || { echo "    FAILED: copy"; return 1; }
  for p in "$@"; do
    if ( cd "$dst" && git apply -R --check "$p" ) >/dev/null 2>&1; then ( cd "$dst" && git apply -R "$p" ) || { echo "    FAILED: reversing $(basename "$p")"; return 1; }
      echo "    reversed $(basename "$p")"
    else echo "    FAILED: $(basename "$p") does not reverse cleanly in the copy"; return 1; fi
  done
  echo "    sources differ from the live tree in: $(diff -rq "$LIVE/src" "$dst/src" 2>/dev/null | wc -l) src file(s), $(diff -rq "$LIVE/ggml/src/ggml-cuda" "$dst/ggml/src/ggml-cuda" 2>/dev/null | wc -l) cuda file(s)"
  echo "    building (log: $OUT/O_build_${name}_${stamp}.log) ... start $(date +%H:%M:%S)"
  LLAMA_DIR="$dst" BUILD_NAME=build-wsl JOBS="${JOBS:-8}" CUDA_ARCH=89 bash "$CREATOR/build_llama_cuda_v2.sh" > "$OUT/O_build_${name}_${stamp}.log" 2>&1
  if [ ! -x "$dst/build-wsl/bin/llama-server" ]; then echo "    FAILED: no llama-server after the build. Last lines:"; tail -n 6 "$OUT/O_build_${name}_${stamp}.log" | cut -c1-200; return 1; fi
  echo "    built   $(date +%H:%M:%S)"
}

for V in $LIST; do
  case "$V" in
    ctrl)     PATCHES=() ;;
    nodedup)  PATCHES=("$P_DEDUP") ;;
    nodiag2)  PATCHES=("$P_DIAG2") ;;
    nodiag12) PATCHES=("$P_DIAG2" "$P_DIAG") ;;
    noquant)  PATCHES=("$P_QUANT"); echo; echo "NOTE: noquant is a negative control; a crash at bypass 32 is the expected result." ;;
    *) echo "unknown variant '$V' (use: ctrl nodedup nodiag2 nodiag12 noquant)"; continue ;;
  esac
  echo; echo "### ${GROUP} / build $V   start $(date +%H:%M:%S)"
  if iso_make "$V" "${PATCHES[@]}"; then
    run_variant "o_$V" "$SET75" "$(base_flags)" "TIER_SRC=$ISO" ""
  else
    printf 'o_%s\tFAILED (build)\t-\t-\t-\t-\t-\t-\t-\n' "$V" >> "$TSV"
  fi
done
finish
