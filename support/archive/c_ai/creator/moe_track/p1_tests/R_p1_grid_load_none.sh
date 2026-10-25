#!/usr/bin/env bash
# R  The P1 grid under --load-mode none: bank 7.0 / 7.5 / 8.0 GiB x bypass 32 / 9, -ub 2048 -t 6 -tb 8, plus 7.5 GiB at bypass 9 without -ub.
# STATUS: COMPLETED 2026-10-22 13:26 to 15:11 (stamps 132648, 132907, 143932, 144107, 150241, 150419, 150951, 151123). Result: P1 PASS in all six cells, G3 FAIL in all.
GROUP=R_p1_grid_load_none
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
for BY in 32 9; do
  for CELL in 7.0:7300 7.5:7800 8.0:8300; do
    GIB=${CELL%%:*}; CAP=${CELL##*:}
    run_variant "b${GIB}_by${BY}" "tier:n=40,bypass=$BY,cap=$CAP,hot=$H/pelias_hot_set_b${GIB}.txt" "$(base_flags)" "" ""
  done
done
run_variant b7.5_by9_lmnone_ubdefault "tier:n=40,bypass=9,cap=7800,hot=$H/pelias_hot_set_b7.5.txt" "-t $T -tb $TB $LM" "" ""
finish
