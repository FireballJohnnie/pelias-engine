#!/usr/bin/env bash
# E  Layout: --n-cpu-moe with the bank (tier n=40, 39, 38). Uses the 7.0 GiB set for VRAM headroom: each layer moved to the GPU adds its experts.
# A start that runs out of VRAM is logged as FAILED. Whether the bank still covers the layers moved to the GPU is not known; read the 'hot bank active' line in the log.
GROUP=E_layout
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
for N in 40 39 38; do
  run_variant n$N "tier:n=$N,bypass=32,cap=7300,hot=$H/pelias_hot_set_b7.0.txt" "$(base_flags)" "" ""
done
finish
