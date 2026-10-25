#!/usr/bin/env bash
# N  Pinned host memory on (base) vs off (GGML_CUDA_NO_PINNED=1). If this build does not read that variable the two runs will simply match.
# ReBAR and PCIe settings cannot be changed from here.
GROUP=N_pinned_memory
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant pinned_on  "$SET75" "$(base_flags)" "" ""
run_variant pinned_off "$SET75" "$(base_flags)" "GGML_CUDA_NO_PINNED=1" ""
finish
