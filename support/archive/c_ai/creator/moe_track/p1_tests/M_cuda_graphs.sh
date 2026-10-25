#!/usr/bin/env bash
# M  CUDA graphs on (base) vs off (GGML_CUDA_DISABLE_GRAPHS=1). The CUDA 13.0 vs 13.2 and arch-flag parts need a rebuild and are not covered here.
GROUP=M_cuda_graphs
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant graphs_on  "$SET75" "$(base_flags)" "" ""
run_variant graphs_off "$SET75" "$(base_flags)" "GGML_CUDA_DISABLE_GRAPHS=1" ""
finish
