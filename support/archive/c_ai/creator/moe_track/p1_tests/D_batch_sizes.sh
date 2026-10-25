#!/usr/bin/env bash
# D  Batch sizes: -ub (micro-batch) and -b (logical batch). -ub above -b is clamped by llama.cpp, so the 4096 case sets both.
GROUP=D_batch_sizes
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant base_ub2048     "$SET75" "$(base_flags)" "" ""
run_variant ub512           "$SET75" "-t $T -tb $TB $LM -ub 512" "" ""
run_variant ub1024          "$SET75" "-t $T -tb $TB $LM -ub 1024" "" ""
run_variant b4096_ub2048    "$SET75" "-t $T -tb $TB $LM -b 4096 -ub 2048" "" ""
run_variant b4096_ub4096    "$SET75" "-t $T -tb $TB $LM -b 4096 -ub 4096" "" ""
finish
