#!/usr/bin/env bash
# G  Scheduler op-offload: --no-op-offload (if the binary does not know it, that variant is logged as FAILED) and the GGML_OP_OFFLOAD_MIN_BATCH threshold.
GROUP=G_scheduler_offload
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant base          "$SET75" "$(base_flags)" "" ""
run_variant no_op_offload "$SET75" "$(base_flags) --no-op-offload" "" ""
# GGML_OP_OFFLOAD_MIN_BATCH appears in your shell history (line 1950), so this build reads it; 1000000 = never offload batch ops to the GPU, 16 = offload from 16 tokens.
run_variant offload_min_1000000 "$SET75" "$(base_flags)" "GGML_OP_OFFLOAD_MIN_BATCH=1000000" ""
run_variant offload_min_16 "$SET75" "$(base_flags)" "GGML_OP_OFFLOAD_MIN_BATCH=16" ""
finish
