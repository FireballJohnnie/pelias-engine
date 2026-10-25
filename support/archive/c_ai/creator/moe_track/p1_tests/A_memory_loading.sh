#!/usr/bin/env bash
# A  Memory loading: default (auto) vs --load-mode none/mmap/mlock/mmap+mlock/dio. (this build has no --no-mmap/--mlock; mlock modes may fail on the WSL memory-lock limit and are then logged as FAILED.)
GROUP=A_memory_loading
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant default_load "$SET75" "-t $T -tb $TB -ub $UB" "" ""
run_variant load_none    "$SET75" "$(base_flags)" "" ""
run_variant lm_mmap      "$SET75" "-t $T -tb $TB -ub $UB --load-mode mmap" "" ""
run_variant lm_mlock     "$SET75" "-t $T -tb $TB -ub $UB --load-mode mlock" "" ""
run_variant lm_mmap_mlock "$SET75" "-t $T -tb $TB -ub $UB --load-mode mmap+mlock" "" ""
run_variant lm_dio       "$SET75" "-t $T -tb $TB -ub $UB --load-mode dio" "" ""
finish
