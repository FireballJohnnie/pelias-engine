#!/usr/bin/env bash
# C  Threads: -t (generation), -tb (batch), --poll, --prio on the tier. Base is -t 6 -tb 8.
GROUP=C_threads
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant base_t6_tb8 "$SET75" "$(base_flags)" "" ""
run_variant t5_tb8      "$SET75" "-t 5 -tb 8 $LM -ub $UB" "" ""
run_variant t7_tb8      "$SET75" "-t 7 -tb 8 $LM -ub $UB" "" ""
run_variant t6_tb6      "$SET75" "-t 6 -tb 6 $LM -ub $UB" "" ""
run_variant t6_tb10     "$SET75" "-t 6 -tb 10 $LM -ub $UB" "" ""
run_variant t6_tb12     "$SET75" "-t 6 -tb 12 $LM -ub $UB" "" ""
run_variant poll0       "$SET75" "$(base_flags) --poll 0" "" ""
run_variant poll100     "$SET75" "$(base_flags) --poll 100" "" ""
run_variant prio2       "$SET75" "$(base_flags) --prio 2" "" ""
finish
