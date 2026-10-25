#!/usr/bin/env bash
# Q  Bypass 32 with the default load and -ub 2048 (-t 6 -tb 8), 7.5 and 8.0 GiB sets.   STATUS: COMPLETED 2026-10-22 14:11 to 14:13 (stamps 141137, 141319). Result: P1 HOLD (prompt reading 921 and 902 tok/s).
GROUP=Q_bypass32_default_load
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant b75_by32 "tier:n=40,bypass=32,cap=7800,hot=$H/pelias_hot_set_b7.5.txt" "-t 6 -tb 8 -ub 2048" "" ""
run_variant b80_by32 "tier:n=40,bypass=32,cap=8300,hot=$H/pelias_hot_set_b8.0.txt" "-t 6 -tb 8 -ub 2048" "" ""
finish
