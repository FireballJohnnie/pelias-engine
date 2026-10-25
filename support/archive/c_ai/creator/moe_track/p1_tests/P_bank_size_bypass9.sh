#!/usr/bin/env bash
# P  Hot-set size sweep at bypass 9, default load, default -ub, -t 6 -tb 8.   STATUS: COMPLETED 2026-10-22 13:00 to 13:16 (stamps 125811, 130522, 130854, 131225, 131550; 6.1 GiB also at 114018).
# This script repeats it; no need to run it again unless you want the same-session comparison. Same commands as the original runs.
GROUP=P_bank_size_bypass9
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant b61 "tier:n=40,bypass=9,cap=6400" "-t 6 -tb 8" "" ""
run_variant b70 "tier:n=40,bypass=9,cap=7300,hot=$H/pelias_hot_set_b7.0.txt" "-t 6 -tb 8" "" ""
run_variant b75 "tier:n=40,bypass=9,cap=7800,hot=$H/pelias_hot_set_b7.5.txt" "-t 6 -tb 8" "" ""
run_variant b80 "tier:n=40,bypass=9,cap=8300,hot=$H/pelias_hot_set_b8.0.txt" "-t 6 -tb 8" "" ""
run_variant b85 "tier:n=40,bypass=9,cap=8800,hot=$H/pelias_hot_set_b8.5.txt" "-t 6 -tb 8" "" ""
finish
