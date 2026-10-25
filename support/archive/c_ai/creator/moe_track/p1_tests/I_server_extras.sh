#!/usr/bin/env bash
# I  Server extras: context checkpoints (--ctx-checkpoints, the log shows 32 by default) and the prompt-reuse test (5d).
GROUP=I_server_extras
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant base      "$SET75" "$(base_flags)" "" ""
run_variant ctxcp2    "$SET75,ctxcp=2" "$(base_flags)" "" ""
run_variant ctxcp8    "$SET75,ctxcp=8" "$(base_flags)" "" ""
run_variant reuse_5d  "$SET75,reuse=1" "$(base_flags)" "" "--tests reuse"
finish
