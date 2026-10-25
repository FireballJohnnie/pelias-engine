#!/usr/bin/env bash
# L  Speculative decoding with the MTP head in the main GGUF (--spec-type draft-mtp), n_max 2 and 3, plus the gauntlet's MTP acceptance test (5e).
# Whether this tier build loads with MTP is not known; a failed start is logged as FAILED.
GROUP=L_speculative_mtp
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant base "$SET75" "$(base_flags)" "" ""
run_variant mtp2 "$SET75,mtp=2" "$(base_flags)" "" "--tests mtp"
run_variant mtp3 "$SET75,mtp=3" "$(base_flags)" "" "--tests mtp"
finish
