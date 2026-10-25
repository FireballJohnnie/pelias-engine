#!/usr/bin/env bash
# F  KV cache type and flash-attn. Base = q8_0 KV with flash-attn on. Flash-attn off is run with f16 KV (a quantised V cache needs flash-attn).
GROUP=F_kv_attention
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
run_variant base_q8_fa_on "$SET75" "$(base_flags)" "" ""
run_variant kv_f16_fa_on  "$SET75" "$(base_flags) --cache-type-k f16 --cache-type-v f16" "" ""
run_variant kv_f16_fa_off "$SET75" "$(base_flags) --cache-type-k f16 --cache-type-v f16 -fa off" "" ""
finish
