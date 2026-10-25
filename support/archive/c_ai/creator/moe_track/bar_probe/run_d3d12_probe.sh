#!/usr/bin/env bash
# run_d3d12_probe.sh - builds and runs d3d12_bar_probe.cpp (D3D12 shared buffer -> CUDA import test) in WSL2.
# Run from WSL:  bash <project>/moe_track/bar_probe/run_d3d12_probe.sh
# Needs: g++, git (network to github.com once, for the header-only DirectX-Headers; no sudo, nothing installed system-wide).
# Uses 5 heap kinds x 2 paths x 64 MiB for a few seconds, plus CPU store timing (loads one core + PCIe briefly):
# refuses to run while a test is running (FORCE=1 overrides).  Output: results/d3d12_bar_probe_<stamp>.txt
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/results" "$HERE/third_party"
OUT="$HERE/results/d3d12_bar_probe_$(date +%Y%m%d_%H%M%S).txt"
SHA=461ef5138bd30922b44fbad77c4e47662812d2d5    # DirectX-Headers commit the probe was compile-checked against
DXH="$HERE/third_party/DirectX-Headers"

if [ "${FORCE:-0}" != 1 ] && pgrep -f 'W_bank70_extras|X_b70_mtp2_repeat|gauntlet\.py|llama-server|llama-bench|_lib\.sh' >/dev/null 2>&1; then
  echo "A test looks like it is still running. Wait for it, or run: FORCE=1 bash $0"; exit 1
fi

if [ ! -f "$DXH/include/directx/d3d12.h" ]; then
  echo "fetching DirectX-Headers ($SHA) into third_party/ ..."
  rm -rf "$DXH"; mkdir -p "$DXH" && cd "$DXH" && git init -q && git remote add origin https://github.com/microsoft/DirectX-Headers.git \
    && git fetch -q --depth 1 origin "$SHA" && git checkout -q FETCH_HEAD \
    || { echo "FETCH FAILED (network?)"; exit 1; }
  cd "$HERE"
fi

{
echo "## d3d12_bar_probe $(date -Is)  kernel $(uname -r)  headers $(git -C "$DXH" rev-parse --short HEAD 2>/dev/null)"
if g++ -std=c++17 -O1 -mavx2 -Wall -I"$DXH/include" -I"$DXH/include/wsl/stubs" -o "$HERE/d3d12_bar_probe" "$HERE/d3d12_bar_probe.cpp" -ldl; then
  timeout 180 "$HERE/d3d12_bar_probe"; echo "exit code $?"
else
  echo "BUILD FAILED"
fi
echo "## end"
} 2>&1 | tee "$OUT"
echo "saved: $OUT"
