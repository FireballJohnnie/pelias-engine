#!/usr/bin/env bash
# probe_O  READ-ONLY. Gathers what is needed to write the O script (isolating each patch). Changes nothing: git apply -R --check only tests.
# Run: bash <project>/moe_track/p1_tests/probes/probe_O.sh
T=$HOME/llama-pelias-tier; B=/mnt/c/AI/cl-sndbx-bkp/stuart-creator; cd "$T" || exit 1
echo "== git"; git log --oneline -3 | cut -c1-120; git status --short | head -n 30; git diff --stat | tail -n 12
echo; echo "== can each patch be reversed alone on the current tree? (test only)"
for p in $B/2026-10-21/finish3_pin_fix/pin_cold_to_cpu.patch $B/2026-10-21/finish3_cuda_hole_fix/cuda_hole_fix.patch \
         $B/2026-10-22/t4_helper_fix/mm_ids_dedup.patch $B/2026-10-22/t4_helper_fix/mmq_dump_diag.patch \
         $B/2026-10-22/t4_helper_fix/mmq_dump_diag2.patch $B/2026-10-22/t4_helper_fix/quantize_signed_id.patch \
         <project>/moe_track/patches/pelias_tier_v1.patch; do
  printf '%-26s ' "$(basename $p)"
  if git apply -R --check "$p" >/dev/null 2>&1; then echo "reverses cleanly"; else echo "does NOT reverse cleanly on its own"; fi
done
echo; echo "== build directories and binaries"; ls -d build* 2>/dev/null
ls -la --time-style=long-iso build-wsl/bin/llama-server build-wsl/bin/libggml-cuda.so.0 2>/dev/null | cut -c1-160
echo; echo "== CMake cache (build-wsl)"
grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CMAKE_CUDA_FLAGS|GGML_CUDA[A-Z_0-9]*|GGML_NATIVE|GGML_CCACHE|CMAKE_CXX_FLAGS):" build-wsl/CMakeCache.txt 2>/dev/null | cut -c1-170
echo; echo "== CUDA toolkit"; command -v nvcc; nvcc --version 2>&1 | tail -n 3
echo; echo "== resources"; df -h "$HOME" | tail -n 1; free -g | head -n 2; echo "cores: $(nproc)"
echo; echo "== earlier build logs (time taken)"; ls -la --time-style=long-iso $HOME/p1/build_*.log 2>/dev/null | cut -c1-160 | tail -n 5
l=$(ls -t $HOME/p1/build_*.log 2>/dev/null | head -n 1); [ -n "$l" ] && { echo "$l"; tail -n 4 "$l" | cut -c1-160; }
