#!/usr/bin/env bash
# build_llama_cuda.sh 1.0.0   (new file; run it in WSL)
#
# Builds llama-server, llama-quantize and llama-perplexity WITH CUDA for the RTX 4070 (compute capability 8.9) into  <llama.cpp>/build-wsl .
# Nothing else is touched: the existing build/ folder (CPU only, Debug) stays as it is, and nothing outside build-wsl is written.
#
# Why this exists: the plain `cmake -DGGML_CUDA=ON` stopped with "Unable to find cuda_runtime.h in /usr/include / Unable to find cudart
# library".  CMake looked for a system CUDA toolkit; yours is the one under  ~/stuart_creator_engine/cuda_shim  (built from the pip wheels
# for the flash-attn build), so it has to be told where the toolkit root is.  This script looks for a folder that has all three of
# include/cuda_runtime.h, a libcudart and an nvcc, prints what it found for every place it tried, and only then runs CMake.
#
#   bash build_llama_cuda.sh                 find the toolkit, configure, build
#   DRY_RUN=1 bash build_llama_cuda.sh       find the toolkit and print the commands, build nothing
#
# Optional overrides (environment variables):
#   LLAMA_DIR=<llama.cpp>   BUILD_NAME=build-wsl   CUDA_ARCH=89   JOBS=8
#   CUDA_ROOT=/some/toolkit/root         use exactly this toolkit root
#   ALLOW_UNSUPPORTED=1                  add -allow-unsupported-compiler (only if nvcc rejects the g++ version)
#   FA_ALL_QUANTS=1                      compile flash-attention for every KV-cache type pair (long compile; only if a q8_0 KV cache errors)
set -u

LLAMA="${LLAMA_DIR:-<llama.cpp>}"
NAME="${BUILD_NAME:-build-wsl}"
ARCH="${CUDA_ARCH:-89}"
JOBS="${JOBS:-8}"
DRY="${DRY_RUN:-0}"

say() { printf '%s\n' "$*"; }
die() { printf 'STOP: %s\n' "$*" >&2; exit 2; }

[ -f "$LLAMA/CMakeLists.txt" ] || die "$LLAMA/CMakeLists.txt not found (set LLAMA_DIR)"
if ! command -v cmake >/dev/null 2>&1; then
    [ "$DRY" = "1" ] || die "cmake is not installed (sudo apt install cmake)"
    say "(cmake not found; continuing because DRY_RUN=1)"
fi

# ---- one candidate: print "header | libcudart | nvcc" for it ---------------------------------------------------------------
header_of() {   # a folder that holds cuda_runtime.h
    local r="$1" d
    for d in "$r/include" "$r/targets/x86_64-linux/include"; do
        [ -f "$d/cuda_runtime.h" ] && { printf '%s' "$d"; return 0; }
    done
    return 1
}
cudart_of() {   # an unversioned libcudart.so (what CMake links), or failing that the versioned one (reported, not enough)
    local r="$1" d f
    for d in "$r/lib64" "$r/lib" "$r/lib/x86_64-linux-gnu" "$r/targets/x86_64-linux/lib"; do
        for f in "$d/libcudart.so" "$d/libcudart_static.a"; do
            [ -e "$f" ] && { printf '%s' "$f"; return 0; }
        done
    done
    return 1
}
cudart_versioned_of() {
    local r="$1" d f
    for d in "$r/lib64" "$r/lib" "$r/lib/x86_64-linux-gnu" "$r/targets/x86_64-linux/lib"; do
        for f in "$d"/libcudart.so.*; do
            [ -e "$f" ] && { printf '%s' "$f"; return 0; }
        done
    done
    return 1
}
nvcc_of() {
    local r="$1"
    [ -x "$r/bin/nvcc" ] && { printf '%s' "$r/bin/nvcc"; return 0; }
    return 1
}

# ---- the places to try, in order --------------------------------------------------------------------------------------------
cands=()
add() { [ -n "${1:-}" ] && [ -d "$1" ] && cands+=("$1"); return 0; }
add "${CUDA_ROOT:-}"
if [ -z "${CUDA_ROOT:-}" ]; then
    add "${CUDA_HOME:-}"
    add "${CUDA_PATH:-}"
    add "$HOME/stuart_creator_engine/cuda_shim"
    add /usr/local/cuda
    add /usr/local/cuda-13.0
    add /usr/local/cuda-13
    if NV="$(command -v nvcc 2>/dev/null)"; then
        add "$(dirname "$(dirname "$NV")")"
        add "$(dirname "$(dirname "$(readlink -f "$NV")")")"
    fi
    for d in "$HOME"/stuart_creator_engine/venv/lib/python3*/site-packages/nvidia/cu13 \
             "$HOME"/stuart_creator_engine/venv/lib/python3*/site-packages/nvidia/cuda_runtime \
             "$HOME"/stuart_creator_engine/venv/lib/python3*/site-packages/nvidia/cuda_nvcc; do
        add "$d"
    done
    add /usr
fi

ROOT=""; NVCC=""; seen=" "
say "Looking for a CUDA toolkit root (needs include/cuda_runtime.h, a libcudart and an nvcc):"
for c in "${cands[@]:-}"; do
    [ -n "$c" ] || continue
    real="$(readlink -f "$c" 2>/dev/null || printf '%s' "$c")"
    case "$seen" in *" $real "*) continue ;; esac
    seen="$seen$real "
    h="$(header_of "$c")" || h=""
    l="$(cudart_of "$c")" || l=""
    lv="$(cudart_versioned_of "$c")" || lv=""
    n="$(nvcc_of "$c")" || n=""
    if [ -n "$l" ]; then ltxt="$l"; elif [ -n "$lv" ]; then ltxt="only the versioned $lv (CMake cannot link that)"; else ltxt="none"; fi
    say "  $c"
    say "      cuda_runtime.h : ${h:-none}"
    say "      libcudart      : $ltxt"
    say "      nvcc           : ${n:-none}"
    if [ -z "$ROOT" ] && [ -n "$h" ] && [ -n "$l" ] && [ -n "$n" ]; then
        ROOT="$c"; NVCC="$n"
    fi
done

if [ -z "$ROOT" ]; then
    say ""
    say "No complete CUDA toolkit root was found, so CMake would fail the same way again."
    say "Send me the block above, plus the output of:   command -v nvcc; readlink -f \"\$(command -v nvcc)\"; ls \"\$HOME/stuart_creator_engine/cuda_shim\" \"\$HOME/stuart_creator_engine/cuda_shim/lib64\""
    exit 3
fi
say ""
say "Using CUDA root: $ROOT   (nvcc: $NVCC)"

export CUDA_HOME="$ROOT"
export PATH="$ROOT/bin:$ROOT/nvvm/bin:$PATH"

FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release
       -DCUDAToolkit_ROOT="$ROOT" -DCMAKE_CUDA_COMPILER="$NVCC")
[ "${ALLOW_UNSUPPORTED:-0}" = "1" ] && FLAGS+=("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
[ "${FA_ALL_QUANTS:-0}" = "1" ] && FLAGS+=(-DGGML_CUDA_FA_ALL_QUANTS=ON)

CONF=(cmake --fresh -S "$LLAMA" -B "$LLAMA/$NAME" "${FLAGS[@]}")
BUILD=(cmake --build "$LLAMA/$NAME" -j "$JOBS" --target llama-server llama-quantize llama-perplexity)

say "Configure : ${CONF[*]}"
say "Build     : ${BUILD[*]}"
if [ "$DRY" = "1" ]; then
    say "DRY_RUN=1: nothing was run."
    exit 0
fi

"${CONF[@]}" || { say "STOP: the configure step failed; the lines above say why (send them to me)."; exit 4; }
"${BUILD[@]}" || { say "STOP: the build failed; the last error lines above say why (send them to me)."; exit 5; }

say ""
say "Built. Checking the three programs:"
status=0
for t in llama-server llama-quantize llama-perplexity; do
    p="$LLAMA/$NAME/bin/$t"
    if [ -x "$p" ]; then
        miss="$(ldd "$p" 2>/dev/null | grep -c 'not found')"
        say "  ok   $p   (libraries not found: $miss)"
        [ "$miss" != "0" ] && { ldd "$p" | grep 'not found'; status=6; }
    else
        say "  MISSING $p"; status=6
    fi
done
say ""
say "Next:  \$PY stuart_pelias0.2.py export --adapter <the phase-2 run>/final"
exit "$status"
