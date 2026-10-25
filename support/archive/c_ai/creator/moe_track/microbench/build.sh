#!/usr/bin/env bash
# File: <project>\moe_track\microbench\build.sh   (WSL: <project>/moe_track/microbench/build.sh)
# Compiles the three micro-benchmarks into microbench/bin/. Compiles only; runs nothing.
#   bash microbench/build.sh            build what can be built, say what was skipped
#   ARCH=sm_89 bash microbench/build.sh  GPU architecture for the .cu files (RTX 4070 = sm_89, the default)
# Needs g++ (02) and nvcc (03, 04). nvcc is looked for on PATH, then in /usr/local/cuda/bin and /usr/local/cuda-*/bin.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"
BIN="$HERE/bin"
ARCH="${ARCH:-sm_89}"
mkdir -p "$BIN"

find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return 0; fi
    local c
    for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}

built=(); skipped=()

if command -v g++ >/dev/null 2>&1; then
    if g++ -O3 -mavx2 -fopenmp -std=c++17 "$HERE/02_ram_bw_test.cpp" -o "$BIN/02_ram_bw_test"; then
        built+=("02_ram_bw_test")
    else
        skipped+=("02_ram_bw_test (compile error above)")
    fi
else
    skipped+=("02_ram_bw_test (g++ not found: sudo-free fix is not possible here; the build tools come with the llama.cpp build setup)")
fi

if NVCC="$(find_nvcc)"; then
    moe_log "nvcc: $NVCC  (arch $ARCH)"
    if "$NVCC" -O3 -std=c++17 -arch="$ARCH" "$HERE/03_handoff_latency.cu" -o "$BIN/03_handoff_latency"; then
        built+=("03_handoff_latency")
    else
        skipped+=("03_handoff_latency (nvcc error above)")
    fi
    if "$NVCC" -O3 -std=c++17 -arch="$ARCH" -Xcompiler "-fopenmp -mavx2" "$HERE/04_dual_bus_test.cu" -o "$BIN/04_dual_bus_test"; then
        built+=("04_dual_bus_test")
    else
        skipped+=("04_dual_bus_test (nvcc error above)")
    fi
else
    skipped+=("03_handoff_latency, 04_dual_bus_test (nvcc not found; build_llama_cuda.sh uses the same toolkit, check 'ls /usr/local/cuda*/bin/nvcc')")
fi

echo
moe_log "built:   ${built[*]:-nothing}"
if [ "${#skipped[@]}" -gt 0 ]; then
    moe_log "skipped:"
    printf '           %s\n' "${skipped[@]}"
    exit 1
fi
exit 0
