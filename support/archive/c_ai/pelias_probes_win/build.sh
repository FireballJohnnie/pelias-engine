#!/usr/bin/env bash
# build.sh - build pelias_probe on WSL2 / Linux. Usage: ./build.sh   (env: ARCH=sm_89  CUDA_HOME=/usr/local/cuda)
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
ARCH="${ARCH:-sm_89}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
  echo "nvcc not found. Expected /usr/local/cuda/bin/nvcc (CUDA toolkit for WSL). Nothing was built." >&2
  exit 1
fi
command -v g++ >/dev/null || { echo "g++ not found. Nothing was built." >&2; exit 1; }
echo "nvcc:  $("$NVCC" --version | tail -1)"
echo "g++:   $(g++ --version | head -1)"
echo "arch:  $ARCH"
mkdir -p build
echo "[1/4] kernels.cu (nvcc)";   "$NVCC" -std=c++20 -O2 -arch="$ARCH" -c kernels.cu -o build/kernels.o
echo "[2/4] cpu_bw.cpp (g++ -mavx2)"; g++ -std=c++20 -O3 -mavx2 -c cpu_bw.cpp -o build/cpu_bw.o
echo "[3/4] main.cpp (g++)";      g++ -std=c++20 -O2 -Wall -Wextra -I"$CUDA_HOME/include" -c main.cpp -o build/main.o
echo "[4/4] link (nvcc)";         "$NVCC" -arch="$ARCH" build/main.o build/cpu_bw.o build/kernels.o -o build/pelias_probe -ldl -lpthread
echo "built: $PWD/build/pelias_probe"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
./build/pelias_probe list | tr '\n' ' '; echo
echo "Next: ./build/pelias_probe selftest --out ./selftest_out"
