#!/usr/bin/env bash
# run_bar_probe.sh - READ-ONLY probe: can anything in WSL2 map VRAM for CPU writes (BAR1)?
# Run from WSL:  bash <project>/moe_track/bar_probe/run_bar_probe.sh
# Needs: gcc. Uses ~1 GiB VRAM for a few seconds. Refuses to run while a test is running (FORCE=1 overrides).
# Output: results/bar_probe_<stamp>.txt (same text as the screen).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/results"
OUT="$HERE/results/bar_probe_$(date +%Y%m%d_%H%M%S).txt"

if [ "${FORCE:-0}" != 1 ] && pgrep -f 'W_bank70_extras|gauntlet\.py|llama-server|llama-bench|_lib\.sh' >/dev/null 2>&1; then
  echo "A test looks like it is still running (W / gauntlet / llama-server). Wait for it, or run: FORCE=1 bash $0"
  exit 1
fi

{
echo "## bar_probe $(date -Is)  kernel $(uname -r)"

echo; echo "## A. nvidia-smi BAR1 and PCI view (guest)"
nvidia-smi -q -d MEMORY | sed -n '/FB Memory Usage/,/Conf Compute/p'
echo "PCI devices with vendor 10de in guest: $(grep -l '^0x10de' /sys/bus/pci/devices/*/vendor 2>/dev/null | wc -l)"
ls -l /dev/dxg /dev/nvidia* 2>&1 | head -5

echo; echo "## B. WSL graphics/compute libraries present"
ls -l /usr/lib/wsl/lib 2>/dev/null | grep -E 'libcuda|libd3d12|libdxcore|libnvidia|libvulkan|vulkan' 
echo "-- Vulkan loader / ICDs / tools:"
ls /usr/lib/x86_64-linux-gnu/libvulkan.so* 2>&1 | head -3
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1 | head -10
command -v vulkaninfo || echo "vulkaninfo: not installed"
ls /usr/include/vulkan/vulkan.h 2>&1 | head -1
echo "-- GPU_UPLOAD strings in libd3d12core.so:"
strings /usr/lib/wsl/lib/libd3d12core.so 2>/dev/null | grep -iE 'GPU_UPLOAD|GpuUpload' | head -10 || true
ls -l /usr/lib/wsl/lib/libd3d12core.so 2>&1
echo "-- external-memory / VMM entry points in libcuda:"
nm -D /usr/lib/wsl/lib/libcuda.so.1 2>/dev/null | grep -E 'cuImportExternalMemory|cuExternalMemoryGetMappedBuffer|cuMemCreate|cuMemMap|cuMemAddressReserve|cuMemGetHandleForAddressRange' || echo "(nm unavailable or none found)"

echo; echo "## C. enum values in YOUR cuda.h (to cross-check the numbers the probe uses)"
H=/usr/local/cuda/include/cuda.h
if [ -f "$H" ]; then
  grep -E 'CU_DEVICE_ATTRIBUTE_(INTEGRATED|CAN_MAP_HOST_MEMORY|UNIFIED_ADDRESSING|MANAGED_MEMORY|PAGEABLE_MEMORY_ACCESS|CONCURRENT_MANAGED_ACCESS|HOST_REGISTER_SUPPORTED|VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED|HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED|HANDLE_TYPE_WIN32_HANDLE_SUPPORTED|GPU_DIRECT_RDMA_SUPPORTED|DMA_BUF_SUPPORTED) *=' "$H" | sed 's/^ *//' | head -20
  grep -E 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_[A-Z0-9_]+ *=|CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD *=|CU_POINTER_ATTRIBUTE_P2P_TOKENS *=' "$H" | sed 's/^ *//' | head -12
else
  echo "cuda.h not found at $H"
fi

echo; echo "## D. driver probe (builds bar_probe.c with gcc, runs it)"
if gcc -std=gnu11 -O1 -Wall -o "$HERE/bar_probe" "$HERE/bar_probe.c" -ldl; then
  "$HERE/bar_probe"
else
  echo "BUILD FAILED (is gcc installed?)"
fi
echo; echo "## end"
} 2>&1 | tee "$OUT"
echo "saved: $OUT"
