#!/usr/bin/env bash
# run_wsl_all.sh - run the pelias probe suite inside WSL2 (or native Linux).
#   ./run_wsl_all.sh quick     short pass over every probe (about 10 minutes)
#   ./run_wsl_all.sh full      the real measurements (about 70 minutes; closes nothing for you - close other GPU apps first)
#   ./run_wsl_all.sh tdr       display-timeout test, stage by stage, asks before each stage (screen may flicker)
#   ./run_wsl_all.sh census    read-only machine facts and driver census only, no GPU load
# Environment (all optional):
#   PELIAS_GGUF=/path/to/model.gguf   adds the model-file read test (the path is never written to results)
#   ONLY="h5 h6"   run just these      SKIP="m3 h4"   leave these out
#   TAG=stock      label for the run (use after resetting Afterburner to stock)
#   YES=1          do not ask for confirmation (not used for the tdr stages)
#   SANITIZE=1     also run selftest under compute-sanitizer (slow)
#   DRIVE_DIR=...  where results are copied (default: your backup drive)
set -u
MODE="${1:-quick}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/build/pelias_probe"
DRIVE_DIR="${DRIVE_DIR:-/mnt/c/AI/cl-sndbx-bkp/stuart-creator/2026-10-23/probe_results/wsl}"
TAG="${TAG:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${STAMP}_${MODE}${TAG:+_$TAG}"
OUT="$HOME/pelias_probe_results/$RUN_NAME"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

case "$MODE" in quick|full|tdr|census) ;; *) echo "usage: $0 [quick|full|tdr|census]"; exit 1 ;; esac
mkdir -p "$OUT"
LOG="$OUT/run.log"
SUMMARY="$OUT/summary.tsv"
printf 'probe\texit_code\tseconds\n' > "$SUMMARY"

say() { echo "$@" | tee -a "$LOG"; }
DRIVE_WARNED=0
sync_out() {
  if [ -d "$(dirname "$DRIVE_DIR")" ] || mkdir -p "$DRIVE_DIR" 2>/dev/null; then
    mkdir -p "$DRIVE_DIR/$RUN_NAME" 2>/dev/null && cp -r "$OUT/." "$DRIVE_DIR/$RUN_NAME/" 2>/dev/null && return 0
  fi
  if [ "$DRIVE_WARNED" = 0 ]; then echo "(could not copy to $DRIVE_DIR - results stay in $OUT)"; DRIVE_WARNED=1; fi
  return 1
}
trap 'say "interrupted"; sync_out; exit 130' INT TERM

# ---------------------------------------------------------------- machine state + census (read-only)
CENSUS="$OUT/census.txt"
sec() { { echo "### $1"; shift; "$@" 2>&1; echo; } >> "$CENSUS"; }
shsec() { { echo "### $1"; bash -c "$2" 2>&1; echo; } >> "$CENSUS"; }
NVSMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
WSLCFG="$(ls /mnt/c/Users/*/.wslconfig 2>/dev/null | head -1)"

stable_state() {
  echo "kernel=$(uname -r)"
  echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "mem_total_kib=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
  echo "gpu=$("$NVSMI" --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader 2>/dev/null | head -1)"
  echo "nvcc=$("$NVCC" --version 2>/dev/null | tail -1)"
  echo "wslconfig_sha=$([ -n "$WSLCFG" ] && sha256sum "$WSLCFG" | cut -c1-16 || echo none)"
  echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
  echo "memlock=$(ulimit -l)"
  echo "probe_sources_sha=$(cat "$HERE"/*.h "$HERE"/*.cpp "$HERE"/*.cu 2>/dev/null | sha256sum | cut -c1-16)"
}
stable_state > "$OUT/machine_state.txt"
SHA="$(sha256sum "$OUT/machine_state.txt" | cut -c1-16)"
say "run: $RUN_NAME   machine_state_sha: $SHA"

driver_census() {
  : > "$CENSUS"
  sec "date" date
  sec "uname -a" uname -a
  shsec "/proc/version" 'cat /proc/version'
  shsec "wsl detection" 'grep -qi microsoft /proc/version && echo "WSL2 kernel" || echo "not WSL"'
  shsec "device nodes (dxg = the WSL2 GPU bridge; nvidia* = native driver)" 'ls -l /dev/dxg /dev/nvidia* /dev/dri /dev/nvidia-uvm* 2>&1'
  shsec "native NVIDIA driver files (absent on WSL2)" 'ls -l /proc/driver/nvidia 2>&1; cat /proc/driver/nvidia/version 2>&1'
  shsec "/usr/lib/wsl/lib listing" 'ls -l /usr/lib/wsl/lib 2>&1'
  shsec "expected libraries present?" 'for f in libcuda.so.1 libcuda.so libnvidia-ml.so.1 libnvidia-ml.so libnvidia-ptxjitcompiler.so.1 libnvcuvid.so.1 libnvidia-encode.so.1 libd3d12.so libd3d12core.so libdxcore.so libcudadebugger.so.1 libnvidia-opticalflow.so.1; do if [ -e /usr/lib/wsl/lib/$f ]; then echo "present  $f"; else echo "missing  $f"; fi; done'
  shsec "ldconfig entries (cuda/nvidia/dxcore)" 'ldconfig -p | grep -iE "cuda|nvidia|dxcore|d3d12" | head -60'
  shsec "kernel modules (lsmod)" 'lsmod 2>&1 | head -30; echo "--- modinfo nvidia:"; modinfo nvidia 2>&1 | head -5'
  shsec "dmesg lines about dxg / nvidia / vmbus (may be empty without root)" 'dmesg 2>&1 | grep -iE "dxg|nvidia|vmbus|hv_" | head -60'
  shsec "PCI devices (vendor 0x10de = NVIDIA)" 'ls /sys/bus/pci/devices 2>/dev/null | while read d; do echo "$d vendor=$(cat /sys/bus/pci/devices/$d/vendor) device=$(cat /sys/bus/pci/devices/$d/device) class=$(cat /sys/bus/pci/devices/$d/class)"; done; echo "(lspci:)"; lspci 2>&1 | head -20'
  shsec "ulimit -a (ulimit -l is RLIMIT_MEMLOCK)" 'ulimit -a'
  shsec "/proc/self/limits" 'cat /proc/self/limits'
  shsec "memory" 'free -m; echo; swapon --show; echo; grep -E "MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|HugePages|AnonHuge|Hugepagesize" /proc/meminfo'
  shsec "transparent hugepages and vm settings" 'for f in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag /proc/sys/vm/swappiness /proc/sys/vm/overcommit_memory /proc/sys/vm/max_map_count; do echo "$f: $(cat $f 2>&1)"; done'
  shsec "cpu" 'lscpu 2>&1 | head -25; echo; grep -m1 flags /proc/cpuinfo | tr " " "\n" | grep -E "^(avx|avx2|avx512.*|fma|bmi1|bmi2|sse4_2|aes|amx.*)$" | tr "\n" " "'
  shsec "/etc/wsl.conf" 'cat /etc/wsl.conf 2>&1'
  shsec ".wslconfig (Windows side, read only)" '[ -n "'"$WSLCFG"'" ] && cat "'"$WSLCFG"'" || echo "not found under /mnt/c/Users"'
  shsec "environment (only CUDA / NVIDIA / WSL / LD_LIBRARY_PATH / PATH)" 'env | grep -E "^(CUDA|NVIDIA|WSL|LD_LIBRARY_PATH|PATH)[A-Za-z_]*=" '
  shsec "toolchain" 'echo "g++: $(g++ --version 2>&1 | head -1)"; echo "gcc: $(gcc --version 2>&1 | head -1)"; echo "cmake: $(cmake --version 2>&1 | head -1)"; echo "nvcc: $('"$NVCC"' --version 2>&1 | tail -2 | tr "\n" " ")"; ls -d /usr/local/cuda* 2>&1; echo "compute-sanitizer: $(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>&1)"; echo "nsys: $(command -v nsys || echo missing)  ncu: $(command -v ncu || echo missing)"'
  shsec "nvidia-smi (summary)" "\"$NVSMI\""
  shsec "nvidia-smi (csv query)" "\"$NVSMI\" --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,power.limit,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,memory.total,memory.used,memory.free,utilization.gpu --format=csv"
  shsec "nvidia-smi -q (full)" "\"$NVSMI\" -q"
  shsec "who is using the GPU from inside WSL" "\"$NVSMI\" --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1; ps -eo pid,comm,rss --sort=-rss 2>/dev/null | head -8"
}

# ---------------------------------------------------------------- running probes
ARGS=(--out "$OUT" --machine-sha "$SHA")
[ -n "$TAG" ] && ARGS+=(--tag "$TAG")
QF=""
[ "$MODE" = "quick" ] && QF="--quick"

TENSOR_BLOCKED=0
wanted() {
  local n="$1" x
  if [ "$TENSOR_BLOCKED" = 1 ] && { [ "$n" = h3 ] || [ "$n" = h10 ] || [ "$n" = h4 ]; }; then return 1; fi
  if [ -n "${ONLY:-}" ]; then for x in $ONLY; do [ "$x" = "$n" ] && return 0; done; return 1; fi
  for x in ${SKIP:-}; do [ "$x" = "$n" ] && return 1; done
  return 0
}

run_probe() {   # run_probe <name> <timeout_seconds> [extra args...]
  local name="$1" tmo="$2"; shift 2
  local t0 rc
  say ""
  say "=== $name  ($(date +%H:%M:%S), timeout ${tmo}s)"
  t0=$(date +%s)
  timeout --kill-after=30 "$tmo" "$BIN" "$name" $QF "$@" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 124 ]; then say "[TIMEOUT] $name was stopped after ${tmo}s"; fi
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ] && [ "$rc" -ne 4 ] && [ "$rc" -ne 124 ]; then say "[EXIT $rc] $name ended abnormally (137 = killed, often out of memory)"; fi
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$SUMMARY"
  sync_out
  return "$rc"
}

# ---------------------------------------------------------------- preflight
preflight() {
  if [ ! -x "$BIN" ]; then echo "Not built yet: run ./build.sh first."; exit 1; fi
  local temp mem
  temp="$("$NVSMI" --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)"
  mem="$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)"
  say "GPU temperature now: ${temp:-unknown} C   |   free host memory: ${mem} GiB"
  if [ -n "${temp:-}" ] && [ "${temp:-0}" -ge 60 ] 2>/dev/null; then say "WARNING: the card is already warm (${temp} C). Let it idle first for clean numbers."; fi
  if [ "$MODE" = "full" ] && awk -v m="$mem" 'BEGIN{exit !(m < 35)}'; then say "WARNING: under 35 GiB free - the pinned-memory probes (P-H7, P-M3) will stop early to protect the machine."; fi
  say "Close browsers, games, video and any other GPU or LLM program before this run."
  if [ -z "${PELIAS_GGUF:-}" ]; then say "Note: PELIAS_GGUF is not set, so the model-file read test (m4) will be skipped."; fi
}
confirm() {
  [ "${YES:-0}" = 1 ] && return 0
  read -r -p "$1 [Enter to start, Ctrl-C to cancel] " _ || true
}

# ---------------------------------------------------------------- go
driver_census
sync_out
if [ "$MODE" = "census" ]; then
  say "census written: $CENSUS"
  say "Copied to: $DRIVE_DIR/$RUN_NAME"
  exit 0
fi
preflight

if [ "$MODE" = "tdr" ]; then
  say ""
  say "DISPLAY-TIMEOUT TEST. Each stage runs one long GPU kernel. Windows may reset the display driver: the screen can flicker or go black for a few seconds, and programs using the GPU can crash. Save your work first."
  for st in 2s 5s 30s queued graph; do
    wanted "h13_$st" || continue
    read -r -p "Run stage '$st'? Type YES to run, anything else to stop: " ans || ans=""
    if [ "$ans" != "YES" ]; then say "stopped before stage $st"; break; fi
    run_probe h13 240 --stage "$st"
    rc=$?
    if [ "$rc" -ne 0 ]; then say "Stopped: stage $st did not pass. Do not run the longer stages. Results are saved."; break; fi
  done
  say ""; say "Copied to: $DRIVE_DIR/$RUN_NAME"
  exit 0
fi

[ "$MODE" = "full" ] && confirm "About to run the FULL suite (about 70 minutes, runs the GPU hard)."

T_SELF=300; T_STATE=60
if [ "$MODE" = "quick" ]; then T_SHORT=300; T_H4=300; T_MEM=600; T_M3=600; else T_SHORT=900; T_H4=2700; T_MEM=1800; T_M3=3000; fi

run_probe selftest "$T_SELF"
rc_self=$?
if [ "$rc_self" = 4 ]; then
  TENSOR_BLOCKED=1
  say "The tensor-core self-checks failed (everything else passed). Skipping h3, h10 and h4 and running the rest."
elif [ "$rc_self" -ne 0 ] && [ "${FORCE:-0}" != 1 ]; then
  say "The self-test failed, so the measurements cannot be trusted. Stopping. (Set FORCE=1 to continue anyway.)"
  sync_out
  say "Copied to: $DRIVE_DIR/$RUN_NAME"
  exit 3
fi
if [ "${SANITIZE:-0}" = 1 ]; then
  CS="$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null)"
  if [ -n "$CS" ]; then
    say "=== compute-sanitizer memcheck on selftest"
    timeout --kill-after=30 1800 "$CS" --tool memcheck --error-exitcode 9 "$BIN" selftest --out "$OUT" --tag sanitizer 2>&1 | tee "$OUT/compute_sanitizer_selftest.txt" | tail -25 | tee -a "$LOG"
  else
    say "compute-sanitizer not found; skipped"
  fi
fi

wanted state && run_probe state "$T_STATE"
wanted h8  && run_probe h8  120
wanted h1  && run_probe h1  120
wanted h2  && run_probe h2  "$T_SHORT"
wanted h3  && run_probe h3  "$T_SHORT"
wanted h10 && run_probe h10 "$T_SHORT"
wanted h4  && run_probe h4  "$T_H4"
wanted h5  && run_probe h5  "$T_SHORT"
wanted h6  && run_probe h6  "$T_SHORT"
wanted h12 && run_probe h12 "$T_SHORT"
wanted h7  && run_probe h7  "$T_MEM"
wanted m3  && run_probe m3  "$T_M3"
wanted m4  && run_probe m4  "$T_MEM"
wanted h9  && run_probe h9  "$T_SHORT"

say ""
say "================ summary ($RUN_NAME)"
grep -E '^\[(PASS|FAIL|REC|ERROR|SKIP|TIMEOUT|EXIT)' "$LOG" | tee -a "$OUT/summary.txt"
sync_out
say ""
say "Results in WSL:   $OUT"
say "Copied to drive:  $DRIVE_DIR/$RUN_NAME"
say "Windows path:     $(echo "$DRIVE_DIR/$RUN_NAME" | sed 's|^/mnt/c/|C:\\|; s|/|\\|g')"
say "Next: the display-timeout test is separate (./run_wsl_all.sh tdr). Tell Claude when the run is done."
