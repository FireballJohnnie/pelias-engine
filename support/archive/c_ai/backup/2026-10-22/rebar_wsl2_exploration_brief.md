# ReBAR / CPU-to-VRAM access on WSL2 — exploration brief

Written 2026-10-22 12:30 (Pelias, RTX 4070 12 GB, Ryzen 7 5800X3D, WSL2). Status: PARKED side branch of Stuart-Creator / Pelias-Tier. Re-open only if the author's own testing from this file finds a viable route.

Label key: [MEASURED] seen in output on this machine; [DERIVED] follows from measured facts; [ESTIMATE] my reasoning, not checked; [UNMEASURED] no number yet, pathway given; [AUTHOR-STATED] said by the author or written in his ledger.

---

## STATUS UPDATE 2026-10-22 21:2x (read this first; sections below are the 12:30 original)

- Premise change. nvidia-smi inside WSL2 reports BAR1 Total 16,384 MiB (Used 0 MiB in the one-shot probe and in the 3-minute sampling during series W) [MEASURED]. ReBAR is on at the host (HWiNFO agreed) [MEASURED]. The guest still has no NVIDIA PCI device, only /dev/dxg [MEASURED]. So "the BAR is invisible" is outdated; "the guest cannot map it" still stands [DERIVED]. Whether any WSL2 path puts data through the BAR: [UNMEASURED].
- Author question (NVIDIA Profile Inspector / PowerShell). Sources found describe the Profile Inspector rBAR options as per-application driver-profile settings (Tom's Hardware); no source says they reach CUDA under WSL2 [UNMEASURED]. The BAR is already 16 GiB, so that setting is not the limiting factor [DERIVED]. NVIDIA's WSL guide lists no BAR, PCI or GPUDirect support and states: no concurrent CPU/GPU access, pinned memory limited, no full managed memory.
- Test pathway, in order: (1) Windows-side read-only `nvidia-smi -q -d MEMORY` BAR1 lines (author runs; compare with the WSL view); (2) Section 6 checks for route B (D3D12 upload heap) while a transfer-heavy bench runs, sampling BAR1 used; (3) native Linux boot (E1): there nvidia.ko exists, so gdrcopy applies; bench CPU-write latency and bandwidth through BAR1 against cudaMemcpy and compare with the 94 us handoff [UNMEASURED]. That bench is the "ReBAR module" starting point.

---

## FINAL WSL2 CHECK 2026-10-22 22:34 (all external-memory types with the real shared handle; file bar_probe/results/d3d12_bar_probe_20261022_223420.txt) - WSL2 BRANCH CLOSED, native Linux tabled

- [MEASURED] DEFAULT-heap control, real handle 0x28 (committed, flags 1 = dedicated) and 0x29 (heap, flags 0). Return codes (801 = NOT_SUPPORTED, 1 = INVALID_VALUE): committed: type 5 801, 2 801, 3 801, 6 801, 7 801, 1 801, 9 INVALID_VALUE. heap: type 4 801, 2 801, 3 801, 6 INVALID_VALUE, 7 INVALID_VALUE, 1 801, 9 801.
- [DERIVED] The INVALID_VALUE cases flip with the flags value (type 9: 1 with flags 1, 801 with flags 0; types 6 and 7: 1 with flags 0, 801 with flags 1), so they are argument-validation failures (the dedicated flag is required for the D3D11 resource types and not allowed for dma-buf), and the type-support check, which comes after validation, gives 801 once the arguments are valid. Every external-memory type (1, 2, 3, 4, 5, 6, 7, 9) has at least one NOT_SUPPORTED result with arguments that pass validation. Type 8 (NVSCIBUF) is a Tegra type and was not tried.
- [DERIVED] Verdict: CUDA in WSL2 imports no external-memory handle type, so no host-visible-VRAM allocation from D3D12 or Vulkan can reach CUDA here, with or without BAR. Together with GPUUploadHeapSupported = 0, the WSL2 ReBAR route is closed.
- Tabled for native Linux (E1), low priority: re-read attributes 116 (GPU_DIRECT_RDMA) and 124 (DMA_BUF); Vulkan HOST_VISIBLE+DEVICE_LOCAL allocation exported as OPAQUE_FD / DMABUF_FD and imported into CUDA; the 4 KiB write-plus-flag latency against the 13.3 us resident path. Only worth doing if the handoff term ever becomes the limiter (about 2-3% of a token in the ceiling model).

---

## D3D12 PROBE RESULT 2026-10-22 22:30 (d3d12_bar_probe, real shared handles; file bar_probe/results/d3d12_bar_probe_20261022_223022.txt) - ROUTE RETIRED ON WSL2

- [MEASURED] D3D12 works in WSL2: DXCore lists 'NVIDIA GeForce RTX 4070', D3D12CreateDevice OK, UMA=0, CacheCoherentUMA=0, IsolatedMMU=1. OPTIONS16 GPUUploadHeapSupported = 0: GPU upload heaps (CPU-visible VRAM) are NOT available on this adapter/runtime.
- [MEASURED] Control: a plain DEFAULT (VRAM) buffer, created as committed resource and as heap+placed, gets a valid shared handle (0x28, 0x29), but cuImportExternalMemory returns CUDA_ERROR_NOT_SUPPORTED (801) for types 5 (D3D12_RESOURCE), 4 (D3D12_HEAP) and 2 (OPAQUE_WIN32) with those real handles. So CUDA in WSL2 refuses D3D12/Win32 handle import even for an ordinary resource, independent of BAR.
- [MEASURED] UPLOAD, GPU_UPLOAD, CUSTOM L1+WRITE_COMBINE and CUSTOM L0+WRITE_COMBINE shared buffers all fail at creation with E_INVALIDARG (0x80070057), as committed resource and as heap. Whether plain UPLOAD failing is caused by the SHARED flag combination or by my parameters is UNKNOWN; it does not change the verdict because the import control already fails.
- [MEASURED] Earlier dummy-handle result: the same types returned INVALID_VALUE (1) with NULL handles but NOT_SUPPORTED (801) with real handles, so CUDA validates arguments BEFORE checking type support; the 801-vs-1 difference in the first probe carried no information about support (this corrects the reading both of us used).
- [DERIVED] Two independent blockers on WSL2: (a) no GPU upload heap on this adapter; (b) CUDA will not import D3D12/Win32 handles. (b) alone closes 'CPU-mapped BAR1 allocation imported into CUDA'. Not tested with real handles: external-memory types 3 (OPAQUE_WIN32_KMT), 6/7 (D3D11, no D3D11 in WSL), 9 (DMABUF_FD); low probability [ESTIMATE]. Native Linux (E1) re-check of attributes 116/124 and Vulkan OPAQUE_FD import remains the only open item, low priority.

---

## PROBE RESULT 2026-10-22 22:09 (bar_probe, WSL2, RTX 4070, driver API 13040; file bar_probe/results/bar_probe_20261022_220908.txt)

- [MEASURED] BAR1 used = 0 MiB before, after a 1 GiB cuMemAlloc, after VMM handles, after free. Guest has 0 PCI devices with vendor 10de; /dev/nvidia* absent; only /dev/dxg.
- [MEASURED] Attributes: CAN_MAP_HOST_MEMORY 1, UNIFIED_ADDRESSING 1, MANAGED_MEMORY 1, PAGEABLE_MEMORY_ACCESS 0, CONCURRENT_MANAGED_ACCESS 0, HOST_REGISTER_SUPPORTED 1, VMM 1, POSIX_FD handle type (VMM) 1, WIN32 handle type 0, GPU_DIRECT_RDMA 0, DMA_BUF 0. All enum numbers used match the local cuda.h.
- [MEASURED] cuMemGetHandleForAddressRange DMA_BUF_FD = NOT_SUPPORTED; cuImportExternalMemory OPAQUE_FD (real fd) = NOT_SUPPORTED; OPAQUE_WIN32, WIN32_KMT, D3D12_HEAP, D3D12_RESOURCE, D3D11 types = INVALID_VALUE with dummy NULL handles (type not rejected up front, but a dummy proves nothing about a real handle); P2P_TOKENS and cuMemCreate+gpuDirectRDMACapable = INVALID_DEVICE; plain cuMemCreate and POSIX_FD cuMemCreate succeed (this validates the property struct layout used by the probe).
- [MEASURED] Vulkan: loader present, ICD files for lvp, gfxstream, intel, radeon, asahi, nouveau; NO NVIDIA ICD and no dzn (D3D12-backed) ICD; vulkaninfo and vulkan.h not installed.
- [MEASURED] libd3d12core.so (dated Oct 2023) DOES contain D3D12_HEAP_TYPE_GPU_UPLOAD strings (including 'check GPUUploadHeapSupported via OPTIONS16'): the runtime knows the heap type; the earlier [ESTIMATE] that it might lack it was wrong. Whether this adapter reports GPUUploadHeapSupported is [UNMEASURED].
- [DERIVED] Route A (Vulkan HOST_VISIBLE+DEVICE_LOCAL to CUDA OPAQUE_FD) is closed on WSL2: no hardware Vulkan ICD, and CUDA rejects OPAQUE_FD import. gdrcopy-style paths are closed (RDMA 0, no nvidia.ko). The only open WSL2 route is D3D12 shared heap/resource to CUDA via a real handle (types 4/5), [UNMEASURED]; attribute 104 = 0 makes it unlikely [ESTIMATE].
- Value ceiling [DERIVED]: handoff is about 2-3% of a token in the ceiling model and the resident host-flag path is already 13.3 us median, so even a working BAR route is a small gain. Recommendation recorded in RUNBOOK: close the WSL2 ReBAR branch; native Linux (E1) re-read of attributes 116/124 and Vulkan-to-CUDA OPAQUE_FD is the only remaining check, low priority.

---

## 1. The question

Can the CPU write directly into GPU memory (through the PCIe BAR window that ReBAR enlarges) from inside WSL2, and would that make the CPU-to-GPU handoff in Pelias-Tier faster?

What ReBAR is: it lets the CPU see all of VRAM through the BAR1 window instead of a small slice. It only affects CPU-initiated loads and stores to VRAM. It does not change GPU-side speed or GPU-to-host copies by itself.

## 2. Measured facts on this machine

| Item | Value | Label |
|---|---|---|
| BAR1 total / used (nvidia-smi) | 16,384 MiB / 0 MiB | [MEASURED] |
| PCIe link | 4.0 x16 | [MEASURED] (earlier window) |
| Kernel | 6.18.54.1-microsoft-standard-WSL2 | [MEASURED] |
| Module support | CONFIG_MODULES=y, CONFIG_MODULE_UNLOAD=y | [MEASURED] |
| Kernel headers | /usr/src/linux-headers-6.18.54.1-microsoft-standard-WSL2 present | [MEASURED] |
| /lib/modules | 6.18.40.1 and 6.18.54.1 | [MEASURED] |
| `modinfo nvidia` | "Module nvidia not found" | [MEASURED] |
| /proc/driver/nvidia | does not exist | [MEASURED] |
| /dev/gdrdrv, gdr module, gdrcopy package | none present | [MEASURED] |
| /dev/nvidia* | none present | [MEASURED] |
| /dev/dxg | exists, crw-rw-rw- root, device 10,258 | [MEASURED] |
| PCI devices in the guest | 4 total: vendor 0x1414 x1, vendor 0x1af4 (virtio) x3 | [MEASURED] |
| lspci GPU line | `e6d7:00:00.0 3D controller: Microsoft Corporation Basic Render Driver` | [MEASURED] |
| NVIDIA vendor ID 0x10de in guest | absent | [MEASURED] |
| nvcc on PATH | CUDA 13.0.88 (not 13.2) | [MEASURED]; whether the llama.cpp build used it: [UNMEASURED] |
| nvidia-smi driver shown | CUDA UMD 13.4 | [MEASURED] (earlier window) |

Files in /usr/lib/wsl/lib [MEASURED]:

- libcuda.so / .so.1 / .so.1.1 — 188,024 bytes, dated Sep 18
- libd3d12.so — 801,840 bytes, dated Oct 20 2023
- libd3d12core.so — 6,880,344 bytes, dated Oct 20 2023
- libdxcore.so — 942,048 bytes, dated Mar 31 2024
- libcudadebugger.so.1, libnvidia-ml.so.1, nvidia-smi
- libnvidia-gpucomp.so (symlink name carries 615.78.05)

Note: libd3d12core.so is dated Oct 2023. If the "GPU upload heap" feature (Section 4, route B) arrived in a D3D12 runtime later than that, this copy may not have it [ESTIMATE]. Check 2 in Section 6 tests it.

## 3. Why a guest kernel module probably cannot work (route A)

- WSL2 gives the guest a paravirtual GPU, not the physical card. The guest PCI bus has no NVIDIA device, only a Microsoft "Basic Render Driver" stand-in [MEASURED].
- With no 0x10de device in the guest there is no BAR resource for Linux to map, whatever module is loaded [DERIVED].
- gdrcopy's `gdrdrv` needs the Linux NVIDIA kernel driver (`nvidia.ko`) to export P2P symbols. That module is absent [MEASURED], and installing a Linux NVIDIA driver in WSL2 is not the supported setup [ESTIMATE].
- Passthrough of the real card into the guest (Discrete Device Assignment) is not normally offered for a consumer card on Windows client [ESTIMATE].

Status: BLOCKED unless someone finds a way to put the physical device or its BAR into the guest. Do not spend time on module-building until a route below shows a CPU-visible VRAM mapping exists at all.

## 4. Routes that are still open (no custom kernel module)

### Route B — D3D12 "GPU upload heap" through libd3d12 in WSL2

Idea [ESTIMATE]: Windows D3D12 can create a heap in VRAM that the CPU may write to; it relies on ReBAR. WSL2 exposes D3D12 to Linux programs through libd3d12 and libdxcore. A small Linux C++ program could create that heap, map it, write a few bytes from the CPU and time it.

Needs to be true:
1. The d3d12core in WSL supports that heap type (see Section 6, check 2).
2. The WSL D3D12 layer returns a CPU pointer for that heap.
3. For this to help Pelias, CUDA must be able to read that same memory. That is route D, and it is the hard part.

Build note [ESTIMATE, verify]: the open-source DirectX-Headers project (Microsoft) provides D3D12 headers usable on Linux/WSL.

### Route C — talk to /dev/dxg directly (dxgkrnl ioctls)

Idea [ESTIMATE]: the WSL kernel's dxgkrnl driver exposes lock/map operations for video-memory allocations. A CPU-visible allocation might map host-side BAR-backed pages into the guest.

Needs to be true: the ioctl interface is documented in the kernel tree's uapi header, and a CPU-visible vidmem allocation is permitted by the host driver. Higher effort than route B; try it only if B is blocked only by the old runtime. Check 3 in Section 6 finds the header.

### Route D — CUDA-side interop (needed by both B and C)

Unknown [UNMEASURED]: can WSL's libcuda import a D3D12 resource, or any CPU-writable VRAM, as CUDA memory? Pathway: read the CUDA device attributes for this GPU (Section 6, check 4): GPUDirect RDMA support, virtual memory management support, host-register support, and which external-memory handle types are supported.

Baseline that already works [DERIVED from earlier microbench]: GPU reads a pinned host flag written by the CPU. That path involves no BAR write at all (see Section 5).

## 5. What it could buy, and the gate before spending effort

Handoff numbers (from `moe_track/microbench/03_handoff_latency.cu`, logs under `logs/floors/`):

| Mode | Median | Label |
|---|---|---|
| pingpong_flag (resident kernel, mapped flag) | 3.36 us | [MEASURED] |
| pingpong_4k (flag plus 4 KB read) | 13.35 us (p99 18.7) | [MEASURED] |
| launch_after_kick (launch plus sync per layer) | 94.40 us (p99 123.0) | [MEASURED] |
| spin_wait | 46.17 us | [MEASURED] |
| copy_roundtrip | 175.0 us | [MEASURED] |

The status ledger (line 52) quotes "5.05 us median round-trip" for Microbenchmark 2c [AUTHOR-STATED]. I could not match 5.05 us to any log line I opened; the logs above give different numbers. Reconcile before using either.

Per-token budget at the tier's measured speed (44.2 to 44.7 tok/s): about 22.4 to 22.6 ms per token [DERIVED].

Upper bounds on what a faster CPU-to-GPU doorbell could save [DERIVED from the microbench modes]:
- if the real engine uses a resident kernel: about 0.5 ms per token, roughly 2%;
- if it uses a launch-plus-sync path: about 2.6 to 3.8 ms per token, roughly 12 to 17%.

Which design the real tier build pays is [UNMEASURED]. Evidence so far:
- The two tier source files (`llama-pelias-tier.cpp`, `.h`) contain zero hits for doorbell, cudaHostAlloc, cudaHostRegister, cudaStreamSynchronize, cudaEventQuery [MEASURED].
- No `PELIAS_TIER` string was found under `ggml/src` [MEASURED].
- The ledger (line 219) lists the doorbell handoff as "Not Written Yet" [AUTHOR-STATED].
- So the tier most likely uses stock ggml scheduler handoffs [ESTIMATE].
- The server logs do not print a graph-split count [MEASURED]; "graphs reused" lines are not the same thing [ESTIMATE].

Open gap: the tier's measured speed is about 8.5 ms per token slower than the ceiling model predicted (about 13.9 ms at 71.8 tok/s) [DERIVED]. Handoff cost could explain part of it, not all [ESTIMATE].

GATE: do not invest in a ReBAR module until the real per-token handoff cost of the tier build is measured. Pathway: find the number of CUDA-to-CPU scheduler splits per token (needs a source of that count, not yet found), then multiply by a measured per-split sync cost.

## 6. Read-only checks to run (WSL), one at a time

1. Is a native Linux NVIDIA driver visible at all (repeat, for the record):
   `ls /dev/nvidia* /dev/gdrdrv 2>&1; lspci | grep -i nvidia`
2. Does this d3d12core know the GPU upload heap? (If the string is absent, route B is probably blocked by the old runtime.)
   `strings /usr/lib/wsl/lib/libd3d12core.so | grep -iE "GPU_UPLOAD|GpuUpload" | head -n 5`
3. Is the dxgkrnl ioctl header in the installed kernel headers (route C)?
   `find /usr/src/linux-headers-$(uname -r) -iname "*d3dkmthk*" -o -iname "*dxgkrnl*" 2>/dev/null | head -n 10`
4. CUDA attributes (route D). First find the enum names and values in the header, then query them:
   `grep -nE "GPU_DIRECT_RDMA|VIRTUAL_MEMORY_MANAGEMENT|HOST_REGISTER_SUPPORTED|HANDLE_TYPE" /usr/local/cuda/include/cuda.h | head -n 20`
   (Querying needs a small program or a ctypes one-liner; ask before creating a file.)
5. Is a D3D12 header set available for building a test (route B)?
   `ls /usr/include/directx /usr/include/wsl 2>&1 | head; apt list --installed 2>/dev/null | grep -iE "directx|dxheaders" | head`

## 7. Native Linux plan (the parked fallback)

When the native Linux boot exists:
- Confirm BAR1: `lspci -vv -s <bus id>` and read the Region lines. Expect 16 GiB if ReBAR is active [ESTIMATE].
- gdrcopy: install `nvidia` driver, build gdrcopy and its `gdrdrv`, then run its sanity and latency tests (test names to verify: `gdrcopy_sanity`, `gdrcopy_copybw`, `gdrcopy_pplat`) [ESTIMATE].
- RISK [ESTIMATE]: GPUDirect RDMA / P2P pinning has historically been limited on GeForce cards. If gdrcopy refuses the 4070, the CPU-to-VRAM store test needs another route, such as a CUDA dma-buf export; look up the current CUDA documentation before planning that.
- Compare against the Section 5 baseline: a store into VRAM versus the pinned-host-flag handoff, with identical tests on both.

## 8. Decision record

- Route A (guest module): blocked by evidence in Section 2, pending any new idea.
- Routes B, C, D: open; B check 2 comes first because it is one read-only command.
- Nothing here changes the current N-ladder or tier plan.
- Constraints that still apply: no PowerShell, create files only when the author asks, delete nothing.
