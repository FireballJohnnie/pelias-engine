# PELIAS RESEARCH UPDATE & MASTER VERIFICATION AUDIT (`pelias-research-update.md`)

**Unified Cross-Check of `pelias-engine` and `pelias-trainer`, Complete Derivation & Probe Ledger, Ceiling Gap Attribution, Blueprint Synchronization, and the Master C++20/CUDA Universal Unified-Bus Architecture**.

| Field | Value | Source & Verification Date |
| --- | --- | --- |
| **Document** | `pelias-research-update.md` (supersedes draft alias `pelias-research.update.md`)

 | Master Verification Audit (`2026-10-23`)

 |
| **Date** | `2026-10-23`<br> | Repository Ledger & Master Notes Lock (`2026-10-23`)

 |
| **Author & Maintainer** | the user (`Copyright (c) 2026 John Lipka and the Pelias contributors`)

 | `PELIAS_TRAINER_MASTER_NOTES.md` & `README.md` (`2026-10-20` to `2026-10-23`)[cite: 2, 3] |
| **Repositories** | [`pelias-engine`](https://www.google.com/search?q=https://github.com/<account>/pelias-engine) (Inference) & `pelias-trainer` (LoRA/QAT Fine-Tuning)[cite: 1, 3, 4] | Private repositories, interfaced via standard GGUF (`2026-10-23`)[cite: 2, 3] |
| **Pinned Oracle** | [`llama.cpp`](https://github.com/ggerganov/llama.cpp) commit `2ca15f540` (`b11399`) with 4 carried patches (`mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu`) and 2 dropped diagnostic patches (`mmq_dump_diag`, `mmq_dump_diag2`) (`READ-SRC` / `DECISION`)[cite: 1, 2, 4] | Author Ruling (`2026-10-22`) & `PELIAS_ENGINE_LINUX_BLUEPRINT.md`[cite: 1, 4] |
| **Target Hardware** | NVIDIA GeForce RTX 4070 12 GB (`12,281 MB` GDDR6X, [Ada Lovelace `AD104](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)`, `sm_89`, 46 SMs, 5,888 CUDA cores, 184 Tensor Cores, 36 MiB L2, 192-bit bus, `504 GB/s`; OC: `965 mV` @ `~2,775–2,790 MHz` core, `+1000 MHz` mem) + AMD Ryzen 7 5800X3D ([Family `19h` Zen 3](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf), 8C/16T, AVX2 only, no AVX-512, 96 MB 3D V-Cache victim L3) + 64 GB (`4×16 GB`) DDR4-3200 (`51.2 GB/s` theoretical peak) + PCIe 4.0 x16 (`26.1–26.2 GB/s` measured) + CUDA `13.0.88`[cite: 1, 4, 6] | `MEASURED` via HWiNFO (`2026-10-20`), `nvcc`, microbenches `02`–`04` (`2026-10-20`), & `PUBLISHED` vendor specs (`2026-10-23`)[cite: 1, 4, 6] |
| **Hard VRAM Ceiling** | **`10.9 GiB`** (Author ruling, `2026-10-22`; enforced by runtime memory governor against `10.4 / 10.9 / 11.2 GiB` candidates)[cite: 1, 2, 4] | `DECISION` (`2026-10-22`, `PELIAS_ENGINE_LINUX_BLUEPRINT.md`)[cite: 1, 2, 4] |
| **Honesty Labels** | **MEASURED** (run on the author's machine, in a ledger or raw file) · **READ-SRC** (read from source or from the served model file's header) · **PUBLISHED** (external vendor/academic specification, cited with URL and check date in Section 8) · **DERIVED** (scripted arithmetic from labelled inputs) · **ESTIMATE** (engineering judgement, paired with a named probe) · **UNVERIFIED** (not relied on until its named probe passes) · **DECISION** (a design choice or author ruling with a recorded reason) · **AUTHOR-STATED** (stated by the author; no log attached)

 | Enforced across all 231 labelled statements in the Trainer Master Scaffold (`MEASURED 24 · READ-SRC 42 · PUBLISHED 33 · DERIVED 72 · ESTIMATE 23 · UNVERIFIED 19 · DECISION 18`) (`2026-10-23`)

 |

---

## 1. The Core Discovery: Turning Consumer Hardware into a Universal Unified-Bus System

The foundational breakthrough driving both `pelias-engine` and `pelias-trainer` emerged from measuring actual hardware floors while solving a practical constraint: fine-tuning and serving **Qwen3.6-35B-A3B** (`20.22 GiB` in `Q4_K_M`, `753` tensors) inside a strict **`10.9 GiB` VRAM ceiling** on a single 12 GB consumer RTX 4070 paired with 64 GB of DDR4-3200 system RAM (`READ-SRC` / `DECISION`, `2026-10-20` to `2026-10-23`).

### 1.1 The Physical Dual-Bus Discovery & Why Unknowns Had to Be Derived

Standard inference and training frameworks treat PCIe offload as a serial fallback where either the GPU stalls waiting for host-to-device copies or the CPU executes layers while the GPU sits idle (`MEASURED` on stock/tier splits, `2026-10-22`)[cite: 1, 3, 4]. By writing standalone hardware microbenchmarks (`02_ram_bw_test.cpp`, `03_handoff.cu`, `04_dual_bus.cu`, `2026-10-20`) on the reference machine, four physical facts were established[cite: 1, 4]:

1. **Additive Dual-Bus Memory Saturation (`47.62 GB/s` = `93%` of Theoretical DDR4-3200 Peak):**
* Theoretical dual-channel DDR4-3200 peak bandwidth is **`51.2 GB/s`** (`PUBLISHED` JEDEC DDR4-3200 spec, checked `2026-10-23`)[cite: 1, 6].
* Standalone CPU AVX2 reads (`02_ram_bw_test.cpp`) achieve **`43.8–45.9 GB/s`** (`45.81 GB/s` on `4 KiB` pages; `44.86 GB/s` on `2 MiB` pages) (`MEASURED`, Engine Ledger §1, `2026-10-20`).


* Standalone PCIe 4.0 x16 GPU reads (`04_dual_bus.cu`) achieve **`26.1 GB/s`** (`26.1–26.2 GB/s` via the `uint4` zero-copy checksum kernel over `cudaHostAlloc` pinned memory) (`MEASURED`, Engine Ledger §1, `2026-10-20`)[cite: 1, 2, 4, 6].
* When the Ryzen 7 5800X3D and the RTX 4070 read cold experts from the same DDR4-3200 memory controller **simultaneously** (`04_dual_bus.cu`), they do not halve each other's throughput; instead, the CPU sustains **`26.56 GB/s` (`26.6 GB/s`)** while the GPU simultaneously sustains **`21.06 GB/s`** over PCIe, achieving a combined **`47.62 GB/s` (`47.6 GB/s`)**—**`93.0%` of the physical `51.2 GB/s` DRAM pin ceiling** and **`1.04×`** the CPU alone (`MEASURED` / `DERIVED`, Engine Ledger §1, `2026-10-20`, `PELIAS_TRAINER_README_main.md` §9)[cite: 1, 3, 4].


2. **Sub-20-Microsecond Shared-Memory Doorbell Coordination (`13.3 µs` vs. `94 µs` OS Sync):**
* Coordinating CPU and GPU execution 40 times per token via standard CUDA stream synchronization (`cudaStreamSynchronize`) costs **`94 µs/layer` (`3.76 ms/token`)** in isolated microbenchmarking (`03_handoff.cu`) and **`4.6–5.1 ms/token`** in live WSL2 end-to-end token handoffs (`MEASURED`, `2026-10-20` & `2026-10-22`)[cite: 1, 4].
* Replacing OS/driver synchronization with a resident-kernel shared-memory doorbell flag (`03_handoff.cu`) drops round-trip handoff latency to **`13.3 µs` median (`0.53 ms/token`)** and **`18.7 µs` p99 (`0.75 ms/token`)**, with raw one-way flag visibility taking **`3.4 µs`** (`MEASURED`, `2026-10-20`)[cite: 1, 4].


3. **Asymmetric Compute + Bandwidth Coupling Across Inference and Training:**
* **In Inference (`pelias-engine`, Token-Major):** The always-on `1.41B`-parameter spine (`~0.80 GiB` in served mix) and a water-filled static bank of the **`4,170` hottest experts (`7.5 GiB`, incl. `40` zero slots = `72.6 MiB`)** remain locked in `504 GB/s` GDDR6X VRAM, absorbing **`82.73%`** of all routed expert byte traffic (`READ-SRC` / `DERIVED` from `P-E1` trace of `121,160` tokens, `2026-10-23`). Only the **`17.27%`** cold expert byte misses traverse the `47.62 GB/s` unified DDR4 + PCIe bus in parallel with GPU hot-expert execution (`DERIVED`, `2026-10-20` to `2026-10-23`)[cite: 1, 2, 4].


* **In Training (`pelias-trainer`, Layer-Major):** Because all **`19,503,513,600 B` (`18.164 GiB` trunk / `18.65 GiB` with `blk.40`)** of frozen quantized experts must be traversed once in forward and once in backward (`79` fetches, `35.8 GiB/step`), streaming them over the **`26.1 GB/s`** PCIe bus takes a fixed **`1.49 s/step`** regardless of token batch size (`DERIVED` from `MEASURED` bandwidth, `PELIAS_TRAINER_MASTER_NOTES.md` §1.7, `2026-10-23`). Grouping $K$ micro-batch windows per layer sweep (`D5`) hides `100%` of the PCIe transfer behind GPU Tensor Core compute once a step processes at least **`1,495 / 2,242 / 2,989` tokens** at `1,000 / 1,500 / 2,000 tok/s` (`DERIVED`, `2026-10-23`)[cite: 2].




4. **Why Universality Requires Deriving Unknowns From First Principles:**
* Turning an off-the-shelf consumer motherboard into a deterministic unified-bus system exposes undocumented hardware/OS interactions: Windows/WSL2 [`VidMm` forcing `4-KB` `DXGK_PTE` page tables](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address) on system RAM (`PUBLISHED`, checked `2026-10-23`), `ggml`'s `210-byte` `Q6_K` block geometry breaking `16-byte` (`uint4`) PCIe coalescing (`READ-SRC`, `P-E9`, `2026-10-23`), Zen 3's [50/50 SMT Load Queue split and post-`LOCK` FP stall penalty](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) (`PUBLISHED`, checked `2026-10-23`)[cite: 1, 6], and CPU `256`-element `Q8_K` vs. GPU `32`-element `q8_1` activation quantization spread (`READ-SRC`, `2026-10-23`).


* Every one of those unknowns is derived, bounded, and paired with an executable probe below so the architecture can scale from this single RTX 4070 + 5800X3D reference system to **any consumer GPU/CPU pair, any MoE or dense LLM, and any out-of-core sparse scientific or graphics workload** (`AUTHOR-STATED` goal / `DECISION`, `2026-10-20` to `2026-10-23`).





---

## 2. Zero-Hallucination Self-Audit & Reconciliation of Prior Claims

To maintain absolute data integrity across both repositories, this section audits the external research plan (`pelias_engine_hardware_verified_research_plan.md`, dated `2026-10-23`)[cite: 6] and internal working notes (`PELIAS_TRAINER_MASTER_NOTES.md`, dated `2026-10-23`)[cite: 2] against the raw benchmark ledgers (`benchmarks/RESULTS.md` §1–9.4)[cite: 1, 2, 4].

### 2.1 Formal Retirement of the `"57.0 tok/s"` Windows-Native Figure (`G1` / `X10`)

* **The Discrepancy:** In `pelias_engine_hardware_verified_research_plan.md` (§1, §2.1, §2.2, §3, dated `2026-10-23`), an outside draft claimed a live Windows 11 decode speed of `"57.0 tok/s (17.54 ms/token)"` and derived a `"6.67 ms/token (166.7 µs/layer)"` gap to the ceiling.


* **The Ledger Truth (`MEASURED`):** In `pelias-engine/benchmarks/RESULTS.md` (§9.4, `benchmarks/raw/windows-native-2026-10-23/`), the only recorded Windows 11 native runs are **`47.3` and `46.8 tok/s` (`21.14 / 21.37 ms/token`, client token-gap p50 `21.53 ms`) at short context (`ctx0`)** and **`53.7` and `54.1 tok/s` (`18.62 / 18.48 ms/token`, client token-gap p50 `18.28 ms`) at `12k` context** (`MEASURED`, `2026-10-23`).


* **Origin of the Mistyped `57.0`:** The number `57.0` appears in the repository logs only once: as the upper bound of a 3-request range at `4k` context depth for an **`8.5 GiB` WSL2 bank** whose median was `53.7 tok/s` (`55.7 tok/s` short) and whose peak VRAM (**`11.4 GiB`**) violated the **`10.9 GiB`** ceiling (`MEASURED`, `pelias-research-update.md` §1.1 & `PELIAS_TRAINER_MASTER_NOTES.md` §1.15, `2026-10-23`).


* **Resolution (`DECISION`):** The `"57.0 tok/s"` figure is permanently retired across all engine and trainer documents. All speed-to-ceiling calculations use the six verified benchmark entries in `benchmarks/RESULTS.md` (§8.2, §9.2, §9.4, dated `2026-10-22` & `2026-10-23`).



### 2.2 Strict Separation of Measured E0 Sub-Timings vs. Unmeasured Intra-Layer Splits

* **What is `MEASURED` in E0 (`benchmarks/raw/e0-2026-10-22/` & `RUNBOOK.md`, `2026-10-22`):** On WSL2 Reference Cell A (`7.0 GiB` bank, bypass 32, `-ub 4096`), a timer-free short-context token takes **`24.6 ms`** inside the server loop (streaming client token-gap p50 is `26.33 ms` at `ctx0` and `23.34 ms` at `12k`; with timer+CUDA graphs ON `31.99 / 28.18 ms`; with CUDA graphs OFF `36.50 / 32.56 ms`) (`MEASURED`, `2026-10-22`). Host-side split sums across the **`82 splits/token` (`41 CPU + 41 CUDA`)** decompose the `24.6 ms` token into:


* **CPU cold expert compute:** **`10.1 ms`** (vs. `2.86 ms` assumed in the `2026-10-20` ceiling model $\rightarrow$ **`+7.2 ms` gap**) (`MEASURED` / `DERIVED`, `2026-10-22`).


* **GPU compute:** **`6.6 ms`** (`2.75 ms` host CUDA split time with CUDA graphs ON vs. `8.04 ms` with CUDA graphs OFF; vs. `7.73 ms` assumed in the ceiling model $\rightarrow$ **`-1.1 ms` faster than model**) (`MEASURED` / `DERIVED`, `2026-10-22`).


* **Sync + activation copy handoff:** **`5.1 ms`** (`4.6–5.0+ ms` in `PELIAS_ENGINE_LINUX_BLUEPRINT.md` Sources note; vs. `0.25 ms` assumed in the ceiling model $\rightarrow$ **`+4.85 to +4.9 ms` gap**) (`MEASURED` / `DERIVED`, `2026-10-22`)[cite: 1, 4].
* **Copy-back / kernel launch overhead:** **`2.8 ms`** (vs. `0.00 ms` assumed in the ceiling model $\rightarrow$ **`+2.8 ms` gap**) (`MEASURED` / `DERIVED`, `2026-10-22`).


* **CPU and GPU execution overlap:** **Zero (`0%`)** (`MEASURED` — the CPU and CUDA graph splits execute strictly serially in the `llama.cpp` tier build, `2026-10-22`)[cite: 1, 4].


* **What is `DERIVED` (FLOP Shares) vs. `UNVERIFIED` (Sub-Layer Microsecond Splits):** Within any MoE layer, the CPU cold expert branch can only overlap with GPU hot-expert + shared-expert execution because router top-8 selection causally depends on the attention/DeltaNet output (`READ-SRC` / `DERIVED`, `2026-10-23`)[cite: 1, 4]. While the forward FLOP breakdown at $T = 8,192$ is exact (`DERIVED` in `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` §6.10, `2026-10-23`: out of **`6.74 GFLOP/token`**, DeltaNet projections are **`2.02 GFLOP` [`30%`]**, routed experts **`2.01 GFLOP` [`30%`]**, `lm_head` **`1.02 GFLOP` [`15%`]**, attention scores/values **`0.67 GFLOP` [`10%`]**, attention projections **`0.55 GFLOP` [`8%`]**, shared experts **`0.25 GFLOP` [`4%`]**, DeltaNet chunk core **`0.18 GFLOP` [`3%`]**, and routers **`0.04 GFLOP` [`1%`]**), the exact microsecond division of the `6.6 ms` batch-1 GPU time between spine attention/DeltaNet and hot-bank experts is **`UNVERIFIED` until Probe `P-E2` (`e0_timeline.patch` per-layer CUDA event timers) runs** (`2026-10-23`)[cite: 1, 2, 4].

### 2.3 Complete Audit of the 18 Claims (`G1–G18`) from the Outside Research Report

Every claim from `pelias_engine_hardware_verified_research_plan.md` (`2026-10-23`)[cite: 6] was cross-checked against the repository ledgers and primary vendor documentation (`PELIAS_TRAINER_MASTER_NOTES.md` §1.16 & `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` §16.4, `2026-10-23`):

| ID | Claim in Outside Report (`2026-10-23`)[cite: 6] | Verification Against Repo Data & Primary Sources (`2026-10-23`)

 | Label / Verdict | Disposition in Engine & Trainer Plans |
| --- | --- | --- | --- | --- |
| **G1** | Windows-native runs at `57.0 tok/s (17.54 ms/token)`, gap `6.67 ms` (`166.7 µs/layer`)[cite: 6]. | Repo ledger (`benchmarks/RESULTS.md` §9.4, `2026-10-23`) records `47.3 / 46.8 tok/s` short and `53.7 / 54.1 tok/s` at `12k`.

 | **Rejected (Not in repo)**<br> | Recomputed in Section 4.2 from the six logged runs.

 |
| **G2** | WDDM `VidMm` manages system memory strictly in `4-KB` pages (`DXGK_PTE`)[cite: 6]. | Primary Microsoft Learn page (*[GPU Virtual Address](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address)*) verified `2026-10-23`: *"System memory, which is always allocated at a 4-KB granularity"* / *"System memory is always managed in 4-KB pages"*.

 | **PUBLISHED (True)**<br> | Context for `P-E11`; does **not** explain the `9.59 vs. 26.2 GB/s` gap because the `26.2 GB/s` checksum kernel reads the exact same `4-KB` pinned pages (`DERIVED`, `2026-10-23`).

 |
| **G3** | [WSL `3.0.4` (`ce88638`)](https://github.com/microsoft/WSL/releases), kernel `6.18.54.1`, removed bundled `perf` (`#41904`), `64 KB` stack (`#41888`)[cite: 6]. | `wsl --version` in `raw/e0-2026-10-22/machine_state/` confirms WSL `3.0.4.0`, kernel `6.18.54.1-1` (`MEASURED`, `2026-10-22`); individual PR numbers unverified on fetched release page.

 | **Version MEASURED; PRs UNVERIFIED**<br> | No change needed: Engine E0/E2 already uses in-process `cudaEventRecord` timers (`DECISION`)[cite: 1, 4]. |
| **G4** | `4-KB` guest pages in `mmap` cause the `2×` prefill swing (`863` vs. `1,680 tok/s`); [`madvise(MADV_HUGEPAGE)`](https://docs.kernel.org/admin-guide/mm/transhuge.html) fixes it[cite: 6]. | `--load-mode none` is the **fast** mode (`1,667–1,680 tok/s`) and uses ordinary anonymous `4-KB` memory; `.wslconfig` already sets `transparent_hugepage=always`; CPU `2 MiB` page test (`02_ram_bw_test.cpp`) got `44.86 GB/s` vs. `45.81 GB/s` for `4 KiB` (`MEASURED`, Ledger §1, `2026-10-20`)[cite: 1, 2, 4]. | **Contradicted by Repo Data**<br> | Kept as Engine open unknown: inspect `/proc/meminfo` `AnonHugePages` per load mode (`E11` / `P-E11`)[cite: 1, 2, 4]. |
| **G5** | Windows: allocate cold bank with [`VirtualAlloc(MEM_LARGE_PAGES)`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc) + [`SeLockMemoryPrivilege`](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support), staging buffer with `cudaHostAlloc`[cite: 6]. | Win32 APIs exist as documented (`PUBLISHED`, verified `2026-10-23`); benefit on 5800X3D DTLB unmeasured on Windows[cite: 1, 6]. | **UNVERIFIED Hypothesis**<br> | Added as Windows-native A/B test if Windows-native is promoted (`Decision 4`).

 |
| **G6** | Pin 8 worker threads to the 8 physical Zen 3 cores (`1T` mode); Load/Store/Retire queues are split 50/50 in `2T` mode ([AMD Family 19h SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) §2.14, Table 3)[cite: 6]. | AMD Family 19h SOG (pub. `56665` rev. `3.00`) verified `2026-10-23`: §2.10.3 & §2.14 Table 3 confirm queue partitioning in SMT mode; stock `llama.cpp` ledger §8.8 shows 5–7 threads beat 10 (`MEASURED`, `2026-10-20`)[cite: 1, 6]. | **Partly PUBLISHED + MEASURED**<br> | Adopted as A/B test `E-A1` in Engine Phase E4 (`8` pinned physical cores vs. `16` SMT).

 |
| **G7** | Use `acquire`/`release` (`MOV`) instead of `seq_cst` (`LOCK XCHG`) on handoff flags; avoid FP instructions right after `LOCK` ([AMD Family 19h SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) §2.15)[cite: 6]. | AMD SOG §2.15 quote verified word-for-word on `2026-10-23`: *"Following a LOCKed instruction, refrain from using floating point instructions as long as possible"* (`PUBLISHED`)[cite: 1, 6]. | **PUBLISHED (True); benefit UNVERIFIED**<br> | Adopted as coding rule and A/B test `E-A2` in Engine Phase E6.

 |
| **G8** | Use `_mm_prefetch(..., _MM_HINT_NTA)` on cold expert streaming to protect the `96 MB` victim L3 cache[cite: 6]. | Zen 3 L3 is a victim cache (`PUBLISHED`, [AMD SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) §2.1/2.6, verified `2026-10-23`); speed effect unmeasured.

 | **UNVERIFIED Hypothesis**<br> | Adopted as A/B test `E-A3` in Engine Phase E4.

 |
| **G9** | `16-byte` (`uint4`) loads staged through `__shared__` memory lift GPU zero-copy compute reads from `9.59` to `26.2 GB/s`[cite: 6]. | This is the engine's own `wide_load.cu` (`E0`/`E5`) hypothesis; the outside report misattributed the author's measured numbers to the Ada whitepaper (`2026-10-23`)[cite: 1, 2, 4, 6]. | **UNVERIFIED Hypothesis**<br> | Tested by Probe `P-E10` (`wide_load.cu`, including `Q6_K` `210 B` block staging).

 |
| **G10** | Pin `16–24 MB` of the RTX 4070's `36 MB` L2 cache via [`cudaAccessPropertyPersisting`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)[cite: 6]. | CUDA API exists (`PUBLISHED`, verified `2026-10-23`); max persisting size under WDDM/WSL2 unmeasured.

 | **UNVERIFIED Hypothesis**<br> | A/B test `E-A4` in Engine E5 (query `cudaDevAttrMaxPersistingL2CacheSize` first).

 |
| **G11** | Hold [`/dev/cpu_dma_latency = 0`](https://docs.kernel.org/power/pm_qos_interface.html) on Linux; run `nvidia-smi -lgc 2475,2475`[cite: 6]. | First is standard Linux PM QoS (`PUBLISHED`, verified `2026-10-23`); **second (`2475 MHz`) would underclock the author's `~2,775–2,790 MHz` overclock (`X14`)**.

 | **Clock Lock REJECTED**<br> | `/dev/cpu_dma_latency` kept for Native Linux E1 only; `2475 MHz` lock rejected (`DECISION`).

 |
| **G12** | Native Linux projected at `72–86+ tok/s`[cite: 6]. | Zero native-Linux runs exist yet; Phase E1 measures them[cite: 1, 2, 4]. | **UNVERIFIED (Not used)**<br> | Replaced by actual Phase E1 measurements when run[cite: 1, 2, 4]. |
| **G13** | Gate thresholds (`≥22 / ≥25 GB/s`, `p99 ≤ 12 µs`) and Oct 23–Nov 22 calendar[cite: 6]. | Invented by the outside report; `PELIAS_ENGINE_LINUX_BLUEPRINT.md` (`2026-10-22`) has its own phase gates[cite: 1, 2, 4]. | **Not Adopted**<br> | Author sets all dates and post-E0/E1 gates (`DECISION`)[cite: 1, 2, 4]. |
| **G14** | RTX 4070 specs: 46 SMs, 5,888 cores, 192-bit bus, `504 GB/s`, 36 MB L2, `sm_89`[cite: 6]. | Matches [NVIDIA Ada GPU Architecture Whitepaper v2.0](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) and Trainer Scaffold §2.1 (`2026-10-23`).

 | **PUBLISHED (Consistent)**<br> | Locked in hardware fact base[cite: 1, 2, 4]. |
| **G15** | Pinned host memory capped at `≤ 32 GB` under WDDM[cite: 6]. | `cudaHostAlloc` proven to `20,448 MiB` on WSL2 despite `RLIMIT_MEMLOCK = 64 MiB` (`MEASURED`, `2026-10-20`); `>20.4 GiB` untested.

 | **UNVERIFIED above 20.4 GiB**<br> | Tested up to `32 GiB` by Probe `P-H7` and Engine `E11`.

 |
| **G16** | WSL2 runs at `44–54 tok/s` short and `43–49 tok/s` at `12k`[cite: 6]. | True for the best recorded cell (`7.5 GiB` bank, bypass 32, `-ub 2048`: `53.5 / 53.0 / 49.0 tok/s`, Ledger §8.2), not for Reference Cell A (`37.9 / 42.1 tok/s`, Ledger §9.2) (`MEASURED`, `2026-10-22`)[cite: 1, 2, 4]. | **MEASURED (Config-specific)**<br> | Both cells explicitly separated in Section 4.2.

 |
| **G17** | [`/dev/dxg` VMBus](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-faq) adds `25–35 µs/layer` (`1.0–1.4 ms/token`), explaining the Win11 vs. WSL2 gap[cite: 6]. | `1.0–1.4 ms` is at most **`19–27%`** of the measured **`5.02–5.27 ms/token`** (`5.1–5.2 ms`) Win11–WSL2 difference on Reference Cell A (`DERIVED`, `2026-10-23`).

 | **UNVERIFIED & Partial**<br> | Addressed via E0 handoff probe (`94 µs` sync vs. `13.3 µs` doorbell) and CUDA graphs[cite: 1, 2, 4]. |
| **G18** | Windows-native thread pinning + [`SetThreadInformation`](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadinformation) (`HighQoS`) + `timeBeginPeriod(1)` cuts `5–21%` spread[cite: 6]. | Win32 APIs exist (`PUBLISHED`, verified `2026-10-23`); launch-to-launch spread is `5–21%` (`MEASURED` on WSL2, Blueprint `2026-10-22`); Win11 pinning untested[cite: 1, 2, 4, 6]. | **UNVERIFIED Hypothesis**<br> | Tested only if Windows-native is promoted (`Decision 4`).

 |

---

## 3. Complete Derived Fact Ledger, Scripts & Probes (`PELIAS_TRAINER_MASTER_NOTES.md` §1.1–1.17)

Every architectural decision in `pelias-engine` and `pelias-trainer` rests on 17 derived findings documented in `PELIAS_TRAINER_MASTER_NOTES.md` (`2026-10-23`)[cite: 2] and `PELIAS_TRAINER_README_main.md` (`2026-10-20` to `2026-10-23`)[cite: 3]. Each entry below records the exact finding, the script or calculation that derived it, its honesty label, what depends on it, and the named probe that verifies any remaining unknown.

### 3.1 Master Table of All 17 Derived System & Model Findings

| ID | Finding & Exact Numerical Values | How It Was Derived (Script / Source / Formula) | Honesty Label & Date | Downstream Dependencies & Verification Probes |
| --- | --- | --- | --- | --- |
| **F1 (`1.1`)** | **Served Model Exact Byte & Tensor Map:** `753` tensors, **`20.22 GiB` (`21,713,463,296 B`)** in `Q4_K_M` (file dated `2026-10-20 01:08`; companion `Q5_K_M` is `25,347,532,800 B` / `23.61 GiB`, dated `2026-10-20 00:49`). Contains **`40` trunk layers (`blk.0`–`blk.39`) + `1` MTP block (`blk.40`, `block_count = 41`)**. Routed experts: `ffn_gate_exps` and `ffn_up_exps` are `Q4_K` (`144 MiB` each) in all `41` blocks; `ffn_down_exps` is `Q6_K` (`210 MiB` each $\rightarrow$ **`498 MiB`/layer**) in **`21` blocks** (`20` trunk layers `0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39` + `blk.40`) and `Q4_K` (`144 MiB` each $\rightarrow$ **`432 MiB`/layer**) in **`20` trunk blocks**. `lm_head` (`output.weight`) is `Q6_K`, token embedding (`token_embd.weight`) is `Q4_K`, and all routers (`ffn_gate_inp`, `41 × 2048 × 256 = 21,495,808` params, `82.0 MiB`/copy) are `F32`.

 | Parsed directly from the served GGUF binary header using `gguf_min.py` / pinned `llama.cpp` (`2ca15f540`) Python reader and tabulated by tensor name, shape, and `ggml_type`; zero reliance on external model cards[cite: 2, 4]. | **READ-SRC** (`2026-10-20` & `2026-10-23`)[cite: 2, 3] | Governs every memory table in Engine (`E3`, `E5`) and Trainer (`§2.2`, `§2.3`, `Appendix A`), the core byte budget (`D22`), and the byte-splice exporter (`io/gguf_splice`)[cite: 1, 2, 4]. |
| **F2 (`1.2`)** | **Hybrid Layer Topology (`30` DeltaNet + `10` Full GQA Attention):** Of the `40` trunk layers, **`30` are Gated DeltaNet** (linear attention) layers and **`10` are Full GQA Attention** layers located at indices **`3, 7, 11, 15, 19, 23, 27, 31, 35, 39`**[cite: 2]. Each layer has `256` routed experts, top-`8` active per token, expert intermediate width `512`, `1` always-on shared expert, hidden dimension `2,048`, and vocabulary `248,320`.

 | Read from `llama.cpp` (`2ca15f540`) graph builder `src/models/qwen35moe.cpp`, where `(layer + 1) % 4 != 0` selects DeltaNet, cross-checked against per-layer tensor names in `F1`[cite: 2]. | **READ-SRC** (`2026-10-23`)[cite: 2] | Sets the trainable LoRA module count (`F3`), separate E7/T6–T7 spine kernels (`attention.cu` vs. `deltanet.cu`), and the `P-E1` layer-type residency test (`F9`)[cite: 1, 2, 4]. |
| **F3 (`1.3`)** | **Trainable Spine ("Core") & LoRA State Footprint:** The trainable core (shared experts + DeltaNet projections + attention projections) comprises **`1,409.0 M` parameters**[cite: 2]. LoRA adapters attach to **`310` modules** (`30 × 3` DeltaNet projections + `10 × 4` attention projections + `40 × 3` shared-expert projections + `30 × 2` DeltaNet gates/norms). At rank $r = 16$ ([rsLoRA](https://arxiv.org/abs/2312.03732) $\alpha/\sqrt{r}$ scaling), trainable parameters total **`21.17 M`**, consuming **`323 MiB`** (`0.32 GiB`) across fp32 master weights, fp32 gradients, and fp32 Adam $m, v$ moments (`161 MiB` at $r=8$; `646 MiB` at $r=32$).

 | Scripted summation of parameter dimensions across all `310` target matrices from `F1` and `F2` ($\sum (d_{\text{in}} r + r d_{\text{out}}) \times 16\text{ bytes}$ for master + grad + $m$ + $v$)[cite: 2]. | **DERIVED** (`2026-10-23`)[cite: 2] | Anchors the `7.90 GiB` Trainer VRAM budget (`§4.1`, `§7`, `Appendix E`), optimizer step time, and shard merge memory.

 |
| **F4 (`1.4`)** | **The "Delta vs. Step" 4-Bit Merge Erase Problem:** When a learned LoRA update $\Delta W = \frac{\alpha}{\sqrt{r}} B A$ is merged into a base weight stored in `Q4_K` (`4.5` bits/weight), typical element updates fall inside a single `Q4_K` quantization step and are erased upon re-quantization to `Q4_K`[cite: 2]. Resolving this requires raising core tensor precision (`Q6_K` / `Q8_0`), training through the exact `ggml` quantizer via **Mode Q** (Quantization-Aware LoRA), or serving an unmerged adapter[cite: 2]. | Analytical step-size comparison against LoRA update magnitudes in Trainer Scaffold `§4.3` and `§4.10`[cite: 2]. | **DERIVED** (mechanism); magnitude on `Qwen3.6-35B-A3B` is **UNVERIFIED** (`2026-10-23`)[cite: 2] | Directly tested by **`P-E6`** (bit-exact `ggml` quantizer port), **`P-E7`** (do served weights sit cleanly on their quantization grid?), and **`P-E8`** (does Mode Q close the merge gap?)[cite: 2]. |
| **F5 (`1.5`)** | **Core Byte Sizes by Quantization Type & Retirement of the Blanket 8-Bit Core:** Exact byte sizes of the `1,409.0 M`-parameter core by format: Served mix = **`820.3 MiB`** (`~0.80 GiB`); All `Q6_K` (`6.5625 bpw`) = **`1,102.3 MiB` (`+282.0 MiB`)**; All `Q8_0` (`8.5 bpw`) = **`1,427.7 MiB` (`+607.4 MiB`)**; All `bf16` (`16 bpw`) = **`2,687.5 MiB`**. Raising individual core classes from served mix to `Q6_K` costs: shared experts **`+25.7 MiB`**, attention **`+65.5 MiB`**, DeltaNet out **`+61.9 MiB`**, DeltaNet in **`+127.8 MiB`**, $\alpha/\beta$ **`+1.0 MiB`** (total **`+282.0 MiB`**). Because `+607.4 MiB` (`Q8_0`) exceeds available VRAM headroom (`F6`), a blanket 8-bit core cannot fit next to the `7.5 GiB` bank[cite: 2].

 | Computed from exact `ggml` block sizes (`Q4_K` = `144 B / 256` = `4.5 bpw`; `Q6_K` = `210 B / 256` = `6.5625 bpw`; `Q8_0` = `34 B / 32` = `8.5 bpw`) applied to each tensor class in `F1`[cite: 2, 4]. | **DERIVED** (`2026-10-23`)[cite: 2] | Drives the per-class byte-budget solver (`D22`, `§4.4`, `§9.4`, `Appendix E`) and `Decision 2` / `Decision 3` (`P-E5`).

 |
| **F6 (`1.6`)** | **True VRAM Headroom ($\Delta_{\max}$) Under the `10.9 GiB` Ceiling:** With the **`10.9 GiB`** ceiling and a **`0.1 GiB`** safety margin (`10.8 GiB` usable), the `7.5 GiB` bank peaks at **`10.40–10.44 GiB`** at default prefill micro-batch (`-ub 4096`, stamps `130854`, `132648`), leaving **$\Delta_{\max} = \mathbf{0.36\text{ GiB}}$ (`369 MiB`)**; at **`-ub 2048`** (stamps `132907`, `141137`), it peaks at **`10.67–10.69 GiB`**, leaving **$\Delta_{\max} \approx \mathbf{0.11\text{ GiB}}$ (`113 MiB`)**. An `8.0 GiB` bank (`11.0–11.19 GiB` peak; `11.17 GiB` in blueprint risk table) and an `8.5 GiB` bank (`11.4 GiB` peak) violate the ceiling[cite: 1, 2, 4].

 | $\Delta_{\max} = 10.9\text{ GiB} - 0.1\text{ GiB} - \text{Peak}_{\text{MEASURED}}$ across `benchmarks/RESULTS.md` §8–9. At `-ub 2048` (`113 MiB` room), shared experts (`+25.7 MiB`) + attention (`+65.5 MiB`) + $\alpha/\beta$ (`+1.0 MiB`) = **`+92.2 MiB`** fit at `Q6_K` with `20.8 MiB` spare while DeltaNet uses Mode Q at `+0 B`.

 | **DERIVED** from **MEASURED** peaks (`2026-10-22` & `2026-10-23`)

 | Verified on the native engine arena allocator by **`P-E4`** and enforced end-to-end by **`P-X5`**.

 |
| **F7 (`1.7`)** | **Fixed Per-Step Expert Streaming Time (`1.49 s/step`) & Batch-Hiding Thresholds:** Streaming the `19,503,513,600 B` (`18.164 GiB` trunk / `18.65 GiB` with `blk.40`, `79` fetches = `35.8 GiB/step`) of frozen routed experts twice per optimizer step (forward + backward) at the measured **`26.1 GB/s`** PCIe floor takes **`1.49 s/step`** regardless of token batch size. The transfer is `100%` hidden behind GPU compute once a step holds **`1,495` tokens** (at `1,000 tok/s`), **`2,242` tokens** (at `1,500 tok/s`), or **`2,989` tokens** (at `2,000 tok/s`)[cite: 2].

 | $t_{\text{stream}} = \frac{2 \times 19.5035 \times 10^9\text{ B}}{26.1 \times 10^9\text{ B/s}} = 1.4945\text{ s}$; $N_{\text{hide}} = t_{\text{stream}} \times R_{\text{compute}}$[cite: 2]. | **DERIVED** from **MEASURED** `26.1 GB/s` floor (`2026-10-20` & `2026-10-23`)[cite: 2, 4] | Motivates the Layer-Major $K$-window streamer (`D5`, `bus/stage_streamer`, `§5`, `§13`).

 |
| **F8 (`1.8`)** | **Training & Forward FLOP/Token and Overclocked AD104 Peaks:** At $T = 8,192$, training requires **`20.39 GFLOP/token`** (forward is **`6.74 GFLOP/token`**; backward costs ~1 forward instead of 2 because frozen experts skip `wgrad` and only compute transposed-dequant `dgrad` `G5/G6`). RTX 4070 (`46` SMs) theoretical peaks: **`58.3 TFLOPS` bf16** at stock boost (**`65.4 TFLOPS`** at author's `~2,775–2,790 MHz` OC) and **`233.2 TOPS` int8** (**`261.4 TOPS`** at OC).

 | Operation-level FLOP summation in Trainer Scaffold `§6.10`; SM count (`46/128`) and clock ratio scaling from published [NVIDIA Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) specs.

 | **DERIVED** (`2026-10-23`); achieved kernel efficiency is **ESTIMATE / UNVERIFIED**[cite: 2] | Anchored by Wave 0 probes **`P-H3`** (measured tensor rates & clocks) and **`P-K1`–`P-K3**` (bf16, int8, and in-register dequant efficiency)[cite: 2]. |
| **F9 (`1.9`)** | **Full-Attention (`10`) vs. DeltaNet (`30`) Routing Equivalence (`P-E1`):** Across `84,503` fit and `36,657` held-out decode steps (`121,160` total tokens $\times 40$ layers $\times 8$ experts), the `7.5 GiB` bank (`4,170` experts) achieves **`0.8273`** overall byte-weighted hit rate and **`0.639`** any-miss layer share (`0.361` zero-miss share). Full-attention layers (`10`) have a **`0.8299`** hit rate and **`0.3636`** zero-miss share vs. **`0.8265`** hit rate and **`0.3602`** zero-miss share for the `30` DeltaNet layers[cite: 2]. Dedicating bank space to hold all `10` full-attention layers `100%` resident would consume **`4.605 GiB` (`4,715 MiB = 4 × 432 + 6 × 498`)** of the `7.5 GiB` bank and **reduce** overall hit rate from `0.8273` to **`0.7114`**[cite: 2]. Full-attention coverage vs. top-$k$ experts per layer: `64 → 0.118`, `128 → 0.519`, `192 → 0.831`, `224 → 0.941`, `256 → 1.0` (`90% / 95% / 99%` coverage requires **`210 / 227 / 247`** experts)[cite: 2].

 | Executed Python analysis script (`P-E1`, `2026-10-23`) over the engine's recorded routing trace (`collect_trace.py`)[cite: 1, 2, 4]. | **MEASURED** (trace) & **DERIVED** (analysis, `2026-10-23`)

 | Refutes the hypothesis that full-attention layers require priority residency; keeps full-attention priority strictly as gated optional experiment **Mode `L-Z**` with a hard kill criterion (`§16.5 B1.1`)[cite: 2]. |
| **F10 (`1.10`)** | **Router Locality Adaptation (`Mode L`) Cost, Literature & Exchange Rate:** Training only the `41 × 2048 × 256 = 21.5 M` fp32 router weights (`82.0 MiB`/copy) with frozen experts costs **`+0.32 GiB`** VRAM (grad + Adam $m,v$ + frozen reference router) and **`0.206%`** of training FLOPs (`41.9 MFLOP/token` `wgrad`)[cite: 2]. From the engine's bank-size sweep, the empirical exchange rate is **`~+1 tok/s` per `+0.01` hit-rate gain**[cite: 2]. Literature audit (`2026-10-23`): [*Cacheable by Design?* (arXiv:2608.18261)](https://arxiv.org/abs/2608.18261) found post-hoc untrained router clamping breaks its `1%` PPL gate (`+0.8–2.0%` PPL at `~30%` fewer misses; `+4.0–5.8%` at `50%` in §7.1 Table 6), whereas training cuts rerouting PPL penalty from `+7.7%` to `+0.9%` (§7.1), and [*ReMoE* (arXiv:2605.27081)](https://arxiv.org/abs/2605.27081) lifted adjacent-token reuse `27.3% → 34.5%` (`+26%` relative) with frozen experts.

 | Parameter/FLOP arithmetic (`§7.12`, `§16.5 B1`); exchange rate derived from Engine Ledger §8 bank sweep; full-text verification of `arXiv:2608.18261` and `arXiv:2605.27081` on `2026-10-23`.

 | Costs **DERIVED**; literature **PUBLISHED**; benefit on `Qwen3.6-35B-A3B` **UNVERIFIED** (`2026-10-23`)

 | Unlocks only if Wave 0 probes **`P-L1`** (Belady vs. static gap), **`P-L2`** (clamp curve), **`P-E2`**, and **`P-E3`** pass, and ships only if 4 pre-declared gates pass (`CE ≤ +1%`, `Δh ≥ +0.03`, domain breadth, live engine speedup `G-L4`).

 |
| **F11 (`1.11`)** | **Zero-Miss Layer Share ($Z_l$) vs. Byte-Weighted Hit Rate ($h$) Objective:** In `P-E1`, `36.1%` of layer-steps have `0` misses and `63.9%` have $\ge 1$ miss (`1` miss: `27.9%`; `2`: `16.5%`; `3`: `9.3%`; `4`: `5.1%`; `5`: `2.8%`; `6`: `1.4%`; `7`: `0.7%`; `8`: `0.3%`; conditional mean **`2.16` misses per missing layer**). If a layer's execution time $T_l(m) \approx t_{\text{hot}} + \mathbb{I}[m > 0](c_0 + c_1 m)$ has a large fixed wake/handoff cost $c_0 \gg c_1$, maximizing the zero-miss layer share $Z_l$ beats maximizing byte-weighted hit rate $h$.

 | Derived from the `P-E1` miss-count histogram (`1.3816` mean misses $\div 0.639 = 2.16$) and the E0 handoff structure (`§16.5 B1.7`).

 | Histogram **MEASURED / DERIVED**; $c_0$ vs. $c_1$ superiority **UNVERIFIED** (`2026-10-23`)

 | Decided directly by Probe **`P-E2`** (per-layer CUDA event timer regression of layer latency vs. miss count $m \in \{0..8\}$ in E0).

 |
| **F12 (`1.12`)** | **`Q6_K` `210 B` Block / `420 B` Row Geometry & Zero-Cost `2 MiB` GGUF Alignment:** In `ggml`, each `256`-weight `Q6_K` super-block is **`210 B`** (`ql[128] + qh[64] + scales[16] + d[2]`), so one `512`-element expert row is **`420 B` (`2` blocks; `420 mod 16 = 4`)**—not a multiple of `16 B` (`uint4`). Padding `420 B → 432 B` costs **`+2.86%`** per row (**`+120 MiB`** over the 20 trunk `Q6_K` layers = `0.65%` of expert bytes) and cannot live inside standard GGUF (`repack_gguf.py` owns any row padding). Conversely, all `123` (`41 × 3`) expert tensors (`144 MiB` or `210 MiB`) are exact multiples of **`2 MiB`**, so placing them last in the GGUF after a `<2 MiB` `pelias.pad` header entry aligns all `123` expert tensors to `2 MiB` boundaries at **zero tensor byte cost**.

 | Read from `ggml-common.h` (`block_q6_K`) in `llama.cpp 2ca15f540` and verified against tensor byte sizes in `F1` (`§16.5 B4`, `Appendix E`).

 | Block/tensor geometry **READ-SRC & DERIVED**; bandwidth gain **UNVERIFIED** (`2026-10-23`)

 | Audited by **`P-E9`** (tensor/row/block alignment check), **`P-E10`** (`wide_load.cu` `16-B` staged zero-copy reads: `9.59 → 26.2 GB/s`), and **`P-E11`** (`2 MiB` page effect)[cite: 1, 2, 4]. |
| **F13 (`1.13`)** | **Three-Tier KL Divergence Gates (`K1`, `K2`, `K3`) & Float Noise Floor ($\sigma_{\text{float}}$):** The KL divergence values **`0.017353` (batch 16)** and **`0.018286` (batch 1024)** (`MEASURED`, Engine KL Ledger) represent the floating-point reduction noise floor $\sigma_{\text{float}}$ of the **current served GGUF** across `16 × 1,024`-token chunks (`kl_gate.py`)[cite: 1, 2, 4]. Because a newly trained file has new weights, verification requires three distinct gates: **`K1`** (trainer vs. `llama.cpp` on the *same* GGUF), **`K2`** (exported GGUF vs. trainer reference), and **`K3`** (`pelias-engine` vs. `llama.cpp` on the exported GGUF).

 | Derived from Engine KL gate ledger (`kl_gate.py`) and Ruling H6 (`2026-10-22`: float-spread governs; strict `KL ≤ 0.005` / `top-1 ≥ 99%` logged for information)[cite: 1, 2, 4]. | Current file floor **MEASURED**; new file floor **UNVERIFIED** (`2026-10-23`)[cite: 1, 2, 4] | **`P-Q4a`** reproduces `0.017353 / 0.018286` on the current file; **`P-Q4b`** re-measures $\sigma_{\text{float}}$ on every newly exported GGUF (`§10`, `§16.5 B3`).

 |
| **F14 (`1.14`)** | **Bimodal Activation Quantization (`Q8_K` 256-elem CPU vs. `q8_1` 32-elem GPU):** Source audit of `llama.cpp 2ca15f540` proved that both CPU AVX2 and GPU CUDA K-quant dot-product kernels accumulate in **`int32` with `fp32` scales**, refuting an earlier hypothesis of fp16/bf16 accumulator drift[cite: 2]. The actual numerical difference between hot (GPU) and cold (CPU) expert execution is **input activation quantization granularity**: the CPU quantizes activations in **`256`-element super-blocks (`Q8_K`, `292 B`)** while the GPU quantizes in **`32`-element blocks (`q8_1`, `36 B`)**, compounded by router 8th-vs-9th logit near-ties.

 | Direct source inspection of `ggml-cpu` and `ggml-cuda` (`mmq` / `vecdotq`) in `llama.cpp 2ca15f540` (`§16.5 B5`)[cite: 2]. | Kernel arithmetic **READ-SRC**; outlier KL impact **UNVERIFIED** (`2026-10-23`)[cite: 2] | Verified by **`P-E13`** (accumulator/reduction tree audit), **`P-E3`** (8th-vs-9th router logit margin), and **`P-E12`** (correlation $r$ between outlier ratio $\rho = \max |
| **F15 (`1.15`)** | **Live Decode Speed vs. Modelled Ceiling Across All 6 Logged Runs:** Against the `10.87–11.34 ms/token` (`88.2–92.0 tok/s`) ceiling, WSL2 Cell A (`7.0 GiB`, `-ub 4096`) runs at **`37.9 tok/s` (`26.39 ms`, gap `15.05–15.52 ms`)** short and **`42.1 tok/s` (`23.75 ms`, gap `12.41–12.88 ms`)** at `12k`; Windows-native Cell A (`7.0 GiB`, `-ub 4096`) runs at **`47.3 / 46.8 tok/s` (`21.14 / 21.37 ms`, gap `9.80–10.50 ms`)** short and **`53.7 / 54.1 tok/s` (`18.62 / 18.48 ms`, gap `7.14–7.75 ms`)** at `12k`; WSL2 Best Cell (`7.5 GiB`, `-ub 2048`, gauntlet client) runs at **`53.5 / 53.0 / 49.0 tok/s`** at short / `4k` / `12k` (`18.69 / 18.87 / 20.41 ms`, gap `7.35–9.54 ms`).

 | Scripted calculation $\Delta t = \frac{1000}{\text{tok/s}} - t_{\text{ceiling}}$ and $\frac{\Delta t}{40\text{ layers}}$ across `benchmarks/RESULTS.md` §8.2, §9.2, §9.4 (`§16.4.1`).

 | **DERIVED** from **MEASURED** runs (`2026-10-22` & `2026-10-23`)

 | Full breakdown in Section 4 below; drives `Decision 4` (Windows-native promotion) and Phase E0/E1/E6/E8 gates[cite: 1, 2, 4]. |
| **F16 (`1.16`)** | **Audit of the 18 Outside Report Claims (`G1–G18`):** All 18 claims audited against repo ledgers and primary vendor specifications.

 | Cross-checked against `benchmarks/RESULTS.md`, [Microsoft WDDM docs](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address), [AMD Family 19h SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf), and [NVIDIA Ada docs](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) (`2026-10-23`).

 | Individual labels in Section 2.3 (`2026-10-23`)

 | Valid hypotheses converted into A/B tests `E-A1` through `E-A7` in Engine Blueprint Unknowns table.

 |
| **F17 (`1.17`)** | **Deliberately Excluded Designs & Quantitative Kill Proofs:** (1) *Expert consolidation* (`128` experts $\times 3 \times 2048 \times 512$ over `10` layers = **`4.03 B` params** $\rightarrow$ **`16.1 GB`** fp32 master + **`32.2 GB`** Adam moments = `48.3 GB`, impossible on 12 GB VRAM / 64 GB RAM)[cite: 2]; (2) *Inference-time router clamping* (excluded by author ruling & `arXiv:2608.18261` Table 6)[cite: 1, 2, 4]; (3) *v0.3 "zero-sync routing plan"* (left out after review showed it could not work as drafted)[cite: 3]; (4) *`mmq_dump_diag` / `mmq_dump_diag2` patches* (dropped by `2026-10-22` ruling)[cite: 4]; (5) *Bundled Python/PyTorch/PEFT/Unsloth inside the new C++/CUDA binary* (excluded by ground-up rule and AGPL/LGPL license hygiene)[cite: 2, 3]. | Parameter byte arithmetic (`§16.5 B1.6`), source/license audits (`2026-10-20` to `2026-10-23`), and author rulings (`2026-10-22`)[cite: 2, 3, 4]. | **DERIVED** & **DECISION** (`2026-10-20` to `2026-10-23`)[cite: 2, 3, 4] | Prevents wasted engineering cycles and keeps the Apache-2.0 repository clean[cite: 2, 3, 4]. |

### 3.2 Complete Inventory of Repository Scripts, Microbenchmarks, Patches, and Baseline Packages

Every derived or measured number in the Pelias record traces to a specific executable script, microbenchmark, patch, or baseline package[cite: 1, 2, 3, 4, 5]:

| Artifact / Script / Program | Repository & Location | Exact Function & Recorded Output | Status & Label |
| --- | --- | --- | --- |
| **`gguf_min.py` & `expert_layer_sizes.json**` | `pelias-engine/tools/` (reused in `pelias-trainer`)[cite: 4] | Reads GGUF v3 header (`753` tensors, `21,713,463,296 B`), verifies `block_count = 41`, and emits exact per-layer byte sizes (`498 MiB` for the 21 `Q6_K`-down blocks, `432 MiB` for the 20 `Q4_K`-down blocks)[cite: 1, 2, 4]. | **READ-SRC / MEASURED** (`2026-10-20` to `2026-10-23`)[cite: 2, 4] |
| **`collect_trace.py`** | `pelias-engine/tools/`[cite: 4] | Records router top-8 expert selections across `84,503` fit decode steps and `36,657` held-out decode steps (`121,160` tokens $\times 40$ layers $\times 8$ experts)[cite: 1, 2, 4]. | **MEASURED** (`2026-10-20` to `2026-10-23`)[cite: 1, 2, 4] |
| **`export_hot_set.py` & `PELIAS_HOT_SET` profiles** | `pelias-engine/tools/` & `profiles/`[cite: 4] | Solves water-filling across the `40` trunk layers (`84` to `161` experts/layer at `7.5 GiB` = `4,170` experts incl. `40` zero slots = `72.6 MiB`), achieving `0.8273` held-out byte-weighted hit rate (`0.806` at `7.0 GiB`); paired with C++ `hot_set_io` via `P-E14`[cite: 1, 2, 4]. | **DERIVED / SIMULATED** (`2026-10-20` to `2026-10-23`)[cite: 1, 2, 4] |
| **`kl_gate.py`** | `pelias-engine/tools/`[cite: 4] | Evaluates mean KL divergence and top-1 token agreement across `16` chunks of `1,024` tokens (`16,384` tokens) against untouched `llama.cpp 2ca15f540`; established stock float noise floor $\sigma_{\text{float}} = \mathbf{0.017353}$ (batch 16) and $\mathbf{0.018286}$ (batch 1024)[cite: 1, 2, 4]. | **MEASURED** (`2026-10-20` to `2026-10-22`)[cite: 1, 2, 4] |
| **`speed.py` & `p1_tests` (`run_floors.sh`, `run_base_cell.sh`, `run_sweep.sh`)** | `pelias-engine/tools/` & `scripts/`[cite: 4] | Benchmarks prompt prefill and token decode at `short` (`ctx0`), `4,000`, and `12,000` context depths (3 repeats, 2 warm-ups); logged launch-to-launch spread (`5%–21%`) and all WSL2 / Win11 cells in `benchmarks/RESULTS.md` §8–9[cite: 1, 2, 4]. | **MEASURED** (`2026-10-20` to `2026-10-23`)[cite: 1, 4] |
| **`repack_gguf.py`** | `pelias-engine/tools/` (`D25` / `R17`)[cite: 1, 4] | Engine-owned offline packer (`D25`): consumes standard or `--layout expert-last` GGUFs and optionally pads `Q6_K` rows from `420 B → 432 B` (`+120 MiB` over 20 layers) if `P-E10` proves row padding is needed[cite: 1, 2, 4]. | **DECISION / PLANNED** (gated on `P-E10`)[cite: 1, 2, 4] |
| **Microbench `02` (`02_ram_bw_test.cpp` / `bench/ram_bw.cpp`)** | `pelias-engine/bench/`[cite: 4] | Measures standalone Ryzen 7 5800X3D AVX2 cold-expert read bandwidth: **`43.8–45.9 GB/s`** (`45.81 GB/s` on `4 KiB` pages; `44.86 GB/s` on `2 MiB` pages)[cite: 1, 4]. | **MEASURED** (`2026-10-20`)[cite: 1, 4] |
| **Microbench `03` (`03_handoff.cu` / `bench/handoff.cu`)** | `pelias-engine/bench/`[cite: 4] | Measures per-layer CPU↔GPU coordination latency on WSL2: **`94 µs`** for `cudaStreamSynchronize`, **`13.3 µs` median / `18.7 µs` p99** for resident-kernel doorbell ping-pong, and **`3.4 µs`** one-way flag visibility[cite: 1, 4]. | **MEASURED** (`2026-10-20`)[cite: 1, 4] |
| **Microbench `04` (`04_dual_bus.cu` / `bench/dual_bus.cu`)** | `pelias-engine/bench/`[cite: 4] | Measures standalone PCIe zero-copy reads (**`26.1–26.2 GB/s`** checksum kernel; **`9.59 GB/s`** narrow-field compute kernel) and simultaneous CPU+GPU dual-bus reads (**`47.62 GB/s` = `26.56 GB/s` CPU + `21.06 GB/s` GPU**)[cite: 1, 4]. | **MEASURED** (`2026-10-20`)[cite: 1, 4] |
| **Microbenches `05` (`pin_modes.cu`), `06` (`overlap.cu`), `07` (`wide_load.cu`)** | `pelias-engine/bench/`[cite: 4] | Tests `cudaHostAlloc` (`20,448 MiB` proven on WSL2) vs. `cudaHostRegister` (`pin_modes.cu`), concurrent CPU AVX2 + GPU hot-expert compute (`overlap.cu`), and `16-byte` (`uint4`) shared-memory staged zero-copy reads (`wide_load.cu`, `P-E10`)[cite: 1, 4]. | `pin_modes` **MEASURED** (`2026-10-20`); `overlap` & `wide_load` **E0 PROBES**[cite: 1, 4] |
| **Carried `llama.cpp` Patches (`4` kept, `2` dropped) & E0 Instrumentation** | `pelias-engine/` reference tree[cite: 1, 4] | **Carried (`2026-10-22` ruling):** `mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu`[cite: 1, 4]. **Dropped:** `mmq_dump_diag`, `mmq_dump_diag2`[cite: 4]. **E0 measurement-only patches:** `e0_timeline.patch` (`P-E2`) and `hit_counter.patch` (live hit rate `0.7753–0.7793` at `7.0 GiB`)[cite: 1, 4]. | **DECISION & MEASURED** (`2026-10-22`)[cite: 1, 4] |
| **`pelias-trainer` Baseline Packages (`v0.1`, `v0.2.1`, `v0.3`, MTP Aligner, Tests)** | `pelias-trainer/` (`49` files, **`725,010 B` / `~708 KiB**`, counted `2026-10-20`)[cite: 1, 3] | `v0.2.1` (`peliasv0.2.tar`, `17` files `102,542 B` + launcher `166,308 B`, written `2026-10-19 01:52–03:11`): **Run 1** started `08:52` (`616` steps, `202.7 min`, median **`247.85 tok/s`**); **Run 2** continued (`114` steps, `37.9 min`, median **`56.45 tok/s`**); exported `Q5_K_M` (`23.61 GiB`, `2026-10-20 00:49`) and `Q4_K_M` (`20.22 GiB`, `2026-10-20 01:08`)[cite: 1, 3, 5]. `v0.1` (`15` files `84,381 B` + launcher `166,297 B`, alternate path without 8-bit layer 0 / grouped MM / fused Triton dequant). `v0.3` patch (`2` files `23,778 B`). MTP draft-head aligner (`6` files `90,903 B`, incl. C++ hidden-state dumper). CPU tests (`7` files `90,801 B`)[cite: 3]. | **MEASURED / RECORDED** (`2026-10-19` to `2026-10-20`)[cite: 1, 3, 5] |

---

## 4. Engine Cross-Check Against the Speed Ceiling (`PELIAS_ENGINE_CROSSCHECK_SPEED_TO_CEILING.md`)

This section serves directly as the verified content for `PELIAS_ENGINE_CROSSCHECK_SPEED_TO_CEILING.md`.

### 4.1 The Modelled Speed Ceiling (`88.2–92.0 tok/s`, `10.87–11.34 ms/token`)

The analytical ceiling model (`2026-10-20`, updated per the `2026-10-22` author ruling) sums three terms across the `40` trunk MoE layers of `Qwen3.6-35B-A3B` (`Q4_K_M`, `20.22 GiB`, `256` experts/layer, top-`8` active)[cite: 1, 2, 4]:

$$T_{\text{ceiling}} = T_{\text{spine}} + T_{\text{cold\_overlap}} + T_{\text{handoff}}$$

| Ceiling Model Input | Value | Honesty Label & Source |
| --- | --- | --- |
| **GPU Spine Time per Token ($T_{\text{spine}}$)** | **`7.73 ms`** (`7.4 ms` split discrepancy open; shown as range until E0 per-token timers close) | **ESTIMATE / ASSUMED** (Author ruling `2026-10-22`: re-derive from E0 timers; E0 WSL2 measured **`6.6 ms`** GPU compute with CUDA graphs ON and `2.75 ms` host CUDA split time vs. `8.04 ms` with CUDA graphs OFF)[cite: 1, 4] |
| **Hot Bank Residency ($h$)** | `7.5 GiB` bank (`4,170` experts incl. `40` zero slots = `72.6 MiB`) $\rightarrow$ **`0.8273`** byte-weighted hit rate (`0.83` rounded in blueprint) | **SIMULATED / DERIVED** (`P-E1` on `84,503` fit / `36,657` held-out decode steps, `2026-10-23`; live `7.0 GiB` bank measured **`0.7753–0.7793`** at 1-token vs. `0.806` simulated in E0 `2026-10-22`)[cite: 1, 2, 4] |
| **Cold Expert Bandwidth (Unified Bus)** | CPU alone: **`43.8–45.9 GB/s`**; PCIe zero-copy checksum kernel alone: **`26.1–26.2 GB/s`**; **Combined dual-bus: `47.62 GB/s` (`26.56 GB/s` CPU + `21.06 GB/s` GPU = `93.0%` of `51.2 GB/s` DDR4-3200 peak)** | **MEASURED** (Engine Ledger §1, microbenches `02` and `04`, `2026-10-20`)[cite: 1, 2, 4] |
| **Zero-Copy Compute Kernel Rate** | **`9.59 GB/s`** today vs. **`26.2 GB/s`** checksum kernel target | **MEASURED** (`9.59 GB/s` adds **`+0.47 ms/token`** to the ceiling: `10.87 ms → 11.34 ms`, `2026-10-20`)[cite: 1, 4] |
| **Cold Expert Read Time per Token ($T_{\text{cold\_overlap}}$)** | **`2.86 ms`** (at `26.2 GB/s` zero-copy share) to **`3.33 ms`** (at `9.59 GB/s`), assuming `100%` overlap between CPU cold compute and GPU hot expert compute | **DERIVED** (`2026-10-20`)[cite: 1, 4] |
| **Per-Layer Handoff Latency ($T_{\text{handoff}}$)** | **`0.25 ms/token`** (`~6.25 µs/layer` modelled; vs. **`13.3 µs`** median / **`18.7 µs`** p99 resident doorbell = `0.53–0.75 ms/token`, **`3.4 µs`** flag alone, **`94 µs`** stream sync = `3.76 ms/token`) | **ASSUMED** (`0.25 ms` in model) vs. **MEASURED** (`13.3 µs` doorbell / `94 µs` sync, microbench `03`, `2026-10-20`)[cite: 1, 4] |
| **Resulting Modelled Ceiling** | **`10.87 ms/token` (`92.0 tok/s`)** with `26.2 GB/s` zero-copy; **`11.34 ms/token` (`88.2 tok/s`)** with `9.59 GB/s` kernel (range **`88.2–90.9 tok/s`** under the `7.4 ms` ruling) | **DERIVED** (`2026-10-20` & `2026-10-22` author ruling)[cite: 1, 2, 4] |

### 4.2 Master Speed-to-Ceiling Comparison Table (All Recorded Runs)

Every row below is computed directly from the six logged runs in `benchmarks/RESULTS.md` ($\frac{1000}{\text{tok/s}}$ compared against `10.87–11.34 ms`, and divided by `40` layers) (`DERIVED` from `MEASURED` runs, Trainer Scaffold §16.4.1 & Master Notes §1.15, `2026-10-23`):

| Recorded Run & Configuration | Measuring Client | Context | Speed (`tok/s`) | Token Time (`ms/token`) | Total Gap to Ceiling (`10.87–11.34 ms`) | Per-Layer Gap (`µs/layer`) | Label & Ledger Ref |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1. WSL2 Reference Cell A** (`7.0 GiB` bank, bypass 32, `-ub 4096`, timer off) | Streaming client (§9.2) | Short (`ctx0`) | **`37.9`** | `26.39` (p50 `26.33`) | **`15.05–15.52 ms`** (`15.0–15.5 ms`) | **`376–388 µs`** | **MEASURED / DERIVED** (`2026-10-22`)

 |
| **2. WSL2 Reference Cell A** (`7.0 GiB` bank, bypass 32, `-ub 4096`, timer off) | Streaming client (§9.2) | `12k` | **`42.1`** | `23.75` (p50 `23.34`) | **`12.41–12.88 ms`** (`12.4–12.9 ms`) | **`310–322 µs`** | **MEASURED / DERIVED** (`2026-10-22`)

 |
| **3. Windows-Native Cell A (`r1 / r2`)** (`7.0 GiB` bank, bypass 32, `-ub 4096`) | Streaming client (§9.4) | Short (`ctx0`) | **`47.3 / 46.8`** | `21.14 / 21.37` (p50 `21.53`) | **`9.80–10.50 ms`** (`9.8–10.5 ms`) | **`245–262 µs`** | **MEASURED / DERIVED** (`2026-10-23`)

 |
| **4. Windows-Native Cell A (`r1 / r2`)** (`7.0 GiB` bank, bypass 32, `-ub 4096`) | Streaming client (§9.4) | `12k` | **`53.7 / 54.1`** | `18.62 / 18.48` (p50 `18.28`) | **`7.14–7.75 ms`** (`7.2–7.8 ms`) | **`179–194 µs`** | **MEASURED / DERIVED** (`2026-10-23`)

 |
| **5. WSL2 Best Recorded Cell** (`7.5 GiB` bank, bypass 32, `-ub 2048`) | Gauntlet client (§8.2) | Short / `4k` | **`53.5 / 53.0`** | `18.69 / 18.87` | **`7.35–7.82 ms`** (short) | **`184–196 µs`** (short) | **MEASURED / DERIVED** (`2026-10-22`)

 |
| **6. WSL2 Best Recorded Cell** (`7.5 GiB` bank, bypass 32, `-ub 2048`) | Gauntlet client (§8.2) | `12k` | **`49.0`** | `20.41` | **`9.07–9.54 ms`** (`12k`) | **`227–239 µs`** (`12k`) | **MEASURED / DERIVED** (`2026-10-22`)

 |

### 4.3 Six Verified Findings Explaining the Gap Between Live Runs and the Ceiling

1. **Zero CPU/GPU Overlap + Stream Sync Handoff in the `llama.cpp` Tier Build (`+13.8 ms` on WSL2 Cell A, `MEASURED`):**
In the E0 timeline breakdown (`2026-10-22`, `RUNBOOK.md` & `benchmarks/RESULTS.md` §9), the `24.6 ms` internal short-context token time on WSL2 Cell A exceeds the `10.87 ms` (`10.8 ms`) model by `+13.73–13.8 ms` because[cite: 1, 4]:
* CPU cold expert compute takes **`10.1 ms`** (`+7.24 ms` over the `2.86 ms` model) and executes **strictly serially (`0%` overlap)** with the **`6.6 ms`** GPU compute (`MEASURED`, `2026-10-22`)[cite: 1, 4].
* Stream-sync handoff and activation copies across `/dev/dxg` take **`5.1 ms`** (`4.6–5.0+ ms` timer-free; `+4.85 ms` over the `0.25 ms` model) plus **`2.8 ms`** of copy-back/launch overhead (`MEASURED`, `2026-10-22`)[cite: 1, 4].


2. **The `5.1–5.2 ms/token` Windows-Native vs. WSL2 Delta Matches the WSL2 Sync Handoff Cost (`MEASURED` / `DERIVED`):**
On Reference Cell A (identical `7.0 GiB` bank, bypass 32, `-ub 4096`, and streaming client), Windows 11 native is faster than WSL2 by **`5.25–5.02 ms/token` at short context** (`21.14 / 21.37 ms` vs. `26.39 ms`, **`+24.8% / +23.5%`**) and **`5.13–5.27 ms/token` at `12k` context** (`18.62 / 18.48 ms` vs. `23.75 ms`, **`+27.6% / +28.5%`**) (`MEASURED` / `DERIVED`, `2026-10-23`). That **`5.1–5.2 ms/token`** gain matches the **`4.6–5.1 ms/token`** timer-free sync-and-copy handoff overhead measured on WSL2 (`DERIVED`, `2026-10-23`)[cite: 1, 2, 4].


3. **Configuration Settings vs. OS Effect (`+15.6 tok/s` on WSL2 Alone, `MEASURED` / `DERIVED`):**
Moving WSL2 from Reference Cell A (`7.0 GiB` bank, `-ub 4096`: `37.9 tok/s` short) to the §8.2 cell (`7.5 GiB` bank, `-ub 2048`: `53.5 tok/s` short) gained **`+15.6 tok/s` (`-7.70 ms/token`)**, which is **`1.66×` larger** than the Windows-vs-WSL2 gain on Cell A (`+9.4 tok/s` short) (`DERIVED`, `2026-10-23`). However, at `12k` context the ranking flips: Windows-native Cell A (`53.7 / 54.1 tok/s`) beats WSL2 §8.2 (`49.0 tok/s`). Because the two cells used different measuring clients (streaming client in §9.2/§9.4 vs. gauntlet client in §8.2) and Windows-native has not yet been tested at `7.5 GiB` + `-ub 2048`, **the pure OS effect at optimal settings is currently unmeasured** (`PELIAS_TRAINER_MASTER_NOTES.md` §1.15, `2026-10-23`).


4. **The `63.9%` Any-Miss Share & Fixed Per-Layer Trip Cost (`P-E1` `MEASURED` / `DERIVED`, `P-E2` `UNVERIFIED`):**
While the `7.5 GiB` bank achieves a **`0.8273`** byte-weighted hit rate across `36,657` held-out decode steps (`P-E1`, `2026-10-23`), the per-layer miss count distribution is:


* `0` misses: **`0.361`** (`36.1%` of layer-steps are `100%` VRAM-resident).


* `1` miss: **`0.279`**; `2` misses: **`0.165`**; `3` misses: **`0.093`**; `4` misses: **`0.051`**; `5` misses: **`0.028`**; `6` misses: **`0.014`**; `7` misses: **`0.007`**; `8` misses: **`0.003`** (`MEASURED` trace / `DERIVED` histogram, `2026-10-23`).


* Consequently, **`63.9%` of layer-steps (`25.56` out of `40` trunk layers per token) incur at least one cold miss**. On the `36.1%` of layers with `0` misses, the CPU cold bus sits completely idle (unused bandwidth that cannot be banked for later layers), while on the `63.9%` of layers that miss, the conditional mean is **`2.16` misses per missing layer** (`1.3816 ÷ 0.639`, `DERIVED` from `P-E1`) and every missing layer pays the fixed CPU wake/handoff overhead $c_0$ (`P-E2`).




5. **`82` Splits/Token (`41 CPU + 41 CUDA`) Includes the 41st Block (`blk.40` MTP, `READ-SRC` / `MEASURED`):**
The served GGUF contains **`block_count = 41`** (`40` trunk layers + `1` MTP block `blk.40` with `498 MiB` of `Q6_K`-down routed experts) (`READ-SRC`, `2026-10-23`). E0 host split telemetry logged **`82 splits/token = 41 CPU + 41 CUDA`** (`MEASURED`, `2026-10-22`), whereas the `10.87 ms` ceiling model divides by `40` layers[cite: 1, 4]. Furthermore, when MTP speculative decoding (`draft 2/3` at a `7.0 GiB` bank) is enabled on WSL2, `12k` decode speed jumps from **`45.4 tok/s` (base) to `51.5–52.9 tok/s` (`MEASURED`)**.


6. **Non-16-Byte `Q6_K` Geometry (`420 B` rows / `210 B` blocks) & Narrow Zero-Copy Reads (`P-E9` `READ-SRC`, `P-E10` `UNVERIFIED`):**
In `20` trunk layers (`0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39`) plus `blk.40`, `ffn_down_exps` is `Q6_K` (`210 MiB` per layer, `498 MiB` total layer experts vs. `432 MiB` for the `20` `Q4_K`-down layers) (`READ-SRC`, `2026-10-23`). Each `Q6_K` block is **`210 B` (`210 mod 16 = 2`)** and each `512`-element row is **`420 B` (`2` blocks, `420 mod 16 = 4`)** (`READ-SRC`, `P-E9`, `2026-10-23`). Narrow field reads over PCIe achieve only **`9.59 GB/s`** vs. **`26.2 GB/s`** for the `uint4` checksum kernel (`MEASURED`, `2026-10-20`); whether `16-byte` shared-memory stream staging lifts compute reads to `≥20–26.2 GB/s` is tested by **`P-E10`**[cite: 1, 2, 4].



---

## 5. Engine Master Notes, Dependency Chains & Mandatory Blueprint Edits (`PELIAS_ENGINE_MASTER_NOTES.md`)

*This section provides the complete reasoning trail, invalidation matrix,probe registry, and 10-point synchronization patch for `PELIAS_ENGINE_MASTER_NOTES.md` and `PELIAS_ENGINE_LINUX_BLUEPRINT.md`[cite: 1, 2, 4].*

### 5.1 The 10 Mandatory Edits to `PELIAS_ENGINE_LINUX_BLUEPRINT.md` (From Trainer Scaffold §16.7 & Master Notes §3.4)

These 10 edits synchronize `PELIAS_ENGINE_LINUX_BLUEPRINT.md` (`2026-10-22`, commit `34f744f`)[cite: 4] with the verified findings from `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` (§16.7) and `PELIAS_TRAINER_MASTER_NOTES.md` (§3.4, `2026-10-23`):

1. **Scope Line ([`arXiv:2608.18261`](https://arxiv.org/abs/2608.18261) Correction):** Replace the blanket statement in *Goal and scope* that router changes are out of scope because *"a published study found every setting broke its 1% perplexity gate"*[cite: 4]. State accurately that **inference-time router clamping on an untrained model** is rejected (`+0.8% to +2.0%` PPL at `~30%` fewer misses and `+4.0% to +5.8%` at `50%` in [arXiv:2608.18261](https://arxiv.org/abs/2608.18261) §7.1 Table 6, verified `2026-10-23`), while **trainer-side router locality adaptation (`Mode L`)** is accepted as a gated input because §7.1 stacking showed training cuts the rerouting PPL cost from `+7.7%` to `+0.9%` and [*ReMoE* (arXiv:2605.27081)](https://arxiv.org/abs/2605.27081) demonstrated `+26%` relative adjacent-token reuse (`27.3% → 34.5%`) with frozen experts (`PUBLISHED`, verified `2026-10-23`).


2. **VRAM Budget & Headroom Table (`10.9 GiB` Ruling):** Add the measured VRAM peak and headroom table (`§16.3.1` of Trainer Scaffold & `§1.6` of Master Notes):


* Hard Ceiling: **`10.9 GiB`** (`DECISION`, ruled `2026-10-22`)[cite: 1, 2, 4].
* `8.0 GiB` bank peaks at **`11.0–11.19 GiB`** (`11.17 GiB` in blueprint risk table); `8.5 GiB` bank peaks at **`11.4 GiB`** $\rightarrow$ both exceed the `10.9 GiB` ceiling and do not fit (`MEASURED`, `2026-10-22`)[cite: 1, 2, 4].
* `7.5 GiB` bank (`4,170` experts incl. `72.6 MiB` of zero slots) peaks at **`10.40–10.44 GiB` at the default prefill micro-batch (`-ub 4096`)** (stamps `130854`, `132648`) and **`10.67–10.69 GiB` at `-ub 2048**` (stamps `132907`, `141137`, the setting that achieves `1,667–1,680 tok/s` prefill at `12k`) (`MEASURED`, `2026-10-22`)[cite: 1, 2, 4].
* Resulting $\Delta_{\max}$ for any extra trainer core bytes (with a `0.1 GiB` safety margin): **`0.36 GiB` (`369 MiB`) at default micro-batch** or **`~0.11 GiB` (`113 MiB`) at `-ub 2048**` (`DERIVED`; re-measured on the native engine allocator by **`P-E4`**).




3. **Inputs Table CPU Bandwidth Correction (`X8`):** Clarify in *Where the numbers stand* (`PELIAS_ENGINE_LINUX_BLUEPRINT.md`) that **standalone CPU read bandwidth is `43.8–45.9 GB/s` (`MEASURED`, microbench `02`, `2026-10-20`)**; the `26.6 GB/s` (`26.56 GB/s`) figure in the table is the CPU's concurrent share while the GPU simultaneously reads `21.06 GB/s` over PCIe (**`47.62 GB/s`** combined, `MEASURED`, microbench `04`, `2026-10-20`)[cite: 1, 4].
4. **Phase E3 Oracle Gate (`K1 / K2 / K3`):** Update the Phase E3 gate so the GGUF reader and oracle harness also ingest trainer-exported GGUFs (`merged-core.gguf` + `layout_manifest.json`) and evaluate the three distinct KL gates (**`K1`** trainer vs. `llama.cpp` on same GGUF; **`K2`** exported GGUF vs. trainer reference; **`K3`** engine vs. `llama.cpp` on exported GGUF) against the file's measured float spread $\sigma_{\text{float}}$ (**`P-Q4a`** reproduces `0.017353` at batch 16 and `0.018286` at batch 1024 on the current served file; **`P-Q4b`** re-measures on any newly trained file)[cite: 1, 2, 4].
5. **Phase E5 Slot Table & `Q6_K` 210-Byte Block Handling:** Specify that the E5 bank loader supports variable per-layer capacities (`84` to `161` experts/layer in the `7.5 GiB` water-fill) from the `PELIAS_HOT_SET` file, and that `wide_load.cu` (**`P-E10`**) must handle `Q6_K`'s `210 B` blocks / `420 B` rows via stream-aligned `16-byte` (`uint4`) shared-memory staging (or consume `432 B` row-padded packs from the engine's `repack_gguf.py` if `P-E10` proves padding is required)[cite: 1, 2, 4].
6. **Phase E11 Startup Probe (`THP` & `AnonHugePages`):** Add logging of `.wslconfig` (`transparent_hugepage=always swiotlb=65536`) and `/proc/meminfo` `AnonHugePages` during model load (**`P-E11`**) to test the true cause of the `2×` load-mode prefill swing (`863–929` vs. `1,667–1,680 tok/s` at `12k`)[cite: 1, 2, 4].
7. **Phase E12 Hot-Set Objective (`B1.7` / `P-E2`) & Mode L Extras:** Add the choice between byte-weighted hit maximization ($h$) and **zero-miss layer share ($Z_l$) maximization** (decided by **`P-E2`**'s measurement of fixed miss cost $c_0$ vs. marginal miss cost $c_1$), and add trainer Mode L candidates to the E12 gated extras[cite: 1, 2, 4].
8. **Unknowns Table A/B Additions (`E-A1` to `E-A7`):** Add the audited hypotheses from Section 2.3 (`E-A1` 8 physical cores `1T` vs. 16 SMT; `E-A2` `acquire`/`release` vs. `seq_cst`; `E-A3` `PREFETCHNTA`; `E-A4` L2 `cudaAccessPropertyPersisting`; `E-A5` `AnonHugePages` load-mode check; `E-A6` native Linux floors; `E-A7` pinned memory `>20.4 GiB`)[cite: 1, 2, 4, 6].
9. **New Section — Trainer Co-Design Contract:** Insert the two-way milestone handoff table (Section 5.3 below) and the rule that **the engine owns the packed layout and `repack_gguf.py` (`R17` / `D25`)**, while the trainer emits standard GGUFs (optionally with `--layout expert-last` so all `123` expert tensors sit on `2 MiB` boundaries at zero byte cost via a `<2 MiB` `pelias.pad` header entry, **`P-E9`**).


10. **Sources Section Update:** Add `benchmarks/RESULTS.md` sections 1–9.4, `raw/e0-2026-10-22/`, `raw/windows-native-2026-10-23/`, the full-text verification of [`arXiv:2608.18261`](https://arxiv.org/abs/2608.18261) and [`arXiv:2605.27081`](https://arxiv.org/abs/2605.27081) (`2026-10-23`), and `pelias_engine_hardware_verified_research_plan.md` (`2026-10-23`) marked explicitly as an audited hypothesis list rather than a primary measurement source[cite: 1, 2, 4, 6].

### 5.2 Dependency Graph & Invalidation Matrix (`PELIAS_TRAINER_MASTER_NOTES.md` §2.1–2.2 & `PELIAS_ENGINE_MASTER_NOTES.md` §2)

The dependency chain across both repositories is strictly directed (`A ──► B` means `B` requires `A`)[cite: 2]:

```text
served GGUF bytes (F1: 753 tensors, 20.22 GiB) ──► core size by type (F5: 820.3 / 1102.3 / 1427.7 MiB) ──► byte-budget solver (D22) ──► core export
                                                                ▲                                                      ▲
engine bank size (7.5 GiB, 4,170 experts) ─────────► VRAM headroom Δ_max (F6: 0.36 GiB / ~0.11 GiB) ───────────────────┘
        ▲
engine hit-rate-vs-bank sweep (MEASURED) ──────────► locality exchange rate (F10: ~+1 tok/s per +0.01 h) ──► Mode L gates (G-L1..G-L4)
        ▲                                                                        ▲
routing trace (P-E1, F9: 121,160 steps, h=0.8273, any-miss=0.639) ───────────────┴──► decides Mode L-Z & zero-miss lever Z_l (F11, P-E2)

expert stream bytes (F7: 18.164 GiB × 2) + PCIe BW (26.1 GB/s MEASURED) ──► 1.49 s/step (1,495–2,989 tok/step) ──► Layer-Major K-window scheduler (D5)
FLOP/token (F8: 20.39 GFLOP train / 6.74 fwd) + GPU peaks (P-H3) + kernel efficiency (P-K1..K3) ──► achieved tok/s ──► trainer speed targets

KL float spread σ_float (F13: 0.017353 / 0.018286, P-Q4a/b) + bimodal audit (F14, P-E12/13) ──► K1/K2/K3 gates ──► accepts Mode Q / Mode B / core types
Q6_K row geometry (F12: 210 B block / 420 B row, P-E9) + wide-load probe (P-E10) + THP probe (P-E11) ──► 2 MiB GGUF pad & engine repack_gguf.py (D25)

```

Whenever any left-hand input below changes, the dependent right-hand derivations must be re-run by script (`PELIAS_TRAINER_MASTER_NOTES.md` §2.2 & `pelias-research-update.md` §3.2):

| If this input changes... | ...then re-compute / re-run | Reason |
| --- | --- | --- |
| **Engine bank size (`7.5 GiB`) or non-bank VRAM (`P-E4`)** | `F6` ($\Delta_{\max}$: `0.36` / `~0.11 GiB`) $\rightarrow$ `F5` Trainer byte-budget solver (`D22`) $\rightarrow$ Core export types; `F10` value of locality; simulated & live hit rate $h$ | Moves both the core's byte room under `10.9 GiB` and the resident expert count (`4,170`).

 |
| **Prefill micro-batch (`-ub 4096` vs. `-ub 2048`)** | Peak VRAM (`10.40–10.44` vs. `10.67–10.69 GiB`), $\Delta_{\max}$ (`369 MiB` vs. `113 MiB`), and prefill speed (`1,667–1,680 tok/s` at `12k`) | Scratch activation buffers scale linearly with `-ub`[cite: 1, 2, 4]. |
| **Pinned `llama.cpp` oracle commit (`2ca15f540`)** | `F2` (layer rule), `F12` (`Q6_K` layout), `F14` (CPU/GPU accumulators `P-E13`), and the bit-exact quantizer port (`P-E6`) | Graph structure, block layouts, and reference kernels are read directly from commit `2ca15f540`[cite: 2, 4]. |
| **`P-E2` per-layer miss cost regression ($c_0$ vs. $c_1$)** | `export_hot_set.py` / C++ `hot_set_io` objective (byte-hit $h$ vs. zero-miss share $Z_l$) and Mode L layer weights $c_l$ | If $c_0 \gg c_1$, eliminating 1-miss layers beats raising average hit rate (`B1.7`).

 |
| **`P-E10` wide-load zero-copy rate (`9.59 → 26.2 GB/s`)** | Ceiling model cold-read term (`3.33 ms → 2.86 ms`), E12 zero-copy third-branch gate (`≥15 GB/s`), and whether `Q6_K` rows need `420 → 432 B` padding in `repack_gguf.py` | Decides cold-path PCIe compute bandwidth and packed layout[cite: 1, 2, 4]. |
| **Measured expert stream bandwidth (`26.1 GB/s`)** | `F7` (`1.49 s/step` transfer time and `1,495 / 2,242 / 2,989` tokens-per-step hiding thresholds) | Stream time equals $2 \times \text{bytes} \div \text{bandwidth}$[cite: 2]. |
| **Clock settings (overclock `~2,775–2,790 MHz` on or off)** | `F8` theoretical peak rates (`58.3` vs. `65.4 TFLOPS` bf16; `233.2` vs. `261.4 TOPS` int8) and `P-H3` | Tensor Core throughput scales linearly with SM clock frequency.

 |
| **Hard VRAM ceiling (`10.9 GiB`, author ruling)** | `F6` ($\Delta_{\max}$) and the runtime memory governor (`hal/vram_governor`, Trainer Chapter 15) | Every static and dynamic arena buffer is budgeted against `10.9 GiB`[cite: 1, 2, 4]. |
| **Engine adds `Q8_0` spine weight kernels (`P-E5`, `Decision 3`)** | `F5` (`Q8_0` candidate becomes selectable in `D22` for classes that fit within $\Delta_{\max}$) | The engine must have a kernel for every weight `ggml_type` exported by the trainer.

 |
| **Target OS selection (`Decision 4`: WSL2, Win11, Native Linux)** | Handoff floor (`13.3 µs` WSL2 doorbell vs. Win11 vs. Native Linux `E1`), pinned cap (`P-H7`/`E11`), and trainer default platform (`P-H11`) | Windows-native is `5.1–5.2 ms/token` faster at decode on Cell A (`MEASURED`, `2026-10-23`)[cite: 1, 2, 4]. |
| **Served GGUF weights (a newly trained export)** | `F1`, `F13` (`P-Q4b` $\sigma_{\text{float}}$ noise floor of the new file), `F9` (`P-E1` routing trace), and `PELIAS_HOT_SET` profile (`R18`) | New weights shift both logit rounding noise and router top-8 frequencies.

 |

### 5.3 The Two-Way Engine ↔ Trainer Milestone Contract (`PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` §16.7)

| Engine Milestone[cite: 1, 2, 4] | What `pelias-engine` Releases to `pelias-trainer`<br> | What `pelias-trainer` Releases / What `pelias-engine` Verifies

 |
| --- | --- | --- |
| **E0** (Per-layer timers) | `P-E2` ($c_0, c_1$ per miss & layer type) $\rightarrow$ layer weights $c_l$ in Mode L (`B1.2`)

 | —

 |
| **E1 / E11** (Platform floors) | Native Linux, WSL2, and Win11 floor table $\rightarrow$ platform default (`P-H11`)[cite: 1, 2, 4] | —

 |
| **E3** (Reader & Oracle) | Reproduction of stock KL floor (`0.017353` batch 16 / `0.018286` batch 1024, `B3`, `P-Q4a`)[cite: 1, 2, 4] | `layout_manifest.json` and `merged-core.gguf` parse identically; `K3` gate passes

 |
| **E5** (Bank & Wide-Load) | `P-E10` (`uint4` staging) & `P-E11` (`2 MiB` pages) $\rightarrow$ manifest flags & pack spec[cite: 1, 2, 4] | C++ `PELIAS_HOT_SET` writer matches `export_hot_set.py` byte-for-byte (`P-E14`)

 |
| **E7** (Full 40-Layer Spine) | `P-E5` supported weight type matrix (`Q4_K`, `Q6_K`, `F32`; optional `Q8_0`) & `P-E13` accumulator audit (`int32` dot + `fp32` scale)[cite: 1, 2, 4] | Trainer exports only supported core types (`D22`)

 |
| **E8 / E9** (Decode & VRAM Planner) | True $\Delta_{\max}$ (`P-E4`) at shipped prefill micro-batch[cite: 1, 2, 4] | Merged core fits under `10.9 GiB` at the `7.5 GiB` bank (`P-X5`) with `≤2%` decode regression

 |
| **E12** (Gated Extras) | MTP draft acceptance (`P-T8`) & live bank hit rate[cite: 1, 2, 4] | Re-profiled hot sets & Mode L candidates (`G-L4`, `G-Z4`)

 |

### 5.4 Master Verification Probe Registry (Wave 0 & Cross-Repository Probes)

In `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` (Chapter 11 & Chapter 16) and `PELIAS_TRAINER_MASTER_NOTES.md` (§3.2, §3.5), there are **99 named probes** (`46` Wave 0 probes that require zero trainer code), ensuring that every `ESTIMATE` (`23`) and `UNVERIFIED` (`19`) statement is converted into a `MEASURED` fact before the stage that relies on it[cite: 1, 2, 5]:

| Probe ID | Category | Exact Question Answered / Measurement Protocol | Target Gate / Dependent Decision |
| --- | --- | --- | --- |
| **`P-H3`** | Hardware (`Wave 0`) | What are the RTX 4070's real sustained `bf16` TFLOPS, `int8` TOPS, and SM clock frequencies under the `~2,775–2,790 MHz` overclock vs. stock (`F8`)?[cite: 2] | Re-bases `600 / 1,200 / 2,000 tok/s` trainer speed targets[cite: 2]. |
| **`P-H7`** | Hardware (`Wave 0`) | How much pinned host memory (`cudaHostAlloc` / `cudaHostRegister`) can be locked above the proven `20,448 MiB` up to `32 GiB` on WSL2, Win11, and Native Linux (`G15`)?

 | Sets host arena pool limit in `hal/host_alloc_*` (`E11`)[cite: 1, 2, 4]. |
| **`P-H8`** | Hardware (`Wave 0`) | How much free VRAM exists at process start across WSL2, Win11, and Native Linux before any engine/trainer allocation?[cite: 2] | Calibrates the `10.9 GiB` runtime VRAM governor (`hal/vram_governor`).

 |
| **`P-H11`** | Hardware (`Wave 0`) | What is the 3-way OS floor comparison (Native Linux vs. WSL2 vs. Win11) on DMA stream rate, doorbell handoff, and launch jitter?

 | Selects default platform (`Decision 4`, `E1`)[cite: 1, 2, 4]. |
| **`P-K1`–`P-K3**` | Kernels (`Wave 0`) | What fraction of `P-H3` peak do the `bf16` GEMM path (`P-K1`), `int8` path (`P-K2`), and [MARLIN](https://arxiv.org/abs/2408.11743)-style in-register dequant MMA path (`P-K3`, memory-bound vs. compute-bound) achieve (`F8`)?

 | Locks Stage T3/T4 forward and backward (`G1–G6`) kernel design.

 |
| **`P-Q4a` / `P-Q4b**` | Accuracy (`Wave 0` / Export) | Does our oracle harness reproduce `0.017353` (batch 16) and `0.018286` (batch 1024) on the current GGUF (`P-Q4a`), and what is $\sigma_{\text{float}}$ on a newly exported GGUF (`P-Q4b`, `F13`)?

 | Sets pass thresholds for `K1`, `K2`, `K3` and Engine Phase `E3`[cite: 1, 2, 4]. |
| **`P-S7`** | Spine (`Stage T7`) | Does the chunked `C=64` [WY/UT](https://arxiv.org/abs/2406.06484) [Gated DeltaNet](https://arxiv.org/abs/2412.06464) kernel match the sequential fp32 warp-recurrence oracle within float tolerance?

 | Rung-2 parity gate for both Engine prefill/decode (`E7`) and Trainer (`T7`)[cite: 1, 4]. |
| **`P-L1`–`P-L10**` | Locality (`Wave 0` / Mode L) | `P-L1`: Belady optimal vs. static water-fill hit gap; `P-L2`: untrained clamp PPL curve (`arXiv:2608.18261` check); `P-L3`–`P-L10`: router drift, entropy, and cross-domain generalization (`F10`).

 | Decides `Decision 5` (whether to build Mode L / Mode L-Z at all).

 |
| **`P-E1`** | Engine-Fit (**DONE**) | Analyzed `84,503` fit + `36,657` held-out decode steps: `h = 0.8273`, any-miss `0.639`, full-attn `0.8299` vs. DeltaNet `0.8265`, full-attn 100% residency drops $h$ to `0.7114` (`F9`).

 | **COMPLETED (`2026-10-23`, `MEASURED` / `DERIVED`)**.

 |
| **`P-E2`** | Engine-Fit (`Wave 0` / `E0`) | Regresses per-layer execution latency (`e0_timeline.patch`) against miss count $m \in \{0..8\}$ to extract fixed wake/handoff cost $c_0$ and marginal miss cost $c_1$ (`F11`).

 | Decides byte-weighted hit $h$ vs. zero-miss share $Z_l$ (`B1.7`, `E12`)[cite: 1, 2, 4]. |
| **`P-E3`** | Engine-Fit (`Wave 0`) | Measures the router logit margin between the 8th and 9th ranked expert across the trace (`F10`, `F14`).

 | Determines feasibility of low-KL router nudging (Mode L) and near-tie branch flips.

 |
| **`P-E4`** | Engine-Fit (`E8`/`E9`) | Measures exact non-bank VRAM consumption of the native `pelias-engine` arena across bank sizes and contexts (`F6`).

 | Replaces `llama.cpp`-derived $\Delta_{\max}$ (`0.36` / `~0.11 GiB`) in `D22`.

 |
| **`P-E5`** | Engine-Fit (`E7`) | Confirms the exact matrix of spine weight quantization types (`Q4_K`, `Q6_K`, `F32`, optional `Q8_0`) supported by `pelias-engine` (`F5`).

 | Governs `Decision 3` and allowed export types in `D22`.

 |
| **`P-E6`** | Engine-Fit (`Wave 0`) | Verifies that our standalone C++/CUDA port of `ggml` quantizers (`Q4_K`, `Q6_K`, `Q8_0`, `q8_1`, `Q8_K`) is bit-exact against `llama.cpp 2ca15f540` (`F4`).

 | Mandatory gate before Stage T3/T12 and Engine E4/E5[cite: 1, 2, 4]. |
| **`P-E7`** | Engine-Fit (`Wave 0`) | Dequantizes and re-quantizes the served `Q4_K_M` core tensors to check whether they sit cleanly on their quantization grid (`F4`)[cite: 2]. | Quantifies baseline round-trip quantization error before LoRA merge[cite: 2]. |
| **`P-E8`** | Engine-Fit (`Stage T12`) | Measures whether Mode Q (Quantization-Aware LoRA through the exact `ggml` quantizer) closes the merge degradation gap (`F4`)[cite: 2]. | Decides whether DeltaNet projections can stay at `Q4_K` (`+0 B`) under `-ub 2048`.

 |
| **`P-E9`** | Engine-Fit (`Wave 0`, **DONE**) | Audits alignment of all `753` tensors, `123` expert tensors (`2 MiB` multiples), and `Q6_K` rows (`420 B`) / blocks (`210 B`) (`F12`).

 | **COMPLETED (`2026-10-23`, `READ-SRC`)**; drives `--layout expert-last`.

 |
| **`P-E10`** | Engine-Fit (`E0` / `E5`) | Benchmarks `wide_load.cu` (`16-B` `uint4` shared-memory staging) on `Q4_K` (`144 B`) and `Q6_K` (`210 B`) cold experts (`9.59 → 26.2 GB/s`, `F12`, `G9`)[cite: 1, 2, 4]. | Decides whether `repack_gguf.py` (`D25`) must pad `Q6_K` rows `420 → 432 B`.

 |
| **`P-E11`** | Engine-Fit (`E0` / `E11`) | Logs `/proc/meminfo` `AnonHugePages` per load mode and tests whether `2 MiB` pages alter zero-copy or CPU read speed (`F12`, `G2`, `G4`)[cite: 1, 2, 4]. | Locks `host_alloc_wsl.cpp` and `host_alloc_linux.cpp` (`E11`)[cite: 1, 4, 6]. |
| **`P-E12`** | Engine-Fit (`Wave 0`) | Measures correlation $r$ between per-256-block activation outlier ratio $\rho = \max | x |
| **`P-E13`** | Engine-Fit (`Wave 0`, **DONE**) | Audits accumulator types and reduction trees in `llama.cpp 2ca15f540` (`int32` dot + `fp32` scale) and `pelias-engine` (`F14`).

 | **COMPLETED (`2026-10-23`, `READ-SRC`)** for `llama.cpp`; enforced in Engine E4/E5.

 |
| **`P-E14`** | Engine-Fit (`E5`) | Verifies that the C++ `PELIAS_HOT_SET` writer reproduces `export_hot_set.py` output byte-for-byte.

 | Locks shared `bank/hot_set_solver` between Engine and Trainer.

 |
| **`P-T8` / `P-T9` / `P-X5**` | End-to-End (`T13`–`T14`) | `P-T8`: MTP draft head overlap/agree rate (`blk.40`); `P-T9`: C++/CUDA trainer matches/beats `v0.2.1` on identical recipe; `P-X5`: exported GGUF fits `<10.9 GiB` at `7.5 GiB` bank with `≤2%` decode regression.

 | Final release gates for `pelias-trainer` and `pelias-engine`.

 |

### 5.5 The 6 Open Author Decisions (`PELIAS_TRAINER_MASTER_NOTES.md` §3.1)

1. **Is the `7.5 GiB` bank a hard ruling or a derived optimum?** Treated as `DERIVED` (largest `0.5 GiB` step under the `10.9 GiB` ceiling; `8.0 GiB` peaks at `11.0–11.19 GiB`).


2. **Which prefill micro-batch (`-ub`) ships in production?** Sets $\Delta_{\max}$ to **`0.36 GiB` (`369 MiB`)** at default `-ub 4096` (where all-`Q6_K` core `+282.0 MiB` fits with `87 MiB` spare) vs. **`~0.11 GiB` (`113 MiB`)** at `-ub 2048` (where shared experts `+25.7 MiB` + attention `+65.5 MiB` + $\alpha/\beta$ `+1.0 MiB` = `+92.2 MiB` fit at `Q6_K` with `20.8 MiB` spare, and DeltaNet projections use Mode Q at `0` extra bytes).


3. **Will `pelias-engine` add `Q8_0` weight kernels for the spine?** Required if any core class is exported as `Q8_0` (`P-E5`); today the engine plans `Q4_K`/`Q6_K`/`F32` weights and `q8_0` KV cache[cite: 1, 2, 4].
4. **Promote Windows-native from `"not in scope"` to a supported build target?** Windows-native is `5.1–5.2 ms/token` (`+24–28%`) faster at decode on Cell A (`MEASURED`, 2 launches; needs a 3rd launch per side to lock as a formal claim)[cite: 1, 2, 4].
5. **Is Mode L / L-Z worth building after Wave 0 probes?** Decided after `P-L1` (Belady vs. static gap), `P-L2` (untrained clamp cost curve), `P-E2` (miss cost by layer), and `P-E3` (8th-vs-9th logit margins) run in Wave 0 with zero trainer code.


6. **Are there any Windows-native runs newer than `raw/windows-native-2026-10-23/`?** Until a newer raw log is added, `47.3 / 46.8 tok/s` short and `53.7 / 54.1 tok/s` at `12k` are the sole official Windows-native numbers.



---

## 6. Plain-English Summary for `pelias-engine` (`README_PLAIN_ENGLISH.md` Ready Draft)

*Drafted to pair with `PELIAS_TRAINER_README_PLAIN_ENGLISH.md` (`2026-10-23`)[cite: 5], `PELIAS_TRAINER_README_main.md` (`2026-10-20` to `2026-10-23`)[cite: 3], and `PELIAS_ENGINE_LINUX_BLUEPRINT.md` (`2026-10-22`)[cite: 4], obeying Standing Rule 7 (no claims of novelty; every number carries its honesty label)[cite: 1, 3, 5].*

> ### Pelias-Engine in Plain English
> 
> 
> **What problem does it solve?**
> Large AI language models usually require expensive datacenter graphics cards because the entire model must sit inside fast graphics memory (VRAM) to run quickly[cite: 1, 5]. Our target model, **Qwen3.6-35B-A3B**, has 35 billion parameters and takes up **20.22 GiB** on disk (`READ-SRC`, `2026-10-20`). A consumer **NVIDIA RTX 4070** graphics card has only **12 GB** of VRAM, and to keep the system safe while Windows is running, we enforce a strict ceiling of **10.9 GiB** (`DECISION`, `2026-10-22`)[cite: 1, 2, 4].
> 
> 
> **How does Pelias fit a 20.22 GiB model into a 10.9 GiB card without slowing to a crawl?**
> Qwen3.6-35B-A3B is a **Mixture-of-Experts (MoE)** model[cite: 1, 3, 5]. Although it contains 256 specialist "experts" in each of its 40 layers (plus 1 multi-token prediction layer), it only uses **8 experts per layer** for any single word (token), alongside a small **1.41-billion-parameter "spine"** (attention, DeltaNet, and shared experts) that runs on every token (`READ-SRC`, `2026-10-23`)[cite: 1, 2, 5].
> Instead of splitting the layers clumsily between the graphics card and the processor, **Pelias-Engine turns a standard consumer PC into a unified two-bus system**:
> 
> 
> 1. **The Spine and the "Hot Bank" Live on the GPU:** The always-on spine (`~0.80 GiB`) and a pre-profiled bank of the **4,170 most popular experts (`7.5 GiB`)** stay permanently inside the RTX 4070's fast VRAM (`READ-SRC` / `DERIVED`, `2026-10-23`). Tracing 121,160 generated tokens showed that this `7.5 GiB` bank handles **82.73%** of all expert work (`MEASURED` trace / `DERIVED` split, `2026-10-23`).
> 
> 
> 2. **The "Cold" Experts Live in Pinned System RAM:** The remaining less-used experts stay in the computer's 64 GB of DDR4-3200 system memory (`DECISION`, `2026-10-20`)[cite: 1, 2, 4].
> 3. **Both Buses Work at the Same Time:** Whenever a token needs an expert that is not in the GPU's hot bank, the **AMD Ryzen 7 5800X3D** processor computes its share of cold experts using AVX2 instructions while the **RTX 4070** runs the hot experts and can read cold experts directly across the PCIe 4.0 x16 bus (`MEASURED` / `DECISION`, `2026-10-20`)[cite: 1, 2, 4]. Measured together on this machine, the CPU (`26.56 GB/s`) and GPU (`21.06 GB/s`) read system memory at a combined **47.62 GB/s**—**93%** of the physical speed limit of DDR4-3200 memory (`MEASURED`, `2026-10-20`)[cite: 1, 3, 4].
> 4. **A 13.3-Microsecond Doorbell:** Normally, making the GPU and CPU wait on each other 40 times per token wastes **4.6 to 5.1 milliseconds per token** in operating-system overhead (`MEASURED` on WSL2, `2026-10-22`)[cite: 1, 4]. Pelias replaces slow operating-system waits with a shared-memory "doorbell" flag that takes **13.3 microseconds** (median, `18.7 µs` p99, `MEASURED`, `2026-10-20`)[cite: 1, 4].
> 
> 
> **Where do the numbers stand today?**
> * **Hardware Ceiling Model:** **88.2 to 92.0 tokens/second** (`10.87–11.34 ms/token`, `DERIVED` from measured bandwidths and three inputs currently being closed in Phase E0, `2026-10-20` to `2026-10-22`)[cite: 1, 2, 4].
> * **Current Prototype (built on `llama.cpp`):**
> * **WSL2 Linux (`7.5 GiB` bank, `-ub 2048`):** **53.5 tok/s** on short prompts, **53.0 tok/s** at 4k context, and **49.0 tok/s** at 12k context (`MEASURED`, `2026-10-22`).
> 
> 
> * **WSL2 Reference Cell (`7.0 GiB` bank, `-ub 4096`):** **37.9 tok/s** short and **42.1 tok/s** at 12k context (`MEASURED`, `2026-10-22`).
> 
> 
> * **Windows 11 Native (same `7.0 GiB` Reference Cell):** **47.3 / 46.8 tok/s** short and **53.7 / 54.1 tok/s** at 12k context (`MEASURED`, `2026-10-23`), plus **51.5 to 52.9 tok/s** at 12k on WSL2 when multi-token speculative decoding (MTP) is turned on (`MEASURED`, `2026-10-22`).
> 
> 
> 
> 
> * **Accuracy Guarantee:** Every step is checked against an untouched reference build of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (commit `2ca15f540`)[cite: 1, 2, 4]. An output is accepted only if its divergence (KL) stays within `llama.cpp`'s own floating-point batch noise floor (**0.017353** at batch 16 and **0.018286** at batch 1024, `MEASURED` / `DECISION`, `2026-10-22`)[cite: 1, 2, 4].
> 
> 

---

## 7. Master Pelias Blueprint: Ground-Up C++20/CUDA Co-Existing Application & Universal Substrate

### 7.1 Why Unifying `pelias-engine` and `pelias-trainer` in C++20/CUDA Is Architecturally Exact

In `PELIAS_TRAINER_README_main.md` (§5–6, `2026-10-20`), the two repositories were separated only because the `v0.2.1` trainer was written in Python/PyTorch (`49` files, `725,010 B` / `~708 KiB` of source code) while the engine was C++/CUDA[cite: 1, 3]. Now that both are ground-up **C++20 + CUDA 13.0.88 (`sm_89`)** designs targeting the exact same machine, model (`Qwen3.6-35B-A3B`), GGUF file (`753` tensors, `21,713,463,296 B`), and oracle (`llama.cpp 2ca15f540`) (`2026-10-23`)[cite: 1, 2, 4], building them as a single cohesive codebase (`pelias`) with a shared core library (`libpelias_core` + `libpelias_substrate`) provides five concrete engineering advantages:

1. **Zero Forward-Pass Divergence (`K1` = `E7` by Construction):**
The engine's Phase E7 spine (`attention.cu`, `deltanet.cu`, `norm.cu`, `router.cu`, `lm_head.cu`)[cite: 4] and the trainer's Stages T3–T8 forward pass (`G1/G2` dense GEMMs, `E1–E5` norms/SwiGLU, `R1` fp32 router, `A1/A2` IMRoPE + GQA attention, `D1–D5/D7` DeltaNet) compute the **exact same mathematical function** (`§2.4` of Trainer Scaffold). Compiling them from one `src/core/spine/` directory guarantees that what is trained is bit-for-bit what is served[cite: 1, 2, 4].


2. **One `10.9 GiB` VRAM Governor, Arena Allocator, and Tri-OS Host Memory Layer (`hal/`):**
Both applications enforce the **`10.9 GiB`** ceiling (`Governor`), ban dynamic `cudaMalloc` inside the hot loop (`Arena`), and use `cudaHostAlloc` on WSL2 (`20,448 MiB` proven, where `cudaHostRegister` returns a device pointer $\neq$ host pointer)[cite: 1, 2, 4]. One `hal/` module serves both the Engine's **`7.5 GiB` static bank (`10.40–10.69 GiB` peak)** and the Trainer's **`7.90 GiB` streaming layout** (`2 × 498 MiB` expert slots + `0.80 GiB` spine + `0.32 GiB` LoRA/Adam + `2.50 GiB` working set).


3. **One Bit-Exact Quantizer/Dequantizer Suite (`block_types.h`, `G8`, `G9`, `P-E6`):**
Both the engine (`E4/E5`) and the trainer (`T3/T12` and Mode Q `§4.10`) require bit-exact ports of `ggml`'s `Q4_K` (`144 B`), `Q6_K` (`210 B`), `Q8_0` (`34 B`), `q8_1` (`36 B` GPU `32`-element activation block), and `Q8_K` (`292 B` CPU `256`-element activation block)[cite: 1, 2, 4]. Implementing **Mode B (bimodal-faithful forward, `D24`)** in the trainer becomes immediate when the engine's CPU `Q8_K` and GPU `q8_1` activation quantizers live in the shared core.


4. **In-Memory Closed Loop for Mode L Router Training & Hot-Set Profiling (`D21`, `D25`):**
The routing `Tape` (`40 layers × N tokens × top-8`) and the C++ `PELIAS_HOT_SET` water-filling / zero-miss solver (`P-E14`) are shared between `pelias trace` (engine) and `pelias train` / `pelias export` (trainer)[cite: 1, 2, 4]. After training finishes, the unified binary can switch from `TrainerArena` to `EngineArena` in-process and run the `P-X5` live decode speed and `K3` KL gates immediately[cite: 1, 2, 4].
5. **Single Wave 0 + E0/E1 Probe & Oracle Harness (`pelias probe` & `pelias verify`):**
Unifies the trainer's `46` Wave 0 probes (`P-H`, `P-M`, `P-K`, `P-Q`, `P-S`, `P-D`, `P-L`, `P-E`) with the engine's E0/E1 microbenchmarks (`02_ram_bw`, `03_handoff`, `04_dual_bus`, `05_pin_modes`, `06_overlap`, `07_wide_load`)[cite: 1, 4], writing one `probe_constants.json` and machine-state hash.



### 7.2 Master Co-Existing Architecture & Universal Substrate Diagram

To achieve the stretch goal of making Pelias **universal across consumer hardware, MoE LLMs, dense LLMs, and non-LLM sparse workloads** (`PELIAS_TRAINER_README_main.md` Footnote 1 & `PELIAS_ENGINE_LINUX_BLUEPRINT.md` Phase E13)[cite: 1, 3, 4], the C++20/CUDA architecture separates **Layer 0 (`libpelias_substrate`: The Domain-Agnostic Unified-Bus Hardware Abstraction)** from **Layer 1 (`libpelias_core`: Shared Model I/O, Spine Kernels & Verification Oracle)** and **Layer 2 (`pelias-engine` & `pelias-trainer` via zero-overhead C++20 Concepts)**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                         PELIAS MASTER C++20 / CUDA 13.0.88 APPLICATION                      │
│              One Binary (`pelias`) · Three Targets (`libpelias_core`, `engine`, `trainer`)  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 0: `libpelias_substrate` (Universal Consumer Unified-Bus Hardware Abstraction)        │
│   ├─ `hal/vram_governor` & `arena`: Hard 10.9 GiB ceiling, zero-malloc step/token loops     │
│   ├─ `hal/host_alloc_{linux,wsl,win32}`: Pinned DMA pools, 2 MiB THP / LargePage probes     │
│   ├─ `bus/doorbell_handoff`: 13.3 µs lock-free acquire/release + warp `__nanosleep` sync    │
│   ├─ `bus/cpu_threadpool`: Physical core (`1T`) pinning, `PREFETCHNTA`, zero-`LOCK` FMA     │
│   ├─ `bus/stage_streamer`: 3-stream DMA (`S_compute`, `S_w` H2D, `S_x` boundary D2H/H2D)    │
│   ├─ `bank/hot_set_solver`: Water-fill (byte-hit h) & Zero-Miss (Z_l) residency optimizer   │
│   └─ `quant/ggml_exact`: Bit-exact Q4_K, Q6_K, Q8_0, q8_1 (32), Q8_K (256) CPU/GPU kernels │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: `libpelias_core` (Shared Model I/O, Spine Kernels & Verification Oracle)           │
│   ├─ `io/gguf_splice`: 753-tensor mmap reader, byte-splice exporter, SHA-256 payload proof, │
│   │                    expert-last 2 MiB alignment (`pelias.pad`), `layout_manifest.json`   │
│   ├─ `tok/libllama_bridge`: Exact GGUF tokenizer & Jinja chat-template (`reasoning_content`)│
│   ├─ `spine/forward_kernels`:                                                               │
│   │    • RMSNorm (+1 folded), Gated RMSNorm (`ssm_norm`), per-head L2Norm (`eps = 1e-6`)    │
│   │    • Attention (`10` trunk + `blk.40`): joint Q+gate, IMRoPE `[11,11,10,0]`, GQA `256`  │
│   │    • DeltaNet (`30` layers): 4-tap causal conv1d + SiLU, chunked `C=64` WY/UT inverse   │
│   │      (used by BOTH Engine Prefill & Trainer Forward) + sequential fp32 warp recurrence  │
│   │      (used by BOTH Engine Decode & Trainer Rung-2 Oracle `P-S7`!)                       │
│   │    • Router (`41 × 2048 × 256` F32): fp32 FMA top-8 argmax, 8-way softmax, `Tape` I/O   │
│   └─ `verify/oracle_2ca15f540`: Evaluates K1 (Parity), K2 (Export), K3 (Engine) vs σ_float  │
├──────────────────────────────────────────┬──────────────────────────────────────────────────┤
│ LAYER 2A: `pelias-engine` (Inference)    │ LAYER 2B: `pelias-trainer` (Ground-Up LoRA/QAT)  │
│   • Execution: Token-Major, Dual-Bus     │   • Execution: Layer-Major, K windows/sweep      │
│   • VRAM Bank: 7.5 GiB (4,170 experts)   │   • Pinned Store: 18.65 GiB experts (79 fetches, │
│   • Hot Branch: GPU `expert_q4k/q6k.cu`  │     35.8 GiB/step = 1.49s @ 26.1 GB/s, hidden)   │
│   • Cold Branch: Parallel CPU AVX2 +     │   • Core: 310 LoRA modules (21.17M params, fp32) │
│     GPU 16-B (`uint4`) staged zero-copy  │   • Backward: Transposed-dequant dgrad (`G5/G6`),│
│   • State: `q8_0` KV cache + GDN state   │     FA2 bwd (`A3`), DeltaNet reverse-state (`D6`)│
│   • Decode: CUDA-graphed spine + MTP     │   • Loss: Chunked 256-pos fp32 CE, step-wide norm│
│   • Server: OpenAI HTTP + VS Copilot     │   • Gated Modes: Mode Q, Mode L/L-Z/L-C, Mode B/N│
└──────────────────────────────────────────┴──────────────────────────────────────────────────┘

```

### 7.3 How the Unified-Bus Substrate Scales Universally Beyond Qwen3.6 and Beyond LLMs

Because **Layer 0 (`libpelias_substrate`)** is parameterized over any staged computation graph with conditional block activation and auto-calibrates itself to any consumer PC via the startup capability probe (`E11` / `pelias probe`)[cite: 1, 3, 4]:

1. **Universal Hardware Portability Across Any Consumer PC (`E11` / `hal/`):**
* On any consumer desktop or workstation (PCIe 3.0/4.0/5.0, DDR4/DDR5, 8 GB–24 GB GPUs, AMD Zen or Intel Core CPUs across Native Linux, WSL2, or Windows 11), `pelias probe` measures the machine's exact standalone CPU RAM bandwidth ($B_{\text{CPU}}$), standalone PCIe zero-copy bandwidth ($B_{\text{GPU}}$), concurrent dual-bus ceiling ($B_{\text{dual}}$), doorbell latency ($t_{\text{doorbell}}$), and usable VRAM ceiling ($V_{\max}$)[cite: 1, 2, 4].
* Those measured constants automatically parameterize the `HotSetSolver` water-fill capacity, the CPU/GPU cold-branch split ratio, and the trainer's compute-hiding batch threshold $N_{\text{hide}} = \frac{2 \times \text{Bytes}_{\text{stream}}}{B_{\text{GPU}}} \times R_{\text{compute}}$[cite: 1, 2, 4].


2. **Other MoE LLMs (`E13` Plugin Interface):**
* Any Mixture-of-Experts architecture (such as `Gemma 4 26B-A4B`, `Qwen3-Coder`, or DeepSeek MoE variants mentioned in `PELIAS_TRAINER_README_main.md` §10) plugs into Layer 1 by declaring its `ModelSpec` (layer pattern, expert dimensions, top-$k$ routing rule, and quantization types) without touching a single line of Layer 0 unified-bus code[cite: 1, 3, 4].


3. **Dense LLMs (`PELIAS_TRAINER_README_main.md` Footnote 1 Stretch Goal):**
* For a dense LLM (`12B`–`70B`) that exceeds `10.9 GiB` VRAM, the trainer's **Layer-Major Streamer (`bus/stage_streamer`, §5.4–5.6)** works out of the box—streaming frozen quantized dense FFN/projection layers into the two `498 MiB` VRAM slots over $K$ micro-batches while training LoRA adapters in VRAM.


* On the inference side, `pelias-engine` partitions dense FFN weight matrices into static column/row tiles split across the `504 GB/s` VRAM hot bank and the `47.62 GB/s` CPU+PCIe unified bus, coordinated by the same **`13.3 µs`** doorbell handoff[cite: 1, 3, 4].


4. **Non-LLM Sparse & Out-of-Core Workloads:**
* Any domain where a `20–64 GB` state or parameter graph sits in system RAM and is queried sparsely per step can execute directly on `libpelias_substrate` using `HotColdBankManager`, `DoorbellCoordinator` (`13.3 µs`), and `StageMajorStreamer` (`26.1 GB/s`) without modification[cite: 1, 4]:
* **Out-of-Core Graph Neural Networks (GNNs) & Knowledge Graphs:** High-degree hub nodes remain pinned in the GPU VRAM hot bank (`80%+` hit rate), while cold neighborhood subgraphs are aggregated concurrently by CPU AVX2 threads and GPU `uint4` zero-copy reads across the unified bus.


* **3D Gaussian Splatting & Out-of-Core Ray-Tracing BVH Trees:** Frustum-visible primary splats/BVH nodes reside in the GPU hot bank while secondary/cold LOD splats stream over PCIe via `16-byte` shared-memory staging.


* **Block-Sparse Scientific PDE & Finite-Element Solvers:** Active high-gradient mesh blocks compute in VRAM Tensor Cores while quiescent boundary blocks update in parallel on the CPU across pinned DDR4/DDR5.







---

## 8. Verified Citations, Dates & Hyperlinks Registry (`CITATIONS.md` Reference)

Every source used across `pelias-research-update.md`, `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`, `PELIAS_TRAINER_MASTER_NOTES.md`, `PELIAS_TRAINER_README_main.md`, and `PELIAS_ENGINE_LINUX_BLUEPRINT.md` is cataloged below with its verification date, full hyperlink, and exact scope[cite: 1, 2, 3, 4, 6].

### 8.1 Primary Hardware, OS, and Compiler Documentation (`PUBLISHED`)

1. **NVIDIA Corporation (Checked `2026-10-22` & `2026-10-23`):**
* *CUDA on WSL User Guide (Release 13.0.3 Archive)* — [https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html](https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html) (Current: [https://docs.nvidia.com/cuda/wsl-user-guide/index.html](https://docs.nvidia.com/cuda/wsl-user-guide/index.html))[cite: 1, 4, 6].
* **Verified Scope:** Documents `/dev/dxg` WDDM paravirtualization, absence of full Unified/Managed Memory support on WSL2, pinned system memory limits via host WDDM memory manager, absence of guest PCIe BAR1 visibility, and NVML/profiler restrictions[cite: 1, 2, 4, 6].


2. **Microsoft Corporation (Fetched & Verified `2026-10-23`, published `2024-12-19`):**
* *GPU Virtual Address (Windows Drivers / WDDM `VidMm` Specification)* — [https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address) and *GPU Segments* — [https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-segments](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-segments).


* **Verified Scope:** Verifies under *"Physical page size"* (`DXGK_PTE` multilevel translation): *"System memory, which is always allocated at a 4-KB granularity"* and *"System memory is always managed in 4-KB pages, while memory segments can be managed at either 4 KB or 64-KB granularity as determined by the KMD"* (`G2`). Also see *WDDM GPU Paravirtualization (`dxgkrnl` / `/dev/dxg` FAQ)* — [https://learn.microsoft.com/en-us/windows/ai/directml/gpu-faq](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-faq)[cite: 6].




3. **Advanced Micro Devices, Inc. (Fetched & Verified `2026-10-23`):**
* *Software Optimization Guide for AMD Family 19h Processors (Publication #56665, Rev. 3.00)* — [https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf).


* **Verified Scope:** Verifies Zen 3 (`Ryzen 7 5800X3D`) `64`-entry L1 DTLB, `2,048`-entry L2 DTLB, `512 KB` L2 cache, and `96 MB` shared **victim L3 cache** (§2.1, §2.6); static 50/50 SMT partitioning of the Load Queue, Store Queue, and Retire Queue (`256` macro-ops in `1T` mode vs. `128`/thread in `2T` mode, §2.10.3 & §2.14 Table 3, `G6`); and §2.15 exact quote: *"Following a LOCKed instruction, refrain from using floating point instructions as long as possible"* (`G7`).




4. **NVIDIA Corporation (Verified `2026-10-23`):**
* *NVIDIA Ada GPU Architecture Whitepaper (v2.0)* — [https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) and *CUDA C++ Programming Guide (v13.0)* — [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).


* **Verified Scope:** Verifies GeForce RTX 4070 (`AD104`, `sm_89`) hardware specifications (`46` SMs, `5,888` CUDA cores, `184` 4th-gen Tensor Cores, `36 MiB` L2 cache, `192-bit` GDDR6X bus at `504 GB/s`, `100 KB` L1/Shared memory per SM, `G14`), mapped pinned memory coalescing rules (`cudaHostAllocMapped`), L2 persistence (`cudaAccessPropertyPersisting`, `G10`), CUDA Graphs (`cudaStreamBeginCapture`), and `__nanosleep(ns)`.




5. **The Linux Kernel Documentation & WSL Release Ledger (Verified `2026-10-23`):**
* *Transparent Hugepage Support (`transhuge`)* — [https://docs.kernel.org/admin-guide/mm/transhuge.html](https://docs.kernel.org/admin-guide/mm/transhuge.html) and *PM QoS Interface (`cpu_dma_latency`)* — [https://docs.kernel.org/power/pm_qos_interface.html](https://docs.kernel.org/power/pm_qos_interface.html)[cite: 1, 6].
* *Microsoft WSL Releases (`3.0.4`, `2026-10-20`)* — [https://github.com/microsoft/WSL/releases](https://github.com/microsoft/WSL/releases) (WSL `3.0.4.0` and kernel `6.18.54.1-1` verified on test machine via `raw/e0-2026-10-22/machine_state/`, `G3`)[cite: 1, 6].


6. **Microsoft Win32 API Documentation (Verified `2026-10-23`):**
* *Large-Page Support (`MEM_LARGE_PAGES`)* — [https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support), *VirtualAlloc* — [https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc), *SetThreadAffinityMask* — [https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask), *SetThreadSelectedCpuSets* — [https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadselectedcpusets](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadselectedcpusets), and *SetThreadInformation (`THREAD_POWER_THROTTLING_STATE`)* — [https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadinformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadinformation) (`G5`, `G18`).





### 8.2 Peer-Reviewed & Preprint Literature (`PUBLISHED`)

1. **Routing Locality & Cache-Aware MoE (Read & Verified `2026-10-23`):**
* *Cacheable by Design?* (`arXiv:2608.18261`) — [https://arxiv.org/abs/2608.18261](https://arxiv.org/abs/2608.18261) (Read in full `2026-10-23`: §7 Table 5 pre-registered negative result at 137M/340M from scratch; §7.1 Table 6 untrained inference-time router clamp costs `+0.8% to +2.0%` PPL at `~30%` miss reduction and `+4.0% to +5.8%` at `50%`; §7.1 stacking result reduces rerouting PPL cost from `+7.7%` to `+0.9%`)[cite: 1, 2, 4].
* *ReMoE: Router-Only Fine-Tuning for MoE Locality* (`arXiv:2605.27081`) — [https://arxiv.org/abs/2605.27081](https://arxiv.org/abs/2605.27081) (Read `2026-10-23`: router-only fine-tuning of DeepSeek-V2-Lite with frozen experts lifts adjacent-token reuse from `27.3%` to `34.5%` [`+26%` relative] within benchmark noise).


* *Cache-Aware Joint Router Adaptation* (`arXiv:2609.04895`) — [https://arxiv.org/abs/2609.04895](https://arxiv.org/abs/2609.04895) (Abstract read `2026-10-23`; not relied upon).


* *Sticky Routing* (`arXiv:2607.08780`) — [https://arxiv.org/abs/2607.08780](https://arxiv.org/abs/2607.08780) (PDF unreadable `2026-10-23`; marked `UNVERIFIED`, not relied upon).




2. **DeltaNet, Quantization, LoRA & Kernels (Verified `2026-10-23`):**
* Songlin Yang, Jan Kautz, Ali Hatamizadeh — *Gated Delta Networks: Improving Mamba2 with Delta Rule* (`arXiv:2412.06464`, submitted `2024-12-09`, v3 `2025-03-06`, ICLR 2025) — [https://arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464) (Governs the `30` Gated DeltaNet layers and adaptive decay/delta state update $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$).


* Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim — *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (`arXiv:2406.06484`, submitted `2024-06-10`, v6 `2025-01-15`) — [https://arxiv.org/abs/2406.06484](https://arxiv.org/abs/2406.06484) (Governs the chunked `C=64` WY/UT representation used in Engine prefill and Trainer forward/backward).


* Zihan Wang et al. — *Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models (ESFT)* (`arXiv:2407.01906`, submitted `2024-07-02`, EMNLP 2024 `2024-11-12`) — [https://arxiv.org/abs/2407.01906](https://arxiv.org/abs/2407.01906).


* Erik Wijmans et al. — *Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy / CCE)* (`arXiv:2411.09009`, submitted `2024-11-13`) — [https://arxiv.org/abs/2411.09009](https://arxiv.org/abs/2411.09009) (Governs the `256`-position chunked fp32 linear + cross-entropy loss over the `248,320` vocabulary without materializing full logit matrices).


* Damjan Kalajdzievski — *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA)* (`arXiv:2312.03732`, submitted `2023-11-28`) — [https://arxiv.org/abs/2312.03732](https://arxiv.org/abs/2312.03732) (Governs the $\gamma_r = \alpha/\sqrt{r}$ scaling on the `310` LoRA modules).


* Elias Frantar et al. — *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models* (`arXiv:2408.11743`, posted `2024-08-21`) — [https://arxiv.org/abs/2408.11743](https://arxiv.org/abs/2408.11743) (Governs swap-AB dequant-in-register MMA pipelining tested in `P-K3`).


* *llama.cpp Quantization Study* (`arXiv:2601.14277`, verified `2026-10-23`) — [https://arxiv.org/abs/2601.14277](https://arxiv.org/abs/2601.14277) (Llama-3.1-8B perplexity delta reference: `Q4_K_M` `+0.24`, `Q6_K` `+0.03`, `Q8_0` `+0.01`).


* Nicholas J. Higham — *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, §3.1) (Governs floating-point summation error bounds for CPU/GPU cold-path parity tests).





### 8.3 Internal Repository Ledgers, Code Artifacts & Third-Party Software (`MEASURED` / `READ-SRC`)

1. **`pelias-engine` ([https://github.com/<account>/pelias-engine](https://www.google.com/search?q=https://github.com/<account>/pelias-engine), Private, Apache-2.0 proposed)[cite: 1, 3, 4]:**
* `pelias_tier_blueprint.md` (`2026-10-20`), HWiNFO hardware report (`2026-10-20`), `RUNBOOK.md` (`2026-10-20` to `2026-10-22`), and `p1_tests/results/` (`K` and `A` results, `2026-10-22`)[cite: 1, 4].
* `PELIAS_ENGINE_LINUX_BLUEPRINT.md` (`2026-10-22`, commit `34f744f`)[cite: 1, 4].
* `benchmarks/RESULTS.md` (§1–9.4, up to `2026-10-23`), `benchmarks/raw/e0-2026-10-22/` (including machine state: WSL `3.0.4.0`, kernel `6.18.54.1-1`), and `benchmarks/raw/windows-native-2026-10-23/` (`47.3 / 46.8 tok/s` short, `53.7 / 54.1 tok/s` at `12k`)[cite: 1, 2, 4].


2. **`pelias-trainer` (Private, Apache-2.0 proposed)[cite: 1, 2, 3, 5]:**
* `pelias-trainer 0.2.1` (`peliasv0.2.tar`, `49` total repo files = `725,010 B` / `~708 KiB`; package written `2026-10-19 01:52–03:11`; **Run 1** started `2026-10-19 08:52`: `616` steps, `202.7 min`, median **`247.85 tok/s`**; **Run 2**: `114` steps, `37.9 min`, median **`56.45 tok/s`**; exported `Q5_K_M` `25,347,532,800 B` / `23.61 GiB` on `2026-10-20 00:49` and `Q4_K_M` `21,713,463,296 B` / `20.22 GiB` on `2026-10-20 01:08`)[cite: 1, 3, 5].
* `README.md` (`2026-10-20`, focus note `2026-10-21`, C++/CUDA note `2026-10-23`) and `README_PLAIN_ENGLISH.md` (`2026-10-23`)[cite: 1, 3, 5].
* `docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` (`2026-10-23`, scaffold v1.0) and `docs/PELIAS_TRAINER_MASTER_NOTES.md` (`2026-10-23`).




3. **Upstream Oracle, Base Model & Naming Provenance (Checked `2026-10-20` to `2026-10-23`):**
* [`llama.cpp`](https://github.com/ggerganov/llama.cpp) (MIT License, pinned commit `2ca15f540` / `b11399`)[cite: 1, 3, 4].
* Base model **Qwen3.6-35B-A3B** (Apache-2.0 License, checked `2026-10-20`; local snapshot read from `unsloth/Qwen3.6-35B-A3B` cache; no weights, adapters, or training datasets are stored in either repository)[cite: 3, 5].
* Unrelated open-source geocoder name check: [`pelias/pelias`](https://github.com/pelias/pelias) (MIT License, Geocode Earth, verified `2026-10-20` in `PELIAS_TRAINER_README_main.md` §8)[cite: 3].