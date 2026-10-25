# Research update and master verification audit (core discovery, retirement of the 57.0 figure, E0 split, claims G1 to G18)

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Parts 1.1 and 1.1a (master lines 141 to 298). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
>
> **Status.** The engine is HELD (author, 2026-10-24 06:07). This is documentation of a design; no engine source code exists in this repository. Labels (MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED) are used as defined in the blueprint; a DECISION marked "proposed" is not ruled.
>
> **Part numbers in the text below** refer to the master document. Where each one lives in this repository:
>
> | Reference in the text | Where it is in this repository |
> |---|---|
> | Part 1.6 (reconciliation of the research update) | `docs/spec/RECONCILIATION.md` |
> | Part 1.1, 1.1a (research update: core discovery, retirement of 57.0, E0 split, audit G1 to G18) | `docs/spec/RESEARCH_UPDATE_AUDIT.md` |
> | Part 1.2 to 1.5 (machine, engine facts, drifted assumptions, claim-by-claim review) | `docs/spec/MACHINE_AND_DRIFT.md` |
> | Part 3.1 (engine results ledger) | `benchmarks/RESULTS.md` and `benchmarks/raw/` |
> | Part 3.3, 3.4, 3.5 (Wave 0 results, modelled ceiling, speed-to-ceiling table, attribution register) | `docs/spec/WAVE0_AND_CEILING.md` |
> | Part 4.2 to 4.5, 4.8, 4.9 (engine components, script inventory, patches, microbenchmarks) | `docs/spec/PATCHES_AND_TOOLS.md` |
> | Part 5.3, 7.13 to 7.15, 8.10 (queued items, what is needed, open author decisions, measurement registers, the hold) | `docs/spec/ENGINE_QUEUE_AND_MEASURABLES.md` |
> | Part 8 (history) | summarized in `docs/PELIAS_ENGINE_LINEAGE.md`; the full records stay in the author's local archive |
> | Part 4.10 (replacement map) | `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, section 5 |
> | Part 5.0 (synchronization register) | `docs/spec/SYNC_REGISTER.md` |
> | Part 5.1, 7.2 to 7.6 (synchronized engine blueprint) | `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` |
> | Part 5.2 (trainer co-design contract) | `docs/spec/TRAINER_CONTRACT.md` |
> | Part 6 (proprietary architecture and server specification) | `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` |
> | Part 7.1, 7.16 (order of operations, open decisions O1 to O14) | `docs/spec/OPEN_DECISIONS_AND_ORDER.md` |
> | Part 7.9, 7.10, 7.11 (probe registry, decision register) | `docs/spec/PROBE_REGISTRY.md` |
> | Other parts (1.7, 2, 3.2, 4.1, 4.6, 4.7, 5.4, 7.7, 7.8, 7.12, 8.3 to 8.9, 8.11 to 8.13) | not copied; they stay in the local master document (they concern the trainer, quote private paths and names, or repeat files that are already in this repository, such as `support/archive/c_ai/creator/moe_track/UNFINISHED.md` for 4.6) |
> 

---

## 1.1 Header, core discovery, retirement of the mistyped Windows speed, the E0 split breakdown and the complete audit of the outside report

<!-- SPLICE id="P1.1" src="pelias-research-update.md (received 2026-10-23 19:59, 870 lines)" lines="1-182" shift="+3" ws="1" edited="0" -->
> **Source (verbatim):** pelias-research-update.md (received 2026-10-23 19:59, 870 lines), lines 1-182. Contains lines 1-182: header table, section 1 (core discovery), section 2 (2.1 retirement of the "57.0" Windows figure, 2.2 E0 split, 2.3 audit G1-G18). Heading levels shifted by +3 for this document only. Table rows that the source breaks over several lines are joined back (whitespace only).

### PELIAS RESEARCH UPDATE & MASTER VERIFICATION AUDIT (`pelias-research-update.md`)

**Unified Cross-Check of `pelias-engine` and `pelias-trainer`, Complete Derivation & Probe Ledger, Ceiling Gap Attribution, Blueprint Synchronization, and the Master C++20/CUDA Universal Unified-Bus Architecture**.

| Field | Value | Source & Verification Date |
| --- | --- | --- |
| **Document** | `pelias-research-update.md` (supersedes draft alias `pelias-research.update.md`) | Master Verification Audit (`2026-10-23`) |
| **Date** | `2026-10-23`<br> | Repository Ledger & Master Notes Lock (`2026-10-23`) |
| **Author & Maintainer** | the user (`Copyright (c) 2026 John Lipka and the Pelias contributors`) | `PELIAS_TRAINER_MASTER_NOTES.md` & `README.md` (`2026-10-20` to `2026-10-23`)[cite: 2, 3] |
| **Repositories** | [`pelias-engine`](https://www.google.com/search?q=https://github.com/<account>/pelias-engine) (Inference) & `pelias-trainer` (LoRA/QAT Fine-Tuning)[cite: 1, 3, 4] | Private repositories, interfaced via standard GGUF (`2026-10-23`)[cite: 2, 3] |
| **Pinned Oracle** | [`llama.cpp`](https://github.com/ggerganov/llama.cpp) commit `2ca15f540` (`b11399`) with 4 carried patches (`mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu`) and 2 dropped diagnostic patches (`mmq_dump_diag`, `mmq_dump_diag2`) (`READ-SRC` / `DECISION`)[cite: 1, 2, 4] | Author Ruling (`2026-10-22`) & `PELIAS_ENGINE_LINUX_BLUEPRINT.md`[cite: 1, 4] |
| **Target Hardware** | NVIDIA GeForce RTX 4070 12 GB (`12,281 MB` GDDR6X, [Ada Lovelace `AD104](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)`, `sm_89`, 46 SMs, 5,888 CUDA cores, 184 Tensor Cores, 36 MiB L2, 192-bit bus, `504 GB/s`; OC: `965 mV` @ `~2,775–2,790 MHz` core, `+1000 MHz` mem) + AMD Ryzen 7 5800X3D ([Family `19h` Zen 3](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf), 8C/16T, AVX2 only, no AVX-512, 96 MB 3D V-Cache victim L3) + 64 GB (`4×16 GB`) DDR4-3200 (`51.2 GB/s` theoretical peak) + PCIe 4.0 x16 (`26.1–26.2 GB/s` measured) + CUDA `13.0.88`[cite: 1, 4, 6] | `MEASURED` via HWiNFO (`2026-10-20`), `nvcc`, microbenches `02`–`04` (`2026-10-20`), & `PUBLISHED` vendor specs (`2026-10-23`)[cite: 1, 4, 6] |
| **Hard VRAM Ceiling** | **`10.9 GiB`** (Author ruling, `2026-10-22`; enforced by runtime memory governor against `10.4 / 10.9 / 11.2 GiB` candidates)[cite: 1, 2, 4] | `DECISION` (`2026-10-22`, `PELIAS_ENGINE_LINUX_BLUEPRINT.md`)[cite: 1, 2, 4] |
| **Honesty Labels** | **MEASURED** (run on the author's machine, in a ledger or raw file) · **READ-SRC** (read from source or from the served model file's header) · **PUBLISHED** (external vendor/academic specification, cited with URL and check date in Section 8) · **DERIVED** (scripted arithmetic from labelled inputs) · **ESTIMATE** (engineering judgement, paired with a named probe) · **UNVERIFIED** (not relied on until its named probe passes) · **DECISION** (a design choice or author ruling with a recorded reason) · **AUTHOR-STATED** (stated by the author; no log attached) | Enforced across all 231 labelled statements in the Trainer Master Scaffold (`MEASURED 24 · READ-SRC 42 · PUBLISHED 33 · DERIVED 72 · ESTIMATE 23 · UNVERIFIED 19 · DECISION 18`) (`2026-10-23`) |

---

#### 1. The Core Discovery: Turning Consumer Hardware into a Universal Unified-Bus System

The foundational breakthrough driving both `pelias-engine` and `pelias-trainer` emerged from measuring actual hardware floors while solving a practical constraint: fine-tuning and serving **Qwen3.6-35B-A3B** (`20.22 GiB` in `Q4_K_M`, `753` tensors) inside a strict **`10.9 GiB` VRAM ceiling** on a single 12 GB consumer RTX 4070 paired with 64 GB of DDR4-3200 system RAM (`READ-SRC` / `DECISION`, `2026-10-20` to `2026-10-23`).

##### 1.1 The Physical Dual-Bus Discovery & Why Unknowns Had to Be Derived

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

#### 2. Zero-Hallucination Self-Audit & Reconciliation of Prior Claims

To maintain absolute data integrity across both repositories, this section audits the external research plan (`pelias_engine_hardware_verified_research_plan.md`, dated `2026-10-23`)[cite: 6] and internal working notes (`PELIAS_TRAINER_MASTER_NOTES.md`, dated `2026-10-23`)[cite: 2] against the raw benchmark ledgers (`benchmarks/RESULTS.md` §1–9.4)[cite: 1, 2, 4].

##### 2.1 Formal Retirement of the `"57.0 tok/s"` Windows-Native Figure (`G1` / `X10`)

* **The Discrepancy:** In `pelias_engine_hardware_verified_research_plan.md` (§1, §2.1, §2.2, §3, dated `2026-10-23`), an outside draft claimed a live Windows 11 decode speed of `"57.0 tok/s (17.54 ms/token)"` and derived a `"6.67 ms/token (166.7 µs/layer)"` gap to the ceiling.


* **The Ledger Truth (`MEASURED`):** In `pelias-engine/benchmarks/RESULTS.md` (§9.4, `benchmarks/raw/windows-native-2026-10-23/`), the only recorded Windows 11 native runs are **`47.3` and `46.8 tok/s` (`21.14 / 21.37 ms/token`, client token-gap p50 `21.53 ms`) at short context (`ctx0`)** and **`53.7` and `54.1 tok/s` (`18.62 / 18.48 ms/token`, client token-gap p50 `18.28 ms`) at `12k` context** (`MEASURED`, `2026-10-23`).


* **Origin of the Mistyped `57.0`:** The number `57.0` appears in the repository logs only once: as the upper bound of a 3-request range at `4k` context depth for an **`8.5 GiB` WSL2 bank** whose median was `53.7 tok/s` (`55.7 tok/s` short) and whose peak VRAM (**`11.4 GiB`**) violated the **`10.9 GiB`** ceiling (`MEASURED`, `pelias-research-update.md` §1.1 & `PELIAS_TRAINER_MASTER_NOTES.md` §1.15, `2026-10-23`).


* **Resolution (`DECISION`):** The `"57.0 tok/s"` figure is permanently retired across all engine and trainer documents. All speed-to-ceiling calculations use the six verified benchmark entries in `benchmarks/RESULTS.md` (§8.2, §9.2, §9.4, dated `2026-10-22` & `2026-10-23`).



##### 2.2 Strict Separation of Measured E0 Sub-Timings vs. Unmeasured Intra-Layer Splits

* **What is `MEASURED` in E0 (`benchmarks/raw/e0-2026-10-22/` & `RUNBOOK.md`, `2026-10-22`):** On WSL2 Reference Cell A (`7.0 GiB` bank, bypass 32, `-ub 4096`), a timer-free short-context token takes **`24.6 ms`** inside the server loop (streaming client token-gap p50 is `26.33 ms` at `ctx0` and `23.34 ms` at `12k`; with timer+CUDA graphs ON `31.99 / 28.18 ms`; with CUDA graphs OFF `36.50 / 32.56 ms`) (`MEASURED`, `2026-10-22`). Host-side split sums across the **`82 splits/token` (`41 CPU + 41 CUDA`)** decompose the `24.6 ms` token into:


* **CPU cold expert compute:** **`10.1 ms`** (vs. `2.86 ms` assumed in the `2026-10-20` ceiling model $\rightarrow$ **`+7.2 ms` gap**) (`MEASURED` / `DERIVED`, `2026-10-22`).


* **GPU compute:** **`6.6 ms`** (`2.75 ms` host CUDA split time with CUDA graphs ON vs. `8.04 ms` with CUDA graphs OFF; vs. `7.73 ms` assumed in the ceiling model $\rightarrow$ **`-1.1 ms` faster than model**) (`MEASURED` / `DERIVED`, `2026-10-22`).


* **Sync + activation copy handoff:** **`5.1 ms`** (`4.6–5.0+ ms` in `PELIAS_ENGINE_LINUX_BLUEPRINT.md` Sources note; vs. `0.25 ms` assumed in the ceiling model $\rightarrow$ **`+4.85 to +4.9 ms` gap**) (`MEASURED` / `DERIVED`, `2026-10-22`)[cite: 1, 4].
* **Copy-back / kernel launch overhead:** **`2.8 ms`** (vs. `0.00 ms` assumed in the ceiling model $\rightarrow$ **`+2.8 ms` gap**) (`MEASURED` / `DERIVED`, `2026-10-22`).


* **CPU and GPU execution overlap:** **Zero (`0%`)** (`MEASURED` — the CPU and CUDA graph splits execute strictly serially in the `llama.cpp` tier build, `2026-10-22`)[cite: 1, 4].


* **What is `DERIVED` (FLOP Shares) vs. `UNVERIFIED` (Sub-Layer Microsecond Splits):** Within any MoE layer, the CPU cold expert branch can only overlap with GPU hot-expert + shared-expert execution because router top-8 selection causally depends on the attention/DeltaNet output (`READ-SRC` / `DERIVED`, `2026-10-23`)[cite: 1, 4]. While the forward FLOP breakdown at $T = 8,192$ is exact (`DERIVED` in `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` §6.10, `2026-10-23`: out of **`6.74 GFLOP/token`**, DeltaNet projections are **`2.02 GFLOP` [`30%`]**, routed experts **`2.01 GFLOP` [`30%`]**, `lm_head` **`1.02 GFLOP` [`15%`]**, attention scores/values **`0.67 GFLOP` [`10%`]**, attention projections **`0.55 GFLOP` [`8%`]**, shared experts **`0.25 GFLOP` [`4%`]**, DeltaNet chunk core **`0.18 GFLOP` [`3%`]**, and routers **`0.04 GFLOP` [`1%`]**), the exact microsecond division of the `6.6 ms` batch-1 GPU time between spine attention/DeltaNet and hot-bank experts is **`UNVERIFIED` until Probe `P-E2` (`e0_timeline.patch` per-layer CUDA event timers) runs** (`2026-10-23`)[cite: 1, 2, 4].

##### 2.3 Complete Audit of the 18 Claims (`G1–G18`) from the Outside Research Report

Every claim from `pelias_engine_hardware_verified_research_plan.md` (`2026-10-23`)[cite: 6] was cross-checked against the repository ledgers and primary vendor documentation (`PELIAS_TRAINER_MASTER_NOTES.md` §1.16 & `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` §16.4, `2026-10-23`):

| ID | Claim in Outside Report (`2026-10-23`)[cite: 6] | Verification Against Repo Data & Primary Sources (`2026-10-23`) | Label / Verdict | Disposition in Engine & Trainer Plans |
| --- | --- | --- | --- | --- |
| **G1** | Windows-native runs at `57.0 tok/s (17.54 ms/token)`, gap `6.67 ms` (`166.7 µs/layer`)[cite: 6]. | Repo ledger (`benchmarks/RESULTS.md` §9.4, `2026-10-23`) records `47.3 / 46.8 tok/s` short and `53.7 / 54.1 tok/s` at `12k`. | **Rejected (Not in repo)**<br> | Recomputed in Section 4.2 from the six logged runs. |
| **G2** | WDDM `VidMm` manages system memory strictly in `4-KB` pages (`DXGK_PTE`)[cite: 6]. | Primary Microsoft Learn page (*[GPU Virtual Address](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-address)*) verified `2026-10-23`: *"System memory, which is always allocated at a 4-KB granularity"* / *"System memory is always managed in 4-KB pages"*. | **PUBLISHED (True)**<br> | Context for `P-E11`; does **not** explain the `9.59 vs. 26.2 GB/s` gap because the `26.2 GB/s` checksum kernel reads the exact same `4-KB` pinned pages (`DERIVED`, `2026-10-23`). |
| **G3** | [WSL `3.0.4` (`ce88638`)](https://github.com/microsoft/WSL/releases), kernel `6.18.54.1`, removed bundled `perf` (`#41904`), `64 KB` stack (`#41888`)[cite: 6]. | `wsl --version` in `raw/e0-2026-10-22/machine_state/` confirms WSL `3.0.4.0`, kernel `6.18.54.1-1` (`MEASURED`, `2026-10-22`); individual PR numbers unverified on fetched release page. | **Version MEASURED; PRs UNVERIFIED**<br> | No change needed: Engine E0/E2 already uses in-process `cudaEventRecord` timers (`DECISION`)[cite: 1, 4]. |
| **G4** | `4-KB` guest pages in `mmap` cause the `2×` prefill swing (`863` vs. `1,680 tok/s`); [`madvise(MADV_HUGEPAGE)`](https://docs.kernel.org/admin-guide/mm/transhuge.html) fixes it[cite: 6]. | `--load-mode none` is the **fast** mode (`1,667–1,680 tok/s`) and uses ordinary anonymous `4-KB` memory; `.wslconfig` already sets `transparent_hugepage=always`; CPU `2 MiB` page test (`02_ram_bw_test.cpp`) got `44.86 GB/s` vs. `45.81 GB/s` for `4 KiB` (`MEASURED`, Ledger §1, `2026-10-20`)[cite: 1, 2, 4]. | **Contradicted by Repo Data**<br> | Kept as Engine open unknown: inspect `/proc/meminfo` `AnonHugePages` per load mode (`E11` / `P-E11`)[cite: 1, 2, 4]. |
| **G5** | Windows: allocate cold bank with [`VirtualAlloc(MEM_LARGE_PAGES)`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc) + [`SeLockMemoryPrivilege`](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support), staging buffer with `cudaHostAlloc`[cite: 6]. | Win32 APIs exist as documented (`PUBLISHED`, verified `2026-10-23`); benefit on 5800X3D DTLB unmeasured on Windows[cite: 1, 6]. | **UNVERIFIED Hypothesis**<br> | Added as Windows-native A/B test if Windows-native is promoted (`Decision 4`). |
| **G6** | Pin 8 worker threads to the 8 physical Zen 3 cores (`1T` mode); Load/Store/Retire queues are split 50/50 in `2T` mode ([AMD Family 19h SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) §2.14, Table 3)[cite: 6]. | AMD Family 19h SOG (pub. `56665` rev. `3.00`) verified `2026-10-23`: §2.10.3 & §2.14 Table 3 confirm queue partitioning in SMT mode; stock `llama.cpp` ledger §8.8 shows 5–7 threads beat 10 (`MEASURED`, `2026-10-20`)[cite: 1, 6]. | **Partly PUBLISHED + MEASURED**<br> | Adopted as A/B test `E-A1` in Engine Phase E4 (`8` pinned physical cores vs. `16` SMT). |
| **G7** | Use `acquire`/`release` (`MOV`) instead of `seq_cst` (`LOCK XCHG`) on handoff flags; avoid FP instructions right after `LOCK` ([AMD Family 19h SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) §2.15)[cite: 6]. | AMD SOG §2.15 quote verified word-for-word on `2026-10-23`: *"Following a LOCKed instruction, refrain from using floating point instructions as long as possible"* (`PUBLISHED`)[cite: 1, 6]. | **PUBLISHED (True); benefit UNVERIFIED**<br> | Adopted as coding rule and A/B test `E-A2` in Engine Phase E6. |
| **G8** | Use `_mm_prefetch(..., _MM_HINT_NTA)` on cold expert streaming to protect the `96 MB` victim L3 cache[cite: 6]. | Zen 3 L3 is a victim cache (`PUBLISHED`, [AMD SOG](https://kib.kiev.ua/x86docs/AMD/Optimization/56665_3.00_Software%20Optimization%20Guide%20for%20AMD%20Family%2019h%20Processors%20%28PUB%29.pdf) §2.1/2.6, verified `2026-10-23`); speed effect unmeasured. | **UNVERIFIED Hypothesis**<br> | Adopted as A/B test `E-A3` in Engine Phase E4. |
| **G9** | `16-byte` (`uint4`) loads staged through `__shared__` memory lift GPU zero-copy compute reads from `9.59` to `26.2 GB/s`[cite: 6]. | This is the engine's own `wide_load.cu` (`E0`/`E5`) hypothesis; the outside report misattributed the author's measured numbers to the Ada whitepaper (`2026-10-23`)[cite: 1, 2, 4, 6]. | **UNVERIFIED Hypothesis**<br> | Tested by Probe `P-E10` (`wide_load.cu`, including `Q6_K` `210 B` block staging). |
| **G10** | Pin `16–24 MB` of the RTX 4070's `36 MB` L2 cache via [`cudaAccessPropertyPersisting`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)[cite: 6]. | CUDA API exists (`PUBLISHED`, verified `2026-10-23`); max persisting size under WDDM/WSL2 unmeasured. | **UNVERIFIED Hypothesis**<br> | A/B test `E-A4` in Engine E5 (query `cudaDevAttrMaxPersistingL2CacheSize` first). |
| **G11** | Hold [`/dev/cpu_dma_latency = 0`](https://docs.kernel.org/power/pm_qos_interface.html) on Linux; run `nvidia-smi -lgc 2475,2475`[cite: 6]. | First is standard Linux PM QoS (`PUBLISHED`, verified `2026-10-23`); **second (`2475 MHz`) would underclock the author's `~2,775–2,790 MHz` overclock (`X14`)**. | **Clock Lock REJECTED**<br> | `/dev/cpu_dma_latency` kept for Native Linux E1 only; `2475 MHz` lock rejected (`DECISION`). |
| **G12** | Native Linux projected at `72–86+ tok/s`[cite: 6]. | Zero native-Linux runs exist yet; Phase E1 measures them[cite: 1, 2, 4]. | **UNVERIFIED (Not used)**<br> | Replaced by actual Phase E1 measurements when run[cite: 1, 2, 4]. |
| **G13** | Gate thresholds (`≥22 / ≥25 GB/s`, `p99 ≤ 12 µs`) and Oct 23–Nov 22 calendar[cite: 6]. | Invented by the outside report; `PELIAS_ENGINE_LINUX_BLUEPRINT.md` (`2026-10-22`) has its own phase gates[cite: 1, 2, 4]. | **Not Adopted**<br> | Author sets all dates and post-E0/E1 gates (`DECISION`)[cite: 1, 2, 4]. |
| **G14** | RTX 4070 specs: 46 SMs, 5,888 cores, 192-bit bus, `504 GB/s`, 36 MB L2, `sm_89`[cite: 6]. | Matches [NVIDIA Ada GPU Architecture Whitepaper v2.0](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) and Trainer Scaffold §2.1 (`2026-10-23`). | **PUBLISHED (Consistent)**<br> | Locked in hardware fact base[cite: 1, 2, 4]. |
| **G15** | Pinned host memory capped at `≤ 32 GB` under WDDM[cite: 6]. | `cudaHostAlloc` proven to `20,448 MiB` on WSL2 despite `RLIMIT_MEMLOCK = 64 MiB` (`MEASURED`, `2026-10-20`); `>20.4 GiB` untested. | **UNVERIFIED above 20.4 GiB**<br> | Tested up to `32 GiB` by Probe `P-H7` and Engine `E11`. |
| **G16** | WSL2 runs at `44–54 tok/s` short and `43–49 tok/s` at `12k`[cite: 6]. | True for the best recorded cell (`7.5 GiB` bank, bypass 32, `-ub 2048`: `53.5 / 53.0 / 49.0 tok/s`, Ledger §8.2), not for Reference Cell A (`37.9 / 42.1 tok/s`, Ledger §9.2) (`MEASURED`, `2026-10-22`)[cite: 1, 2, 4]. | **MEASURED (Config-specific)**<br> | Both cells explicitly separated in Section 4.2. |
| **G17** | [`/dev/dxg` VMBus](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-faq) adds `25–35 µs/layer` (`1.0–1.4 ms/token`), explaining the Win11 vs. WSL2 gap[cite: 6]. | `1.0–1.4 ms` is at most **`19–27%`** of the measured **`5.02–5.27 ms/token`** (`5.1–5.2 ms`) Win11–WSL2 difference on Reference Cell A (`DERIVED`, `2026-10-23`). | **UNVERIFIED & Partial**<br> | Addressed via E0 handoff probe (`94 µs` sync vs. `13.3 µs` doorbell) and CUDA graphs[cite: 1, 2, 4]. |
| **G18** | Windows-native thread pinning + [`SetThreadInformation`](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadinformation) (`HighQoS`) + `timeBeginPeriod(1)` cuts `5–21%` spread[cite: 6]. | Win32 APIs exist (`PUBLISHED`, verified `2026-10-23`); launch-to-launch spread is `5–21%` (`MEASURED` on WSL2, Blueprint `2026-10-22`); Win11 pinning untested[cite: 1, 2, 4, 6]. | **UNVERIFIED Hypothesis**<br> | Tested only if Windows-native is promoted (`Decision 4`). |

---

<!-- /SPLICE P1.1 sha256="a4cbf29aa7a0ae0aed4176067a339dd7359686bb937d204bee32ea7a99578ec0" -->

## 1.1a Assembly note: what in block 1.1 (and in the research-update blocks of Parts 2, 3, 5 and 7) is superseded, and by what

The research update was received on 2026-10-23 19:59 and compared line by line with the trainer master that evening (1.6). The author's rulings of 2026-10-23 and 2026-10-24 then moved some of its figures. The update's text is kept word for word wherever it appears in this document, so each place where it is out of date is listed here, with the block that governs instead. Nothing below is a new measurement; every replacing figure is in a spliced block.

| # | Research-update statement (where) | Governs instead | Source of the replacement |
|---|---|---|---|
| 1 | Hard VRAM ceiling 10.9 GiB (header table; F6; 5.1 edit 2; 5.2 matrix; 5.5 decision 2; section 7 summary) | **10.7 GiB**, author ruling 2026-10-23 (D26), after P-H8 measured 10.83 GiB (WSL2) and 10.85 GiB (Windows-native) free after the context exists. Ruled 2026-10-24 (D28) that the number is measured at start (smaller of CUDA free and NVML free after the context, minus a margin); 10.7 GiB stays this machine's value until the margin is measured | Part 5.1 table "VRAM budget and headroom"; Part 7.5 and 7.11 (D26, D28) |
| 2 | Headroom for extra core bytes 0.36 GiB (369 MiB) at the default micro-batch and about 0.11 GiB (113 MiB) at -ub 2048 (F6; 5.1 edit 2; 5.5 decision 2) | Whole-card reading at 10.7 GiB with a 0.1 GiB margin: 0.16 GiB (0.20 at the lowest peak) at the default micro-batch and none at -ub 2048 (-0.09 GiB). Process-only reading: about 1.2 / 0.9 GiB. The 0.36 / 0.11 pair was computed against 10.9 GiB | Part 5.1 table; Part 7.12 |
| 3 | The 10.40 to 10.44 GiB peaks are "the default prefill micro-batch (-ub 4096)" (F6; 5.1 edit 2; 5.5 decision 2) | The micro-batch of those two runs (stamps 130854 and 132648) is **not recorded** in the files (engine ledger 8.2: "inferred from the rate"). The 7.5 GiB bank at -ub 4096 is unmeasured; by the 1:1 bank-to-peak slope seen at -ub 2048 it would be about 11.2 GiB [ESTIMATE], over the ceiling | Reconciliation D2 (1.6); blueprint unknown "P-E4 at -ub 4096" (Part 7.6) |
| 4 | Trainer VRAM layout "7.90 GiB (2 x 498 MiB slots + 0.80 spine + 0.32 LoRA/Adam + 2.50 working set)" (F3 area) | The four named items sum to 4.59 GiB. The master's 13-row table sums to 7.90 GiB; the update's line is an incomplete summary, not a different design | Reconciliation D8 |
| 5 | "231 labelled statements: 24 / 42 / 33 / 72 / 23 / 19 / 18" (header table) | Not adopted. Counting tokens, the master has 24 / 42 / 33 / 74 / 21 / 19 / 17 = 230; tokens are not statements. A per-row census is open item N5 | Reconciliation D3, N5 |
| 6 | E0 split "10.1 / 6.6 / 5.1 / 2.8 ms = 24.6 ms" labelled MEASURED, and "zero overlap" labelled MEASURED (2.2; 4.3 finding 1) | **DERIVED.** The parts are cell-B (timer on) medians with the timer cost taken out of sync-and-copy, so 5.0 is a lower bound (4.6 at 12k); 2.8 is copy-back 2.3 plus launch 0.5; zero overlap is derived from the parts summing to wall time (30.5 against 30.3 ms) | Reconciliation D4; engine ledger 9.2 (Part 3.1) |
| 7 | "2.75 ms host CUDA split (graphs on) against 8.04 ms (graphs off)" (2.2; 4.1 table) | Not found in the ledger or in the engine's summaries. Not adopted until found | Reconciliation D4 |
| 8 | The 82 splits per token "include the 41st block blk.40 (MTP)", READ-SRC / MEASURED (4.3 finding 5) | The 82 is measured. What the 41st CPU split is, is **UNVERIFIED**: the server log says "hot bank active: 40 layers" and the last attach line is layer 39. P-E2's per-layer timers (by layer index) settle it | Reconciliation D5 |
| 9 | The Windows-native gain of 5.1 to 5.2 ms "matches" the WSL2 sync-and-copy cost of 4.6 to 5.1 ms (4.3 finding 2) | A hypothesis. The ledger itself says "consistent, not proven" (9.4), and the update's own G17 caps the VMBus part at 19 to 27 % of the gap. P-H11 tests it | Reconciliation D6; engine ledger 9.4 |
| 10 | Decision 1 (the 7.5 GiB bank) "treated as DERIVED" (5.5) | An open author decision. The bank need not be a multiple of 0.5 GiB: at the default micro-batch the free room would hold roughly 7.8 GiB if the slope is 1:1 [ESTIMATE] | Reconciliation D7; Part 7.5 |
| 11 | LoRA 310 = 30 x 3 + 10 x 4 + 40 x 3 + 30 x 2 "gates/norms" (F3) | 10 x 4 attention + 30 x 5 DeltaNet + 40 x 3 shared expert = 310 (READ-SRC from the v0.2 source). Norms take no LoRA. Same total, different parts | Reconciliation D1 |
| 12 | P-E13 "DONE (READ-SRC)" (5.4) | Partly done: the llama.cpp side was read; the engine side is open because the engine kernels are not written | Reconciliation D9 |
| 13 | "-ub 2048, the setting that achieves 1,667 to 1,680 prompt tok/s at 12k" (5.5) | 1,680.3 is stamp 132907 (--load-mode none, -ub 2048); the best-cell launch 141137 (default load, -ub 2048) read 920.75 at 12k. The load mode, not -ub, is the 2x factor | Reconciliation D10 |
| 14 | 8.5 GiB bank "median 53.7 (55.7 short), peak 11.4 GiB" (5.1 edit 2) | Two different launches: stamp 131550 (55.7 short, peak 11.4) and the sweep run b85 (8,631 MiB bank, 4k: 53.7, range 51.4 to 57.0, peak 11.28) | Reconciliation D11 |
| 15 | Section 7 Layer 1 lists `tok/libllama_bridge` ("exact GGUF tokenizer and Jinja chat-template") and `verify/oracle_2ca15f540` inside the shared core; Layer 0 lists `quant/ggml_exact` ("bit-exact" kernels) | Replaced: no llama.cpp or ggml code is linked, copied or required at run time (D37; author ruling 2026-10-24 08:15). The tokenizer and template renderer are native (Part 6.13). The quantizers stay bit-exact in their bytes and arithmetic but are original code that reproduces the layout and arithmetic (Part 6.11), and the module is named `quant/`. The oracle is an external, read-only harness in `tests/oracle/`, never part of the core | Part 5.1 (engine blueprint, R2); Part 6 |
| 16 | "Windows-native" speed figure "57.0 tok/s" in outside claim G1 | Retired. Candidate origins (A3): the upper bound of a 3-request range at 4k on WSL2 (b85 sweep, median 53.7, range 51.4 to 57.0), and a prompt-read rate of 57.0 in a 2026-10-21 speed file. The recorded Windows-native runs are 47.3 and 46.8 tok/s short and 53.7 and 54.1 at 12k | 1.1 (update 2.1); reconciliation A3 |

**What stays as the update has it.** The arithmetic holds: the reconciliation recomputed 86 figures from their stated inputs and one check failed on purpose (row 4). The 18 verdicts of the audit table match the master's 16.4. The ten mandatory blueprint edits, the two-way contract, the dependency chain and the six open decisions are the master's 16.7 and the notes' 2.1 to 3.1 restated; where the update adds the unified program (its section 7), that is new and is kept as the update wrote it in Part 5.4 and Part 6.1, corrected by row 15.
