# Wave 0 hardware results, the modelled speed ceiling, the speed-to-ceiling table and the attribution register

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Parts 3.3, 3.4 and 3.5 (master lines 1459 to 1612). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 3.3 Wave 0 hardware results on WSL2 and Windows-native (trainer master 11.12)

<!-- SPLICE id="P3.3" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="1671-1716" shift="+1" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 1671-1716. Heading levels shifted by +1 for this document only.

### 11.12 Wave 0 results so far — the hardware set (2026-10-23/24)

The hardware probes of 11.2 exist as one standalone program (C++/CUDA; one JSON per probe in the schema of 11.11; it never overwrites a result). It was run in full on **WSL2** (2026-10-23, 68 minutes) and on **Windows-native** (2026-10-23/24, 68 minutes; built with nvcc 13.4 and Visual Studio 18 Professional), with the overclock on and nothing else using the GPU. Every step exited 0. P-M4 has not run (it needs the model path). The raw files are kept in the author's backup folder, not in this repository. Everything below is MEASURED unless marked; "A / B" means WSL2 / Windows-native.

| Probe | Result (A / B) | What it changes |
|---|---|---|
| P-H8 free VRAM | Total 11.99 GiB. After the CUDA context exists: CUDA free **10.83 / 10.85** GiB, NVML free 10.95 / 10.85. Context cost 154 / 138 MiB. NVML used before the context 914 / 1,031 MiB (held by Windows). BAR1 total 16,384 MiB; used 0 / 16,355 MiB. The same 10.83 is in 21 engine logs (a fixed driver budget; inference) | The ceiling (10.7, D26); only 0.13 / 0.15 GiB of physical room above it |
| P-H1 device | 46 SMs; opt-in shared memory 101,376 B (99 KiB) per block; the driver reports 5 async copy engines | Tile constants as planned |
| P-H2 bandwidth | VRAM read 523 / 534, write 448 / 454, copy kernel 473 / 481, `cudaMemcpy` device-to-device 495 / 504 GB/s (the last two count read + write). Host DRAM read, 1 thread 37.1 / 37.9, best at 4 threads 43.7 / 44.8 GB/s | Roofline inputs; VRAM read is above the 504 GB/s stock spec (the memory is overclocked) |
| P-H3 tensor rates (60 s each, sustained) | bf16 65.2, fp16 64.9, int8 259.5, fp8-e4m3 129.8 T(FL)OPS, identical on both platforms to 0.1; 99.8 % of the burst rate; SM clock 2,760–2,775 MHz; 0.511 / 0.511 / 2.044 / 1.022 per SM per GHz | Replaces the DERIVED overclock peaks 65.4 / 261 (−0.3 % / −0.7 %). int8 is 3.98× and fp8 1.99× bf16: the 4× assumption holds. Stock-clock values are not measured |
| P-H10 canary | 10,000 GEMM runs, 0 mismatches, 1 distinct hash, CPU-check error ≤ 1 % of tolerance (both platforms) | R3: no bit flip seen at the overclock in this run |
| P-H4 soak (30 min) | 17,685 / 17,700 cycles, 0 mismatches. SM clock median 2,760 (per-minute means 2,760.0–2,760.1); 43–55 / 49–56 °C (median 52 / 53); power median ≈ 107 W of the 200 W limit. Memory clock under load: 11,251–11,501 MHz on A, a flat 11,501 on B. The driver's throttle-reason field reads "gpu_idle" (raw mask 1) on every sample on both platforms while the GPU is flat out | The throttle-reason field is unusable; the stop rule of 15.3/15.8 uses clocks and temperature |
| P-H5 transfers (256 MiB) | Pinned H2D 26.3 / 26.3 and D2H 26.3 / 26.3 GB/s; pageable H2D 10.3 / 11.8, D2H 10.8 / 11.8; zero-copy kernel read 26.3; three streams together 26.1 / 26.2. Small copies (median): 4 KiB 73 / 31 µs, 1 MiB 113 / 72 µs | 5.5 uses 26.1 (about 1 % low). The 20 GB/s pass line is met |
| P-H6 contention | H2D 26.1 alone; with 4 / 8 / 16 CPU reader threads 26.0 / 25.3 / 23.6 on A (worst ratio 0.90; B worst 0.87); combined CPU + H2D 46.6–48.7 GB/s on A (engine prior 47.6) | Roofline slowdown factors |
| P-H7 pinned limit | **32 GiB** (the largest size tried; 34.4 GB) held on both: 0 MiB swapped in, DMA round trips pass. Allocation 0.76–1.0 s per GiB on A, 0.069 s per GiB on B. Host available memory falls 1.01 GiB per pinned GiB (A 45.9 → 13.7; B 55.6 → 23.4). The real limit is not found | R7 closed. The 28–33 GiB plan of 5.3 fits |
| P-M3 host hold | 24 GiB pinned + 4 GiB pageable for 30 min: 0 mismatches, 0 MiB swapped in, H2D during the hold median 25.8 / 26.0 GB/s. 10-ms-sleep overshoot: A median 0.082, p99 0.375, max 1.70 ms; B median 6.14, p99 6.48, max 16.9 ms; no stall over 50 ms on either | The B figure is probably the default 15.6 ms Windows timer tick, not a stall (INFERENCE; the Windows build asks for no finer timer), so the two are not comparable yet. No jitter threshold exists until an idle run does |
| P-H9 graphs and flush cost | Eager launch 4.70 / 2.18 µs per kernel, graph replay 0.716 / 0.707 µs (85 % / 68 % saving; A's eager figure was 2.25 µs in the short pass, so it depends on run length); counters exact; parameter update works; 10⁵ replays. `cudaStreamQuery` flush: idle stream 0.098 / 0.091 µs; busy stream median 76.6 / 8.05 µs per call; empty launch + flush, pipelined, 48.6 / 24.2 µs against 5.67 / 2.89 µs for the launch alone | Graphs adopted (pass: ≥ 30 %). The flush is host time, hidden while the GPU has queued work (5.4) |
| P-H12 overlap | Five copy engines. Ordinary submission: the work runs one after another on the GPU timeline (all-three efficiency 0.47 / 0.52 for many short tensor launches, 0.00 for one long launch, 0.39 / 0.42 for the VRAM-read kernel). With a flush after each enqueue, compute first, interleaved copies: **0.92 on both**; H2D + D2H pair 0.84 | The flush rule (D27, 5.4); `S_w` and `S_x` overlap at 0.84, not 1.0 |
| P-H13 display timeout | Stages of 2, 5 and 30 s (one kernel per SM, about 60 W), a 10 s queued set and a 10 s graph: all completed, device usable afterwards, no errors. B: no display-driver events; the author saw nothing on screen on either path. The device attribute "kernel exec timeout" reads 1 and the Windows timeout is at its default (2 s) | Not a constraint on this machine for ≤ 30 s of this load. Heavier kernels and other display setups are unmeasured |

**What these results change in the design.**

1. **The ceiling is 10.7 GiB** (D26). The card has 11.99 GiB, but CUDA reports only 10.83 / 10.85 GiB free once a context exists, identically on every run and in 21 engine logs; the first P-H8 rule (free ≥ ceiling + 0.3 = 11.2) could never pass. What the number is measured against is open (16.3 X15; 16.7 decision 7).
2. **The streamer must flush** (D27): on both platforms copy/compute overlap exists only with a `cudaStreamQuery` after each enqueue (5.4, R6).
3. **R7 is closed**: 32 GiB pinned was held for 30 minutes on both platforms and the limit is above what was tried.
4. **The peak-rate inputs of the performance model are measured** (13.2). The tables of 13.3 were computed with 65.4 / 261 and move by at most 0.7 %.
5. **The two platforms agree on what the GPU does** (P-H2, P-H3, P-H5, P-H12). They differ in host-side costs: the flush (77 vs 8 µs), pinned allocation time (0.8–1.0 vs 0.07 s per GiB), small-copy latency (73 vs 31 µs for 4 KiB) and the memory clock under load (11,251–11,501 vs 11,501 MHz; cause not examined). That is the P-H11 data for P-H3, P-H5 and P-H9. It says nothing about the engine's +25–28 % decode difference, which needs the engine's own runs.

**Still unmeasured, each with the way to measure it.**

* P-M4 (model-file read speed): needs the model path (`PELIAS_GGUF` or `-Gguf`); the path is never written to a result.
* Stock-clock values of P-H3, P-H10 and P-H4: reset the overclock tool, rerun with the tag `stock`.
* The Windows jitter figure: ask for a finer timer in the Windows build (`timeBeginPeriod(1)` or a high-resolution waitable timer), rerun P-M3; an idle run (`--pinned-gib 0 --pageable-gib 0`) gives the comparison for a threshold.
* The pinned limit above 32 GiB: raise `--max-gib`.
* The memory clock under load (11,251 vs 11,501 MHz, and 11,501 at idle): `nvidia-smi dmon` across idle → load on each platform.
* Resizable BAR: no on/off comparison exists. Switch it off in the BIOS (BAR1 should then read 256 MiB), rerun P-H2, P-H5, P-H6 and P-H12 with the tag `rebar_off`, and switch it back on.
* The display timeout for full-power tensor kernels, and for a screen on the same GPU in laptop display modes.
* VRAM drift while the machine is in use (the disturbance test). No run logged GPU memory over time: the 30-minute soak's sampler records clocks, temperature and power only, so an app starting mid-run was never seen. What exists: the memory Windows held before our context was 914 and 1,070 MiB on the two WSL runs and 1,015 and 1,031 MiB on the two Windows runs, while CUDA free stayed 10.830 / 10.846 GiB. Way to measure: add NVML used and CUDA free (peak and floor) to the soak's sampler, then during a soak open a browser, play a video and start a game and record the largest move; the margin is that drift plus a small buffer. This needs a change to the probe source, which has not been made.
* Whether memory is held between probes. Before 2026-10-24 only P-H8 logged memory before its context. The run scripts now write `vram_log.tsv`, so the next runs show it; if a probe fails, `RETRY_ON_FAIL=1` shows whether flushing matters.
* Hash time for the model key (3.3, U4): time a SHA-256 over the model file, next to the read rate of P-M4.
* The power plan (Balanced during every run): rerun on High performance.
* Whether the engine's own copy/compute overlap improves with the flush: engine side (held); add `cudaStreamQuery` after its enqueues and compare the token rate.
* Native Linux: not measured (tabled).

---

<!-- /SPLICE P3.3 sha256="d0e9efa57322461ed327dc423a73ff16636e6bf1d6ca16aed55399d1d937cf0d" -->

## 3.4 The modelled speed ceiling, the speed-to-ceiling comparison table across all recorded runs, and the six verified gap findings

<!-- SPLICE id="P3.4" src="pelias-research-update.md (received 2026-10-23 19:59, 870 lines)" lines="304-390" shift="+2" ws="1" edited="0" -->
> **Source (verbatim):** pelias-research-update.md (received 2026-10-23 19:59, 870 lines), lines 304-390. Heading levels shifted by +2 for this document only. Table rows that the source breaks over several lines are joined back (whitespace only).

### 4. Engine Cross-Check Against the Speed Ceiling (`PELIAS_ENGINE_CROSSCHECK_SPEED_TO_CEILING.md`)

This section serves directly as the verified content for `PELIAS_ENGINE_CROSSCHECK_SPEED_TO_CEILING.md`.

#### 4.1 The Modelled Speed Ceiling (`88.2–92.0 tok/s`, `10.87–11.34 ms/token`)

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

#### 4.2 Master Speed-to-Ceiling Comparison Table (All Recorded Runs)

Every row below is computed directly from the six logged runs in `benchmarks/RESULTS.md` ($\frac{1000}{\text{tok/s}}$ compared against `10.87–11.34 ms`, and divided by `40` layers) (`DERIVED` from `MEASURED` runs, Trainer Scaffold §16.4.1 & Master Notes §1.15, `2026-10-23`):

| Recorded Run & Configuration | Measuring Client | Context | Speed (`tok/s`) | Token Time (`ms/token`) | Total Gap to Ceiling (`10.87–11.34 ms`) | Per-Layer Gap (`µs/layer`) | Label & Ledger Ref |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1. WSL2 Reference Cell A** (`7.0 GiB` bank, bypass 32, `-ub 4096`, timer off) | Streaming client (§9.2) | Short (`ctx0`) | **`37.9`** | `26.39` (p50 `26.33`) | **`15.05–15.52 ms`** (`15.0–15.5 ms`) | **`376–388 µs`** | **MEASURED / DERIVED** (`2026-10-22`) |
| **2. WSL2 Reference Cell A** (`7.0 GiB` bank, bypass 32, `-ub 4096`, timer off) | Streaming client (§9.2) | `12k` | **`42.1`** | `23.75` (p50 `23.34`) | **`12.41–12.88 ms`** (`12.4–12.9 ms`) | **`310–322 µs`** | **MEASURED / DERIVED** (`2026-10-22`) |
| **3. Windows-Native Cell A (`r1 / r2`)** (`7.0 GiB` bank, bypass 32, `-ub 4096`) | Streaming client (§9.4) | Short (`ctx0`) | **`47.3 / 46.8`** | `21.14 / 21.37` (p50 `21.53`) | **`9.80–10.50 ms`** (`9.8–10.5 ms`) | **`245–262 µs`** | **MEASURED / DERIVED** (`2026-10-23`) |
| **4. Windows-Native Cell A (`r1 / r2`)** (`7.0 GiB` bank, bypass 32, `-ub 4096`) | Streaming client (§9.4) | `12k` | **`53.7 / 54.1`** | `18.62 / 18.48` (p50 `18.28`) | **`7.14–7.75 ms`** (`7.2–7.8 ms`) | **`179–194 µs`** | **MEASURED / DERIVED** (`2026-10-23`) |
| **5. WSL2 Best Recorded Cell** (`7.5 GiB` bank, bypass 32, `-ub 2048`) | Gauntlet client (§8.2) | Short / `4k` | **`53.5 / 53.0`** | `18.69 / 18.87` | **`7.35–7.82 ms`** (short) | **`184–196 µs`** (short) | **MEASURED / DERIVED** (`2026-10-22`) |
| **6. WSL2 Best Recorded Cell** (`7.5 GiB` bank, bypass 32, `-ub 2048`) | Gauntlet client (§8.2) | `12k` | **`49.0`** | `20.41` | **`9.07–9.54 ms`** (`12k`) | **`227–239 µs`** (`12k`) | **MEASURED / DERIVED** (`2026-10-22`) |

#### 4.3 Six Verified Findings Explaining the Gap Between Live Runs and the Ceiling

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

<!-- /SPLICE P3.4 sha256="09f369b5af4460d933d38c7eb1e6a8eb98b3de94a0e72ba88e36703c14606973" -->

## 3.5 Attribution register: which part of the gap to the ceiling is known, how it is labelled, and which proprietary module owns it (new text)

The six findings of 3.4 are the research update's. The corrections of Part 1.1a change three of their labels. This register restates each finding once, with the governing label, and adds the one column the update does not have: the module of Part 6 that would own the term in the proprietary engine. The last column is a hypothesis for each row, not a result [UNVERIFIED]; no row has been tested on a non-llama runtime.

| # | Term of the gap | Quantity (from 3.4 and the ledgers) | Governing label | What is not established | Test | Owner in Part 6 |
|---|---|---|---|---|---|---|
| 1 | Total gap, WSL2 Cell A, short context | 26.39 ms/token against 10.87 to 11.34 ms: 15.05 to 15.52 ms (376 to 388 µs per layer). At 12k: 12.41 to 12.88 ms | MEASURED runs, DERIVED gap | Only the sum is known; the parts below are one launch's medians | E0 (done once), P-E2 | Whole engine |
| 2 | CPU cold-expert compute and no overlap with the GPU | 10.1 ms of CPU cold compute against 2.86 ms in the model (+7.24 ms), serial with 6.6 ms of GPU compute | **DERIVED** (the update says MEASURED; D4) | Zero overlap is derived from the parts summing to wall time (30.5 against 30.3 ms), not observed directly | P-E2 per-layer timers; E6 overlap gate | CPU threadpool and doorbell coordinator (6.6, 6.7) |
| 3 | Stream-sync handoff and activation copies | 4.6 to 5.0 ms or more timer-free (5.1 ms with the timer on); model 0.25 ms | **DERIVED**, a lower bound | The doorbell's 13.3 µs is a microbenchmark, not a feature of a running engine | E6 handoff gate; P-H9 | Doorbell coordinator (6.6) |
| 4 | Copy-back and launch | 2.8 ms (copy-back 2.3 plus launch 0.5) | **DERIVED** (the update's 2.75 and 8.04 ms graph figures are not adopted; D4) | Cost of the same step with CUDA graphs on and off is not in the ledger | E8 graph gate | Decode loop with static graphs (6.7) |
| 5 | Windows-native against WSL2 on the same cell | 5.25 to 5.02 ms at short and 5.13 to 5.27 ms at 12k (24.8 % and 23.5 %; 27.6 % and 28.5 %) | MEASURED / DERIVED | "Matches the WSL2 sync cost" is a **hypothesis** (D6); the pure OS effect at the best settings is unmeasured because the two cells used different clients | P-H11; three launches per side | Platform layer, two builds (6.5) |
| 6 | Settings against operating system | WSL2 Cell A to the 7.5 GiB / -ub 2048 cell: +15.6 tok/s (-7.70 ms), 1.66 times the Windows-against-WSL2 gain on Cell A; at 12k the ranking flips | DERIVED | Different measuring clients; Windows-native not run at 7.5 GiB and -ub 2048 | Windows-native run of the best cell | Memory plan and VRAM governor (6.9) |
| 7 | Any-miss share | 63.9 % of layer-steps (25.56 of 40 layers per token) have at least one cold expert; 36.1 % have none; mean 2.16 misses per missing layer; byte-weighted hit rate 0.8273 on 36,657 held-out steps | MEASURED trace, DERIVED histogram; the fixed per-layer cost is UNVERIFIED | c0 against c1 | P-E2 | Hot-set solver (6.9) |
| 8 | The 41st CPU split | 82 splits per token = 41 CPU + 41 CUDA; block_count 41 | The 82 is MEASURED; that the 41st is blk.40 is **UNVERIFIED** (D5) | The server log says 40 layers hot-banked | P-E2 by layer index; MTP off against on | Model reader and plan (6.8, 6.9) |
| 9 | Q6_K row geometry | 210-byte blocks, 420-byte rows in 20 trunk layers plus blk.40; 9.59 GB/s against 26.2 GB/s | READ-SRC geometry, MEASURED rates; the lift from staging is UNVERIFIED | Whether staging reaches 20 to 26.2 GB/s | P-E10 | Spine and expert kernels (6.10) |
| 10 | Behaviour the tier took from llama.cpp | Not quantified | UNVERIFIED | What the server did that the earlier work did not account for (author, 2026-10-24) | The llama-server catalogue (6.14) | Server (6.12, 6.14) |

**What the register says in one sentence.** The measured gap is large (about 15 ms of 26.39 ms on the WSL2 reference cell), its biggest identified parts (rows 2 and 3) are derived from one launch's timers, and no test has yet changed one part while holding the others, so the ceiling's 88.2 to 92.0 tok/s is a model, not a target the tier was shown to approach [DERIVED].
