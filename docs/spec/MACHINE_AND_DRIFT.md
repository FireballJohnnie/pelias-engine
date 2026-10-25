# Machine and toolchain, engine facts, drifted assumptions, and the claim-by-claim review of the outside report

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Parts 1.2 to 1.5 (master lines 299 to 443). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 1.2 Machine and toolchain (trainer master 2.1)

<!-- SPLICE id="P1.2" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="191-212" shift="+1" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 191-212. Heading levels shifted by +1 for this document only.

### 2.1 Machine and toolchain

| Item | Value | Label |
|---|---|---|
| GPU | RTX 4070 12 GB (AD104, `sm_89`): 46 SMs, 5,888 CUDA cores, 184 tensor cores, 36 MiB L2, 504 GB/s, 200 W, boost 2,475 MHz | PUBLISHED (Tom's Hardware review) |
| GPU overclock | memory +1000 MHz; core ≈ 2,775–2,790 MHz at 965 mV; **no ECC** | author-stated; stability is a probe (P-H4, P-H10) |
| Shared memory | 100 KiB per SM; 99 KiB max per block on CC 8.9 | PUBLISHED (LSU GPU course table); probe P-H1 reads the real values |
| CPU / RAM | Ryzen 7 5800X3D (8c/16t, **AVX2 only**, 96 MB L3); 4×16 GB DDR4-3200 (51.2 GB/s peak) | author-stated / PUBLISHED |
| Host bandwidth | CPU alone 45.9 GB/s; GPU zero-copy read alone 26.1 GB/s; together 47.6 GB/s = 93 % of peak | MEASURED (engine ledger) |
| OS | Windows 11 Pro (build 26300) host + WSL2 (kernel 6.18.54.1; 48 GB cap, 48 GB swap on D:); the trainer is designed to build on both WSL2 and Windows-native; the hardware probes were built and run on both (11.12), the trainer itself has not | author-stated / DECISION; MEASURED (build number, from the Windows facts run) |
| Pinned memory | **32 GiB (= 34.4 GB) pinned and held on both WSL2 and Windows-native** (P-H7; the largest size tried, so the limit is **not found**). The engine's earlier 20,448 MiB was the largest size it tried, not a limit. `RLIMIT_MEMLOCK` = 64 MiB does not stop `cudaHostAlloc`; use `cudaHostAlloc` (`cudaHostRegister` gives a device pointer ≠ host pointer on WSL2) | MEASURED (P-H7, 2026-10-23/24; engine ledger for the pointer rule) |
| Driver / toolkit | driver 617.42 (CUDA 13.4 capable); WSL2: `nvcc 13.0.88`; Windows: `nvcc 13.4` and Visual Studio 18 Professional (used for the probe build); **avoid toolkit 13.2** | author-stated; MEASURED (Windows facts run); PUBLISHED (Unsloth warning) |
| llama.cpp | commit `2ca15f540` with carried patches `mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu` | READ-SRC |
| VRAM ceiling | **10.7 GiB**, absolute (author ruling 2026-10-23; 10.9 from 2026-10-22 until P-H8 measured the free memory). The basis it is measured on is open: 16.3 X15 and 16.7 decision 7 | author-stated (DECISION) |
| Idle VRAM | 579 MiB after a fresh WSL restart; Windows `dwm` can hold ~3 GB after long uptime — gate on `nvidia-smi` before a run. 2026-10-23/24: NVML used before the CUDA context 914 MiB (WSL2 run) and 1,031 MiB (Windows run); with WSL off `nvidia-smi` showed 683–744 MiB used; the GPU-memory counter put `dwm` at 884 MiB | MEASURED (earlier sessions; P-H8; Windows facts run) |
| Tensor-core peaks (4070) | BF16/FP16 FP32-acc **58.3 TFLOPS** (65.4 at 2.775 GHz); INT8 **233.2 TOPS** (261.4 at OC); FP8 FP32-acc 116.6 | DERIVED from published RTX 4090 rows by SM × clock (46/128 × 2.475/2.52); **P-H3 has replaced the overclocked values with 65.2 / 259.5 (row below); the stock values are still DERIVED** |
| Ada FP16 with FP16 accumulate | twice the FP32-acc rate on 4090 (330 vs 165 TFLOPS); marketing "233 (466)" for the 4070 are *sparse FP16-acc* figures — not usable for training | PUBLISHED (aiwiki table, NVIDIA forum thread) |
| Free VRAM (P-H8) | Total 11.99 GiB. Once the CUDA context exists CUDA reports **10.83 GiB free on WSL2 and 10.85 on Windows-native** (NVML 10.95 / 10.85); the same 10.83 appears in 21 engine logs, so it behaves as a fixed driver budget and not as desktop load (inference). The context costs 154 / 138 MiB; Windows holds 914 / 1,031 MiB before the process starts | MEASURED (P-H8, 2026-10-23/24; engine logs) |
| Sustained tensor rates (P-H3) and soak (P-H4) | bf16 65.2, fp16 64.9, int8 259.5, fp8-e4m3 129.8 T(FL)OPS at an SM clock of 2,760–2,775 MHz, 99.8 % of the burst rate, the same on both platforms; 30-minute soak: 0 output mismatches, a flat clock, 43–56 °C, ≈ 107 W of the 200 W limit | MEASURED (11.12) |
| Memory and PCIe (P-H2, P-H5) | VRAM read 523–534 GB/s, write 448–454; host DRAM read best 43.7–44.8 GB/s (4 threads); PCIe 4.0 x16: pinned H2D and D2H 26.3 GB/s each at 256 MiB, pageable 10–12 GB/s | MEASURED (11.12) |
| Windows-side settings | Hardware GPU scheduling off; display-timeout (TDR) settings at the Windows default; power plan Balanced; Resizable BAR on (BAR1 total 16,384 MiB); fixed 32 GiB page file; 63.9 GiB RAM as 4 × DDR4-3200; no display-driver errors in the 24 h before the runs | MEASURED (Windows facts run, 2026-10-23); the effect of the power plan and of ReBAR is not tested |

<!-- /SPLICE P1.2 sha256="ab3d57589f98149c0b9dc7bc28e552b692c972facd4fca987b69815dc434a15e" -->

## 1.3 Engine facts the trainer depends on (trainer master 2.7)

<!-- SPLICE id="P1.3" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="387-402" shift="+1" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 387-402. Heading levels shifted by +1 for this document only.

### 2.7 Engine facts the trainer depends on

All from the pelias-engine ledger and raw files (MEASURED unless marked). The trainer is the engine's *upstream*: whatever it writes, the engine serves.

| Fact | Value | Why the trainer cares |
|---|---|---|
| Hot-bank economics | 7.0 GiB bank; live hit rate 0.775–0.779 (1-token), 0.733 (2–31 tokens); simulated 0.806; layers 0–2 lowest (0.61–0.65), layer 20 highest (0.90–0.92); by domain: agent 0.790, code 0.753, persona 0.799, reasoning 0.674; domain hot-set overlaps (Jaccard) are low | Fine-tuning moves the routing distribution. The hot bank must be re-profiled on the *trained* model. The routing tape (D7) provides the statistics for free |
| Float spread | stock-vs-stock KL between llama.cpp batch configurations is 0.0174 / 0.0183 on the engine's test set — larger than the 0.005 strict limit; `tier8_nobank` 0.0017, `tier8` 0.0106; reruns of the same configuration are deterministic. (Column meaning to be re-verified; P-Q4 re-measures on the trainer's eval set) | Every "KL ≤ x" gate in this document is set relative to the measured float spread, never to a round number (author ruling H6) |
| Dequant parity | the engine's cold kernel dequantizes bit-identically to ggml on sampled blocks | P-Q1 extends this to every tensor of the file |
| The MMQ crash class | the 9–31-token MMQ crash was `ids ? ids[blockIdx.x] : blockIdx.x` mixing signed and unsigned 32-bit so a `-1` sentinel became 4,294,967,295 (fix `quantize_signed_id`) | Rule for our kernels: no signed/unsigned sentinel mixing for expert ids; tests with empty experts and holes (rung 1) |
| MTP speculation | draft 2/3 at a 7.0 GiB bank: 12k-token decode 51.5–52.9 tok/s vs 45.4 base | Training the MTP block alongside the trunk (D12) protects this gain; P-T8 measures acceptance before/after |
| Host bandwidth | CPU alone 45.9 GB/s, GPU zero-copy alone 26.1 GB/s, together 47.6 GB/s (93 % of DRAM peak) | PCIe streaming and CPU work (boundary copies, data pipeline) compete for the same DRAM; P-H6 |
| Pinned-memory rules on WSL2 | `cudaHostAlloc` pointers are identical host/device; `cudaHostRegister` are not; 20.4 GiB pinned works (the trainer's P-H7 later held 32 GiB on WSL2 and on Windows-native) | Allocation policy in `hal_arena` (5.3) |
| Memory ceiling | engine P1 reference cell: 7.0 GiB bank, ubatch 4096 → peak 10.7 GiB; the 7.5 GiB bank peaked at **10.40–10.44 GiB at the default micro-batch** (stamps 130854, 132648) and at **10.67–10.69 GiB at `-ub 2048`** (stamps 132907, 141137), the setting that gives the fast 1,680 tok/s prefill; the 8.0 GiB bank at 11.0–11.19 GiB and the 8.5 GiB bank at 11.4 GiB (> ceiling). These peaks are whole-card figures (NVML used, including the Windows desktop's share) | Headroom under the 10.7 GiB ceiling at the 7.5 GiB bank is **0.26–0.30 GiB at the default micro-batch and 0.01–0.03 GiB at `-ub 2048`; less a 0.1 GiB margin = ≈ 0.16 GiB and none (−0.09) for everything the core adds** (DERIVED on the whole-card basis; at the earlier 10.9 ceiling it was ≈ 0.36 and ≈ 0.11). Read instead as a limit on the process's own allocations, the same peaks would leave ≈ 1.2 and ≈ 0.9 GiB (16.3.1; decision 7). The engine's own non-bank footprint will differ from the tier's, so P-E4 re-measures it at the production micro-batch. An all-Q8_0 core (+0.59 GiB) fits at neither; all-Q6_K (+0.28 GiB) fits at neither on the whole-card reading; the loss of bank to a smaller bank is ≈ −0.03 hit per 0.59 GiB (DERIVED by interpolating the engine's static-bank curve 7.0 → 6.41 GiB: 0.805 → 0.778). P-E4 measures the real peak with real merged cores; Mode L (7.12) is the trainer-side lever that could buy hit rate back |
| Machine state | BAR1 16 GiB-used mystery on WSL2; THP set to `always`; HAGS off; Afterburner settings; file-read speeds | Recorded in the engine notes; P-H8 re-checks real free VRAM before runs and (2026-10-23/24) records BAR1 on both platforms: total 16,384 MiB, used 0 MiB inside WSL2 and 16,355 MiB on Windows-native (meaning not examined) |

<!-- /SPLICE P1.3 sha256="13a2f6676d9f146981268bd7661e5ecf1d0e5967292f289a29bca730ff54f669" -->

## 1.4 Drifted assumptions and discrepancies, with the 7.5 GiB budget at 10.7 GiB (trainer master 16.3 and 16.3.1)

<!-- SPLICE id="P1.4" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="2092-2128" shift="+1" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 2092-2128. Heading levels shifted by +1 for this document only.

### 16.3 Drifted assumptions and discrepancies — register

Every line is something that was assumed, stated, or inconsistent, with what the evidence shows and what was done. "This doc" means the trainer plan; "engine" means the engine blueprint (changes in 16.7).

| # | Assumption or discrepancy | Evidence | Resolution |
|---|---|---|---|
| X1 | An all-Q8_0 core is the safe default (this doc, first draft) | +0.59 GiB vs 0.16 GiB / none free at the 10.7 ceiling (DERIVED from the engine's peaks; 0.36 / ≈ 0.11 at 10.9); engine kernels planned for Q4_K/Q6_K/F32 only | D4 revised; D22, D23; 4.3, 9.4 |
| X2 | The earlier summary of arXiv 2608.18261 ("every setting broke its 1 % gate", in the engine blueprint) | Paper read in full: untrained rerouting costs +0.8…+2.0 % at ≈ 30 % fewer misses and +4…+5.8 % at 50 %; training from scratch at 137 M/340 M never met "≥ 30 % at ≤ +1 %"; but rerouting after locality training costs +0.9 % vs +7.7 % | 7.12 table; engine blueprint scope line to be corrected (16.7). The paper is evidence that **inference-time** clamping is costly and that training changes the price — not that router training is impossible |
| X3 | 2 MiB huge pages are *required* to avoid GPU/CPU TLB trouble | CPU read test: 44.86 GB/s with 2 MiB pages vs 45.81 GB/s with 4 KiB (MEASURED, ledger §1); WSL is already booted with `transparent_hugepage=always`; GPU effect not measured | Optional; P-E11 decides. Not a requirement of the exporter |
| X4 | `uint4` coalescing is the *established* cause of the 9.59 vs 26.2 GB/s gap | The ledger itself calls the byte-load cause "unverified". The checksum kernel (26.2) and the compute kernel (9.59) read the same pinned pages, so page size alone cannot be the cause (DERIVED) | Hypothesis, probe P-E10. The WDDM 4-KiB statement is true (16.4 G2) but does not explain this gap |
| X5 | "KL ≤ 0.017353 / 0.018286 against the oracle" | Stock-vs-stock at batch 16 / 1024 against a batch-8 reference, 16 chunks × 1,024 tokens, on the base model | Three KLs, three gates (16.5 B3); initial values are the engine's numbers; re-measured on the trainer's tokens (P-Q4) |
| X6 | Zero spill on the 10 full-attention layers is needed because they are latency-critical | Hit 0.8299 vs 0.8265 elsewhere; zero-miss 0.3636 vs 0.3602; full residency costs 4.605 GiB and lowers the overall hit to 0.7114 (P-E1). Per-layer latency cost by layer type is **unmeasured** (P-E2) | Optional L-Z; weights uniform until P-E2 |
| X7 | FP16/BF16 accumulator divergence between the AVX2 and CUDA reduction trees | K-quant dot products accumulate in int32, scales in fp32; the real difference is Q8_K (256-element blocks, CPU) vs q8_1 (32-element blocks, CUDA) activation quantization | 16.5 B5; P-E12 measures, P-E13 audits the engine's kernels |
| X8 | Engine blueprint: "CPU read of cold experts 26.6 GB/s" | 26.56 GB/s is the CPU's share while the GPU reads at the same time; CPU alone reads 43.8–45.9 GB/s; the combined 47.62 = 26.56 + 21.06 | Engine table to be corrected (16.7). The ceiling uses 47.6 combined, which already assumes the GPU share at the checksum-kernel rate |
| X9 | Engine blueprint scope: "a Windows-native build is not in scope" | Ledger §9.4: Windows-native decode is 25 % (short) and 28 % (12k) faster than the WSL cell (two Windows launches vs one WSL launch; not yet a claim); the difference is ≈ 5.2 ms per token | Decision for the author (16.7). The trainer re-measures both platforms for the exported model either way (9.5 step 4) |
| X10 | An outside report's Windows figure of "57.0 tok/s (17.54 ms/token)" | Recorded Windows-native runs: 47.3 and 46.8 tok/s short, 53.7 and 54.1 at 12k. 57.0 appears in the logs only as the top of a 3-request range at 4k depth for an 8.5 GiB WSL bank (median 53.7). The author says the number may have been mistyped | Not used. Gap analysis recomputed from the recorded runs (16.4.1) |
| X11 | Engine bank peak "10.40–10.44 GiB at 7.5 GiB" taken as the headroom for all settings | The same bank peaks at 10.67–10.69 GiB at `-ub 2048` | Both stated everywhere (2.7, 9.4, P-E4); the engine blueprint should get a VRAM-budget row (16.7) |
| X12 | Engine blueprint: "After E13 the lessons go back into the pelias-trainer" | The trainer's byte budget, hot-set writer and gates depend on E-phase results now | A two-way contract (16.7): which engine milestone releases which trainer input |
| X13 | An outside report: pinned memory is capped at ≤ 32 GB under WDDM | **Refuted on this machine:** P-H7 held 32 GiB (= 34.4 GB) of `cudaHostAlloc` memory on WSL2 and on Windows-native with nothing swapped in (2026-10-23/24); the engine's earlier 20,448 MiB was the largest size it had tried, not a limit. The limit above 32 GiB is not found | P-H7 raised to `--max-gib` above 32 if more is ever needed |
| X14 | An outside report: lock GPU clocks at 2475 MHz (`nvidia-smi -lgc`) | The card runs overclocked (core ≈ 2790 MHz at 965 mV, memory +1000 MHz); 2475 would be an under-clock, and clock-lock support under WSL is unverified | Not adopted. Clock policy stays with the author; the trainer logs the clocks (15.3) |
| X15 | "The ceiling" is one number with one meaning | Three different quantities carry the name. (a) The engine's ruling (10.9, engine blueprint) is compared with whole-card peaks: NVML used, including Windows' share (MEASURED, engine ledger). (b) The engine's gates say "VRAM at most 11.2 GiB" (P1 and G3, whole card): that is where the 11.2 in the author's memory and in the first P-H8 rule comes from. (c) P-H8, the governor's plan sum and `cudaMemGetInfo` see only the process. The card holds 0.89 / 1.01 GiB for Windows before the process starts and 1.04 / 1.14 GiB once its context exists (NVML used, A / B, P-H8); CUDA free after the context is 10.83 / 10.85 GiB | Decision 7 (16.7), ruled 2026-10-24 for a probe-measured, process-only ceiling. Headroom under both readings is in 16.3.1; the other chapters still print the whole-card figures (the engine's basis, and the stricter) and say so |
| X16 | First P-H8 rule: free after the context ≥ ceiling + 0.3 (11.2 GiB) | Unreachable on this card: CUDA free is 10.83 GiB (WSL2) and 10.85 (Windows-native), the same on every run and in 21 engine logs, so it is a fixed driver budget (inference), not desktop load; NVML used before the context moved by about 0.16 GiB between runs (sign-out, restart) while CUDA free did not move | Rule replaced: the smaller of CUDA free and NVML free after the context ≥ 10.7 (margin 0, a margin can be added); ceiling lowered to 10.7 by the author (D26) |

#### 16.3.1 The 7.5 GiB budget, in one table

| Quantity | Value | Label |
|---|---|---|
| VRAM ceiling | 10.7 GiB | DECISION (author ruling 2026-10-23; 10.9 from 2026-10-22 until then) |
| Bank | 7.5 GiB, 4,170 experts, including one zero slot per layer (72.6 MiB) | READ-SRC (engine hot-set sim) |
| Whole-card peak, default micro-batch | 10.40–10.44 GiB → non-bank 2.90–2.94 GiB → headroom to 10.7: 0.26–0.30 GiB | MEASURED / DERIVED |
| Whole-card peak, `-ub 2048` | 10.67–10.69 GiB → non-bank ≈ 3.2 GiB → headroom to 10.7: 0.01–0.03 GiB | MEASURED / DERIVED |
| Whole-card peak, 8.0 GiB bank | 11.0–11.19 GiB → over the ceiling | MEASURED |
| Δ_max for the core's extra bytes (0.1 GiB margin), whole-card reading (W) | **0.16 GiB** (default micro-batch; 0.20 at the lowest peak) / **none** (`-ub 2048`: −0.09 GiB) | DERIVED; P-E4 re-measures with the engine's own allocator |
| Δ_max, process-only reading (P): the same peaks minus the ≈ 1.0 GiB Windows already holds before the process starts | engine's own use ≈ 9.4 / 9.7 GiB; headroom to 10.7 ≈ 1.3 / 1.0 GiB, ≈ **1.2 / 0.9 GiB** after the margin | DERIVED; the Windows share moved by about 0.16 GiB between probe runs, so P-E4 must record NVML used before the engine starts as well as at the peak |

The "whole card" figure includes the Windows desktop's share, so it is a ceiling-relevant number rather than the engine's own footprint. **Consequences of the two readings (decision 7; ruled 2026-10-24 for the probe-measured, process-only ceiling, with the whole-card figures kept as the stricter planning figures until P-E4 re-measures; 3.3, U2).** On the whole-card reading (W) the earlier conclusions get worse, not better: an all-Q6_K core (+0.28 GiB) fits at neither micro-batch, so classes are raised selectively; for example shared experts + attention + DeltaNet-out to Q6_K is +153.1 MiB (25.7 + 65.5 + 61.9, Appendix E), which fits the 0.16 GiB (163.8 MiB) at the default micro-batch. Taking the rest from the bank costs hit rate: the shortfall of an all-Q6_K core is 0.12 GiB at the default micro-batch and 0.37 GiB at `-ub 2048`, which at the engine's exchange rate (−0.027 hit per −0.59 GiB, ≈ +1 tok/s per +0.01 hit) is about −0.005 hit (−0.5 tok/s) and −0.017 hit (−1.7 tok/s) (DERIVED by interpolation; P-E4 and P-X5 measure it). On the process-only reading (P) the room is about 1 GiB and the 8-bit core (+0.59 GiB) would fit at both micro-batches. The author's choice decides which of these designs is built.

<!-- /SPLICE P1.4 sha256="d3e27857a1738be6cba87dc71d2fb0a3acc7af271e3c81b907da81f5cd914278" -->

## 1.5 The outside research report, claim by claim, and the gap analysis from the recorded runs (trainer master 16.4 and 16.4.1)

<!-- SPLICE id="P1.5" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="2129-2170" shift="+1" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 2129-2170. Heading levels shifted by +1 for this document only.

### 16.4 The outside research report — what was checked and what happened to each claim

The report was produced by another AI system from the engine blueprint and the author's description. It is useful as a **list of hypotheses**; as a source it fails its own claim of being "zero-hallucination": its reference numbers in the body are off by one against its reference list throughout (for example the AMD guide is cited as [Ref 5] in the text and numbered 6 in the list; the THP kernel document is [Ref 6] in the text and 7 in the list), and it attributes the engine's own measured 9.59 → 26.2 GB/s figures to the NVIDIA Ada whitepaper. Each claim below was checked against the engine ledger and, where possible, against the cited primary source.

| # | Claim | Check | Verdict | Where it went |
|---|---|---|---|---|
| G1 | Windows-native 57.0 tok/s (17.54 ms/token); gap 6.67 ms (167 µs/layer) | Recorded Windows runs 46.8–47.3 short, 53.7–54.1 at 12k (ledger §9.4); X10 | **Not supported as stated** | Recomputed in 16.4.1 |
| G2 | WDDM `VidMm` manages system memory only in 4-KB pages | Fetched the cited Microsoft page: *"System memory is always managed in 4-KB pages, while memory segments can be managed at either 4 KB or 64-KB granularity as determined by the KMD."* (section "Physical page size"); the page mentions no 2 MB option | **True** | Context for P-E11. It does not show that 4-KB mappings limit the engine: the checksum kernel already reads those pages at 26.2 GB/s |
| G3 | WSL 3.0.4, kernel 6.18.54.1 | Matches the pasted `wsl --version` (machine state). The release notes could not be confirmed (the page fetched showed older releases only), so the PR numbers (#41904 "perf removed", #41888 stack size) are unchecked | **Version true; details unverified** | The engine already plans in-process CUDA-event timing; no change |
| G4 | The 2× prefill difference between load modes comes from 4-KB guest pages; `madvise(MADV_HUGEPAGE)` fixes it | `--load-mode none` is the **fast** mode and is ordinary anonymous memory (4-KB pages, per the engine's audit-script comment); WSL has `transparent_hugepage=always`; the engine's own planned test is to read `AnonHugePages` while each mode serves | **Unsupported; contradicted by the direction of the evidence** | Engine unknown "cause of the load-mode effect" kept; P-E11 |
| G5 | Windows: allocate the CPU cold bank with `MEM_LARGE_PAGES` (needs `SeLockMemoryPrivilege`), the small staging buffer with `cudaHostAlloc` | API facts are as documented; benefit unmeasured; the CPU gain from 2 MiB pages on WSL was nil | **Plausible, unmeasured** | Engine test E0.6-style, only if Windows-native is adopted (X9) |
| G6 | Pin 8 workers to the 8 physical cores (1T) rather than 16 SMT threads; Zen 3 partitions load/store/retire queues in 2T mode (AMD guide §2.14, Table 3) | §2.14/Table 3 exist; the extracted text cannot show the classification of those three queues; the retire queue "can hold up to 256 macro ops or 128 per thread in SMT mode" (§2.10.3). On stock llama.cpp 5–7 generation threads beat 10 (ledger §8.8) | **Partly verified; test it** | Engine E4 A/B (8 pinned vs 16) |
| G7 | Use acquire/release, not seq_cst, for handoff flags; after a `LOCK` avoid FP instructions (AMD §2.15) | §2.15 quote verified exactly: *"Following a LOCKed instruction, refrain from using floating point instructions as long as possible."* On x86 a seq_cst store compiles to `XCHG` (locked); an acquire load and a release store are plain `MOV` | **True; benefit unmeasured** | Engine coding rule where the memory model allows it; A/B in E6 |
| G8 | `PREFETCHNTA` for cold streaming reads to spare the 96 MB victim L3 | Mechanism plausible (L3 is a victim cache); unmeasured | **Hypothesis** | Engine E4 A/B |
| G9 | `uint4` loads staged through shared memory lift zero-copy compute reads from 9.59 to 26.2 GB/s | The engine's own E0.5/E5 item; mechanism plausible; not measured; the citation in the report (Ada whitepaper) is wrong | **Hypothesis (already on the engine's list)** | P-E10 |
| G10 | Pin 16–24 MB of the 36 MB L2 for spine activations and DeltaNet state (`cudaAccessPolicyWindow`) | The API exists; the achievable persisting size on this card under WDDM/WSL is unmeasured | **Hypothesis** | Engine E5 A/B (query `cudaDevAttrMaxPersistingL2CacheSize` first) |
| G11 | Hold `/dev/cpu_dma_latency = 0`; lock GPU clocks | The first applies to native Linux; X14 for the second | **Native Linux only / not adopted** | Engine E1 only |
| G12 | Native Linux projected at 72–86+ tok/s | No measurement exists; the engine's E1 phase is the measurement | **Not used** | — |
| G13 | Gate thresholds (≥ 22 / ≥ 25 GB/s zero-copy; ≥ 27 GB/s cold; handoff p99 ≤ 12 µs native) and the calendar | Invented by the report; the engine blueprint has its own gates and the author sets the dates | **Not adopted** | Candidate gates may be set after the E0/E1 floors |
| G14 | RTX 4070: 46 SMs, 5,888 cores, 192-bit bus, 504 GB/s, 36 MB L2, compute capability 8.9 | Consistent with published AD104 data and with this document's 2.1 | **Consistent** | — |
| G15 | Pinned host memory capped at ≤ 32 GB | X13 | **Unverified** | E11 / P-H7 |
| G16 | WSL2 runs at 44–54 tok/s short and 43–49 at 12k | Engine ledger §8.2: 53.5 / 53.0 / 49.0 tok/s with the 7.5 GiB set, bypass 32, `-ub 2048`. The reference cell A (7.0 GiB, bypass 32, `-ub 4096`, timer off) gives 37.9 / 42.1 | **Supported for the best recorded cell, not for the reference cell** | 16.4.1 (row added). The 5.1–5.2 ms Windows–WSL difference is a like-for-like figure on the reference cell only |
| G17 | The `/dev/dxg` VMBus costs 25–35 µs per layer (1.0–1.4 ms per token) and explains why Windows is faster | Arithmetic: 1.0–1.4 ms is 19–27 % of the 5.1–5.2 ms measured difference on the reference cell. The 57.0 baseline it explains is itself unsupported (G1) | **Unmeasured; at most a fifth to a quarter of the measured gap** | Engine handoff probe (the ledger has 94 µs sync against 13.3 µs resident-doorbell handoff on WSL) |
| G18 | Windows-native: pin workers to physical cores, turn off power throttling (`SetThreadInformation`), `timeBeginPeriod(1)`, to cut launch-to-launch spread (stated as 5–21 %) | The calls exist as documented; the spread figure was not checked against the ledger here; no Windows-native thread-pinning run exists | **Hypothesis, Windows-native only** | Engine, only if Windows-native is adopted (decision 4 in 16.7) |

#### 16.4.1 The gap to the ceiling, recomputed from the recorded runs

The ceiling model gives 10.87 ms per token (92.0 tok/s) with the checksum-rate GPU read and 11.34 ms (88.2 tok/s) with today's 9.59 GB/s compute kernel (engine ledger §4, blueprint). Per-layer figures divide by 40.

| Run (recorded) | tok/s | ms/token | Gap to 10.87–11.34 ms | µs per layer |
|---|---|---|---|---|
| WSL cell A, short | 37.9 | 26.39 | 15.0–15.5 ms | 376–388 |
| WSL cell A, 12k | 42.1 | 23.75 | 12.4–12.9 ms | 310–322 |
| Windows-native r1 / r2, short | 47.3 / 46.8 | 21.14 / 21.37 | 9.8–10.5 ms | 245–262 |
| Windows-native r1 / r2, 12k | 53.7 / 54.1 | 18.62 / 18.48 | 7.2–7.8 ms | 179–194 |
| WSL best recorded cell (ledger §8.2: 7.5 GiB set, bypass 32, `-ub 2048`), short / 4k / 12k | 53.5 / 53.0 / 49.0 | 18.69 / 18.87 / 20.41 | 7.4–9.5 ms (7.35–7.82 for the short run, 9.07–9.54 for the 12k run) | 184–239 |
| *(the report's figure, for comparison)* | *57.0* | *17.54* | *6.2–6.7 ms* | *155–167* |

The §8.2 row uses the same ceiling as the others; that ceiling was not recomputed for a 7.5 GiB set. The §8.2 speeds come from the gauntlet run (ledger §8.2), the other rows from the author's streaming client (ledger §9.2); two measuring clients were used, so comparisons across that line are indicative, not exact (DERIVED, caveats).

The Windows–WSL difference is 5.1–5.2 ms per token at both depths (DERIVED), about the engine's measured timer-free sync-and-copy handoff on WSL (≥ 4.6–5.0 ms). That difference is measured on the reference cell only. On WSL, the §8.2 cell's short-context run (53.5) is 15.6 tok/s above the reference cell's (37.9), more than the whole Windows–WSL difference on the reference cell (47.3 against 37.9), so the cell's settings appear to matter more than the operating system. At 12k the order is different: Windows reference cell 53.7 / 54.1 against WSL §8.2 49.0. Windows-native has not been run in the §8.2 configuration, and the two sets used different measuring clients, so the OS effect at equal settings is unmeasured (DERIVED comparison of recorded runs). The ledger notes that three launches per side are still needed, so this is a consistent pattern, not a result. If the author has a newer Windows run than those in the repository, it should be added to the ledger with its raw file before it is used anywhere.

<!-- /SPLICE P1.5 sha256="4e939d8db70e6e36b39c2fcc5fec6c80aa0f29fb7750b9f1be2adc485991ccda" -->
