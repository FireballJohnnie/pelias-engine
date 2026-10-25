# Pelias-Engine lineage: how the plan and the numbers came to be what they are

Written 2026-10-25, as part of the consolidation recorded in the blueprint as `[sync 11]`. The engine is **HELD** (author, 2026-10-24 06:07). This is documentation: it contains no engine code and lifts no hold.

**What this file is for.** The author asked that old material which is not otherwise listed in this repository be noted together with *how it led to the new derived results*. This file does that. Section 2 is the timeline. Section 3 takes each figure or plan that changed and gives the earlier value, the current value, what changed it, its label and where the evidence is. Section 4 is the lineage of the tier build (the patches on llama.cpp). Section 5 is the register of old material: every working folder that was searched, what is in the archive from it and which result it leads to. Section 6 lists the discrepancies the search turned up. Section 7 lists superseded figures that must not be quoted as data. Section 8 says what the records show and what they cannot answer.

**Labels** are the engine's eight: MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED. A DECISION marked "proposed" is not ruled.

**Where the full records are.** The complete history (the handoffs, rundowns and the runbook, word for word) is kept in the author's local archive and is not in this repository, because those files name private scripts and equations. What follows was taken from them and from the files in this repository; where a statement rests only on the local archive, the evidence column says so.

---

## 1. The short answer

**What the records show.** The project began as a **training** problem: fine-tuning a 35-billion-parameter mixture-of-experts model on one 12 GB card, with the 4-bit experts streamed layer by layer from RAM. Measuring how fast the experts could be streamed, and what stood idle meanwhile, raised the **inference** question; that question became pelias-engine. The first answer was built as **a patch set on a pinned llama.cpp commit**, because a new runtime would have needed CUDA graphs, DeltaNet and attention kernels, a tokenizer, a sampler, a chat template and a server, and llama.cpp had all of them. Even so, the first tier blueprint listed the full ground-up engine as its **last, trigger-based phase** and said "I do not recommend starting here". The engine blueprint of 2026-10-22 then made the ground-up engine the plan, in 14 gated phases (E0 to E13), with the llama.cpp tier as the measuring vehicle. On 2026-10-24 at 08:15 the author ruled "ground up means the ground up" (trainer decision D37), and the blueprint now implements that [sync R2].

**Why the speed stopped where it did.** Every decode speed recorded in this project, on WSL2 and on Windows-native, came from the **patched llama.cpp build**. No run of a non-llama server exists. The modelled ceiling (88.2 to 92.0 tok/s) rests on measured hardware rates and a derived time split; the gap to the live figures (44 to 54 tok/s on short prompts and 43 to 49 at 12k tokens on WSL2) is attributed in `docs/spec/WAVE0_AND_CEILING.md` but **not isolated**, because the one experiment that would isolate it, a server that is not llama.cpp, has not been built. The author's inference of 2026-10-24 08:43 (the ceiling was not reached because the runs were not yet on a new server) is consistent with the records and is not yet proven.

**Why the confusion arose.** The author's words of 08:20 and 08:21: the llama.cpp work was understood as **a way to learn how to build the Pelias server from the ground up**, including a wish to collect every llama parameter, and not as a study of llama.cpp for its own sake. The records agree. What they do not contain is the **catalogue** of llama-server's parameters and behaviours that such a study would have produced; it was never written (`docs/spec/PROPRIETARY_RUNTIME_SPEC.md`, 6.14).

---

## 2. Timeline and decision chain

Times are the author's local time on the dates shown. "Evidence" gives the file in this repository; "local archive" means the record is kept on the author's drive and is not in this repository.

| When | What happened | Evidence |
|---|---|---|
| 2026-10-18, 23:16 | The first handoff is written for a reviewing model. The author's intent: fine-tune on one 12 GB card without truncating long samples. The architecture to be kept: a GPU-resident spine with 4-bit experts streamed from pinned RAM | local archive |
| 2026-10-18, 23:59 | The author on writing a new program: "if we have to write our own dang program from the ground up and call it Pelias, i'm down for it." [AUTHOR-STATED] | local archive |
| 2026-10-19, 00:38 to 02:03 | A review finds that the existing trainer cannot load the model on a 12 GB card; two from-scratch trainers (v0.1, v0.2) are delivered with CPU tests | trainer repository |
| 2026-10-19, 08:52 | The first logged Pelias training run starts | trainer repository |
| 2026-10-20, 00:49 and 01:08 | The 5-bit and 4-bit files are exported (Q5_K_M, Q4_K_M) | trainer repository |
| 2026-10-20, 02:40 | The author chooses Q4_K_M | local archive |
| 2026-10-20, 03:25 | The ladder of experts moved to the GPU stops at 12 layers; the card is at 10.2 to 10.4 GiB | local archive |
| 2026-10-20 | The first tier blueprint puts a ground-up engine in Phase 8, "trigger-based"; the three-lane blueprint gives "why not a new runtime"; blueprint v2 repeats it as "why a patch set on llama.cpp and not a new runtime" | `support/archive/c_ai/creator/benchmarks/pelias_tier_blueprint.md`, `pelias_blueprint_v2.1_review_and_patches.md`; the three-lane and v2 files are withheld (local archive) |
| 2026-10-20, 21:32 to 21:44 | The two private repositories are created and the first documents are pushed | repository history |
| 2026-10-20, 23:55 | The end goal is added to both READMEs: make Pelias universal for mixture-of-experts models (a goal, not a result) | `README.md` |
| 2026-10-21, 04:09 to 04:18 | Gates G1 and G2 pass at a 6.1 GiB bank in the simulation | `support/archive/c_ai/creator/moe_track/sim/sim_result_6p1.json`; `benchmarks/RESULTS.md` 8.5 |
| 2026-10-21, 07:45 to 17:1x | The tier is built into llama.cpp; the first run crashes; the first measured speed with the hot bank follows; the crash for chunks of 9 or more tokens takes a day to explain | `benchmarks/RESULTS.md` 8.4; `support/archive/c_ai/backup/2026-10-21/p1_logs/` |
| 2026-10-21, 17:30 | A handoff for a new conversation: where we are, commands, decisions waiting | `support/archive/c_ai/backup/2026-10-21/HANDOFF_2026-10-21.md` |
| 2026-10-22, 14:2x | The crash is fixed: a signed id converted to an unsigned value in the activation quantizer | `patches/stage2-tier-cuda/quantize_signed_id.patch`; `benchmarks/RESULTS.md` 8.4 |
| 2026-10-22, 17:0x | The author directs that the engine targets Linux with a WSL2-compatible path | local archive |
| 2026-10-22, 18:0x | The engine blueprint is written: 14 gated phases, E0 to E13 | `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`; the unedited copy is `support/archive/c_ai/backup/2026-10-22/docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` |
| 2026-10-22, 22:4x | Author rulings: 10.9 GiB ceiling (later superseded), the float spread governs the accuracy gate, four patches carried and two dropped, the P1 reference cell | blueprint, "Decisions needed from the author" |
| 2026-10-22, 23:1x to 23:3x | The E0 hit counter and timeline results; the E0 gap attribution (DERIVED) | `benchmarks/raw/e0-2026-10-22/`; `benchmarks/RESULTS.md` 9; `support/archive/c_ai/creator/moe_track/e0/` |
| 2026-10-23, 01:3x | The Windows-native runs are recorded; commit 34f744f in the engine repository | `benchmarks/raw/windows-native-2026-10-23/` |
| 2026-10-23, 15:2x to 15:5x | The trainer master is finished and pushed; a pause is reached | trainer repository |
| 2026-10-23, 19:59 | The outside research update arrives; it is read in full and compared with the trainer and the engine ledger | `docs/spec/RESEARCH_UPDATE_AUDIT.md`, `docs/spec/RECONCILIATION.md` |
| 2026-10-23, 21:2x to 2026-10-24, 03:1x | The Wave 0 hardware probes are written and run on WSL2 and on Windows-native | `support/archive/c_ai/pelias_probes_win/`; `support/archive/c_ai/backup/2026-10-23/probe_results/`; `docs/spec/WAVE0_AND_CEILING.md` |
| 2026-10-23, 22:2x | The author sets the VRAM ceiling at 10.7 GiB (trainer D26) | `docs/spec/MACHINE_AND_DRIFT.md` |
| 2026-10-24, 06:07 | The engine is put on **HOLD**; the research update is logged for the engine | `support/archive/c_ai/backup/2026-10-24/engine_hold/ENGINE_HOLD_QUEUE.md`; `docs/spec/ENGINE_QUEUE_AND_MEASURABLES.md` |
| 2026-10-24, 06:21 | The draft is to be built as a WSL version and a Windows version; native Linux is held | blueprint `[sync R1]` |
| 2026-10-24, 07:39 | The author's final decisions on the micro-batch, the Q8_0 question and the export default | blueprint `[sync R4]` |
| 2026-10-24, 08:05 and 08:15 | The bf16 shards are found; "ground up means the ground up" | blueprint `[sync R2]` |
| 2026-10-24, 08:20 to 08:50 | The author explains the confusion and asks what else must be built from the ground up; the text census of the tokenizer, the template and the llama-server surface is made | `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` (6.13, 6.14) |
| 2026-10-24, 09:09 and 09:11 | The master specification is requested, with the history ("i want to know why we got to where we got") | local archive |
| 2026-10-24, 10:10 | "next time we talk, we're making this server. ground. up." Read as the direction for the next conversation; **not recorded as lifting the hold** (O13) | local archive; `docs/spec/OPEN_DECISIONS_AND_ORDER.md` |
| 2026-10-25, 18:14 | This consolidation: every file of the working tree searched, the engine documents synchronized, the old material copied into `support/` with a manifest | `support/MANIFEST.tsv`; blueprint `[sync 11]` |

---

## 3. Old figure, current figure, what changed it

Each row is one thing that was believed or planned at one time and is different now. Nothing here is a new measurement; the evidence column points to the file that holds the number.

| # | Item | Earlier | Now | What changed it | Label | Evidence |
|---|---|---|---|---|---|---|
| L1 | Decode speed on Windows-native | An outside draft of 2026-10-23 claimed "57.0 tok/s (17.54 ms/token)" and derived a gap of 6.67 ms per token from it | Recorded runs: 47.3 and 46.8 tok/s short, 53.7 and 54.1 at 12k. The 57.0 figure is **retired** and is never quoted as data | The ledger holds no such run. The only places 57.0 occurs: the upper bound of a 3-request range at 4k on WSL2 (8.5 GiB bank, bypass 9, median 53.7, peak 11.4 GiB, which also broke the 10.9 GiB ceiling of the time), and a prompt-reading rate on a 61-token prompt in a speed file of 2026-10-21. Whether the outside draft took it from either is not provable | MEASURED (the runs); DECISION (retirement) | `docs/spec/RESEARCH_UPDATE_AUDIT.md` (2.1, rows G1 and 16); `docs/spec/RECONCILIATION.md` (D11, A3); `benchmarks/raw/p1-sweep-2026-10-22/results/P_bank_size_bypass9_b85_20261022_191010.log`; `benchmarks/RESULTS.md` 9.4 |
| L2 | Cost of the CPU and GPU handoff per layer | 94 µs with a stream synchronization | 13.3 µs median and 18.7 µs p99 with a resident-kernel doorbell (microbenchmark 03, WSL2, 2026-10-20). Gate E6: handoff p99 no worse than 18.7 µs | A measured replacement, not a ruling. In the live tier the timer-free handoff is at least 4.6 to 5.0 ms per token (about 5.1 in the split) against 0.25 ms in the ceiling model, so the floor is not reached by the tier | MEASURED (floor); DERIVED (live handoff) | `support/archive/c_ai/creator/moe_track/microbench/03_handoff_latency.cu`; `benchmarks/raw/hardware-floors/03_handoff_*`; `benchmarks/RESULTS.md` 9.2 |
| L3 | Modelled ceiling against the live token | About 92 tok/s from three measured inputs and three assumed ones (spine time 7.73 ms, hit rate 0.83, perfect CPU and GPU overlap) | Ceiling 10.87 ms (92.0 tok/s) with the 26.2 GB/s read, 11.34 ms (88.2 tok/s) with today's 9.59 GB/s kernel, 88.2 to 90.9 under the 7.4 ms spine ruling of 2026-10-22. The live token is 24.6 ms (WSL2 Cell A): cold experts on the CPU 10.1 ms, GPU compute 6.6 ms, handoff 5.1 ms, copy-back 2.8 ms, with no overlap | The E0 per-token timeline of 2026-10-22. The parts are medians of the timer-on cell with the timer cost taken out, so the handoff is a lower bound, and the "zero overlap" is derived from the parts summing to the wall time (30.5 against 30.3 ms) | DERIVED (not isolated: all runs were on patched llama.cpp) | `docs/spec/WAVE0_AND_CEILING.md` (4.1 to 4.3); `docs/spec/RESEARCH_UPDATE_AUDIT.md` (row 6); `docs/spec/RECONCILIATION.md` (D4); `benchmarks/raw/e0-2026-10-22/` |
| L4 | Hit rate of the bank | Simulated: 0.762 to 0.765 at 6.1 GiB, 0.806 at 7.0, 0.827 at 7.5, 0.865 at 8.5 (held-out, byte-weighted) | Live hit rate at 7.0 GiB: 0.7753 to 0.7793 | The E0 hit counter patch. The LRU trigger in the risk table (live rate well below the simulation) is not met | SIMULATED, then MEASURED (live) | `benchmarks/RESULTS.md` 8.5, 8.6, 9.1; `support/archive/c_ai/creator/moe_track/e0/e0_hit_counter.patch`; `support/archive/c_ai/creator/moe_track/sim/` |
| L5 | Size of the VRAM ceiling | The blueprint of 2026-10-22 named three options (10.4, 10.9, 11.2 GiB); 11.2 is the figure the P1 and G3 gates use, compared with the whole card. Ruled **10.9 GiB** on 2026-10-22 | **10.7 GiB** (author, 2026-10-23, trainer D26), after probe P-H8 measured 10.83 GiB (WSL2) and 10.85 GiB (Windows-native) of CUDA free memory once a context exists. On 2026-10-24 (D28) the author ruled that the number is measured at start (the smaller of CUDA free and NVML free after the context, minus a margin), with 10.7 staying this machine's value until the margin is measured | The free memory is a fixed driver budget (21 engine logs show the same 10.83), so a rule needing 11.2 free cannot be met on this card. "The ceiling" also names three different quantities (the engine's whole-card comparison, the gates' 11.2, the probe-measured process-only reading); the process-only reading was ruled for 10.7 on 2026-10-24. The origin of the 10.4 option is not recorded | DECISION (AUTHOR-STATED); MEASURED (P-H8) | `docs/spec/MACHINE_AND_DRIFT.md` (rows X15, X16 and the machine table); blueprint "Decisions needed from the author"; `docs/spec/WAVE0_AND_CEILING.md` |
| L6 | Size of the hot bank | About 5.5 GiB "per the project notes" (what is left of the card), then 6.1 GiB (the first set, 3,426 experts, 2026-10-21), 7.0 GiB (the P1 reference cell), 7.5 GiB (4,170 experts including 40 zero slots) | 7.5 GiB is **derived**, not a ruling (O1). The sweep at bypass 9: 6.1 to 8.5 GiB gave short-prompt generation 44.2 to 55.7 tok/s and a GPU peak of 9.0 to 11.4 GiB. Peaks against the 10.7 ceiling: 7.5 GiB, 10.40 to 10.44 GiB at the default micro-batch and 10.67 to 10.69 at `-ub 2048`; 8.0 GiB, 11.0 to 11.19; 8.5 GiB, 11.4 (over) | The ceiling (L5) and the sweep. One launch each; two launches labelled 6.1 GiB differ by 3.5 tok/s at short, so a 0.5 GiB step is not separable but the trend over 2.4 GiB is | MEASURED (speeds, peaks); DERIVED (hit rates, 7.5 as the best step) | `benchmarks/RESULTS.md` 8.6; `docs/spec/MACHINE_AND_DRIFT.md` (memory ceiling row); `support/archive/c_ai/creator/logs/forks/` (leaderboard and per-run stats) |
| L7 | Bypass value (chunks smaller than this take the bank path) | 9, the production setting while the crash was unexplained (a chunk of 9 or more never reaches the dual path) | **32** is the largest useful value: after the fix the bank path is faster up to a chunk of 24, ties from 32, and the stock path's time rises from 250 to 403 ms between chunks of 24 and 32 (cause ESTIMATE: the scheduler's rule for offloading batches of 32 or more). Chunks of 25 to 31 were not measured. Bypass 32 with `--load-mode none -ub 2048` has not been run | The crash fix (L8) | MEASURED; ESTIMATE (cause of the step) | `benchmarks/RESULTS.md` 8.7 |
| L8 | The crash for chunks of 9 or more tokens | Two hypotheses: ids of -1 not skipped (`cuda_hole_fix`), then repeated placeholder rows in the sorted-by-expert helper (`mm_ids_dedup`); neither removed it | Cause found 2026-10-22: in the non-scatter activation quantizer, `ids ? ids[blockIdx.x] : blockIdx.x` converts the id -1 to 4,294,967,295 before widening, so the guard `if (i01 < 0) return;` never fires. Fix: `quantize_signed_id`. Before the fix at bypass 32, prompts of 1 to 12 tokens passed and 13 to 34 failed; after it, 8 to 61 tokens passed | Two diagnostic patches printed the id map and buffer geometry: faulting address minus buffer start = 0x7FFFFFFF800 bytes = 4,294,967,295 rows x 2,048 bytes. The two sanitizer readings of 2026-10-21 (a write 2.4 GB past an 8 KB buffer, a read 8.8 TB past a 2 MiB buffer) are consistent with a wrapped index; they were not decoded at the time | MEASURED | `benchmarks/RESULTS.md` 8.4; `patches/stage2-tier-cuda/`; raw sweeps and sanitizer logs `support/archive/c_ai/backup/2026-10-21/p1_logs/` (`san_N13.log`, `san_N34.log`, `sched_N13.log`, `sweep_*`, `sweep2_*`, `sweep9_*`, `diag_*`); the before-files `support/archive/c_ai/backup/2026-10-21/finish3_cuda_hole_fix/` |
| L9 | Which patches are kept | Six loose patches. Ruling of 2026-10-22: carry `mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu`; drop `mmq_dump_diag` and `mmq_dump_diag2` | "Carry" now means: kept as the reference record and as the patched oracle build; no patched llama.cpp code enters the engine [sync R2]. The current Windows tier tree **contains** all five CUDA-part patches, the two diagnostics included (inert unless `PELIAS_MMID_DUMP=1`). Only `quantize_signed_id` is the real fix; whether `mm_ids_dedup` is needed is untested | [sync R2]; the comparison of 2026-10-25 (section 4). Whether the Windows tree should drop the diagnostics is open | MEASURED (what the tree contains); DECISION (open) | blueprint "Decisions needed from the author"; `docs/spec/PATCHES_AND_TOOLS.md`; `support/tier_build/` |
| L10 | Platform | Engine targets Linux (native or WSL2), tested on WSL2; Windows-native "not supported at first" | WSL2 and Windows-native, built as two versions of one code base; native Linux held and to be a separate entity [sync R1]. Windows-native decoded 25% and 28% faster than WSL2 in the recorded runs (two launches against one) | Author ruling of 2026-10-24 06:21, made for the trainer draft; applying it to the engine is the blueprint's extension of the ruling (O12, open for correction). Three launches per side are needed before a speed claim | AUTHOR-STATED (the ruling); MEASURED (speeds, thin) | blueprint `[sync R1]`; `benchmarks/raw/windows-native-2026-10-23/`; `benchmarks/RESULTS.md` 9.4 |
| L11 | What "engine" means | The first tier blueprint: a patch set on llama.cpp, with a ground-up engine as the last, trigger-based phase ("I do not recommend starting here"). The 2026-10-22 blueprint: a ground-up engine in 14 phases, llama.cpp as the measuring vehicle | Strictly proprietary runtime: no llama.cpp, ggml, libllama, common or JSON-library code is linked, copied or required at run time. The pinned commit 2ca15f540 (b11399) is an external, read-only test oracle under `tests/oracle/` | Author ruling of 2026-10-24 08:15 (trainer D37), reason AUTHOR-STATED: routing through llama brought in behaviour that was not accounted for | AUTHOR-STATED; DECISION | blueprint `[sync R2]`; `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` (6.3); `support/archive/c_ai/creator/benchmarks/pelias_tier_blueprint.md` |
| L12 | Meaning of a failed E8 gate | The kill criterion: fold the findings back into a llama.cpp patch series | **Removed** by `[sync R2]`, because the product is ground-up. Whether to restore it is open (O11) | The same ruling as L11 | DECISION (open) | blueprint "Risks and kill criteria"; `docs/spec/OPEN_DECISIONS_AND_ORDER.md` |
| L13 | Accuracy gate | "No worse than stock" had two readings | The float spread governs (ruling 2026-10-22): no worse than the natural spread between two stock runs at different batch sizes. Stock: KL 0.017353 at batch 16 and 0.018286 at batch 1024. Tier: 0.010568 and 0.011856 at batch 8 (below), 0.017866 at batch 16 (above stock by 0.000513, smaller than either standard error). The E3 gate keeps the stock floor as its reference and adds the trainer gates K1, K2, K3 | The author's ruling and the sweeps of 2026-10-21 and 2026-10-22 | MEASURED; DECISION | `benchmarks/RESULTS.md` 8.3; blueprint E3 `[sync 4]`; `support/archive/c_ai/creator/moe_track/sim/kl_gate.py` |
| L14 | Prefill micro-batch | Left at the default (about 300 prompt tokens/s at 12k in the bank sweep) | An engine setting (D31, D34). `-ub 2048` reaches 1,680 prompt tokens/s at 12k with `--load-mode none` (peak 10.67 GiB; the script's gate P1 passed on this launch), 921 with the default load at bypass 32. Headroom against 10.7 GiB: 0.26 to 0.30 GiB at the default, 0.01 to 0.03 at `-ub 2048` | The sweep of 2026-10-22 and the author's rulings of 2026-10-24 07:39. The micro-batch of the 10.40 to 10.44 GiB runs is not written in the files | MEASURED; DECISION | `benchmarks/RESULTS.md` 8.2, 8.6; blueprint `[sync R4]`; `docs/spec/RECONCILIATION.md` (D2) |

---

## 4. Lineage of the tier build (the patches on llama.cpp 2ca15f540)

The tier build is the measuring vehicle: every recorded speed in this repository came from it. Until 2026-10-25 the repository held the six loose patches but not their base. The base and the current Windows tree were located in the search of the working folders, and the steps below were checked by applying the patches to a clean checkout of commit 2ca15f540 and by comparing files byte for byte.

| Step | What it is | Where | Check |
|---|---|---|---|
| 0 | llama.cpp commit 2ca15f540 (b11399), the pinned upstream | external | [MEASURED] `git apply --check` of steps 1 and 5 on a clean checkout |
| 1 | `pelias_tier_v1.patch` (7 files: `ggml-cpu.c`, `repack.cpp`, `src/CMakeLists.txt`, `llama-graph.cpp`, `llama-graph.h`, `llama-model.cpp`, `llama.cpp`) plus the new sources `llama-pelias-tier.cpp` and `llama-pelias-tier.h` (2026-10-20) | `support/archive/c_ai/creator/moe_track/patches/` (the build script `09_build_pelias_tier.sh` is beside them) | [MEASURED] applies to 2ca15f540. `RESULTS.md` 8 said these were not in the repository; they are now |
| 2 | The stage-2 CUDA series, in this order: `cuda_hole_fix`, `mm_ids_dedup`, `mmq_dump_diag`, `mmq_dump_diag2`, `quantize_signed_id` (files `ggml-cuda/mmid.cu`, `mmq.cu`, `quantize.cu`) | `patches/stage2-tier-cuda/` | [MEASURED] the changed lines of the current tree's three CUDA files equal the five patches in this order |
| 3 | `pin_cold_to_cpu.patch` (pins the cold branch to the CPU backend; 7 lines rewritten, 2 added in `llama-graph.cpp`) | `patches/stage2-tier-cuda/`; its base file is `support/archive/c_ai/backup/2026-10-21/finish3_pin_fix/llama-graph.cpp.before_pin` | [MEASURED] the base file equals `llama-graph.cpp` produced by step 1; the patch applies to it; the result equals the current tree's `llama-graph.cpp` (checksums begin 413692ef and b26bb372) |
| 4 | `win_fseek.patch` (64-bit file seek for Windows: `off_t` is 32 bits there and the model file is over 20 GiB) | `support/archive/c_ai/creator/moe_track/e0/` | [MEASURED] step 1's `llama-pelias-tier.cpp` plus this patch equals the current tree's source once the Windows line endings are removed (checksum begins 7e12e717) |
| 5 | **Current Windows-native tier tree** = steps 1 to 4: one patch on 2ca15f540 (10 files, applies cleanly) and the two new sources. The header equals step 1's. The source has Windows (CRLF) line endings | `support/tier_build/` | [MEASURED] as above |
| E0 | `e0_timeline.patch` (per-stage timeline of one token) and `e0_hit_counter.patch` (live hit counter), the measurement-only builds behind L3 and L4 | `support/archive/c_ai/creator/moe_track/e0/` (with `e0_timeline_analyze.py` and `token_latency.py`) | [MEASURED] not in the current tree: its header equals step 1's and it touches neither `ggml-backend.cpp` nor `llama-context.cpp` |

Files in this repository that carry these steps: `patches/stage2-tier-cuda/` (steps 2 and 3), `support/tier_build/` (step 5), and the archive paths above (steps 1, 4 and E0).

---

## 5. Register of old material that was not listed before

These are the working folders that the search of the whole C:\AI tree found. For each: where the copy is, how many files are in the repository (the others are withheld or duplicate; see `support/MANIFEST.tsv`) and the result it leads to. Paths are under `support/archive/c_ai/`.

| Folder | Files here | What it is | Leads to |
|---|---|---|---|
| `creator/moe_track/microbench/` | 8 | The first microbenchmarks: `01_gguf_expert_sizes.py`, `02_ram_bw_test.cpp`, `03_handoff_latency.cu`, `04_dual_bus_test.cu`, `05_n_sweep.sh`, `06_verify_gate_g0.py`, `07_pin_modes.cu`, `build.sh` | The hardware floors of the ceiling model: 43.8 to 45.9 GB/s CPU read, 94 µs against 13.3 µs handoff, 26.1 to 26.2 GB/s zero-copy, 47.62 GB/s combined, pinned-memory modes. Gate G0 (is the floor model trustworthy) |
| `creator/moe_track/sim/` | 6 | `export_hot_set.py`, `kl_gate.py`, `waterfill_sim.py`, `predictor_eval.py`, `sim_result.json`, `sim_result_6p1.json` | The hot sets, the cache-policy simulation (static 0.765 against global LRU 0.865 at 6.1 GiB), the KL gate and the stock noise floor (L4, L6, L13) |
| `creator/moe_track/trace/` | 7 | The router-trace collector (`moe_trace.cpp`, `05_collect_traces.py`, `build_prompts.py`, `summarize_traces.py`) | The 121,160-step trace (84,503 fit, 36,657 held out) behind every hot set |
| `creator/moe_track/fork/` | 6 | The fork gauntlet scripts, the prompt-reuse and MTP-acceptance tests and the launch-parity check | The speed leaderboard of 138 fork runs in `creator/logs/forks/` |
| `creator/moe_track/mtp_align/` | 6 | The draft-head (MTP) aligner and a C++ hidden-state dumper | The trainer side of the unused MTP block (the 41st CPU split is UNVERIFIED as that block) |
| `creator/moe_track/pelias_serve/`, `pelias_serve_wide/` | 14 and 1 | The 14-file CUDA prototype: cold-weights view, miss splitter, doorbell handoff, Q4_K and Q6_K zero-copy kernels, DeltaNet state, hot-aware MTP policy, parity harness; and the 16-byte-load variant of the kernel (hypothesis unconfirmed) | The module designs of `docs/spec/PROPRIETARY_RUNTIME_SPEC.md`; the 9.59 GB/s kernel and the wide-load question (P-E10) |
| `creator/moe_track/patches/` | 5 | The base tier patch, its two sources and build script, the v0.3 patch layer | Section 4, step 1 |
| `creator/moe_track/e0/` | 8 | The E0 measurement patches, analysis scripts and the Windows seek patch | L3, L4, section 4 |
| `creator/moe_track/bar_probe/` | 8 | BAR-size probes (a C version and a Direct3D 12 version) and four result files of 2026-10-22 | The resizable-BAR question (E-A6), held with native Linux |
| `creator/moe_track/tests/`, `p1_tests/` | 33 and 156 | Unit and hardware tests of the prototype; the P1 test scripts and their results | The P1 sweep and gates; the unit tests the proprietary runtime's tests start from |
| `creator/moe_track/lib/` | 8 | CUDA shim, floor model, gates, `gguf_min.py`, launcher, speed log, tensor reader | The GGUF reader's oracle and the launcher flags the clients use today |
| `creator/moe_track/` (top) | 8 | `README.md`, `UNFINISHED.md`, `env.sh`, the numbered run scripts and `MANIFEST.sha256` | What was unfinished when the prototype stopped (the list the blueprint's component status builds on) |
| `creator/logs/speed_server/` | 458 | Speed-server logs of the prototype runs | The 44 to 54 tok/s record |
| `creator/logs/forks/` | 138 | Per-run statistics and a leaderboard of the fork runs | L6, L7 |
| `creator/logs/moe_trace/` | 99 | Trace summaries and the KL comparison logs | L13 |
| `creator/logs/floors/`, `phase1/`, `phase2/`, `server_*.log` | 4, 3, 2 and 48 | The hardware-floor audit runs and the first server logs | L2; the first recorded speeds |
| `creator/benchmarks/` | 4 | The first tier blueprint, the v2.1 review and patches, published reference numbers, community reports for 12 GB cards | L11 (why a patch set first) |
| `creator/pelias-engine-release/` | 4 | The drive's earlier release folder (2026-10-21): README, CITATIONS, `SCAFFOLD.md`, `RESULTS.md` | Older than the repository (section 6, item 5) |
| `creator/` (root) | 5 | CUDA build scripts for llama.cpp and the machine survey scripts | How the tier build was compiled |
| `backup/2026-10-20/` | 9 | The first backups of the prototype folders and their checksum lists | A dated copy of `moe_track` |
| `backup/2026-10-21/` | 97 | The raw P1 crash and bank sweeps (`p1_logs/`), the sanitizer logs, the before-files of the hole fix and the pin fix, the wide-load check, the handoff of that evening | L7, L8, section 4 |
| `backup/2026-10-22/` | 8 | The unedited copy of the engine blueprint, the terminal logs and checkers of the P1 tests, the upload copy of the README and ledger, the ReBAR exploration brief | The base the synchronized blueprint was edited from |
| `backup/2026-10-23/probe_results/` | 183 | Wave 0 hardware probe results: 106 files for Windows-native, 77 for WSL2 | The measured machine facts (P-H8 free memory, pinned-memory limits, copy-engine overlap) behind L5 |
| `backup/2026-10-24/` | 3 | The engine hold queue and the two interim Windows launcher scripts (`win_bat/start_pelias_server.bat`, `stop_pelias_server.bat`) | The HOLD and what waits behind it; the full flag set and the `PELIAS_TIER_*` variables the clients' server runs with today (the parameters the proprietary server must accept, `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` 6.14) |
| `backup/design_docs_first_conversation/` | 2 | The first conversation's execution-plan status ledger and state map | The early plan |
| `pelias_probes_win/` | 14 | The Wave 0 probe program: sources, run scripts for WSL2 and Windows, checksums | `plan/probe_constants` and the start-up probe |
| `win_build/` | 5 | Build and configure logs and the first Windows-native run summary | L10 |
| `support/tier_build/` (outside the archive) | 3 | The current Windows-native tier build | Section 4, step 5 |
| Withheld, in the author's drive and in `support/local_only/` (not pushed) | 126 | Prompt corpora, the earlier handoffs, ledgers and plans that name private scripts or equations, environment listings, machine reports | Nothing in the engine documents depends on them: what they say about the engine is already in the files above and in `docs/spec/` |

---

## 6. Discrepancies found while searching

| # | What was found | Evidence | What to do |
|---|---|---|---|
| 1 | **The two diagnostic patches.** The ruling of 2026-10-22 dropped `mmq_dump_diag` and `mmq_dump_diag2`; the current Windows tier tree still contains them | Section 4, step 2 | Open decision: keep the tree as it is (no speed effect is recorded either way) or rebuild with three patches. Listed in the blueprint's decisions table |
| 2 | **Microbenchmark numbering.** The research update and the blueprint name the microbenchmark sources as 02 `ram_bw`, 03 `handoff`, 04 `dual_bus`, 05 `pin_modes`, 06 `overlap`, 07 `wide_load`. The files on the drive are `02_ram_bw_test.cpp`, `03_handoff_latency.cu`, `04_dual_bus_test.cu`, `05_n_sweep.sh`, `06_verify_gate_g0.py`, `07_pin_modes.cu`; no source called `overlap` or `wide_load` exists in any searched folder | Section 5 | The names in the blueprint tree (`bench/overlap.cu`, `bench/wide_load.cu`) are planned names. Overlap was measured by the E0 timeline and later by Wave 0 probe P-H12; the wide-load kernel exists only as a variant of the prototype kernel (hypothesis unconfirmed, P-E10 not run). The same holds for the tools and run scripts: the research update names `collect_trace.py`, `export_hot_set.py`, `kl_gate.py`, `speed.py`, `repack_gguf.py` and `run_floors.sh`, `run_base_cell.sh`, `run_sweep.sh`; the files on the drive are `trace/05_collect_traces.py`, `sim/export_hot_set.py`, `sim/kl_gate.py`, `lib/speedlog.py` (the speed client scripts are withheld) and the lettered P1 scripts `p1_tests/A_memory_loading.sh` to `X_b70_mtp2_repeat.sh`; no file called `repack_gguf.py` was found (it is planned, decision D25). The replacement map in the scaffold uses the real file names |
| 3 | **Patch base.** `RESULTS.md` section 8 said the base tier patch and `llama-pelias-tier.{h,cpp}` were not in the repository | Section 4 | Fixed by the archive; `RESULTS.md` is unchanged (it is a ledger of what was true when written) |
| 4 | **Raw folder not cited.** `benchmarks/raw/p1-sweep-2026-10-22/` is in the repository but no line of `RESULTS.md` cites it by path | `benchmarks/raw/p1-sweep-2026-10-22/` (README, run order, results, terminal logs) | Cited here and in the scaffold. The 2026-10-21 raw sweeps that precede the fix are in `support/archive/c_ai/backup/2026-10-21/p1_logs/` |
| 5 | **The drive's `pelias-engine-release` folder is older than the repository** (2026-10-21; the repository holds the files as pushed later and everything after) | `support/archive/c_ai/creator/pelias-engine-release/` | Left as it is on the drive. Refreshing it from the repository is offered, not done |
| 6 | **Two copies of the blueprint.** The unedited one in `backup/2026-10-22/docs/` differs from the repository's earlier version only by the account name written as `<account>` and the E0 update line | `support/archive/c_ai/backup/2026-10-22/docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` | None. The synchronized blueprint is `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` |
| 7 | **Ceiling quoted as "about 92", "88.2 to 90.9" and "88.2 to 92.0".** Not a conflict: 92.0 uses the 26.2 GB/s read, 88.2 the 9.59 GB/s kernel, and 90.9 the 7.4 ms spine ruling of 2026-10-22 | L3 | Quote the range with its input |
| 8 | **Date of the 10.7 GiB ceiling.** The number was set on 2026-10-23 (D26); its reading (process-only, measured at start) was ruled on 2026-10-24 (D28) | L5 | Both dates are right for different parts |
| 9 | **Names in the repository.** The approved copyright line occurs in `README.md` (already there), in two of the new spec files that quote the master specification verbatim (`docs/spec/PROPRIETARY_RUNTIME_SPEC.md`, `docs/spec/RESEARCH_UPDATE_AUDIT.md`) and in two archived files; the repository owner's account name occurs only in `README.md` and the blueprint, which already had it; the new documents (this file, the scaffold, `support/README.md`) carry neither | the scan recorded in `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, section 9.3 | None |
| 10 | **Not searched or copied:** `C:\Ai-Prposed` (trainer C++/CUDA proposals) and `D:\AI-Old\New folder\Stuart-Creator` (an older copy of the project folder) are connected folders but outside the request, which named `C:\AI` | the folder list of the session | Say so if they should be searched too |
| 11 | **Reply text in pushed raw logs.** 94 raw log files already in the repository at `34f744f` (231 lines, all under `benchmarks/raw/`) carry the first words of the model's reply after `answer starts:`. The archive copies of the same kind of log have that text replaced by `[reply text removed]` | `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, section 7.3 (N9) and 9.3 | Left as pushed. Redacting them is the author's decision; it would change the hashes of those files |

---

## 7. Superseded figures: kept as history, never quoted as data

| Figure | Why it is not data | Replaced by |
|---|---|---|
| 57.0 tok/s on Windows-native | No such run exists (L1) | 47.3 and 46.8 short; 53.7 and 54.1 at 12k |
| 10.9 GiB VRAM ceiling | Superseded 2026-10-23 (L5) | 10.7 GiB (reading ruled 2026-10-24) |
| 11.2 GiB as "the ceiling" | It is the whole-card figure of the P1 and G3 gates; unreachable as a free-memory rule on this card | L5 |
| 92 tok/s as a target or result | A model output from three measured and three assumed inputs | L3 |
| The E0 split labelled MEASURED | It is DERIVED from timer-on medians | L3 |
| Hit rates 0.762 to 0.865 as live rates | Simulated, byte-weighted, held out | Live 0.7753 to 0.7793 at 7.0 GiB (L4) |
| The 6.1 GiB bank figures as the current bank | The first set only (L6) | The sweep and O1 |
| "Four patches carried" as a description of the Windows tree | The tree carries the five CUDA-part patches (L9) | Section 4 |
| `-ub 4096` as the micro-batch of the 10.40 to 10.44 GiB runs | The files do not record it (reconciliation D2) | "The default micro-batch, unrecorded" |
| Any speed from a non-llama server | None exists | The first run of the ground-up server |

---

## 8. What the records show, and what they do not

**The author's three questions of 2026-10-24, 08:43, answered from the records.**

| Question | Answer from the records |
|---|---|
| What else must be built from the ground up? | Beyond the kernels and the decode loop, decision D37 adds the tokenizer with Unicode and NFC data, the chat-template renderer, the GGUF reader and writer, SHA-256, the JSON reader and writer, and the block quantizers. The records add the items D37 does not name but the server needs: raw sockets and an HTTP and SSE layer, the sampler, tool-call parsing, reasoning handling, the prompt and state cache, the request log. All are specified as modules in `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` (6.4) |
| Is every parameter mapped, so the server can be built? | **No.** The launcher parameters the clients use today are mapped. The catalogue of everything else llama-server does has never been written; only rough counts exist (361 option definitions, 36 distinct POST route strings, about 86 request-field names at commit 2ca15f540). What the clients actually send is unmeasured |
| Will the trainer take longer? | Yes, by items that are not yet measured: the tokenizer, the renderer, the reader and writer, SHA-256, the JSON code and the quantizers join the trainer's own schedule. No estimate exists and none is given |

**Whether the author's inference holds.** The ceiling was not reached; the records neither confirm nor refute that the cause is llama.cpp. All recorded runs were on the patched build, so the cost of llama.cpp's scheduler, graph and copy path is not separated from the cost of the tier patch, the platform (Windows-native decoded 25% and 28% faster than WSL2, two launches against one), the unused MTP block (a possible factor, UNVERIFIED) or the CPU path. The experiment that separates them is the first run of a server that is not llama.cpp, with the E0 timers on.

**Why the records read as they do (a pattern, not a verdict).**

1. The patch set on llama.cpp was chosen as the fastest way to measure the tier idea. It was never ruled a replacement for the ground-up engine.
2. The measuring work produced a great deal that is reusable: the hardware floors, the ceiling model, the hot-set method, the handoff figures, the gates. It produced no specification of what a ground-up server must do, because the work was done through llama.cpp and its behaviour was inherited, not studied.
3. The ground-up decision was made on 2026-10-24 and is written into the trainer master (D37) and into the blueprint (`[sync R2]`). The engine is held; the trainer is not.

**What the records cannot answer.**

| Question | Why not | How it would be answered |
|---|---|---|
| How much of the gap to the ceiling is llama.cpp | No non-llama run exists | The first run of the proprietary engine with the E0 timers |
| What the clients need from the server | Not captured | Request captures and the catalogue |
| How long the ground-up pieces take | Unmeasured | Measured as each is built |
| Whether the 41st CPU split is the unused MTP block | UNVERIFIED | Compare with the block off and on |
| Whether the first tier blueprint's cost estimate ("months") still holds | An ESTIMATE written before the trainer, the probes and the rulings of 2026-10-24 | Not estimated here |

---

## 9. Sources

`docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`; `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`; `docs/spec/` (all files); `benchmarks/RESULTS.md` and `benchmarks/raw/`; `patches/stage2-tier-cuda/`; `support/MANIFEST.tsv`, `support/README.md`, `support/tier_build/`, `support/archive/c_ai/`. The history in sections 1 and 2 is taken word for word from Part 8 of the author's local master document, with the persona and dataset details left out.
