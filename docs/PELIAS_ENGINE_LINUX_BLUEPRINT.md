# Pelias-Engine: build blueprint, synchronized (WSL2 + Windows-native draft; native Linux held) [sync R1]

2026-10-22 (original blueprint, engine commit 34f744f); synchronized on 2026-10-24 and consolidated with the project's working folders on 2026-10-25

Pelias-Engine is built in 14 gated phases, E0 to E13. The first line of engine code is written in E2, after the ceiling inputs are measured (E0) and the platform floors are known (E1: WSL2 and Windows-native; native Linux is held) [sync R1]. This is a plan only; no code is written yet. **Synchronization note (2026-10-24):** every mark in square brackets below is a queued edit applied in place. `[sync 1]` to `[sync 10]` are the ten mandatory edits (research update 5.1 = trainer master 16.7 = trainer notes 3.4); `[sync R1]` to `[sync R4]` apply author rulings of 2026-10-24 (R1 draft built for WSL2 and Windows-native, native Linux held; R2 llama.cpp is a read-only test oracle and never part of the runtime; R3 the 10.7 GiB ceiling; R4 the 07:39 decisions that touch the engine). The engine design and engine code stay HELD (author, 2026-10-24 06:07). **Consolidation note (2026-10-25) [sync 11]:** the same edits, which until then existed only in the author's local master document, are now applied to this file; the supporting records were added under `docs/spec/`, `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, `docs/PELIAS_ENGINE_LINEAGE.md` and `support/`. These are records only: no engine source code is written or changed, and `benchmarks/` and `patches/` are untouched.

## Document map (2026-10-25) [sync 11]

| File | What it holds |
|---|---|
| `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` (this file) | The engine plan: goal, numbers, components, phases E0 to E13, gates, risks, decisions, unknowns, sources |
| `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md` | The verified file inventory, the scaffold target tree, the phase and gate map, the replacement map, the verification results and the open gaps |
| `docs/PELIAS_ENGINE_LINEAGE.md` | How the project reached its current numbers and plan: old figure, new figure, what changed it, and the old material that was not listed before |
| `docs/spec/` | The proprietary runtime and server specification, open decisions, probe registry, research-update audit, Wave 0 results and ceiling model, trainer contract, synchronization register |
| `benchmarks/` | `RESULTS.md` (claim-to-log ledger) and `raw/` (redacted logs); unchanged |
| `patches/` | The stage-2 loose patches on llama.cpp 2ca15f540; unchanged |
| `support/` | `tier_build/` (the current Windows-native tier build as a patch and two new sources), `archive/` (redacted copies of the working folders), `MANIFEST.tsv` (source path, size and hash of every file considered), `README.md` |

## Goal and scope

The goal is decode speed near the ceiling model's 92 tok/s [DERIVED, model of 2026-10-20; three inputs still assumed], on one 12 GB card plus system RAM. The stretch goal is an engine others can use on other MoE models and GPUs.

- **Today:** the llama.cpp-based tier decodes at 44 to 54 tok/s on short prompts and 43 to 49 tok/s at 12k tokens, over all launches and load modes [MEASURED, WSL2]. Gate G3 asks for at least 55.0 tok/s at 12k. The six recorded runs that anchor the gap analysis are tabulated in `benchmarks/RESULTS.md` (sections 8.1, 9.2 and 9.4) and `docs/spec/WAVE0_AND_CEILING.md` (WSL2 Cell A 37.9 short / 42.1 at 12k; Windows-native Cell A 47.3 and 46.8 short, 53.7 and 54.1 at 12k; WSL2 best cell 53.5 / 53.0 / 49.0 at short / 4k / 12k) [sync R1]. All of these runs were made on the patched llama.cpp tier build: no run of a non-llama server exists yet, so the cause of the gap to the ceiling is not isolated [UNVERIFIED].
- **First model:** Qwen3.6-35B-A3B in Q4_K_M. It has 40 MoE layers, 256 experts per layer and 8 used per token [MEASURED, trace summary].
- **Reference machine:** RTX 4070 (12,281 MB), Ryzen 7 5800X3D (8 cores, 16 threads, AVX2, no AVX-512), 4 x 16 GB DDR4-3200, PCIe 4.0 x16, CUDA 13.0.88 on WSL2 and 13.4.92 on Windows-native [MEASURED, HWiNFO report and nvcc].
- **Target system [sync R1]:** WSL2 and Windows-native, built as two versions of one code base. Author ruling of 2026-10-24 06:21 [AUTHOR-STATED]: "the draft is built as a WSL version AND a Windows version (decision 4 is answered for the draft; Linux-native is held and will be a separate entity later)". The ruling was made for the trainer draft; this synchronization applies it to the engine, and the author may correct that. Native Linux is tabled.
- **Not in scope [sync 1] [sync R1]:** inference-time changes to the router. Clamping or re-ranking the router's choices on an untrained model is rejected: +0.8% to +2.0% perplexity at about 30% fewer misses, and +4.0% to +5.8% at 50% (arXiv 2608.18261 section 7.1 Table 6 [PUBLISHED, full text verified 2026-10-23]). **Trainer-side router locality (Mode L) is not out of scope**: it is an input to the engine, accepted as a gated experiment, because the same study's section 7.1 stacking shows that training cuts the rerouting perplexity cost from +7.7% to +0.9%, and ReMoE (arXiv 2605.27081) reports +26% relative adjacent-token reuse (27.3% to 34.5%) with frozen experts [PUBLISHED, verified 2026-10-23]. The benefit on this model is UNVERIFIED (probes P-L1, P-L2, P-E2, P-E3). The earlier wording, "every setting broke its 1% perplexity gate", was a summary of the inference-time result only. Also not in scope: training itself, which belongs to pelias-trainer. A Windows-native build is no longer out of scope (see Target system).
- **Strictly proprietary runtime [sync R2]:** no llama.cpp, ggml, libllama, common or JSON-library code is linked, copied or required at run time (trainer master D37; author ruling 2026-10-24 08:15, reason AUTHOR-STATED: routing through llama brought in behaviour that was not accounted for). The pinned commit 2ca15f540 (b11399), built unchanged, is an **external, read-only test oracle** that lives only under `tests/oracle/`: it supplies logits, KL, top-1 and kernel-parity references and nothing else. Where the engine's bytes must equal ggml's (block layouts, quantizer arithmetic), the layout and arithmetic are re-expressed in original code after reading the reference, and the file's provenance line says so. The one third-party library in the draft is Dear ImGui (MIT), used for the interface only, and not in the engine.
- **Working rules:** one step at a time; CPU-side logic is tested in the sandbox; GPU runs are done by the author; no file is created without being asked.

## Where the numbers stand

The ceiling is a sum of inputs, and three of them are still assumed. Closing those is phase E0. All measurements below were taken on WSL2.

| Input | Value | Status |
| --- | --- | --- |
| Handoff, sync per layer | 94 µs | measured |
| Handoff, resident-kernel ping-pong | 13.3 µs median, 18.7 µs p99 | measured |
| CPU read of cold experts [sync 3] | 43.8 to 45.9 GB/s with the CPU alone (microbench 02, 2026-10-20). 26.6 GB/s (26.56) is the CPU's share while the GPU reads at the same time (21.06 GB/s over PCIe; 47.62 GB/s combined, microbench 04) | measured |
| GPU zero-copy read, checksum kernel | 26.2 GB/s | measured |
| GPU zero-copy read, compute kernel | 9.59 GB/s | measured; 16-byte-load fix untested |
| PCIe alone | 26.1 GB/s | measured |
| CPU + zero-copy together | 47.6 GB/s | measured |
| GPU spine per token | 7.73 ms | assumed |
| Hit rate of the 7.5 GiB bank | 0.83 held-out | simulated; live rate not measured |
| CPU and GPU overlap | perfect | assumed |
| Resulting ceiling | about 92 tok/s; 88.2 with the 9.59 GB/s kernel | derived |
| Live speed, llama.cpp tier | 44 to 54 tok/s short, 43 to 49 at 12k | measured |

The gap between about 92 and the live figures is what E0 must explain before design choices are locked. Status after E0 (2026-10-22, see Sources): the timeline shows no CPU and GPU overlap and a timer-free handoff of at least 4.6 to 5.0 ms per token; the 7.4 against 7.73 ms spine question stays open; the breakdown is DERIVED, not MEASURED (reconciliation D4).

## Components

```text
 requests ─┐
 weights  ─┤
 memory plan ─┼─►  DECODE LOOP  (40 MoE layers, one pass per token)
 hot-set profile ─┘        │
                           ▼
                      GPU spine  (attention / DeltaNet, norms, router)
                           │
                  router splits the 8 chosen experts
                    ┌──────┴───────┐
                    ▼              ▼
              HOT branch (GPU)   COLD branch (CPU)      run in parallel,
              bank experts       AVX2 Q4_K/Q6_K         handoff flag between them
                    └──────┬───────┘
                           ▼
                         MERGE ──► next layer

 Oracle tests (llama.cpp 2ca15f540: external, read-only, tests/oracle/ only) sit outside the engine and check every part. [sync R2]
```

The top row feeds the decode loop: requests, weights, the memory plan and the hot-set profile. For each of the 40 MoE layers the loop runs the GPU spine, the router splits the 8 chosen experts into hot and cold, the two branches run in parallel and exchange a handoff flag, and the merge feeds the next layer. The oracle tests (llama.cpp 2ca15f540) sit outside the engine and check every part; nothing from llama.cpp is linked into, started by, or required at run time by the engine process [sync R2].

## Proposed scaffolding and file names

These names are a proposal. Nothing is created yet. The letter after each entry is the phase that first creates it (E2 creates the whole empty tree and its build files; later phases fill the files in).

```text
pelias-engine/
  README.md  LICENSE (Apache-2.0)  CMakeLists.txt  CMakePresets.json  .clang-format
  cmake/
    pelias_options.cmake        build options: CUDA archs (89 first), CPU-only test build   E2
    probe_flags.cmake           feature checks (AVX2, CUDA version, WSL build flag, Windows build flag) [sync R1]         E2
  docs/
    BLUEPRINT.md  RUNBOOK.md  decisions/  floors/ (WSL2 and Windows-native tables; native Linux held) [sync R1]           E0
  include/pelias/
    status.h  log.h  trace.h  timer.h                    errors, logging, per-token timers   E2
    model/   gguf.h  tensor_map.h  expert_table.h  arch.h      reader and plugin interface   E3
    mem/     host_alloc.h  vram_plan.h  probe.h                 memory strategy and probe     E2/E11
    cpu/     expert.h  threadpool.h                              cold-expert path              E4
    gpu/     bank.h  zerocopy.cuh  handoff.cuh  expert.cuh  spine.cuh   hot path             E5-E7
    engine/  layer.h  decode_loop.h  prefill.h  kvcache.h  state.h  sampler.h              E6-E9
    server/  http.h  tokenizer.h  chat_template.h  tools.h  api_openai.h                  E10
  src/                      same folders as include/, one .cpp or .cu per header
    cpu/expert_q4k_avx2.cpp  expert_q6k_avx2.cpp  threadpool.cpp                        E4
    gpu/expert_q4k.cu  expert_q6k.cu  bank.cu  zerocopy.cu  handoff.cu                   E5/E6
    gpu/spine/attention.cu  deltanet.cu  norm.cu  router.cu  lm_head.cu                 E7
    mem/host_alloc_wsl.cpp  host_alloc_win32.cpp  probe.cpp  vram_plan.cpp  (host_alloc_linux.cpp is held with native Linux) [sync R1]              E2/E11
    model/arch_moe.cpp                                                                E3
    main.cpp                    command line: serve, bench, trace, probe                  E8
  tools/                        offline, Python
    collect_trace.py  export_hot_set.py  kl_gate.py  speed.py  repack_gguf.py (later)   E3 (reused)
  bench/                        standalone measurement programs
    ram_bw.cpp  handoff.cu  dual_bus.cu  pin_modes.cu  overlap.cu  wide_load.cu         E0/E1
  tests/
    unit/    CPU-only: partition, slot table, quantisation                              E2+
    parity/  kernel and layer parity against the oracle                                 E4-E6
    oracle/  llama.cpp 2ca15f540 harness, fixed prompts, checksums: external, read-only, never linked or built into the engine [sync R2]   E3
  profiles/                     hot-set files (PELIAS_HOT_SET format) and launcher profiles  E5/E10
  scripts/                      run_floors.sh  run_base_cell.sh  run_sweep.sh               E1
```

**Proprietary-server additions to this tree (operating-system socket layer, memory-mapped GGUF reader, native tokenizer and chat-template renderer, arena, doorbell coordinator) are specified in `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` [sync R2]; the tree above stays the engine-phase file map.** Existing work that moves in: microbench programs 02 to 07 become `bench/`; `export_hot_set.py`, the trace collector and `kl_gate.py` become `tools/`; the P1 scripts become `scripts/`. The patches and benchmark results already in the GitHub repo stay where they are as the reference record. [sync R2] The patches are patches on llama.cpp: they are a record and a reference for the patched oracle build, and no patched llama.cpp code enters the engine's source tree. The replacement map in `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md` (section 5) maps each one to the module that replaces what it did.

## Order of operations

Each phase starts only when the previous gate is met. A failed gate stops that branch and the reason is logged.

| Phase | What gets built or decided | Reuses | Gate to move on |
| --- | --- | --- | --- |
| E0 Close the measurements | Finish the P1 sweep (C to N, J, B, O). Run five ground-up measurements: per-token time breakdown, live hit rate, CPU and GPU overlap, wide-load kernel, cause of the load-mode effect. Get the four author rulings. | p1_tests, microbench 02 to 07 | Spine, hit rate and overlap are measured, and the gap between about 92 tok/s and the live figures is explained. |
| E1 Platform floors [sync R1] | Re-run the floor tests (02, 03, 04, 07) and the base cell on WSL2 and on Windows-native. Native Linux floors (boot into native Linux, test ReBAR) are held with native Linux; E-A6 stays open. | microbench programs; the trainer's Wave 0 hardware probes (master 11.12) measure the same quantities on both platforms | A floor table for WSL2 and Windows-native side by side; probe contents chosen from the differences. |
| E2 Skeleton | Repo layout, CMake, CUDA arch 89 first, CPU-only test build, test harness, trace and timer hooks, Apache-2.0 licence. | pelias-engine repo | Clean build on WSL2 and on Windows-native (MSVC and nvcc 13.4.92 for the Windows tree) [sync R1]; CPU tests pass in the sandbox. |
| E3 Model reader and oracle | The engine's own memory-mapped GGUF reader [sync R2] (`docs/spec/PROPRIETARY_RUNTIME_SPEC.md`), tensor map, expert size table. External oracle harness against pinned llama.cpp 2ca15f540 (b11399): logits, KL, top-1. [sync 4] The reader and the harness also ingest trainer-exported GGUFs (`merged-core.gguf` with `layout_manifest.json`) and evaluate the three KL gates against the file's measured float spread: K1 (trainer against llama.cpp on the same GGUF), K2 (exported GGUF against the trainer reference), K3 (engine against llama.cpp on the exported GGUF). P-Q4a reproduces 0.017353 at batch 16 and 0.018286 at batch 1024 on the current served file; P-Q4b re-measures the spread on any newly trained file. | gguf_min.py, expert_layer_sizes.json, kl_gate.py | Reads the Q4_K_M file; the oracle reproduces the stock noise floor (KL 0.017353 at batch 16 vs 0.018286 at batch 1024 [MEASURED]); a trainer-exported GGUF and its layout manifest parse identically, and K3 holds on it [sync 4]. |
| E4 CPU expert kernels | AVX2 kernels for Q4_K gate/up and Q6_K down; 8-expert path; thread pool pinned to cores. | 02_ram_bw_test.cpp | Throughput at or above llama.cpp's CPU expert path (about 27 GB/s effective [DERIVED]); output matches the oracle. |
| E5 GPU expert kernels and bank | 16-byte-load kernels; bank loader and slot table from the hot-set file; zero-copy read kernel for cold experts. [sync 5] The slot table supports variable per-layer capacities (84 to 161 experts per layer in the 7.5 GiB water-fill) read from the `PELIAS_HOT_SET` file. `wide_load.cu` (P-E10) must handle Q6_K's 210-byte blocks (420-byte rows) by stream-aligned 16-byte (`uint4`) shared-memory staging, or consume 432-byte row-padded packs from the engine's `repack_gguf.py` if P-E10 shows padding is required. | PELIAS_HOT_SET format, 03, 04, 07 | Zero-copy compute kernel reaches the checksum rate (26.2 GB/s target, from 9.59); bank experts match the oracle. |
| E6 One layer end to end | Router, hot/cold partition, GPU hot branch and CPU cold branch in parallel, merge. Handoff by resident-kernel flag, stream sync as fallback. | tier dual-branch design, 03 | Layer latency within the E0 model; output within the oracle noise floor; handoff p99 no worse than 18.7 µs on native Linux. |
| E7 Full spine | Attention, DeltaNet layers, norms, router, lm_head, q8_0 KV cache; all 40 layers; first full token path. | kernel behaviour read from llama.cpp 2ca15f540 as a specification only (read-only; the engine's kernels are its own code) [sync R2] | Full-model KL vs stock no worse than the stock noise floor, until the H6 ruling sets the limit. |
| E8 Decode loop | CUDA graphs for the static spine, layer pipelining, CPU/GPU overlap, sampler, hit counters and timers. | | First at least the llama.cpp tier result, then G3 (55.0 tok/s or more at 12k), then the ceiling table from E0 and E1. |
| E9 Prefill and long context | Chunked prefill, large-batch path, bypass rule, contexts up to 131,072, VRAM planner. | | Prompt reading at least 1,670 tok/s at 12k (llama.cpp tier) and aimed at 2,238 (stock with load mode none) [MEASURED]; peak VRAM within the author's ceiling. |
| E10 Server | Tokenizer, chat template, tool-call parsing, OpenAI-compatible streaming, prefix cache for recurrent state: all the engine's own code [sync R2] (native byte-level BPE tokenizer, chat-template subset renderer, raw-socket HTTP and SSE server; specified in `docs/spec/PROPRIETARY_RUNTIME_SPEC.md`). | current launcher flags | Drop-in for the launcher; persona probe not-pass 12 of 111 or fewer (baseline 9), needles 12/12 to 120k, a Visual Studio Copilot session works. |
| E11 Capability probe | Detect WSL2 or Windows-native, test pinned-memory limits and bandwidth, read BAR size, choose the memory strategy. [sync 6] Also log the WSL configuration line as it is (`transparent_hugepage=always swiotlb=65536`) and `/proc/meminfo` `AnonHugePages` during model load (P-E11), to test the true cause of the 2x load-mode prefill swing (863 to 929 against 1,667 to 1,680 tok/s at 12k). | NVIDIA WSL guide | The same code base passes the E8 to E10 gates on WSL2 and on Windows-native [sync R1]. |
| E12 Gated extras | Zero-copy third branch, prefetch and small dynamic region, MTP, precision tiers, ReBAR path, LRU bank. Each only if its own test says go. [sync 7] Hot-set objective: byte-weighted hit maximization (h) against zero-miss layer share (Z_l) maximization, decided by P-E2's measurement of the fixed miss cost c0 against the marginal miss cost c1. Trainer Mode L candidates (re-profiled hot sets) enter the extras list, each judged by its own gates G-L4 and G-Z4 (measured decode against the unchanged baseline C0). | blueprint phase 7 | Zero-copy: GPU rate at least 15 GB/s and combined at least 1.25x CPU alone. LRU: only if the live hit rate is well below 0.83. |
| E13 Hardening and generality | Model-plugin interface for other MoEs, other GPUs, packaging, regression suite, docs. | | A second MoE model runs through the plugin interface; regression suite green. |

After E13 the lessons go back into the pelias-trainer. [sync 9] This is superseded by the two-way contract in `docs/spec/TRAINER_CONTRACT.md`: each engine milestone releases a named input to the trainer now, not at the end (trainer master 16.3 X12).

## Verification ladder

Every phase is checked against the same oracle: llama.cpp commit 2ca15f540 (b11399), built unchanged. It is an external, read-only test oracle under `tests/oracle/`; the engine never links or starts it [sync R2]. Each rung must pass before the next is run.

1. **Unit tests, CPU only (sandbox):** partition completeness, slot-table invariants, quantisation parity.
2. **Kernel parity:** each CPU and GPU kernel against a reference, in the style of `test-backend-ops`.
3. **Layer parity:** one MoE layer against the oracle.
4. **Model accuracy:** the existing KL gate, 16 chunks of 1,024 tokens, with mean KL and top-1 agreement, run as K1, K2 and K3 when the file is trainer-exported [sync 4]. Stock noise floor: KL 0.017353 (batch 16) and 0.018286 (batch 1024) [MEASURED]. The pass limit waits for the H6 ruling.
5. **Speed:** the speed tool at depths short, 4,000 and 12,000, 3 repeats and 2 warm-ups, against a stock run made the same day. Launch-to-launch spread has run 5% to 21% [MEASURED], so results carry the spread and repeat launches.
6. **Capability:** persona probe, needles, tools, recall, and real Visual Studio use.

Every run records its label, command, commit hash and input checksums.

## WSL2 and Windows-native first; native Linux held [sync R1]

One code base builds two versions, a WSL2 build and a Windows-native build [sync R1]. What differs is how the GPU and memory are reached, and the operating-system calls (POSIX `sys/socket.h` and `mmap` against Winsock2 and `CreateFileMapping`/`MapViewOfFile`) which live in one thin layer (`docs/spec/PROPRIETARY_RUNTIME_SPEC.md`). Native Linux is held and will be a separate entity later (author, 2026-10-24 06:21).

- **Startup probe (E11):** detects WSL2, tests pinned-copy bandwidth and limits, reads the BAR size, and picks the host-memory and transfer strategy. This matters: on WSL2 the load mode changed prompt speed by about 2x (863 to 929 vs 1,667 to 1,680 tok/s at 12k) [MEASURED].
- **Avoid what WSL2 lacks:** NVIDIA documents that WSL2 has no full managed (unified) memory support, limits pinned system memory, supports only some NVML queries and has limited profilers ([CUDA on WSL User Guide](https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html)). The engine uses explicit allocations and copies, treats pinned memory as capped with a fallback, does its own timing with CUDA events, and keeps OS-specific calls in a thin allocation and threading layer.
- **Native-only paths:** ReBAR and any faster transfer route are optional and switched on only when the probe finds them. The HWiNFO report shows Resizable BAR on at the host; nvidia-smi inside WSL2 reports BAR1 total 16,384 MiB with 0 MiB used (one sample taken mid-run), while the guest shows no NVIDIA PCI device [MEASURED].
- **Windows-native [sync R1]:** the recorded Windows-native decode of the tier build is 25% (short) and 28% (12k) faster than the WSL cell, a difference of about 5.2 ms per token (two Windows launches against one WSL launch; not yet a claim: three launches per side are needed; ledger section 9.4, trainer master 16.3 X9). The cause is not isolated; P-H11 and E1 test it.
- **What may differ on native Linux, all unmeasured there (held):** handoff latency, the pinned-memory cap, host-memory behaviour and ReBAR. E1 measures them on this machine.

## Risks and kill criteria

| Risk | What stops or changes the plan |
| --- | --- |
| E0 shows the live hit rate far below the simulated 0.83 | Move the bank design toward an LRU or small dynamic region (E12) before E5 is locked. |
| Handoff on Windows-native or native Linux is not lower than on WSL2 (native Linux held) [sync R1] | The ceiling stays near 88 to 92 tok/s [DERIVED]; plan around it instead of assuming gains. |
| Spine kernels (attention, DeltaNet) are a large build | Start with the engine's own slow reference kernels (the sequential fp32 DeltaNet recurrence and the blocked attention kernel, as in the trainer's `kern/` plan) behind a clean interface and replace them one at a time with the fast ones; llama.cpp's kernels are a read-only specification and a test oracle, never a dependency [sync R2]. |
| GPU and CPU rounding differ from the oracle | Set the pass limit from the stock noise floor after the H6 ruling. |
| VRAM headroom: the 8.0 GiB bank peaked at 11.17 GiB of 12 [MEASURED] | The bank size is capped by the author's VRAM ceiling ruling. |
| E8 does not beat the llama.cpp tier result | [sync R2] The original kill criterion, "fold the findings back into a llama.cpp patch series", is removed: the author ruled ground-up on 2026-10-24 (D37). At that gate, log the per-token timers against the E0 breakdown and decide with the author whether to continue. The llama.cpp tier remains the interim measuring build and the reference record. **Open for the author:** whether to restore the original fallback. |
| Scope creep into the server | Keep E10 thin: only what the existing launcher and clients need. |

## Decisions needed from the author

| Decision | Why it blocks | Needed by |
| --- | --- | --- |
| H6 definition: strict limit or float-spread reading (RULED 2026-10-22: float-spread governs; strict KL 0.005 / top-1 99% is reported for information only) | Sets the accuracy pass limit for every phase from E3 | E3 |
| VRAM ceiling: 10.4, 10.9 or 11.2 GiB (RULED 2026-10-22: 10.9 GiB; **SUPERSEDED [sync R3] by 10.7 GiB**, author ruling 2026-10-23 (trainer master D26), after P-H8 measured 10.83 GiB (WSL2) and 10.85 GiB (Windows-native) of CUDA free memory once a context exists; ruled 2026-10-24 (D28) that the number is measured at start: the smaller of CUDA free and NVML free after the context, minus a margin, with 10.7 GiB staying this machine's value until the margin is measured) | Caps the bank size and the E9 planner | E5 |
| Fate of the six loose patches and the base tier patch (RULED 2026-10-22: carry mm_ids_dedup, quantize_signed_id, cuda_hole_fix and pin_cold_to_cpu; drop mmq_dump_diag and mmq_dump_diag2) [sync R2: "carry" means kept as the reference record and as the patched oracle build; no patched llama.cpp code enters the engine] | Decides what is carried into the engine as reference | E2 |
| The 7.4 ms split discrepancy (author ruling 2026-10-22: re-derive from E0 per-token timers; until then the ceiling is shown as a range 88.2 to 90.9 tok/s [DERIVED]) | One of the inputs to the spine time | E0 |
| Hit-rate counter and per-token timer patches (measurement builds, in a copy of the tree) | Needed to measure the assumed inputs | E0 |
| CUDA minimum version and GPU architectures beyond 89 | Fixes the build matrix and the generality goal | E2 |
| Which extras enter E12 first | Order of the gated extras | E12 |
| Thinking on or off in the speed tests | Parked until the main sweep is done | after E0 |
| Repository: new folders in FireballJohnnie/pelias-engine, or a separate repo | Fixes the scaffolding tree and where the existing patches and results live | E2 |
| Is the 7.5 GiB bank a ruling or a derived value? May the bank be a non-multiple of 0.5 GiB? [sync 2] | The core's room is set by it; at the default micro-batch the free room would hold roughly 7.8 GiB if the slope is 1:1 [ESTIMATE, reconciliation D7]; 8.0 GiB does not fit | E5 |
| Which prefill micro-batch does the engine ship with? [sync R4] | RULED 2026-10-24 07:39 (trainer D31, D34): an engine setting; the trainer no longer depends on it. The engine's own peak still moves with it: 0.26 to 0.30 GiB of headroom at the default and 0.01 to 0.03 GiB at `-ub 2048` (whole-card reading, 10.7 GiB ceiling) | E9 |
| Does the engine add Q8_0 weights or kernels? [sync R4] | RULED 2026-10-24 07:39 (trainer D32): not needed; no Q8_0 or Q6_K core inflation by default. Stays an optional Advanced/Developer override, usable only if the engine supports it (P-E5) | E7 |
| Is Windows-native promoted from "not in scope"? [sync R1] | Answered for the draft by the author's ruling of 2026-10-24 06:21 (WSL2 and Windows-native both built); three launches per side are needed before a speed claim | E1 |
| Is Mode L / L-Z worth building after P-L1, P-L2, P-E2 and P-E3 report? [sync 7] | Decides whether any router training is built and whether the extras list in E12 gets Mode L candidates | E12 |
| What is the 10.7 GiB ceiling a ceiling of? [sync R3] | RULED 2026-10-24 for a probe-measured, process-only reading (the whole-card figures stay as the stricter planning figures until P-E4 re-measures); the margin is set by the disturbance test | E5 |
| Does the Windows tier tree keep the two diagnostic patches (`mmq_dump_diag`, `mmq_dump_diag2`) that the ruling of 2026-10-22 dropped? [sync 11] | MEASURED 2026-10-25: the current Windows tier tree (`support/tier_build/`) carries the CUDA changes of all five stage-2 patches line for line, the two diagnostics included (inert unless `PELIAS_MMID_DUMP=1`). A rebuild without them is a three-patch tree; no speed effect is recorded either way | E1 |
| Does "ground up" apply to the engine, and does the kill criterion at E8 (fallback to a llama.cpp patch series) stay? [sync R2] | The product ruling (D37) is ground-up; the engine is HELD and the author has not yet confirmed the engine part | E2 |

## Unknowns and how each gets measured

| Unknown | Why it matters | Measurement pathway |
| --- | --- | --- |
| Live hit rate | Replaces the assumed 80% in the ceiling | A counter in a measurement-only build of the tier, in a copy of the tree |
| GPU spine time (assumed 7.73 ms) | Largest fixed term per token | Per-token timeline from timers inside the tier (CUDA events) |
| CPU and GPU overlap | The ceiling assumes it is perfect | Standalone program running cold-expert math on the CPU while the GPU runs hot-expert math on real Q4_K shapes |
| Wide-load GPU kernel | Cold-read rate 9.59 vs 26.2 GB/s | A 16-byte-load variant of the kernel, same test |
| Native Linux handoff, pinned cap, ReBAR | Decide the probe and may move the ceiling | E1: same floor tests on native Linux |
| Cause of the load-mode effect | Guides host-memory design | Read AnonHugePages in /proc/meminfo while each mode serves |
| Cost of long context | The model has no context term | Speed at depths up to 128k once E8 runs |
| CPU expert kernel limit | Bounds the cold path | E4 standalone benchmark |
| Thinking on or off | May change results and run time a lot | Revisit after the main sweep |
| E-A1 [sync 8]: thread count and pinning, 8 physical cores (one thread each) against 16 SMT threads | The ledger shows 5 to 7 threads beating 10 on stock; Zen 3 splits the load queue between SMT threads (audited hypothesis from the outside report, UNVERIFIED) | A/B in E4, with its own go/no-go |
| E-A2: acquire/release against seq_cst on the handoff flags | Atomic ordering cost on the doorbell (audited hypothesis, UNVERIFIED) | A/B in E6 |
| E-A3: `PREFETCHNTA` on cold reads | Possible cold-read gain on the CPU path (audited hypothesis, UNVERIFIED) | A/B in E4 |
| E-A4: L2 persisting window (`cudaAccessPropertyPersisting`) | Possible gain for the spine or hot tensors (audited hypothesis, UNVERIFIED) | Query `cudaDevAttrMaxPersistingL2CacheSize` first, then A/B in E5 to E8 |
| E-A5: `AnonHugePages` per load mode | The engine's open load-mode unknown (the same item as the row above, carried here under its A/B name) | E11, P-E11 |
| E-A6: native Linux floors | The engine's E1 as first written; held with native Linux [sync R1] | Held |
| E-A7: pinned memory above 20.4 GiB | The engine had proven 20,448 MiB, the largest size it tried. The trainer's P-H7 held 32 GiB of `cudaHostAlloc` memory on WSL2 and on Windows-native (MEASURED 2026-10-23/24, trainer master 16.3 X13), so the 20.4 GiB figure was not a limit on this machine; the limit above 32 GiB is not found | Engine-side re-check in E11 |
| P-E4 at `-ub 4096` [sync 2] | The 7.5 GiB bank was never measured at `-ub 4096`: the stamps 130854 and 132648 ran at the default micro-batch, whose value the files do not record (ledger 8.2; reconciliation D2). By the 1:1 slope seen at `-ub 2048` it would peak near 11.2 GiB [ESTIMATE], over the ceiling | P-E4 adds this case |
| P-H13 on Windows-native [reconciliation 7.5] | Display-driver timeout on long kernels or graphs; the card also drives the display | Trainer's P-H13 (done on both platforms, master 11.12); engine re-check at E6 |
| Behaviour llama.cpp contributed that the ground-up engine must account for [sync R2] | The author's reason for ground-up: routing through llama brought in behaviour that was not accounted for. One candidate: the unused MTP block `blk.40` that stock llama-server skips with an "unused tensor" warning at each start; whether it explains why the VRAM ceiling was not reached is UNVERIFIED | Compare VRAM and speed with MTP off and on; the per-token timers of E0; the llama-server catalogue (`docs/spec/PROPRIETARY_RUNTIME_SPEC.md`, section 6.14) |

## Sources

- Project documents, read this session: `pelias_tier_blueprint.md` (2026-10-20), `RUNBOOK.md` (hardware floors of 2026-10-20; results of 2026-10-22), `p1_tests/results/` (K and A results, 2026-10-22), and the HWiNFO report of 2026-10-20.
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html), fetched 2026-10-22: WSL2 limits on managed memory, pinned memory, NVML and profilers.
- "Cacheable by Design?", arXiv 2608.18261: routing study and the negative result on forcing router locality. Cited from the tier blueprint; not re-fetched this session.
- Numbers from the running P1 sweep (C to N, J, B, O) are not in this document yet; the E0 results replace the assumed inputs when they arrive.
- [sync 10] Added by the synchronization of 2026-10-24: `benchmarks/RESULTS.md` sections 1 to 9.4; `benchmarks/raw/e0-2026-10-22/`; `benchmarks/raw/windows-native-2026-10-23/`; the full-text verification of arXiv 2608.18261 and arXiv 2605.27081 (2026-10-23); and `pelias_engine_hardware_verified_research_plan.md` (2026-10-23), marked as an audited hypothesis list and not as a primary measurement source; the trainer master (16.3 to 16.7 and 11.12) and notes (3.4); the reconciliation of the research update (D1 to D11, A1 to A8, N1 to N5; `docs/spec/RECONCILIATION.md`); the author's rulings of 2026-10-24 (R1 to R4).
- E0 update 2026-10-22 (tier build, WSL2): live hit rate 0.775 to 0.779 for the 7.0 GiB bank against 0.806 simulated, not far below, so the LRU trigger in the risk table is not met; the per-token timeline shows no CPU and GPU overlap and a timer-free handoff of at least 4.6 to 5.0 ms per token (model: 0.25 ms). The 7.4 against 7.73 ms spine question stays open. Details: `benchmarks/RESULTS.md` section 9 and `benchmarks/raw/e0-2026-10-22/`.
- [sync 11] Added by the consolidation of 2026-10-25 (`docs/PELIAS_ENGINE_LINEAGE.md` has the detail): (1) the base tier patch `pelias_tier_v1.patch` and the sources `llama-pelias-tier.cpp/.h` of 2026-10-20, which `benchmarks/RESULTS.md` section 8 said were not in the repository, are in `support/archive/c_ai/creator/moe_track/patches/`; the patch applies cleanly to llama.cpp 2ca15f540 [MEASURED, `git apply --check`]. (2) The current Windows-native tier tree is stored in `support/tier_build/` as one patch on 2ca15f540 (10 files; applies cleanly) and two new sources; its CUDA part equals the five stage-2 patches applied in the order `cuda_hole_fix`, `mm_ids_dedup`, `mmq_dump_diag`, `mmq_dump_diag2`, `quantize_signed_id` [MEASURED, changed lines compared file by file], and its other files equal `pelias_tier_v1.patch` apart from `src/llama-graph.cpp`, where 7 lines are rewritten and 2 added to pin the cold branch to the CPU (the change of `pin_cold_to_cpu.patch`; `src/llama-graph.cpp` equals the v1 result with that patch applied, and the patch's base file `llama-graph.cpp.before_pin` is archived in `support/archive/c_ai/backup/2026-10-21/finish3_pin_fix/`) [MEASURED]. The tree's `llama-pelias-tier.cpp` equals the v1 source plus `support/archive/c_ai/creator/moe_track/e0/win_fseek.patch` (64-bit file seek for Windows), saved with Windows line endings, and its `llama-pelias-tier.h` equals the v1 header [MEASURED, byte comparison]. (3) The Wave 0 hardware probe program (sources, run scripts) and its results for WSL2 and Windows-native (2026-10-23 and 2026-10-24) are in `support/archive/c_ai/pelias_probes_win/` and `support/archive/c_ai/backup/2026-10-23/probe_results/`. (4) The original raw logs of the 2026-10-21 crash sweeps (before the fix: prompts of 1 to 12 tokens pass and 13 to 34 fail at bypass 32) are in `support/archive/c_ai/backup/2026-10-21/p1_logs/`. (5) The microbenchmarks 01 to 07, the E0 patches and analysis scripts, the router-trace and simulation tools and the 14-file CUDA prototype package are in `support/archive/c_ai/creator/moe_track/`; the replacement map in `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md` says which module replaces each.
