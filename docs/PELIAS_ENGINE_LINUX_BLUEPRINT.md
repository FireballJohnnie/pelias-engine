# Pelias-Engine (Linux): build blueprint

2026-10-22

Pelias-Engine is built in 14 gated phases, E0 to E13. The first line of engine code is written in E2, after the ceiling inputs are measured (E0) and the native-Linux floors are known (E1). This is a plan only; no code is written yet.

## Goal and scope

The goal is decode speed near the ceiling model's 92 tok/s [DERIVED, model of 2026-10-20; three inputs still assumed], on one 12 GB card plus system RAM. The stretch goal is an engine others can use on other MoE models and GPUs.

- **Today:** the llama.cpp-based tier decodes at 44 to 54 tok/s on short prompts and 43 to 49 tok/s at 12k tokens, over all launches and load modes [MEASURED, WSL2]. Gate G3 asks for at least 55.0 tok/s at 12k.
- **First model:** Qwen3.6-35B-A3B in Q4_K_M. It has 40 MoE layers, 256 experts per layer and 8 used per token [MEASURED, trace summary].
- **Reference machine:** RTX 4070 (12,281 MB), Ryzen 7 5800X3D (8 cores, 16 threads, AVX2, no AVX-512), 4 x 16 GB DDR4-3200, PCIe 4.0 x16, CUDA 13.0.88 [MEASURED, HWiNFO report and nvcc].
- **Target system:** native Linux first. The same binary also runs on WSL2 (see the compatibility section).
- **Not in scope:** changing the router (a published study found every setting broke its 1% perplexity gate [PUBLISHED, arXiv 2608.18261, via the tier blueprint; not re-fetched]), training, and a Windows-native build.
- **Working rules:** one step at a time; CPU-side logic is tested in the sandbox; GPU runs are done by the author; no file is created without being asked.

## Where the numbers stand

The ceiling is a sum of inputs, and three of them are still assumed. Closing those is phase E0. All measurements below were taken on WSL2.

| Input | Value | Status |
| --- | --- | --- |
| Handoff, sync per layer | 94 µs | measured |
| Handoff, resident-kernel ping-pong | 13.3 µs median, 18.7 µs p99 | measured |
| CPU read of cold experts | 26.6 GB/s | measured |
| GPU zero-copy read, checksum kernel | 26.2 GB/s | measured |
| GPU zero-copy read, compute kernel | 9.59 GB/s | measured; 16-byte-load fix untested |
| PCIe alone | 26.1 GB/s | measured |
| CPU + zero-copy together | 47.6 GB/s | measured |
| GPU spine per token | 7.73 ms | assumed |
| Hit rate of the 7.5 GiB bank | 0.83 held-out | simulated; live rate not measured |
| CPU and GPU overlap | perfect | assumed |
| Resulting ceiling | about 92 tok/s; 88.2 with the 9.59 GB/s kernel | derived |
| Live speed, llama.cpp tier | 44 to 54 tok/s short, 43 to 49 at 12k | measured |

The gap between about 92 and the live figures is what E0 must explain before design choices are locked.

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

 Oracle tests (llama.cpp 2ca15f540) sit outside the engine and check every part.
```

The top row feeds the decode loop: requests, weights, the memory plan and the hot-set profile. For each of the 40 MoE layers the loop runs the GPU spine, the router splits the 8 chosen experts into hot and cold, the two branches run in parallel and exchange a handoff flag, and the merge feeds the next layer. The oracle tests (llama.cpp 2ca15f540) sit outside the engine and check every part.

## Proposed scaffolding and file names

These names are a proposal. Nothing is created yet. The letter after each entry is the phase that first creates it (E2 creates the whole empty tree and its build files; later phases fill the files in).

```text
pelias-engine/
  README.md  LICENSE (Apache-2.0)  CMakeLists.txt  CMakePresets.json  .clang-format
  cmake/
    pelias_options.cmake        build options: CUDA archs (89 first), CPU-only test build   E2
    probe_flags.cmake           feature checks (AVX2, CUDA version, WSL build flag)         E2
  docs/
    BLUEPRINT.md  RUNBOOK.md  decisions/  floors/ (WSL2 and native Linux tables)           E0
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
    mem/host_alloc_linux.cpp  host_alloc_wsl.cpp  probe.cpp  vram_plan.cpp              E2/E11
    model/arch_moe.cpp                                                                E3
    main.cpp                    command line: serve, bench, trace, probe                  E8
  tools/                        offline, Python
    collect_trace.py  export_hot_set.py  kl_gate.py  speed.py  repack_gguf.py (later)   E3 (reused)
  bench/                        standalone measurement programs
    ram_bw.cpp  handoff.cu  dual_bus.cu  pin_modes.cu  overlap.cu  wide_load.cu         E0/E1
  tests/
    unit/    CPU-only: partition, slot table, quantisation                              E2+
    parity/  kernel and layer parity against the oracle                                 E4-E6
    oracle/  llama.cpp 2ca15f540 harness, fixed prompts, checksums                      E3
  profiles/                     hot-set files (PELIAS_HOT_SET format) and launcher profiles  E5/E10
  scripts/                      run_floors.sh  run_base_cell.sh  run_sweep.sh               E1
```

Existing work that moves in: microbench programs 02 to 07 become `bench/`; `export_hot_set.py`, the trace collector and `kl_gate.py` become `tools/`; the P1 scripts become `scripts/`. The patches and benchmark results already in the GitHub repo stay where they are as the reference record.

## Order of operations

Each phase starts only when the previous gate is met. A failed gate stops that branch and the reason is logged.

| Phase | What gets built or decided | Reuses | Gate to move on |
| --- | --- | --- | --- |
| E0 Close the measurements | Finish the P1 sweep (C to N, J, B, O). Run five ground-up measurements: per-token time breakdown, live hit rate, CPU and GPU overlap, wide-load kernel, cause of the load-mode effect. Get the four author rulings. | p1_tests, microbench 02 to 07 | Spine, hit rate and overlap are measured, and the gap between about 92 tok/s and the live figures is explained. |
| E1 Native Linux floors | Boot this machine into native Linux. Re-run the floor tests (02, 03, 04, 07) and the base cell. Test ReBAR. | microbench programs | A floor table for WSL2 and native Linux side by side; probe contents chosen from the differences. |
| E2 Skeleton | Repo layout, CMake, CUDA arch 89 first, CPU-only test build, test harness, trace and timer hooks, Apache-2.0 licence. | pelias-engine repo | Clean build on native Linux and WSL2; CPU tests pass in the sandbox. |
| E3 Model reader and oracle | GGUF reader, tensor map, expert size table. Oracle harness against pinned llama.cpp 2ca15f540 (b11399): logits, KL, top-1. | gguf_min.py, expert_layer_sizes.json, kl_gate.py | Reads the Q4_K_M file; the oracle reproduces the stock noise floor (KL 0.017353 at batch 16 vs 0.018286 at batch 1024 [MEASURED]). |
| E4 CPU expert kernels | AVX2 kernels for Q4_K gate/up and Q6_K down; 8-expert path; thread pool pinned to cores. | 02_ram_bw_test.cpp | Throughput at or above llama.cpp's CPU expert path (about 27 GB/s effective [DERIVED]); output matches the oracle. |
| E5 GPU expert kernels and bank | 16-byte-load kernels; bank loader and slot table from the hot-set file; zero-copy read kernel for cold experts. | PELIAS_HOT_SET format, 03, 04, 07 | Zero-copy compute kernel reaches the checksum rate (26.2 GB/s target, from 9.59); bank experts match the oracle. |
| E6 One layer end to end | Router, hot/cold partition, GPU hot branch and CPU cold branch in parallel, merge. Handoff by resident-kernel flag, stream sync as fallback. | tier dual-branch design, 03 | Layer latency within the E0 model; output within the oracle noise floor; handoff p99 no worse than 18.7 µs on native Linux. |
| E7 Full spine | Attention, DeltaNet layers, norms, router, lm_head, q8_0 KV cache; all 40 layers; first full token path. | upstream kernels as reference | Full-model KL vs stock no worse than the stock noise floor, until the H6 ruling sets the limit. |
| E8 Decode loop | CUDA graphs for the static spine, layer pipelining, CPU/GPU overlap, sampler, hit counters and timers. | | First at least the llama.cpp tier result, then G3 (55.0 tok/s or more at 12k), then the ceiling table from E0 and E1. |
| E9 Prefill and long context | Chunked prefill, large-batch path, bypass rule, contexts up to 131,072, VRAM planner. | | Prompt reading at least 1,670 tok/s at 12k (llama.cpp tier) and aimed at 2,238 (stock with load mode none) [MEASURED]; peak VRAM within the author's ceiling. |
| E10 Server | Tokenizer, chat template, tool-call parsing, OpenAI-compatible streaming, prefix cache for recurrent state. | current launcher flags | Drop-in for the launcher; persona probe not-pass 12 of 111 or fewer (baseline 9), needles 12/12 to 120k, a Visual Studio Copilot session works. |
| E11 Capability probe | Detect WSL2, test pinned-memory limits and bandwidth, read BAR size, choose the memory strategy. | NVIDIA WSL guide | The same binary passes the E8 to E10 gates on WSL2 and native Linux. |
| E12 Gated extras | Zero-copy third branch, prefetch and small dynamic region, MTP, precision tiers, ReBAR path, LRU bank. Each only if its own test says go. | blueprint phase 7 | Zero-copy: GPU rate at least 15 GB/s and combined at least 1.25x CPU alone. LRU: only if the live hit rate is well below 0.83. |
| E13 Hardening and generality | Model-plugin interface for other MoEs, other GPUs, packaging, regression suite, docs. | | A second MoE model runs through the plugin interface; regression suite green. |

After E13 the lessons go back into the pelias-trainer.

## Verification ladder

Every phase is checked against the same oracle: llama.cpp commit 2ca15f540 (b11399), built unchanged. Each rung must pass before the next is run.

1. **Unit tests, CPU only (sandbox):** partition completeness, slot-table invariants, quantisation parity.
2. **Kernel parity:** each CPU and GPU kernel against a reference, in the style of `test-backend-ops`.
3. **Layer parity:** one MoE layer against the oracle.
4. **Model accuracy:** the existing KL gate, 16 chunks of 1,024 tokens, with mean KL and top-1 agreement. Stock noise floor: KL 0.017353 (batch 16) and 0.018286 (batch 1024) [MEASURED]. The pass limit waits for the H6 ruling.
5. **Speed:** the speed tool at depths short, 4,000 and 12,000, 3 repeats and 2 warm-ups, against a stock run made the same day. Launch-to-launch spread has run 5% to 21% [MEASURED], so results carry the spread and repeat launches.
6. **Capability:** persona probe, needles, tools, recall, and real Visual Studio use.

Every run records its label, command, commit hash and input checksums.

## Linux first, WSL2 compatible

One code base and one binary serve both. WSL2 is Linux, so a Linux engine runs there; what differs is how the GPU and memory are reached.

- **Startup probe (E11):** detects WSL2, tests pinned-copy bandwidth and limits, reads the BAR size, and picks the host-memory and transfer strategy. This matters: on WSL2 the load mode changed prompt speed by about 2x (863 to 929 vs 1,667 to 1,680 tok/s at 12k) [MEASURED].
- **Avoid what WSL2 lacks:** NVIDIA documents that WSL2 has no full managed (unified) memory support, limits pinned system memory, supports only some NVML queries and has limited profilers ([CUDA on WSL User Guide](https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html)). The engine uses explicit allocations and copies, treats pinned memory as capped with a fallback, does its own timing with CUDA events, and keeps OS-specific calls in a thin allocation and threading layer.
- **Native-only paths:** ReBAR and any faster transfer route are optional and switched on only when the probe finds them. The HWiNFO report shows Resizable BAR on at the host; nvidia-smi inside WSL2 reports BAR1 total 16,384 MiB with 0 MiB used (one sample taken mid-run), while the guest shows no NVIDIA PCI device [MEASURED].
- **What may differ on native Linux, all unmeasured there:** handoff latency, the pinned-memory cap, host-memory behaviour and ReBAR. E1 measures them on this machine.

## Risks and kill criteria

| Risk | What stops or changes the plan |
| --- | --- |
| E0 shows the live hit rate far below the simulated 0.83 | Move the bank design toward an LRU or small dynamic region (E12) before E5 is locked. |
| Handoff on native Linux is not lower than on WSL2 | The ceiling stays near 88 to 92 tok/s [DERIVED]; plan around it instead of assuming gains. |
| Spine kernels (attention, DeltaNet) are a large build | Start with upstream kernels as reference behind a clean interface and replace them one at a time. |
| GPU and CPU rounding differ from the oracle | Set the pass limit from the stock noise floor after the H6 ruling. |
| VRAM headroom: the 8.0 GiB bank peaked at 11.17 GiB of 12 [MEASURED] | The bank size is capped by the author's VRAM ceiling ruling. |
| E8 does not beat the llama.cpp tier result | Stop the from-scratch track at that point and fold the findings back into a llama.cpp patch series. |
| Scope creep into the server | Keep E10 thin: only what the existing launcher and clients need. |

## Decisions needed from the author

| Decision | Why it blocks | Needed by |
| --- | --- | --- |
| H6 definition: strict limit or float-spread reading (RULED 2026-10-22: float-spread governs; strict KL 0.005 / top-1 99% is reported for information only) | Sets the accuracy pass limit for every phase from E3 | E3 |
| VRAM ceiling: 10.4, 10.9 or 11.2 GiB (RULED 2026-10-22: 10.9 GiB) | Caps the bank size and the E9 planner | E5 |
| Fate of the six loose patches and the base tier patch (RULED 2026-10-22: carry mm_ids_dedup, quantize_signed_id, cuda_hole_fix and pin_cold_to_cpu; drop mmq_dump_diag and mmq_dump_diag2) | Decides what is carried into the engine as reference | E2 |
| The 7.4 ms split discrepancy (author ruling 2026-10-22: re-derive from E0 per-token timers; until then the ceiling is shown as a range 88.2 to 90.9 tok/s [DERIVED]) | One of the inputs to the spine time | E0 |
| Hit-rate counter and per-token timer patches (measurement builds, in a copy of the tree) | Needed to measure the assumed inputs | E0 |
| CUDA minimum version and GPU architectures beyond 89 | Fixes the build matrix and the generality goal | E2 |
| Which extras enter E12 first | Order of the gated extras | E12 |
| Thinking on or off in the speed tests | Parked until the main sweep is done | after E0 |
| Repository: new folders in FireballJohnnie/pelias-engine, or a separate repo | Fixes the scaffolding tree and where the existing patches and results live | E2 |

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

## Sources

- Project documents, read this session: `pelias_tier_blueprint.md` (2026-10-20), `RUNBOOK.md` (hardware floors of 2026-10-20; results of 2026-10-22), `p1_tests/results/` (K and A results, 2026-10-22), and the HWiNFO report of 2026-10-20.
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/archive/13.0.3/wsl-user-guide/index.html), fetched 2026-10-22: WSL2 limits on managed memory, pinned memory, NVML and profilers.
- "Cacheable by Design?", arXiv 2608.18261: routing study and the negative result on forcing router locality. Cited from the tier blueprint; not re-fetched this session.
- Numbers from the running P1 sweep (C to N, J, B, O) are not in this document yet; the E0 results replace the assumed inputs when they arrive.
- E0 update 2026-10-22 (tier build, WSL2): live hit rate 0.775 to 0.779 for the 7.0 GiB bank against 0.806 simulated, not far below, so the LRU trigger in the risk table is not met; the per-token timeline shows no CPU and GPU overlap and a timer-free handoff of at least 4.6 to 5.0 ms per token (model: 0.25 ms). The 7.4 against 7.73 ms spine question stays open. Details: `benchmarks/RESULTS.md` section 9 and `benchmarks/raw/e0-2026-10-22/`.
