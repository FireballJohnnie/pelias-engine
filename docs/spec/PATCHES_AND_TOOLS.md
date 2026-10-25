# Engine components and layout, inventory of scripts, microbenchmarks and patches, the blueprint as first written, the llama.cpp patches and the microbenchmarks

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Parts 4.2, 4.3, 4.4, 4.5, 4.8 and 4.9 (master lines 1686 to 1928 and 2500 to 2533). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 4.2 Engine components, repository layout, build footprint and bundling (pelias-engine README)

<!-- SPLICE id="P4.2" src="pelias-engine README.md (engine head 34f744f)" lines="238-308" shift="+2" ws="0" edited="0" -->
> **Source (verbatim):** pelias-engine README.md (engine head 34f744f), lines 238-308. Heading levels shifted by +2 for this document only.

### 8. Status of every component

| Component | Status |
|---|---|
| Plain-layout, aligned copy of the expert tensors from a GGUF; GPU pointers per matrix | built; **verified on the real 40-layer file** |
| GPU zero-copy reads of that copy | built; verified (26.2 GB/s checksum kernel) |
| Miss splitter (rows divided between GPU and CPU; runs on host and device) | built; tested |
| Doorbell handoff | built; resident ping-pong measured; in-stream with a real CPU workload **not yet** |
| Cold-path GPU kernels (Q4_K and Q6_K only; other quant formats not written) | built; correct on synthetic data on the GPU; **speed 9.59 GB/s, to improve** |
| CPU worker | tested against real ggml on a CPU-only build; the always-polling version is **not written** |
| Hot bank patch for llama.cpp (cold on CPU only) | built for CUDA; decodes on the test machine (+5 to +7% over stock, measured); strict accuracy limits not met (stock does not meet them either; decision open); crash for batches of 9 or more tokens **fixed**, loose patches in `patches/stage2-tier-cuda/` |
| Cold path inside llama.cpp (device-side planning, graph ops, CPU worker, timeout fallback) | **not written** |
| Router traces, hot-set simulation | run on recorded traces: static and adaptive-cache hit rates simulated (derived); no live hit-rate counter yet |
| Measure-anything framework and go/no-go report | planned |

The detailed list, with the checks required before a release, is in `docs/STATUS.md` (not yet in this repository; it is kept in the author's development notes until release).

### 9. Repository layout, footprint and bundling

*This is the target layout. Until release the files live in the author's development folder and are copied here in stages; apart from this README and the LICENSE, nothing in this layout is in the repository yet.*

```
pelias-engine/
  README.md  LICENSE  CITATIONS.md  NOTICE  THIRD_PARTY_NOTICES.md  CONTRIBUTING.md  SECURITY.md  CHANGELOG.md  CITATION.cff
  docs/            design, measured results, porting guide, methodology, status
  src/
    core/          splitter, handoff, weights view, cold-path kernels, CPU pool, quant geometry
    portability/   the thin GPU API layer
    experimental/  profiler, state snapshots, speculative-decoding policy
  patches/
    llama.cpp/     the hot-bank patch (+ cold path later), build script, pinned base commit
  tools/
    measure/       hardware floors: RAM, PCIe, combined, handoff, pinning
    trace/         router-choice recording
    sim/           hot-bank simulation, predictor evaluation
    launcher/      server launcher
    parity/        reference dumps and comparison
    common/        shared helpers
  tests/           unit and hardware tests, CPU stand-in for the GPU API
  scripts/         numbered run-order scripts
  benchmarks/      RESULTS.md (claim-to-log ledger) and raw/ (redacted logs)
  trainer/         (later) the training engine and its patches
```

#### 9.1 Size of a build (proposed; the label says which numbers are measured)

| Item | Size | Label |
|---|---|---|
| Source repository, engine-only first release | existing 78 files: 828 KB; with the files still to be written, a few MB at most (mostly text) | measured + estimate |
| Front-page files (README, LICENSE, CITATIONS) | about 54 KB (README about 35 KB, LICENSE about 11 KB, CITATIONS about 8 KB) | measured |
| Patched llama.cpp build (CUDA, one GPU architecture) | unmeasured; path: `du -sh` of the build directory after the first patched build | unmeasured |
| Model file on disk (test model, Q4_K_M GGUF) | 20.22 GiB, of which 18.16 GiB are routed experts and about 2.06 GiB everything else (derived) | measured |
| Plain-layout copy of the experts in RAM (pinned) | 18.16 GiB, in addition to whatever llama.cpp holds; if llama.cpp also keeps its own CPU-side copy, both together are about 36.3 GiB (derived) against 64 GB of RAM; pointing llama.cpp's expert tensors at the single pinned copy is an option | measured + derived |
| VRAM | the spine, KV cache and compute buffers take about 4.8 GiB at 131k context with experts on the CPU (measured); the hot bank gets what is left of the 12 GB card (about 5.5 GiB per the project notes, derived) | measured + derived |

#### 9.2 Where it runs first

* **Tested:** Windows 11 with WSL2 (Ubuntu 26.04.1 LTS), NVIDIA RTX 4070 (sm_89), x86-64. Sources and the model sit on the Windows (NTFS) side and are read through the WSL mount; build outputs sit on the WSL ext4 disk.
* **Filesystem:** the engine copies the expert weights into RAM at start, so decode does not read the model file; any filesystem the GGUF can be read from works (design property, **unverified** beyond this machine). The load itself ran at 0.52 GB/s from the NTFS mount (**measured**); the same load from ext4 or a native Linux NVMe is **unmeasured**.
* **Initial target:** Linux (native or WSL2), NVIDIA CUDA, x86-64. **Not supported at first:** native Windows, AMD, Intel Arc (section 5 explains what they need), macOS (unified memory already; see section 6).
* Transparent huge pages were on for the measurements; they changed bandwidth by under 1% (**measured**), so they are optional.

#### 9.3 Bundling with the Pelias trainer

| | Size | Label |
|---|---|---|
| Trainer, all parts: Pelias 0.2.1 package (17 files, 102,542 B) and 0.2 launcher (1 file, 166,308 B), Pelias 0.1 package (15 files, 84,381 B) and 0.1 launcher (1 file, 166,297 B), v0.3 patch layer (2 files, 23,778 B), draft-head aligner (6 files, 90,903 B), 7 tests assigned by file name (90,801 B) | 49 files, 725,010 B (about 708 KiB) | measured; total derived |
| Trainer runtime environment (PyTorch, transformers, PEFT, Triton, GPU kernels, whatever else the author's environment holds) | installed, never bundled; size unmeasured; path: `du -sh` of its virtual environment | unmeasured |

**Decision (author, 2026-10-20): the trainer gets a separate repository, linked from this README** (repository: [`pelias-trainer`](https://github.com/<account>/pelias-trainer)). Reasons: different language and dependencies (Python / PyTorch against C++ / CUDA); the licence review its dependencies need (section 11) would otherwise hold up the engine; different release pace while the trainer is still being refined; a smaller clone and simpler CI for engine users; and the two meet at one clean interface, the GGUF file. The engine is model-agnostic, so a user can run it with any GGUF, trained with Pelias or not. If a single repository is wanted later, the trainer folder can be added to this one once its licence review is done, or the two can be tied together with a git submodule, without changing the engine.

<!-- /SPLICE P4.2 sha256="019554a8cafd46a725929398c3a92f9265aac5b100608bc25dc6837c58f68148" -->


## 4.3 Inventory of repository scripts, microbenchmarks, patches and baseline packages (research update 3.2)

<!-- SPLICE id="P4.3" src="pelias-research-update.md (received 2026-10-23 19:59, 870 lines)" lines="283-303" shift="+1" ws="1" edited="0" -->
> **Source (verbatim):** pelias-research-update.md (received 2026-10-23 19:59, 870 lines), lines 283-303. Heading levels shifted by +1 for this document only. Table rows that the source breaks over several lines are joined back (whitespace only).

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

<!-- /SPLICE P4.3 sha256="c5e07572adb72b984e46513f50055a85ece64d0ff987fbd29fb136dbc3f85dd4" -->


## 4.4 The engine blueprint as first written: scaffolding tree with bench/, tools/, tests/ and scripts/, and the existing work that moves in

<!-- SPLICE id="P4.4" src="PELIAS_ENGINE_LINUX_BLUEPRINT.md (engine head 34f744f, dated 2026-10-22)" lines="63-103" shift="+2" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_ENGINE_LINUX_BLUEPRINT.md (engine head 34f744f, dated 2026-10-22), lines 63-103. Original text, before any edit; the synchronized version is in Part 5.1. Heading levels shifted by +2 for this document only.

### Proposed scaffolding and file names

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

<!-- /SPLICE P4.4 sha256="2995e08982b00c7537021f926ca801b29b5dd8b9185442756cf9a6681e6365c4" -->


## 4.5 Generated inventories of what exists today (new text; counted from the files on 2026-10-24)

Counts and sizes below were produced by a script from the engine checkout (head 34f744f), the trainer repository (local head 953c9b1, three commits ahead of the pushed head c8e6ec4) and the staged backup folder. They are MEASURED by count. They show what is in each repository today and, by what is absent, which prototype work has not been moved in.

### 4.5.1 The engine repository (tracked files outside the raw logs)

| File | Bytes |
|---|---|
| `CITATIONS.md` | 6,842 |
| `LICENSE` | 11,358 |
| `README.md` | 40,538 |
| `benchmarks/RESULTS.md` | 34,573 |
| `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` | 18,518 |
| `patches/stage2-tier-cuda/cuda_hole_fix.patch` | 2,022 |
| `patches/stage2-tier-cuda/emulate_helper.py` | 4,065 |
| `patches/stage2-tier-cuda/mm_ids_dedup.patch` | 2,296 |
| `patches/stage2-tier-cuda/mmq_dump_diag.patch` | 6,028 |
| `patches/stage2-tier-cuda/mmq_dump_diag2.patch` | 3,192 |
| `patches/stage2-tier-cuda/pin_cold_to_cpu.patch` | 2,896 |
| `patches/stage2-tier-cuda/quantize_signed_id.patch` | 813 |
| **12 files** | **133,141** |

Raw logs under `benchmarks/raw/` (tracked, redacted): `cold-path-tests` 2 files; `e0-2026-10-22` 27 files; `gpu-read-and-handoff-tests` 4 files; `hardware-floors` 21 files; `p1-sweep-2026-10-22` 226 files; `stage2-tier-cuda` 103 files; `stock-llama-cpp-speed` 15 files; `windows-native-2026-10-23` 7 files. Total tracked files in the engine repository: 417.

**Not in the engine repository.** The microbenchmark programs 02 to 07, `gguf_min.py`, `expert_layer_sizes.json`, `collect_trace.py`, `export_hot_set.py`, `kl_gate.py`, `speed.py`, the `p1_tests` scripts, the base tier patch (`llama-pelias-tier.h` and `.cpp`, `pelias_tier_v1.patch`, `09_build_pelias_tier.sh`) and the two E0 measurement-only patches (`e0_timeline.patch`, `hit_counter.patch`) are described by the sources (research update 3.2, blueprint, prototype list) but are **not tracked** in the checkout: the engine README (section 9) says that until release the files live in the author's development folder. Their text is therefore not available to this document; what is known about each is from the ledger and the lists spliced in 4.3, 4.4 and 4.6 [READ-SRC for the lists; the files themselves UNVERIFIED here].

### 4.5.2 The trainer repository (tracked files by directory)

| Directory | Tracked files |
|---|---|
| `src/kern` | 30 |
| `src/hal` | 15 |
| `benchmarks` | 12 |
| `src/model` | 11 |
| `src/train` | 11 |
| `src/io` | 10 |
| `(root)` | 9 |
| `src/data` | 9 |
| `src/flow` | 9 |
| `src/intake` | 9 |
| `src/verify` | 9 |
| `src/warn` | 9 |
| `src/export` | 7 |
| `src/plan` | 7 |
| `src/project` | 7 |
| `src/sched` | 7 |
| `src/startup` | 7 |
| `tests` | 7 |
| `resources` | 6 |
| `src/stream` | 5 |
| `src/tok` | 5 |
| `src/config` | 3 |
| `src/help` | 3 |
| `src/ingest` | 3 |
| `src/serve` | 3 |
| `cmake` | 2 |
| `docs` | 2 |
| `include` | 2 |
| `src` | 2 |
| `src/cli` | 2 |
| `src/ui` | 2 |
| `tools` | 2 |
| `third_party` | 1 |
| **Total** | **228** |

Of these, 185 are under `src/`. The `tok` and `io/gguf_reader` files and the `serve` files are stubs; `src/serve/serve_command.h` is a declaration-free stub whose comment says it is to print the server command for a finished GGUF, held until unification (status in the file: NOT IMPLEMENTED). **It is not a server.** The trainer repository therefore holds no HTTP, socket, tokenizer or chat-template code at all, only these stubs [READ-SRC, stub files read 2026-10-24].

Stub check (bytes): `src/io/gguf_reader.cc` 299; `src/io/gguf_reader.h` 352; `src/serve/CMakeLists.txt` 267; `src/serve/serve_command.cc` 336; `src/serve/serve_command.h` 384; `src/tok/CMakeLists.txt` 240; `src/tok/chat_template.cc` 302; `src/tok/chat_template.h` 352; `src/tok/tokenizer.cc` 332; `src/tok/tokenizer.h` 386.

### 4.5.3 The Wave 0 probe program (staged backup, source only)

| File | Bytes |
|---|---|
| `2026-10-23/trainer_master/probes/README_PROBES.md` | 3,202 |
| `2026-10-23/trainer_master/probes/build.sh` | 1,366 |
| `2026-10-23/trainer_master/probes/cudautil.h` | 11,334 |
| `2026-10-23/trainer_master/probes/hostlib.h` | 20,714 |
| `2026-10-23/trainer_master/probes/kernels.cu` | 14,707 |
| `2026-10-23/trainer_master/probes/kernels.h` | 3,307 |
| `2026-10-23/trainer_master/probes/main.cpp` | 13,776 |
| `2026-10-23/trainer_master/probes/nvml_dyn.h` | 10,762 |
| `2026-10-23/trainer_master/probes/probes_a.h` | 24,188 |
| `2026-10-23/trainer_master/probes/probes_b.h` | 50,573 |
| `2026-10-24/trainer_master/probes/run_windows_all.ps1` | 27,286 |
| `2026-10-24/trainer_master/probes/run_wsl_all.sh` | 16,652 |
| **12 files** | **197,867** |

The probe program is the GPU source that produced the Wave 0 results of Part 3.3 on the author's machine. It is a measurement tool, not product code; Part 4.10 shows how its measurements become the start-up probe of the proprietary runtime.


## 4.8 The llama.cpp patches: what each did, why it was written, the ruling, and how the proprietary runtime handles the same hazard (new text)

The facts in the first four columns are from the engine ledger (Part 3.1, section 8.4 and the patch headers) and the research update inventory (4.3). The last column is new and is a **DECISION (proposed)**.

| Patch | What it did | Why it was written for the prototype | Author ruling of 2026-10-22 | In the proprietary runtime |
|---|---|---|---|---|
| `quantize_signed_id.patch` | In the non-scatter activation quantizer, widens each side of the conditional to 64 bits before the comparison, so the id -1 stays negative and the `if (i01 < 0) return;` guard fires | The tier gave ggml kernels a branch in which some rows name no expert (id -1, "this branch does nothing"). The conditional converted -1 to 4,294,967,295 and a row 4,294,967,295 was read; this was the cause of the crash for chunks of 9 or more tokens (ledger 8.4, MEASURED). It is the real fix | Carried; carry is read as reference record and patched oracle build only ([sync R2]) | No kernel ever receives a negative id. The partition step writes a hot list and a cold list, each holding only valid ids with a count, so there is no "hole" to skip. Unit test in `tests/unit/`: feed every combination of hit and miss for 1 to 64 tokens and check that no list contains a negative value |
| `cuda_hole_fix.patch` | Pre-fills the id map with -1 and the output with 0 (asynchronous memset), and skips negative ids in the quantizer | First hypothesis for the same crash. It did not remove the crash alone | Carried (same reading) | Same as above. The bank's zero slot (4,170 experts at 7.5 GiB include 72.6 MiB of zero slots) is a real slot holding zero weights, so a miss routed to it contributes exact zeros without a special path in a kernel |
| `mm_ids_dedup.patch` | Counts each (token, expert) pair once in the sorted-by-expert helper | Second hypothesis (rows that repeat the placeholder expert). It did not remove the crash. Whether it is still needed is **UNTESTED** | Carried | Each (token, expert) pair is counted once by construction in the partition (the pair list is built once per layer). The same unit test checks that no pair appears twice |
| `pin_cold_to_cpu.patch` | Pins the cold branch's operations to the CPU backend | The cold branch carries -1 ids that only the CPU kernel understands; the scheduler must not give it to the GPU | Carried | There is no scheduler that assigns operations to backends. The cold branch is a work item written by the GPU into the doorbell mailbox and read by the CPU workers (6.6, 6.7) |
| `mmq_dump_diag.patch`, `mmq_dump_diag2.patch` | Printed the id map after the helper and the buffer geometry when `PELIAS_MMID_DUMP=1`; otherwise inert | Diagnostics that found the cause (faulting address minus buffer start equals 0x7FFFFFFF800 bytes = 4,294,967,295 rows x 2,048 bytes) | Dropped | The runtime has a debug trace (`trace.h`, E2) that dumps the partition lists per layer when switched on; it is not a patch of a foreign file |
| `emulate_helper.py` | Emulates the helper's arithmetic in Python | Used to reason about the helper's output without a GPU | No ruling recorded | Not needed once the partition is the project's own code; the unit test replaces it |
| Base tier patch: `llama-pelias-tier.{h,cpp}`, `pelias_tier_v1.patch`, `09_build_pelias_tier.sh` | The tier itself: the hot bank in VRAM, the cold path on the CPU, the bypass rule for large prompt chunks, the hot-set file reader, the statistics file | The only way to measure the idea quickly was to patch a pinned llama.cpp (Part 8.2) | Not in the engine repository (4.5.1) | `bank/`, `gpu/`, `cpu/`, `engine/` of 6.4. The environment variables become the parameters of 6.14 |
| `e0_timeline.patch`, `hit_counter.patch` | Measurement-only: a per-stage timeline of one token (E0) and a live hit counter (0.7753 to 0.7793 at 7.0 GiB) | Needed to split the token time into stages before any engine existed | Not in the engine repository (4.5.1) | The per-token timers and hit counters are part of the engine (`timer.h`, `trace.h`, E2 and E8) and are on in the benchmark build |


## 4.9 The microbenchmarks, tools, scripts and the Wave 0 program: why each was written and what happens to it (new text)

| Existing item | Why it was written for the prototype | Proprietary counterpart | What happens to the original |
|---|---|---|---|
| `02_ram_bw_test.cpp` (`bench/ram_bw.cpp`) | To learn what the CPU can read from RAM before deciding that cold experts could be computed on the CPU | `bench/` stays as a measurement program; the start-up probe (`plan/probe_constants`) repeats a short version on the user's machine | Kept as the reference measurement; its result (43.8 to 45.9 GB/s) is the check for the probe |
| `03_handoff.cu` (`bench/handoff.cu`) | To price the CPU and GPU coordination per layer: stream sync 94 µs, resident-kernel ping-pong 13.3 µs median and 18.7 µs p99 | `bus/doorbell_handoff` (6.6) and its gate in E6 | Kept as the standalone reference; the engine's test must not be slower than its p99 |
| `04_dual_bus.cu` | To learn whether zero-copy reads and CPU reads can run at the same time without slowing each other (47.62 GB/s together) | The same measurement is a start-up probe; the overlap rule is part of the roofline solver | Kept |
| `05_pin_modes.cu` | To learn how much pinned memory each platform allows | `hal/pinned_pool` and the pageable fallback chain (6.9); P-H7 | Kept as a probe |
| `06_overlap.cu`, `07_wide_load.cu` | To test concurrent CPU and GPU expert work, and 16-byte staged reads (P-E10) | `gpu/expert_*` kernels and E5 | Kept; E5 must reach the checksum rate |
| `gguf_min.py`, `expert_layer_sizes.json` | To read the GGUF header without loading 20 GiB and to get the per-layer expert bytes | `io/gguf_reader` and the expert table (6.8) | Kept as the oracle for the reader's tensor list |
| `collect_trace.py` | To record router choices per token from a running server, because no engine could write them | The engine's `trace` sub-command (6.4 `cli/`) | Kept to check the trace file format; the engine writes the same format |
| `export_hot_set.py` and the hot-set profiles | To turn a trace into the per-layer resident set under a byte budget | `bank/hot_set_solver` and `hot_set_io` (6.4); byte-identical output is probe P-E14 | Kept as the oracle for P-E14 |
| `kl_gate.py` | To compare logits of a patched build against an untouched llama.cpp | `verify/` comparators; the llama.cpp side stays an external oracle in `tests/oracle/` | Kept; its protocol (16 chunks of 1,024 tokens) is the reference for the comparators |
| `speed.py`, `p1_tests` scripts (`run_floors.sh`, `run_base_cell.sh`, `run_sweep.sh`) | To measure decode speed through a server at short, 4,000 and 12,000 tokens | These are clients of the HTTP server and stay clients; the engine's `bench` sub-command repeats the measurement without a socket | Kept in `scripts/`; they work against any server that speaks the same API, which is a test of the server |
| `repack_gguf.py` (D25, R17) | To lay the experts out in the order and alignment the engine wants | The layout rule is the engine's; the trainer's writer writes it (`io/gguf_writer`, expert-last with `pelias.pad`, 6.8) | Kept as the oracle for the writer's layout; the engine's `cli/` has a `repack` sub-command |
| Wave 0 probe program (4.5.3; 12 files) | To measure the machine for the trainer plan and to settle the 10.7 GiB ceiling | `plan/probe_constants` and the start-up probe (6.9); the probe code is the starting point | Kept as the reference measurement; the start-up probe must reproduce its numbers on this machine |
| Trainer packages v0.1, v0.2.1, v0.3 and the MTP aligner (4.1) | The earlier trainer, written in Python on PyTorch, `transformers`, PEFT and Triton (trainer README, section 4), which produced the two exports and the two logged training runs | `train/`, `stream/`, `kern/`, `export/` of the trainer tree | Kept as the record and as the numeric reference where the master names it; not used for per-layer parity (master 10) |
| Launcher scripts and the numbered shell scripts of the prototype | To start the tier build with the right flags and environment | `cli/serve` with the parameters of 6.14 | Kept as the record of the flags the clients use today |
