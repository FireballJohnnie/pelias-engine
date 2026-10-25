# Pelias-Engine master scaffold

Written 2026-10-25, as part of the consolidation recorded in the blueprint as `[sync 11]`. The engine is **HELD** (author, 2026-10-24 06:07). This is documentation: it holds no engine code and lifts no hold. Nothing in it is pushed; the push waits for the author's word.

**What this file is.** A verified map of the whole engine directory: what is in it (section 2), what came from the search of the working tree (section 3), the target tree the engine is built into (section 4), which module replaces each piece of existing work and where that piece sits today (section 5), the phases and their gates with their recorded status (section 6), what is still open (section 7), what the search covered and left out (section 8), and how to check all of it again (section 9). The reasons behind the numbers are in `docs/PELIAS_ENGINE_LINEAGE.md`.

**Labels** are the engine's eight: MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED. A DECISION marked "proposed" is not ruled. Status in section 6 is taken only from recorded facts.

---

## 1. Verification summary

The repository was verified twice: first as it stood at commit `34f744f` (before any change of 2026-10-25), then as it stands now (the working tree before the local commit that carries this file). Documents were corrected where a check found them wrong; no check was changed to make it pass.

| Check | Result | How to repeat |
|---|---|---|
| Repository at `34f744f`: 417 files, 13,272,048 bytes | No zero-byte file, every text file decodes as UTF-8, every `.json` parses, no broken relative link in the Markdown. One raw-file reference is a glob (`raw/stage2-tier-cuda/speed/20261022_*`), not a missing file. One raw folder is not cited by path in `benchmarks/RESULTS.md`: `benchmarks/raw/p1-sweep-2026-10-22/` (cited in the lineage, section 6, item 4) | Check out `34f744f` in a scratch clone and run the same checks (the first-pass script is not part of the repository); section 9 holds the checks for the current tree |
| Current tree outside `support/archive` and `support/local_only`: 439 files | 433 text files all UTF-8; 90 JSON files all parse; 23 Markdown files with 3 relative links, all resolve; 257 of 257 repository paths cited in `docs/PELIAS_ENGINE_LINEAGE.md`, this file and `support/README.md` exist; two raw timeline files (`benchmarks/raw/e0-2026-10-22/timeline/B_graphs_on.host.tsv`, `C_graphs_off.host.tsv`) mix record types of 3, 4 and 9 fields by design (event logs) | section 9 |
| `support/MANIFEST.tsv`: 1,559 rows | Every pushable copy matches its recorded SHA-256 (1,374 archive files and 3 tier-build files); each of the 55 duplicate rows names a repository file with the same hash; all 1,374 files under `support/archive` are listed | `python3 docs/scaffold/verify_inventory.py` |
| Tier-build lineage | Both tier patches apply to llama.cpp 2ca15f540; the archived pre-pin file, the pin patch, the Windows seek patch and the current tree agree byte for byte (table in section 9) | section 9 |
| Privacy split | `support/local_only/` is ignored by git, nothing under it is tracked or staged (0 lines of `git status` mention it); privacy scan of every pushable file: see section 9 | section 9 |

**Findings of the verification** (details and evidence in `docs/PELIAS_ENGINE_LINEAGE.md`, section 6):

1. The base tier patch and its sources, said by `benchmarks/RESULTS.md` section 8 to be missing, were found and are in the archive; the current Windows-native tier tree was found and is in `support/tier_build/`.
2. The Windows tier tree still contains the two diagnostic patches that the ruling of 2026-10-22 dropped (open decision, blueprint decisions table).
3. Several file names in the research update, the blueprint tree and the README layout are target names, not the names of existing files (microbenchmarks 05 to 07, `overlap.cu`, `wide_load.cu`, the numbered P1 scripts, `collect_trace.py`, `speed.py`, `repack_gguf.py`). Section 5 gives the real names.
4. `benchmarks/raw/p1-sweep-2026-10-22/` is in the repository but is not cited by path in the ledger.
5. The drive's older `pelias-engine-release` folder (2026-10-21) is behind the repository; its files are archived here as they were.

---

## 2. Repository inventory

Every file of the repository outside `support/archive` (covered by section 3) and `support/local_only` (withheld) is listed in `docs/scaffold/REPO_INVENTORY.tsv` with its group, size, line count, SHA-256, status against commit `34f744f` and a note. Two files are left out of the inventory so that they can record its result: the inventory itself and this file.

**Totals (inventory):** 437 files, 14,341,232 bytes. Against `34f744f`: 415 unchanged, 2 modified, 20 new.

| Group | Files | Bytes | Unchanged | Modified | New |
|---|---|---|---|---|---|
| `root` | 5 | 60,814 | 2 | 1 | 2 |
| `benchmarks` | 1 | 34,573 | 1 | 0 | 0 |
| `benchmarks/raw/cold-path-tests` | 2 | 10,575 | 2 | 0 | 0 |
| `benchmarks/raw/e0-2026-10-22` | 27 | 10,049,167 | 27 | 0 | 0 |
| `benchmarks/raw/gpu-read-and-handoff-tests` | 4 | 27,206 | 4 | 0 | 0 |
| `benchmarks/raw/hardware-floors` | 21 | 87,792 | 21 | 0 | 0 |
| `benchmarks/raw/p1-sweep-2026-10-22` | 226 | 2,428,695 | 226 | 0 | 0 |
| `benchmarks/raw/stage2-tier-cuda` | 103 | 424,678 | 103 | 0 | 0 |
| `benchmarks/raw/stock-llama-cpp-speed` | 15 | 41,362 | 15 | 0 | 0 |
| `benchmarks/raw/windows-native-2026-10-23` | 7 | 49,758 | 7 | 0 | 0 |
| `docs` | 2 | 77,402 | 0 | 1 | 1 |
| `docs/scaffold` | 1 | 5,742 | 0 | 0 | 1 |
| `docs/spec` | 11 | 393,061 | 0 | 0 | 11 |
| `patches/stage2-tier-cuda` | 7 | 21,312 | 7 | 0 | 0 |
| `support (wrapper)` | 2 | 582,510 | 0 | 0 | 2 |
| `support/tier_build` | 3 | 46,585 | 0 | 0 | 3 |

**Named files** (the rest are the raw logs of `benchmarks/raw/`, listed in the inventory):

| File | Status | What it is |
|---|---|---|
| `.gitattributes` | new | Keeps archived line endings (and so the hashes) unchanged on checkout |
| `.gitignore` | new | Withholds support/local_only/ from git |
| `CITATIONS.md` | unchanged | Verified citations |
| `LICENSE` | unchanged | Apache-2.0 standard text |
| `README.md` | modified | Front page: what the engine is, measured results, plan, layout; 9.0 lists what the repository holds today (updated 2026-10-25) |
| `benchmarks/RESULTS.md` | unchanged | Claim-to-log ledger of the engine |
| `docs/PELIAS_ENGINE_LINEAGE.md` | new | How the plan and the numbers came to be what they are; register of old material; discrepancies |
| `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` | modified | The engine plan, phases E0 to E13, synchronized with the rulings of 2026-10-24 and consolidated 2026-10-25 |
| `docs/scaffold/verify_inventory.py` | new | Standard-library check of this inventory and of support/MANIFEST.tsv |
| `docs/spec/ENGINE_QUEUE_AND_MEASURABLES.md` | new | Queued items, what is needed, open author decisions, measurement registers, the hold |
| `docs/spec/MACHINE_AND_DRIFT.md` | new | Machine facts, engine facts, drifted assumptions, claim-by-claim review |
| `docs/spec/OPEN_DECISIONS_AND_ORDER.md` | new | Master order of operations and the open decisions O1 to O14 |
| `docs/spec/PATCHES_AND_TOOLS.md` | new | Engine components, script inventory, patches, microbenchmarks |
| `docs/spec/PROBE_REGISTRY.md` | new | Probe registry and decision register |
| `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` | new | Proprietary architecture and server specification |
| `docs/spec/RECONCILIATION.md` | new | Reconciliation of the research update with the ledgers |
| `docs/spec/RESEARCH_UPDATE_AUDIT.md` | new | Research update: core discovery, retirement of 57.0, E0 split, audit G1 to G18 |
| `docs/spec/SYNC_REGISTER.md` | new | Every edit made to the blueprint on 2026-10-24 and where it landed |
| `docs/spec/TRAINER_CONTRACT.md` | new | Trainer co-design contract |
| `docs/spec/WAVE0_AND_CEILING.md` | new | Wave 0 results, modelled ceiling, speed-to-ceiling table, attribution register |
| `patches/stage2-tier-cuda/cuda_hole_fix.patch` | unchanged | Stage-2 CUDA patch (first hypothesis for the crash: ids of -1 not skipped); carried by the ruling of 2026-10-22 |
| `patches/stage2-tier-cuda/emulate_helper.py` | unchanged | Helper script of the stage-2 runs |
| `patches/stage2-tier-cuda/mm_ids_dedup.patch` | unchanged | Stage-2 CUDA patch (second hypothesis: repeated placeholder rows in the sorted-by-expert helper); carried by the ruling of 2026-10-22; whether it is needed is untested |
| `patches/stage2-tier-cuda/mmq_dump_diag.patch` | unchanged | Diagnostic patch (prints the id map); dropped by the ruling of 2026-10-22 but still in the current Windows tier tree |
| `patches/stage2-tier-cuda/mmq_dump_diag2.patch` | unchanged | Diagnostic patch (prints buffer geometry); dropped by the ruling of 2026-10-22 but still in the current Windows tier tree |
| `patches/stage2-tier-cuda/pin_cold_to_cpu.patch` | unchanged | Graph patch (llama-graph.cpp) that keeps the cold-expert work on the CPU; carried by the ruling of 2026-10-22 |
| `patches/stage2-tier-cuda/quantize_signed_id.patch` | unchanged | The fix of the crash for chunks of 9 or more tokens (a signed id widened to an unsigned value); carried by the ruling of 2026-10-22 |
| `support/MANIFEST.tsv` | new | One row per file found by the search of the working tree |
| `support/README.md` | new | Layout, classes, redactions, privacy split and how to check the archive |
| `support/tier_build/pelias_tier_current_vs_2ca15f540.patch` | new | Current Windows-native tier tree as one patch on llama.cpp 2ca15f540 (10 files) |
| `support/tier_build/src/llama-pelias-tier.cpp` | new | Tier source, Windows line endings (equals the v1 source plus win_fseek.patch) |
| `support/tier_build/src/llama-pelias-tier.h` | new | Tier header (equals the v1 header) |

**Raw logs** (`benchmarks/raw/`):

| Folder | Files | Bytes | What it holds |
|---|---|---|---|
| `benchmarks/raw/cold-path-tests/` | 2 | 10,575 | Cold-path kernel test runs (one failure, then all pass) |
| `benchmarks/raw/e0-2026-10-22/` | 27 | 10,049,167 | E0 results: per-token timeline, hit counter, KL rerun, machine state (WSL2) |
| `benchmarks/raw/gpu-read-and-handoff-tests/` | 4 | 27,206 | GPU read of pinned memory and handoff tests |
| `benchmarks/raw/hardware-floors/` | 21 | 87,792 | Hardware floors: RAM bandwidth, handoff, dual bus, huge pages (2026-10-20) |
| `benchmarks/raw/p1-sweep-2026-10-22/` | 226 | 2,428,695 | P1 sweep: run order, results and terminal logs |
| `benchmarks/raw/stage2-tier-cuda/` | 103 | 424,678 | Stage-2 tier runs: hot sets, H6 KL, terminal output, simulation, speed |
| `benchmarks/raw/stock-llama-cpp-speed/` | 15 | 41,362 | Stock llama.cpp speed runs (2026-10-19 to 2026-10-20) |
| `benchmarks/raw/windows-native-2026-10-23/` | 7 | 49,758 | Windows-native tier build runs: configure log, per-token tables, tier stats |

---

## 3. The support archive (what the search of the working tree added)

The whole C:\AI tree was searched (section 8). `support/MANIFEST.tsv` has 1,559 rows, one for each file found that bears on the engine and was not in the repository as a file of the same content; 55 of them turned out identical to a repository file after redaction and are not copied again. `support/README.md` explains the layout, the classes and the redactions.

| Class | Files | Meaning |
|---|---|---|
| PUSH-VERBATIM | 516 | copied byte for byte |
| PUSH-REDACTED | 858 | copied after replacing private paths and names (counts per file in the manifest) |
| TIER-BUILD | 3 | the current Windows-native tier tree, in `support/tier_build/` |
| DUPLICATE-OF-REPO | 55 | identical to a repository file; not copied again |
| LOCAL-ONLY | 126 | withheld from the repository (kept on the drive and in `support/local_only/`, ignored by git) |
| NOT-COPIED | 1 | a 10.3 MB mailbox of the last two engine commits; the commits are in the history |

The pushable archive is 1,374 files, 13,328,596 bytes on disk.

| Group | In the repository | Withheld | Duplicate of a repository file | Not copied | Bytes of the pushable files (source) |
|---|---|---|---|---|---|
| `top` | 0 | 5 | 0 | 0 | 0 |
| `creator/(root)` | 5 | 6 | 0 | 0 | 80,691 |
| `creator/benchmarks` | 4 | 1 | 0 | 0 | 43,736 |
| `creator/logs` | 753 | 98 | 46 | 0 | 5,548,365 |
| `creator/moe_track` | 274 | 1 | 7 | 0 | 3,537,234 |
| `creator/pelias-engine-release` | 4 | 0 | 0 | 0 | 63,325 |
| `backup/2026-10-20` | 9 | 2 | 0 | 0 | 129,459 |
| `backup/2026-10-21` | 97 | 0 | 0 | 0 | 2,499,318 |
| `backup/2026-10-22` | 8 | 1 | 0 | 0 | 117,094 |
| `backup/2026-10-23` | 195 | 3 | 0 | 1 | 1,072,503 |
| `backup/2026-10-24` | 3 | 1 | 0 | 0 | 6,235 |
| `backup/2026-10-25` | 0 | 0 | 2 | 0 | 0 |
| `backup/(root)` | 1 | 5 | 0 | 0 | 5,719 |
| `backup/design_docs_first_conversation` | 2 | 3 | 0 | 0 | 13,526 |
| `pelias_probes_win` | 14 | 0 | 0 | 0 | 197,039 |
| `win_build` | 5 | 0 | 0 | 0 | 114,671 |
| `tier_build` | 3 | 0 | 0 | 0 | 46,585 |

The register that says what each group is and which result it leads to is `docs/PELIAS_ENGINE_LINEAGE.md`, section 5.

---

## 4. The scaffold target tree

Nothing in this tree exists yet apart from the documents. It is the tree the engine is built into when the hold is lifted. Two views agree: the blueprint's file map (4.1) and the proprietary-runtime module map (4.2). The blueprint states that its tree stays the engine-phase file map and that the proprietary-server additions are specified in `docs/spec/PROPRIETARY_RUNTIME_SPEC.md`.

### 4.1 The engine-phase file map (the blueprint, word for word)

Blueprint note: the letter after each entry is the phase that first creates it (E2 creates the whole empty tree and its build files; later phases fill the files in). Nothing is created yet.

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

### 4.2 The module map by layer (from `docs/spec/PROPRIETARY_RUNTIME_SPEC.md`, 6.4)

Names follow the trainer's tree where a module exists there. "Shared" means one source linked by both applications. **First built** gives the engine phase and the trainer stage. The full table, with the files and roles of each module, is in the spec; the replaced prototype piece is in section 5.

**Layer 0: libpelias_substrate (hardware abstraction, shared)**

| Module | Blueprint path | Owner | First built |
|---|---|---|---|
| `os/file_map` | (new; the blueprint has a plain reader in `model/gguf.h`) | Shared | T1, E3 |
| `os/net` | (new) | Shared (engine first) | E10 |
| `os/threads` | `cpu/threadpool.h` (OS part) | Shared | E4 |
| `os/large_pages` | (new) | Shared | E11 |
| `hal/device`, `stream`, `event` | `gpu/*` glue | Shared | T1, E2 |
| `hal/arena` | `mem/vram_plan.h` | Shared | T1, E2 |
| `hal/vram_governor` | `mem/vram_plan.h`, `mem/probe.h` | Shared | T1, E2, E11 |
| `hal/pinned_pool` and backends `host_alloc_wsl`, `host_alloc_win32` | `mem/host_alloc.h`, `src/mem/host_alloc_{wsl,win32}.cpp` | Shared | T1, E2, E11 |
| `hal/sync_audit` | - | Shared | T1 |
| `plan/` | `mem/vram_plan.h`, `mem/probe.h` | Shared | T1, E11 |
| `bus/doorbell_handoff` | `gpu/handoff.cuh` and the CPU side | Engine (substrate for reuse) | E6 |
| `bus/cpu_threadpool` | `cpu/threadpool.h` | Engine | E4 |
| `bus/stage_streamer` | (trainer `stream/`) | Trainer | T10 |
| `bank/hot_set_solver`, `hot_set_io` | `tools/export_hot_set.py`, `profiles/` | Shared | E5, T12 (P-E14) |
| `quant/` | `cpu/expert_q*_avx2.cpp`, `gpu/expert_q*.cu` | Shared | T1b, T3, E4, E5 |

**Layer 1: libpelias_core (model input and output, spine kernels, comparators; shared)**

| Module | Blueprint path | Owner | First built |
|---|---|---|---|
| `io/gguf_reader` | `model/gguf.h`, `tensor_map.h`, `expert_table.h` | Shared | T1, E3 |
| `io/gguf_writer`, `gguf_splice` | (trainer `export/`) | Trainer (substrate for reuse) | T1b, T12 |
| `io/sha256`, `io/json`, `io/safetensors` | - | Shared | T1 |
| `tok/` | `server/tokenizer.h`, `server/chat_template.h` | Shared | T2, E10 |
| `kern/` (spine family) | `gpu/spine/attention.cu`, `deltanet.cu`, `norm.cu`, `router.cu`, `lm_head.cu` | Shared (levels L2 of sharing) | T4 to T6, E7 |
| `kern/` (expert and matrix family) | `gpu/expert_q4k.cu`, `expert_q6k.cu`, `zerocopy.cu`, `bank.cu` | Engine (hot and cold branch) and Trainer (GEMM, LoRA); L3 not shared | E4, E5, T3 |
| `model/` | `model/arch.h`, `arch_moe.cpp` | Shared (spec); Trainer (stores) | E3, T7 |
| `verify/` | `tools/kl_gate.py` | Shared | E3, T12 |

**Layer 2A: pelias-engine (inference and server)**

| Module | Blueprint path | First built |
|---|---|---|
| `engine/` | `engine/layer.h`, `decode_loop.h`, `prefill.h`, `kvcache.h`, `state.h`, `sampler.h` | E6 to E9 |
| `cpu/` | `src/cpu/expert_q4k_avx2.cpp`, `expert_q6k_avx2.cpp` | E4 |
| `gpu/` | `src/gpu/bank.cu`, `zerocopy.cu`, `handoff.cu` | E5 to E7 |
| `server/` | `server/http.h`, `tools.h`, `api_openai.h` | E10 |
| `cli/` | `src/main.cpp` | E8, E11 |

Layer 2B (the trainer) is in the trainer repository; `docs/spec/TRAINER_CONTRACT.md` is the contract between the two.

### 4.3 Directories of the target tree and the existing material that moves in

| Directory | Holds | Existing material that moves in (section 5) |
|---|---|---|
| `bench/` | standalone measurement programs | microbenchmarks `02`, `03`, `04`, `07` and the Wave 0 probe program |
| `tools/` | offline Python tools | `export_hot_set.py`, `kl_gate.py`, `05_collect_traces.py`, `gguf_min.py` |
| `scripts/` | numbered run-order scripts | the lettered P1 scripts `A` to `X`, `05_n_sweep.sh`, `06_verify_gate_g0.py` |
| `tests/unit/`, `tests/parity/` | CPU-only unit tests; kernel and layer parity against the oracle | the 33 test programs of `creator/moe_track/tests/` |
| `tests/oracle/` | the external, read-only llama.cpp 2ca15f540 harness (never linked or built into the engine) | nothing moves in; `support/tier_build/` and `patches/stage2-tier-cuda/` are the record of the patched oracle build |
| `profiles/` | hot-set files and launcher profiles | the hot-set files of the runs (`benchmarks/raw/stage2-tier-cuda/hot-sets/`) |
| `docs/` | design, measured results, floors | this repository's `docs/`, `benchmarks/` and `support/` |

---

## 5. The replacement map

Which module of the proprietary runtime replaces each group of existing work, where that work sits today, and what is done with the original. Rows 1 to 13 are the replacement map of the master specification (4.10), word for word, with the column "Where it is today" added; the file-level table follows. Paths under `support/archive/c_ai/` are given from `creator/`, `backup/` or the folder name onward.

| # | Group of existing work (master 4.10) | Written for the prototype because | Where it is today | Replaced by (spec 6.4 onward) | Phase or stage | Label |
|---|---|---|---|---|---|---|
| 1 | llama.cpp as the engine (kernels, graph, scheduler) | It was the fastest way to measure the tier idea | external; pinned 2ca15f540 (b11399); the tier build is `support/tier_build/` | `engine/`, `gpu/`, `cpu/`, `kern/` | E4 to E9, T4 to T6 | DECISION (D37, AUTHOR-STATED) |
| 2 | llama.cpp as the HTTP server | The clients already spoke to it | external; the flags the clients use today are in `support/archive/c_ai/backup/2026-10-24/win_bat/start_pelias_server.bat` and `support/archive/c_ai/creator/moe_track/lib/llama_launcher.sh` | `server/` and `os/net` (6.12) | E10 | DECISION (proposed) |
| 3 | llama.cpp's GGUF reader | Part of the build | external; the prototype's header reader is `support/archive/c_ai/creator/moe_track/lib/gguf_min.py` | `io/gguf_reader` and `os/file_map` (6.8) | E3, T1 | DECISION (proposed) |
| 4 | llama.cpp's tokenizer and Jinja renderer | Part of the server | external | `tok/` (6.13) | T2, E10 | DECISION (proposed); effort unmeasured |
| 5 | llama.cpp's quantizers and dequantizers | Part of ggml | external (ggml); the prototype's block definitions are `support/archive/c_ai/creator/moe_track/pelias_serve/pelias_quant.h` | `quant/` (6.11) | T1b | DECISION (proposed); parity P-E6 |
| 6 | llama.cpp's sampler, prompt cache and slots | Part of the server | external | `engine/sampler`, `engine/state`, `server/` | E8, E10 | UNVERIFIED which behaviours the clients rely on (6.14) |
| 7 | The tier patches (4.8) | The tier lived inside ggml | `patches/stage2-tier-cuda/`; base `support/archive/c_ai/creator/moe_track/patches/`; current tree `support/tier_build/`; lineage in `docs/PELIAS_ENGINE_LINEAGE.md` section 4 | `bank/`, `gpu/`, `bus/` | E5, E6 | DECISION (proposed) |
| 8 | Environment variables of the tier (`PELIAS_HOT_SET`, `PELIAS_TIER_BYPASS_N`, `PELIAS_TIER_MAX_MIB`, `PELIAS_TIER_VERBOSE`, `PELIAS_TIER_STATS_FILE`) | The patch read them at start | set in `support/archive/c_ai/backup/2026-10-24/win_bat/start_pelias_server.bat`; read in `support/tier_build/src/llama-pelias-tier.cpp` | Parameters of 6.14 | E10, E11 | DECISION (proposed) |
| 9 | Microbenchmarks 02 to 07 | Each priced one hardware fact | `support/archive/c_ai/creator/moe_track/microbench/` (file names in 5.1) | `bench/` and the start-up probe | E0, E1, E11 | Kept as reference |
| 10 | Python tools (`collect_trace.py`, `export_hot_set.py`, `kl_gate.py`, `speed.py`, `repack_gguf.py`) | No engine could write traces, the hot set or the packed file | `support/archive/c_ai/creator/moe_track/sim/`, `support/archive/c_ai/creator/moe_track/trace/`, `support/archive/c_ai/creator/moe_track/lib/` (file names in 5.1) | `cli/` sub-commands, `bank/`, `verify/`, `io/gguf_writer` | E3 to E5, T12 | Kept as oracles |
| 11 | The Wave 0 probe program | The trainer's plan needed measured rates | `support/archive/c_ai/pelias_probes_win/`; results `support/archive/c_ai/backup/2026-10-23/probe_results/` | `plan/probe_constants`, `startup/` | T1, E11 | Kept as reference |
| 12 | The Python trainer packages | The first trainer | trainer repository | The trainer tree (master 5.2) | T0 to T14 | The record; D37 |
| 13 | llama.cpp itself, as an oracle | The only trusted implementation of the model | external | `tests/oracle/` (external, read-only, test-only) | E3, T12 | AUTHOR-STATED lock, Part 6.3 |

### 5.1 File level: the real names

The blueprint tree and the research update use target names. This table gives the file that exists and the target it becomes. A dash means no such file was found in any searched folder.

| Target (blueprint or research update) | File that exists | Becomes | Note |
|---|---|---|---|
| `tools/gguf_min.py`, `expert_layer_sizes.json` | `support/archive/c_ai/creator/moe_track/lib/gguf_min.py`; `support/archive/c_ai/creator/moe_track/microbench/01_gguf_expert_sizes.py`; `benchmarks/raw/hardware-floors/expert_layer_sizes.json` | `io/gguf_reader` oracle | tested by `support/archive/c_ai/creator/moe_track/tests/test_gguf_min.py` |
| `bench/ram_bw.cpp` (microbench 02) | `support/archive/c_ai/creator/moe_track/microbench/02_ram_bw_test.cpp` | `bench/ram_bw.cpp` | 43.8 to 45.9 GB/s [MEASURED] |
| `bench/handoff.cu` (microbench 03) | `support/archive/c_ai/creator/moe_track/microbench/03_handoff_latency.cu` | `bench/handoff.cu` | 94 µs against 13.3 µs median, 18.7 µs p99 [MEASURED] |
| `bench/dual_bus.cu` (microbench 04) | `support/archive/c_ai/creator/moe_track/microbench/04_dual_bus_test.cu` | `bench/dual_bus.cu` | 47.62 GB/s combined [MEASURED] |
| `bench/pin_modes.cu` (named "05" in the research update) | `support/archive/c_ai/creator/moe_track/microbench/07_pin_modes.cu` | `bench/pin_modes.cu` | numbered 07 on the drive |
| (gate G0 scripts) | `support/archive/c_ai/creator/moe_track/microbench/05_n_sweep.sh`, `06_verify_gate_g0.py` | `scripts/` | numbered 05 and 06 on the drive; not the programs the research update calls 05 and 06 |
| `bench/overlap.cu` (named "06") | - | P-H12 of the Wave 0 program (`support/archive/c_ai/pelias_probes_win/probes_b.h`) and the E0 timeline (`support/archive/c_ai/creator/moe_track/e0/e0_timeline.patch`) | no source of this name exists; overlap was derived from the E0 timeline and probed in P-H12 |
| `bench/wide_load.cu` (named "07") | -; nearest: `support/archive/c_ai/creator/moe_track/pelias_serve_wide/zc_mmvq_q4k_q6k.cuh` | E5 kernel (P-E10) | a variant of the prototype kernel using 16-byte loads; the hypothesis is unconfirmed and P-E10 has not run |
| `tools/collect_trace.py` | `support/archive/c_ai/creator/moe_track/trace/05_collect_traces.py`, `moe_trace.cpp`, `build_prompts.py`, `summarize_traces.py` | `cli/` sub-command `trace` | 121,160-step trace |
| `tools/export_hot_set.py` | `support/archive/c_ai/creator/moe_track/sim/export_hot_set.py` | `bank/hot_set_solver`, `hot_set_io`; oracle for P-E14 | water-fill; `waterfill_sim.py` and `predictor_eval.py` beside it |
| `tools/kl_gate.py` | `support/archive/c_ai/creator/moe_track/sim/kl_gate.py` | `verify/` comparators | protocol: 16 chunks of 1,024 tokens |
| `tools/speed.py` | `support/archive/c_ai/creator/moe_track/lib/speedlog.py`; the speed client scripts are withheld | clients of the HTTP server; `cli/ bench` | run against any server that speaks the same API |
| `tools/repack_gguf.py` | - | `io/gguf_writer`, `cli/ repack` | planned (D25, R17); gated on P-E10 |
| `scripts/run_floors.sh`, `run_base_cell.sh`, `run_sweep.sh` | `support/archive/c_ai/creator/moe_track/p1_tests/A_memory_loading.sh` to `X_b70_mtp2_repeat.sh`, `_lib.sh`, `probes/` | `scripts/` | the lettered P1 scripts; results in `benchmarks/raw/p1-sweep-2026-10-22/results/` |
| E0 timeline and hit counter | `support/archive/c_ai/creator/moe_track/e0/e0_timeline.patch`, `e0_hit_counter.patch`, `e0_timeline_analyze.py`, `token_latency.py` | `engine/` per-token timers and hit counters | measurement-only; not in the Windows tier tree |
| Cold-path prototype | `support/archive/c_ai/creator/moe_track/pelias_serve/` (14 files) | `model/` (`cold_weights_view`, `cold_pool`), `bus/doorbell_handoff` (`handoff.cuh`, `miss_splitter.h`), `kern/` expert family (`zc_mmvq_q4k_q6k.cuh`) | spec 6.4, "Replaces in the prototype"; kept as reference |
| Windows 64-bit seek | `support/archive/c_ai/creator/moe_track/e0/win_fseek.patch` | `os/file_map` (Windows backend) | in the Windows tier tree |
| Launchers | `support/archive/c_ai/creator/moe_track/lib/llama_launcher.sh`; `support/archive/c_ai/backup/2026-10-24/win_bat/` | `cli/ serve` with the parameters of 6.14 | the record of the flags the clients use today |
| Router-trace and fork tools | `support/archive/c_ai/creator/moe_track/trace/`, `support/archive/c_ai/creator/moe_track/fork/` | `cli/ trace`; the fork gauntlet stays a test client | speed leaderboard in `support/archive/c_ai/creator/logs/forks/` |
| BAR-size probes | `support/archive/c_ai/creator/moe_track/bar_probe/` | start-up probe (E11) | ReBAR question E-A6, held with native Linux |
| Draft-head (MTP) aligner | `support/archive/c_ai/creator/moe_track/mtp_align/` | trainer side | the unused MTP block is a possible factor (UNVERIFIED) |
| Test programs of the prototype | `support/archive/c_ai/creator/moe_track/tests/` (33), `support/archive/c_ai/creator/moe_track/p1_tests/` | `tests/unit/`, `tests/parity/` | CPU-side tests run in the sandbox |

---

## 6. Phase and gate map, E0 to E13

Gates are the blueprint's, word for word. Status is only what the records show. The order of work above the phases (trainer first, engine held, then unify) is in `docs/spec/OPEN_DECISIONS_AND_ORDER.md`, section 7.1.

| Phase | Gate to move on (blueprint) | Status as recorded | Specification |
|---|---|---|---|
| E0 Close the measurements | Spine, hit rate and overlap are measured, and the gap between about 92 tok/s and the live figures is explained. | Partly done; the gate is not recorded as met. P1 sweep: the results folder holds cells A to U, W and X (no V) in `benchmarks/raw/p1-sweep-2026-10-22/`. Five measurements: per-token breakdown and live hit rate measured on 2026-10-22 (WSL2, Reference Cell A; the split of the gap is DERIVED, not isolated); overlap derived (the parts add up to the wall time; 0% overlap) and probed in P-H12; wide-load kernel (P-E10) not run; cause of the load-mode effect not established (E-A5 and P-E11 not run). Four author rulings obtained on 2026-10-22 (ceiling, accuracy gate, patches, reference cell) | 7.2; `docs/spec/WAVE0_AND_CEILING.md` |
| E1 Platform floors [sync R1] | A floor table for WSL2 and Windows-native side by side; probe contents chosen from the differences. | Partly done. The Wave 0 probe program ran on WSL2 (2026-10-23) and on Windows-native (2026-10-23/24), every step exiting 0 (`docs/spec/WAVE0_AND_CEILING.md`); a Windows-native run of the tier build was recorded on 2026-10-23 (`benchmarks/raw/windows-native-2026-10-23/`; two Windows launches against one WSL launch, so not yet a claim). The floor programs 02, 03, 04 and 07 themselves were run on WSL2 on 2026-10-20 (`benchmarks/raw/hardware-floors/`); a Windows-native run of them is not in the records searched. Three launches per side are not done. Native Linux is held | `docs/spec/WAVE0_AND_CEILING.md`, `docs/spec/PROBE_REGISTRY.md` |
| E2 Skeleton | Clean build on WSL2 and on Windows-native (MSVC and nvcc 13.4.92 for the Windows tree) [sync R1]; CPU tests pass in the sandbox. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` 6.4, 6.5, 6.9 |
| E3 Model reader and oracle | Reads the Q4_K_M file; the oracle reproduces the stock noise floor (KL 0.017353 at batch 16 vs 0.018286 at batch 1024 [MEASURED]); a trainer-exported GGUF and its layout manifest parse identically, and K3 holds on it [sync 4]. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.8, 6.11; `docs/spec/TRAINER_CONTRACT.md` |
| E4 CPU expert kernels | Throughput at or above llama.cpp's CPU expert path (about 27 GB/s effective [DERIVED]); output matches the oracle. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.7, 6.11 |
| E5 GPU expert kernels and bank | Zero-copy compute kernel reaches the checksum rate (26.2 GB/s target, from 9.59); bank experts match the oracle. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.9, 6.10; P-E10, P-E14 |
| E6 One layer end to end | Layer latency within the E0 model; output within the oracle noise floor; handoff p99 no worse than 18.7 µs on native Linux. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.6 |
| E7 Full spine | Full-model KL vs stock no worse than the stock noise floor, until the H6 ruling sets the limit. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.10 |
| E8 Decode loop | First at least the llama.cpp tier result, then G3 (55.0 tok/s or more at 12k), then the ceiling table from E0 and E1. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.7; open decision O11 |
| E9 Prefill and long context | Prompt reading at least 1,670 tok/s at 12k (llama.cpp tier) and aimed at 2,238 (stock with load mode none) [MEASURED]; peak VRAM within the author's ceiling. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.7 |
| E10 Server | Drop-in for the launcher; persona probe not-pass 12 of 111 or fewer (baseline 9), needles 12/12 to 120k, a Visual Studio Copilot session works. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.12, 6.13, 6.14 |
| E11 Capability probe | The same code base passes the E8 to E10 gates on WSL2 and on Windows-native [sync R1]. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.9; P-E11 |
| E12 Gated extras | Zero-copy: GPU rate at least 15 GB/s and combined at least 1.25x CPU alone. LRU: only if the live hit rate is well below 0.83. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | `docs/spec/PROBE_REGISTRY.md`; open decision O2 |
| E13 Hardening and generality | A second MoE model runs through the plugin interface; regression suite green. | Not started. The engine is HELD (author, 2026-10-24 06:07); no engine source code exists | 6.4 (plug-in interface) |

---

## 7. Open gaps and decisions

### 7.1 Open decisions O1 to O14 (`docs/spec/OPEN_DECISIONS_AND_ORDER.md`, 7.16)

| # | Decision | Status |
|---|---|---|
| O1 | Is the 7.5 GiB bank a ruling, or the current best step under the 10.7 GiB ceiling? May the bank be a non-multiple of 0.5 GiB? | Open; this document treats 7.5 as derived (master 16.7, decision 1) |
| O2 | Is Mode L (and L-Z) worth building after the Wave 0 probes P-L1, P-L2, P-E2 and P-E3 report? | Open |
| O3 | What does a blank answer in the parameter sheet do (knowledge-only rows, or the base model drafts an answer for the user to review)? | Open; asked 2026-10-24, no answer |
| O4 | May Expert mode switch off the clean-stop guard, or only change its limits? | Open; proposed: limits only; not yet asked |
| O5 | May Claude read the author's verification script and the gate set it names? | Open; go-ahead awaited |
| O6 | May Claude look up the published request formats of the named clients? | Open; go-ahead awaited |
| O7 | The support-bundle wording (trainer U10) | Open; clarification asked |
| O8 | Add the hold-queue line "catalogue llama.cpp's parameters and behaviours as the spec for the Pelias server"? | Open; asked twice, unanswered |
| O9 | Dear ImGui: keep for the interface only, write an own layer, or build headless first? | Open |
| O10 | The server's policy for special-token text inside user content: pass, escape or refuse | Open; this document proposes `pass` as the default because it reproduces the behaviour the clients have today, with every occurrence counted |
| O11 | May the original E8 kill criterion (fall back to a llama.cpp patch series) be restored? | Open; removed in this document by `[sync R2]` |
| O12 | Does the ruling "WSL2 and Windows-native, native Linux held" apply to the engine as well as to the trainer draft? | Applied by `[sync R1]`; the author may correct it |
| O13 | When does the engine hold lift, and does the ground-up engine work start before the trainer is "top tier"? | Held at the author's word |
| O14 | Order of work: the Batch C probe source, or the llama-server catalogue? | The 09:09 request (this document) takes precedence; both are waiting |

### 7.2 The research update's claims G1 to G18 and their verdicts (`docs/spec/RESEARCH_UPDATE_AUDIT.md`)

| Id | Claim (shortened; full text in the audit) | Verdict |
|---|---|---|
| G1 | Windows-native runs at `57.0 tok/s (17.54 ms/token)`, gap `6.67 ms` (`166.7 µs/layer`)[cite: 6]. | Rejected (Not in repo) |
| G2 | WDDM `VidMm` manages system memory strictly in `4-KB` pages (`DXGK_PTE`)[cite: 6]. | PUBLISHED (True) |
| G3 | WSL `3.0.4` (`ce88638`), kernel `6.18.54.1`, removed bundled `perf` (`#41904`), `64 KB` stack (`#41888`)[cite:... | Version MEASURED; PRs UNVERIFIED |
| G4 | `4-KB` guest pages in `mmap` cause the `2×` prefill swing (`863` vs. `1,680 tok/s`); `madvise(MADV_HUGEPAGE)`... | Contradicted by Repo Data |
| G5 | Windows: allocate cold bank with `VirtualAlloc(MEM_LARGE_PAGES)` + `SeLockMemoryPrivilege`, staging buffer wit... | UNVERIFIED Hypothesis |
| G6 | Pin 8 worker threads to the 8 physical Zen 3 cores (`1T` mode); Load/Store/Retire queues are split 50/50 in `2... | Partly PUBLISHED + MEASURED |
| G7 | Use `acquire`/`release` (`MOV`) instead of `seq_cst` (`LOCK XCHG`) on handoff flags; avoid FP instructions rig... | PUBLISHED (True); benefit UNVERIFIED |
| G8 | Use `_mm_prefetch(..., _MM_HINT_NTA)` on cold expert streaming to protect the `96 MB` victim L3 cache[cite: 6]... | UNVERIFIED Hypothesis |
| G9 | `16-byte` (`uint4`) loads staged through `__shared__` memory lift GPU zero-copy compute reads from `9.59` to `... | UNVERIFIED Hypothesis |
| G10 | Pin `16–24 MB` of the RTX 4070's `36 MB` L2 cache via `cudaAccessPropertyPersisting`[cite: 6]. | UNVERIFIED Hypothesis |
| G11 | Hold `/dev/cpu_dma_latency = 0` on Linux; run `nvidia-smi -lgc 2475,2475`[cite: 6]. | Clock Lock REJECTED |
| G12 | Native Linux projected at `72–86+ tok/s`[cite: 6]. | UNVERIFIED (Not used) |
| G13 | Gate thresholds (`≥22 / ≥25 GB/s`, `p99 ≤ 12 µs`) and Oct 23–Nov 22 calendar[cite: 6]. | Not Adopted |
| G14 | RTX 4070 specs: 46 SMs, 5,888 cores, 192-bit bus, `504 GB/s`, 36 MB L2, `sm_89`[cite: 6]. | PUBLISHED (Consistent) |
| G15 | Pinned host memory capped at `≤ 32 GB` under WDDM[cite: 6]. | UNVERIFIED above 20.4 GiB |
| G16 | WSL2 runs at `44–54 tok/s` short and `43–49 tok/s` at `12k`[cite: 6]. | MEASURED (Config-specific) |
| G17 | `/dev/dxg` VMBus adds `25–35 µs/layer` (`1.0–1.4 ms/token`), explaining the Win11 vs. WSL2 gap[cite: 6]. | UNVERIFIED & Partial |
| G18 | Windows-native thread pinning + `SetThreadInformation` (`HighQoS`) + `timeBeginPeriod(1)` cuts `5–21%` spread[... | UNVERIFIED Hypothesis |

### 7.3 Open items found or confirmed by this consolidation

| # | Item | Where |
|---|---|---|
| N1 | Keep or drop the two diagnostic patches in the Windows tier tree | blueprint decisions table; lineage L9 |
| N2 | The llama-server catalogue (361 option definitions, 36 distinct POST route strings, about 86 request fields at 2ca15f540: rough counts only) was never written; what the clients send is unmeasured (O6, O8) | `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` 6.14 |
| N3 | The Batch C probe source (`kernels_c`, `probes_c`, extended `main.cpp`, run scripts) is not written | `docs/spec/PROBE_REGISTRY.md` |
| N4 | The first run of a non-llama server, which would isolate the gap to the ceiling, does not exist | lineage L3 and section 8 |
| N5 | Wide-load kernel rate (P-E10), huge-page effect (P-E11), E0 timeline on the proprietary runtime (P-E2), VRAM peaks and disturbance margin (P-E4, O1): unmeasured | `docs/spec/ENGINE_QUEUE_AND_MEASURABLES.md` |
| N6 | Three launches per side are needed before a Windows-native against WSL2 speed claim (the recorded gain rests on two launches against one) | blueprint, platform section |
| N7 | Whether the drive's older `pelias-engine-release` folder should be refreshed from the repository | lineage section 6, item 5 |
| N8 | File names that exist only as targets (section 5.1) are renamed or created at E2, not before | section 5.1 |
| N9 | Raw logs already pushed at `34f744f` carry the first words of the model's reply after "answer starts:" (231 lines in 94 files). The archive copies have that text removed. The pushed logs are left as they are; whether to redact them is the author's decision, and it would change those files' hashes | section 9.3 |
| N10 | Two connected folders were not searched or copied because the request named C:\AI: `C:\Ai-Prposed` (the trainer's C++/CUDA proposals) and `D:\AI-Old\New folder\Stuart-Creator` (an older copy of the project folder) | section 8 |

The blueprint's own list of unknowns and how each gets measured (including the audited hypotheses E-A1 to E-A7) is the section "Unknowns and how each gets measured" of `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`.

---

## 8. Scope, search method and what was not copied

**Method.** Every file under C:\AI (7,621 files at the time) was listed with its size, date and SHA-256; text files were scored for engine-related content (the engine, the tier build, the hot bank, the handoff, the ceiling, the probes, the server) and grouped by folder. Selection was by content and folder, not by moving whole folders. Each selected file was then classified for privacy (section 3), redacted where needed, copied with its hash recorded, and the copy was verified against the hash. The listing and the selection are on the drive in the folder of 2026-10-25 (`engine_scrape`).

| Outcome | Files | Reason |
|---|---|---|
| Selected (first pass) | 1,553 | engine research track, probes, run logs, project records and plans; listed in the manifest |
| Added after the first pass | 3 | the two interim Windows launcher scripts and the sandbox archive of 2026-10-24 (both had been left out because their folders were on the exclusion list) |
| Current Windows-native tier tree | 3 | saved as a patch and two source files in `support/tier_build/` |
| Identical to a file already in the repository | 301 | not listed again |
| Same content as another selected file | 257 | folded into the selected copy; the other paths are recorded as aliases in the selection list on the drive |
| Stock llama.cpp source and build output | 4,399 | third-party and regenerable; the changed files are saved as the tier patch |
| Third-party source | 80 | not engine material |
| Trainer work, earlier prototype or this assembly's own working files | 557 | kept in the drive's backup tree (trainer repository, UI framework work in progress, master builders, earlier sandbox work) |
| Trainer, persona, setup or unrelated material | 404 | setup and training logs, trainer releases, persona tests; not engine material |
| Router trace data over 1 MB, and files over 30 MB | 41 | location and size recorded; the traces can be regenerated with the trace tools |
| Binary files | 28 | built executables and archives |
| A file that looked like a credential | 1 | not opened and not copied |

Of the 7,621 files listed: 1,553 selected, 301 already in the repository, 257 duplicates folded, 5,510 excluded (4,399 + 80 + 557 + 404 + 41 + 28 + 1 = 5,510). The 3 later additions and the 3 tier-build files bring the manifest to 1,559 rows.

**Not copied, by decision.**

- `C:\Ai-Prposed` (the trainer's C++/CUDA proposals) and `D:\AI-Old\New folder\Stuart-Creator` (an older copy of the project folder) are connected to the session but outside the request, which named C:\AI. Say so if they should be searched too.
- The trainer's own documents, bundles and framework work are in the trainer repository and the drive's backup tree. The trainer master's engine chapters (16.3 to 16.7, 11.12) are spliced into `docs/spec/`.
- The older `pelias-engine-release` folder on the drive was not touched.
- The mailbox of the last two engine commits (10.3 MB, redundant with the repository history).
- The author's master specification (`PELIAS_PROPRIETARY_MASTER_SERVER.md`): a local archive that quotes backup paths and names. Its engine parts are in `docs/spec/`; a copy sits in `support/local_only/master/` in the working directory, ignored by git.

**What "everything" means here.** The repository now holds, or points to a withheld copy of, every file in C:\AI that bears on the engine, apart from the items above. A file found later is added to the manifest with the same procedure; the search can be rerun from the listing on the drive.

---

## 9. Verification commands and results

Run on 2026-10-25 against the working tree on top of commit `34f744f`, before the local commit that carries this file. The checks change nothing in the history.

**9.1 Inventory and manifest** (`python3 docs/scaffold/verify_inventory.py`; exit status 0):

```text
inventory: 437 rows, 437 match
tree: 439 files outside support/archive and support/local_only, 2 left out on purpose
manifest: 1559 rows (DUPLICATE-OF-REPO 55, LOCAL-ONLY 126, NOT-COPIED 1, PUSH-REDACTED 858, PUSH-VERBATIM 516, TIER-BUILD 3); 1432 copies or duplicate targets match
archive: 1374 files, all listed
git: no file under support/local_only is tracked
local manifest: 126 rows, 0 problems
RESULT: all checks passed
```

**9.2 Tier-build lineage** (patches applied to a clean checkout of llama.cpp 2ca15f540 in a scratch directory):

| Check | Result |
|---|---|
| base commit of the checkout | pass |
| git apply --check pelias_tier_v1.patch on 2ca15f540 | pass |
| git apply --check pelias_tier_current_vs_2ca15f540.patch on 2ca15f540 | pass |
| llama-graph.cpp after v1 equals the archived before_pin file | pass |
| before_pin plus pin_cold_to_cpu.patch equals llama-graph.cpp of the current patch | pass |
| v1 llama-pelias-tier.cpp plus win_fseek.patch equals the tier_build source without CR | pass |
| tier_build header equals the v1 header | pass |

**9.3 Privacy scan** (counts of occurrences; no value is printed). Pattern families: the local account name, the personal name outside the approved copyright line, the repository owner account name, a Hugging Face token pattern, a Windows profile path other than the redacted form, a home path other than the redacted form, an e-mail address, the names of private scripts and equations, persona design wording.

| Pattern | In files new since 34f744f | Added to files that existed at 34f744f | Already present at 34f744f | In the archive |
|---|---|---|---|---|
| the local account name | 0 | 0 | 0 | 0 |
| the personal name outside the approved copyright line | 0 | 0 | 2 | 0 |
| the repository owner account name | 0 | 0 | 3 | 0 |
| a Hugging Face token pattern | 0 | 0 | 0 | 0 |
| a Windows profile path other than the redacted form | 0 | 0 | 0 | 0 |
| a home path other than /home/user | 0 | 0 | 0 | 3 |
| an e-mail address | 0 | 0 | 0 | 0 |
| a private script or equation name | 0 | 0 | 0 | 0 |
| persona design wording | 0 | 0 | 231 | 0 |

Already present in the repository at 34f744f (not added by this consolidation):

- the personal name outside the approved copyright line: 2 occurrence(s) in 1 file(s) (`README.md`); the author line of the README, written by the author.
- the repository owner account name: 3 occurrence(s) in 2 file(s) (`README.md`, `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`).
- persona design wording: 231 occurrence(s) in 94 file(s); these are the first words of a model reply after "answer starts:" in raw logs (open item N9).

Non-zero counts outside what was already present:

- a home path other than /home/user: 3 occurrence(s) in 2 file(s) (`support/archive/c_ai/backup/2026-10-23/inputs/nvidia_skills_catalog_20261023.txt`, `support/archive/c_ai/creator/moe_track/tests/test_pelias_v03.py`).

The home-path hits were read: one is a sandbox path inside an old test script (a one-letter folder name, not the user's account), and two are example paths inside a vendor skills catalogue (a generic cloud-image account). None names the user or the machine.

**9.4 Git state.** `support/local_only/` is matched by the rule `.gitignore line 3: support/local_only/`; at the time of the checks the index held nothing. Nothing is pushed: the push waits for the author's word.

**9.5 Links and paths** (current tree): 3 relative Markdown links (0 broken); 257 cited repository paths, 0 not found. The two ragged TSV files are the event logs named in section 1.

