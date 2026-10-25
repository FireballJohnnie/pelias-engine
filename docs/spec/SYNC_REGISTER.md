# Synchronization register: every edit made to the engine blueprint on 2026-10-24, and where it landed

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Part 5.0 (master lines 2556 to 2578) followed by the file `blueprint_edits_applied.md` (the 42 edits, old text and new text). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. The edits were applied to `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` on 2026-10-25; the old text in each entry is the blueprint as it stood at engine commit 34f744f.
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

## 5.0 The synchronization register (new text): every edit, its source, and where it landed

The ten mandatory edits are the ones the research update (5.1), the trainer master (16.7) and the trainer notes (3.4) agree on. Where this document's version of an edit differs from the research update's wording, the difference and the reason are in the last column. The four `R` marks apply author rulings of 2026-10-24 that the blueprint (dated 2026-10-22) did not yet reflect.

| Mark | What it changes | Source | Where it landed | Difference from the source wording |
|---|---|---|---|---|
| `[sync 1]` | The router sentence in the scope line: inference-time router clamping on an untrained model is out of scope, with the published study's figures stated correctly; trainer-side router training (Mode L) is an input, not excluded | Update 5.1 item 1; master 16.7 (1) | Part 5.1, "Goal and scope" ("Not in scope") and the synchronization note | None in substance |
| `[sync 2]` | Adds the VRAM budget and headroom table | Update 5.1 item 2; master 16.3.1 | Part 5.1, new heading after "Where the numbers stand"; Part 7.5 (bank decision); Part 7.6 | The update used the superseded 10.9 GiB ceiling (headroom 0.36 and 0.11 GiB) and the label "-ub 4096" for the 10.40 to 10.44 GiB peaks. This document uses the 10.7 GiB ceiling and copies master 16.3.1 word for word (headroom 0.16 GiB at the default micro-batch, none at -ub 2048, whole-card reading); the micro-batch of the 10.40 to 10.44 GiB runs is unrecorded (reconciliation D2) |
| `[sync 3]` | The CPU-read row: 43.8 to 45.9 GB/s with the CPU alone; 26.6 GB/s is its share while the GPU reads | Update 5.1 item 3 | Part 5.1, "Where the numbers stand" | None |
| `[sync 4]` | The E3 gate reads trainer-exported GGUFs and checks three KL gates (K1, K2, K3) against the file's measured float spread | Update 5.1 item 4; master 16.7 (4) | Part 7.2 (E3); Part 7.3 (ladder) | None |
| `[sync 5]` | The E5 bank loader supports a different capacity per layer (84 to 161 experts per layer at 7.5 GiB) and handles the Q6_K 210-byte blocks (420-byte rows) | Update 5.1 item 5 | Part 7.2 (E5) | None |
| `[sync 6]` | The start-up probe logs the memory-page settings and `AnonHugePages` during model load (P-E11) | Update 5.1 item 6 | Part 7.2 (E11) | Platform wording follows R1 |
| `[sync 7]` | The E12 hot-set objective (byte-weighted hit against zero-miss layer share, decided by P-E2) and the Mode L candidates in the gated extras | Update 5.1 item 7 | Part 7.2 (E12); Part 7.5 | None |
| `[sync 8]` | Adds E-A1 to E-A7 to the unknowns table as audited hypotheses | Update 5.1 item 8 | Part 7.6 | None |
| `[sync 9]` | Adds the trainer co-design contract; the engine owns the packed layout and `repack_gguf.py` (R17, D25); the trainer emits standard GGUFs, optionally with `--layout expert-last` (a pad entry under 2 MiB, P-E9) | Update 5.1 item 9; master 16.7 | Part 5.2; Part 7.2 (after E13) | The contract table is copied from the master, whose last column reads 10.7 |
| `[sync 10]` | Sources: the engine ledger sections 1 to 9.4, the E0 and Windows-native raw folders, the full-text checks of the two papers, and the outside hardware plan marked as an audited hypothesis list | Update 5.1 item 10 | Part 5.1, "Sources" | None |
| `[sync R1]` | The draft is built as a WSL2 version and a Windows-native version; native Linux is held and will be a separate entity | Author ruling 2026-10-24 06:21 (AUTHOR-STATED), made for the trainer draft | Title, goal, target system, "Not in scope", the scaffolding tree (`host_alloc_win32`, Windows build flag), E1, E2, E11, the platform section, risks, decisions, unknowns | **Applying it to the engine is this document's extension of the ruling.** Flagged so the author can correct it (Part 7.16, O12) |
| `[sync R2]` | llama.cpp is never part of the runtime: it is an external, read-only test oracle under `tests/oracle/`; the engine's reader, tokenizer, renderer and server are its own | Author ruling 2026-10-24 08:15 (AUTHOR-STATED); trainer D37 | Working rules, components, scaffolding tree, E3, E7, E10, ladder, risks (E8 kill criterion), decisions, unknowns | The original E8 kill criterion ("fold the findings back into a llama.cpp patch series") is **removed**; whether to restore it is open for the author (O11) |
| `[sync R3]` | The VRAM ceiling is 10.7 GiB, not 10.9 | Trainer D26, D28; author ruling 2026-10-24 | Part 7.5 (decision text); the new VRAM table | The ruling of 2026-10-22 (10.9) stays visible and is marked superseded |
| `[sync R4]` | The 07:39 rulings that touch the engine: the prefill micro-batch is an engine setting; no Q8_0 or Q6_K core inflation by default | Trainer D31, D32, D34 | Synchronization note; Part 7.5 | None |

**How the edits were checked.** Each edit replaces one anchored string of the blueprint, and the builder refuses to run unless the anchor occurs exactly once. After the build, the verifier re-applies the same list of edits to the source and compares the result with the block in this document, character for character. A block with edits is therefore proven to be "source plus the listed edits" and nothing else.

# Blueprint edits applied in PELIAS_PROPRIETARY_MASTER_SERVER.md

Every edit below was applied in place to the engine blueprint (engine head 34f744f, dated 2026-10-22) inside the master document. The pelias-engine repository itself is unchanged. Each entry shows the text replaced and the text that replaces it, word for word; headings are shown at the level they have in the master document. The segment letter is the blueprint segment (A title, B goal and scope, C numbers, D components, E scaffolding, F order of operations, G ladder, H platform, I risks, J decisions, K unknowns, L sources).

# Edit 1. Segment A, edit 1. Marks: [sync R1]

**Replaced:**

> Pelias-Engine (Linux): build blueprint

**With:**

> Pelias-Engine: build blueprint, synchronized (WSL2 + Windows-native draft; native Linux held) [sync R1]

# Edit 2. Segment A, edit 2. Marks: (no mark: wording kept consistent with a marked edit)

**Replaced:**

>
> 2026-10-22
>

**With:**

>
> 2026-10-22 (original blueprint, engine commit 34f744f); synchronized in this document on 2026-10-24
>

# Edit 3. Segment A, edit 3. Marks: [sync R1]

**Replaced:**

> after the ceiling inputs are measured (E0) and the native-Linux floors are known (E1).

**With:**

> after the ceiling inputs are measured (E0) and the platform floors are known (E1: WSL2 and Windows-native; native Linux is held) [sync R1].

# Edit 4. Segment A, edit 4. Marks: [sync 1] [sync 10] [sync R1] [sync R4]

**Replaced:**

> This is a plan only; no code is written yet.

**With:**

> This is a plan only; no code is written yet. **Synchronization note (2026-10-24):** every mark in square brackets below is a queued edit applied in place. `[sync 1]` to `[sync 10]` are the ten mandatory edits (research update 5.1 = trainer master 16.7 = trainer notes 3.4); `[sync R1]` to `[sync R4]` apply author rulings of 2026-10-24 (R1 draft built for WSL2 and Windows-native, native Linux held; R2 llama.cpp is a read-only test oracle and never part of the runtime; R3 the 10.7 GiB ceiling; R4 the 07:39 decisions that touch the engine). The engine stays HELD: no file in the pelias-engine repository is changed by this document.

# Edit 5. Segment B, edit 1. Marks: [sync R1]

**Replaced:**

> - **Today:** the llama.cpp-based tier decodes at 44 to 54 tok/s on short prompts and 43 to 49 tok/s at 12k tokens, over all launches and load modes [MEASURED, WSL2]. Gate G3 asks for at least 55.0 tok/s at 12k.

**With:**

> - **Today:** the llama.cpp-based tier decodes at 44 to 54 tok/s on short prompts and 43 to 49 tok/s at 12k tokens, over all launches and load modes [MEASURED, WSL2]. Gate G3 asks for at least 55.0 tok/s at 12k. The six recorded runs that anchor the gap analysis are tabulated in Part 3 of this document (WSL2 Cell A 37.9 short / 42.1 at 12k; Windows-native Cell A 47.3 and 46.8 short, 53.7 and 54.1 at 12k; WSL2 best cell 53.5 / 53.0 / 49.0 at short / 4k / 12k) [sync R1]. All of these runs were made on the patched llama.cpp tier build: no run of a non-llama server exists yet, so the cause of the gap to the ceiling is not isolated [UNVERIFIED].

# Edit 6. Segment B, edit 2. Marks: (no mark: wording kept consistent with a marked edit)

**Replaced:**

> CUDA 13.0.88 [MEASURED, HWiNFO report and nvcc].

**With:**

> CUDA 13.0.88 on WSL2 and 13.4.92 on Windows-native [MEASURED, HWiNFO report and nvcc].

# Edit 7. Segment B, edit 3. Marks: [sync R1]

**Replaced:**

> - **Target system:** native Linux first. The same binary also runs on WSL2 (see the compatibility section).

**With:**

> - **Target system [sync R1]:** WSL2 and Windows-native, built as two versions of one code base. Author ruling of 2026-10-24 06:21 [AUTHOR-STATED]: "the draft is built as a WSL version AND a Windows version (decision 4 is answered for the draft; Linux-native is held and will be a separate entity later)". The ruling was made for the trainer draft; this synchronization applies it to the engine, and the author may correct that. Native Linux is tabled.

# Edit 8. Segment B, edit 4. Marks: [sync 1] [sync R1]

**Replaced:**

> - **Not in scope:** changing the router (a published study found every setting broke its 1% perplexity gate [PUBLISHED, arXiv 2608.18261, via the tier blueprint; not re-fetched]), training, and a Windows-native build.

**With:**

> - **Not in scope [sync 1] [sync R1]:** inference-time changes to the router. Clamping or re-ranking the router's choices on an untrained model is rejected: +0.8% to +2.0% perplexity at about 30% fewer misses, and +4.0% to +5.8% at 50% (arXiv 2608.18261 section 7.1 Table 6 [PUBLISHED, full text verified 2026-10-23]). **Trainer-side router locality (Mode L) is not out of scope**: it is an input to the engine, accepted as a gated experiment, because the same study's section 7.1 stacking shows that training cuts the rerouting perplexity cost from +7.7% to +0.9%, and ReMoE (arXiv 2605.27081) reports +26% relative adjacent-token reuse (27.3% to 34.5%) with frozen experts [PUBLISHED, verified 2026-10-23]. The benefit on this model is UNVERIFIED (probes P-L1, P-L2, P-E2, P-E3). The earlier wording, "every setting broke its 1% perplexity gate", was a summary of the inference-time result only. Also not in scope: training itself, which belongs to pelias-trainer. A Windows-native build is no longer out of scope (see Target system).

# Edit 9. Segment B, edit 5. Marks: [sync R2]

**Replaced:**

> - **Working rules:** one step at a time; CPU-side logic is tested in the sandbox; GPU runs are done by the author; no file is created without being asked.

**With:**

> - **Strictly proprietary runtime [sync R2]:** no llama.cpp, ggml, libllama, common or JSON-library code is linked, copied or required at run time (trainer master D37; author ruling 2026-10-24 08:15, reason AUTHOR-STATED: routing through llama brought in behaviour that was not accounted for). The pinned commit 2ca15f540 (b11399), built unchanged, is an **external, read-only test oracle** that lives only under `tests/oracle/`: it supplies logits, KL, top-1 and kernel-parity references and nothing else. Where the engine's bytes must equal ggml's (block layouts, quantizer arithmetic), the layout and arithmetic are re-expressed in original code after reading the reference, and the file's provenance line says so. The one third-party library in the draft is Dear ImGui (MIT), used for the interface only, and not in the engine.
> - **Working rules:** one step at a time; CPU-side logic is tested in the sandbox; GPU runs are done by the author; no file is created without being asked.

# Edit 10. Segment C, edit 1. Marks: [sync 3]

**Replaced:**

> | CPU read of cold experts | 26.6 GB/s | measured |

**With:**

> | CPU read of cold experts [sync 3] | 43.8 to 45.9 GB/s with the CPU alone (microbench 02, 2026-10-20). 26.6 GB/s (26.56) is the CPU's share while the GPU reads at the same time (21.06 GB/s over PCIe; 47.62 GB/s combined, microbench 04) | measured |

# Edit 11. Segment C, edit 2. Marks: (no mark: wording kept consistent with a marked edit)

**Replaced:**

> The gap between about 92 and the live figures is what E0 must explain before design choices are locked.

**With:**

> The gap between about 92 and the live figures is what E0 must explain before design choices are locked. Status after E0 (2026-10-22, see Sources): the timeline shows no CPU and GPU overlap and a timer-free handoff of at least 4.6 to 5.0 ms per token; the 7.4 against 7.73 ms spine question stays open; the breakdown is DERIVED, not MEASURED (reconciliation D4).

# Edit 12. Segment D, edit 1. Marks: [sync R2]

**Replaced:**

>  Oracle tests (llama.cpp 2ca15f540) sit outside the engine and check every part.

**With:**

>  Oracle tests (llama.cpp 2ca15f540: external, read-only, tests/oracle/ only) sit outside the engine and check every part. [sync R2]

# Edit 13. Segment D, edit 2. Marks: [sync R2]

**Replaced:**

> The oracle tests (llama.cpp 2ca15f540) sit outside the engine and check every part.

**With:**

> The oracle tests (llama.cpp 2ca15f540) sit outside the engine and check every part; nothing from llama.cpp is linked into, started by, or required at run time by the engine process [sync R2].

# Edit 14. Segment E, edit 1. Marks: [sync R1]

**Replaced:**

> floors/ (WSL2 and native Linux tables)

**With:**

> floors/ (WSL2 and Windows-native tables; native Linux held) [sync R1]

# Edit 15. Segment E, edit 2. Marks: [sync R1]

**Replaced:**

> feature checks (AVX2, CUDA version, WSL build flag)

**With:**

> feature checks (AVX2, CUDA version, WSL build flag, Windows build flag) [sync R1]

# Edit 16. Segment E, edit 3. Marks: [sync R1]

**Replaced:**

> mem/host_alloc_linux.cpp  host_alloc_wsl.cpp  probe.cpp  vram_plan.cpp

**With:**

> mem/host_alloc_wsl.cpp  host_alloc_win32.cpp  probe.cpp  vram_plan.cpp  (host_alloc_linux.cpp is held with native Linux) [sync R1]

# Edit 17. Segment E, edit 4. Marks: [sync R2]

**Replaced:**

> oracle/  llama.cpp 2ca15f540 harness, fixed prompts, checksums                      E3

**With:**

> oracle/  llama.cpp 2ca15f540 harness, fixed prompts, checksums: external, read-only, never linked or built into the engine [sync R2]   E3

# Edit 18. Segment E, edit 5. Marks: [sync R2]

**Replaced:**

> Existing work that moves in: microbench programs 02 to 07 become `bench/`;

**With:**

> **Proprietary-server additions to this tree (operating-system socket layer, memory-mapped GGUF reader, native tokenizer and chat-template renderer, arena, doorbell coordinator) are specified in Part 6 of this document [sync R2]; the tree above stays the engine-phase file map.** Existing work that moves in: microbench programs 02 to 07 become `bench/`;

# Edit 19. Segment E, edit 6. Marks: [sync R2]

**Replaced:**

> The patches and benchmark results already in the GitHub repo stay where they are as the reference record.

**With:**

> The patches and benchmark results already in the GitHub repo stay where they are as the reference record. [sync R2] The patches are patches on llama.cpp: they are a record and a reference for the patched oracle build, and no patched llama.cpp code enters the engine's source tree. Part 4.10 maps each one to the module that replaces what it did.

# Edit 20. Segment F, edit 1. Marks: [sync R1]

**Replaced:**

> | E1 Native Linux floors | Boot this machine into native Linux. Re-run the floor tests (02, 03, 04, 07) and the base cell. Test ReBAR. | microbench programs | A floor table for WSL2 and native Linux side by side; probe contents chosen from the differences. |

**With:**

> | E1 Platform floors [sync R1] | Re-run the floor tests (02, 03, 04, 07) and the base cell on WSL2 and on Windows-native. Native Linux floors (boot into native Linux, test ReBAR) are held with native Linux; E-A6 stays open. | microbench programs; the trainer's Wave 0 hardware probes (master 11.12) measure the same quantities on both platforms | A floor table for WSL2 and Windows-native side by side; probe contents chosen from the differences. |

# Edit 21. Segment F, edit 2. Marks: [sync R1]

**Replaced:**

> | Clean build on native Linux and WSL2; CPU tests pass in the sandbox. |

**With:**

> | Clean build on WSL2 and on Windows-native (MSVC and nvcc 13.4.92 for the Windows tree) [sync R1]; CPU tests pass in the sandbox. |

# Edit 22. Segment F, edit 3. Marks: [sync R2] [sync 4]

**Replaced:**

> | E3 Model reader and oracle | GGUF reader, tensor map, expert size table. Oracle harness against pinned llama.cpp 2ca15f540 (b11399): logits, KL, top-1. |

**With:**

> | E3 Model reader and oracle | The engine's own memory-mapped GGUF reader [sync R2] (Part 6), tensor map, expert size table. External oracle harness against pinned llama.cpp 2ca15f540 (b11399): logits, KL, top-1. [sync 4] The reader and the harness also ingest trainer-exported GGUFs (`merged-core.gguf` with `layout_manifest.json`) and evaluate the three KL gates against the file's measured float spread: K1 (trainer against llama.cpp on the same GGUF), K2 (exported GGUF against the trainer reference), K3 (engine against llama.cpp on the exported GGUF). P-Q4a reproduces 0.017353 at batch 16 and 0.018286 at batch 1024 on the current served file; P-Q4b re-measures the spread on any newly trained file. |

# Edit 23. Segment F, edit 4. Marks: [sync 4]

**Replaced:**

> | Reads the Q4_K_M file; the oracle reproduces the stock noise floor (KL 0.017353 at batch 16 vs 0.018286 at batch 1024 [MEASURED]). |

**With:**

> | Reads the Q4_K_M file; the oracle reproduces the stock noise floor (KL 0.017353 at batch 16 vs 0.018286 at batch 1024 [MEASURED]); a trainer-exported GGUF and its layout manifest parse identically, and K3 holds on it [sync 4]. |

# Edit 24. Segment F, edit 5. Marks: [sync 5]

**Replaced:**

> | 16-byte-load kernels; bank loader and slot table from the hot-set file; zero-copy read kernel for cold experts. |

**With:**

> | 16-byte-load kernels; bank loader and slot table from the hot-set file; zero-copy read kernel for cold experts. [sync 5] The slot table supports variable per-layer capacities (84 to 161 experts per layer in the 7.5 GiB water-fill) read from the `PELIAS_HOT_SET` file. `wide_load.cu` (P-E10) must handle Q6_K's 210-byte blocks (420-byte rows) by stream-aligned 16-byte (`uint4`) shared-memory staging, or consume 432-byte row-padded packs from the engine's `repack_gguf.py` if P-E10 shows padding is required. |

# Edit 25. Segment F, edit 6. Marks: [sync R2]

**Replaced:**

> | Attention, DeltaNet layers, norms, router, lm_head, q8_0 KV cache; all 40 layers; first full token path. | upstream kernels as reference |

**With:**

> | Attention, DeltaNet layers, norms, router, lm_head, q8_0 KV cache; all 40 layers; first full token path. | kernel behaviour read from llama.cpp 2ca15f540 as a specification only (read-only; the engine's kernels are its own code) [sync R2] |

# Edit 26. Segment F, edit 7. Marks: [sync R2]

**Replaced:**

> | Tokenizer, chat template, tool-call parsing, OpenAI-compatible streaming, prefix cache for recurrent state. |

**With:**

> | Tokenizer, chat template, tool-call parsing, OpenAI-compatible streaming, prefix cache for recurrent state: all the engine's own code [sync R2] (native byte-level BPE tokenizer, chat-template subset renderer, raw-socket HTTP and SSE server; specified in Part 6). |

# Edit 27. Segment F, edit 8. Marks: [sync 6] [sync R1]

**Replaced:**

> | Detect WSL2, test pinned-memory limits and bandwidth, read BAR size, choose the memory strategy. | NVIDIA WSL guide | The same binary passes the E8 to E10 gates on WSL2 and native Linux. |

**With:**

> | Detect WSL2 or Windows-native, test pinned-memory limits and bandwidth, read BAR size, choose the memory strategy. [sync 6] Also log the WSL configuration line as it is (`transparent_hugepage=always swiotlb=65536`) and `/proc/meminfo` `AnonHugePages` during model load (P-E11), to test the true cause of the 2x load-mode prefill swing (863 to 929 against 1,667 to 1,680 tok/s at 12k). | NVIDIA WSL guide | The same code base passes the E8 to E10 gates on WSL2 and on Windows-native [sync R1]. |

# Edit 28. Segment F, edit 9. Marks: [sync 7]

**Replaced:**

> | Zero-copy third branch, prefetch and small dynamic region, MTP, precision tiers, ReBAR path, LRU bank. Each only if its own test says go. |

**With:**

> | Zero-copy third branch, prefetch and small dynamic region, MTP, precision tiers, ReBAR path, LRU bank. Each only if its own test says go. [sync 7] Hot-set objective: byte-weighted hit maximization (h) against zero-miss layer share (Z_l) maximization, decided by P-E2's measurement of the fixed miss cost c0 against the marginal miss cost c1. Trainer Mode L candidates (re-profiled hot sets) enter the extras list, each judged by its own gates G-L4 and G-Z4 (measured decode against the unchanged baseline C0). |

# Edit 29. Segment F, edit 10. Marks: [sync 9]

**Replaced:**

> After E13 the lessons go back into the pelias-trainer.

**With:**

> After E13 the lessons go back into the pelias-trainer. [sync 9] This is superseded by the two-way contract in Part 5.2: each engine milestone releases a named input to the trainer now, not at the end (trainer master 16.3 X12).

# Edit 30. Segment G, edit 1. Marks: [sync R2]

**Replaced:**

> Every phase is checked against the same oracle: llama.cpp commit 2ca15f540 (b11399), built unchanged.

**With:**

> Every phase is checked against the same oracle: llama.cpp commit 2ca15f540 (b11399), built unchanged. It is an external, read-only test oracle under `tests/oracle/`; the engine never links or starts it [sync R2].

# Edit 31. Segment G, edit 2. Marks: [sync 4]

**Replaced:**

> with mean KL and top-1 agreement. Stock noise floor:

**With:**

> with mean KL and top-1 agreement, run as K1, K2 and K3 when the file is trainer-exported [sync 4]. Stock noise floor:

# Edit 32. Segment H, edit 1. Marks: [sync R1]

**Replaced:**

> Linux first, WSL2 compatible

**With:**

> WSL2 and Windows-native first; native Linux held [sync R1]

# Edit 33. Segment H, edit 2. Marks: [sync R1]

**Replaced:**

> One code base and one binary serve both. WSL2 is Linux, so a Linux engine runs there; what differs is how the GPU and memory are reached.

**With:**

> One code base builds two versions, a WSL2 build and a Windows-native build [sync R1]. What differs is how the GPU and memory are reached, and the operating-system calls (POSIX `sys/socket.h` and `mmap` against Winsock2 and `CreateFileMapping`/`MapViewOfFile`) which live in one thin layer (Part 6). Native Linux is held and will be a separate entity later (author, 2026-10-24 06:21).

# Edit 34. Segment H, edit 3. Marks: [sync R1]

**Replaced:**

> - **What may differ on native Linux, all unmeasured there:**

**With:**

> - **Windows-native [sync R1]:** the recorded Windows-native decode of the tier build is 25% (short) and 28% (12k) faster than the WSL cell, a difference of about 5.2 ms per token (two Windows launches against one WSL launch; not yet a claim: three launches per side are needed; ledger section 9.4, trainer master 16.3 X9). The cause is not isolated; P-H11 and E1 test it.
> - **What may differ on native Linux, all unmeasured there (held):**

# Edit 35. Segment I, edit 1. Marks: [sync R1]

**Replaced:**

> | Handoff on native Linux is not lower than on WSL2 |

**With:**

> | Handoff on Windows-native or native Linux is not lower than on WSL2 (native Linux held) [sync R1] |

# Edit 36. Segment I, edit 2. Marks: [sync R2]

**Replaced:**

> | Start with upstream kernels as reference behind a clean interface and replace them one at a time. |

**With:**

> | Start with the engine's own slow reference kernels (the sequential fp32 DeltaNet recurrence and the blocked attention kernel, as in the trainer's `kern/` plan) behind a clean interface and replace them one at a time with the fast ones; llama.cpp's kernels are a read-only specification and a test oracle, never a dependency [sync R2]. |

# Edit 37. Segment I, edit 3. Marks: [sync R2]

**Replaced:**

> | E8 does not beat the llama.cpp tier result | Stop the from-scratch track at that point and fold the findings back into a llama.cpp patch series. |

**With:**

> | E8 does not beat the llama.cpp tier result | [sync R2] The original kill criterion, "fold the findings back into a llama.cpp patch series", is removed: the author ruled ground-up on 2026-10-24 (D37). At that gate, log the per-token timers against the E0 breakdown and decide with the author whether to continue. The llama.cpp tier remains the interim measuring build and the reference record. **Open for the author:** whether to restore the original fallback. |

# Edit 38. Segment J, edit 1. Marks: [sync R3]

**Replaced:**

> | VRAM ceiling: 10.4, 10.9 or 11.2 GiB (RULED 2026-10-22: 10.9 GiB) |

**With:**

> | VRAM ceiling: 10.4, 10.9 or 11.2 GiB (RULED 2026-10-22: 10.9 GiB; **SUPERSEDED [sync R3] by 10.7 GiB**, author ruling 2026-10-23 (trainer master D26), after P-H8 measured 10.83 GiB (WSL2) and 10.85 GiB (Windows-native) of CUDA free memory once a context exists; ruled 2026-10-24 (D28) that the number is measured at start: the smaller of CUDA free and NVML free after the context, minus a margin, with 10.7 GiB staying this machine's value until the margin is measured) |

# Edit 39. Segment J, edit 2. Marks: [sync R2: "carry" means kept as the reference record and as the patched oracle build; no patched llama.cpp code enters the engine]

**Replaced:**

> (RULED 2026-10-22: carry mm_ids_dedup, quantize_signed_id, cuda_hole_fix and pin_cold_to_cpu; drop mmq_dump_diag and mmq_dump_diag2)

**With:**

> (RULED 2026-10-22: carry mm_ids_dedup, quantize_signed_id, cuda_hole_fix and pin_cold_to_cpu; drop mmq_dump_diag and mmq_dump_diag2) [sync R2: "carry" means kept as the reference record and as the patched oracle build; no patched llama.cpp code enters the engine]

# Edit 40. Segment J, edit 3. Marks: [sync 2] [sync R4] [sync R4] [sync R1] [sync 7] [sync R3] [sync R2]

**Replaced:**

> | Repository: new folders in <account>/pelias-engine, or a separate repo | Fixes the scaffolding tree and where the existing patches and results live | E2 |

**With:**

> | Repository: new folders in <account>/pelias-engine, or a separate repo | Fixes the scaffolding tree and where the existing patches and results live | E2 |
> | Is the 7.5 GiB bank a ruling or a derived value? May the bank be a non-multiple of 0.5 GiB? [sync 2] | The core's room is set by it; at the default micro-batch the free room would hold roughly 7.8 GiB if the slope is 1:1 [ESTIMATE, reconciliation D7]; 8.0 GiB does not fit | E5 |
> | Which prefill micro-batch does the engine ship with? [sync R4] | RULED 2026-10-24 07:39 (trainer D31, D34): an engine setting; the trainer no longer depends on it. The engine's own peak still moves with it: 0.26 to 0.30 GiB of headroom at the default and 0.01 to 0.03 GiB at `-ub 2048` (whole-card reading, 10.7 GiB ceiling) | E9 |
> | Does the engine add Q8_0 weights or kernels? [sync R4] | RULED 2026-10-24 07:39 (trainer D32): not needed; no Q8_0 or Q6_K core inflation by default. Stays an optional Advanced/Developer override, usable only if the engine supports it (P-E5) | E7 |
> | Is Windows-native promoted from "not in scope"? [sync R1] | Answered for the draft by the author's ruling of 2026-10-24 06:21 (WSL2 and Windows-native both built); three launches per side are needed before a speed claim | E1 |
> | Is Mode L / L-Z worth building after P-L1, P-L2, P-E2 and P-E3 report? [sync 7] | Decides whether any router training is built and whether the extras list in E12 gets Mode L candidates | E12 |
> | What is the 10.7 GiB ceiling a ceiling of? [sync R3] | RULED 2026-10-24 for a probe-measured, process-only reading (the whole-card figures stay as the stricter planning figures until P-E4 re-measures); the margin is set by the disturbance test | E5 |
> | Does "ground up" apply to the engine, and does the kill criterion at E8 (fallback to a llama.cpp patch series) stay? [sync R2] | The product ruling (D37) is ground-up; the engine is HELD and the author has not yet confirmed the engine part | E2 |

# Edit 41. Segment K, edit 1. Marks: [sync 8] [sync R1] [sync 2] [sync R2]

**Replaced:**

> | Thinking on or off | May change results and run time a lot | Revisit after the main sweep |

**With:**

> | Thinking on or off | May change results and run time a lot | Revisit after the main sweep |
> | E-A1 [sync 8]: thread count and pinning, 8 physical cores (one thread each) against 16 SMT threads | The ledger shows 5 to 7 threads beating 10 on stock; Zen 3 splits the load queue between SMT threads (audited hypothesis from the outside report, UNVERIFIED) | A/B in E4, with its own go/no-go |
> | E-A2: acquire/release against seq_cst on the handoff flags | Atomic ordering cost on the doorbell (audited hypothesis, UNVERIFIED) | A/B in E6 |
> | E-A3: `PREFETCHNTA` on cold reads | Possible cold-read gain on the CPU path (audited hypothesis, UNVERIFIED) | A/B in E4 |
> | E-A4: L2 persisting window (`cudaAccessPropertyPersisting`) | Possible gain for the spine or hot tensors (audited hypothesis, UNVERIFIED) | Query `cudaDevAttrMaxPersistingL2CacheSize` first, then A/B in E5 to E8 |
> | E-A5: `AnonHugePages` per load mode | The engine's open load-mode unknown (the same item as the row above, carried here under its A/B name) | E11, P-E11 |
> | E-A6: native Linux floors | The engine's E1 as first written; held with native Linux [sync R1] | Held |
> | E-A7: pinned memory above 20.4 GiB | The engine had proven 20,448 MiB, the largest size it tried. The trainer's P-H7 held 32 GiB of `cudaHostAlloc` memory on WSL2 and on Windows-native (MEASURED 2026-10-23/24, trainer master 16.3 X13), so the 20.4 GiB figure was not a limit on this machine; the limit above 32 GiB is not found | Engine-side re-check in E11 |
> | P-E4 at `-ub 4096` [sync 2] | The 7.5 GiB bank was never measured at `-ub 4096`: the stamps 130854 and 132648 ran at the default micro-batch, whose value the files do not record (ledger 8.2; reconciliation D2). By the 1:1 slope seen at `-ub 2048` it would peak near 11.2 GiB [ESTIMATE], over the ceiling | P-E4 adds this case |
> | P-H13 on Windows-native [reconciliation 7.5] | Display-driver timeout on long kernels or graphs; the card also drives the display | Trainer's P-H13 (done on both platforms, master 11.12); engine re-check at E6 |
> | Behaviour llama.cpp contributed that the ground-up engine must account for [sync R2] | The author's reason for ground-up: routing through llama brought in behaviour that was not accounted for. One candidate: the unused MTP block `blk.40` that stock llama-server skips with an "unused tensor" warning at each start; whether it explains why the VRAM ceiling was not reached is UNVERIFIED | Compare VRAM and speed with MTP off and on; the per-token timers of E0; the llama-server catalogue (Part 6.14) |

# Edit 42. Segment L, edit 1. Marks: [sync 10]

**Replaced:**

> - Numbers from the running P1 sweep (C to N, J, B, O) are not in this document yet; the E0 results replace the assumed inputs when they arrive.

**With:**

> - Numbers from the running P1 sweep (C to N, J, B, O) are not in this document yet; the E0 results replace the assumed inputs when they arrive.
> - [sync 10] Added by the synchronization of 2026-10-24: `benchmarks/RESULTS.md` sections 1 to 9.4 (Part 3.1); `benchmarks/raw/e0-2026-10-22/`; `benchmarks/raw/windows-native-2026-10-23/`; the full-text verification of arXiv 2608.18261 and arXiv 2605.27081 (2026-10-23); and `pelias_engine_hardware_verified_research_plan.md` (2026-10-23), marked as an audited hypothesis list and not as a primary measurement source; the trainer master (16.3 to 16.7 and 11.12) and notes (3.4); the reconciliation of the research update (D1 to D11, A1 to A8, N1 to N5; Part 1.6); the author's rulings of 2026-10-24 (R1 to R4).
