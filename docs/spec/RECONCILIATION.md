# Reconciliation of the outside research update with the trainer notes (D1 to D11, A1 to A8, N1 to N5)

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Part 1.6 (master lines 444 to 553). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 1.6 Reconciliation of the research update with the trainer master: verdict, map, corrections D1-D11, adopted suggestions A1-A8, gaps N1-N5, WSL2 points

<!-- SPLICE id="P1.6" src="RESEARCH_UPDATE_RECONCILIATION_TRAINER.md (2026-10-23)" lines="1-103" shift="+3" ws="0" edited="0" -->
> **Source (verbatim):** RESEARCH_UPDATE_RECONCILIATION_TRAINER.md (2026-10-23), lines 1-103. Binding corrections: where this section and the research update disagree, this section wins. Its sections 7 to 9 are placed in Parts 5 and 7. Heading levels shifted by +3 for this document only.

### Research update (2026-10-23) reconciled with the pelias-trainer master

Working copy, 2026-10-23. Compared: `16d2a764-pelias-research-update.md` (870 lines, 109,635 B) against the trainer master (2,357 lines, repo head `61db12b`), the trainer master notes (216 lines), the engine ledger and raw files (engine head `34f744f`, read only), and the trainer repo README. Nothing was run on the GPU, nothing was pushed, no repo file was changed.

Labels as in the master: MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION. Every number below that I computed myself was computed by `recheck_update.py` (kept with this file).

---

#### 1. Verdict

1. The update is mostly our own notes and master restated. Its F1–F17 are our notes 1.1–1.17 under new numbers; its G1–G18 table has the same 18 verdicts as master 16.4; its 10 engine edits, milestone contract, dependency chain and six open decisions are master 16.7 and notes 2.1–3.1; its probe table is master Chapter 11. Its section 7 (one unified C++20/CUDA program) is the only large piece that is new.
2. Its arithmetic holds. I recomputed 86 figures from their stated inputs (dual-bus sum, stream time and the three hiding thresholds, expert bytes to the byte, Q6_K padding, core sizes, headroom, LoRA state, Mode L cost, FLOP shares, peak-rate scaling, all eight gap rows, the Windows–WSL deltas). One check failed, on purpose: its "7.90 GiB layout" parenthetical sums to 4.59 GiB (see D8).
3. It differs from our data in 11 places (section 3). Our data is used in every case. None of them changes a trainer design decision; two of them (D2, D4) would have put wrong labels or wrong settings into the engine blueprint if copied over.
4. It gives the trainer no new probe category. Everything it asks for that touches the trainer is already a Wave 0 probe in master 11.2–11.10. What it does change: it confirms that the next step is the Wave 0 platform and transfer probes (11.10 steps 1–2), and I found four gaps while checking that neither the update nor the master covers, plus one priority note (section 5: N1, N3, N4, N5 are gaps; N2 is already P-H3 and only needs to be treated as the first number to get).
5. Everything that is engine-only or unified-only is parked in sections 7 and 8 and not started, as you asked.

---

#### 2. Map: update section, where we already have it, status

| Update section | Our source | Status |
|---|---|---|
| Header table (hardware, ceiling, pins, labels) | master 2.1, 16.3, Appendix D | Matches. New label `AUTHOR-STATED` proposed (A2). Its label census (231) is not adopted (D3) |
| 1 Core discovery (dual bus, doorbell, stream time, thresholds) | master 1, 5.4–5.5; notes 1.7, 1.15 | Matches (recomputed) |
| 2.1 Retire "57.0" | notes 1.15, master 16.4.1 | Agrees. Candidate origin found in raw files (A3) |
| 2.2 E0 sub-timings | engine `raw/e0-2026-10-22/README.md` section 2 | Numbers match the ledger's derived decomposition. Label is wrong, two figures not found (D4) |
| 2.3 Claims G1–G18 | master 16.4 | Same 18 verdicts. Its E-A1…E-A7 list is the same as master 16.6 (A4) |
| 3.1 F1–F17 | notes 1.1–1.17 | Same content. LoRA module breakdown differs (D1) |
| 3.2 Script and artifact inventory | engine tools, repo README | Matches (baseline package sizes and v0.2.1 run figures are in the repo README and master) |
| 4 Engine cross-check | master 16.4.1, engine ledger 8.2, 9.2, 9.4 | All eight gap rows recomputed, match |
| 5.1 Ten engine blueprint edits | master 16.7 | Identical list. Parked (section 7) |
| 5.2 Dependency graph, invalidation matrix | notes 2.1, 2.2 | Matches; three rows to add to notes 2.2 (A1) |
| 5.3 Milestone contract | master 16.7 | Matches |
| 5.4 Probe registry | master 11 | Matches; P-E13 status differs (D9) |
| 5.5 Six open decisions | master 16.7, notes 3.1 | Same six. It answers #1 for the author (D7) |
| 6 Engine plain-English README | none yet | Parked (section 7) |
| 7 Unified architecture, universal substrate | master 5.1–5.2 (layout rule only) | New. Parked, with seam work for the trainer (A5, A6, A7) |
| 8 Citations | master source list | Every source I checked is already there (ESFT 2407.01906, 2609.04895, 2607.08780, 2601.14277, 2312.03732, 2408.11743, 2411.09009, 2412.06464, 2406.06484, Higham) |

---

#### 3. Where the update disagrees with our data (our data used)

| # | Update says | Our data | Evidence | Effect |
|---|---|---|---|---|
| D1 | LoRA 310 = 30×3 DeltaNet projections + 10×4 attention + 40×3 shared expert + 30×2 "DeltaNet gates/norms" | 10×4 attention + 30×5 DeltaNet + 40×3 shared expert = 310 | READ-SRC from the v0.2 source (master 2.3, 7.1). Norms take no LoRA. Same total, different parts | None on sizes (21.17 M, 323 MiB recomputed). Keep ours |
| D2 | "Default prefill micro-batch (`-ub 4096`)": stamps 130854 and 132648 peak 10.40–10.44 GiB; so all-Q6_K core (+282 MiB) "fits at `-ub 4096`" | Those stamps ran at the default micro-batch, value not written in the files (ledger 8.2: "inferred from the rate"). The 7.0 GiB reference cell at `-ub 4096` already peaks 10.7 GiB | Ledger 8.2 (table row 7.5 GiB / stamp 130854; the note on the two further launches); master 2.7, 1.6 | 7.5 GiB at `-ub 4096` is unmeasured. By the 1:1 bank-to-peak slope seen at `-ub 2048` (7.5 → 8.0 GiB: 10.69 → 11.19) it would be about 11.2 GiB [ESTIMATE], over the ceiling. Add `-ub 4096` as a case of P-E4 |
| D3 | 231 labelled statements: 24 / 42 / 33 / 72 / 23 / 19 / 18 | Token counts in the master now: 24 / 42 / 33 / 74 / 21 / 19 / 17 = 230 | `grep -o` per label | Not adopted. Tokens are not statements. Fix: a census script that counts per table row (N5) |
| D4 | E0 split 10.1 / 6.6 / 5.1 / 2.8 ms = 24.6 ms, "MEASURED"; zero overlap "MEASURED"; "2.75 ms host CUDA split (graphs on) vs 8.04 ms (off)" | Ledger labels the breakdown DERIVED. Parts are cell-B (timer on) medians with the timer cost taken out of sync-and-copy (so 5.0 is a lower bound; 4.6 at 12k). 2.8 = copy back 2.3 + launch 0.5. Zero overlap is derived from the parts summing to wall time (30.5 vs 30.3 ms) | `raw/e0-2026-10-22/README.md` section 2; ledger 9.2 | Label fixed to DERIVED when copied anywhere. 2.75 and 8.04 not found in the ledger or summaries: not adopted until found |
| D5 | The 82 splits per token (41 CPU + 41 GPU) "include the 41st block `blk.40` (MTP)", READ-SRC / MEASURED | The 82 is measured. The server log says "hot bank active: 40 layers" and the last attach line is layer 39. What the 41st CPU split is, is not known | E0 README; `P_bank_size_bypass9_b85` log; `server_N10.log` | UNVERIFIED. P-E2's per-layer timers (by layer index) settle it. Matters later for the MTP decision (P-T8) |
| D6 | Windows–WSL gain of 5.1–5.2 ms "matches" the 4.6–5.1 ms WSL sync-and-copy cost | The ledger itself says "consistent, not proven" (9.4). The update's own G17 caps the VMBus part at 19–27 % of the gap | Ledger 9.4 | Stays a hypothesis; P-H11 tests it |
| D7 | Decision 1 (7.5 GiB bank) "treated as DERIVED (largest 0.5 GiB step under the ceiling)" | Open author decision. The arithmetic is true but the bank need not be a multiple of 0.5 GiB: at the default micro-batch the 0.36 GiB of free room would hold roughly 7.8 GiB if the slope is 1:1 [ESTIMATE] | Ledger 8.2 | Stays open for the author (master 16.7 item 1). See N4: the room has a price |
| D8 | Trainer VRAM layout "7.90 GiB (2×498 MiB slots + 0.80 spine + 0.32 LoRA/Adam + 2.50 working set)" | Master 5.3 has 13 rows that sum to 7.90 (context 0.60, core 0.80, lm_head 0.39, embedding 0.27, routers 0.09, slots 0.97, LoRA 0.32 + 0.04, boundaries 0.75, working set 2.50, CE chunk 0.65, tape 0.02, margin 0.50) | Recomputed: the update's four items sum to 4.59 | Update's line is an incomplete summary, not a different design |
| D9 | P-E13 "DONE (READ-SRC)" | llama.cpp side read (notes 1.14); engine side open because the engine kernels are not written. Master row has no "done" marker | master 11 | Keep ours: partly done |
| D10 | "`-ub 2048`, the setting that achieves 1,667–1,680 prompt tok/s at 12k" | 1,680.3 is stamp 132907 (`--load-mode none -ub 2048`). The best-cell launch 141137 (default load, `-ub 2048`) read 920.75 at 12k. The load mode, not `-ub`, is the 2× factor | Ledger 8.2 lines 173, 243 | Wording only; G4 / P-E11 keep the cause open |
| D11 | 8.5 GiB bank "median 53.7 (55.7 short), peak 11.4 GiB" | Two different launches: stamp 131550 (55.7 short, peak 11.4, ledger 8.2) and the P1 sweep run `b85` (8,631 MiB bank, 4k: 53.7, range 51.4–57.0, peak 11.28) | Ledger 8.2 line 239; sweep log | Not a conflict, but they are two runs |

---

#### 4. Suggestions adopted or adjusted (every one kept)

| # | From the update | Adjustment to our material |
|---|---|---|
| A1 | Invalidation matrix has rows we lack | Add three rows to notes 2.2: P-E2 result → hot-set objective (hit vs zero-miss share) and Mode L layer weights; P-E10 result → ceiling cold-read term, row padding in the engine pack; target OS choice → handoff floor, pinned cap, trainer platform default (P-H11) |
| A2 | Label `AUTHOR-STATED` | Adopt as the 8th label. The master already writes "author-stated / DECISION" in places (OS row, ceiling); the engine ReBAR brief also uses it |
| A3 | Origin of "57.0" | Found in two raw files, neither Windows: `P_bank_size_bypass9_b85_20261022_191010.log` (4k context, WSL2, median 53.7, range 51.4–57.0, 3/3) and a 2026-10-21 speed file (prompt-read rate 57.0 on a 61-token prompt). Notes 1.15 should say this instead of "the author thinks it was mistyped". Whether the outside report took it from one of these is not provable |
| A4 | E-A1…E-A7 engine A/B list | No change: master 16.6 already carries all seven with the same meanings (E-A5 AnonHugePages by load mode, E-A6 native-Linux floors, E-A7 pinned above 20 GiB) |
| A5 | One program, shared libraries | Trainer T0 build rule (proposed D26, "seams"): one static library per top-level directory with declared link dependencies, `pelias::` namespaces, and a CMake check that `hal`, `io`, `quant`, `kern`, `verify` include nothing from `model`, `sched`, `train`, `export`. Then lifting them into a shared core later is a move, not a rewrite. Master 5.1 already has the `kern` → `hal` rule; this extends it |
| A6 | "K1 = E7 by construction" | True only for code that is genuinely shared: norms, RoPE, DeltaNet recurrence, router, quantizers, GGUF I/O. The weight matmuls differ by role (training: dequantize to bf16 tensor-core path; serving: int8 `mmq` with `q8_1` activations). So K1 and K3 stay as gates, and Mode B (D24) is how the two matmuls are brought together. Record as "levels of sharing": L0 types, L1 quantizers, L2 non-matmul spine, L3 matmul (not shared) |
| A7 | Universal substrate beyond LLMs | Derivation route, for the universal step: any workload that has (a) a state larger than VRAM, (b) a skewed access trace, (c) per-step compute it can overlap, reduces to the same three measurements we already take: access trace → hit-rate-versus-bank curve (the engine's `collect_trace.py` / `export_hot_set.py` logic), bus constants (`probe_constants.json`), and the stream-hiding threshold N_hide. The benefit bound is 1 / (h/504 + (1−h)/47.6) per byte read, so a workload qualifies when its measured hit curve beats the break-even for its compute-to-byte ratio. No GNN, splatting or PDE claim is made until a trace exists |
| A8 | `pelias probe` writing constants and a machine-state hash | Already master 5.2 (`plan/probe_constants`, `roofline_solver`) and 11.11 (`machine_state_sha`). Keep; make the engine's microbenchmarks write the same schema when the programs are unified |

---

#### 5. Gaps I found while checking (not in the update, not in the master)

| # | Gap | Evidence | Proposed handling |
|---|---|---|---|
| N1 | Driver profile settings are missing from master 2.1 and from the machine-state record: NVIDIA Profile Inspector has **CUDA Force P2 State off**, **power management: maximum performance**, **prefer no system-memory fallback**. Also `.wslconfig`: `memory=48GB`, `processors=16`, `autoMemoryReclaim=disabled`, `swiotlb=65536` | Engine `raw/e0-2026-10-22/machine_state/MACHINE_STATE.md`. The Force P2 setting is consistent with the memory clock reading 11501 MHz (nvidia-smi paste). The maximum-performance mode would also explain why the card read 2490 MHz at low load [inference, not measured] | Add to master 2.1 (with the HWiNFO and Afterburner facts, once you say yes) and to every result file's machine-state block. Effect of "no fallback": going over the ceiling gives an allocation error, not a silent spill to system RAM |
| N2 | The sustained clock and power under tensor load are unknown. The 2490 MHz paste is a low-load reading, not a load reading; the curve point is 2779 MHz at 965 mV; power limit 100. So the 65.4 TFLOPS / 261 TOPS rows are clock-scaled ceilings | nvidia-smi paste, Afterburner screenshots | Already P-H3 (≥ 60 s per type, NVML clock / power / throttle log, flag if sustained < 85 % of burst). Make its output the speed-target base. It is the single most valuable measurement for the speed story |
| N3 | No row for the Windows display-driver timeout (TDR) on long kernels or graphs. The 5800X3D has no integrated graphics, so the 4070 also drives the display | Master has `Xid` scanning (P-H4) and a stop-condition watchdog (15.8) but nothing on TDR | New probe P-H13: one spinning kernel of 2, 5, 30 s, and one graph of long nodes, with the desktop active, on WSL2 and Windows-native. Expected to pass (compute preemption); unverified. If it fails: no kernel or graph over about 1 s, which the plan already respects (per-layer windows are milliseconds) |
| N4 | Free VRAM room has a price. Rule of thumb in the ledger: about +1 tok/s per +0.01 hit; bank sweep slope 0.038–0.046 hit per GiB. So 1 GiB of bank ≈ 3.8–4.6 tok/s, and the all-Q6_K core's +282 MiB ≈ 1.0–1.3 tok/s (about 2–2.4 % of 53.5), the same size as the 2 % decode-regression gate in P-X5 | Ledger 8.2 rows 7.5 / 8.5 GiB (52.4 → 55.7 tok/s, hit 0.827 → 0.865); master Appendix E | Adjust D22: choose the smallest byte increase that passes the quality gates, not the largest that fits, and print the price next to every candidate. If the bank is fixed (decision 1), bytes under the ceiling are free |
| N5 | No script counts labels per statement, so no label total can be quoted | D3 | Add a census mode to `check_master.py` (count per table row and per bullet; output per chapter) |

---

#### 6. WSL2 points the update raises, and where each stands for the trainer

| Point | Trainer exposure | Covered by | Open |
|---|---|---|---|
| 4-KB system-memory pages under WDDM (G2) | The 18.65 GiB expert store and the boundary store are pinned, 4-KB mapped; the stream is sequential DMA, not scattered reads | P-H5 (by memory type), P-E11 | None new. The checksum kernel already reads those pages at 26.2 GB/s |
| Transparent huge pages (G4) | Applies to pageable staging, not to `cudaHostAlloc` stores | P-E11, P-M4 | Load-mode cause stays open (engine) |
| `/dev/dxg` call cost, 94 µs sync (G17) | Trainer rule: enqueue only, host waits once per step (master 5.6). Launch cost assumed 5–10 µs | Master 6.9, P-H9 | Add one sync-cost line to P-H9 so the 94 µs is re-measured on the trainer's own build |
| Pinned-memory cap (G15) | The plan needs 24–29 GiB pinned; 20.4 GiB is the proven maximum. **This is the largest WSL2 risk to the trainer** | P-H7, P-M3, fallbacks in 5.3 | First script to run |
| Host copy and compute overlap | The 1.49 s stream is hidden only if copies overlap compute and each other | P-H12, P-H6 | Run with P-H7 |
| Windows-native build (G5, G18, 5.1–5.2 ms decode gain) | Design target, untested for the trainer | P-H11, two build trees (master 5.2) | Needs MSVC + nvcc 13.4 programs; the engine wants a third launch per side |
| Driver / `.wslconfig` settings | See N1 | machine-state record | Add to master 2.1 |
| Display-driver timeout | See N3 | none yet | P-H13 |
| BAR1 reading 16,355 MiB used with nothing running (unidentified) | Could eat VRAM headroom | P-H8 (free VRAM at start) | Refuse-to-start rule already there (free ≥ 11.2 GiB) |
| ReBAR CPU-to-VRAM route | Parked: guest has `/dev/dxg` only, no `/dev/nvidia*`, no GDRCopy | engine ReBAR brief (parked 2026-10-22) | Stays parked |

---

<!-- /SPLICE P1.6 sha256="fda4023a109338294b9d6bff1218e2dde7e69c811b97f964ffda01ce21abab24" -->
