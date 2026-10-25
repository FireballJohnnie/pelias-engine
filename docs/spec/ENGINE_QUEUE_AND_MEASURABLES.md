# Queued engine items, what is needed, the open author decisions, measurement registers, and the engine hold of 2026-10-24

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Parts 5.3, 7.13, 7.14, 7.15 and 8.10 (master lines 2814 to 2838, 4116 to 4292 and 5086 to 5125). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 5.3 Queued engine-side and unified-program items from the reconciliation (parked until the author lifts the hold)

<!-- SPLICE id="P5.3" src="RESEARCH_UPDATE_RECONCILIATION_TRAINER.md (2026-10-23)" lines="104-121" shift="+2" ws="0" edited="0" -->
> **Source (verbatim):** RESEARCH_UPDATE_RECONCILIATION_TRAINER.md (2026-10-23), lines 104-121. Heading levels shifted by +2 for this document only.

### 7. Engine-side items (parked until the trainer is top tier)

1. Blueprint edits: the 10 in the update are the 10 in master 16.7. Two corrections before anyone applies them: edit 2 must not say "`-ub 4096` default" (D2), and edit 6 should quote the `.wslconfig` line as it is (`transparent_hugepage=always swiotlb=65536`).
2. Cross-check file: use the E0 breakdown with the DERIVED label and the "at least 5.0 | 4.6 ms" bound (D4); state the 41-CPU-split question as open (D5); add the `57.0` origin note (A3).
3. Engine master notes: reuse notes 1.1–1.17 and the dependency tables; add A1 rows.
4. Engine plain-English README (the update's section 6 draft): fix before use — the 13.3 µs doorbell is a microbenchmark, not a feature of the running engine; the MTP speed sentence sits under the Windows bullet; both need the labels the repo rules require.
5. Engine unknowns to add: the `-ub 4096` case of P-E4 (D2), P-H13 on Windows-native. (E-A1…E-A7 are already in master 16.6.)
6. Engine decision to flag to the author: whether the bank may be a non-multiple of 0.5 GiB (D7, N4).

### 8. Unified-program items (parked)

1. Layers (substrate, core, engine, trainer) and the one-binary subcommands (`probe`, `verify`, `trace`, `train`, `export`) from the update's section 7. Accept as the target shape; the sharing levels in A6 decide what really merges.
2. In-process switch from a trainer arena to an engine arena for P-X5 and K3: sound if both use the same governor and allocator (the seam rule A5 keeps that possible).
3. Non-LLM workloads: handled by the derivation route in A7. No claim now.
4. Mode B becomes cheap once the CPU `Q8_K` and GPU `q8_1` quantizers live in one library (P-E6 produces them for the trainer first).

---

<!-- /SPLICE P5.3 sha256="a4adf026d2bee996b0d0ddb2120e108720da480783c8c6046662d5a66106e8da" -->


## 7.13 What is needed: decisions, measurements, outside information, work not done, weak points (trainer notes section 3)

<!-- SPLICE id="P7.13" src="PELIAS_TRAINER_MASTER_NOTES.md (trainer notes, local head 953c9b1)" lines="197-267" shift="+2" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_MASTER_NOTES.md (trainer notes, local head 953c9b1), lines 197-267. Heading levels shifted by +2 for this document only.

### 3. What is needed

#### 3.1 Decisions needed from the author (master 16.7)

| # | Decision | Why it matters | What it changes |
|---|---|---|---|
| 1 | Is the 7.5 GiB bank a ruling or a derived value? | The core's room (1.6) is set by it; 8.0 GiB does not fit | Core types, export size |
| 2 | **Ruled 2026-10-24 (07:39):** which prefill micro-batch ships? | It is an engine setting and the trainer no longer depends on it: the default export adds +0 bytes (Mode Q) and the trainer's own micro-batch comes from the planner. The engine's peak still moves with it (0.16 GiB free at the default, none at `-ub 2048`, whole-card reading) | Only which classes the optional override could raise |
| 3 | **Ruled 2026-10-24 (07:39):** will the engine add `Q8_0` kernels? | Not needed: no Q8_0 or Q6_K core by default | Q8_0 stays an Advanced/Developer override, usable only if the engine supports it |
| 4 | Promote Windows-native from "not in scope"? | It is 5.1–5.2 ms per token faster at decode (1.15) | Which platform the trainer's exports are tuned for |
| 5 | Is Mode L / L-Z worth building after the Wave 0 probes? | Benefit is unverified and the evidence is mixed (1.9, 1.10) | Whether any router training is built |
| 6 | Is there a Windows run newer than the repo's? | The gap table uses the recorded runs only | 1.15 |
| 7 | What is the 10.7 GiB ceiling a ceiling of? **Ruled 2026-10-24 by the author: it comes from the probe results at start** (process-only reading; the follow-on that keeps 10.7 for this machine until the margin is measured and keeps the whole-card figures as the stricter planning figures is a proposal, not yet agreed) | Whole card: 0.16 GiB / none for the core. Process only: about 1.2 / 0.9 GiB and the 8-bit core would fit (1.18, master 16.3.1) | Core types, export size, and whether the engine's bank is re-examined |
| 8 | What does a blank answer in the parameter sheet do? | The author's list makes code answers from hand-written templates; a description alone makes only knowledge rows | Knowledge-only rows (flagged), or a base-model draft the user reviews before it trains |
| 9 | May Expert mode switch off the clean-stop guard, or only change its limits? | The guard writes an emergency checkpoint when VRAM, RAM or swap go wrong | Whether a forced run can lose work. Proposed: limits only |

#### 3.2 Measurements needed (each has a probe in master Chapter 11; Wave 0 needs no trainer code)

**Speed foundation** — without these the speed story is arithmetic on assumed efficiency:
* P-H3 real tensor rates and clocks; P-H8 free VRAM at start (both done 2026-10-23/24, master 11.12).
* P-K1 efficiency of the bf16 path; P-K2 efficiency of the int8 path; P-K3 whether in-register dequantization is memory-bound or compute-bound; P-Q10 FP8 behaviour. These four run as one batch (Batch C, master D35).

**Accuracy foundation** — without these a "close to llama.cpp" claim cannot be made:
* P-E6 is our port of the ggml quantizers bit-exact for `Q4_K`, `Q6_K`, `Q8_0`?
* P-E7 do the served weights sit exactly on their grid?
* P-E8 does Mode Q close the merge gap, and at what cost?
* P-Q4a reproduce the engine's KL floor; P-Q4b re-measure the float spread on the new file.
* P-E13 which accumulator types and reduction trees do the engine's kernels and llama.cpp's use?

**Engine fit** — these decide how the two repositories lock together:
* P-E2 cost of a layer against its miss count (decides the zero-miss lever, 1.11).
* P-E3 logit margin between the 8th and 9th expert (is zero-spill feasible for this router at all?).
* P-E4 how much VRAM the engine really leaves, per bank size and context.
* P-E5 which weight types the engine's kernels will support.
* P-E9 alignment of the served file's tensors, rows and blocks.
* P-E10 do 16-byte staged loads lift zero-copy reads from 9.59 toward 26.2 GB/s, and does `Q6_K` need row padding?
* P-E11 do 2 MiB pages change zero-copy read speed?
* P-E12 does the CPU/GPU expert-output difference grow with activation outliers?
* P-E14 does the trainer's hot-set writer reproduce the engine's file byte for byte?

**Baseline equivalence:**
* P-T9 does the new trainer, on v0.2's recipe, learn at least as well as v0.2?

**Base and Stage A (master D33):**
* P-X0 the reference base: llama.cpp's own conversion and `llama-quantize Q4_K_M` of the bf16 shards, and how a third-party Q4_K_M differs from it.
* P-X6 does Stage A reproduce that reference tensor for tensor?
* P-X7 how far is each Q4_K_M from the full bf16 model (the author's hypothesis of near-zero drift is tested here)?

#### 3.3 Information needed from outside this repository
* The author's new engine information (pending at the pause).
* Any Windows-native run newer than the one in the engine repo (decision 6).
* The engine project's own cost-per-miss data, or permission to run P-E2 (1.11).
* The original outside research report, if it is to be kept with the record (the master holds a claim-by-claim table of it, 16.4).

#### 3.4 Work not yet done
* **No trainer code** exists. The stages T0–T14 (master 12) have not started. The hardware probe program exists and has run (P-M4 is built but has not run); every other probe is still a specification.
* **Stage A (bf16 → `pelias_base_0step.gguf`)** is specified (master 12.2, T1b) and not written. The shard headers have been read (§1.21), so `ingest` can be specified from measured names and shapes. The shard bytes are still needed for P-X0, P-X6 and P-X7.
* **Batch C** (P-K1, P-K2, P-K3, P-Q10) is specified as one run (master 11.3, D35); its source is not written yet.
* **Edits to the engine blueprint** are listed in master 16.7 and are **proposed, not made**: scope line; VRAM-budget table; CPU-read row; E3 gate (the oracle reads trainer GGUFs); E5 (per-layer capacities, `Q6_K` 210-byte blocks); E11 (huge pages); E12 (hot-set objective); unknowns table; contract section; sources. They wait for the author to lift the pause.
* **Engine side:** the cross-check against the speed ceiling, the engine master notes, the engine plain-English README, and the engine repository update.

#### 3.5 Weak points in this design (so they are not missed)
* **The margin under the measured ceiling is unmeasured** (decision 7 is ruled; the disturbance test in master 11.12 sets the margin). Until then the core's room is 0.16 GiB / none on the whole-card figures or about 1.2 / 0.9 GiB on the process-only figures.
* **Efficiency is unmeasured.** Every tokens-per-second figure for the new trainer is conditional on it (1.8).
* **The size of the merge loss on the core is unmeasured** (1.4); the whole precision plan has an escalation ladder (master 4.8) for the case it is worse than expected.
* **The hot-set and locality ideas are hypotheses** (1.9–1.11). Nothing here depends on them to produce a working trainer.
* **The author's guess that 70–80 % of the earlier draft was compliant is not confirmed.** The compliance matrix has 22 rows: 7 already met, 4 needing correction, 6 new specification, 3 resting on premises the engine's data does not support, 2 waiting on probes. There is no natural denominator for a percentage.
* **Label counts in the master** (occurrences, 2026-10-24): MEASURED 36 · READ-SRC 49 · PUBLISHED 34 · DERIVED 83 · ESTIMATE 23 · UNVERIFIED 26 · DECISION 29. Each ESTIMATE and UNVERIFIED claim is paired with a probe.

#### 3.6 How to keep this file true
When a probe returns, change its entry's label from ESTIMATE or UNVERIFIED to MEASURED, record the date and the raw file, and update the rows of 2.2 that depend on it. Never replace a derived number with a guess. If a result contradicts the master, correct the master first, then this file.
<!-- /SPLICE P7.13 sha256="b4c9a76205d84daeb9667149e7a4cd70c774575e49aa6ccf750da31280d3cd43" -->


## 7.14 The six open author decisions as the research update restates them, and the proposed definition of "top tier" with the first scripts

<!-- SPLICE id="P7.14a" src="pelias-research-update.md (received 2026-10-23 19:59, 870 lines)" lines="589-607" shift="+1" ws="1" edited="0" -->
> **Source (verbatim):** pelias-research-update.md (received 2026-10-23 19:59, 870 lines), lines 589-607. Heading levels shifted by +1 for this document only. Table rows that the source breaks over several lines are joined back (whitespace only).

### 5.5 The 6 Open Author Decisions (`PELIAS_TRAINER_MASTER_NOTES.md` §3.1)

1. **Is the `7.5 GiB` bank a hard ruling or a derived optimum?** Treated as `DERIVED` (largest `0.5 GiB` step under the `10.9 GiB` ceiling; `8.0 GiB` peaks at `11.0–11.19 GiB`).


2. **Which prefill micro-batch (`-ub`) ships in production?** Sets $\Delta_{\max}$ to **`0.36 GiB` (`369 MiB`)** at default `-ub 4096` (where all-`Q6_K` core `+282.0 MiB` fits with `87 MiB` spare) vs. **`~0.11 GiB` (`113 MiB`)** at `-ub 2048` (where shared experts `+25.7 MiB` + attention `+65.5 MiB` + $\alpha/\beta$ `+1.0 MiB` = `+92.2 MiB` fit at `Q6_K` with `20.8 MiB` spare, and DeltaNet projections use Mode Q at `0` extra bytes).


3. **Will `pelias-engine` add `Q8_0` weight kernels for the spine?** Required if any core class is exported as `Q8_0` (`P-E5`); today the engine plans `Q4_K`/`Q6_K`/`F32` weights and `q8_0` KV cache[cite: 1, 2, 4].
4. **Promote Windows-native from `"not in scope"` to a supported build target?** Windows-native is `5.1–5.2 ms/token` (`+24–28%`) faster at decode on Cell A (`MEASURED`, 2 launches; needs a 3rd launch per side to lock as a formal claim)[cite: 1, 2, 4].
5. **Is Mode L / L-Z worth building after Wave 0 probes?** Decided after `P-L1` (Belady vs. static gap), `P-L2` (untrained clamp cost curve), `P-E2` (miss cost by layer), and `P-E3` (8th-vs-9th logit margins) run in Wave 0 with zero trainer code.


6. **Are there any Windows-native runs newer than `raw/windows-native-2026-10-23/`?** Until a newer raw log is added, `47.3 / 46.8 tok/s` short and `53.7 / 54.1 tok/s` at `12k` are the sole official Windows-native numbers.



---

<!-- /SPLICE P7.14a sha256="518147b9535e90c51892be5ac2a0ac45a013cd090dca59140ab0849365463942" -->

<!-- SPLICE id="P7.14b" src="RESEARCH_UPDATE_RECONCILIATION_TRAINER.md (2026-10-23)" lines="122-143" shift="+2" ws="0" edited="0" -->
> **Source (verbatim):** RESEARCH_UPDATE_RECONCILIATION_TRAINER.md (2026-10-23), lines 122-143. Heading levels shifted by +2 for this document only.

### 9. Proposal: what "top tier" means for the trainer, and the first scripts

The master's own definition (11.10 and Chapter 12): Wave 0 ends with a one-page summary that restates each design switch with its measured value, and the T0 exit gate is that you have read it and signed off the switches. Proposed gate for "top tier":

1. This reconciliation applied to the master and notes (section 4 and 5 items, corrections in section 3), pushed.
2. Wave 0 steps 1–6 of master 11.10 run on this machine, results applied (every ESTIMATE the speed, memory and accuracy claims stand on is replaced by a measurement or has a dated kill criterion).
3. The author decisions that touch the trainer ruled: 2 (micro-batch), 3 (`Q8_0` in the engine), 4 (Windows-native), 5 (Mode L), and 1 and 6 for the record.
4. Wave 0 summary read and signed off; backup and handoff current.

First scripts (C++20/CUDA programs, one JSON result each, run from WSL, machine-state record first), in the master's order:

| Batch | Probes | Time on the machine (ESTIMATE) |
|---|---|---|
| A (platform) | P-H8, P-H1, P-H2, P-H3, P-H10, then P-H4 (30-minute soak) | about 1 hour |
| B (transfers and memory) | P-H5, P-H6, P-H12, P-H7, P-M3, P-M4, P-H9, P-H13 (new), P-H11 | about 1 hour plus the 30-minute pinned hold |
| C (kernel efficiency) | P-K1, P-K2, P-K3, P-Q10 | later; decides whether the tile design is reopened |
| D onward | model-file and engine-fact probes (11.10 steps 4–6) | later |

Batch A and B need no model file and no trainer code. P-H3 and P-H4 are the ones that tell us what the overclock really delivers under load.

---

<!-- /SPLICE P7.14b sha256="4b65b1e6fb52298b8ccd5bd7288935e3e480511607bc0fbc7038e0a4d2bcc941" -->


## 7.15 Measurement registers: MEASURABLES (trainer master 16.6) and the bridges (16.5)

<!-- SPLICE id="P7.15" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="2297-2335" shift="+1" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 2297-2335. Heading levels shifted by +1 for this document only.

### 16.6 MEASURABLES — what has to be derived, how, how it is used, and where it applies

This is the separate register the author asked for. "Owner" says who runs it. "State" is **done**, **ready** (no code needed, can run in Wave 0) or **needs code**. Section numbers are in this document; "engine" means the engine blueprint.

| ID | Datum to derive | Proposed method (short; full text in 11.9 unless noted) | How it is used | Applies in | Owner / state |
|---|---|---|---|---|---|
| P-E1 | Does the full-attention layer type route differently; what does full residency cost; how fast does all-8-inside coverage grow with set size | Read-only run of the engine's trace simulator (script spec below) | Decides whether layer weights are uniform; gives the coverage curve of B1.1 | 7.12, 16.5 B1, 14.3 | trainer / **done** |
| P-E2 | Cost of a layer as a function of its miss count, by layer type and context depth | Engine's per-layer timers (`e0_timeline.patch`); regress latency on misses with layer type as a factor | c_l in the loss weights; whether the engine should optimise zero-miss share instead of byte-hit (B1.7) | 7.12, B1.2, B1.7, engine E5/E12 | engine / ready |
| P-E3 | Logit margins between in-set and out-of-set experts; 8th–9th gap | Router logits via the P-S1 callback over the trace corpus; sets R_l of 96 … 224 | δ of the hinge; feasibility of L-Z; near-tie flip rate | B1.2, B1.5, B5 | trainer / ready |
| P-E4 | VRAM left for the core, by bank, prefill micro-batch and context | Engine reference cell with byte-only core variants (`llama-quantize --tensor-type`); NVML peak | Δ_max for the solver | 9.4, B2, 16.3.1, 1.6 | engine / ready |
| P-E5 | Weight types the engine will support | Read the blueprint (E4/E5/E7), ask the author | Output type set of D22 | 9.4, B2 | author / ready |
| P-E6 | Bit-exactness of our own quantizers | CPU implementation vs ggml's `quantize_row_*_ref` (reference only) on random, outlier-heavy and real blocks, then GPU vs CPU | Gate for Mode Q and the exporter | 4.10, 9.3, B2 | trainer / needs code (small) |
| P-E7 | Do the served base weights sit exactly on their grid | Fraction of blocks with Q(deq(block)) = block, per class and type | Validates the erasure model of 4.3 | 4.3, 9.4 | trainer / needs P-E6 |
| P-E8 | Does Mode Q close the merge gap, and at what step cost | Slice, then full model; post-export CE through llama.cpp | Adopt Mode Q per class | 4.10, 9.4 | trainer / needs trainer |
| P-E9 | Alignment of tensors, rows and blocks; does the expert-last layout load identically | Header-only read; load the reordered file in llama.cpp, compare logits | Layout planner choice; manifest | B4, 9.3 | trainer / header part **done** |
| P-E10 | Does a 16-B, shared-memory-staged kernel reach the checksum rate; does Q6_K need padding | Engine `wide_load.cu` variants, Q6_K rows raw vs 432 B | Pack layout; manifest flags | B4, engine E5 | engine / ready |
| P-E11 | Does 2-MiB backing change zero-copy read speed (WSL2, Windows) | 4 KiB vs 2 MiB-backed pinned buffers; read `AnonHugePages` per load mode | Keep 2-MiB alignment optional or require it | B4, 16.3 X3, engine unknowns | engine / ready |
| P-E12 | Does CPU/GPU expert-output difference grow with activation outliers | P-S1 dumps; per-block ρ vs relative difference | Mode B/N adopt or close | B5 | trainer / ready |
| P-E13 | Accumulator types and reduction trees in the engine's and llama.cpp's kernels | Static audit + injected-outlier test against fp64 | The int32/fp32 rule for the engine | B5 | engine / when kernels exist |
| P-E14 | Does the C++ hot-set writer match `export_hot_set.py` byte for byte | Same inputs through both; run the engine's parser | Trainer may own the writer | B4 | trainer / needs code (small) |
| P-Q3 | Which bit-widths and types preserve the learned delta, per class | Merge sweep on the v0.2 adapter | Type per class | 4.3, 9.4 | trainer / ready |
| P-Q4 | σ_float: the stock-vs-stock KL spread; reproduction of the engine's two numbers | Engine protocol on the current served GGUF (step a), then on the new GGUF and trainer tokens (step b) | Every KL gate | B3, 10.3 | trainer / ready |
| P-S1/S6 | Per-layer reference dumps; where the served KL excess sits | `eval-callback` dumps on fixed sequences | Fixtures for rungs 2–3, P-E3, P-E12 | 10.2 | trainer / ready |
| P-L1/L2/L9 | Static-vs-Belady gap; cost of an untrained clamp on this model; router-LoRA support | 11.8 | Size of the Mode L prize | 7.12, B1 | trainer / ready |
| P-H11 | Faster platform for training: WSL2 vs Windows-native | Same bench on both | Platform default; the engine saw +25–28 % decode on Windows-native | 5.1, 15.7, 16.3 X9 | trainer / ready |
| P-X5 | Engine regression with the real merged core | Reference cell, 3 launches per side | Accept or reject an export | 9.5 | engine+trainer / after export |

**Engine-owned measurements the outside report suggested (no trainer dependency; recorded so none is lost).** E-A1 thread count and pinning, 8 physical vs 16 SMT (ledger shows 5–7 beat 10 on stock); E-A2 acquire/release vs seq_cst on handoff flags; E-A3 `PREFETCHNTA` on cold reads; E-A4 L2 persisting window (query `cudaDevAttrMaxPersistingL2CacheSize` first); E-A5 `AnonHugePages` per load mode (the engine's open unknown); E-A6 native-Linux floors (the engine's E1); E-A7 pinned memory above 20 GiB. Each is an A/B in the engine phase named in 16.7, with its own go/no-go.

**P-E1 script specification (method of the run already done; the code is not part of this document).**

1. Load the engine's merged decode routing trace (121,160 decode steps × 40 layers × 8 experts) and the per-layer expert byte sizes; take its 70 / 30 fit / held-out split.
2. Per-layer expert frequency from the fit split; the engine's water-filling selection at 7.5 GiB less the zero slots → baseline resident mask. *Check:* held-out byte-weighted hit 0.8273 and any-miss share 0.639, 4,170 experts — both reproduced exactly.
3. For the held-out steps: per-layer hit, per-layer zero-miss share (all 8 inside), the histogram of misses per layer-step, means over the 10 full-attention layers (indices 3, 7, …, 39) against the other 30.
4. Scenario B: mark the ten full-attention layers fully resident (4.605 GiB), water-fill the rest with the bytes left, report the same.
5. Coverage curve: rank each layer's experts by fit frequency; a held-out step is all-8-inside for set size s iff its worst-ranked expert has rank < s; report the share for s = 64 … 256 and the s needed for 90 / 95 / 99 %.

Results (DERIVED, 2026-10-23): baseline hit 0.8273, any-miss 0.639, miss histogram 0.361 / 0.279 / 0.165 / 0.093 / 0.051 / 0.028 / 0.014 / 0.007 / 0.003 for 0…8 misses; full-attention layers hit 0.8299, others 0.8265; zero-miss 0.3636 vs 0.3602; baseline gives the full-attention layers 1,045 experts (1.878 GiB) and the others 3,125 (5.55 GiB). Scenario B: full-attention layers 1.000, others 0.6164, overall 0.7114, zero-miss 0.3197, any-miss 0.6803. Coverage curve and s needed: table in B1.1. The script runs in seconds on the engine's own data and is re-run on every Mode L candidate.

<!-- /SPLICE P7.15 sha256="1ece555ef5d05ea14fbe483714a44facb96e4ea916c336d70082b9b5982abf95" -->


## 8.10 Engine HOLD and queue of 2026-10-24 06:07, word for word

<!-- SPLICE id="P8.10" src="2026-10-24/engine_hold/ENGINE_HOLD_QUEUE.md" lines="1-33" shift="+3" ws="0" edited="0" -->
> **Source (verbatim):** 2026-10-24/engine_hold/ENGINE_HOLD_QUEUE.md, lines 1-33. Heading levels shifted by +3 for this document only.

### pelias-engine: HOLD and queue (logged 2026-10-24 06:07)

Provenance: original working note (Claude, for the author). Not part of any repository.

#### The hold

The author (06:07, 2026-10-24): "there will still be an overall pause until instructed to continue the pelias-engineer [pelias-engine]." Engine work stays paused until the author says to continue. Nothing in pelias-engine is changed, read for new work, or run while the hold lasts (engine head 34f744f stays untouched). Governing order unchanged: (1) trainer comparison and scripts; (2) hard stop until pelias-trainer is "top tier"; (3) make it universal; (4) engine (HELD); (5) unify.

#### What is queued for the engine, and where it lives

The author's request: log `pelias-research-update.md` so that it is looked at and taken into consideration for the engine, "worked on in the same way as" the trainer.

- The report itself, unedited: `2026-10-23/trainer_master/inputs/pelias-research-update_20261023.md` (received 2026-10-23 19:59 as `16d2a764-pelias-research-update.md`; 109,635 bytes, 870 lines, sha256 6a6fb784...c89e52). It covers the trainer, the engine and a unified program.
- What was already done with it, for the trainer only: `2026-10-23/trainer_master/notes/RESEARCH_UPDATE_RECONCILIATION_TRAINER.md` (sha 73edd360...169c3). Its section 7 (engine items, six points) and section 8 (unified items, four points) are the parked engine-side list; the D-, A- and N-items it names (D2, D4, D5, D7, A1, A3, N4) are the ones that touch the engine.
- Still held from the task list: cross-check of the master against pelias-engine for speed-to-ceiling gains; the engine's own master notes (every low-documentation discovery, derivations, interdependencies); the engine plain-English README; engine repo updates with redactions.

#### "The same way as the trainer": the procedure to repeat when the hold is lifted

1. Read the report in full for the engine's side (line by line; sections 1, 2.2, 4, 5, 6, 7 first).
2. Check every figure against the engine's ledger and raw files (read only), recompute the arithmetic by script, keep the script.
3. List where the report differs from the engine's own data, with the evidence; the engine's data is used (the trainer list was D1-D11).
4. List suggestions adopted or adjusted (A-list) and gaps found while checking (N-list).
5. For each new item write a derivation plan: method, what it is used for, where it applies (as done for the trainer).
6. Propose the engine's "top tier" gate and its first scripts, in the master's labels (MEASURED / READ-SRC / PUBLISHED / DERIVED / ESTIMATE / UNVERIFIED / DECISION / AUTHOR-STATED).
7. Wait for the author's rulings (open engine decisions 1-6 in master 16.7, including whether the bank may be a non-multiple of 0.5 GiB), then apply, back up with hashes, update INDEX and RUNBOOK, push only on the author's go-ahead.

#### Known corrections to carry into that work (from the trainer reconciliation, section 7)

- Blueprint edit 2 must not say "-ub 4096 default" (D2); edit 6 should quote the .wslconfig line as it is (transparent_hugepage=always swiotlb=65536).
- Cross-check file: E0 breakdown labelled DERIVED with the "at least 5.0 | 4.6 ms" bound (D4); the 41-CPU-split question stays open (D5); add the "57.0" origin note (A3).
- Engine README draft (report section 6): the 13.3 us doorbell is a microbenchmark, not a feature of the running engine; the MTP speed sentence sits under the Windows bullet; both need labels.
- Engine unknowns to add: the "-ub 4096" case of P-E4 (D2); P-H13 on Windows-native.
- New facts since then that the engine should reuse (trainer side, measured on this machine, 2026-10-23/24): ceiling 10.7 GiB (decision 7); Windows-native vs WSL host-side numbers (flush, pinned allocation, launch); display-timeout test completed on both sides.
<!-- /SPLICE P8.10 sha256="02a1b0d63aeb42c885d2530958f9fd3f3a2feff899b1181a801e9b7eeac527f0" -->
