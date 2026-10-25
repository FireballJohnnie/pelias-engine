# Trainer co-design contract

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Part 5.2 (master lines 2763 to 2813). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 5.2 Trainer co-design contract [sync 9]

The trainer and the engine are two applications built on one shared base (6.4). The contract below states, milestone by milestone, what each side releases to the other and what each side checks. It is the edit `[sync 9]`. Two points are decided in it: **the engine owns the packed layout** (and with it `repack_gguf.py`, R17 and D25), and **the trainer emits standard GGUFs**, optionally in the expert-last layout, which puts all 123 expert tensors on 2 MiB boundaries at no cost in file size by means of a pad entry under 2 MiB (the pad key is `pelias.pad`; probe P-E9). The facts behind the layout are in Part 1 (alignment of the 123 expert tensors: all 16-byte aligned, 9 4 KiB-aligned, none 2 MiB-aligned; data starts at byte 10,991,104; `general.alignment` absent, so 32).

<!-- SPLICE id="P5.2a" src="PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1)" lines="2340-2351" shift="+0" ws="0" edited="0" -->
> **Source (verbatim):** PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md (trainer master, local head 953c9b1), lines 2340-2351. The contract table from trainer master 16.7. Its last column reads "10.7" where the research update's copy of the table (next block) still reads 10.9.

**Two-way contract.**

| Engine milestone | Releases to the trainer | Trainer output the engine checks |
|---|---|---|
| E0 per-layer timers | P-E2 → layer weights c_l (B1.2) | — |
| E1/E11 platform results | Platform default for training (P-H11) | — |
| E3 reader and oracle | Reproduction of the stock KL floor (B3) | Layout manifest and exported GGUF parse identically; K3 gate holds on the exported GGUF |
| E5 bank and wide-load kernel | P-E10, P-E11 → manifest flags, pack decision | Hot-set file accepted by the slot-table loader; parity P-E14 |
| E7 full spine | P-E5 type matrix, P-E13 audit | Core types within the supported set |
| E8/E9 decode loop, VRAM planner | Δ_max (P-E4) at the shipped micro-batch | Peak VRAM at the 7.5 GiB bank ≤ 10.7 GiB with the real merged core |
| E12 extras | LRU only if live hit rate is well below simulation; MTP | Mode L candidates: re-profiled hot sets, measured decode vs C0 (G-L4, G-Z4) |

<!-- /SPLICE P5.2a sha256="4322d81b3f7ccd4118f3746cfd242107fad7d17bd9f11d563ed6ac9f6ab315a3" -->

<!-- SPLICE id="P5.2b" src="pelias-research-update.md (received 2026-10-23 19:59, 870 lines)" lines="471-500" shift="+1" ws="1" edited="0" -->
> **Source (verbatim):** pelias-research-update.md (received 2026-10-23 19:59, 870 lines), lines 471-500. The research update's copy of the same contract, kept for the record. Heading levels shifted by +1 for this document only. Table rows that the source breaks over several lines are joined back (whitespace only).

### 5.3 The Two-Way Engine ↔ Trainer Milestone Contract (`PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` §16.7)

| Engine Milestone[cite: 1, 2, 4] | What `pelias-engine` Releases to `pelias-trainer`<br> | What `pelias-trainer` Releases / What `pelias-engine` Verifies |
| --- | --- | --- |
| **E0** (Per-layer timers) | `P-E2` ($c_0, c_1$ per miss & layer type) $\rightarrow$ layer weights $c_l$ in Mode L (`B1.2`) | — |
| **E1 / E11** (Platform floors) | Native Linux, WSL2, and Win11 floor table $\rightarrow$ platform default (`P-H11`)[cite: 1, 2, 4] | — |
| **E3** (Reader & Oracle) | Reproduction of stock KL floor (`0.017353` batch 16 / `0.018286` batch 1024, `B3`, `P-Q4a`)[cite: 1, 2, 4] | `layout_manifest.json` and `merged-core.gguf` parse identically; `K3` gate passes |
| **E5** (Bank & Wide-Load) | `P-E10` (`uint4` staging) & `P-E11` (`2 MiB` pages) $\rightarrow$ manifest flags & pack spec[cite: 1, 2, 4] | C++ `PELIAS_HOT_SET` writer matches `export_hot_set.py` byte-for-byte (`P-E14`) |
| **E7** (Full 40-Layer Spine) | `P-E5` supported weight type matrix (`Q4_K`, `Q6_K`, `F32`; optional `Q8_0`) & `P-E13` accumulator audit (`int32` dot + `fp32` scale)[cite: 1, 2, 4] | Trainer exports only supported core types (`D22`) |
| **E8 / E9** (Decode & VRAM Planner) | True $\Delta_{\max}$ (`P-E4`) at shipped prefill micro-batch[cite: 1, 2, 4] | Merged core fits under `10.9 GiB` at the `7.5 GiB` bank (`P-X5`) with `≤2%` decode regression |
| **E12** (Gated Extras) | MTP draft acceptance (`P-T8`) & live bank hit rate[cite: 1, 2, 4] | Re-profiled hot sets & Mode L candidates (`G-L4`, `G-Z4`) |

<!-- /SPLICE P5.2b sha256="175e1324009efa22740466cb978bf151ec6a392d1d1aa8c669e3c622262ced6d" -->

**What the proprietary runtime adds to the contract (new text; DECISION, proposed).**

| Point | Statement |
|---|---|
| One reader, one writer | The engine's reader and the trainer's writer are the shared `io/gguf_reader` and `io/gguf_writer` (6.4, 6.8). They are written once and the contract is checked in both directions: the reader opens a trainer export with its `layout_manifest.json`; the writer's output opens in the external oracle (`tests/oracle/`) |
| The layout is a parameter, not a patch | `layout` is `standard` or `expert-last`. The reader accepts both; the writer takes the layout from the manifest the engine publishes. A file in either layout holds the same tensor bytes (P-E9 checks that the reordered file gives identical logits in the oracle) |
| The pad entry | The pad is a string metadata entry of fewer than 2 MiB, placed so that the first expert tensor starts on a 2 MiB boundary. The reader skips unknown keys; the oracle's loader is expected to do the same, which P-E9 confirms |
| Alignment is checked at load | The reader refuses a file whose expert tensors are not 16-byte aligned, and logs (does not refuse) a missing 2 MiB alignment |
| The hot-set writer | The trainer may own the writer of the `PELIAS_HOT_SET` file, if its output is byte-identical to `export_hot_set.py` (P-E14); the engine's reader parses it either way |
| The tokenizer and the renderer | One implementation (6.13), written by the first stage that needs it (the trainer's T2) and reused by the engine's server (E10). The contract carries the parity gates P-D1 and P-D2 |
| The oracle | Both applications use the same external oracle build and the same fixed prompts; the engine's E3 gate reproduces the stock noise floor before any other result is accepted |
| Who decides a conflict | The contract table above is the record. A change to it is a change to both repositories and waits for the author while the engine is held |
