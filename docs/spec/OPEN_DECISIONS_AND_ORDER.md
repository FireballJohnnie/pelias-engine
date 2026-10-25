# Order of operations and the open decisions O1 to O14

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Parts 7.1 and 7.16 (master lines 3486 to 3510 and 4293 to 4317). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
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

## 7.1 The master order of operations (new text)

**The governing order (AUTHOR-STATED; the project's standing rule).** (1) Trainer comparison and scripts. (2) A hard stop until the trainer is "top tier". (3) Make it universal. (4) The engine, which is **HELD** at the author's word of 2026-10-24 06:07 and stays held until the author says otherwise. (5) Unify. This document does not lift the hold. It specifies the engine so that the work is ready when the hold is lifted, and it assigns every shared item to the **trainer stage that needs it first**, because those stages are not held.

**Rules that apply to every step.** A step starts only when the gate of the step before it is met; a failed gate stops that branch and the reason is logged (blueprint rule; master 12). Build the most uncertain thing first, in the cheapest form that can be tested against something trusted (master 12). One step at a time, and the author runs every GPU program. Nothing is pushed without the author's word.

| # | What is built or decided | Trainer stage | Engine phase | Part 6 items | Gate that closes it | Status as recorded |
|---|---|---|---|---|---|---|
| 1 | Measurements of the machine, and the author's switches | T0 | E0 and E1 | 6.9 (ceiling), 6.6 (handoff figures) | T0: the author reads and signs the one-page switch summary. E0: the gap between the modelled ceiling and the live figures is explained | The hardware probes ran on WSL2 and on Windows-native on 2026-10-23 and 2026-10-24 (Part 3.3). The E0 split is DERIVED, not isolated (Part 1). The probes outside Wave 0 are specifications that have not run |
| 2 | Foundations: device layer, governor, arenas, file mapping, GGUF reader, memory solver, test harness, skeleton | T1 | E2 and E3 (reader) | 6.5, 6.8, 6.9 | T1: lists all 753 tensors with types; a plan above 10.7 GiB is refused; two clean builds give identical test results. E2: clean build on WSL2 and on Windows-native | Trainer: the host-only build framework exists; the `io/gguf_reader` file is a stub (Part 4.5.2). Engine: no code |
| 3 | Block formats and quantizers (CPU first) | T1b | E3 (oracle), E4 | 6.11 | P-E6: byte-identical to the oracle | Not started |
| 4 | Tokenizer and chat-template renderer, data pipeline | T2 | E10 (reuse) | 6.13 | P-D1 and P-D2: zero mismatches | Not started; the stubs are empty |
| 5 | Kernel families: matrix, elementwise and router, attention, DeltaNet; each reference first | T3 to T6 | E4, E5, E7 | 6.10 | Rung 1 tolerances; kernel parity | Not started |
| 6 | The external oracle harness and the project's comparators | T12 (verify) | E3 | 6.3, 6.17 | The oracle reproduces the stock noise floor; the comparators agree with it | Not started; the engine ledger holds the reference numbers |
| 7 | One layer, then a tiny model, then the full forward pass | T7, T8 | E6, E7 | 6.6, 6.7 | Rung 2 and rung 3 | Not started |
| 8 | Backward pass, LoRA, optimizer; streamer and scheduler | T9, T10 | E5, E6 (bank, handoff) | 6.6, 6.9 | Rung 4; zero allocations in the step loop (P-M5) | Not started |
| 9 | Training loop, evaluation, checkpoints; export, verify and the hand-off to the engine | T11, T12 | E8 (decode loop) | 6.7 | Rungs 6 to 8; hash proof; K1, K2, K3 | Not started |
| 10 | Decode loop, prefill, long context | (none) | E8, E9 | 6.7 | E8: at least the llama.cpp tier result, then 55.0 tok/s at 12k, then the ceiling table. E9: prompt reading at least the tier result | **HELD** |
| 11 | Server: sockets, HTTP, SSE, endpoints, tool calls, request log; the llama-server catalogue | (the trainer's `serve` module, held until unification) | E10 | 6.12, 6.14 | Drop-in for the launcher; the capability probes; a Visual Studio session works; replay of captured client requests | **HELD**; the catalogue is not written (6.14) |
| 12 | Start-up probe and plan card; capability detection | the shared `startup` module | E11 | 6.9 | The same code base passes the E8 to E10 gates on WSL2 and Windows-native | **HELD** |
| 13 | Performance engineering; optional modes; gated extras; second model through the plug-in interface | T13, T14 | E12, E13 | 6.10 | Each mode's own pre-declared gates | Not started |
| 14 | Unify: one binary or one shared base, one set of probes, one release | (after both) | (after both) | 6.4 | Both applications pass their ladders on the shared base | Item 5 of the governing order; not started |

**How to read the two phase lists.** The engine phases E0 to E13 are in 7.2 with the blueprint's own gates, edited in place. The trainer stages T0 to T14 are in 7.7 with their build, test and rollback detail. Where the two lists name the same item (the reader, the tokenizer, the hot-set writer, the quantizers), 7.1 gives the earlier one, and the later one reuses it.

## 7.16 Every open decision in one place (new text)

Collected from 7.5 (engine decisions), 7.12 (the trainer master's nine), 7.13 (notes section 3), Part 6 and the runbook. A decision is closed only by the author's word. **Numbers O1 to O14 are assigned in this document only.**

| # | Decision | Status | What waits on it | Where it is described |
|---|---|---|---|---|
| O1 | Is the 7.5 GiB bank a ruling, or the current best step under the 10.7 GiB ceiling? May the bank be a non-multiple of 0.5 GiB? | Open; this document treats 7.5 as derived (master 16.7, decision 1) | The core's room; the governor's default (6.9) | 7.5, 7.12 |
| O2 | Is Mode L (and L-Z) worth building after the Wave 0 probes P-L1, P-L2, P-E2 and P-E3 report? | Open | Whether any router training is built; the E12 extras | 7.5, 7.12 |
| O3 | What does a blank answer in the parameter sheet do (knowledge-only rows, or the base model drafts an answer for the user to review)? | Open; asked 2026-10-24, no answer | The intake (trainer U7) | 7.12, 7.13 |
| O4 | May Expert mode switch off the clean-stop guard, or only change its limits? | Open; proposed: limits only; not yet asked | Safety rails (trainer U11) | 7.12 |
| O5 | May Claude read the author's verification script and the gate set it names? | Open; go-ahead awaited | The trainer's data gates (7.13) | 7.13 |
| O6 | May Claude look up the published request formats of the named clients? | Open; go-ahead awaited | The catalogue's client column (6.14) | 6.14, 7.13 |
| O7 | The support-bundle wording (trainer U10) | Open; clarification asked | The warnings and bundle (trainer) | 7.13 |
| O8 | Add the hold-queue line "catalogue llama.cpp's parameters and behaviours as the spec for the Pelias server"? | Open; asked twice, unanswered | The whole of 6.14 | 6.14, Part 8 |
| O9 | Dear ImGui: keep for the interface only, write an own layer, or build headless first? | Open | The trainer's `ui` module; the third-party list (the draft names it as the one third-party library) | 5.1, 6.4 |
| O10 | The server's policy for special-token text inside user content: pass, escape or refuse | Open; this document proposes `pass` as the default because it reproduces the behaviour the clients have today, with every occurrence counted | 6.13, 6.12 | 6.13 |
| O11 | May the original E8 kill criterion (fall back to a llama.cpp patch series) be restored? | Open; removed in this document by `[sync R2]` | The meaning of a failed E8 gate | 5.0, 7.4 |
| O12 | Does the ruling "WSL2 and Windows-native, native Linux held" apply to the engine as well as to the trainer draft? | Applied by `[sync R1]`; the author may correct it | The engine's platform list | 5.0 |
| O13 | When does the engine hold lift, and does the ground-up engine work start before the trainer is "top tier"? | Held at the author's word | All of Part 6 that is marked engine; E1 to E13 | 7.1 |
| O14 | Order of work: the Batch C probe source, or the llama-server catalogue? | The 09:09 request (this document) takes precedence; both are waiting | 6.14, the open trainer probe batch | Part 8 |

**Measurements still needed before the numbers they affect may be used** (full list in 7.13 and in the probe registry, 7.9): the engine's own VRAM peaks and the disturbance margin (O1; P-E4), the E0 timeline (P-E2), the wide-load kernel rate (P-E10), the huge-page effect (P-E11), the load rate from the WSL2 disk (P-M4), the efficiency of an own GEMM (P-K1 to P-K3), the hit rate of the real workload against the simulation, the effort of the tokenizer, renderer, server and reader, and the client requests (6.14).

**Weak points** that the existing documents already name (7.13; Part 2): the E0 breakdown is DERIVED; the 41st CPU split being the MTP block is UNVERIFIED; the Windows–WSL2 match of 5.1 to 5.2 ms is a hypothesis; no run of a non-llama server exists, so the gap to the ceiling is not isolated; the Windows-native speed gain rests on two launches against one.
