# pelias-engine: HOLD and queue (logged 2026-10-24 06:07)

Provenance: original working note (Claude, for the author). Not part of any repository.

## The hold

The author (06:07, 2026-10-24): "there will still be an overall pause until instructed to continue the pelias-engineer [pelias-engine]." Engine work stays paused until the author says to continue. Nothing in pelias-engine is changed, read for new work, or run while the hold lasts (engine head 34f744f stays untouched). Governing order unchanged: (1) trainer comparison and scripts; (2) hard stop until pelias-trainer is "top tier"; (3) make it universal; (4) engine (HELD); (5) unify.

## What is queued for the engine, and where it lives

The author's request: log `pelias-research-update.md` so that it is looked at and taken into consideration for the engine, "worked on in the same way as" the trainer.

- The report itself, unedited: `2026-10-23/trainer_master/inputs/pelias-research-update_20261023.md` (received 2026-10-23 19:59 as `16d2a764-pelias-research-update.md`; 109,635 bytes, 870 lines, sha256 6a6fb784...c89e52). It covers the trainer, the engine and a unified program.
- What was already done with it, for the trainer only: `2026-10-23/trainer_master/notes/RESEARCH_UPDATE_RECONCILIATION_TRAINER.md` (sha 73edd360...169c3). Its section 7 (engine items, six points) and section 8 (unified items, four points) are the parked engine-side list; the D-, A- and N-items it names (D2, D4, D5, D7, A1, A3, N4) are the ones that touch the engine.
- Still held from the task list: cross-check of the master against pelias-engine for speed-to-ceiling gains; the engine's own master notes (every low-documentation discovery, derivations, interdependencies); the engine plain-English README; engine repo updates with redactions.

## "The same way as the trainer": the procedure to repeat when the hold is lifted

1. Read the report in full for the engine's side (line by line; sections 1, 2.2, 4, 5, 6, 7 first).
2. Check every figure against the engine's ledger and raw files (read only), recompute the arithmetic by script, keep the script.
3. List where the report differs from the engine's own data, with the evidence; the engine's data is used (the trainer list was D1-D11).
4. List suggestions adopted or adjusted (A-list) and gaps found while checking (N-list).
5. For each new item write a derivation plan: method, what it is used for, where it applies (as done for the trainer).
6. Propose the engine's "top tier" gate and its first scripts, in the master's labels (MEASURED / READ-SRC / PUBLISHED / DERIVED / ESTIMATE / UNVERIFIED / DECISION / AUTHOR-STATED).
7. Wait for the author's rulings (open engine decisions 1-6 in master 16.7, including whether the bank may be a non-multiple of 0.5 GiB), then apply, back up with hashes, update INDEX and RUNBOOK, push only on the author's go-ahead.

## Known corrections to carry into that work (from the trainer reconciliation, section 7)

- Blueprint edit 2 must not say "-ub 4096 default" (D2); edit 6 should quote the .wslconfig line as it is (transparent_hugepage=always swiotlb=65536).
- Cross-check file: E0 breakdown labelled DERIVED with the "at least 5.0 | 4.6 ms" bound (D4); the 41-CPU-split question stays open (D5); add the "57.0" origin note (A3).
- Engine README draft (report section 6): the 13.3 us doorbell is a microbenchmark, not a feature of the running engine; the MTP speed sentence sits under the Windows bullet; both need labels.
- Engine unknowns to add: the "-ub 4096" case of P-E4 (D2); P-H13 on Windows-native.
- New facts since then that the engine should reuse (trainer side, measured on this machine, 2026-10-23/24): ceiling 10.7 GiB (decision 7); Windows-native vs WSL host-side numbers (flush, pinned allocation, launch); display-timeout test completed on both sides.
