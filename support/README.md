# support/ : everything gathered from the C:\AI tree that bears on the engine

Written 2026-10-25 (the consolidation recorded as [sync 11] in the blueprint). Nothing in this folder is engine code. Nothing here is part of the runtime. It is the evidence and history that the engine documents cite: logs, probe sources and results, prototypes, patches and earlier notes, each copied from a search of the whole C:\AI tree (not moved; the originals stay on the drive).

Start with `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md` (what the engine tree is and where each part goes) and `docs/PELIAS_ENGINE_LINEAGE.md` (how the old figures became the current ones, and which old material is not otherwise listed).

## 1. What is where

| Path | What it holds |
|---|---|
| `MANIFEST.tsv` | One row per file found by the search (1,559 rows): class, group, where it is in this repository, where it came from (redacted path), size, SHA-256 of the source and of the copy, source date, reason, redaction counts, and for duplicates the repository file it equals. Rows for withheld files carry no path and no name. |
| `archive/c_ai/creator/` | The first working tree (`Stuart/Creator`): MoE microbenchmarks, E0 probes, bar probe, simulator, trace tools, fork tests, MTP alignment, the 14-file CUDA serve prototype, the tier patch v1 and its sources, test programs, phase 1 and 2 logs, 493 speed-server logs, fork and routing traces. |
| `archive/c_ai/backup/` | The dated backups (`cl-sndbx-bkp/stuart-creator/`), 2026-10-20 to 2026-10-24: the P1 sweep logs (2026-10-21), the earlier copy of the blueprint (2026-10-22), the Wave 0 probe results for WSL2 and Windows (2026-10-23), the engine hold queue and the two interim Windows launcher scripts with the full flag set (2026-10-24), the first-conversation design documents. |
| `archive/c_ai/pelias_probes_win/` | The Wave 0 probe sources and run scripts for Windows and WSL2 (14 files). |
| `archive/c_ai/win_build/` | Build notes and scripts for the Windows-native tier build (5 files). |
| `tier_build/` | The current Windows-native tier tree, as a patch against llama.cpp commit 2ca15f540 (`pelias_tier_current_vs_2ca15f540.patch`) plus the two new source files (`src/llama-pelias-tier.cpp`, `src/llama-pelias-tier.h`). **A record of the interim build, not part of the product** (the product is ground-up; llama.cpp stays an external, read-only test oracle). |
| `local_only/` | Withheld material (section 3). **Ignored by git.** Only its manifest is described here. |

## 2. Classes

| Class | Meaning | Files |
|---|---|---|
| PUSH-VERBATIM | Copied byte for byte (SHA-256 of copy equals source). | 516 |
| PUSH-REDACTED | Copied after the redactions in section 4; the manifest records the count of each. | 858 |
| TIER-BUILD | The three tier-build files in `tier_build/`. | 3 |
| DUPLICATE-OF-REPO | Byte-identical to a file already in the repository; not copied again. The row names the repository file. | 55 |
| LOCAL-ONLY | Withheld from the repository (section 3); kept on the drive and in `local_only/`, which git ignores. | 126 |
| NOT-COPIED | One file: a 10.3 MB mailbox of the last two engine commits (42e3496, 34f744f). The commits are in the repository history, so a copy adds nothing. | 1 |

Total 1,559. The pushable set is 1,374 archive files (13,328,596 bytes on disk after redaction) plus the 3 tier-build files. `tier_build/src/llama-pelias-tier.h` is byte-identical to the version-1 copy under `archive/c_ai/creator/moe_track/patches/src/`; it is kept in `tier_build/` so that the tier-build set is complete.

## 3. What is withheld, and why

126 files, all kept on the drive and in `local_only/` (never pushed), in five groups:

| Count | Reason |
|---|---|
| 61 | Environment listings of the user's home folder and installed packages. |
| 23 | Prompt corpora (persona, agent, code and reasoning prompt text). |
| 21 | Files that name a private script or equation (the earlier handoffs and ledgers, the backup index and runbook, five plan documents, and the speed scripts and path module that name private scripts). |
| 16 | Machine identifiers (operating-system reports, event logs, journals). |
| 4 | Persona or dataset design detail. |
| 1 | An archive of mixed working files (the sandbox archive of 2026-10-24: trainer, engine and master-builder sources); withheld whole because its entries were not scanned one by one. |

Rule in force: the dataset, the persona and the author's private calculations stay out of every file of every repository. A document that is mostly engine history but names one of them is withheld whole, not edited, so that nothing is paraphrased wrongly. Everything the engine documents quote from those files was taken from the pushable copies or from the master specification.

## 4. Redactions applied to PUSH-REDACTED files

Each substitution is counted per file in the `redactions` column.

| Pattern | Replaced by |
|---|---|
| The user's home folder and models folder | `/home/user`, `~/models/` |
| The user's Windows profile path | `.../Users/user` |
| The project root on C: or D: | `<project>` |
| The llama.cpp clone on C: | `<llama.cpp>` |
| The test model file name | `test-model-<quant>.gguf` |
| The calibration text file name | `calibration-text.txt` |
| Dataset file names | `dataset/<file>`, `<dataset-file>` |
| Reply text after `answer starts:` and in JSON content fields | `[reply text removed]`, `[text removed]` |
| The repository owner account | `<account>` |
| The author's personal name | "the user" (the approved copyright line is kept as it is) |
| Network adapter hardware addresses | `XX-XX-XX-XX-XX-XX` |

After placement, a residual scan over every pushable file found none of: the user's account name, the personal name outside the approved copyright line, the repository account name, a Hugging Face token, or a Windows profile path that is not already redacted. The HF token file was never opened.

## 5. How to check

`python3 docs/scaffold/verify_inventory.py` (standard library only) recomputes the SHA-256 of every file listed in `docs/scaffold/REPO_INVENTORY.tsv` and of every pushable row of `support/MANIFEST.tsv`, and reports any file that is missing, changed or unlisted. The last run is recorded at the end of `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`. Source hashes in the manifest identify the originals on the drive; they cannot be rechecked from a clone of this repository.

## 6. Reading rules

- Labels are the engine's eight (MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED). A number in an archived log is MEASURED only for the run that log records; the lineage document says which later figures replaced which.
- Every recorded speed in this tree comes from the patched llama.cpp tier build. No non-llama run exists.
- Retired figures (for example the old 57.0 tok/s Windows number) remain in the archive because they are history. They are listed in the lineage document with the reason they are retired, and must not be quoted as data.
- The file `archive/c_ai/backup/2026-10-22/docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` is an earlier copy of the blueprint. The current one is `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md`.
