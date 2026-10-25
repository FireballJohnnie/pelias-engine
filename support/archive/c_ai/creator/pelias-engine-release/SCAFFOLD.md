# SCAFFOLD: proposed GitHub layout and file count

*Proposal, 2026-10-20. Nothing is uploaded. Nothing in the development tree (`moe_track\`) has been moved. This file is a planning document and is not part of the published repository.*
*Labels: **measured** (counted or measured on the development tree), **estimate** (my guess at work that is not written yet).*

## 1. Proposed repository

* Name: `pelias-engine` (**pending a name check**: "Pelias" is also an unrelated open-source geocoder, unverified).
* Fresh git history (never push the development tree's history). Default branch `main`. First tag `v0.1.0`, marked pre-release, with a limitations note.
* Private first; public only after the release gate in `moe_track\UNFINISHED.md` (all testing for the current model, then sections F1-F10).

## 2. Layout, with file counts

"Existing" = files that already exist in the development tree and would be copied (counted from `MANIFEST.sha256`, **measured**). "New" = files still to be written (**estimate**).

```
pelias-engine/                                   existing 2  new ~9     README.md, LICENSE (drafts exist); NOTICE, THIRD_PARTY_NOTICES.md,
                                                                         CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md,
                                                                         CITATION.cff, .gitignore, .gitattributes
  .github/                                       existing 0  new ~5     CI workflow, issue templates, pull-request template
  docs/                                          existing 2  new ~7     RUN_ORDER.md, STATUS.md; design, porting guide, methodology,
                                                                         benchmark protocol, reproducibility, limitations/FAQ
  src/
    core/                                        existing 8  new ~5     splitter, handoff, weights view, cold-path kernels, CPU pool, quant geometry;
                                                                         new: polling CPU worker, device-side planner, kernels for other formats
    portability/                                 existing 1  new ~2     pl_cuda.h; new: HIP and SYCL layers (only after the R1-R3 probes pass)
    experimental/                                existing 4  new 0      profiler, state snapshots, speculative-decoding policy
  patches/llama.cpp/                             existing 4  new ~3     tier patch, build script; new: v2 patch (cold path), notes
  tools/
    measure/                                     existing 8  new ~6     hardware floors; new: the universal measurement framework
    trace/                                       existing 7  new 0
    sim/                                         existing 4  new 0
    launcher/                                    existing 1  new 0
    parity/                                      existing 2  new 0
    common/                                      existing 6  new ~1
  tests/                                         existing 26 new ~5     25 tests plus the CPU stand-in (tests/support/cuda_shim.h)
  scripts/                                       existing 5  new ~1     numbered run-order scripts + env.example.sh
  benchmarks/                                    existing 0  new ~10    results ledger, methodology, raw-log summaries (JSON / CSV)
  trainer/                                       NOT in the first release (see section 4)
```

## 3. Totals

| | Files | Size |
|---|---|---|
| Existing files in the first (engine-only) release | **78** (**measured**) | about 0.8 MB (828 KB, **measured**) |
| Draft front-page files already written | 2 (README.md, LICENSE) | about 38 KB |
| New files to write | about **54** (range 40-65, **estimate**) | not estimated |
| **Approximate total at first release** | **about 135 files** (range 120-150, **estimate**) | a few MB at most, mostly text |

The held-back files (section 4) would add 22 existing files (289 KB, **measured**) if they are released later.

How the counts add up: the development manifest lists 100 files (1,117 KB). Of those, 6 fork-comparison files and 8 trainer files are held back, plus 8 tests that only test those held-back parts (`test_fork_scripts.py`, `test_merge_blk40.py`, `test_mtp_numpy.py`, `test_mtp_torch_fake.py`, `test_dump_hidden.py`, `test_trainer_io.py`, `test_pelias_v03.py`, `fake_torch.py`): 100 - 22 = 78. The manifest file itself is not released.

## 4. Not uploaded in the first release

* **Held back** (22 files): `fork/` (compares third-party forks whose licences are unchecked, and one script may contain private text), the trainer (`mtp_align/`, `patches/pelias_v03_patches.py`, `stuart_pelias0.3.py`: licence review of Unsloth-related dependencies, and the trainer is still being refined), and the 8 tests tied to them. The trainer is better as its own repository.
* **Never uploaded:** model files (GGUF, safetensors, adapters), the Hugging Face cache, raw logs, virtual environments, build outputs, `__pycache__`, personal or project-private data (persona text, datasets, seed prompts until checked), the development `MANIFEST*.sha256`, the backup folder with `INDEX.md` and `RUNBOOK.md` (a cleaned `docs/STATUS.md` replaces them), anything that looks like a token or key.

## 5. Upload stages (each one waits for the release gate)

| Stage | What | Files |
|---|---|---|
| 0. Listing | `README.md` and `LICENSE` only, in a private repository. Needs the name, licence and copyright holder confirmed first. | 2 |
| 1. Measurement | `tools/measure/`, `tools/common/`, `docs/`, `benchmarks/` (the hardware-floor tools and their logs) | about 33 |
| 2. Engine | `src/`, `patches/`, `tests/`, `scripts/`, the rest of `tools/` once the integration and its end-to-end benchmark exist | about 95 |
| 3. Universal framework | the measure-anything tool and its validation runs | about 6 |
| 4. Trainer | separate repository, after refinement and licence review | 16+ |

## 6. Before anything moves

* The scrub list (hard-coded paths in 25 files, the project name in 98 of 102 text files, SPDX headers, a fresh secret scan) is `moe_track\UNFINISHED.md`, section F7. The file-by-file move map is section G.
* Copy, do not move. A script rewrites paths and include lines in the copies, the CPU stand-in tests run in the new tree, and the results are compared with the development tree before it is touched.
