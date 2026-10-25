# UNFINISHED.md — what is NOT finished, and why

Written 2026-10-20 with the package. Everything not listed here was written, and tested in a sandbox (18 test groups, `tests/run_all_tests.sh`; all passed).
The sandbox has **no GPU, no nvcc, no PyTorch, no real model** — so every item below is something only your machine can settle.

## A. Not written (needs a real GPU engine)

| What | Why it is not here |
|---|---|
| **E4 — single-stream spine executor** (runs all 40 layers: DeltaNet + attention + hot/cold experts, with the cold pool and snapshots wired in) | It is a whole inference engine. Its GPU kernels can't be compiled or checked without a GPU, and nothing else (E1, E2, E7) can be checked until it exists. |
| **E7 — llama-server adapter** (serve E4 through the llama.cpp HTTP API) | Needs E4 first, plus llama.cpp internals (`llama_memory_recurrent`, graph hooks) and the cold pool reading **plain-layout** weights (llama.cpp's CPU "repack" buffer is not plain). Not verifiable offline. |
| **E5 / E6 / E0 wired into a running engine** | The pieces exist and are tested (below), but there is no engine for them to plug into yet. |

## B. Written and tested, but only off-GPU / off-torch — first real run is yours

Run these in the order shown in README.md; each has a self-check that fails loudly.

- **CUDA code never compiled with nvcc and never run on a GPU:** `microbench/03_handoff_latency.cu`, `microbench/04_dual_bus_test.cu`, `pelias_serve/e0_layer_profiler.cuh`, `e5_deltanet_state.cuh`, `e6_hot_aware_mtp.cuh`, `lib/pl_cuda.h`. Their logic was tested on the CPU through a serial CUDA stand-in (`lib/cuda_shim.h`; the E6 block was emulated with threads + a barrier). Expect possible nvcc compile errors on first build (typos, arch flags).
- **Pelias-Tier GPU path (Phase 6):** tested against a CPU-only llama.cpp build (bit-exact when off, float-order-exact when on). The CUDA backend path has not run. Gate H6 (`13_phase6_run.sh kl-base` / `kl-compare`) is the real check.
- **Torch code never run with real torch/transformers:** `mtp_align/10_train_blk40_mtp_lora.py` + `mtp_torch.py` (only checked against a numpy-backed fake torch and an independent numpy implementation — run `selftest` first), and `patches/pelias_v03_patches.py` (peak-VRAM guard, KV-cached probe, optional offload). The cached probe checks itself on the first prompt and falls back to the old loop if it disagrees; offload is **off by default** and fails loud. Try them with the `smoke` stage first.
- **`mtp_align/11_merge_blk40_and_export.py`** is numpy-only and fully tested, but the commands it *prints* (`convert_hf_to_gguf.py` on the MTP-only folder, `llama-quantize`) have never been run.
- **MTP numpy reference** (`mtp_block_numpy.py`) is checked against the torch fake and itself, **not** against a real llama.cpp MTP run (no tiny Qwen3.5-MoE-with-MTP model exists to test with).

## C. Not possible as the roadmap wrote it (I changed it; the code follows the corrected version)

- **v0.3 "zero-sync routing plan" — not implemented.** The draft's `_plan` returned 3 values (the real one returns 4, so it would crash at layer 1), and removing the host sync would need the expert function rewritten around one grouped GEMM over all 256 experts (~1.6 GiB extra VRAM vs 0.2 GiB per chunk). Only a real run can say if that fits. Details at the top of `patches/pelias_v03_patches.py`.
- **v0.3 offload** — the draft kept `Y_moe` on the host, which saves no PCIe traffic. Implemented instead: only each layer's input is offloaded, and it only matters above 4096 training tokens (≈0.5 GiB saved at 4096).
- **E5 snapshot cost** — the draft said 48 MiB / 0.095 ms. From the model geometry it is **62.9 MiB per 30-layer slot, ≈0.5 ms**. And a rejected draft costs a rollback **plus a re-decode** of the accepted token, so break-even acceptance is ≥ 0.5 even with a free draft.
- **E6 "≥ 15 % gain"** — my policy model (`pelias_serve/e6_policy_model.py`, made-up inputs) suggests **≈ 1.07×** at best. Treat it as an estimate: measure with `e0_layer_profiler` and `fork/5e_mtp_acceptance.py`, then re-run the model with real numbers.
- **Roadmap text points at things that don't exist:** `sim/08_verify_gate_h6_and_mtp.py` (gates live in `lib/gates.py`), `patches/pelias_tier.h` and `pelias_q6k_ext.cuh` (Phase 6 is done at the graph level in `patches/src/llama-pelias-tier.{h,cpp}`).
- **Roadmap §12 launcher env vars** `PELIAS_HOT_PROFILE`, `PELIAS_MISS_MODE`, `PELIAS_GGUF_SHA256`, `PELIAS_CPU_THREADS` are not read by the delivered patch. The launcher sets the ones it does read: `PELIAS_HOT_SET`, `PELIAS_TIER_BYPASS_N`, `PELIAS_TIER_MAX_MIB`. Hot set file: `logs/pelias_hot_set.txt` (from `sim/export_hot_set.py`), not `sim/hot_profile.json`.

## D. Assumptions about llama.cpp / forks that I could not verify

- Fork flags used by `fork/gauntlet.py` for the **Schwerz** and **Parmesan** families are taken from their docs, not from a build; the first start will tell (the gauntlet prints a note for each).
- `fork/5e_mtp_acceptance.py`: the per-request field `speculative.n_max` and the response counters `draft_n` / `draft_n_accepted` exist in the pinned source (checked). Whether `n_max: 0` really switches drafting off for MTP is untested; if the server still drafts, the script marks its paired baseline invalid and uses the stored N=28 speed file.
- `--spec-type draft-mtp`, `-md`, `--load-mode`, `--prio`, `--n-cpu-moe` exist at the pinned commit (checked in the source). `GGML_CUDA_DISABLE_FUSION=1` for `--mtp` comes from the roadmap (llama.cpp issue #29168), not tested.

## E. Decisions / measurements that are yours

- **Launcher defaults** (`-t 7 --prio 2 --load-mode none`) are what roadmap §12 bakes in, but your current production flags don't have them and Phase 1 (`01_phase1_os_tuning.sh`) is what decides whether they help. To drop one: `LLAMA_LOAD_MODE= llama --serve --stuart-creator …` (empty = omit), `LLAMA_THREADS=…`, `LLAMA_PRIO=…`. Check with `llama --serve --stuart-creator --print` (starts nothing).
- **All gates** (G0, G1, G2, H6, H9, P1, MTP acceptance, prompt reuse) need runs on the real model. The scripts compute and print PASS/FAIL; I have no real numbers for any of them.
- **Track 2 parity (E1/E2):** `pelias_serve/ref_dump.cpp` makes the reference dumps from stock llama.cpp and `e1_e2_parity_harness.py` compares them. The comparison is done; the other side (`pelias_serve_*` dumps from the E4 executor) can't exist until E4 does.


---

# Added 2026-10-20 evening: the road to a universal build on GitHub (nothing in F, G or H has been started)

**Rule from the author:** none of this is prepared for release until all the testing the author can do for the current model is complete. That means the RUNBOOK item "PLAN TO FINISH THE IMMEDIATE BUILD" (finish-1 to finish-6 and checks 1 to 7), then the evidence items in F1 below.
**Target folder (created empty on 2026-10-20, nothing moved into it):** `<project>\pelias-engine-release\` (WSL `<project>/pelias-engine-release/`). Its root holds the draft `README.md` and the draft `LICENSE`.
**Project front page (draft):** `pelias-engine-release\README.md`. Sections A-E above date from the first package and are partly stale (the Pass 1 and cold-path work came after them; see the RUNBOOK).

## F. Task list: universal build, published on GitHub

**F1. Evidence needed before any public claim**
- [ ] Hit rate of a profile-chosen bank on held-out text; end-to-end speed against the best-tuned stock llama.cpp on the same machine; KL divergence against stock logits (RUNBOOK finish-2, 5, 6).
- [ ] Repeat on at least one more machine, and on at least one more MoE model from a different architecture family.
- [ ] Quantisation sweep on the current model family: at least Q4_K_M and Q6_K now; Q5_K and Q8_0 once their kernels exist (F3). The README's quantisation reasoning (section 6) is a hypothesis until this is done.
- [ ] A dense (non-MoE) model probe of the cold-bandwidth-only effect (README section 6: open question).
- [ ] A server-class machine if one can be borrowed or rented (many-channel DRAM, large VRAM).
- [ ] Before the runs: write down what counts as "works" (thresholds derived from the model, not tuned afterwards).
- [ ] Settle the open question in README section 3: time the CPU worker alone, to see whether the stock CPU path (26-27 GB/s) is compute-bound.

**F2. Universal measurement framework (parked task #17)**
- [ ] One tool that measures, on any system, what the speed model needs: RAM bandwidth by thread count, PCIe alone, combined CPU+GPU, handoff latency, VRAM size, CPU dequantise-and-dot throughput alone, expert sizes and quant types from the GGUF header, the spine from a short layer sweep, the hit rate from a short trace.
- [ ] Output: machine-readable JSON, a go/no-go against stock llama.cpp, and a projected tokens/second with an interval from the measured spreads.
- [ ] Validate its projections against real end-to-end runs before they are published.

**F3. Backends and quantisation formats**
- [ ] GPU kernels for the other formats (Q2_K, Q3_K, Q5_K, Q8_0, the IQ types, MXFP4 if used). Only Q4_K and Q6_K exist.
- [ ] For each non-NVIDIA backend, a tiny probe (like `07_pin_modes.cu` and `03_handoff_latency.cu`) of README requirements R1 (zero-copy host reads), R2 (flag visibility in microseconds) and R3 (a waiting kernel is allowed to wait) BEFORE writing kernels: HIP (AMD), SYCL / Level Zero (Intel Arc), Vulkan.
- [ ] Fallback when R2 or R3 fail: the sync-per-layer handoff (94 us per layer measured on NVIDIA), priced by the model.
- [ ] Verify every vendor API statement in README section 5.1 against the vendor documentation (they are marked unverified today).

**F4. Platforms**
- [ ] Linux native, Windows native (WSL2 done). Huge pages, NUMA and chiplet effects, Intel hybrid-core thread pinning, AVX2 / AVX-512 / AMX paths.

**F5. llama.cpp integration** (details in the RUNBOOK plan, finish-4)
- [ ] Rebase the tier patch onto current master; pin the base commit; decide patch-in-repo versus upstream proposal.
- [ ] CI that builds the patched tree and runs the CPU stand-in tests.
- [ ] Compare with the mainline expert cache (`--moe-cache-mib`) and the other projects listed in README section 10 before any novelty statement.

**F6. Trainer**
- [ ] Decide: separate repository (recommended) or `trainer/` here. Refine it first (author's plan).
- [ ] Open question (author, 2026-10-20): can the Pelias training method improve the quality of 1-3 bit quantised models? The engine cannot (accuracy is the file's property); the trainer might. Test path: fine-tune and quantise at 2-3 bit with and without the trainer, compare by KL divergence and task sets. Unverified.
- [ ] Licence review of its dependencies and patches: Unsloth core Apache-2.0, but optional parts AGPL-3.0 and `unsloth-zoo` LGPL-3.0 / AGPL-3.0 per their GitHub pages (checked 2026-10-20).

**F7. Repository hygiene (the scrub)**
Measured on the development tree on 2026-10-20 by grep (102 text files): 25 contain hard-coded `C:\AI` / `/mnt/c/AI` paths; 98 mention "stuart" (mostly the path in each file's first-line comment); 0 contain "user"; 0 contain a Hugging Face token pattern.
- [ ] Replace hard-coded paths with configuration (`lib/paths.py`, `env.sh`, every header comment that prints a path); the numbered scripts print `moe_track/` paths.
- [ ] Remove personal and project-private names and text (persona names; seed prompts and corpora in `trace/` and `fork/`: check what they contain before they are copied).
- [ ] Secret scan again at release time; start a FRESH git history (never push the development tree's history).
- [ ] `.gitignore`, `.gitattributes`, line endings; SPDX headers on every source file once the licence is confirmed; drop the dev-only `MANIFEST*.sha256`.
- [ ] Make `tests/run_all_tests.sh` find its own location, and document which tests need a llama.cpp source tree and tiny models.
- [ ] List the Pass 1 tests in `run_all_tests.sh` and `test_pelias_serve.py`; add `07_pin_modes.cu` to the build script (also on the RUNBOOK plan).

**F8. Documentation**
- [ ] Design note (the three mechanisms and the speed model), porting guide (requirements R1-R5), methodology (the five labels and how each number is produced), benchmark protocol and the matrix in README section 12 (model type x machine class x GPU vendor x quantisation x size versus memory), reproducibility (pinned commits, exact build lines), limitations, FAQ.

**F9. Legal and community files** (the author decides; nothing here is legal advice)
- [ ] Confirm the licence (draft: Apache-2.0); copyright holder name and year; `NOTICE`; `THIRD_PARTY_NOTICES.md` (llama.cpp / ggml MIT text reproduced).
- [ ] Contribution terms (DCO or CLA), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates, `CHANGELOG.md`, `CITATION.cff`.
- [ ] Name check: "Pelias" is also an open-source geocoder (unverified); decide the project name before the repository is created.
- [ ] Prior-art review (README section 10) and a claims review: no "first ever", no number without a label.
- [ ] Review by someone qualified (the author already noted a lawyer is outside what Claude can do).

**F10. Release mechanics**
- [ ] Private repository first; dry run: a fresh clone on a clean machine builds and passes the tests using only the README; then a pre-release tag (v0.1.0) with a limitations note; then public.

## G. Move map (what goes where; NOTHING has been moved; the development tree stays as is so current testing does not break)

All 100 files in `MANIFEST.sha256` are mapped (checked by script on 2026-10-20). "HOLD" = do not release until the note is resolved. Paths are relative to `moe_track\` on the left and to `pelias-engine-release\` on the right.

| From (moe_track) | To (release) | Files | Notes |
|---|---|---|---|
| `pelias_serve/` miss_splitter.h, handoff.cuh, pelias_defs.h, pelias_quant.h, zc_mmvq_q4k_q6k.cuh, cold_weights_view.h/.cpp, cold_pool.h | `src/core/` | 8 | includes use `../lib/pl_cuda.h`; update relative include paths |
| `pelias_serve/` e0_layer_profiler.cuh, e5_deltanet_state.cuh, e6_hot_aware_mtp.cuh, e6_policy_model.py | `src/experimental/` | 4 | not wired into any engine yet |
| `pelias_serve/` ref_dump.cpp, e1_e2_parity_harness.py | `tools/parity/` | 2 | |
| `lib/pl_cuda.h` | `src/portability/` | 1 | the thin GPU API layer |
| `lib/cuda_shim.h` | `tests/support/` | 1 | the CPU stand-in for the GPU API |
| `lib/llama_launcher.sh` | `tools/launcher/` | 1 | paths and the "stuart-creator" name must be configurable |
| `lib/` floor_model.py, gates.py, gguf_min.py, paths.py, speedlog.py, stensors.py | `tools/common/` | 6 | `paths.py` holds the hard-coded roots |
| `microbench/*` | `tools/measure/` | 8 | becomes the base of the universal framework (F2) |
| `trace/*` | `tools/trace/` | 7 | check the seed prompts first (F7) |
| `sim/*` | `tools/sim/` | 4 | |
| `patches/src/llama-pelias-tier.{h,cpp}`, `patches/pelias_tier_v1.patch`, `patches/09_build_pelias_tier.sh` | `patches/llama.cpp/` | 4 | pin the base commit; MIT notices of llama.cpp stay |
| `tests/*` | `tests/` | 33 | `test_cold_path.cu` needs a build line update; some tests need a llama.cpp tree |
| `00_init_workspace.sh`, `01_phase1_os_tuning.sh`, `12_upgrade_llama_launcher.sh`, `13_phase6_run.sh` | `scripts/` | 4 | print `moe_track/` paths |
| `env.sh` | `scripts/env.example.sh` | 1 | no machine-specific values |
| `README.md` (the development run-order table) | `docs/RUN_ORDER.md` | 1 | rewrite paths |
| `UNFINISHED.md` | `docs/STATUS.md` | 1 | rewrite after the testing is done |
| `fork/*` | **HOLD** `tools/fork-compare/` | 6 | compares third-party llama.cpp forks (Schwerz, Parmesan); their licences are unknown (unverified), and the corpus script may hold private text; probably not released |
| `mtp_align/*`, `patches/pelias_v03_patches.py`, `stuart_pelias0.3.py` | **HOLD** `trainer/` | 8 | trainer decision and licence review (F6) |
| `MANIFEST.sha256` | not released | 1 | development checksum list |
| (new) `pelias-engine-release/README.md`, `LICENSE` | stay in the root | 2 | drafts, already there |

**How to move when the time comes (so nothing breaks):** copy, do not move; a script rewrites the paths and include lines in the copies; the CPU stand-in tests run in the new tree and their results are compared with the development tree's; the development tree is left untouched until the new tree has passed.

## H. Licence and third-party notes (draft; mirrors README section 11)

| Item | Licence | Source |
|---|---|---|
| Pelias code | Apache-2.0 (proposed; author to confirm) | `LICENSE` is the standard text (sha256 `cfc7749b...523d30`) taken from the Debian `common-licenses` copy |
| llama.cpp, ggml | MIT | GitHub repository page, checked 2026-10-20 |
| Qwen3.6-35B-A3B (base model; weights NOT in the repository) | Apache-2.0 | Hugging Face model page, checked 2026-10-20 |
| Unsloth core | Apache-2.0; some optional components AGPL-3.0 | GitHub page, checked 2026-10-20 |
| unsloth-zoo | LGPL-3.0 and AGPL-3.0 | GitHub page, checked 2026-10-20 |
| NVIDIA CUDA toolkit | NVIDIA licence; not redistributed | users install it themselves |
| Schwerz / Parmesan forks | unknown | not checked; nothing from them is shipped |
