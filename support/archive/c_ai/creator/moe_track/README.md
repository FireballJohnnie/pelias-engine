# moe_track — what to run, in order

Folder: `<project>\moe_track\` (WSL: `<project>/moe_track/`).
Nothing here has been run on your machine. The scripts are written not to overwrite earlier results (they stop, or use a new label), and none needs sudo.
**Not finished:** see `UNFINISHED.md` (read it first, it is short).

Stop the server (`llama --stop`) before any step marked **(server off)**. All commands are run from `moe_track/` in WSL.

| # | Step | Command | Gate / result |
|---|------|---------|---------------|
| 0 | folders + tool check | `bash 00_init_workspace.sh` | — |
| 1 | **Phase 1** launch tuning (one change per run) | `bash 01_phase1_os_tuning.sh audit` then `run lmnone`, `run t8` / `t7` / `t6` | P1 |
| 2 | **Phase 2** floors: build, sizes, RAM, handoff, bus, N-sweep | `bash microbench/build.sh` · `python microbench/01_gguf_expert_sizes.py` · `microbench/bin/02_ram_bw_test` **(server off)** · `03_handoff_latency` · `04_dual_bus_test` **(server off)** · `bash microbench/05_n_sweep.sh` | G0 |
| 3 | **Phase 3** record router choices | `bash trace/build.sh` · `python trace/05_collect_traces.py plan` → `run` **(server off)** → `summary` · `python lib/gates.py g1` | G1 |
| 4 | **Phase 4** simulate a VRAM bank | `python sim/waterfill_sim.py` · `python sim/predictor_eval.py` · `python lib/gates.py g2` / `h9` | G2, H9 |
| 5 | **Phase 5** forks + MTP + prompt reuse | `bash fork/05_phase5_fork_gauntlet.sh plan` → `build KEY` → `profile` → `run SPEC…` → `report` · `python fork/5d_prompt_reuse.py` · `python fork/5e_mtp_acceptance.py run` | G3, MTP, reuse |
| 6 | **Phase 6** Pelias-Tier (VRAM hot bank) | `bash patches/09_build_pelias_tier.sh build` · `bash 13_phase6_run.sh hotset` → `speed` → `kl-base` → `kl-compare` → `report` | H6 |
| 7 | launcher upgrade | `bash 12_upgrade_llama_launcher.sh` (shows, writes nothing) → `install` (backs up the old one first) | — |
| 8 | **MTP head alignment** (blk.40) | `python mtp_align/10_train_blk40_mtp_lora.py selftest` → `export-samples` → `bash mtp_align/10a_dump_hidden.sh` **(server off)** → `train` → `python mtp_align/11_merge_blk40_and_export.py --adapter …` (prints the convert / quantize / serve commands) | acceptance |
| 9 | **Pelias v0.3** training patches | `python stuart_pelias0.3.py <stage>` — same stages and options as `stuart_pelias0.2.py`; try `smoke` first. `--v03-off` = plain 0.2, `--v03-offload-xl` = opt-in offload | — |
| 10 | **Track 2** parts (library, not a program) | see below | E1/E2 |

## The new launcher (step 7)

```
llama --serve --stuart-creator [--bg] [--profile long|fast] [--pelias-tier] [--mtp [--mtp-n N] [--mtp-gguf FILE]] [--print] [-- extra flags]
llama --stop        llama --status        llama --version
```
`--print` shows the exact command and environment and starts nothing — use it to look before you launch. `--pelias-tier` needs the step-6 build and hot-set file and refuses without them.
`--mtp-gguf FILE` is for the draft GGUF made in step 8. Overrides (`LLAMA_THREADS`, `LLAMA_PRIO`, `LLAMA_LOAD_MODE=`, `EXPERT_FLAGS`, …) are listed at the top of `lib/llama_launcher.sh`.
To go back: `bash 12_upgrade_llama_launcher.sh restore`.

## Track 2 (`pelias_serve/`)

Header-only parts, each with a test: `cold_pool.h` (E3, CPU expert pool, checked against real ggml), `e5_deltanet_state.cuh` (E5 snapshot/rollback), `e6_hot_aware_mtp.cuh` + `e6_policy_model.py` (E6 draft gate and its estimate), `e0_layer_profiler.cuh` (E0 timings). The engine that would use them (E4) and the server adapter (E7) are **not written** — see `UNFINISHED.md`.
Reference dumps from stock llama.cpp (needed later for E1/E2): `python pelias_serve/e1_e2_parity_harness.py howto` prints the commands; `… selftest` checks the comparison logic.

## Checking the tools themselves (optional)

`tests/run_all_tests.sh` (needs a llama.cpp source tree, two shared-library builds and two tiny models; header explains). These test the code, they are not measurements. Files in `tests/` are never needed for normal use.

## Layout

`lib/` shared helpers, paths, gates, launcher · `microbench/` Phase 2 · `trace/` Phase 3 · `sim/` Phase 4 · `fork/` Phase 5 · `patches/` Phase 6 + Pelias v0.3 patches · `mtp_align/` blk.40 training/export · `pelias_serve/` Track 2 · `tests/` · `MANIFEST.sha256` lists every file with its checksum.
