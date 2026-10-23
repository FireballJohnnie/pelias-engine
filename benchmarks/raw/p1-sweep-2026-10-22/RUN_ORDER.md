# P1 test run order (written 2026-10-22)

Run one script at a time, in WSL, in this order. Each one stops the server on port 8012 and leaves it stopped.
Command form:  `bash /mnt/c/AI/Stuart/Creator/moe_track/p1_tests/<script>`
Results land in `p1_tests/results/` (I read them from C: when you tell me you are done; nothing to paste).
Time = about 1.6 min per variant [DERIVED from the 2026-10-22 log stamps], so totals are [ESTIMATE].

## Part 1: new tests, in this order

| # | Script | Variants | About | What it changes (one thing at a time) | Status |
|---|---|---|---|---|---|
| 1 | `K_logging.sh` | 4 | 7 min | `-lv` level and PELIAS_TIER_VERBOSE | RE-RUN (v2 lost its stats file; v3 sends one SIGTERM) |
| 2 | `A_memory_loading.sh` | 6 | 10 min | `--load-mode`: default, none, mmap, mlock, mmap+mlock, dio | RE-RUN (fixed: this build has no `--no-mmap`/`--mlock`) |
| 3 | `C_threads.sh` | 9 | 15 min | `-t`, `-tb`, `--poll`, `--prio` | to run |
| 4 | `D_batch_sizes.sh` | 5 | 8 min | `-ub` 512/1024, `-b` 4096 | to run |
| 5 | `E_layout.sh` | 3 | 5 min | `--n-cpu-moe` 40/39/38 (38 may run out of VRAM; it is then logged FAILED) | to run |
| 6 | `F_kv_attention.sh` | 3 | 5 min | KV f16 vs q8_0; flash attention off (may run out of VRAM) | to run |
| 7 | `G_scheduler_offload.sh` | 4 | 7 min | `--no-op-offload`, GGML_OP_OFFLOAD_MIN_BATCH | to run |
| 8 | `H_machine_state.sh` | 3 | 5 min | nothing: 3 identical runs + nvidia-smi log = drift and clocks | to run |
| 9 | `I_server_extras.sh` | 4 | 8 min | `--ctx-checkpoints` 2/8; prompt-reuse test | to run |
| 10 | `L_speculative_mtp.sh` | 3 | 6 min | MTP draft 2/3 (the GGUF has MTP tensors; whether the tier build accepts it is unknown) | to run |
| 11 | `M_cuda_graphs.sh` | 2 | 3 min | GGML_CUDA_DISABLE_GRAPHS=1 | to run |
| 12 | `N_pinned_memory.sh` | 2 | 3 min | GGML_CUDA_NO_PINNED=1 | to run |
| 13 | `J_sampling_output.sh` | 6 runs on 4 server launches | 15 min | temp 0, no top-k/top-p, top-k 40 + min-p, `--ignore-eos` (fixed length), penalties, plus a base repeat as drift control | NEW, to run |
| 14 | `B_hot_set_content.sh` | 6 | 12 min | hot-set trace mix: existing set, then rebuilt all / persona / agent / code / reasoning (writes only into `results/B_hot_sets/`) | NEW, to run |
| 15 | `O_patch_isolation.sh` | 3 (default) | UNMEASURED, builds | copy of the tier tree with one patch reversed, built in `~/llama-pelias-tier-iso/`: control, no dedup, no diag2. Live tree untouched. | NEW, run last |

Total about 80 min for rows 1 to 12, plus about 27 min for J and B [ESTIMATE]; O adds a full build per variant (time unmeasured).
Notes: O: run `bash .../O_patch_isolation.sh ctrl nodedup` first if you want less build time; `nodiag12` and `noquant` (expected crash) are optional extras. O deletes nothing; the copies stay for you to remove.
Notes: B: a test rebuild of the "all" set gave 4198 experts vs 4170 in the production b7.5 file, so the script also runs the production set as its control.
Notes: `--load-mode mlock` and `mmap+mlock` may only warn that the memory lock failed (WSL limit) and then behave like mmap; I will check the logs. `--prio 2` may do nothing without extra rights.

## Part 2: reproductions of work already done (no need to run for the speed question)

Header of each script says COMPLETED. Run only if you want a second launch of the same thing.

| Script | Reproduces | My pick |
|---|---|---|
| `R_p1_grid_load_none.sh` (7 variants) | the P1 grid, bank x bypass, `--load-mode none` | RE-RUN once at the end: a second launch of the grid shows how big the launch-to-launch drift is (earlier spreads were 5-21%) |
| `U_crash_sweep.sh` | bypass-32 crash sweep across 12 prompt lengths | RE-RUN once at the end: regression check that the quantize.cu fix still holds on the current build |
| `T_kl_gate.sh` | H6 KL runs (7 runs, several minutes each). `tier8_b85` and `tier16_b75_by32` are RECONSTRUCTED from result file names | optional; only if you want the reconstruction confirmed |
| `P_bank_size_bypass9.sh`, `Q_bypass32_default_load.sh`, `S_stock_layout_ub_load.sh`, `V_bypass_latency_by_length.sh` | bank sweep at bypass 9, default-load bypass 32, stock sweeps, bypass latency by length | not needed |

## Part 3: scripts still to be made (none of A to V is missing; these are things that need a decision or new code)

| Item | Why it is not scripted | What I need from you |
|---|---|---|
| B: live hit-rate counter | needs a patch to the tier (a counter in `llama-pelias-tier.cpp`, printed into the stats file) | your OK to write the patch; it stays a separate loose patch until you say otherwise |
| B: static vs LRU bank | needs new code in the tier (a changing bank) | decision whether it is worth building |
| O: pin_cold_to_cpu, cuda_hole_fix, pelias_tier_v1 | do not reverse alone on the current tree (later patches touch the same lines) | your OK to isolate by hand-editing the sources in a copy |
| J: thinking on/off | speed tool fixes enable_thinking=false; effect on tokens/s per token expected small [ESTIMATE] | nothing unless you want it tested |

The probes are done; their output is in the scripts above.

## Part 4: follow-up series W (added 2026-10-22 after the C to N, J, B sweep was read)

Run after R, U, T, P, Q, S, V and the O re-run are finished (GPU and CPU must be free):
`bash /mnt/c/AI/Stuart/Creator/moe_track/p1_tests/W_bank70_extras.sh`

- 7.0 GiB bank (`pelias_hot_set_b7.0.txt`, cap 7300), 5 cells x 3 launches in round-robin order: `b70_base`, `b70_ub4096` (-b 4096 -ub 4096), `b70_mtp2`, `b70_mtp3`, `b70_mtp3_ub4096`.
- 15 launches, about 35 to 40 minutes [ESTIMATE from logged start times]. A cell that does not fit VRAM is logged FAILED and the run goes on.
- Output: `results/W_bank70_extras.tsv` (one row per launch), one log per launch in `results/`, and a mean-per-cell table (with min-max and highest peak) printed at the end.
- Reads: does MTP or ub 4096 fit under 11.2 GiB at 7.0 GiB; whether mtp3 holds 12k decode of 55 or more across launches (G3 needs 12k >= 55, prompt >= 1150, VRAM <= 11.2).

## After everything: I read `results/`, interpret, and upload the logs in one batch to the GitHub repo (only when you say so, and never while a test is running).
