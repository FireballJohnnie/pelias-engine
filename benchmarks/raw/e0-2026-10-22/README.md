# E0 measurements, 2026-10-22: live hit rate, per-token timeline, KL gate rerun

One machine: RTX 4070 12 GB, Ryzen 7 5800X3D, 4x16 GB DDR4-3200, WSL2 (kernel 6.18.54.1), llama.cpp `2ca15f540` with the tier patch, Qwen3.6-35B-A3B Q4_K_M.
Reference cell: 7.0 GiB hot bank, bypass 32, `--load-mode none`, `-t 6 -tb 8`, `-b 4096 -ub 4096`. The GPU is overclocked (see `machine_state/MACHINE_STATE.md`); nothing here is a claim about other hardware.
The account name in paths was replaced with `user`. Scripts and patches are not part of this batch. The measurement builds were made in a separate copy of the tier tree; the production build was not touched.

## 1. Live hit rate, `hit_counter/` [MEASURED]
A counting callback reads the expert ids of every layer and counts those found in the bank. Three runs of the reference cell, 40 layers each.
| Run | Hit rate, 1-token steps | Selections counted | Hit rate, 2 to 31-token batches | Selections |
|---|---|---|---|---|
| 1 | 0.7753 | 592,000 | 0.7328 | 20,160 |
| 2 | 0.7760 | 633,600 | 0.7325 | 20,160 |
| 3 | 0.7793 | 548,160 | 0.7340 | 20,160 |

- Simulated held-out rate for this bank: 0.806 (`benchmarks/RESULTS.md`, section 8.6). Live is about 0.03 below.
- Lowest layers: 0 (0.61 to 0.63), 1 and 2 (0.63 to 0.65), 34 (0.69). Highest: 20 (0.90 to 0.92), 37 (0.85 to 0.86). Per layer: `hit_rate_by_layer.tsv`.
- Counted on the gauntlet's own prompts, not on the held-out text the bank was chosen from. The 2 to 31-token figure rests on a small sample.
- The counting callback makes the scheduler stop after each counted node, so the speed columns of `E0_hit_counter.tsv` (about 41 to 45 tok/s) are **not valid** and must not be used.

## 2. Per-token timeline, `timeline/` [MEASURED; derived figures marked]
Method: a measurement build stamps the host clock around every scheduler split (`*.host.tsv`) and brackets every GPU graph call with two CUDA events (`*.gpu.tsv`). A streaming client records the arrival time of each generated token (`*_tokens.tsv`) on the same monotonic clock. 300 tokens at short context, then 300 tokens after a 12k-token prompt, temperature 0.
- A: timer off. B: timer on, CUDA graphs on. C: timer on, CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS=1`).
- Each token is 82 splits: 41 CPU splits (the cold experts) alternating with 41 GPU splits.

Median gap between tokens as the client saw it (ms):
| Cell | Short | 12k |
|---|---|---|
| A, timer off | 26.33 | 23.34 |
| B, timer on, graphs on | 31.99 | 28.18 |
| C, timer on, graphs off | 36.50 | 32.56 |
The timer costs 5.7 ms (short) and 4.8 ms (12k) per token. Turning CUDA graphs off costs about 4.5 ms more. The absolute tokens/s of this client (A: 37.9 short, 42.1 at 12k) are lower than the gauntlet's; compare only within these cells.

Where one token goes, cell B, median ms (short | 12k):
| Part | Short | 12k |
|---|---|---|
| Wall time, host clock | 30.3 | 26.5 |
| Cold experts on the CPU (41 splits) | 10.1 | 6.6 |
| Wait for the GPU plus copy, before each CPU split | 17.4 | 17.0 |
| of which GPU compute (sum of CUDA-event durations) | 6.6 | 7.6 |
| Copy back to the GPU | 2.3 | 2.2 |
| GPU launch cost | 0.5 | 0.45 |

The parts sum to the wall time (30.5 against 30.3 ms short). Derived from that:
- The layers run strictly in sequence: GPU, sync and copy, CPU cold experts, copy back, GPU. The hot-expert work is in the next GPU split, so the CPU and GPU do not work at the same time in this graph [DERIVED]. The ceiling model assumes perfect overlap.
- Sync-and-copy beyond GPU compute is 10.75 | 9.45 ms with the timer on. The timer's own cost (5.7 | 4.85 ms) is the most that can sit inside it, so the timer-free figure is at least 5.0 | 4.6 ms per token, 112 to 262 microseconds per layer [DERIVED bounds]. The ceiling model assumes 0.247 ms per token.
- Gap to the ceiling model at the live hit rate (0.776): model 10.8 ms per token (92 tok/s); measured timer-free about 24.6 ms short (client A 26.3 minus 1.7 ms client, sampler and HTTP overhead taken from cell B). Gap 13.8 ms = +7.2 cold experts, +4.9 handoff, +2.8 copy back and launch, -1.1 GPU compute [DERIVED; the 1.7 ms overhead was measured in cell B only, and the timer is assumed not to change compute durations].
- GPU compute (6.6 short, 7.6 at 12k) is at or under the assumed 7.73 ms spine. It includes the hot-bank expert work, so it does **not** settle the 7.4 against 7.73 ms question.
- Open: why the cold-expert time is 10.1 ms for the short prompt and 6.6 ms at 12k (the hit rate by context was not separated); the GPU-to-host clock anchoring in `*.gpu.tsv` is an estimate and was not used for these sums.
- `E0_timeline_overview.png` shows the per-layer and per-token view of cell B.

## 3. KL gate rerun, `kl_rerun/` [MEASURED]
The seven H6 accuracy runs repeated at 23:35, same labels. Every row matches the 18:44 set (`p1-sweep-2026-10-22/results/T_kl_gate_20261022_184430.tsv`) in mean KL, standard error, top-1 and perplexity, to every printed digit. The strict limits (KL 0.005, top-1 99%) are still not met by any tier configuration, and the ruling that the float-spread governs stands.

## 4. Machine state, `machine_state/`
`.wslconfig`, WSL version, the GPU overclock and driver settings (stated by the owner), BAR1 readings and a sequential file-read test. No stock-clock result exists.
