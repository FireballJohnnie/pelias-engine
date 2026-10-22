# Results ledger (engine)

**Status: pre-release, private.** This page ties every number in the front-page README to the raw log it came from, with a label. Nothing here is a finished-engine result: the end-to-end Pelias run has **not** been measured yet (see section 6). A first intermediate result, the hot bank in VRAM with the cold experts on the CPU, is in section 8.

Labels used throughout:

| Label | Meaning |
|---|---|
| **measured** | Printed by a test run; the raw log is in `raw/`. |
| **derived** | Computed from measured or published inputs; the formula is shown. |
| **not in repo yet** | A number the README states, whose raw log is not included in this repository yet. Treat as author-stated until the log is added. |
| **recorded** | Written in the author's run log, whose raw terminal output could not be found. Treat as author-stated until a log is added. |
| **unverified** | A hypothesis, not tested. |

Test machine: Ryzen 7 5800X3D, 4 x 16 GB DDR4-3200 (two channels), RTX 4070 12 GB on PCIe 4.0 x16, Windows 11 with WSL2 (Ubuntu 26.04.1 LTS, 48 GB given to WSL). Stock build: llama.cpp `b11399-2ca15f540`. Model: a fine-tune of Qwen3.6-35B-A3B, shown in logs as `test-model` (model files are not published).

---

## 1. Hardware floors (measured)

Raw: `raw/hardware-floors/`. Summaries are the `*.json` files; the `.txt` files are the console output of the same runs.

| Quantity | Value | Source file |
|---|---|---|
| CPU reading RAM, best stream rate | 43.77 GB/s | `ram_bw_floor.json` |
| CPU reading RAM, best expert-shaped access (4 threads, 4 KB pages) | 45.81 GB/s | `ram_bw_floor.json`, `02_ram_bw_thp_20261020_1758.txt` |
| Same, 2 MB huge pages, 4 threads | 44.86 GB/s (stream 43.77) | `02_ram_bw_thp_20261020_1758.txt` |
| CPU alone, 7 threads | 45.89 GB/s | `dual_bus_contention.json` |
| GPU zero-copy read of pinned RAM, alone | 26.11 GB/s | `dual_bus_contention.json` |
| Host-to-device copy, alone | 26.21 GB/s | `dual_bus_contention.json` |
| CPU + GPU zero-copy at the same time | 26.56 + 21.06 = **47.62 GB/s** | `dual_bus_contention.json` |
| CPU + host-to-device copies | 24.12 + 25.84 = 49.96 GB/s | `dual_bus_contention.json` |
| All three at once | 25.34 + 11.58 + 11.58 = 48.50 GB/s (1.06x CPU alone) | `dual_bus_contention.json` |
| Sync-per-layer handoff | 94.40 us median, 123.01 us p99 | `handoff_latency.json` |
| Empty kernel launch | 33.08 us median | `handoff_latency.json` |
| Resident-kernel ping-pong, flag only | 3.36 us median | `handoff_latency.json` |
| Resident-kernel ping-pong, GPU also reads 4 KB the CPU wrote | 13.34 us median, 18.71 us p99 | `handoff_latency.json` |

**Things to read carefully**

* **One floor test did not pass.** The test's own "greenlight" asks for the combined rate to reach 1.25x the CPU alone (57.4 GB/s). It reached 1.04x (CPU + zero-copy) and 1.06x (all three). `phase_7a_greenlight` is `false` in `dual_bus_contention.json`. The test's note explains why: dual-channel DDR4-3200 peaks at 51.2 GB/s, so 57.4 GB/s cannot be reached on this machine when the CPU alone already reads 45.9. The front-page README claims 93% of the DRAM peak (47.62 of 51.2 GB/s), not a 1.25x gain.
* **Run-to-run spread.** CPU alone read 42.43 GB/s in `04_dual_bus_20261020_1744.txt`, 43.72 in `04_dual_bus_thp_20261020_1758.txt`, and 45.89 in the final run (`04_dual_bus_rev2_20261020_1808.txt`). The README quotes the final (highest) run. Combined rates in all runs sat between 47.6 and 50.0 GB/s.
* **Superseded handoff run.** `03_handoff_20261020_1744.txt` printed `launch_after median 0.00 us` and its own warning "2 CPU-side waits timed out; treat the numbers as invalid". The test was revised; `03_handoff_rev2_20261020_1808.txt` is the run the README uses. The invalid run is kept for transparency and is **not** used for any claim.
* **Aborted attempts.** `02_ram_bw_20261020_1737.txt` and `02_ram_bw_20261020_1744.txt` stopped because a server was running ("results would be distorted"). `03_handoff_20261020_1737.txt` and `04_dual_bus_20261020_1737.txt` stopped on a missing binary. No results came from these.
* **Pinned-memory modes** (`pin_modes.json`, 2.3 GiB; `pin_modes_20gib.json`, 20 GiB; memlock limit 65,536 KiB, about 47 GiB available at start). `cudaHostAlloc` mapped memory reached the full 20,448 MiB with device and host pointers identical. `cudaHostRegister` mapped (device pointer) also reached 20,448 MiB, with different device and host addresses. Variants asked through the host pointer reached only the first 144 MiB chunk.
* **Expert sizes** (`expert_layer_sizes.json`): 40 layers x 256 experts, top-8; total routed-expert bytes 19,503,513,600 (18.164 GiB); average 1,904,640 B per expert; 20 layers with Q4_K down projections and 20 with Q6_K; derived active bytes per token 0.6095 GB. The README rounds this to 608 MB (8 x 40 x 1.90 MB).

---

## 2. Stock llama.cpp baselines (measured)

Raw: `raw/stock-llama-cpp-speed/` (`.txt` summary, `.json` full rows, `.jsonl` one row per request). Sampling: temperature 1.0, top-p 0.95, top-k 20, seed 20261019, up to 256 generated tokens, 3 repeats per depth, 2 warm-up requests not counted. Decode tokens/s, median (range), at short (61-token) / 4k / 12k prompt depth.

| Run (file prefix) | Setup | Short | 4k | 12k |
|---|---|---|---|---|
| `20261019_231714_q5km-kvq8_0-ctx32k` | Q5_K_M file, KV q8_0, ctx 32,768, experts on CPU | 29.2 (29.1-29.2) | 28.5 (28.3-29.2) | 27.0 (26.8-27.8) |
| `20261020_020015_q4km-kvq8_0-ctx32k` | Q4_K_M, ctx 32,768, experts on CPU | 32.1 (31.7-32.5) | 30.7 (30.1-31.2), 2 of 3 reps | 30.4 (30.3-30.8) |
| `20261020_025525_q4km-kvq8_0-ctx131k-ub2048` | Q4_K_M, ctx 131,072, ub 2048, experts on CPU | **31.8** (29.9-32.2) | **31.5** (30.7-31.5) | **31.3** (30.7-31.6) |
| `20261020_032031_q4km-ncmoe28-ctx131k-ub2048` | same, `--n-cpu-moe 28` | **41.1** (41.1-41.2) | **40.2** (39.8-40.6) | **38.8** (37.9-40.0) |
| `20261020_174340_q4km-ncmoe28-lmnone-ctx131k-ub2048` | same as above plus the extra flag `--load-mode none` (as recorded in the run note; the audit script's comment in `phase1_audit.txt` says it loads into ordinary anonymous memory, i.e. 4 KB pages) | 41.7 (41.6-42.2) | 41.5 (40.9-41.7) | 40.4 (39.9-40.5) |

* The README's **31.8 / 31.5 / 31.3** are the third row; its **41.1 / 40.2 / 38.8** are the fourth.
* The fifth row is a later repeat with one extra flag and is **not** used in README figures. It is 1.5% to 4.1% higher than the fourth row. Run-to-run differences of that size mean the "2.3x" and "3.0x" multiples in the README should be read as approximate. (Derived check: 95.0 / 41.1 = 2.31 and 95.0 / 41.7 = 2.28.)
* GPU memory in use is for the whole card including the desktop: 10.2-10.4 GiB (fourth row), 10.5-10.6 GiB (fifth), of 12.0 GiB.
* The fifth row's launch record is `raw/hardware-floors/phase1_launch_q4km-ncmoe28-lmnone-ctx131k-ub2048.txt`. Its command line contains `-c 32768` and, later, `-c 131072`; the later one wins, and the server log line in `phase1_run_lmnone_20261020_1741.txt` confirms `n_ctx_slot = 131072`. No launch record is included for the other rows.

---

## 3. Cold-path test (measured, synthetic data)

Raw: `raw/cold-path-tests/`. One layer of the engine's cold path (row splitter, GPU kernels, CPU hand-off, combine) on **synthetic** Q4_K / Q6_K data in mapped pinned memory, against a double-precision reference. Error bounds are derived on paper (rounding-error bound gamma_n from Higham, *Accuracy and Stability of Numerical Algorithms*, section 3.1; `expf` taken as 2 ulp), not tuned to pass.

| Run | File | Result |
|---|---|---|
| 1 | `cold_path_run1_one_failure.txt` | 11 checks passed, **1 failed**: "all rows on the GPU (no handoff)", with gate/up 3 and SwiGLU 3 values outside the bound. |
| 2 | `cold_path_run2_all_pass.txt` | **12 of 12 passed** (2 lane-maths checks, 2 synthetic layers set up, 8 split scenarios). |

* **Cause of the failure in run 1:** ordering between the default stream and a non-blocking stream. The fix was a `cudaDeviceSynchronize()`. No bound was loosened. The failing log is kept on purpose.
* Kernel read speed in run 2: **median 9.59 GB/s** over 8 passes (range 6.49-9.62), against 26.2 GB/s for the checksum kernel. The cold-path kernel is therefore well below the PCIe read rate; a likely cause is byte-wise loads (**unverified**). This number sets the "today's kernel" row in the README's speed table.
* "CPU stand-in worker" times in these logs are placeholders and mean nothing about CPU speed.
* The test is synthetic. It does **not** show that real model outputs match stock; that comparison (KL divergence) is not run yet.
* Build line recorded in the test log: `nvcc -O2 -std=c++17 -arch=sm_89 ... -x cu test_cold_path.cu`; the include path in it points at the development tree layout, which differs from this repository's. The test source is not published yet.

---

## 4. The derived speed ceiling (derived, not measured)

```
time per token = spine + (1 - hit) x (cold bytes per token / cold bandwidth) + 40 x handoff
spine          = 7.73 ms      derived from the stock runs at 0 and 12 GPU layers; assumes cost per layer is linear (untested)
cold bytes     = 608 MB       derived from the measured expert sizes (section 1)
cold bandwidth = 47.6 GB/s    measured, CPU + zero-copy together (section 1)
handoff        = 6.17 us      p99 of the resident-kernel ping-pong in the engine's own handoff test (see section 5)
```

Re-computing the first column from these inputs: 7.73 + 608 / 47.6 + 40 x 0.00617 = 20.75 ms, i.e. 48.2 tokens/s at 0% hit. At 60 / 80 / 90% hit the same formula gives 76.4 / 95.0 / 108.1 tokens/s. The README tables for other machine cases (B-E) reuse the same CPU share, an assumed 0.83 PCIe efficiency and an assumed 0.93 DRAM cap; those assumptions are **unverified** outside the test machine.

These are ceilings. They assume the CPU and GPU shares overlap perfectly. Hit rates of 60-90% are hypothetical until traces give the real one.

---

## 5. Pinning, GPU-read and handoff tests (measured)

Raw: `raw/gpu-read-and-handoff-tests/`. Terminal output saved from the author's runs of the engine's own tests on 2026-10-20 (the saved files are timestamped between 19:23 and 19:57 local time; the order below follows them). Paths, the user name and the model file name were replaced.

| File | What it is | Result |
|---|---|---|
| `handoff_and_host_tests.txt` | Host-only tests (miss splitter, host handoff, cold-weights reader), then the GPU handoff test (a resident kernel and a minimal stand-in CPU worker that copies the request, does 2,048 multiply-adds and posts), then the reader's tests | Handoff correctness: **ALL PASS**, every round combined, accumulator exact. Block 256 was fastest: CPU-side round trip **median 5.05 us, p90 5.64, p99 6.17, max 171.55**; GPU-side median 6.39, p99 7.84. Medians for blocks 32 / 64 / 128 / 256 / 512 / 1024: 10.93 / 7.17 / 5.40 / 5.05 / 5.06 / 5.15 us. The file ends with `FAILED`: that is the reader's real-file test, run with the literal placeholder path `/path/to/model.gguf`, which does not exist. It is not a handoff failure. |
| `gpu_read_4layers_run1_pin_check_failed.txt` | Real-file test, 4 layers (1.95 GiB) | **FAIL.** `pin()` reported 0 of 12 regions ok because the GPU address differs from the CPU address, and this run's check required them to be identical. The file loaded and byte-verified (4.0 s, 0.52 GB/s). |
| `gpu_read_4layers_run2_pin_ok.txt` | Same test | **ALL PASS.** 12 regions pinned in 0.30 s, 4 of 4 layers readable by the GPU; the GPU-minus-CPU address offset was constant within the run. The check now reads "the GPU address may differ from the CPU address". |
| `gpu_read_4layers_run3_gpu_read_ok.txt` | Same test plus the GPU read | **ALL PASS.** 12 regions pinned in 0.33 s; a GPU kernel read 12 matrices (1.95 GiB) through the device pointers: 0 whole-tensor checksum mismatches, 108 sampled rows, 0 wrong; checksum-read speed 26.23 / 25.85 / 26.26 GB/s (median 26.23; includes the hash arithmetic). The offset differed from run 2 (-136887765303296 against -125428842889216 bytes), so it must never be hard-coded. |

The stand-in CPU worker means the handoff numbers do not yet include a real CPU workload or other kernels in the stream (section 6). The "emulated round trip (32 host threads, NOT hardware)" line in the first file is a host-only check and says nothing about the GPU.

**Full-model figures with no raw log yet.** The README's 40-layer figures come from the same test run on all 40 layers: 18.16 GiB loaded in 37.7 s (0.52 GB/s) and byte-verified, 120 regions pinned (0 failed) in 0.82 s, 40 of 40 layers readable, 120 matrices with 0 checksum mismatches, 1,080 sampled rows with 0 wrong, checksum-read 26.21 / 26.17 / 26.21 GB/s. They are recorded in the author's run log (label: **recorded**), but the raw terminal output was not found. The 4-layer runs above agree with them on load rate (0.52 GB/s) and read speed (26.2 GB/s). To close the gap: repeat the run on the full file with the output saved to a file, and add it here.

The 6.17 us figure feeds the ceiling in section 4. With the microbenchmark's 18.71 us p99 (a flag plus a 4 KB read, in section 1) instead, the 0%-hit ceiling would be 7.73 + 12.77 + 40 x 0.01871 = 21.25 ms, or 47.1 tokens/s (derived), about 2% lower.

---

## 6. Not measured yet, and the path to each number

| Unknown | Path |
|---|---|
| End-to-end tokens/s of the full engine (stages 3 and 4) | stage 2 is measured (section 8); run the stage 3 and 4 builds of the README ledger and log them here |
| Hit rate of a profile-chosen bank on held-out text, in a live run | simulated 0.765 at 6.1 GiB and 0.827 at 7.5 GiB (sections 8.5, 8.7); add a live hit-rate counter to the patch and log it |
| Accuracy against stock, decision on the limit | measured (section 8.3): the strict limits are not met even by stock at another batch size; the tier is inside that spread at batch 8 and 16. Open: which definition the author adopts (strict or float spread) |
| Overlap of CPU and GPU shares in a real layer | time the real CPU worker and kernel together |
| Whether the stock CPU path is compute-bound | time the CPU worker alone |
| Spine linearity | one more `--n-cpu-moe` point |
| Wider-load cold-path kernel | rewrite and re-run section 3 |
| Whether the helper-deduplication patch is still needed after the quantiser fix (section 8.4) | rebuild without `mm_ids_dedup.patch` and repeat the crash sweep and the batch-16 accuracy run |
| Why `--load-mode none` costs decode speed on the tier build (46.9 against 52.4 tokens/s short at bypass 9, different launches, section 8.6) | repeat 3 launches per setting; write the launch settings into the speed file |
| Stock path at chunks of 25 to 31 tokens, and the cause of the step between 24 and 32 (section 8.8) | sweep chunks 25 to 31; read the scheduler's offload rule |

---

## 7. About the logs in `raw/`

* Copies were redacted before publication: the local project and user paths, the model's file name (replaced by `test-model` / `<project>`), and one model reply text were removed. Numbers were not altered.
* Logs not listed here (other test and server logs) are **not** published; they are outside this engine's scope and were not reviewed for release.
* The model files, adapters and training data are not in this repository.
* `raw/stage2-tier-cuda/` follows the same redaction. In addition, model reply text in the speed reports was replaced by `[reply text removed]`, and the commands that produced the terminal output were cut: only their output is kept. The model's reply to the speed tool's prompts is not part of any number here. Some terminal output of 2026-10-22 (the passing crash sweep, the sanitizer runs after the fix, the memory-dump and geometry diagnostics, the batch-16 accuracy run's console and the chunk-latency sweep) was pasted into a chat and **not saved as files**; it is not in `raw/`. Where a number below comes from that output it is marked "raw output not kept". `raw/stage2-tier-cuda/hot-sets/` holds the expert lists (and their simulated hit rates), `patches/stage2-tier-cuda/` the loose patches. The base tier patch and the tier source are **not** in this repository.

---

## 8. Stage 2: the hot bank on CUDA (2026-10-21)

**What ran.** A patched llama.cpp (the "tier", on `2ca15f5`) that keeps a static bank of hot experts in VRAM (about 6.1 GiB, in all 40 layers, chosen from recorded router traces) and leaves every expert on the CPU as well. For small batches each layer is computed twice, hot experts from the bank and the rest from the CPU copy, and the two results are added; for larger batches the stock path runs. The base patch is not in this repository; the later loose patches (section 8.4) are in `patches/stage2-tier-cuda/`. Later days' runs in sections 8.7 to 8.9 used bank sizes of 6.4 to 8.5 GiB. Same machine and model as sections 1 to 2. "Stock" below is `--n-cpu-moe 28` (the best layout that left room for other GPU use).

### 8.1 Matched speed comparisons (measured)

Generation tokens/s, median of three requests (ranges are in the raw files); prompt tokens/s at 12k context; GPU peak is the whole card including the desktop. Raw: `raw/stage2-tier-cuda/speed/` (file stamp in the first column).

| Stamp | Run | Short | 4k | 12k | Prompt at 12k | GPU peak GiB |
|---|---|---|---|---|---|---|
| 121851 | stock, context 32,768 | 41.2 | 42.3 | 40.2 | 419.0 | 8.4 |
| 121147 | tier, bypass 9 | 44.3 | 44.5 | 43.4 | 279.8 | 9.0 |
| 132536 | stock, repeat | 42.0 | 41.9 | 41.6 | 408.5 | 8.4 |
| 132830 | tier, repeat | 44.5 | 42.7 | 45.1 | 287.7 | 8.9 |
| 133216 | stock, `-ub 2048` | 41.9 | 42.4 | 41.8 | 1253.2 | 8.6 |
| 133352 | tier, `-ub 2048` | 44.7 | 46.1 | 43.2 | 935.5 | 9.2 |

* **Derived:** mean gain over the three pairs +2.8 / +2.2 / +2.7 tokens/s at short / 4k / 12k (about +5 to +7%). Two launches of the same setting differ by up to 1.4 (stock, 12k: 40.2 vs 41.6) and 1.8 (tier, 4k: 44.5 vs 42.7), so a gain below that is noise.
* **Prompt reading (measured, derived reason).** Tier over stock at 12k: 0.67 and 0.70 at the default micro-batch, 0.75 at `-ub 2048`. The tier copies the experts of all 40 layers to the GPU for each large chunk; stock copies 28 layers' worth: 28/40 = 0.70.
* **Single runs of other settings (measured, one launch each, not separable from the drift above).** Stamps 122349 (`-t 7`): 46.8 / 47.2 / 44.8. 122732 (`--poll 100`): 45.8 / 45.9 / 45.1. 123101 (context 131,072 with `-ub 2048`): 44.2 / 45.1 / 46.8, 12k range 40.9 to 47.6, GPU peak 10.6 GiB. These files share one label, so the setting of each is **recorded** from the run order and is not written inside the files.
* **Not a valid baseline:** an older stored stock figure for 12k context (38.78) is stale; compare only with the matched runs above.
* **Ceiling (derived).** With the simulated hit rate 0.765, spine 7.73 ms, 608 MB of cold experts per token at 26.6 GB/s (the CPU's share, section 1) and 40 x 94.4 us of per-layer sync: 1 / (7.73 + 0.235 x 22.86 + 3.78 ms) = about 59 tokens/s. Measured 44 to 47 is about 75 to 80% of it; the gap is **unexplained**.

### 8.2 Gates from the author's gate script (not published)

* **2026-10-21:** prompt-reading regression on hold; 12k-context target (generation at least 55 tokens/s and prompt reading at least 1,150 tokens/s) failed (best tier run 46.8 and 935.5).
* **2026-10-22, bypass 32, 7.5 GiB set, `-ub 2048`, default load** (stamp 141137; `raw/stage2-tier-cuda/terminal/gauntlet_ub2048_bypass32_hot7.5_8.0_reports.txt`). Measured: generation 53.51 short, 53.0 at 4k, 49.05 at 12k; prompt reading 920.75 tokens/s at 12k; GPU peak 10.69 GiB. Gate P1: **hold**. Gate G3: **fail** (49.05 against at least 55.0; 921 against at least 1,150; the 11.2 GiB peak limit is met).
* **The script's percentages use its stored baseline** (generation 41.14 short, 38.78 at 12k; prompt 1,192.45), which section 8.1 calls stale at 12k. Against the matched stock run at `-ub 2048` (stamp 133216: 41.9, 41.8, prompt 1,253.2) the figures are, derived: short +27.7%, 12k +17.3%, prompt reading at 12k -26.5%.
* **Noise:** the script's candidate spread was 19.3% at depth 0 and 16.1% at depth 12,000 (baseline 0.3% and 5.5%). A gain smaller than that is not separable.
* **8.0 GiB set at bypass 32** (stamp 141319): 55.3 / 54.6 / 48.7, prompt at 12k 901.5, GPU peak 11.19 GiB. G3 fails on generation at 12k and on prompt reading.

### 8.3 Accuracy against stock (measured)

Raw: `raw/stage2-tier-cuda/h6/`. Reference: the stock build at batch 8 (`tier_ref8`). 16 chunks of 1,024 tokens, 8,176 tokens scored, Q4_K_M, KV q8_0, same text for every row. "Tier" = the tier build with all 40 layers' experts on the CPU plus the bank.

| Candidate | Batch | Mean KL (nats) | Std. error | Top-1 agreement | Perplexity (cand. / ref.) | p99 KL |
|---|---|---|---|---|---|---|
| tier, bank off | 8 | 0.001731 | 0.000090 | 98.740% | 4.112086 / 4.098874 | 0.024949 |
| tier, 6.1 GiB set (first run) | 8 | 0.010568 | 0.000428 | 97.126% | 4.111559 / 4.098874 | 0.146915 |
| same, repeated | 8 | 0.010568 | 0.000428 | 97.126% | 4.111559 / 4.098874 | 0.146915 |
| tier, 8.5 GiB set | 8 | 0.011856 | 0.000729 | 97.248% | 4.102311 / 4.098874 | 0.148767 |
| stock, `-b 16 -ub 16` | 16 | 0.017353 | 0.000668 | 96.490% | 4.117205 / 4.098874 | 0.235869 |
| stock, `-ub 1024` | 1024 | 0.018286 | 0.000969 | 96.318% | 4.130661 / 4.098874 | 0.233612 |
| tier, bypass 32, 7.5 GiB set | 16 | 0.017866 | 0.000805 | 96.159% | 4.137987 / 4.098874 | 0.232517 |

* **Strict limits** (mean KL at most 0.005, top-1 at least 99%): **not met by any bank run, and not met by stock itself at another batch size** (0.017353 at batch 16, 0.018286 at 1,024).
* **Float-spread definition** (no worse than the natural spread between two stock runs at different batch sizes): at batch 8 the tier (0.010568 and 0.011856) is below the stock spread (0.017353 and 0.018286); at batch 16 the tier (0.017866) is above stock (0.017353) by 0.000513, smaller than either standard error (0.000805, 0.000668). **Which definition counts is the author's decision.**
* **The bank adds KL, but not more than the stock spread.** Bank off: 0.001731; bank on: 0.010568 at the same batch and placement. The repeat of the 0.010568 run gave identical numbers to six digits, so the run is deterministic and the figure is not launch noise.
* **Bank size of the batch-8 rows.** The files do not record the hot set. The first run (`tier8`, 2026-10-21) used the 6.1 GiB set, the only one that existed then; the repeat (`tier8_rep`, stamp 115348 on 2026-10-22) is presumed to be the same set (**not recorded**). The 8.5 GiB row and the 7.5 GiB batch-16 row are named in their file names.
* **The dual path ran.** The stats files show 40 layers with the bank; dual-path layer builds were 3,280 at 8 tokens for the batch-8 rows and 3,280 at 16 tokens for the batch-16 row.
* **Reading (derived).** Every row has a long tail (p99 0.025 to 0.236, maximum 0.41 to 4.6; stock included), so the tail is not specific to the bank. A likely cause is top-k routing flips from small numeric differences, **not tested**. Also not tested: which tokens flip, and whether the bank changes the flip rate beyond the amount the batch size does.

### 8.4 Crash for chunks of 9 or more tokens: cause found and fixed (measured)

**Real limit.** The server splits a prompt of N tokens into a chunk of N-4 and a final chunk of 4 (`terminal/chunk_split_log_N12_N61.txt`: N=12 gave 8 + 4; N=61 gave 57 + 4). The dual path is used only when a chunk is smaller than the bypass value, so a prompt of 13 tokens (chunk 9) was the first to fail at bypass 32. Before the fix, at bypass 32: prompts of 1 to 12 passed, 13 to 34 failed, 61 passed (`terminal/crash_sweep_bypass32_tokens_1-34.txt`, `terminal/hole_fix_build_and_sweep2.txt`). At bypass 9 (the production setting) the dual path never sees a chunk of 9 or more: prompts of 8, 12 to 17 and 61 tokens passed 3 of 3 each (`terminal/bypass9_vs_13_latency_N13_crash.txt`); at bypass 13 the prompts of 8 and 12 passed and those of 13 to 17 and 61 failed 3 of 3 each (same file).

**Chain of evidence.**

1. Sanitizer before any fix: the first fault was a 4-byte write in the scatter variant of the activation quantiser (`terminal/sanitizer_before_hole_fix_N13_N34.txt`). A fix there (skip negative ids in both quantiser paths; pre-fill the id map with -1 and the output with 0, `hole_fix`) did not remove the crash.
2. A second hypothesis, rows that repeat the placeholder expert inside the sorted-by-expert helper, led to a patch that counts each expert once per row (`mm_ids_dedup.patch`). It also did not remove the crash. Whether it is still needed is **untested**.
3. Two diagnostic patches (`mmq_dump_diag.patch`, `mmq_dump_diag2.patch`, switched on by `PELIAS_MMID_DUMP=1`, otherwise inert) printed the id map after the helper and the buffer geometry. Result (raw output not kept): the faulting address minus the activation buffer's start equals 0x7FFFFFFF800 bytes = 4,294,967,295 rows x 2,048 bytes each, so a row index of 4,294,967,295 was read. Every bank-path chunk of 9 or more tokens has unused rows with id -1.
4. **Cause.** In the non-scatter quantiser, `const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;` mixes a signed 32-bit id with an unsigned block index. The conditional operator converts -1 to 4,294,967,295 before widening, so the following `if (i01 < 0) return;` never fires.
5. **Fix** (`quantize_signed_id.patch`): widen each side to 64 bits first, `(int64_t) ids[blockIdx.x] : (int64_t) blockIdx.x`.

**Result after the fix** (raw output not kept): fresh server per length, prompts of 8 to 61 tokens at bypass 32, no CUDA error in any run.

**Explained now, not before.** The two sanitizer readings of 2026-10-21 (a write 2.4 GB past an 8 KB buffer, a read 8.8 TB past a 2 MiB buffer) are consistent with a wrapped row index; they were not decoded at the time.

### 8.5 Cache-policy simulation (derived, on recorded router traces)

Raw: `raw/stage2-tier-cuda/sim/cache_policy_simulation_6p1gib.json`. 121,160 decode steps: 84,503 to choose the bank, 36,657 held out. Hit rate is byte-weighted: bytes of the chosen experts found in VRAM over bytes of all chosen experts. Bank 6.1 GiB, 3,426 experts.

| Policy at 6.1 GiB | Held-out hit rate |
|---|---|
| Static bank | 0.765 |
| Global LRU pool | 0.865 |
| Per-layer LRU, same slots | 0.862 |
| Half static, half LRU | 0.858 |

Static bank by size (GiB: hit rate): 2.0: 0.43, 3.0: 0.54, 4.0: 0.63, 5.0: 0.70, 5.6: 0.74, 6.0: 0.76, 8.0: 0.85. By domain with a bank from all domains: agent 0.790, code 0.753, persona 0.799, reasoning 0.674. **Caveats:** the LRU inserts on every miss, the copy traffic that costs is not counted, the caches start empty, and the split is a fixed 50/50. These are paper numbers, not runs.

### 8.6 Bank size sweep, bypass 9 (measured; derived hit rates)

Raw: `raw/stage2-tier-cuda/speed/20261022_*` and `raw/stage2-tier-cuda/hot-sets/`. Default micro-batch (the prompt rate of about 300 tokens/s at 12k points to it; the setting is **not written in the files**). Generation tokens/s, median of three. Held-out hit rate is the simulated byte-weighted figure stored in each hot-set file.

| Hot set (GiB) | Stamp | Short | 4k | 12k | GPU peak GiB | Held-out hit rate (simulated) |
|---|---|---|---|---|---|---|
| 6.1 | 114018 | 44.2 | 45.1 | 44.8 | 9.0 | 0.762 |
| 6.1 | 125811 | 47.7 | 47.6 | 46.7 | 9.0 | 0.762 |
| 7.0 | 130522 | 48.4 | 50.8 | 49.1 | 9.9 | 0.806 |
| 7.5 | 130854 | 52.4 | 52.5 | 50.0 | 10.4 | 0.827 |
| 8.0 | 131225 | 54.6 | 51.8 | 49.5 | 11.0 | 0.847 |
| 8.5 | 131550 | 55.7 | 52.9 | 53.4 | 11.4 | 0.865 |

* **Trend (measured, one launch each):** generation rises with the bank, short from 44.2 to 55.7 over 6.1 to 8.5 GiB.
* **Noise is larger than section 8.1 stated.** Two launches labelled 6.1 GiB differ by 3.5 at short (44.2 vs 47.7; the settings of the first are not confirmed). A single step of 0.5 GiB is therefore not separable; the trend over 2.4 GiB is.
* **Two more 7.5 GiB launches with other settings** (the settings are in the gauntlet console output, not in the speed files; stamps in the raw `speed/` folder): 132648, `--load-mode none -t 6 -tb 8`: 50.1 / 47.7 / 45.0 generation, 668.0 prompt tokens/s at 12k, peak 10.44 GiB. 132907, `--load-mode none -ub 2048 -t 6 -tb 8`: 46.9 / 44.3 / 45.4, **1,680.3** prompt tokens/s at 12k, peak 10.67 GiB; the script's gate P1 **passed** on this one (generation +14% short, +17% at 12k, prompt reading +41% against its stored baseline, which section 8.2 calls stale at 12k). The rows in the table above ran with the default load and micro-batch (about 300 prompt tokens/s); this is inferred from the rate, not written in the files.

### 8.7 Bypass value (measured; raw output not kept for the crossover)

* **Bank path against stock path by chunk size, after the fix.** The bank path is faster up to a chunk of 24 tokens, ties from 32, and the stock path's time rises from 250 to 403 ms between chunks of 24 and 32. The cause of that step is an **estimate**: the scheduler's rule for offloading batches of 32 or more tokens to the GPU. Chunks of 25 to 31 were not measured.
* **Consequence:** raising the bypass above 32 gains nothing in these runs. Bypass 32 is the largest useful value found.
* **Bypass 32 with `-ub 2048` and the 7.5 GiB set** (section 8.2): generation 53.5 / 53.0 / 49.0 tokens/s, prompt reading 921 tokens/s at 12k. Bypass 9 with `--load-mode none -ub 2048` reached 1,680 prompt tokens/s with generation 46.9 / 45.4 (section 8.6), the only P1 pass so far. Bypass 32 with `--load-mode none -ub 2048` has **not been run**.

### 8.8 Stock layout, micro-batch, load mode and threads (measured, one launch each)

Stock build, context 131,072, KV q8_0, `-ub 2048` unless noted. Generation tokens/s short / 4k / 12k; prompt tokens/s at 12k; GPU peak GiB. Drift between launches is up to 1.4 (section 8.1).

| Setting | Stamp | Gen | Prompt at 12k | Peak |
|---|---|---|---|---|
| `--n-cpu-moe 27` | 120417 | 43.8 / 42.7 / 41.1 | 1,226.2 | 10.6 |
| `--n-cpu-moe 26` | 120758 | 45.8 / 43.9 / 42.8 | 1,222.5 | 11.0 |
| `--n-cpu-moe 25` | 121055 | 45.6 / 44.3 / 44.8 | 1,294.9 | 11.5 |
| 27, `-ub 4096` | 123225 | 43.1 / 42.5 / 42.3 | 1,170.8 | 10.6 |
| 27, `--load-mode none` | 123535 | 43.7 / 42.8 / 41.0 | 2,144.7 | 10.6 |
| 25, `--load-mode none` | 123833 | 45.2 / 42.9 / 43.3 | 2,238.4 | 11.5 |
| 25, `--load-mode none`, `-t 5 -tb 8` | 125013 | 46.6 / 45.2 / 45.9 | 2,230.7 | 11.4 |
| 25, `--load-mode none`, `-t 6 -tb 8` | 124824 | 47.9 / 45.6 / 44.1 | 2,241.5 | 11.4 |
| 25, `--load-mode none`, `-t 7 -tb 8` | 124326 | 46.1 / 45.9 / 45.4 | 2,238.3 | 11.5 |
| 25, `--load-mode none`, `-t 8 -tb 8` | 125203 | 45.7 / 44.6 / 43.1 | 2,235.9 | 11.4 |
| 25, `--load-mode none`, `-t 10 -tb 8` | 125345 | 42.9 / 43.2 / 41.6 | 2,232.9 | 11.4 |

* **`--load-mode none`** about doubles stock prompt reading at 12k (1,226 to 2,145 at 27; 1,295 to 2,238 at 25) with generation unchanged within drift.
* **Threads:** 5 to 7 generation threads are within one drift of each other; 10 is lower (41.6 at 12k vs 44.1 to 45.9).

### 8.9 Not known yet

Live hit rate (only simulated figures exist); why 44 to 55 tokens/s and not the ceiling (section 8.1); whether the helper-deduplication patch is needed; the mechanism of the decode cost of `--load-mode none` on the tier build; whether bypass 32 with `--load-mode none -ub 2048` passes P1; the stock path at chunks of 25 to 31 and the cause of the step at 32; the chunk-size behaviour of the production setting above 61 tokens (not swept).
