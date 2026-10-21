# Results ledger (engine)

**Status: pre-release, private.** This page ties every number in the front-page README to the raw log it came from, with a label. Nothing here is a finished-engine result: the end-to-end Pelias run has **not** been measured yet (see "Not measured yet" at the end).

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
| End-to-end tokens/s of the engine | run the stage 2 and 3 builds of the README ledger and log them here |
| Hit rate of a profile-chosen bank on held-out text | trace router choices on a held-out split, simulate banks |
| Accuracy against stock (KL divergence of logits) | compare engine and stock logits on the same prompts |
| Overlap of CPU and GPU shares in a real layer | time the real CPU worker and kernel together |
| Whether the stock CPU path is compute-bound | time the CPU worker alone |
| Spine linearity | one more `--n-cpu-moe` point |
| Wider-load cold-path kernel | rewrite and re-run section 3 |

---

## 7. About the logs in `raw/`

* Copies were redacted before publication: the local project and user paths, the model's file name (replaced by `test-model` / `<project>`), and one model reply text were removed. Numbers were not altered.
* Logs not listed here (other test and server logs) are **not** published; they are outside this engine's scope and were not reviewed for release.
* The model files, adapters and training data are not in this repository.
