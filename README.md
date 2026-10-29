# PELIAS

**Two-path expert serving for Mixture-of-Experts models on split-memory PCs (CPU RAM + discrete GPU).**
*Goal: make a machine whose memory is split across two buses behave, for MoE decoding, like one memory pool driven at the DRAM controller's full rate.*

> **End goal:** to make Pelias (this engine and its trainer) universal for every LLM that uses a Mixture-of-Experts architecture.¹
>
> ¹ *Footnote. This is a goal, not a result.* There is potential for it to happen, with proper benchmarking and training, but it needs additional testing beyond the author's capabilities. While this work opens up better potential speed and accuracy for MoE models, it also has the potential to improve the overall speed and accuracy of non-MoE models. Both are **stretch goals** (**author-stated**). Today's results come from one machine and one model (Qwen3.6-35B-A3B), and this repository holds no non-MoE results. Path to find out: repeat the ledger's tests on a second MoE model with a different expert layout and on one non-MoE model, and log them in the same format.

**Author and maintainer:** John Lipka. Developed with AI assistance under his direction. Sibling repository: [pelias-trainer](https://github.com/FireballJohnnie/pelias-trainer).

> **Status note (2026-10-27, author decision).** Active work has moved to the combined repository `pelias-master` (private), which rebuilds the engine and the trainer as one C++20/CUDA system. This repository is kept as it is: its records stay valid as the history they describe. The engine will be re-explored here as its own system after `pelias-master` is ready.

> **PRE-RELEASE DRAFT, updated 2026-10-22. Not published.**
> One machine has been measured. A first llama.cpp patch (a static bank of hot experts in VRAM, the rest on the CPU; called the "tier" here) now decodes on that machine and is **measured** at about +5 to +7% over the best stock layout with the 6.1 GiB bank, and up to about +17% at 12k context and +28% at short context with a 7.5 GiB bank at bypass 32 against the matched stock run (derived from single launches; section 4.5; `benchmarks/RESULTS.md` sections 8.2, 8.6). The zero-copy cold-path pieces are built and verified separately and are **not yet integrated into llama.cpp**. The hit rate is **simulated** (0.765 for a static bank at 6.1 GiB), not yet counted in a live run. The strict accuracy limits are **not met** by the tier, nor by stock at another batch size; the tier is inside that spread at batch 8 and 16 (section 7). The crash for batches of 9 or more tokens is **fixed** (cause: a signed/unsigned mix in one CUDA kernel; `benchmarks/RESULTS.md` section 8.4). The loose patches and raw logs are in `patches/stage2-tier-cuda/` and `benchmarks/raw/stage2-tier-cuda/`; the base tier patch and the engine source are **not** in this repository.
> Every number in this file carries one label: **measured** (run on the test machine, with a log), **published** (vendor or standard specification), **derived** (arithmetic on paper from labelled inputs), **estimate** (derived for hardware that has not been tested), or **unverified**.
> No "first ever" claim is made. See *Related work*.
>
> **The test machine's tuning, disclosed (author-stated; confirmed by nvidia-smi on 2026-10-29):** the RTX 4070's memory runs a +1000 MHz overclock in MSI Afterburner (11,501 MHz vs 10,501 stock, about 552 GB/s peak instead of 504), the core about 2,790 MHz undervolted (2.774 GHz measured inside a kernel run), Windows hardware-accelerated GPU scheduling off. Every measured speed here comes from that one machine in that state; numbers for any other hardware are estimates for others to test.

---

## Contents
1. [Origin](#1-origin)
2. [The idea in one page](#2-the-idea-in-one-page)
3. [What the limit really is](#3-what-the-limit-really-is)
4. [Results on the test machine](#4-results-on-the-test-machine)
5. [Applying it to other hardware (theory and estimates)](#5-applying-it-to-other-hardware-theory-and-estimates)
6. [What this could mean for smaller and larger machines](#6-what-this-could-mean-for-smaller-and-larger-machines)
7. [Accuracy](#7-accuracy)
8. [Status of every component](#8-status-of-every-component)
9. [Repository layout, footprint and bundling](#9-repository-layout-footprint-and-bundling)
10. [Related work](#10-related-work)
11. [Licensing and third-party notices](#11-licensing-and-third-party-notices)
12. [Roadmap and how to help](#12-roadmap-and-how-to-help)
13. [Acknowledgements and provenance](#13-acknowledgements-and-provenance)

---

## 1. Origin

Pelias did not start as an inference project. It started as a **training** problem: fine-tuning a 35B-parameter Mixture-of-Experts model on a single 12 GB consumer GPU, where standard 4-bit tooling failed for this model (author-reported). The training engine keeps a small dense "spine" (attention, linear-attention and shared-expert weights) resident in VRAM and **streams the quantised experts layer by layer from pinned host RAM**.

Working out how fast that stream could go, and which parts of the machine were idle while it ran, led to the inference question this repository is about: *if the GPU can read host RAM over PCIe while the CPU reads the same RAM, can the two be driven together so cold experts arrive at the DRAM controller's rate, with a hot set of experts living in VRAM?*

The trainer is a companion project that is being refined; it is not part of the first release of the engine (see [Licensing](#11-licensing-and-third-party-notices) for why that needs a separate review).

## 2. The idea in one page

**Why MoE decoding is a bandwidth problem.** Per generated token each layer reads only its top-k experts (here 8 of 256), but all experts must be *somewhere*. On a consumer GPU they do not fit in VRAM (test model: 18.16 GiB of routed experts, **measured**, against 12 GB of VRAM), so most live in system RAM.

**What stock practice does.** llama.cpp-style offload (`--n-cpu-moe`) puts whole layers' experts in VRAM or in RAM. Layers in RAM are computed **by the CPU alone**. The PCIe link and the GPU's compute sit idle for those layers, and which experts sit in VRAM is decided by layer number, not by how often the router picks them.

**What Pelias does** (three mechanisms):

1. **Hot bank.** Keep the individually most-used experts in VRAM (chosen from recorded router traces), not whole layers. Hits run at VRAM speed.
2. **Split the misses.** For every expert the router picks that is not in the bank, the rows of its matrices are divided between **the CPU** (reads and computes from RAM) and **the GPU** (reads the *same* RAM over PCIe, zero-copy, and computes on the GPU). Both consumers drain the same DRAM at the same time.
3. **Doorbell handoff.** The GPU and the CPU exchange "here is the work" and "here is my partial result" through flags in pinned memory, with a kernel that stays resident and waits (~µs), instead of stopping the GPU stream and synchronising every layer (~94 µs, **measured**).

```
                  hidden state x of this layer
                              |
                 router (GPU) -> top-8 experts
             .----------------+------------------.
        hit: in the VRAM bank               miss: cold, in host RAM
        GPU computes at VRAM speed                |  miss_splitter divides the rows
                                         .--------+--------.
                                CPU rows (RAM -> CPU)   GPU rows (RAM -> PCIe -> GPU, zero-copy)
                                         '--------+--------'
                                   doorbell handoff through pinned flags (microseconds)
                                                  |
                                       combine on the GPU -> next layer
```

**What "unified bus" means here, exactly.** Nothing physical changes: there are still two buses (DRAM to CPU, PCIe to GPU) and the VRAM bus. The aim is that no bandwidth that exists is left idle: the hot set is read at VRAM speed, and the cold set is read by the CPU **and** the GPU together, so the machine's usable cold-weight bandwidth approaches the **DRAM controller's** limit instead of a CPU-only fraction of it.

## 3. What the limit really is

Measured and derived on the test machine (Ryzen 7 5800X3D, 4 x 16 GB DDR4-3200 on two channels, RTX 4070 12 GB on PCIe 4.0 x16):

| Quantity | Value | Label |
|---|---|---|
| VRAM bandwidth, RTX 4070 (192-bit, 21 Gbps GDDR6X) | 504 GB/s | derived from the published memory spec |
| DRAM peak (2 channels x 3200 MT/s x 8 B) | 51.2 GB/s | derived |
| PCIe 4.0 x16 peak (16 GT/s x 16 lanes x 128/130 / 8) | 31.5 GB/s | derived |
| CPU reading RAM alone (7 threads) | 45.9 GB/s (= 90% of DRAM peak) | measured |
| GPU reading RAM over PCIe alone (zero-copy kernel) | 26.1 GB/s (= 83% of PCIe peak) | measured |
| CPU and GPU reading RAM **at the same time** | 26.6 + 21.1 = **47.6 GB/s** (= 93% of DRAM peak) | measured |
| What the stock CPU expert path delivers | about 26-27 GB/s | derived from two stock speed runs |

How to read this honestly:

* **The physical ceiling for cold weights is the DRAM controller. Pelias does not break it.** The two paths together reach 93% of it, because both end at the same DRAM.
* **The limit the project attacks is lower**: the stock CPU path delivers about half of the DRAM peak (26-27 of 51.2 GB/s). Pelias reaches 47.6 GB/s, which is **1.79x** the 26.6 GB/s CPU share (derived; 26.6 is the CPU's rate in the combined run, used here as the stock figure).
* **Not yet known: why the stock path is at 26-27 GB/s.** The CPU alone can *read* 45.9 GB/s, but reading is not computing (dequantise and multiply). If the stock path is limited by CPU compute, the GPU's share adds real throughput (the measured combination gains 1.79x). If one could reach ~46 GB/s of *useful* CPU work, the GPU's share would add only the measured +3.8% (47.6 vs 45.9). **Unverified; the measurement is to run the CPU worker alone and time it.**
* **The CPU carries the larger share of the combined cold read**: 26.6 of 47.6 GB/s = 56% (derived from the measured combined run). The CPU side of the machine is the main engine for cold weights; the GPU adds the rest.
* **The second source of gain is the hit rate**: every expert found in the VRAM bank removes its bytes from the cold term. That number is **unmeasured** (see §4.4).

## 4. Results on the test machine

**Test setup.** Windows 11 with WSL2 (Ubuntu 26.04.1 LTS); llama.cpp commit `2ca15f5` (4 Oct 2026); model: a fine-tune of Qwen3.6-35B-A3B in Q4_K_M GGUF (40 layers, 256 routed experts, top-8; expert tensors 18.16 GiB, **measured**); context 131,072, micro-batch 2048, KV cache q8_0.

Each row below is tied to its raw log, with a label, in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). The engine's own handoff test and a 4-layer run of the pin and GPU-read checks have raw logs (`benchmarks/RESULTS.md` section 5). The full 40-layer figures in section 4.2 (load time, pinning time, checksum-kernel check) are recorded in the author's run log, but their raw output has **not been found yet**.

### 4.1 Stock baseline (what Pelias is compared against) - **measured**

| Configuration | Decode tokens/s (short / 4k / 12k context) |
|---|---|
| Stock llama.cpp, all experts on the CPU | 31.8 / 31.5 / 31.3 |
| Stock llama.cpp, `--n-cpu-moe 28` (12 layers of experts in VRAM; the best layout that fit alongside other GPU use) | 41.1 / 40.2 / 38.8 |

### 4.2 Pelias building blocks - **measured**

| Block | Result |
|---|---|
| Load the 40 layers of expert tensors into an aligned, plain-layout copy | 18.16 GiB in 37.7 s (0.52 GB/s); byte-identical to the file |
| Pin them for GPU access (`cudaHostRegister`) | 120 regions, 0 failed, 0.82 s |
| GPU kernel reading the pinned copy through its device pointers | 26.2 GB/s (checksum kernel); 0 wrong bytes in 120 matrices and 1,080 sampled rows |
| Sync-per-layer handoff (stop the GPU stream and wake it) | 94.4 us median, 123 us p99 per layer |
| Resident-kernel handoff, ping-pong | 5.05 us median, 6.17 us p99 in the engine's own handoff test; in the earlier microbenchmark 13.3 us / 18.7 us with a 4 KB read and 3.4 us for the flag alone |
| Cold-path kernels + splitter + handoff + combine, one layer, 8 split scenarios on synthetic Q4_K/Q6_K data | all 8 correct against a double-precision reference within bounds derived on paper (see §7) |
| Cold-path kernel read speed (computing, from pinned RAM) | **9.59 GB/s median** - well below the 26.2 GB/s the checksum kernel reaches; the kernel loads weights byte by byte, a likely cause (**unverified**); wider loads are the next step |

### 4.3 Modelled end-to-end speed - **derived ceiling, NOT measured**

```
time per token = spine + (1 - hit_rate) x (cold bytes per token / cold bandwidth) + 40 x handoff
spine          = 7.73 ms   (derived from the two stock runs at 0 and 12 GPU layers; assumes the cost per layer is linear - untested)
cold bytes     = 608 MB    (derived: 8 experts x 40 layers x 1.90 MB, from the measured expert sizes)
cold bandwidth = 47.6 GB/s (measured, CPU + zero-copy together)
handoff        = 6.17 us   (measured p99, resident ping-pong)
```

| Hit rate in the VRAM bank (hypothetical) | 0% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|
| Ceiling, tokens/s | 48.2 | 76.4 | 84.7 | 95.0 | 108.1 |
| Ceiling with today's cold-path kernel (9.59 GB/s GPU share, cold bandwidth 36.1 GB/s) | 40.3 | 68.0 | - | 88.2 | 103.5 |

For scale, measured stock: 31.8 tokens/s (all-CPU) and 41.1 (best flags-only layout). At 80% hit rate the ceiling is **2.3x** the flags-only result and **3.0x** the all-CPU result (derived). These are ceilings: they assume the CPU and GPU shares overlap perfectly and that the handoff costs nothing else in a real graph.

### 4.4 Not measured yet, and how it will be measured

| Unknown | Path to a number |
|---|---|
| **Hit rate** of a profile-chosen bank on held-out text, in a live run | live count on the gauntlet's prompts: 0.775 to 0.779 for the 7.0 GiB bank (simulated 0.806), `benchmarks/RESULTS.md` section 9.1; the held-out text count is still to do |
| Overlap of the CPU and GPU shares in a real layer | measured in the llama.cpp tier: none, the layers run in sequence, and the handoff costs at least 4.6 to 5.0 ms per token (`benchmarks/RESULTS.md` section 9.2); the engine's own shares are still to do |
| Spine linearity (is 0.59 ms per moved layer constant?) | one more `--n-cpu-moe` point |
| Handoff with a real CPU workload and other kernels in the stream | in-stream test |
| Whether the stock CPU path is compute-bound (§3) | time the CPU worker alone |
| Effect of float-activation GPU maths on outputs (§7) | KL divergence against stock logits |

### 4.5 Results ledger: derived ceiling first, measured result to follow

Each stage has the ceiling derived from this machine's measured inputs, and an empty cell for the measured end-to-end result. The cells are filled in, with the raw log, as each stage is tested; nothing is entered without a run. Hit rates of 60 / 80 / 90% are hypothetical until the traces give the real one.

| Stage | What runs | Derived ceiling, tok/s at 60 / 80 / 90% hit | Measured tok/s (short context) | Measured hit rate | Accuracy (KL vs stock) |
|---|---|---|---|---|---|
| 0 | Stock llama.cpp, all experts on the CPU | - | **31.8** | - | reference |
| 1 | Stock, `--n-cpu-moe 28` | - | **41.1** | - | same weights |
| 2 | Hot bank in VRAM, cold experts on the CPU only (the llama.cpp patch) | 48.4 / 62.2 / 72.5 | **44.3 to 44.7** in three runs (bank 6.1 GiB, context 32,768, bypass 9); matched stock 41.2 to 42.0. Later single launches: 7.5 GiB bank, bypass 32, `-ub 2048`: **53.5** short, 49.0 at 12k (stock 41.9 / 41.8); bank 8.5 GiB, bypass 9: 55.7 / 53.4 at 12k | 0.765 **simulated** (held-out, byte-weighted); live count on the gauntlet's prompts: **0.775 to 0.779** for the 7.0 GiB bank (section 4.4) | KL 0.0106, top-1 97.13% at batch 8 (limits 0.005 / 99%): **not passed**; stock at batch 16 against the same reference: KL 0.0174, top-1 96.49% |
| 3 | Full Pelias with today's cold-path kernel (9.59 GB/s GPU share) | 68.0 / 88.2 / 103.5 | pending | pending | pending |
| 4 | Full Pelias with a fast kernel (GPU share at the 21 GB/s measured in the combined run) | 76.4 / 95.0 / 108.1 | pending | pending | pending |

How each ceiling is derived (same inputs as §4.3: spine 7.73 ms, 608 MB of cold experts per token, handoff 6.17 us, CPU rate 26.6 GB/s as the CPU's share in the combined run):
* Stages 3 and 4: `1 / (spine + (1 - hit) x 608 MB / cold bandwidth + 40 x 6.17 us)`, with cold bandwidth 36.1 GB/s (26.6 + 9.59) and 47.6 GB/s (measured combined).
* Stage 2: the same with the CPU alone reading the cold experts (26.6 GB/s) and a stream sync per layer instead of the resident handoff (40 x 94.4 us = 3.78 ms per token, measured per-layer cost); this assumes the stock scheduler pays the same per-layer cost, which is **unverified**.

**Stage 2 notes** (details and raw logs: `benchmarks/RESULTS.md` section 8).
* **Speed (measured).** Generation rate, median of three requests, in three matched pairs against stock `--n-cpu-moe 28` on the same context and settings. Mean gain of the tier over stock: +2.8 / +2.2 / +2.7 tokens/s at short / 4k / 12k context (about +5 to +7%). Two launches of the same setting differ by up to 1.4 (stock) and 1.8 (tier) tokens/s, so a smaller gain is noise. **Bank size** (single launches, bypass 9, short context): 44.2 and 47.7 at 6.1 GiB, 48.4 at 7.0, 52.4 at 7.5, 54.6 at 8.0, 55.7 at 8.5 GiB, with GPU peaks of 9.0 to 11.4 GiB.
* **Against the ceiling (derived).** At the simulated hit rate of 0.765 the stage-2 formula gives about 59 tokens/s (spine 7.73 ms + 0.235 x 608 MB / 26.6 GB/s + 3.78 ms). The measured 44 to 47 is about 75 to 80% of that. The gap is **unexplained**.
* **Prompt reading is slower than stock** (0.67 to 0.75 of stock at 12k context, measured; 0.73 at bypass 32 with `-ub 2048`, derived from 921 vs 1,253 tokens/s; bypass 9 with `--load-mode none -ub 2048` reached 1,680 tokens/s at 12k). Derived reason: for each large chunk the tier copies the experts of all 40 layers to the GPU, stock only 28; 28/40 = 0.70.
* **Where it applies.** The dual path runs only for batches of up to 8 tokens (bypass 9); larger batches take the stock path. With bypass 32 the dual path now runs for chunks of up to 31 tokens. Before the fix every prompt of 13 to 34 tokens crashed (a chunk of 9 or more reached a wrong row index in the quantiser); after it, prompts of 8 to 61 tokens ran without a CUDA error. The bank path beats the stock path up to a chunk of 24 tokens and ties from 32, so a bypass above 32 gains nothing in these runs (`benchmarks/RESULTS.md` sections 8.4, 8.7).

## 5. Applying it to other hardware (theory and estimates)

**Everything in this section other than the test machine is an estimate or unverified.** It is a map of what would have to hold, and a model for what it would be worth. If you have one of these machines, the measurements that would replace a row are listed in [§12](#12-roadmap-and-how-to-help); raw logs are welcome.

### 5.1 What the method needs

| # | Requirement | NVIDIA (CUDA) | AMD (ROCm/HIP) | Intel Arc (Level Zero / SYCL) | Any GPU (Vulkan) |
|---|---|---|---|---|---|
| R1 | A GPU kernel can read ordinary host RAM over PCIe, zero-copy | **measured** here (mapped, registered pinned memory) | unverified (host registration with a device pointer exists in HIP) | unverified (host USM allocations) | unverified (importing a host pointer) |
| R2 | CPU and GPU see each other's flag writes within microseconds | **measured** (5-13 us) | unverified (needs coherent host memory) | unverified | unverified |
| R3 | A kernel may wait for the CPU without being killed | **measured** (bounded to 0.5 s, below the 2 s display timeout) | unverified | unverified | unverified |
| R4 | CPU kernels for the quant formats | ggml CPU backend: AVX2 / AVX-512 / others | same | same | same |
| R5 | RAM for the cold experts (plus a second plain-layout copy, 18.16 GiB for the test model), VRAM for the spine, KV cache and bank | - | - | - | - |

If R2 or R3 fails on a platform, the fallback is the slower sync-per-layer handoff (94 us per layer here, 3.8 ms per token over 40 layers, derived), which the model below can price in.

### 5.2 Components

* **CPUs (Intel and AMD).** The CPU share uses ggml's own quantised dot kernels, so any CPU llama.cpp supports can take part. Its throughput (cores, SIMD width, AMX/AVX-512) sets the CPU's share of the cold read and is **unmeasured on every CPU except the test machine**. Hybrid Intel parts probably need threads pinned to performance cores (unverified). Chiplet and NUMA layouts change how much DRAM bandwidth one process can pull (unverified).
* **DDR4 / DDR5.** DRAM peak = channels x MT/s x 8 B (derived). It caps the cold read at about 93% of peak (the one measured ratio). Faster or wider memory raises the cap; whether the CPU share can use it depends on the CPU.
* **PCIe.** Gen3 / 4 / 5 x16 peaks are 15.8 / 31.5 / 63.0 GB/s (derived). The measured efficiency on Gen4 is 83% (one machine). A narrower link (x8) or slower generation shrinks the GPU's share directly.
* **GPU vendors.** The engine's GPU side is a handful of simple kernels (dequantise-and-dot, a flag wait) behind a thin portability layer (`pl_cuda.h`). Porting means R1-R3 above plus rewriting those kernels for the other API.

### 5.3 Estimated cold bandwidth and ceilings

Model as in §4.3 (spine 7.73 ms, 608 MB per token, handoff 6.17 us), with `cold bandwidth = min(0.93 x DRAM peak, CPU share + 0.83 x PCIe peak)`.

| System (hypothetical unless marked) | Cold bandwidth, GB/s | vs stock CPU path | Ceiling tok/s at 60 / 80 / 90% hit |
|---|---|---|---|
| A. **Test machine**: DDR4-3200 x2, PCIe 4.0 x16 (measured combined) | 47.6 | 1.79x | 76.4 / 95.0 / 108.1 |
| B. DDR4-3200 x2, PCIe 3.0 x16 | 39.6 | 1.49x | 70.9 / 90.5 / 105.1 |
| C. DDR5-5600 x2, PCIe 4.0 x16 | 52.7 | 1.98x | 79.4 / 97.2 / 109.5 |
| D. DDR5-6000 x2, PCIe 5.0 x16 | 78.9 | 2.97x | 90.4 / 105.1 / 114.3 |
| E. DDR5-5600 x2, PCIe 4.0 x8 (e.g. an x8 card) | 39.6 | 1.49x | 70.9 / 90.5 / 105.1 |

**Assumptions behind every row (all of them unverified outside row A):** the same CPU share as the test machine (26.6 GB/s) on every CPU; the same 0.83 PCIe efficiency; the same 7.73 ms spine (a faster or slower GPU changes it; its link to VRAM bandwidth is untested); the same 608 MB per token; perfect overlap; the stated hit rates are not predictions (none is measured). Read the table as **how the ceiling moves with DRAM and PCIe**, not as a forecast for any product.

## 6. What this could mean for smaller and larger machines

All consequences follow from one relation and its limits:

```
tokens/s  ~  1 / ( spine + (1 - hit) x active_expert_bytes_per_token / cold_bandwidth + handoff )
```

* **Cold bandwidth is bounded by DRAM**, and hit rate by how much of the routing the VRAM bank covers. The effect is **bounded and additive, not compounding**: no input makes it grow without limit.
* **Machines with little VRAM and plenty of RAM** can run a MoE whose experts fit in RAM; speed is set by the cold term. At 0% hit the test machine's ceiling is 48.2 tokens/s against 32.7 for the same formula at the stock CPU rate (derived), i.e. the cold-read rate alone is worth up to ~1.5x there.
* **Machines with more VRAM** raise the hit rate (unmeasured as a function of bank size), which is the lever that grows the most from 0% to 90% in §4.3.
* **Quantisation level** changes the bytes read per token, so it moves the cold term directly (derived; all of it unverified until measured):
  * *Fewer bits (1-3 bit)*: fewer bytes per token and more experts per GB of VRAM, so the cold term shrinks and the hit rate rises; the engine's relative gain shrinks with it, and the quality loss of such files is a property of the file, not of the engine, and the engine leaves it untouched. People who run such files, and whose model still does not fit in VRAM, would still get the speed benefit (once kernels for those formats exist). Whether the Pelias *training* method can improve the quality of low-bit models is a separate, open question (**unverified**).
  * *More bits (6-bit, 8-bit)*: more bytes per token, so the cold term grows and the relative gain from a faster cold read grows, **provided the experts still fit in RAM**. Example with the §4.3 model, spine held fixed, 80% hit rate: at twice the expert bytes the ceiling is 76.4 tokens/s against 18.7 for the same formula at the stock CPU rate (4.1x), versus 95.0 against 32.7 (2.9x) at the test model's size. But at fixed VRAM a bank of twice-larger experts holds half as many, so the hit rate would be lower (unmeasured). A 16-bit 35B-parameter model is about 70 GB (2 bytes x 35B, derived) and does not fit in 64 GB of RAM.
  * *4-bit and 6-bit* are where the quality/size trade-off is usually acceptable on consumer hardware, which makes them the most relevant cases to benchmark. Whether they show the *largest* improvement is a hypothesis; the formula above suggests higher bit-widths gain more in relative terms when they fit.
  * **Implemented kernels so far: Q4_K and Q6_K only.** Other formats (Q2_K, Q3_K, Q5_K, Q8_0, the IQ types) need their own dequantise-and-dot kernels; the CPU side already supports every format ggml does.
* **Larger models**: if a model's active expert bytes per token double (and the spine stays fixed, which would not be exactly true), the test machine's 80%-hit ceiling falls from 95.0 to 76.4 tokens/s (derived). Capacity rule: the experts (and the plain-layout copy) must fit in RAM.
* **Where it does not help**: models that fit in VRAM; machines whose CPU and GPU already share one memory (Apple silicon, APUs).
* **Dense (non-MoE) models**: the hot bank does not apply, because every token reads every weight and there is no routing skew to exploit. Whether the two-path read of offloaded layers (the cold-bandwidth term only) helps them is an open question (**unverified**); it would be a pure bandwidth effect with no hit-rate term.
* **Server hardware**: many-channel DRAM lets the CPU alone pull far more, and large VRAM puts more of the model on the GPU, so the GPU's share of the cold read is likely a smaller fraction of the total (**unverified**). This is why server machines are in the benchmark matrix below.

More benchmarking on other machines and models is needed before any of this is more than a model.

## 7. Accuracy

* **Expected (derived, not yet measured):** the same quantised weights, the same router choices, and every missed expert computed either on the CPU or on the GPU (nothing is dropped) give the same output as stock llama.cpp up to floating-point order. The GPU path multiplies float activations, whereas ggml first quantises activations to Q8_K and uses an integer dot; the size of that difference is unmeasured.
* **Checked so far (measured):** per-lane maths of the Q4_K and Q6_K kernels reproduces the dequantised weights exactly (10,240 one-hot positions per format); each of the 8 split scenarios agrees with a double-precision split-free reference inside a bound derived on paper (Higham, *Accuracy and Stability of Numerical Algorithms*, §3.1: gamma_n = n u / (1 - n u), u = 2^-24, any summation order; expf at 2 ulp). The bound is loose by four to five orders (worst error / bound 4.1e-5), so it catches wrong scales, signs and row ranges, not rounding-level drift.
* **Measured (2026-10-21 and 22; 16 chunks, 8,176 tokens, reference: stock at batch 8):** tier at batch 8: mean KL 0.0106 nats (limit 0.005), top-1 97.13% (limit 99%), perplexity 4.1116 against 4.0989 (+0.31%, derived). **Controls:** the tier with the bank off: KL 0.0017, top-1 98.74%; stock at batch 16: KL 0.0174, top-1 96.49%; stock at batch 1,024: KL 0.0183, top-1 96.32%; the tier at batch 16 with bypass 32 and a 7.5 GiB set: KL 0.0179, top-1 96.16%. The strict limits are therefore **not met even by stock** at another batch size. Under the design's own rule (no worse than the stock-against-stock spread) the tier passes at batch 8 and is within one standard error of stock at batch 16, but **which definition applies is the author's decision**. Tail (maximum KL 0.4 to 4.6 in every row) is not specific to the bank; the cause is **not tested**. Details: `benchmarks/RESULTS.md` section 8.3.
* **To do:** the author's ruling on the limit, then task-specific test sets.
* **The quality of a served model is the quality of its GGUF.** Training, merging, quantisation and any importance matrix decide it; the engine is neutral. The Pelias trainer (being refined) therefore affects results through the weights it produces, not through the engine.

## 8. Status of every component

| Component | Status |
|---|---|
| Plain-layout, aligned copy of the expert tensors from a GGUF; GPU pointers per matrix | built; **verified on the real 40-layer file** |
| GPU zero-copy reads of that copy | built; verified (26.2 GB/s checksum kernel) |
| Miss splitter (rows divided between GPU and CPU; runs on host and device) | built; tested |
| Doorbell handoff | built; resident ping-pong measured; in-stream with a real CPU workload **not yet** |
| Cold-path GPU kernels (Q4_K and Q6_K only; other quant formats not written) | built; correct on synthetic data on the GPU; **speed 9.59 GB/s, to improve** |
| CPU worker | tested against real ggml on a CPU-only build; the always-polling version is **not written** |
| Hot bank patch for llama.cpp (cold on CPU only) | built for CUDA; decodes on the test machine (+5 to +7% over stock, measured); strict accuracy limits not met (stock does not meet them either; decision open); crash for batches of 9 or more tokens **fixed**, loose patches in `patches/stage2-tier-cuda/` |
| Cold path inside llama.cpp (device-side planning, graph ops, CPU worker, timeout fallback) | **not written** |
| Router traces, hot-set simulation | run on recorded traces: static and adaptive-cache hit rates simulated (derived); no live hit-rate counter yet |
| Measure-anything framework and go/no-go report | planned |

The detailed list, with the checks required before a release, is in `docs/STATUS.md` (not yet in this repository; it is kept in the author's development notes until release).

## 9. Repository layout, footprint and bundling

*This is the target layout. Until release the files live in the author's development folder and are copied here in stages; apart from this README and the LICENSE, nothing in this layout is in the repository yet.*

*Update 2026-10-25: the sentence above is out of date. The repository now holds the documents, ledgers and patches listed in 9.0. The source folders of the target layout below (`src/`, `tools/`, `tests/`, `scripts/`) are still not in it, and the engine is on hold.*

### 9.0 What is in the repository today (2026-10-25)

| Path | What it is |
|---|---|
| `README.md`, `LICENSE`, `CITATIONS.md` | This file, the Apache-2.0 text, and the verified citations |
| `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` | The engine plan (14 gated phases E0 to E13), synchronized with the author's rulings of 2026-10-24 |
| `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md` | The verified inventory of every file, the scaffold target tree, the phase and gate map, and the open gaps |
| `docs/PELIAS_ENGINE_LINEAGE.md` | How the numbers and the plan came to be what they are (old figure, new figure, reason, evidence) |
| `docs/spec/` | The proprietary runtime and server specification and its companion records (open decisions, probe registry, research-update audit, Wave 0 and ceiling, trainer contract, synchronization register) |
| `benchmarks/` | `RESULTS.md` (claim-to-log ledger) and `raw/` (redacted logs of the runs) |
| `patches/stage2-tier-cuda/` | The loose patches on llama.cpp 2ca15f540 (CUDA signed-id fix and diagnostics, cold-branch pinning) |
| `support/` | The current Windows-native tier build (`tier_build/`), redacted copies of the working folders (`archive/`), and `MANIFEST.tsv`; see `support/README.md` |

The engine is **held**; nothing in this table is engine source code. The only code is the llama.cpp patches (a measuring vehicle and test oracle, not the product) and the measurement programs in `support/archive/`.

```
pelias-engine/
  README.md  LICENSE  CITATIONS.md  NOTICE  THIRD_PARTY_NOTICES.md  CONTRIBUTING.md  SECURITY.md  CHANGELOG.md  CITATION.cff
  docs/            design, measured results, porting guide, methodology, status
  src/
    core/          splitter, handoff, weights view, cold-path kernels, CPU pool, quant geometry
    portability/   the thin GPU API layer
    experimental/  profiler, state snapshots, speculative-decoding policy
  patches/
    llama.cpp/     the hot-bank patch (+ cold path later), build script, pinned base commit
  tools/
    measure/       hardware floors: RAM, PCIe, combined, handoff, pinning
    trace/         router-choice recording
    sim/           hot-bank simulation, predictor evaluation
    launcher/      server launcher
    parity/        reference dumps and comparison
    common/        shared helpers
  tests/           unit and hardware tests, CPU stand-in for the GPU API
  scripts/         numbered run-order scripts
  benchmarks/      RESULTS.md (claim-to-log ledger) and raw/ (redacted logs)
  trainer/         (later) the training engine and its patches
```

### 9.1 Size of a build (proposed; the label says which numbers are measured)

| Item | Size | Label |
|---|---|---|
| Source repository, engine-only first release | existing 78 files: 828 KB; with the files still to be written, a few MB at most (mostly text) | measured + estimate |
| Front-page files (README, LICENSE, CITATIONS) | about 54 KB (README about 35 KB, LICENSE about 11 KB, CITATIONS about 8 KB) | measured |
| Patched llama.cpp build (CUDA, one GPU architecture) | unmeasured; path: `du -sh` of the build directory after the first patched build | unmeasured |
| Model file on disk (test model, Q4_K_M GGUF) | 20.22 GiB, of which 18.16 GiB are routed experts and about 2.06 GiB everything else (derived) | measured |
| Plain-layout copy of the experts in RAM (pinned) | 18.16 GiB, in addition to whatever llama.cpp holds; if llama.cpp also keeps its own CPU-side copy, both together are about 36.3 GiB (derived) against 64 GB of RAM; pointing llama.cpp's expert tensors at the single pinned copy is an option | measured + derived |
| VRAM | the spine, KV cache and compute buffers take about 4.8 GiB at 131k context with experts on the CPU (measured); the hot bank gets what is left of the 12 GB card (about 5.5 GiB per the project notes, derived) | measured + derived |

### 9.2 Where it runs first

* **Tested:** Windows 11 with WSL2 (Ubuntu 26.04.1 LTS), NVIDIA RTX 4070 (sm_89), x86-64. Sources and the model sit on the Windows (NTFS) side and are read through the WSL mount; build outputs sit on the WSL ext4 disk.
* **Filesystem:** the engine copies the expert weights into RAM at start, so decode does not read the model file; any filesystem the GGUF can be read from works (design property, **unverified** beyond this machine). The load itself ran at 0.52 GB/s from the NTFS mount (**measured**); the same load from ext4 or a native Linux NVMe is **unmeasured**.
* **Initial target:** Linux (native or WSL2), NVIDIA CUDA, x86-64. **Not supported at first:** native Windows, AMD, Intel Arc (section 5 explains what they need), macOS (unified memory already; see section 6).
* Transparent huge pages were on for the measurements; they changed bandwidth by under 1% (**measured**), so they are optional.

### 9.3 Bundling with the Pelias trainer

| | Size | Label |
|---|---|---|
| Trainer, all parts: Pelias 0.2.1 package (17 files, 102,542 B) and 0.2 launcher (1 file, 166,308 B), Pelias 0.1 package (15 files, 84,381 B) and 0.1 launcher (1 file, 166,297 B), v0.3 patch layer (2 files, 23,778 B), draft-head aligner (6 files, 90,903 B), 7 tests assigned by file name (90,801 B) | 49 files, 725,010 B (about 708 KiB) | measured; total derived |
| Trainer runtime environment (PyTorch, transformers, PEFT, Triton, GPU kernels, whatever else the author's environment holds) | installed, never bundled; size unmeasured; path: `du -sh` of its virtual environment | unmeasured |

**Decision (author, 2026-10-20): the trainer gets a separate repository, linked from this README** (repository: [`pelias-trainer`](https://github.com/FireballJohnnie/pelias-trainer)). Reasons: different language and dependencies (Python / PyTorch against C++ / CUDA); the licence review its dependencies need (section 11) would otherwise hold up the engine; different release pace while the trainer is still being refined; a smaller clone and simpler CI for engine users; and the two meet at one clean interface, the GGUF file. The engine is model-agnostic, so a user can run it with any GGUF, trained with Pelias or not. If a single repository is wanted later, the trainer folder can be added to this one once its licence review is done, or the two can be tied together with a git submodule, without changing the engine.

## 10. Related work

No novelty claim is made until this list has been reviewed against the actual projects. Known neighbours, to be compared before release (details **unverified**):

* llama.cpp's expert placement flags (`--n-cpu-moe`, `--cpu-moe`) and an **open** llama.cpp pull request (PR #29887; open when checked on 2026-10-20, not merged) that adds an expert cache option (`--moe-cache-mib`).
* Hybrid CPU/GPU MoE inference projects such as KTransformers, Fiddler, MoE-Infinity and the ik_llama.cpp fork.
* llama.cpp and ggml themselves, on which the CPU kernels, the quantisation formats and the patch point depend.

Links, authors and dates for each of these are in [`CITATIONS.md`](CITATIONS.md).

## 11. Licensing and third-party notices

*Draft, 2026-10-20. Not legal advice; the author decides before release.*

| Item | Licence | Note |
|---|---|---|
| **Pelias code in this repository** | **Apache License 2.0 (proposed)** | The `LICENSE` file is the standard text. Copyright holder (proposed wording): Copyright (c) 2026 John Lipka; the author confirms before release. A `NOTICE` file and SPDX headers are still to be added. |
| llama.cpp and ggml (patch target; CPU kernels) | MIT (checked on its GitHub page, 2026-10-20) | Patches that modify llama.cpp files stay MIT with the original notices; any binary distribution must reproduce them. |
| NVIDIA CUDA toolkit | NVIDIA's own licence | Not redistributed; users install it themselves. |
| Model weights | not included | The base model (Qwen3.6-35B-A3B) is listed as Apache-2.0 on its Hugging Face page (checked 2026-10-20). A fine-tune carries its own terms. |
| Trainer dependencies | **review needed** | Unsloth core is Apache-2.0, but its GitHub pages (checked 2026-10-20) show AGPL-3.0 for some optional components and LGPL-3.0 / AGPL-3.0 for `unsloth-zoo`. Any trainer code that imports or patches them needs a licence review before it is published. |
| Name | **check needed** | "Pelias" is also the name of an unrelated open-source geocoder ([pelias/pelias](https://github.com/pelias/pelias), MIT licence, maintained by Geocode Earth; **verified 2026-10-20**). This repository has no connection to it. The author chose `pelias-engine` and `pelias-trainer` for the repositories; a naming and trademark check is on the release list. |

The software is provided as is, without warranty (see `LICENSE`).

## 12. Roadmap and how to help

* **Before any release:** finish the remaining testing for the current model (tracked in the author's development notes) - wider loads for the kernel, router traces and hit rate, the llama.cpp build, the integration of the cold path, end-to-end speed and KL divergence - then repeat on at least one more machine and one more MoE model.
* **Universal build:** a tool that measures, on any machine, the numbers the model in §4.3 needs (RAM bandwidth by thread count, PCIe and combined bandwidth, handoff latency, VRAM, expert sizes from the GGUF header, a short layer sweep, a short trace), says whether the engine is expected to beat stock llama.cpp on that machine, and prints a projected speed with an interval. Projections are to be validated against real runs before they are published.
* **What would settle whether it is universal: a benchmark matrix.** Models {MoE, dense} x machines {consumer, server} x GPU vendors {NVIDIA, AMD, Intel} x quantisation {2-3 bit, 4-bit, 5-bit, 6-bit, 8-bit} x model size relative to VRAM and RAM (fits in VRAM, fits in RAM, barely fits). Each cell reports tokens/s against the *best-tuned* stock llama.cpp on the same machine (not its defaults), the hit rate where a bank is used, KL divergence against stock logits, and the raw logs. Cells where Pelias does not win are published too. Until the matrix has more than one machine in it, the claims in §5 and §6 stay estimates.
* **Public demo model (author's decision, 2026-10-20):** after both this engine and the trainer repository are finalised, a model trained on **non-private data** and published as **sample GGUF files in several quantisations, made by the trainer**, so anyone can test the engine without the author's own model. Needs a redistributable dataset (licence **to be checked**), a results ledger from logs, and an engine kernel for each quantisation offered (today only Q4_K and Q6_K). Not started.
* **Help wanted (after release):** measured floors and end-to-end runs from other CPUs, memory configurations and GPUs from NVIDIA, AMD and Intel, with the raw logs.

## 13. Acknowledgements and provenance

*Draft. The author confirms the wording, exact model names and versions before release.*

* **The author's own work.** The research, the design and the hardware measurements are the author's. The project began as a training problem (section 1) and grew into this engine.
* **Third-party code.** The engine patches and builds on **llama.cpp / ggml (MIT)**. Its notice will be reproduced in `THIRD_PARTY_NOTICES.md` (still to be written). Any other code copied or adapted gets the same treatment.
* **Citations.** Every source, with its role (built on, followed in tests, related work, consulted, development tool), is in [`CITATIONS.md`](CITATIONS.md). Each entry was checked by opening the page on 2026-10-20.
* **Models.** The model measured here is the author's own fine-tune of **Qwen3.6-35B-A3B** (Apache-2.0, checked 2026-10-20), made for private use and not distributed. The author's other private models were trained and fine-tuned from base models in the Gemma 4 family (26B A4B and 12B), Qwen3.6-35B-A3B and Qwen3-Coder (exact names and versions **to be confirmed**; base-model licences **to be checked** before release). The released base checkpoints were used only as starting points. **No model weights are part of this repository.**
* **Author and maintainer.** John Lipka. He owns and is responsible for what is published here, including the parts written with AI help.
* **AI assistance.** Built with Claude (Anthropic) and Gemini (Google) as coding, review and writing assistants, under the author's direction. The author ran the hardware measurements.
