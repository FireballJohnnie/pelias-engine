# Pelias-Tier: a ground-up plan for faster decode on one 12 GB card

Draft for review, written 2026-10-20. Nothing here has been built or run unless marked [MEASURED].
Companion to `speed_and_accuracy_proposal.md` (the flag-and-build track). This file is the engineering track.

## 0. How to read this

Labels on every claim:
- [MEASURED] your own logs on PELIAS
- [PUBLISHED] a source I read (section 12); not re-measured here
- [DERIVED] arithmetic from measured numbers, assumptions stated
- [ESTIMATE] my judgment; may be wrong
- [HYPOTHESIS] plausible cause, not yet tested
- [UNVERIFIED] a claim about a tool or flag I have not checked

Accuracy classes (same as the proposal):
- **A** = placement and scheduling, same weights (outputs can differ by rounding)
- **B** = speculative decoding (distribution-preserving, speed only)
- **C** = changes the weights (needs a KL baseline first)

**First step if you read nothing else:** Phase 1a, one read-only command (section 6).

## 1. Thesis: what "like Pelias" means here

From `pelias-training-v3\README.md` and `03_Blueprint...`: Pelias is about 1,700 lines of Python. It worked because it:
1. Decided placement first, from measured bytes (spine resident, experts 4-bit in pinned RAM).
2. Owned only the thin layer: a loader and one custom expert op. Attention, LoRA and the optimizer stayed with libraries.
3. Packed once into a cache with a manifest.
4. Tested the logic on CPU in a sandbox; the GPU numbers came from your runs.
5. Measured against a floor and logged it.

Applied to decode:
- **Place first** from measured traffic (which experts fire), not from a framework default.
- **Own only the expert placement and the expert op.** Do not rebuild DeltaNet, attention, sampling or the server.
- **Decide offline, make runtime trivial:** a hot set chosen once from your own traffic, not an eviction policy guessing live.
- **CPU-side logic is tested in my sandbox; GPU runs are yours.**
- **Every phase has a floor, a prediction, a gate.**

Difference from training: decode touches 8 of 256 experts per layer per token, so the problem is not streaming whole layers. It is which ~2 MB experts are already next to the compute.

## 2. Where we stand [MEASURED]

Q4_K_M, KV q8_0, `-c 131072 -ub 2048`.

| Config | Generation tok/s (short / 4k / 12k) | Prompt tok/s (4k / 12k) | Wait at 12k | GPU in use |
|---|---|---|---|---|
| 32k ctx, default ub, all experts on CPU | 32.1 / 30.7 / 30.4 | 272 / 282 | 42.6 s | 3.1 GiB |
| 131k ctx, ub 2048, all experts on CPU | 31.8 / 31.5 / 31.3 | 827 / 921 | 13.1 s | 4.8 GiB |
| **same + `--n-cpu-moe 28` (final: 12 layers on GPU)** | **41.1 / 40.2 / 38.8** | **1,166 / 1,193** | **10.1 s** | **10.2-10.4 GiB** |

Needles 12/12 to 120k at the all-CPU setting. Persona/capability not yet run on the N=28 setting.
Logs: `logs\speed_server\20261020_025525_...` and `..._032031_q4km-ncmoe28-ctx131k-ub2048`.

## 3. The machine (HWiNFO report, 2026-10-20 03:45)

- CPU Ryzen 7 5800X3D, 8 cores / 16 threads, AVX2 (no AVX-512), 96 MB L3, running at 4.54 GHz.
- RAM 4 x 16 GB DDR4-3200 CL16-18-18-38 (XMP), 2 sticks per channel, dual-rank. Peak 51.2 GB/s [DERIVED]; real read maybe 40-45 GB/s [ESTIMATE, not measured].
- GPU RTX 4070, 12,282 MB, 192-bit, PCIe 4.0 x16 at 16 GT/s, Resizable BAR on. Memory about 504-540 GB/s [DERIVED from the report's 22.5 Gbps]. PCIe about 25 GB/s each way in practice [ESTIMATE; 31.5 theoretical].
- Hyper-V active (that is WSL2); HVCI off.
- Storage: C: Samsung on PCIe 3.0 x4; D: WD SN580 on PCIe 3.0 x2 (chipset slot).

## 4. A model of one token [DERIVED, from two measured points]

> Constants below are superseded by section 13 (expert size about 1.9 MB average, fixed cost about 7.4 ms, CPU path about 25 GB/s).

Per-token time, all experts on CPU: 31.45 ms. At N=28: 24.33 ms. Moving 12 layers saved 7.12 ms.
- Each layer moved to the GPU saves about **0.593 ms**.
- Expert bytes per layer per token are about 16.3 MB (8 x ~2.04 MB; the 2.04 MB comes from blk.40 sizes) [ESTIMATE], so about 0.65 GB per token across 40 layers.
- So the CPU expert path runs at about **27 GB/s**, roughly half of RAM peak.
- Linear extrapolation gives a fixed part of about **7.7 ms/token** (spine, lm_head, DeltaNet, sampling, launches). Two points only; the all-GPU end of the line cannot be measured on this card.

Prediction for a hot bank with hit rate h (hot experts on GPU, cold on CPU in parallel). The VRAM budget (about 5.6 GiB) holds about 2,950 experts, about 74 per layer, about 29% of all experts [DERIVED]. Two scenarios, because I cannot yet separate sync cost s from read bandwidth:

| Hit rate | (i) no sync cost | (ii) 0.1 ms sync per layer, all 40 layers |
|---|---|---|
| 0.70 | 67 tok/s | 57 tok/s |
| 0.80 | 80 tok/s | 64 tok/s |
| 0.90 | 99 tok/s | 73 tok/s |

Published static-pin hit rate on this model: 79% at 30% of experts per layer (Gini 0.65) [PUBLISHED, third-party prompts]. So the central estimate is **hit about 0.75-0.8, about 55-65 tok/s**, range 45-80 [ESTIMATE]. Only Phases 3-4 can confirm.
The cold path shares RAM with everything else: DRAM peak is the hard cap for cold reads.

## 5. Lever map

| # | Lever | Class | Attacks | Expected | Evidence / risk |
|---|---|---|---|---|---|
| 1 | Free flags: huge pages, `--load-mode none`, threads | A | CPU path bandwidth (27 of ~45 GB/s) | +0 to +15% [HYPOTHESIS] | Hyper-V nested paging may cost bandwidth on 4 KB pages. Cheap. |
| 2 | **Static hot bank + parallel hot/cold (Pelias-Tier core)** | A | bytes read from RAM | 55-65 tok/s [ESTIMATE] | Static pin published; design exists in a fork. C++/CUDA work. |
| 3 | Mainline `--moe-cache-mib` (LRU in VRAM) | A | bytes | unknown here | Published: pays off at >=75% hit; one tester saw only 1-5% over tuned `--n-cpu-moe`. [PUBLISHED, earlier research] |
| 4 | ParmesanParty fork as-is | A | bytes | unknown (README gives no speed numbers) | Single maintainer, 17 stars. Wires only Qwen3.5 MoE. [PUBLISHED] |
| 5 | Zero-copy third branch (GPU reads cold experts from pinned RAM, alongside the CPU) | A | cold-path bandwidth | cap 51 GB/s total | Uncharted. WSL2 support unknown. Shares the same RAM. |
| 6 | Cross-layer prefetch into a small dynamic region | A | misses | unknown | Published for other models (Fate); accuracy on this model unknown. |
| 7 | VRAM governor: size context per profile; reuse idle buffers | A | frees 0.5-1.5 GiB = 1-3 layers | +1-3 tok/s [ESTIMATE] | 65,536 ctx frees about 0.66 GiB [DERIVED]. |
| 8 | MTP speculative decoding | B | tokens per pass | -5% to +30% | Mixed published results; 12 GB + DDR4 card saw +2%. [PUBLISHED, earlier research] |
| 9 | Precision tiers (hot Q6, cold Q3) | C | bytes and accuracy | unknown | Needs Q8_0 reference + KL baseline first. |
| 10 | Train or force router locality | C | - | **Do not** | Published pre-registered negative result: every setting broke the 1% perplexity gate; rerouting to cached experts changes outputs. [PUBLISHED, arXiv 2608.18261] |

## 6. Order of operations

Every phase: goal, steps, output, gate. Stop rule: if a gate fails, stop that branch and log why.

### Phase 1: Free checks on the running config (no new code)
Why first: it can change the bandwidth constant the whole model depends on.
- **1a** Read-only. Shows whether WSL can give big pages:
  `cat /sys/kernel/mm/transparent_hugepage/enabled; grep -i hugepages /proc/meminfo | head -3`
- **1b** `--load-mode none` at N=28 (check the flag exists: `llama-server --help | grep -i -A3 load-mode` [UNVERIFIED on this build]). Reason: anonymous memory can use huge pages; file-backed cannot.
- **1c** Threads `-t 8`, `-t 7`, `-t 6`. Read the current thread count from the server log first.
- **1d** Optional, load time only: GGUF on the WSL disk instead of `/mnt/c` (about 155 MB/s today if it is read from `/mnt/c`).
- **Gate:** keep a change only if generation is +3% or better at all three depths and GPU stays <= 11.2 GiB.

### Phase 2: Floors (small programs; I test logic in my sandbox, you run them)
- **2a** PCIe host-to-device and back, pinned vs pageable (Pelias already has pinning and PCIe tests to reuse).
- **2b** RAM read bandwidth with 4/6/8/12/16 threads on a 10+ GB buffer.
- **2c** `llama-bench` sweep of `--n-cpu-moe` (e.g. 40/34/28) to test the linear model with more than two points [flag name UNVERIFIED for this build].
- **2d** Tiny CUDA test: a kernel reading mapped pinned host memory; then run it together with 2b to see the combined bandwidth.
- **Gate G0:** the floor model reproduces 31.8 and 41.1 tok/s within 10%. If not, fix the model before anything else.

### Phase 3: Trace which experts fire, on your real traffic
- Build a tracer on **the same commit (2ca15f540)** in a separate build folder, so logits match the working server. It records the router's top-8 per layer per token, separating prefill from decode.
- **Known trap:** the top-k tensor is a non-contiguous view; a flat copy gives false "uniform" routing. Read the argsort tensor or copy with strides. [PUBLISHED: a study hit exactly this]
- Corpus: persona probes; capability suite; coding-agent transcripts captured while you use Visual Studio Copilot (`llama-server --log-prompts-dir`, listed in the server README [UNVERIFIED on this build]; the prompts contain your code, keep them local); general chat and reasoning. Hold out 20%.
- Output: `logs\moe_trace\*.npz` plus a manifest with checksums.
- **Gate G1:** at least 100k decode tokens across at least 4 task types [ESTIMATE of enough].

### Phase 4: Simulate (pure Python; I can test it fully on synthetic traces)
- Policies: static top-K (uniform, and variable K per layer chosen by marginal gain), LRU, LFU with decay, static + small LRU, and the perfect-knowledge bound.
- Budget sweep 15-45% of experts. Convert hit rate to tok/s with the section 4 model (both scenarios).
- Test generalization: pick the hot set from 80% of traffic, score on the 20%; also train on code, test on chat/persona.
- **Gate G2 (go/no-go for building):** held-out static hit >= 0.75 at the 29% budget (predicts >= 55 tok/s in scenario ii). If held-out hit < 0.65, stop this track. Note: a published study found LRU beat static pinning at a small budget on a different model; if static + LRU beats static by more than 5 points here, the design needs a dynamic region.

### Phase 5: Try what already exists before writing C++
- **5a** Mainline cache: build current master into `~/llama.cpp-moecache` (keep the working build). Confirm with `--help | grep moe-cache`. Record hit rate and speed.
- **5b** ParmesanParty fork: build into `~/llama.cpp-parmesan` (CUDA arch 89, OpenMP on). Its README example: `--override-tensor exps=CPU`, `LLAMA_ARG_MOE_HOT_K` (typical 32-56 on a 24 GB card; our budget suggests about 70), `LLAMA_MOE_HOT_PP_BYPASS_N_TOKENS=64`. The hot set there comes from a runtime counter, not a profile. MIT licence.
- Check both against the stock server: logits KL on fixed prompts, then speed, then probes.
- **Gate G3:** if either reaches >= 55 tok/s at 12k depth and passes the section 7 gates with GPU <= 11.2 GiB, **adopt it** (pin the commit) and stop building. Otherwise continue, now knowing exactly where it falls short.

### Phase 6: Build Pelias-Tier (only if G3 did not settle it)
Keep the patch small, on a pinned commit, split like the fork's (infrastructure / core / graph / model wiring / tests).
- **6.1** `hot_profile.json` from the Phase 4 chooser: per-layer hot expert ids, trace manifest hash.
- **6.2** Hot bank loader: allocate K_l slots per layer in VRAM; copy hot experts once at load; build a slot table (expert id -> slot or -1) on GPU and CPU.
- **6.3** Dual-branch expert op: one masked MUL_MAT_ID for hot (GPU), one for cold (CPU), weighted sum merged. Each selected expert handled exactly once. (Published design.)
- **6.4** Cold CPU branch with a fused expert op and a pinned thread pool.
- **6.5** Prefill path: use the normal path for batches above about 64 tokens. The fork's README reports about 3.5x slower prompt processing otherwise. Reserve VRAM for the prefill offload buffer so `-ub 2048` keeps its speed (an open question, see section 9).
- **6.6** Tests in my sandbox (CPU only): partition completeness, remap invariants, MUL_MAT_ID parity against a reference. On your machine: `test-backend-ops -o MUL_MAT_ID`, then logits KL and top-1 agreement vs stock on fixed prompts.
- **6.7** Speed, then the section 7 gates.
- Effort [ESTIMATE]: 4-8 working sessions; roughly 600-1,500 lines of C++/CUDA/Python.

### Phase 7: Uncharted add-ons (independent; each gated; in this order)
- **7a Zero-copy third branch.** Split cold experts between the CPU and the GPU reading pinned RAM. Only if 2d shows GPU zero-copy >= 15 GB/s and the combined CPU+GPU read is >= 1.25x CPU alone. Needs pinned registration of the cold weights (Pelias already pins 22 GiB in training; whether the inference build registers them is [UNVERIFIED]).
- **7b Prefetch + small dynamic region.** Capture hidden states for about 10k tokens (about 1.6 GB), train a tiny per-layer predictor offline (layer L state -> experts of L+1). No model change. Gate: predicted top-8 recall high enough that misses fall by >= 20% on held-out traces (the right threshold is unknown until measured).
- **7c VRAM governor.** Two launcher profiles: long (131k) and fast (65,536), each with its own hot-bank size. Reuse of idle decode-time buffers is an idea only, unverified.
- **7d MTP** (`--spec-type draft-mtp --spec-draft-n-max 2`) on the best config; the MTP block's experts may need VRAM.
- **7e Precision tiers (Class C).** Only after a Q8_0 reference and KL baseline (proposal Phase 4). Needs a GGUF with split hot/cold tensors; we own the export pipeline.

### Phase 8: Pelias-Serve, the full ground-up engine (trigger-based)
Start only if Phase 6 proves the mechanism but ggml's scheduler overhead or rebase churn blocks further gains, or Phase 7 needs control ggml cannot give.
- Reuse from Pelias: meta-device loader, shard-by-shard fill, 4-bit affine pinned cache, expert streamer (slots, side stream, events).
- Own: expert store, scheduler, decode loop, CUDA graphs for the static spine, CPU expert kernels.
- Must also provide (large): DeltaNet and attention decode kernels (DeltaNet and causal-conv1d kernels exist upstream; check the venv), q8_0 KV, sampler, chat template and tool-call parsing, an OpenAI-compatible streaming server, prefix cache for the recurrent state.
- Effort [ESTIMATE]: months. It would have to beat Phase 6's result to be worth it. I do not recommend starting here; this plan only gets here through the gates.

### Phase 9: Make it permanent
Launcher profiles; pinned build with checksums; rerun probe + capability + needles on every config change; RUNBOOK entry; no auto-pulling llama.cpp; keep the working build.

## 7. Acceptance gates (every candidate)

| Check | Required |
|---|---|
| Speed | generation >= +10% over matching baseline (N=28: 41.1 / 40.2 / 38.8) |
| Prompt | not slower than baseline by more than 5% (1,166 / 1,193) |
| GPU memory | peak <= 11.2 GiB with the display attached |
| Persona/format probe | not-pass <= 12 of 111 (Q4 ctx32k baseline: 9 of 111). Baseline for N=28 still to be measured. |
| Capability | needles 12/12 at 16k-120k, tools 10/10, recall 4/4, 0 loops |
| Numerics (class A) | report logits KL and top-1 agreement vs stock on fixed prompts; set a threshold after seeing the GPU-vs-CPU rounding noise floor |

## 8. Risks and kill criteria

- Hit rate on your traffic is lower than published: stops at G2.
- Per-layer CPU/GPU sync cost larger than modeled: check with the fork in Phase 5 before writing our own.
- WSL2 limits on pinned or mapped memory: Phase 2d decides.
- Prefill regression from the dual path: see 6.5.
- llama.cpp churn: pinned commit, small patch series, never auto-update.
- RAM: WSL is capped at 48 GB; cold experts take about 20+ GB.
- Real Visual Studio use depends on prompt reuse in this hybrid model (reports of full re-processing each turn): separate from this track, proposal Phase 5.
- Do not touch the router (see lever 10).

## 9. What I do not know [UNVERIFIED]

- Whether this build (2ca15f5) has `--load-mode`, `--log-prompts-dir`, or llama-bench `-ncmoe`.
- Whether my sandbox can clone and build llama.cpp CPU-only (needed for the CPU test plan).
- Whether the CPU-resident experts are pinned under `-ot`, and whether mapped pinned memory works under WSL2.
- The fork's real speedups (its README gives none) and whether it works with your GGUF and `-ub 2048`.
- How VRAM is split between the hot bank and the prefill offload buffer.
- Your traffic's actual skew; the 0.65 GB/token figure (derived from an estimated 2.04 MB per expert).
- Whether huge pages are available in this WSL kernel.

## 10. Who does what

- Me: write and test CPU-side logic (trace analysis, simulator, hot-set chooser, profile and GGUF writers), draft patches and unit tests.
- You: run builds and GPU tests, paste logs. One step at a time; no file is created without you asking.

## 11. Proposed folders (not created)

`<project>\moe_track\` with `microbench\`, `trace\`, `sim\`, `patches\`; results in `logs\floors\` and `logs\moe_trace\`. Every run records label, command, commit hash and input checksums.

## 12. Sources

- ParmesanParty/llama.cpp README (fetched 2026-10-20).
- "Cacheable by Design?", arXiv 2608.18261 (fetched 2026-10-20): routing measurements, cache policy simulator, negative result on router training.
- Pelias docs: `pelias-training-v3\README.md`, `03_Blueprint_General_MoE_Training_Tool.md` (read 2026-10-20).
- HWiNFO report from you, 2026-10-20 03:45; your speed logs under `logs\speed_server\`.
- Earlier research (routing skew numbers, mainline `--moe-cache-mib`, MTP results, KV sizes) recorded in `speed_and_accuracy_proposal.md`; not re-fetched this session.

## 13. Update 2026-10-20 (after reviewing the "pin 44% in VRAM, stream the rest over PCIe, reach 70" argument)

**What holds:** decode is bandwidth-bound; about 320 experts (8 x 40 layers) are read per token; keeping some experts resident and fetching the rest is the right family; PCIe about 22 GB/s is your own earlier Pelias test [USER-REPORTED].

**Corrected constants** [DERIVED]
- Expert size: Q4_K_M stores the expert down projection at Q6_K in about half the layers (llama.cpp's rule as I remember it; UNVERIFIED against this file). That gives 1.77 MB per expert in Q4_K-down layers and 2.04 MB in Q6_K-down layers, about 1.9 MB on average. Cross-check: the 12 resident layers cost 5.6 GiB, about 1.96 MB per expert, which matches 8 Q6_K-down layers plus 4 Q4_K-down layers. This replaces the 2.04 MB in section 4; bytes per token are about 0.61 GB, not 0.65.
- VRAM available for experts: about 5.6 GiB (card at 10.4 of 12.0 GiB with the display, 1.6 GiB left), not 8 GB. That is about 3,150 experts, about 79 per layer, about 31% of all experts, not 44%.
- Fixed cost about 7.4 ms per token (spine, lm_head, DeltaNet, sampling, launches); CPU expert path about 25 GB/s. Section 4's 7.7 ms and 27 GB/s were rounded from the same two points; these fit better.

**Four errors in the argument**
1. Pinned fraction is not hit rate. A uniform or whole-layer pin hits about as often as the fraction pinned (about 31%, which is today's 41 tok/s by construction). A profile-guided pin hits about 79% at 30% of experts on this model [PUBLISHED, third-party prompts]. Choosing which experts to pin is the whole gain (Phases 3-4).
2. The fixed cost is missing. The argument's 14.2 ms is only the transfer; adding 7.4 ms gives about 22 ms, about 46 tok/s, even on its own assumptions.
3. The prefetch cannot name layer L+1's experts. Layer L+1's router reads the output of layer L, so its 8 experts are unknown until layer L finishes. Pelias's async copy worked because training streams whole layers, known in advance. In decode only these work: a resident hot set; fetching misses after routing (the overlap window is small); or predicted prefetch (accuracy on this model unknown, Phase 7b).
4. A PyTorch eager decode loop likely has a larger fixed cost than llama.cpp's (CUDA graphs): [ESTIMATE] 15-30 ms before any expert traffic, which would cap it below today's speed. A ground-up engine belongs at ggml level (Phase 6) or native (Phase 8), not in eager PyTorch.

**Also:** the measured runs are not all-CPU (31.8 tok/s with all experts on CPU, 41.1 with `--n-cpu-moe 28`; VRAM 4.8 vs 10.4 GiB). The launcher's default `-ot` override is replaced by `EXPERT_FLAGS`; a log that shows the default is the echoed default or a different run. Stock llama.cpp in your build (2ca15f5) does not fetch per token, but mainline's `--moe-cache-mib` (merged 2026-10-07 per earlier research, UNVERIFIED here) does exactly that; test it in Phase 5a before building anything.

**Revised prediction** (fixed 7.4 ms, 0.61 GB per token; low end adds 4 ms of sync/launch cost for the CPU path or 2 ms for the PCIe path, high end adds none) [ESTIMATE]

| Hit rate | CPU computes misses (25 GB/s) | PCIe uploads misses (22 GB/s) |
|---|---|---|
| 0.31 (uniform pin = today) | 41 | 38 |
| 0.70 | 53-68 | 56-64 |
| 0.79 (profile-guided, published) | 61-80 | 66-76 |
| 0.90 | 72-102 | 82-98 |

Both miss paths read the same RAM, so using both together is capped near 43 GB/s total (Phase 2d).

**Plan changes:** (a) use the corrected constants above; (b) Gate G2 becomes held-out static hit >= 0.75 at the 31% budget; (c) the miss path (CPU compute, PCIe upload, or both) is an explicit comparison in Phases 5-6, with mainline's cache as the PCIe-upload baseline; (d) new Phase 2 item: read the real per-layer expert sizes from the GGUF to replace the derived 1.9 MB; (e) order of phases unchanged.
