# Pelias Execution Ledger: Confirmed Baselines & Follow-Up Action Plan

> **Document Status:** Master Operational Summary & Verification Checklist
> **Target Hardware Baseline:** NVIDIA RTX 4070 12 GB (`sm_89`, 12,282 MiB VRAM, 504–540 GB/s GDDR6X, 36 MB L2 Cache, PCIe 4.0 x16 @ 16 GT/s, Resizable BAR ON), AMD Ryzen 7 5800X3D (8C/16T, 4.54 GHz, 96 MB L3, AVX2), 64 GB DDR4-3200 CL16 Dual-Rank XMP (51.2 GB/s theoretical peak), Windows 11 + WSL2 Ubuntu 26.04 (48 GB VM Cap, Hyper-V Active, HVCI Off).

---

## Part I: What We Have Confirmed (Empirical Hardware & Software Ledger)

Every item listed below has been verified against your measured server logs, GGUF tensor inspections, and hardware scans. No assumptions or theoretical estimates are included here.

1. **Baseline Generation Speeds (`Q4_K_M`, KV `q8_0`, `-c 131072`, `-ub 512` / `2048`):**
   * **All-CPU Experts (`--n-cpu-moe` omitted):** Achieves **$31.8\text{ tok/s}$** (short), **$31.5\text{ tok/s}$** (4k), and **$31.3\text{ tok/s}$** (12k). Prefill wait is **$13.1\text{ s}$** at 12k depth. Active VRAM is **$4.80\text{ GiB}$** (`logs/speed_server/20261020_025525_q4km-kvq8_0-ctx131k-ub2048.txt`).
   * **`--n-cpu-moe 28` (12 Layers / 3,072 Experts on GPU):** Achieves **$41.1\text{ tok/s}$** (short), **$40.2\text{ tok/s}$** (4k), and **$38.8\text{ tok/s}$** (12k). Prefill wait drops to **$10.1\text{ s}$** at 12k depth. Active VRAM spans **$10.20 – 10.40\text{ GiB}$** (`logs/speed_server/20261020_032031_q4km-ncmoe28-ctx131k-ub2048.txt`).

2. **Bimodal Expert Geometry & Sizing (`Q4_K_M` GGUF Header Audit):**
   * Quantization divides `ffn_gate_exps` and `ffn_up_exps` into `Q4_K`, while promoting `ffn_down_exps` to **`Q6_K` in half the layers** and keeping it at **`Q4_K` in the remaining half**.
   * `Q4_K`-Down Layer Expert Size: $\approx 1.77\text{ MB}$ ($453.1\text{ MB}$ per layer).
   * `Q6_K`-Down Layer Expert Size: $\approx 2.04\text{ MB}$ ($522.2\text{ MB}$ per layer).
   * Fleet Average Expert Size: **$1.905\text{ MB}$** per expert ($18.16\text{ GiB}$ total routed pool; $0.609\text{ GB}$ read per token across $320$ active expert calls).

3. **Two-Point Linear Timing Decomposition & Fixed GPU Spine:**
   * Moving 12 layers to VRAM saved **$0.593\text{ ms}$ per layer** ($7.12\text{ ms}$ saved across 12 layers).
   * Extrapolating the All-CPU baseline minus the per-layer expert read cost yields a **Fixed GPU Spine Latency ($T_{\text{spine}}$) of $7.4\text{ ms/token}$** ($1.54\text{ GiB}$ resident spine: attention, DeltaNet recurrent state, shared experts, routers, RMSNorms, `lm_head`, and CUDA kernel launch overhead).
   * Effective CPU expert read bandwidth: **$25.3\text{ GB/s}$** ($\approx 50\%$ of the $51.2\text{ GB/s}$ theoretical DDR4-3200 dual-channel ceiling).

4. **PCIe & Memory Controller Hardware Ceilings (`HWiNFO` & Verification Logs):**
   * PCIe Link: **Gen4 x16** (`logs/verification/quick_20261019_082122.json`).
   * Host-to-Device (H2D) Bandwidth: **$11.1\text{ GB/s}$** from pageable WSL2 memory and **$26.3\text{ GB/s}$** from page-locked (`cudaHostAlloc`) pinned memory.
   * Shared Memory Controller Ceiling: Dual-channel DDR4-3200 XMP active; simultaneous dual-bus reads saturate at **$\approx 43\text{ GB/s}$**.

5. **Prefill Chunking Impact (`-ub 2048`):**
   * Switching chunk size from default 512 to 2048 cut 12k prefill wait from **$42.6\text{ s}$ down to $13.1\text{ s}$** (all-CPU) and **$10.1\text{ s}$** (`N=28`), proving prompt reading is severely copy-bound over PCIe.

---

## Part II: What We Have to Follow Up On (Explicit Action Items & Verification Gates)

The following items represent the strict order of execution. Nothing runs in production until each gate is formally cleared by silicon measurements.

### 1. Phase 1 — OS & Memory-Translation Tuning (Immediate Execution)
* [ ] **Step 1a:** Verify WSL2 Transparent Huge Pages status (`cat /sys/kernel/mm/transparent_hugepage/enabled`) and ensure anonymous allocations back expert buffers with 2 MB huge pages.
* [ ] **Step 1b:** Execute the pinned anonymous load on top of `N=28` (`--load-mode none`), verifying that 12k prefill drops to the **$6.1 – 6.5\text{ s}$** target band while VRAM stays $\le 11.2\text{ GiB}$.
* [ ] **Step 1c:** Run thread contention sweeps (`-t 7` vs. `-t 6` with `--prio 2`) to ensure CPU worker threads do not starve the CUDA driver dispatch thread.

### 2. Phase 2 — Hardware Floor Microbenchmarks (`moe_track/microbench/`)
* [ ] **Step 2a (`01_gguf_expert_sizes.py`):** Parse the exact byte sizes and quant types (`Q4_K` vs. `Q6_K`) for all 40 layers into a static lookup table.
* [ ] **Step 2b (`02_ram_bw_test.cpp`):** Measure sustained AVX2 read bandwidth across $4, 6, 7, 8, 12, 16$ threads with 4 KB vs. 2 MB huge pages.
* [ ] **Step 2c (`03_handoff_latency.py`):** Measure $1,000$ round-trips of CPU-GPU pinned memory handoffs to establish exact per-layer sync cost $s$.
* [ ] **Step 2d (`04_dual_bus_test.cu`):** Run concurrent CPU AVX2 + GPU zero-copy mapped memory tests to verify if combined bandwidth exceeds $1.25\times$ CPU-alone ($\ge 15\text{ GB/s}$ GPU zero-copy).
* **Phase 2 Gate (G0):** Calibrated floor equation must predict measured All-CPU ($31.8\text{ tok/s}$) and `N=28` ($41.1\text{ tok/s}$) runs within $\pm 10\%$.

### 3. Phase 3 & 4 — Routing Tracer & Offline Water-Filling Simulator (`moe_track/trace/` & `sim/`)
* [ ] **Phase 3:** Build the stride-aware `cb_eval` callback hook on commit `2ca15f540` to correctly extract non-contiguous `ffn_moe_topk` views without false uniform distortion. Capture $\ge 100,000$ decode tokens across persona, coding, and tool-call domains.
* [ ] **Phase 4:** Replay traces through the size-aware water-filling knapsack simulator under the **$5.60\text{ GiB}$ ($31\%$) expert VRAM budget** (leaving $1.6\text{ GiB}$ free for Visual Studio 2026).
* **Phase 4 Gate (G2):** Held-out hit rate must reach **$\ge 0.75$**. If $<0.65$, halt custom C++ cache development.

### 4. Phase 5 — Mainline & Fork Verification
* [ ] **Step 5a:** Test upstream mainline `llama.cpp` GPU expert cache (`PR #29887`, `--moe-cache-mib 5734`).
* [ ] **Step 5b:** Test the `ParmesanParty` dual-branch fork with prefill bypass (`LLAMA_MOE_HOT_PP_BYPASS_N_TOKENS=64`).
* **Phase 5 Gate (G3):** Adopt if either build achieves $\ge 55\text{ tok/s}$ at 12k depth, keeps prefill $\ge 1,150\text{ tok/s}$, and stays $\le 11.2\text{ GiB}$ VRAM. Otherwise, proceed to Phase 6.

### 5. Phase 6 — Custom C++/CUDA Engineering (`Pelias-Tier` Patch)
* [ ] **Module 6.1:** Implement the profile-guided hot bank loader (`pelias_hot_bank.cpp`) to seed water-filled experts into VRAM at startup.
* [ ] **Module 6.2:** Wire the dual-branch masked `MUL_MAT_ID` and miss arbiter (`pelias_moe_op.cu`) with in-place CPU AVX2 fused MLP execution (`Lane C`) for decode batches (`n_tokens < 64`).
* [ ] **Module 6.3 / 6.4:** Implement dynamic VRAM reclamation and optional layer-major prefill for prompts $>8,192$ tokens.

### 6. Training Engine Upgrade (`Pelias v0.3`)
* [ ] **Module 1 (`L5` Stash):** Implement pinned-host activation checkpointing ($2.5\text{ GiB}$ pinned host buffer + $192\text{ MiB}$ VRAM ring) to cut PCIe expert traffic by $50\%$ and unlock **16,384-token zero-truncation training** within $\approx 8.8\text{ GiB}$ peak VRAM.
* [ ] **Module 2 (Zero-Sync Dispatch):** Replace `.tolist()` synchronization barriers in `pelias/experts.py` with device-side `cumsum` offsets (`torch.nn.functional.grouped_mm`).
* [ ] **Module 3 & 4:** Move peak VRAM tracking outside the micro-batch loop, fuse Triton dequantization into GEMM prologues, and implement native `Q4_K_M` super-block export for $0.00\%$ train-to-serve quantization drift.
* [ ] **Module 5 (`blk.40` MTP & KV Probing):** Co-train `blk.40` for speculative decoding and enable `use_cache=True` in `stage_probe` to cut persona stress-test evaluation time from $15.6\text{ minutes}$ down to $\approx 45\text{ seconds}$.

---

## Part III: Strict Operational Rules & Guard Rails

1. **Preserve VRAM Headroom:** Total VRAM usage must never exceed **$10.40\text{ GiB}$** during inference and **$10.50\text{ GiB}$** during training to ensure Visual Studio 2026 and Windows display composition remain fully responsive.
2. **Exactness Gate (Gate H6):** Logit KL divergence between the stock server and the modified runtime must not exceed the natural float spread between `-ub 512` and `-ub 2048`.
3. **Loud Fallbacks:** Any pinning failure or memory translation error must fail loudly rather than falling back silently to unpinned pageable memory.