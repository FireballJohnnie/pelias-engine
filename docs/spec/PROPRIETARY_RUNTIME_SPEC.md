# Pelias proprietary runtime and server specification (architecture, file roles, parameters, order of operations)

> **Provenance.** Extracted on 2026-10-25 from `PELIAS_PROPRIETARY_MASTER_SERVER.md` (built 2026-10-24, sha256 begins `56634f43`, held on the author's drive as a local archive and not part of this repository): Part 6 in full (master lines 2892 to 3481). The text is word for word. Only three things differ: headings are moved up one level, the account name is written `<account>`, and this header is added. 
>
> **Status.** The engine is HELD (author, 2026-10-24 06:07). This is documentation of a design; no engine source code exists in this repository. Labels (MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED) are used as defined in the blueprint; a DECISION marked "proposed" is not ruled.
>
> **Part numbers in the text below** refer to the master document. Where each one lives in this repository:
>
> | Reference in the text | Where it is in this repository |
> |---|---|
> | Part 1.6 (reconciliation of the research update) | `docs/spec/RECONCILIATION.md` |
> | Part 1.1, 1.1a (research update: core discovery, retirement of 57.0, E0 split, audit G1 to G18) | `docs/spec/RESEARCH_UPDATE_AUDIT.md` |
> | Part 1.2 to 1.5 (machine, engine facts, drifted assumptions, claim-by-claim review) | `docs/spec/MACHINE_AND_DRIFT.md` |
> | Part 3.1 (engine results ledger) | `benchmarks/RESULTS.md` and `benchmarks/raw/` |
> | Part 3.3, 3.4, 3.5 (Wave 0 results, modelled ceiling, speed-to-ceiling table, attribution register) | `docs/spec/WAVE0_AND_CEILING.md` |
> | Part 4.2 to 4.5, 4.8, 4.9 (engine components, script inventory, patches, microbenchmarks) | `docs/spec/PATCHES_AND_TOOLS.md` |
> | Part 5.3, 7.13 to 7.15, 8.10 (queued items, what is needed, open author decisions, measurement registers, the hold) | `docs/spec/ENGINE_QUEUE_AND_MEASURABLES.md` |
> | Part 8 (history) | summarized in `docs/PELIAS_ENGINE_LINEAGE.md`; the full records stay in the author's local archive |
> | Part 4.10 (replacement map) | `docs/PELIAS_ENGINE_MASTER_SCAFFOLD.md`, section 5 |
> | Part 5.0 (synchronization register) | `docs/spec/SYNC_REGISTER.md` |
> | Part 5.1, 7.2 to 7.6 (synchronized engine blueprint) | `docs/PELIAS_ENGINE_LINUX_BLUEPRINT.md` |
> | Part 5.2 (trainer co-design contract) | `docs/spec/TRAINER_CONTRACT.md` |
> | Part 6 (proprietary architecture and server specification) | `docs/spec/PROPRIETARY_RUNTIME_SPEC.md` |
> | Part 7.1, 7.16 (order of operations, open decisions O1 to O14) | `docs/spec/OPEN_DECISIONS_AND_ORDER.md` |
> | Part 7.9, 7.10, 7.11 (probe registry, decision register) | `docs/spec/PROBE_REGISTRY.md` |
> | Other parts (1.7, 2, 3.2, 4.1, 4.6, 4.7, 5.4, 7.7, 7.8, 7.12, 8.3 to 8.9, 8.11 to 8.13) | not copied; they stay in the local master document (they concern the trainer, quote private paths and names, or repeat files that are already in this repository, such as `support/archive/c_ai/creator/moe_track/UNFINISHED.md` for 4.6) |
> 

---

# Part 6. The 100% proprietary ground-up architecture, server specification and scaffolding

**This part is architecture, file roles, parameters and order of operations. It contains no code.** Operating-system call names are written in prose where the specification needs them, because the author asked for the Winsock2, POSIX socket and file-mapping boundaries to be named.

**How to read it.** 6.1 is the research update's own unified-program section, word for word: its layers 0, 1, 2A and 2B and its seven-line diagram are the starting point the author named. 6.2 onward is new text that (a) corrects 6.1 where the ground-up ruling or the 10.7 GiB ruling changes it, (b) combines its layers with the blueprint's scaffolding tree and the trainer's module tree into one module map, and (c) writes the specification that no document had: the proprietary boundary, the operating-system layer, the process and threading model, the doorbell coordinator, the memory-mapped GGUF reader, the arena and VRAM governor, the quantizers, the kernels, the native tokenizer and chat-template renderer, the raw-socket HTTP and SSE server, the gap register for the llama-server behaviour that was never catalogued, and the order in which it is built.

**Labels in this part.** A statement that is a design choice is a **DECISION (proposed)** until the author rules; it is the strongest label a new design statement gets here. A figure is quoted only from a spliced block and keeps its label. A thing nobody has measured says so, with the probe that would measure it. **No effort or schedule figure is given for any module: they are unmeasured.**

**Status of the whole part.** The engine is HELD (author, 2026-10-24 06:07). Nothing here starts engine work; it is the specification that is ready when the hold is lifted, and the shared items (tokenizer, renderer, GGUF reader, quantizers, SHA-256, JSON) are the ones the trainer needs first (trainer stages T1, T1b, T2), so they are specified to be written once and shared (6.4).

## 6.1 The research update's unified-program section (layers 0, 1, 2A and 2B), word for word

<!-- SPLICE id="P6.1" src="pelias-research-update.md (received 2026-10-23 19:59, 870 lines)" lines="652-758" shift="+2" ws="1" edited="0" -->
> **Source (verbatim):** pelias-research-update.md (received 2026-10-23 19:59, 870 lines), lines 652-758. The starting point the author named. Its out-of-date points are listed and corrected in 6.2. Heading levels shifted by +2 for this document only. Table rows that the source breaks over several lines are joined back (whitespace only).

### 7. Master Pelias Blueprint: Ground-Up C++20/CUDA Co-Existing Application & Universal Substrate

#### 7.1 Why Unifying `pelias-engine` and `pelias-trainer` in C++20/CUDA Is Architecturally Exact

In `PELIAS_TRAINER_README_main.md` (§5–6, `2026-10-20`), the two repositories were separated only because the `v0.2.1` trainer was written in Python/PyTorch (`49` files, `725,010 B` / `~708 KiB` of source code) while the engine was C++/CUDA[cite: 1, 3]. Now that both are ground-up **C++20 + CUDA 13.0.88 (`sm_89`)** designs targeting the exact same machine, model (`Qwen3.6-35B-A3B`), GGUF file (`753` tensors, `21,713,463,296 B`), and oracle (`llama.cpp 2ca15f540`) (`2026-10-23`)[cite: 1, 2, 4], building them as a single cohesive codebase (`pelias`) with a shared core library (`libpelias_core` + `libpelias_substrate`) provides five concrete engineering advantages:

1. **Zero Forward-Pass Divergence (`K1` = `E7` by Construction):**
The engine's Phase E7 spine (`attention.cu`, `deltanet.cu`, `norm.cu`, `router.cu`, `lm_head.cu`)[cite: 4] and the trainer's Stages T3–T8 forward pass (`G1/G2` dense GEMMs, `E1–E5` norms/SwiGLU, `R1` fp32 router, `A1/A2` IMRoPE + GQA attention, `D1–D5/D7` DeltaNet) compute the **exact same mathematical function** (`§2.4` of Trainer Scaffold). Compiling them from one `src/core/spine/` directory guarantees that what is trained is bit-for-bit what is served[cite: 1, 2, 4].


2. **One `10.9 GiB` VRAM Governor, Arena Allocator, and Tri-OS Host Memory Layer (`hal/`):**
Both applications enforce the **`10.9 GiB`** ceiling (`Governor`), ban dynamic `cudaMalloc` inside the hot loop (`Arena`), and use `cudaHostAlloc` on WSL2 (`20,448 MiB` proven, where `cudaHostRegister` returns a device pointer $\neq$ host pointer)[cite: 1, 2, 4]. One `hal/` module serves both the Engine's **`7.5 GiB` static bank (`10.40–10.69 GiB` peak)** and the Trainer's **`7.90 GiB` streaming layout** (`2 × 498 MiB` expert slots + `0.80 GiB` spine + `0.32 GiB` LoRA/Adam + `2.50 GiB` working set).


3. **One Bit-Exact Quantizer/Dequantizer Suite (`block_types.h`, `G8`, `G9`, `P-E6`):**
Both the engine (`E4/E5`) and the trainer (`T3/T12` and Mode Q `§4.10`) require bit-exact ports of `ggml`'s `Q4_K` (`144 B`), `Q6_K` (`210 B`), `Q8_0` (`34 B`), `q8_1` (`36 B` GPU `32`-element activation block), and `Q8_K` (`292 B` CPU `256`-element activation block)[cite: 1, 2, 4]. Implementing **Mode B (bimodal-faithful forward, `D24`)** in the trainer becomes immediate when the engine's CPU `Q8_K` and GPU `q8_1` activation quantizers live in the shared core.


4. **In-Memory Closed Loop for Mode L Router Training & Hot-Set Profiling (`D21`, `D25`):**
The routing `Tape` (`40 layers × N tokens × top-8`) and the C++ `PELIAS_HOT_SET` water-filling / zero-miss solver (`P-E14`) are shared between `pelias trace` (engine) and `pelias train` / `pelias export` (trainer)[cite: 1, 2, 4]. After training finishes, the unified binary can switch from `TrainerArena` to `EngineArena` in-process and run the `P-X5` live decode speed and `K3` KL gates immediately[cite: 1, 2, 4].
5. **Single Wave 0 + E0/E1 Probe & Oracle Harness (`pelias probe` & `pelias verify`):**
Unifies the trainer's `46` Wave 0 probes (`P-H`, `P-M`, `P-K`, `P-Q`, `P-S`, `P-D`, `P-L`, `P-E`) with the engine's E0/E1 microbenchmarks (`02_ram_bw`, `03_handoff`, `04_dual_bus`, `05_pin_modes`, `06_overlap`, `07_wide_load`)[cite: 1, 4], writing one `probe_constants.json` and machine-state hash.



#### 7.2 Master Co-Existing Architecture & Universal Substrate Diagram

To achieve the stretch goal of making Pelias **universal across consumer hardware, MoE LLMs, dense LLMs, and non-LLM sparse workloads** (`PELIAS_TRAINER_README_main.md` Footnote 1 & `PELIAS_ENGINE_LINUX_BLUEPRINT.md` Phase E13)[cite: 1, 3, 4], the C++20/CUDA architecture separates **Layer 0 (`libpelias_substrate`: The Domain-Agnostic Unified-Bus Hardware Abstraction)** from **Layer 1 (`libpelias_core`: Shared Model I/O, Spine Kernels & Verification Oracle)** and **Layer 2 (`pelias-engine` & `pelias-trainer` via zero-overhead C++20 Concepts)**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                         PELIAS MASTER C++20 / CUDA 13.0.88 APPLICATION                      │
│              One Binary (`pelias`) · Three Targets (`libpelias_core`, `engine`, `trainer`)  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 0: `libpelias_substrate` (Universal Consumer Unified-Bus Hardware Abstraction)        │
│   ├─ `hal/vram_governor` & `arena`: Hard 10.9 GiB ceiling, zero-malloc step/token loops     │
│   ├─ `hal/host_alloc_{linux,wsl,win32}`: Pinned DMA pools, 2 MiB THP / LargePage probes     │
│   ├─ `bus/doorbell_handoff`: 13.3 µs lock-free acquire/release + warp `__nanosleep` sync    │
│   ├─ `bus/cpu_threadpool`: Physical core (`1T`) pinning, `PREFETCHNTA`, zero-`LOCK` FMA     │
│   ├─ `bus/stage_streamer`: 3-stream DMA (`S_compute`, `S_w` H2D, `S_x` boundary D2H/H2D)    │
│   ├─ `bank/hot_set_solver`: Water-fill (byte-hit h) & Zero-Miss (Z_l) residency optimizer   │
│   └─ `quant/ggml_exact`: Bit-exact Q4_K, Q6_K, Q8_0, q8_1 (32), Q8_K (256) CPU/GPU kernels │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: `libpelias_core` (Shared Model I/O, Spine Kernels & Verification Oracle)           │
│   ├─ `io/gguf_splice`: 753-tensor mmap reader, byte-splice exporter, SHA-256 payload proof, │
│   │                    expert-last 2 MiB alignment (`pelias.pad`), `layout_manifest.json`   │
│   ├─ `tok/libllama_bridge`: Exact GGUF tokenizer & Jinja chat-template (`reasoning_content`)│
│   ├─ `spine/forward_kernels`:                                                               │
│   │    • RMSNorm (+1 folded), Gated RMSNorm (`ssm_norm`), per-head L2Norm (`eps = 1e-6`)    │
│   │    • Attention (`10` trunk + `blk.40`): joint Q+gate, IMRoPE `[11,11,10,0]`, GQA `256`  │
│   │    • DeltaNet (`30` layers): 4-tap causal conv1d + SiLU, chunked `C=64` WY/UT inverse   │
│   │      (used by BOTH Engine Prefill & Trainer Forward) + sequential fp32 warp recurrence  │
│   │      (used by BOTH Engine Decode & Trainer Rung-2 Oracle `P-S7`!)                       │
│   │    • Router (`41 × 2048 × 256` F32): fp32 FMA top-8 argmax, 8-way softmax, `Tape` I/O   │
│   └─ `verify/oracle_2ca15f540`: Evaluates K1 (Parity), K2 (Export), K3 (Engine) vs σ_float  │
├──────────────────────────────────────────┬──────────────────────────────────────────────────┤
│ LAYER 2A: `pelias-engine` (Inference)    │ LAYER 2B: `pelias-trainer` (Ground-Up LoRA/QAT)  │
│   • Execution: Token-Major, Dual-Bus     │   • Execution: Layer-Major, K windows/sweep      │
│   • VRAM Bank: 7.5 GiB (4,170 experts)   │   • Pinned Store: 18.65 GiB experts (79 fetches, │
│   • Hot Branch: GPU `expert_q4k/q6k.cu`  │     35.8 GiB/step = 1.49s @ 26.1 GB/s, hidden)   │
│   • Cold Branch: Parallel CPU AVX2 +     │   • Core: 310 LoRA modules (21.17M params, fp32) │
│     GPU 16-B (`uint4`) staged zero-copy  │   • Backward: Transposed-dequant dgrad (`G5/G6`),│
│   • State: `q8_0` KV cache + GDN state   │     FA2 bwd (`A3`), DeltaNet reverse-state (`D6`)│
│   • Decode: CUDA-graphed spine + MTP     │   • Loss: Chunked 256-pos fp32 CE, step-wide norm│
│   • Server: OpenAI HTTP + VS Copilot     │   • Gated Modes: Mode Q, Mode L/L-Z/L-C, Mode B/N│
└──────────────────────────────────────────┴──────────────────────────────────────────────────┘

```

#### 7.3 How the Unified-Bus Substrate Scales Universally Beyond Qwen3.6 and Beyond LLMs

Because **Layer 0 (`libpelias_substrate`)** is parameterized over any staged computation graph with conditional block activation and auto-calibrates itself to any consumer PC via the startup capability probe (`E11` / `pelias probe`)[cite: 1, 3, 4]:

1. **Universal Hardware Portability Across Any Consumer PC (`E11` / `hal/`):**
* On any consumer desktop or workstation (PCIe 3.0/4.0/5.0, DDR4/DDR5, 8 GB–24 GB GPUs, AMD Zen or Intel Core CPUs across Native Linux, WSL2, or Windows 11), `pelias probe` measures the machine's exact standalone CPU RAM bandwidth ($B_{\text{CPU}}$), standalone PCIe zero-copy bandwidth ($B_{\text{GPU}}$), concurrent dual-bus ceiling ($B_{\text{dual}}$), doorbell latency ($t_{\text{doorbell}}$), and usable VRAM ceiling ($V_{\max}$)[cite: 1, 2, 4].
* Those measured constants automatically parameterize the `HotSetSolver` water-fill capacity, the CPU/GPU cold-branch split ratio, and the trainer's compute-hiding batch threshold $N_{\text{hide}} = \frac{2 \times \text{Bytes}_{\text{stream}}}{B_{\text{GPU}}} \times R_{\text{compute}}$[cite: 1, 2, 4].


2. **Other MoE LLMs (`E13` Plugin Interface):**
* Any Mixture-of-Experts architecture (such as `Gemma 4 26B-A4B`, `Qwen3-Coder`, or DeepSeek MoE variants mentioned in `PELIAS_TRAINER_README_main.md` §10) plugs into Layer 1 by declaring its `ModelSpec` (layer pattern, expert dimensions, top-$k$ routing rule, and quantization types) without touching a single line of Layer 0 unified-bus code[cite: 1, 3, 4].


3. **Dense LLMs (`PELIAS_TRAINER_README_main.md` Footnote 1 Stretch Goal):**
* For a dense LLM (`12B`–`70B`) that exceeds `10.9 GiB` VRAM, the trainer's **Layer-Major Streamer (`bus/stage_streamer`, §5.4–5.6)** works out of the box—streaming frozen quantized dense FFN/projection layers into the two `498 MiB` VRAM slots over $K$ micro-batches while training LoRA adapters in VRAM.


* On the inference side, `pelias-engine` partitions dense FFN weight matrices into static column/row tiles split across the `504 GB/s` VRAM hot bank and the `47.62 GB/s` CPU+PCIe unified bus, coordinated by the same **`13.3 µs`** doorbell handoff[cite: 1, 3, 4].


4. **Non-LLM Sparse & Out-of-Core Workloads:**
* Any domain where a `20–64 GB` state or parameter graph sits in system RAM and is queried sparsely per step can execute directly on `libpelias_substrate` using `HotColdBankManager`, `DoorbellCoordinator` (`13.3 µs`), and `StageMajorStreamer` (`26.1 GB/s`) without modification[cite: 1, 4]:
* **Out-of-Core Graph Neural Networks (GNNs) & Knowledge Graphs:** High-degree hub nodes remain pinned in the GPU VRAM hot bank (`80%+` hit rate), while cold neighborhood subgraphs are aggregated concurrently by CPU AVX2 threads and GPU `uint4` zero-copy reads across the unified bus.


* **3D Gaussian Splatting & Out-of-Core Ray-Tracing BVH Trees:** Frustum-visible primary splats/BVH nodes reside in the GPU hot bank while secondary/cold LOD splats stream over PCIe via `16-byte` shared-memory staging.


* **Block-Sparse Scientific PDE & Finite-Element Solvers:** Active high-gradient mesh blocks compute in VRAM Tensor Cores while quiescent boundary blocks update in parallel on the CPU across pinned DDR4/DDR5.







---

<!-- /SPLICE P6.1 sha256="69f483efbc16e824ac0248c1119622bbf58478b4f117327875839fdef8558b53" -->

## 6.2 Corrections to 6.1, and what 6.1 still gives (new text)

The research update's section 7 was written before three rulings of 2026-10-23 and 2026-10-24. Its layers and its list of five advantages stay as the starting design; these are the places where it is out of date. The table is complete for 6.1; the same corrections apply wherever the update's text appears in Parts 2, 3 and 5.

| # | 6.1 says | Governs instead | Basis |
|---|---|---|---|
| 1 | A hard 10.9 GiB ceiling in the governor (advantage 2, Layer 0) | The ceiling is measured at start: the smaller of CUDA free and NVML free after the context exists, minus a margin; 10.7 GiB is this machine's value until the margin is measured | D26, D28; Part 1.1a row 1 |
| 2 | The engine's 7.5 GiB bank peaks at 10.40 to 10.69 GiB | Those peaks are whole-card readings at two micro-batches (the default and -ub 2048); the micro-batch of the 10.40 to 10.44 pair is unrecorded | Part 5.1 table; D2 |
| 3 | The trainer's "7.90 GiB streaming layout" as four items | A 13-row plan that sums to 7.90 GiB | D8; trainer master 5.3 |
| 4 | Layer 1 holds `tok/libllama_bridge`, an "exact GGUF tokenizer and Jinja chat-template" | A native tokenizer and template renderer, original code (6.13). No libllama is linked | D37; blueprint R2 |
| 5 | Layer 0 holds `quant/ggml_exact` | `quant/`: the same bytes and arithmetic, written as original code, proven equal to ggml's by tests (6.11) | D37 |
| 6 | Layer 1 holds `verify/oracle_2ca15f540` | The oracle is an external, read-only harness under `tests/oracle/`; the product's `verify/` holds only the product's own comparators (6.3) | Blueprint R2 |
| 7 | "One binary (`pelias`), three targets" and an in-process switch from the trainer arena to the engine arena | Unification is step 5 of the governing order and is held. Until then there are two repositories and one shared core that each links. The in-process switch is an idea, not a measured feature [UNVERIFIED] | Governing order (Part 8.2); reconciliation A5 (1.6) |
| 8 | `cudaHostAlloc` on WSL2 with 20,448 MiB proven | 32 GiB of `cudaHostAlloc` memory held on WSL2 and on Windows-native; the limit above 32 GiB is not found | P-H7, Part 3.3 |
| 9 | `hal/host_alloc_{linux,wsl,win32}` | `host_alloc_wsl` and `host_alloc_win32` are built; `host_alloc_linux` is held with native Linux | Blueprint R1 |
| 10 | Layer 0 `bus/doorbell_handoff` is "13.3 us lock-free" | 13.3 us median and 18.7 us p99 are a microbenchmark of a flag plus a 4 KB read in a resident kernel on WSL2; no running engine has a doorbell, and with a real CPU workload in the stream it is not yet measured | Note to Part 5.4 (reconciliation 7.4); ledger section 5 |
| 11 | Advantage 1: "K1 = E7 by construction" | True only for code that is genuinely shared. The weight matrix products differ by role (training: dequantize to the tensor-core bf16 path; serving: int8 `mmq` with `q8_1` activations), so K1 and K3 stay as gates. Four levels of sharing: L0 types, L1 quantizers, L2 non-matmul spine, L3 matmul (not shared) | Reconciliation A6 |
| 12 | Section 7.3: out-of-core graph, splatting and PDE workloads "without modification" | No claim until a trace exists. A workload qualifies when its measured hit curve beats the break-even for its compute-to-byte ratio | Reconciliation A7 |
| 13 | Layer 2A box: "Server: OpenAI HTTP + VS Copilot" | A line, not a design. The design is 6.12 to 6.14 | This part |

## 6.3 The proprietary boundary

**Definition (DECISION D37, author ruling 2026-10-24 08:15, reason AUTHOR-STATED).** "100% proprietary" means that no code from llama.cpp, ggml, libllama, `common` or any JSON library is linked, copied or required at run time by any Pelias product, and that every component the product needs is written in this project. Where the product's bytes must equal ggml's (block layouts, quantizer arithmetic), the layout and arithmetic are re-expressed in original code after reading the reference, and the file's provenance line says so; credit stays in `CITATIONS.md`. That is an interoperability requirement and not a copy.

| Class | Members | Rule |
|---|---|---|
| **Product code** (original) | Every module in 6.4: the operating-system layer, hardware abstraction, memory governor and arenas, doorbell and threadpool, hot-set solver, quantizers, GGUF reader and writer, SHA-256, JSON reader and writer, tokenizer, chat-template renderer, kernels, decode loop, HTTP and SSE server, trainer | First line of every source file is the provenance line; copyright line "Copyright (c) 2026 John Lipka and the Pelias contributors" |
| **Allowed at run time (not Pelias code)** | The CUDA runtime and driver; NVML, loaded dynamically so the program starts without it (the Wave 0 probe program already does this); the C++ standard library; compiler intrinsics for AVX2; the operating system's own interfaces (POSIX, Win32, Winsock2); cuBLASLt, optional, dense bf16 only, until P-K1 shows whether the project's own GEMM reaches 0.85 of it | Listed in `THIRD_PARTY_NOTICES.md` with the version |
| **Open decision** | Dear ImGui (MIT) for the trainer's interface, the one third-party library in the draft: keep it for the interface only, write an own layer, or ship headless first. It is not part of the engine or the server in any case | Part 7.16, decision O9 |
| **Data, not code** | Unicode Character Database tables (general categories, composition data) used by the tokenizer, generated into source tables by a tool in `tools/` | The tables are Unicode data; the generator is original; the data version is recorded |
| **Forbidden in the product** | llama.cpp, ggml, libllama, `common`; any JSON library; any HTTP library; any regular-expression library; any Jinja library; Python at run time | Checked by BC1 to BC4 below |
| **Test-only and external** | llama.cpp commit 2ca15f540 (b11399), built unchanged, in `tests/oracle/`: supplies logits, KL, top-1 and kernel-parity references and nothing else; the Hugging Face and PyTorch oracle; `gguf-py`; Python 3 | Never linked, never started by the product, never required to run it. Kernel behaviour read from it is a specification only |
| **Offline development tools** | The Python tools of the prototype (`collect_trace.py`, `export_hot_set.py`, `kl_gate.py`, `speed.py`) | Allowed as development aids while they have no C++ twin; each is ported into the product when its output becomes an input of the product (hot-set export first: probe P-E14 proves the C++ writer is byte-identical). The product tree has no Python (trainer principle 10) |

**Boundary checks (DECISION, proposed; new here).** The boundary is enforced by tooling, not by intention:

| # | Check | When |
|---|---|---|
| BC1 | The module dependency graph is acyclic and single-sourced in `tools/check_module_deps.py`; the product modules never include `tests/` or `third_party/` | Every build |
| BC2 | The link map of every product binary contains no symbol from ggml or llama.cpp | Every build |
| BC3 | No product source file includes a header from outside the repository other than the standard library, the CUDA headers, the operating system's headers and the intrinsics headers | Every build |
| BC4 | Every source file's first line is a provenance line; a file whose line says "adapted from" names the licence and appears in `CITATIONS.md` | Every build |
| BC5 | The product binaries start, plan, load a model, decode and serve with the `tests/oracle/` folder absent | Release gate |
| BC6 | No operating-system header is included outside `os/` and the `host_alloc_*` backends | Every build |

## 6.4 The module map: layers 0, 1, 2A and 2B combined with the blueprint tree and the trainer tree (new text)

Names follow the trainer's tree (master 5.2) where a module exists there, the research update's layers (6.1) for the grouping, and the blueprint's tree (Part 5.1) for engine paths. "Shared" means one source in `libpelias_substrate` or `libpelias_core` linked by both applications; "Engine" and "Trainer" mean one application only. **First built** gives the engine phase of Part 7.2 and the trainer stage of Part 7.7; items shared by both are written by whichever stage needs them first (the trainer's T1, T1b and T2), then reused (DECISION, proposed).

**Layer 0: `libpelias_substrate` (domain-agnostic hardware abstraction)**

| Module | Files and roles | Blueprint path | Owner | Replaces in the prototype | First built |
|---|---|---|---|---|---|
| `os/file_map` | Read-only whole-file mapping with advice and unmapping, behind one interface; POSIX file mapping on WSL2, `CreateFileMapping` and `MapViewOfFile` on Windows (6.5, 6.8) | (new; the blueprint has a plain reader in `model/gguf.h`) | Shared | `gguf_min.py`'s Python header read; the prototype's plain-layout copy loader | T1, E3 |
| `os/net` | Listening and connected stream sockets, readiness wait, address and port handling, clean shutdown; POSIX `sys/socket.h` on WSL2, Winsock2 on Windows (6.5, 6.12) | (new) | Shared (engine first) | llama-server's HTTP layer, which the tier used | E10 |
| `os/threads` | Thread creation, core topology, affinity and priority, monotonic clock, signal and console-control handling | `cpu/threadpool.h` (OS part) | Shared | llama.cpp's thread pool, used by the tier through `-t` | E4 |
| `os/large_pages` | Optional huge-page and large-page requests for mapped and allocated memory; off unless P-E11 shows a gain | (new) | Shared | `.wslconfig` experiments (`transparent_hugepage`) | E11 |
| `hal/device`, `stream`, `event` | Device selection, stream and event wrappers, CUDA error checking | `gpu/*` glue | Shared | `lib/pl_cuda.h` (the thin GPU API layer) | T1, E2 |
| `hal/arena` | Bump arenas by lifetime (6.9) | `mem/vram_plan.h` | Shared | The tier's per-call buffers | T1, E2 |
| `hal/vram_governor` | Hard ceiling measured at start (D28); plan sum checked before the first allocation; NVML re-read every N tokens | `mem/vram_plan.h`, `mem/probe.h` | Shared | The `PELIAS_TIER_MAX_MIB` environment variable (7300 in the Windows handoff) | T1, E2, E11 |
| `hal/pinned_pool` and backends `host_alloc_wsl`, `host_alloc_win32` | One pool of pinned host memory from `cudaHostAlloc`; the pool logs the memory-lock limit it sees; the pageable fallback chain of master 15.2 | `mem/host_alloc.h`, `src/mem/host_alloc_{wsl,win32}.cpp` | Shared | `pin_modes` test; the prototype's pinned expert copy | T1, E2, E11 |
| `hal/sync_audit` | Debug-only check of stream ownership after each launch | - | Shared | - | T1 |
| `plan/` | `probe_constants`, `roofline_solver`, `memory_plan`: measured rates in, window sizes, slot counts, bank size and fetch schedule out; the plan card | `mem/vram_plan.h`, `mem/probe.h` | Shared | The author's hand-set launch flags (`-ngl`, `-ot`, bypass values) | T1, E11 |
| `bus/doorbell_handoff` | Shared-memory mailbox and flags between a resident GPU kernel and CPU workers; stream-sync fallback and timeout (6.6) | `gpu/handoff.cuh` and the CPU side | Engine (substrate for reuse) | `pelias_serve/handoff.cuh`, `miss_splitter.h` (never run in an engine) | E6 |
| `bus/cpu_threadpool` | Physical-core-pinned worker pool with AVX2 work items (6.7) | `cpu/threadpool.h` | Engine | llama.cpp's CPU expert path (about 27 GB/s effective, DERIVED) | E4 |
| `bus/stage_streamer` | Three-stream expert and boundary streaming for layer-major work | (trainer `stream/`) | Trainer | v0.2.1 `streamer` module | T10 |
| `bank/hot_set_solver`, `hot_set_io` | Water-fill of resident experts per layer from a trace, zero-miss variant, the `PELIAS_HOT_SET` writer and reader | `tools/export_hot_set.py`, `profiles/` | Shared | `export_hot_set.py` | E5, T12 (P-E14) |
| `quant/` | Block types and CPU and GPU quantizers and dequantizers for Q4_K (144 B), Q6_K (210 B), Q8_0 (34 B), q8_1 (36 B), Q8_K (292 B) (6.11) | `cpu/expert_q*_avx2.cpp`, `gpu/expert_q*.cu` | Shared | ggml's own code, which the tier used in place | T1b, T3, E4, E5 |

**Layer 1: `libpelias_core` (model input and output, spine kernels, comparators)**

| Module | Files and roles | Blueprint path | Owner | Replaces in the prototype | First built |
|---|---|---|---|---|---|
| `io/gguf_reader` | Memory-mapped reader: header, metadata, tensor table, typed views, layout check (6.8) | `model/gguf.h`, `tensor_map.h`, `expert_table.h` | Shared | `gguf_min.py`, `expert_layer_sizes.json` | T1, E3 |
| `io/gguf_writer`, `gguf_splice` | New-GGUF writer, byte-splice of frozen tensors, tensor-payload SHA-256 proof, expert-last layout with the `pelias.pad` entry and `layout_manifest.json` | (trainer `export/`) | Trainer (substrate for reuse) | llama.cpp's converter and `llama-quantize`, used by the prototype export | T1b, T12 |
| `io/sha256`, `io/json`, `io/safetensors` | Own hash, own JSON reader and writer, own safetensors header and shard reader | - | Shared | Python `json`, `hashlib`, `safetensors` | T1 |
| `tok/` | Unicode data tables, NFC normalizer, hand-written pre-tokenizer matcher, byte-level BPE, added and special tokens, chat-template parser and evaluator (6.13) | `server/tokenizer.h`, `server/chat_template.h` | Shared | llama.cpp's tokenizer and Jinja renderer, used inside llama-server | T2, E10 |
| `kern/` (spine family) | RMSNorm, gated RMSNorm, per-head L2 norm, attention with IMRoPE and GQA (10 trunk layers plus the MTP block), DeltaNet (30 layers) with the sequential fp32 recurrence for decode and the chunked path for prefill, router with fp32 top-8, lm_head; every kernel has a slow reference first (6.10) | `gpu/spine/attention.cu`, `deltanet.cu`, `norm.cu`, `router.cu`, `lm_head.cu` | Shared (levels L2 of sharing) | llama.cpp's kernels, used in place | T4 to T6, E7 |
| `kern/` (expert and matrix family) | Hot-branch Q4_K and Q6_K expert kernels, the zero-copy cold-read kernel, trainer GEMM and LoRA kernels. The engine does not need a LoRA kernel when it serves a merged core; the trainer's LoRA epilogue lives here so that an unmerged adapter could be served later (not scheduled) | `gpu/expert_q4k.cu`, `expert_q6k.cu`, `zerocopy.cu`, `bank.cu` | Engine (hot and cold branch) and Trainer (GEMM, LoRA); L3 not shared | `zc_mmvq_q4k_q6k.cuh` and the tier's ggml-cuda path | E4, E5, T3 |
| `model/` | `ModelSpec` from GGUF metadata, tensor map by name, expert size table, architecture plug-in point (E13); the trainer's frozen store and trainable core | `model/arch.h`, `arch_moe.cpp` | Shared (spec); Trainer (stores) | `cold_weights_view.{h,cpp}`, `cold_pool.h` | E3, T7 |
| `verify/` | The project's own KL, top-1 and dump comparators, canaries, gate runner; reads files the external oracle writes | `tools/kl_gate.py` | Shared | `kl_gate.py` | E3, T12 |

**Layer 2A: `pelias-engine` (inference and server)**

| Module | Files and roles | Blueprint path | Replaces in the prototype | First built |
|---|---|---|---|---|
| `engine/` | Decode loop with CUDA graphs for the static spine, layer pipelining, prefill (chunked, bypass rule), q8_0 KV cache, recurrent-state store and its prefix cache, sampler, hit counters and per-token timers | `engine/layer.h`, `decode_loop.h`, `prefill.h`, `kvcache.h`, `state.h`, `sampler.h` | The tier build inside llama.cpp; `e0_timeline.patch`, `hit_counter.patch` | E6 to E9 |
| `cpu/` | AVX2 Q4_K gate and up and Q6_K down kernels; the 8-expert path | `src/cpu/expert_q4k_avx2.cpp`, `expert_q6k_avx2.cpp` | llama.cpp's CPU expert path | E4 |
| `gpu/` | Bank loader and slot table, hot branch, zero-copy cold branch, merge | `src/gpu/bank.cu`, `zerocopy.cu`, `handoff.cu` | `llama-pelias-tier.{h,cpp}` and the CUDA patches | E5 to E7 |
| `server/` | Raw-socket HTTP and SSE server, OpenAI-compatible endpoints, tool-call parser, reasoning splitter, request log (6.12 to 6.14) | `server/http.h`, `tools.h`, `api_openai.h` | llama-server | E10 |
| `cli/` | Sub-commands `serve`, `bench`, `trace`, `probe`, `repack` | `src/main.cpp` | The numbered shell scripts and the launcher | E8, E11 |

**Layer 2B: `pelias-trainer`**

The trainer's own modules are as in master 5.2 and the framework list that follows it (`config`, `ingest`, `data`, `sched`, `stream`, `train`, `export`, `verify`, `cli`, and `project`, `startup`, `intake`, `help`, `warn`, `flow`, `serve`, `ui`). Two connect to this part: the trainer's `serve` module is a stub that is to print the server command for a finished GGUF, and under this part it prints or starts the proprietary server (DECISION, proposed; held until unification); the trainer's `tok` and `io/gguf_reader` stubs are the shared implementations of 6.8 and 6.13.

## 6.5 The operating-system layer (WSL2 and Windows-native; native Linux held)

One code base builds two programs: a WSL2 build (gcc or clang with nvcc 13.0.88) and a Windows-native build (MSVC with nvcc 13.4.92) (author ruling 2026-10-24 06:21 [AUTHOR-STATED]; blueprint R1). Every operating-system difference lives in `os/` and in the `host_alloc_*` backends (check BC6). The table lists each service the product needs and the call family each build uses. Names of calls are given because the author asked for the Winsock2, POSIX and file-mapping boundaries to be named; the exact argument choices are decided when each module is written.

| Service | WSL2 build (POSIX) | Windows-native build (Win32) | Notes |
|---|---|---|---|
| Open and size a file | `open`, `fstat` | `CreateFileW`, `GetFileSizeEx` | Windows file names are UTF-16 internally; the product keeps UTF-8 and converts at this layer |
| Map a file read-only | `mmap` with read protection and a private mapping; `munmap` | `CreateFileMappingW` with read-only page protection, `MapViewOfFile` with read access; `UnmapViewOfFile`, `CloseHandle` | One interface `os/file_map`; the mapping is never written |
| Advise on access | `madvise` (sequential, will-need, do-not-need) | `PrefetchVirtualMemory` for will-need; no direct equivalent of do-not-need | Advice is optional and never changes results |
| Huge or large pages | Transparent huge pages through `madvise`; the `.wslconfig` setting is read, not changed | Large pages need a privilege the user may not have; off by default | Gain unmeasured (P-E11); the engine measured under 1 % from transparent huge pages |
| Pinned host memory | `cudaHostAlloc` | `cudaHostAlloc` | Not an operating-system call; 32 GiB held on both platforms (P-H7). `cudaHostRegister` is avoided where pointer identity matters |
| Start the socket library | none | `WSAStartup`, `WSACleanup` | Called once by `os/net` |
| Create, bind, listen, accept | `socket`, `bind`, `listen`, `accept` | `socket`, `bind`, `listen`, `accept` (Winsock2) | Bind address default is the loopback address, as the clients use today; port is a parameter |
| Exclusive port | A refused second bind is the normal behaviour | `SO_EXCLUSIVEADDRUSE`, so that a second program cannot take a port that is already in use | The clients refuse to start their own server if one already answers on the port |
| Readiness wait | `poll` | `WSAPoll` | A single wait call keeps the server small; effort of other models (completion ports) is not needed at one decode slot |
| Send and receive | `recv`, `send` with the no-signal flag so a closed peer does not raise a signal | `recv`, `send` | A short send is a normal result and the loop continues |
| Close | `shutdown`, `close` | `shutdown`, `closesocket` | |
| Threads and affinity | `pthread_create`, `pthread_setaffinity_np` | `CreateThread` (or the standard thread type), `SetThreadAffinityMask` or `SetThreadGroupAffinity` | Wrapped by `os/threads` |
| Core topology | Read `/sys/devices/system/cpu` sibling lists | `GetLogicalProcessorInformationEx` | Whether WSL2 exposes the hardware's sibling pairs is **UNVERIFIED**; the probe reads it (E-A1) |
| Monotonic clock | `clock_gettime` with the monotonic clock | `QueryPerformanceCounter` | GPU timing uses CUDA events; the host clock is for logs |
| Stop signals | SIGINT and SIGTERM | A console control handler | **First signal: finish or abort the current token, write the statistics file and the request log, release the port, exit. Second signal: exit at once.** llama-server exits at once on a second signal (RUNBOOK, 2026-10-22 15:5x, READ-SRC server.cpp lines 26 to 33); the first-signal behaviour is a DECISION (proposed) |

**What the layer must never do.** It adds no feature. It does not hide a difference that matters to a result: the WSL2 build and the Windows-native build are measured separately (Part 3), because the Windows-native decode of the tier build was 25 % (short) and 28 % (12k) faster on the same cell and the cause is not isolated (P-H11).

*Operating-system call families are named from general knowledge of the POSIX, Win32 and Winsock2 interfaces [PUBLISHED: vendor and standard documentation, not re-fetched for this document]. Each call's exact behaviour on WSL2 and on Windows-native is checked when its module is written, with a test, and nothing in this table is a measurement.*

## 6.6 The doorbell coordinator (shared-memory handoff between GPU and CPU)

**Purpose.** For each of the 40 MoE layers the router chooses 8 experts; the slot table splits them into **hot** (in the VRAM bank) and **cold** (in the pinned host store). The hot branch runs on the GPU and the cold branch on the CPU (and, as a gated extra, on the GPU reading host memory over PCIe). The two branches must run in parallel and join with the least delay. The coordinator is the mechanism that does the join.

**What is known (all MEASURED on WSL2 unless stated; sources in Part 3).**

| Quantity | Value | Where it matters |
|---|---|---|
| Stream synchronization per layer | 94 µs, which is 3.76 ms per token over 40 layers | The slow path. The tier build, which hands off by stream synchronization, showed a timer-free handoff and copy cost of 4.6 to 5.0 ms or more per token [DERIVED] |
| Resident-kernel flag ping-pong with a 4 KB read | 13.3 µs median, 18.7 µs p99 (0.53 to 0.75 ms per token) | The proposed path. A microbenchmark, not a running engine |
| Flag visibility alone | 3.4 µs one way | The floor |
| Standalone handoff test with a stand-in CPU worker (2,048 multiply-adds) | CPU-side round trip 5.05 µs median at block size 256, p99 6.17 µs; GPU-side median 6.39 µs, p99 7.84 µs; block sizes 32, 64, 128, 256, 512, 1024 gave 10.93, 7.17, 5.40, 5.05, 5.06, 5.15 µs | Block size 256 was fastest. No real CPU workload or other kernels were in the stream |
| Modelled handoff in the ceiling | 0.25 ms per token (about 6.25 µs per layer) | An assumption |

**Specification (DECISION, proposed; every parameter is a configuration value, none is hard-coded).**

1. **Mailboxes.** One slot per layer (parameter `handoff.depth`, default 1 per layer, raised if layer pipelining needs two in flight) in the pinned host pool. A slot holds: the token and layer sequence number, the count and ids of the cold experts, their routing weights, and the location of the layer's input activation; a second slot of the same layer holds the reply: the echoed sequence number, a status code and the location of the partial result. Each flag sits alone in its own cache line so that the two sides never share a line with data they are writing (false sharing).
2. **Order of operations for one layer.** (a) The GPU finishes the router and the slot-table lookup and writes the payload; (b) it publishes the sequence number with release ordering; (c) the hot branch starts on the GPU stream at once; (d) CPU workers, spinning on the sequence number with acquire ordering, see it, read the payload, and compute the cold experts (6.7); (e) the last worker to finish writes the partial result and the reply flag; (f) the resident GPU kernel, polling the reply flag with a bounded back-off, sees it and merges the partial result with the hot result; (g) the layer's output feeds the next layer. Whether acquire and release ordering are enough, or sequentially consistent ordering is needed, is the A/B test **E-A2** (UNVERIFIED).
3. **Resident kernel life.** A resident kernel lives for **one token**, or one captured graph launch, and never for the session. Reason: on Windows-native the card also drives the display, and a long-running kernel can be stopped by the display driver's timeout. Probe **P-H13** ran on 2026-10-24 on both platforms: kernels of 2, 5 and 30 seconds, a queued set of 10 seconds and a graph of 10 seconds all completed with no errors and no display-driver events (MEASURED, Part 3.3); heavier kernels and other display setups are unmeasured. A token takes 18 to 26 ms at the recorded speeds (Part 3.4), far below even the default 2-second timeout.
4. **Fallback.** `handoff.mode` is `doorbell` or `stream_sync`. The start-up probe (E11) chooses `stream_sync` when the doorbell is not supported or not faster. At run time, if the reply flag is not posted within `handoff.timeout_us`, the GPU kernel records an error status and exits, the host completes that layer by the synchronized path, and the event is counted and logged. The timeout's value is **unmeasured**: it is set from the p99 of P-E2's per-layer timers plus a margin.
5. **Cancellation.** A client disconnect or a stop signal sets an abort word in the mailbox area; the resident kernel and the workers check it at each layer and drain.
6. **What the gate says.** E6 requires a layer latency within the E0 model, output within the oracle noise floor, and a handoff p99 no worse than 18.7 µs (Part 7.2). In the draft that gate is run on WSL2 and on Windows-native.

**Why this is the first thing to prove.** In the tier build the timeline showed no overlap of the CPU and GPU work and a handoff cost of 4.6 to 5.0 ms or more per token, both DERIVED (Part 3.5 rows 2 and 3). If a real engine with a real CPU worker cannot bring that down, the ceiling of 88.2 to 92.0 tok/s stays a model. This is why E6 is a gated phase and why the coordinator is specified before the server.

## 6.7 The CPU threadpool, the thread model and the decode loop

**Threads (DECISION, proposed; counts are parameters).**

| Thread | Count | Pinned | Role |
|---|---|---|---|
| Control | 1 | No | Start-up, probe, plan, shutdown, signal handling (6.5) |
| Server I/O | 1 acceptor and a bounded number of connection threads (`server.max_connections`) | No; lowest priority | Accept, read requests, write SSE chunks (6.12) |
| Decode | 1 per slot (`engine.slots`, default 1) | Yes, to its own physical core | Owns the CUDA streams and the decode loop; the only thread that touches the GPU |
| Cold-expert workers | `cpu.threads` | Yes, each to a distinct physical core, not the decode thread's core | AVX2 expert kernels; wait on the doorbell |
| Statistics writer | 0 or 1 | No | Per-token timers and counters to the statistics file |

**Core count.** The machine has 8 physical cores and 16 threads (AVX2, no AVX-512). Whether one thread per physical core beats 16 threads is the A/B test **E-A1** in E4, with its own go or no-go (UNVERIFIED hypothesis; Zen 3 splits the load queue between the two threads of a core). The ledger shows that on stock llama.cpp 5 to 7 threads beat 10 [MEASURED]; the prototype's launcher defaulted to 7 threads (roadmap, UNFINISHED section E) and that was not adopted without a test. The first candidate for E4 is one core for the decode thread and seven for workers; E4 decides. **Pinning on WSL2 is itself UNVERIFIED** (6.5): the probe reads the topology and the pool refuses to pin onto sibling threads of one core when it can tell.

**Cold-expert work item (order of operations).** (1) The worker takes the layer's cold list from the mailbox. (2) The activation vector is quantized once to the 8-bit CPU activation format (Q8_K, 292 B per 256 elements) by the first worker or by the GPU side (decided in E4). (3) Each worker takes a contiguous slice of the rows of the gate and up matrices of each cold expert, in a fixed assignment so that repeated runs give identical results on one machine; it computes the dot products with the AVX2 kernels of `cpu/` (Q4_K for gate and up, Q6_K for down in the layers where the down matrix is Q6_K); (4) the activation function and the down projection follow; (5) partial sums are reduced in a fixed order; (6) the reply flag is raised. **Prefetch** hints (`PREFETCHNTA`) on cold reads are the A/B test **E-A3** (UNVERIFIED). **Reads** come from the pinned, aligned, plain-layout copy of the experts (18.16 GiB for the 40 trunk layers; 18.65 GiB with `blk.40`), which the CPU reads at 43.8 to 45.9 GB/s alone and at 26.6 GB/s while the GPU reads at the same time [MEASURED].

**The decode loop, one token (order of operations).**

1. The sampler (6.12) chooses the next token from the logits of the previous step; the embedding row is gathered on the GPU.
2. For layer 0 to 39: the spine part (norms, then attention for the 10 full-attention layers or DeltaNet for the 30 others, residual); the router (fp32, top-8, 8-way softmax); the slot-table lookup; the mailbox post (6.6); the hot branch on the GPU and the cold branch on the workers in parallel; the join; the merge and residual.
3. Final norm, `lm_head`, logits.
4. The static part of each layer's spine is captured once as a CUDA graph and replayed; the hot and cold lists are passed through device memory so the graph does not change. The gain from graphs is gate E8 (the update's figure of 2.75 ms against 8.04 ms is not adopted; Part 1.1a row 7).
5. The token goes to the server thread through a bounded queue; the server formats and sends the SSE chunk (6.12). The decode thread never waits for a client.

**The MTP block.** The served file has 41 blocks; `blk.40` is the multi-token-prediction block with 498 MiB of Q6_K-down experts. The engine treats it as a real 41st block (master D12). It is loaded only when speculative decoding is enabled (a gated extra, E12), because it costs VRAM or host memory. Whether the unused block in stock llama-server explains why the 10.7 GiB ceiling was not reached is **UNVERIFIED**.

**Prefill.** Prompt reading is chunked (parameter `prefill.micro_batch`, an engine setting by the ruling of 2026-10-24 07:39) and uses a bypass rule: the dual hot-and-cold path is used only when a chunk is smaller than `prefill.bypass_tokens` (32 in the Reference Cell A, 9 in the tier's production setting); larger chunks take the large-batch path. The micro-batch moves the engine's peak VRAM (Part 5.1 table: 0.26 to 0.30 GiB of headroom at the default and 0.01 to 0.03 GiB at 2048, whole-card reading, 10.7 GiB ceiling). The gate for E9 is a prompt-reading rate of at least 1,670 tok/s at 12k, aimed at 2,238 [MEASURED references].

## 6.8 The memory-mapped GGUF reader (model file layer)

**Why it is its own code.** The reader is the first thing any program in this project does with the model, the trainer needs it (T1) and the engine needs it (E3), and its layout rules are the interface between them (the engine owns the packed layout; D25). The prototype read the header in Python (`gguf_min.py`) and loaded the expert tensors with a loader built into the tier; neither is available to a ground-up engine.

**Files and roles.** `io/gguf_reader` parses and exposes typed read-only views; `io/layout` checks alignment, the `pelias.pad` entry and the layout manifest; `io/sha256` and `io/json` are the hash and the manifest reader; `model/arch` builds `ModelSpec` from the metadata; `model/tensor_map` indexes tensors by name; `model/expert_table` holds per-layer expert sizes; `os/file_map` does the mapping (6.5).

**Order of operations when a model is opened (DECISION, proposed).**

1. **Resolve the path and check the medium.** A GGUF read through the Windows file system from WSL2 reads at about 0.52 GB/s [MEASURED]; the full 18.16 GiB of experts loaded in 37.7 s at that rate. The plan card says so and advises a copy on the WSL2 disk or a read once into the pinned store; whether the WSL2 disk is faster is P-M4 (unmeasured). The reader itself does not refuse a slow medium.
2. **Open read-only, take the size.** A file smaller than a GGUF header is refused.
3. **Map the whole file read-only** (POSIX file mapping on WSL2; `CreateFileMappingW` and `MapViewOfFile` on Windows). No byte of the mapping is ever written.
4. **Parse the header.** Magic, version (3 is the served file's; other versions are refused with the version in the message), tensor count (753 in the served file), metadata count. The file is little-endian; the build refuses a big-endian host.
5. **Parse the metadata**, with a bounds check on every length read. Unknown value types, string or array lengths that run past the file, and a count that cannot fit are refused with the byte offset. Keys the product needs: the architecture, the alignment (`general.alignment` is absent in the served file, so 32), the block count (41 in the served file: 40 trunk layers and the MTP block), expert count and used count, the tokenizer entries (`tokenizer.ggml.*`, model type `gpt2`), `tokenizer.chat_template`, and the optional `pelias.pad` string, whose value is ignored.
6. **Parse the tensor table.** For each tensor: name, dimensions, type, offset. The byte size is computed from the type's block geometry (Q4_K 144 B per 256, Q6_K 210 B per 256, and so on, 6.11) with overflow-checked 64-bit arithmetic; an offset that is not a multiple of the alignment, a tensor that extends beyond the file, and two tensors that overlap are refused.
7. **Compute the data start** as the end of the table rounded up to the alignment (byte 10,991,104 in the served file, READ-SRC), and **build the tensor map** by name. Views are pointers into the mapping; nothing is copied.
8. **Layout check.** For every expert tensor, record its offset modulo 16 B, 4 KiB and 2 MiB. In the served file all 123 expert tensors are multiples of 16 B, 9 are 4 KiB-aligned, none is 2 MiB-aligned (P-E9, READ-SRC). If a layout manifest is present, compare every field and the file hash. If `pelias.pad` is present, the file was written by `--layout expert-last` (non-expert tensors first, then the 123 expert tensors layer-major, a padding entry of less than 2 MiB so that the first expert tensor starts on a 2 MiB boundary; every expert tensor is a multiple of 2 MiB, so all of them start on 2 MiB boundaries): the reader reports whether the expert tensors are in fact aligned and fails only if `reader.require_expert_alignment` is set. Whether alignment matters for speed is P-E11.
9. **Optional full-file hash** (`reader.verify_sha256`, default off for serving and on for export proofs). The hash time depends on the read rate and the CPU's SHA-256 speed, both unmeasured.
10. **Build `ModelSpec`**: the layer pattern (the layers whose index plus one is a multiple of 4 are attention; the other 30 are DeltaNet), the expert sizes by layer (432 MiB for the 20 layers with Q4_K down and 498 MiB for the 21 blocks with Q6_K down), the tensor names of Appendix A of the master.
11. **Load plan (hand-over to the memory plan, 6.9).** (a) Spine tensors are uploaded into the model arena in VRAM. (b) The expert tensors are copied once into the pinned pool in a plain, aligned layout (the prototype's loader did this and was verified byte for byte on the real 40-layer file). (c) The hot-set file selects the experts uploaded into the bank. (d) After the copy, the expert region of the mapping is released: with the expert-last layout the experts are one contiguous range at the end of the file, so one unmapping releases it while the header and the small tensors stay mapped (on POSIX by unmapping the range; on Windows by mapping the file in two views). The gain is **unmeasured**.
12. **Failure handling.** Every failure is a status code with the byte offset or tensor name; no exception crosses the kernel-launch layer; no partial model state is left behind.

**Hostile or corrupt files.** The reader treats the file as untrusted: every length and offset is checked before use and every multiplication is done in a wide unsigned type with an overflow check. This is a rule taken from a defect the prototype found by sanitizer: a signed 32-bit id of minus one was widened after a conditional and became 4,294,967,295, which the sanitizer saw as, among others, a read 8.8 TB past a 2 MiB buffer (Part 3.1, section 8.4; fixed by widening each side first).

**Trainer-exported files.** A `merged-core.gguf` with its `layout_manifest.json` must parse identically to the served file (E3 gate), and K3 must hold on it. The writer side (`gguf_writer`, `gguf_splice`) belongs to the trainer's export stage and is shared as a library.

## 6.9 The VRAM governor, arenas and the memory plan (the zero-malloc fixed arena)

**Rules, taken from the safety rails (trainer master 15.1 and 15.2; DECISION, binding on the engine by the contract).**

1. Every device allocation goes through an arena registered with the governor at plan time. The plan's sum is checked against the ceiling before the first `cudaMalloc`; a plan above it is **refused, never trimmed silently**.
2. The ceiling is not typed: it is the smaller of CUDA free and NVML free after the context exists, minus a margin (D28). 10.7 GiB is this machine's value (10.83 GiB free minus 0.13) until the disturbance test sets the margin. Until the engine re-measures its own peaks (P-E4), both the process-only and the whole-card readings are applied and the stricter governs.
3. The token loop contains **no** `cudaMalloc`, no `cudaFree`, no `new` and no `malloc` (P-M5). A debug allocator hook turns any such call in the loop into a fatal error with the call site.
4. At run time the governor reads NVML used memory every `governor.nvml_check_every_tokens` tokens (value unmeasured); exceeding the ceiling aborts the request cleanly and reports it.
5. Pinned host memory is one pool from `cudaHostAlloc` (the form that held 32 GiB on WSL2 and on Windows-native); the pool logs the memory-lock limit it sees. If it runs short, the fallback chain of master 15.2 applies in order and each fallback is logged.

**Arenas by lifetime.**

| Arena | Lifetime | Holds | Sized from |
|---|---|---|---|
| `model` | Process | Spine weights, embedding, `lm_head`, routers, the slot table | The model header (6.8) |
| `bank` | Process | The hot experts, with one all-zero slot per layer (72.6 MiB over 40 layers) and per-layer capacities of 84 to 161 experts at 7.5 GiB | The hot-set file and `bank.size_gib` |
| `session` | One slot's conversation | The q8_0 KV cache for the 10 attention layers and the recurrent state of the 30 DeltaNet layers | `kv.context_tokens`, `kv.type` |
| `prefill` | One prompt | Chunk activations, compute buffers | `prefill.micro_batch` |
| `token` | One token | Per-token scratch; reset after each token | The decode graph |
| `host_pinned` | Process | The expert store, mailboxes, staging for uploads | Expert bytes and `handoff.depth` |

**The no-op slot rule (taken from the carried patches).** A hot branch must be able to receive "this expert is not here" without a negative index. A miss, or a padded row, maps to the layer's all-zero slot, so the branch computes exact zeros; the hot branch also receives a count of valid rows. No kernel is passed a negative or unsigned-wrapped index. This replaces the patches that filled the id map with minus one and skipped negative ids in the quantizer, and the patch that counted each expert once per row (Part 4.8).

**VRAM budget at the ceiling (Part 5.1 table, quoted, whole-card reading).** Bank 7.5 GiB (4,170 experts including the zero slots). Whole-card peak at the default micro-batch 10.40 to 10.44 GiB, which leaves 0.26 to 0.30 GiB of headroom to 10.7 GiB; at -ub 2048, 10.67 to 10.69 GiB, which leaves 0.01 to 0.03 GiB; an 8.0 GiB bank peaks at 11.0 to 11.19 GiB and is over the ceiling. The whole-card figure includes the Windows desktop's share (about 1.0 GiB), which is why the process-only reading gives about 1.2 and 0.9 GiB of room. Whether the bank may be a non-multiple of 0.5 GiB (about 7.8 GiB at the default micro-batch if the slope is 1:1 [ESTIMATE]) is open decision O1 (Part 7.16).

**Order of operations at start (DECISION, proposed).**

1. Create the CUDA context. 2. Read CUDA free and NVML free; take the smaller; subtract the margin; that is the ceiling. 3. Open the model (6.8 steps 1 to 10). 4. Run the quick start-up probe (free VRAM, pinned PCIe rate, whether copies need a flush, free host RAM; its running time is unmeasured). 5. Build the plan card: bank size, micro-batch, context, KV type, thread counts, handoff mode; refuse if the sum is above the ceiling. 6. Allocate the arenas and the pinned pool. 7. Load (6.8 step 11). 8. Capture the graphs and run a warm-up token. 9. **Open the port last** (6.12), so that a client never meets a half-ready server. 10. Serve.

**Parameters of the memory plan.**

| Parameter | Meaning | Default | Source and label |
|---|---|---|---|
| `vram.margin_gib` | Subtracted from the measured free memory | 0.13 on this machine | D28; measured by the disturbance test (open) |
| `vram.ceiling_override_gib` | A typed ceiling for Expert mode | none | U11; shown as USER-FORCED |
| `bank.size_gib` | Hot-bank size | 7.5 | Open decision O1; the bank is 4,170 experts |
| `bank.hot_set_file` | Per-layer resident experts | the profile for the model key | `PELIAS_HOT_SET` format; replaces the environment variable of the same name |
| `prefill.micro_batch` | Prefill chunk size | open | Engine setting (ruling 2026-10-24 07:39) |
| `prefill.bypass_tokens` | Chunk size below which the dual path runs | 32 in Cell A; 9 in the tier's production setting | Part 3.4; to be re-derived in E9 |
| `kv.type` | KV cache type | q8_0 | Launcher setting today |
| `kv.context_tokens` | Context length | 32,768 (launcher today); up to 131,072 is the E9 goal | Launcher; blueprint E9 |
| `host.pinned_pool_gib` | Pinned pool size | the expert bytes (18.16 or 18.65 GiB) plus staging | 32 GiB held on both platforms (P-H7); the WSL2 cap is 48 GiB |
| `governor.nvml_check_every_tokens` | NVML re-read period | unmeasured | Master 15.1 |

## 6.10 The shared CUDA kernels: spine, expert and LoRA families

**Rules (trainer master 3.1 principle 5, 5.1; DECISION, binding on the engine).** (1) Every kernel has a slow, obviously-correct reference (CPU or fp64) first; the fast version is admitted only when it matches the reference within the rung-1 tolerance, and each fast variant sits behind a flag. (2) `kern/` depends only on `hal/` and on plain structs of pointers and shapes; it never includes `model/`. Only the scheduler (`sched/` in the trainer, `engine/` in the engine) composes kernels into layers. (3) Target `sm_89` machine code plus `compute_89` PTX; tile constants in one header per kernel, derived from probe P-H1. (4) Inline PTX only for the tensor-core matrix instruction, the matrix load and the asynchronous copy. (5) Debug builds keep device assertions; release builds use status codes.

**Families and what is shared (levels of sharing, reconciliation A6).**

| Level | Content | Shared by engine and trainer? |
|---|---|---|
| L0 | Block types and constants | Yes, one header |
| L1 | Quantizers and dequantizers (6.11) | Yes, one implementation, tested against the oracle |
| L2 | The non-matrix spine: RMSNorm with the +1 folded, gated RMSNorm (`ssm_norm`), per-head L2 norm (eps 1e-6), IMRoPE with sections [11, 11, 10, 0], attention with grouped queries (the 10 trunk full-attention layers and the MTP block), the DeltaNet block (4-tap causal convolution with SiLU, the gates), the router (fp32, top-8, 8-way softmax) | Yes, written once. The DeltaNet recurrence has two forms: the **sequential fp32 recurrence** (decode, and the trainer's rung-2 oracle) and the **chunked form** with chunk 64 (engine prefill and trainer forward) |
| L3 | Weight matrix products | **No.** Training dequantizes to the bf16 tensor-core path; serving uses the int8 path with `q8_1` activations. K1 and K3 stay gates; the bimodal-faithful forward (Mode B, D24) is how the two are brought together |

**Engine-only kernels.**

| Kernel | Role | Known facts | Probes |
|---|---|---|---|
| Hot-branch expert (Q4_K gate and up, Q6_K down) | The experts that sit in the bank | Matches the oracle on the bank's experts is the E5 gate | P-E5 (supported weight-type matrix: Q4_K, Q6_K, F32; Q8_0 optional), P-E13 (accumulators: 32-bit integer dot plus fp32 scale) |
| Zero-copy cold read | The GPU reads cold experts from pinned host memory over PCIe | 26.2 GB/s for the checksum kernel; 9.59 GB/s for the narrow-load compute kernel. Q6_K rows are 420 B (not a multiple of 16 B), so a 16-byte load needs stream-aligned staging in shared memory, or 432-byte padded rows (+120 MiB over 20 layers) written by the engine's `repack` | P-E10 (staging), P-E9 (alignment, done once), P-E11 |
| CPU expert (Q4_K, Q6_K, AVX2) | The cold branch on the workers (6.7) | The throughput target is at or above llama.cpp's CPU expert path (about 27 GB/s effective, DERIVED) | E4 benchmark, E-A3 |
| Slot-table lookup and merge | Maps router output to bank slots, zero-slot rule (6.9) | Part of the E6 layer | E6 gate |

**LoRA kernels.** The trainer's LoRA epilogue (the low-rank update applied to the weight product) lives in `kern/` with the matrix family. The engine serves a **merged** core and needs no LoRA kernel (D25: the exporter writes the merged-core GGUF; the engine owns the packed layout). The kernel is available to the engine only if an unmerged adapter is ever served; that is not scheduled and is not a decision yet.

**How the fast kernels are built (order, from the blueprint risk row).** Start with the engine's own slow reference kernels behind a clean interface (the sequential fp32 DeltaNet recurrence, the blocked attention kernel as in the trainer's `kern/attn_ref_blocked`), then replace them one at a time with the fast ones. llama.cpp's kernels are a read-only specification and a test oracle, never a dependency.

**Lessons carried from the prototype's defects (READ-SRC, Part 3.1 section 8.4).** (1) A "this slot does nothing" value must not be a negative integer that a mixed signed and unsigned expression can widen into a huge index: use the zero slot and a valid-row count (6.9). (2) Index arithmetic in kernels is explicit-width. (3) A kernel is tested with unused rows present, at chunk sizes of 9 and more tokens, which is where the prototype's crash appeared.

## 6.11 The quantizers and block formats (own code, ggml-equal bytes)

**What must be equal to ggml's.** The block layouts and the quantizer arithmetic, because the served file is made of ggml blocks and the engine reads and the trainer writes them. This is an interoperability requirement; the code is original and each file's provenance line says it was written after reading the reference (D37).

| Type | Bytes per block | Elements per block | Where used | Tested by |
|---|---|---|---|---|
| Q4_K | 144 | 256 | Expert and spine weights in the served file (rows of 1,152 B) | P-E6 against the oracle |
| Q6_K | 210 | 256 | Down matrices of 20 trunk layers and `blk.40`; rows of 420 B | P-E6; P-E9 geometry |
| Q8_0 | 34 | 32 | Optional weights and kernels; not needed by default (ruling D32) | P-E5 |
| q8_1 | 36 | 32 | GPU activation quantization for the integer matrix path | P-E6 |
| Q8_K | 292 | 256 | CPU activation quantization for the AVX2 dot | P-E6, P-E13 |

**Exactness criterion (DECISION, proposed).** A quantizer is accepted when its output is byte-identical to the oracle's on a battery of tensors that includes real tensors of the served file and edge cases (zeros, constant blocks, extreme values, denormals); a dequantizer when its output is bit-identical in fp32. The test is P-E6. Whether the CPU and GPU activation paths, which differ by design (Q8_K on the CPU, q8_1 on the GPU), disagree by an amount that matters is the CPU and GPU accumulator audit P-E13 and, at model level, the K3 gate.

**Order of operations when writing them.** (1) Block types and sizes (header only); (2) dequantize references on the CPU; (3) quantize on the CPU (needed by the trainer's Stage A and Mode Q requantizer and by tests); (4) the AVX2 dot kernels; (5) the GPU dequantize and the GPU activation quantizer; (6) the GPU integer matrix path.

## 6.12 The server: raw-socket HTTP/1.1 and Server-Sent Events

**What the server is for.** It is the surface through which the author's clients reach the engine: the Visual Studio 2026 Copilot "bring your own model" provider (a preview feature, Agent mode only, to be tried as an OpenAI provider at `http://localhost:8012/v1`; **untested** when last recorded on 2026-10-21, Part 8.7), the author's gauntlet and speed tools, a web-UI client, odysseus-ai and Ollama-style clients (trainer master 3.3, U8). The launcher parameters those clients rely on today are in 6.14.

**Boundaries.** The server contains its own socket layer (`os/net`, 6.5), its own HTTP/1.1 parser and writer, its own JSON reader and writer (`io/json`), its own chat-template renderer and tokenizer (6.13), its own tool-call parser and its own sampler. It uses no HTTP library, no JSON library and no regular-expression library. It runs in the engine process; there is no second process and no scripting language.

**Routes (DECISION, proposed; none is final until the catalogue of 6.14 and the clients' request logs say so).**

| Route | Status | Purpose and basis |
|---|---|---|
| `POST /v1/chat/completions`, streaming and non-streaming | **Needed** | The chat endpoint every client above uses; the Copilot provider is configured against `/v1` |
| `GET /v1/models` | Proposed | OpenAI-style clients list models before use; the model alias is a parameter (`model.alias`). Whether each client calls it is **UNVERIFIED** |
| A readiness route | Unknown | The Windows client starts only "after the server says ready"; whether it polls a route or reads the log is **not recorded** |
| Completion, tokenize, detokenize, embeddings, properties, slots, metrics and the rest of llama-server's 36 distinct POST routes | Unknown | Built only if a captured client request shows one. The census counted route strings only (Part 6.14) |

**Transport (parameters in brackets).** HTTP/1.1 over a stream socket; default bind address the loopback address and port 8012 (`server.bind_address`, `server.port`); a port already in use is a refusal at start with the owner's reason, never a silent second bind. A bounded number of connections (`server.max_connections`). Persistent connections. A request parser that is a state machine over a bounded buffer: request line, headers (names compared case-insensitively; at most `server.max_header_bytes`), a body with `Content-Length` (at most `server.max_body_bytes`) or with chunked transfer coding, and handling of the "expect 100-continue" header that some clients send. Anything else is answered with the matching error status (bad request, length required, payload too large, header fields too large, not implemented) and the connection is closed. Read and idle timeouts (`server.read_timeout_s`, `server.idle_timeout_s`) end slow or silent clients. The request-line and header limits, the body limit and the timeouts have **no measured values**; defaults are chosen when the module is written and are tested with oversized, truncated and slow inputs.

**Streaming (SSE).** A streaming response is sent with the event-stream content type and no content length; each event is a line beginning with the data field holding one JSON object, followed by a blank line, and the stream ends with the done marker line, as the OpenAI chat-completions streaming convention has it [PUBLISHED convention; the exact bytes every client needs are **UNVERIFIED** until captured]. Every chunk is flushed at once. A client that disconnects is noticed on the next send; the request is aborted, the slot released and the mailbox abort word set (6.6). Heartbeat comment lines during long prefill are optional (`server.sse_heartbeat_s`, default off) and are added only if a client needs them.

**Order of operations for one request (DECISION, proposed).**

1. **Accept and parse** (above). 2. **Validate** the JSON: roles, messages, tools, sampling fields, stream flag. **Unknown fields are ignored and counted**, so that the request log shows which fields real clients send (this is how the catalogue is closed from the clients' side). 3. **Render** the messages and tool list through the chat-template renderer (6.13) into token ids, with the generation prompt, including the opening think marker that the server adds itself. 4. **Check the length**: with context-shift disabled (as the launcher has it today), a prompt longer than the context is refused with an error body, and generation that reaches the end of the context stops with the finish reason "length". 5. **Prefix cache**: find the longest common prefix with the state kept from the last request and reuse the KV cache and the DeltaNet recurrent state; because the recurrent state cannot be rolled back to an arbitrary position, the cache keeps **snapshots** at chosen boundaries (the prototype measured one 30-layer snapshot at 62.9 MiB and about 0.5 ms [DERIVED from the model geometry]; the number of snapshots is `cache.snapshots`, value unmeasured). 6. **Prefill** the remaining tokens (6.7). 7. **Decode** with the sampler. 8. **Stream** deltas (below). 9. **Stop** on the end-of-turn token, a stop string, the maximum token count, the end of the context, a client disconnect or a stop signal. 10. **Finish**: usage counts, per-request statistics, release of the slot.

**Slots and queueing.** The engine has one decode slot by default (`engine.slots` = 1, as the launcher's `-np 1` has it). A request that arrives while the slot is busy waits in a bounded first-in first-out queue (`server.queue_max`) and receives a busy status when the queue is full. What llama-server does in that case is **not catalogued**.

**Sampling.** Defaults from the launcher today: temperature 1.0, top-p 0.95, top-k 20, min-p 0.0 (`sampling.*`); each can be overridden per request. The order in which llama-server applies the sampler stages, the penalties it supports and the meaning of its seed are **READ-SRC items still to be read** (6.14). The random generator is counter-based and seeded by request seed, token index and stage, so a given seed repeats exactly (trainer principle 7).

**Reasoning and tool calls.**

| Item | Behaviour specified | Basis and status |
|---|---|---|
| Think marker | The generation prompt ends with the opening think marker; the model's output up to the closing marker is the reasoning | Master 8.3 and 8.4 (READ) |
| Reasoning field | With the setting the launcher has today (`--reasoning-format deepseek`), reasoning text goes in a separate `reasoning_content` field of each delta and not in `content`; Pelias has the same behaviour as a parameter (`api.reasoning_field`), default `reasoning_content` | Launcher; the clients' needs are **UNVERIFIED** |
| Tool calls | The model writes the native XML form (an outer tool-call element holding a function element with named parameter elements). The server parses it into OpenAI `tool_calls` entries with a function name and an arguments string in JSON. Parameter values arrive as text; their JSON types are taken from the tool's schema | Master 8.3 (READ). The streaming form (argument fragments or one final chunk), parallel calls, and the handling of malformed output are **not catalogued** |
| Template switches | `enable_thinking` (true in the launcher today) is a parameter and a per-request override | Launcher |
| Roles | `system`, `developer`, `user`, `assistant`, `tool`; a missing system prompt, a developer role and unexpected client commands must not make the model or the server fall back or fail | U8 (trainer master 3.3). Unknown roles are rendered as the template defines or refused with a clear error, never guessed |

**Errors.** Every error is a JSON body with a message, a type and a code in the OpenAI style, and an HTTP status. An internal fault (an allocation above the plan, a handoff timeout) fails the request, not the process, and is counted.

**Security and privacy.** Loopback bind by default; no authentication by default; a `server.api_key` parameter is available if a client sends one. The request log records structure only by default: roles, tool names, whether a system prompt is present, field names, token counts, timings. It never records message text unless an explicit debug parameter is on (trainer rail 15.6). Cross-origin headers for a browser-based client are **UNKNOWN**: the web-UI client's needs are in the catalogue.

**Shutdown.** First signal: stop accepting, let the current request finish its token (or abort it), write statistics, close the port. Second signal: exit at once (6.5).

**What is not decided.** Whether any client needs Ollama's own API shape; whether any client needs routes other than the chat endpoint; the exact SSE bytes (comments, usage chunk); the behaviour on a full queue. All are closed by the catalogue and the request logs (6.14), not by guessing.

## 6.13 The native tokenizer and chat-template renderer

**Measured needs (text census of the checkpoint's `tokenizer.json` and `chat_template.jinja`, 2026-10-24; MEASURED by count, approximate because it is a text search; trainer master 8.3).**

| Part | What was counted | What must therefore be written |
|---|---|---|
| Tokenizer | Byte-pair encoding; 248,044 vocabulary entries; 247,587 merges; 33 added tokens (21 special); normalizer NFC; pre-tokenizer: one split pattern (contractions matched without regard to case, runs of letters and marks, single digits, punctuation runs, newline runs, whitespace with a `(?!\S)` look-ahead) followed by the byte-level mapping | UTF-8 handling; Unicode general-category data for letters, marks, numbers and whitespace; NFC composition data; a hand-written matcher for that pattern (no regular-expression library). The tables are data from the Unicode Character Database, not code from llama.cpp |
| Template | 158 lines, 8,057 bytes. Statements: `set` (26), `if`, `elif`, `else` (30, 7, 9), `for` (6), `macro` (1). Expression blocks: 41. Filters: `trim`, `length`, `tojson`, `safe`, `string`. Tests: `string`, `iterable`, `mapping`, `defined`, `none`, `undefined`, `sequence`. String methods: `startswith`, `endswith`, `split`, `rstrip`, `lstrip`. Also `namespace()`, `raise_exception()` and the loop variables `index0`, `first`, `last`, `previtem`, `nextitem` | A small interpreter for a subset of a template language. This template only: a client that sends another template, or a later model, needs the census re-run |

**Files and roles (DECISION, proposed).**

| File | Role |
|---|---|
| `tok/utf8` | Decode, encode and validate UTF-8; the policy for invalid sequences is settled by P-D1 (UNVERIFIED which policy the served behaviour has) |
| `tok/unicode_data` | Generated tables: general category, canonical combining class, composition pairs; the Unicode version is recorded in the file; generated by an original tool in `tools/` |
| `tok/normalizer` | NFC. Whether llama-server applies NFC is **UNVERIFIED**; the parameter `tok.apply_nfc` is set by what P-D1 shows the data needs |
| `tok/pretokenizer` | The hand-written matcher for the split pattern, then the byte-level mapping of each piece |
| `tok/bpe` | Rank table from the file's merges; repeatedly joins the adjacent pair of lowest rank within each piece; a cache for repeated pieces (size is a parameter) |
| `tok/added_tokens` | The 33 added tokens; split the input on special-token strings before BPE when special parsing is on |
| `tok/tokenizer` | Public interface: encode with a special-parsing flag; decode; a **streaming detokenizer** that holds back an incomplete UTF-8 sequence until the next token completes it, so that an SSE chunk never carries half a character |
| `tok/template_lexer`, `template_parser` | Text, statement, expression and comment blocks; whitespace control; a syntax tree for the constructs in the census only |
| `tok/template_eval` | Evaluator over dynamic values (string, number, boolean, none, list, map) with the template language's truthiness; the filters, tests, methods and loop variables above; `tojson` uses the project's JSON writer, and its exact output format (separators, key order, escaping) changes the prompt bytes, so it is settled by P-D2 |
| `tok/chat_template` | Renderer interface: messages, tools and switches (`add_generation_prompt`, `enable_thinking`) in, text out; the tokenizer then turns the text into ids |

**Rules.** (1) The vocabulary, merges, added tokens and template are read **from the GGUF** (`tokenizer.ggml.*`, model type `gpt2`; `tokenizer.chat_template`), never from a second file. (2) A template construct outside the census subset is a **refusal at load**, naming the construct; it is never approximated. (3) The evaluator has hard limits on loop iterations, nesting depth and output size, because a template is data from a file or a client. (4) Parity with the served behaviour is a test, not a dependency: the Hugging Face tokenizer and llama.cpp's tokenizer and renderer are test-only oracles (P-D1, P-D2).

**Traps handled explicitly (master 8.3).** The template splits assistant text on the closing think marker, so a literal think marker inside an answer is silently removed when `reasoning_content` is not supplied (82 of 3,000 turns in the author's test), and the server therefore always passes reasoning separately and round-trips every assistant turn (render, tokenize, detokenize, compare); tool calls use the native XML form and are rendered by the template, never by hand; the `developer` role exists in the newer template; user text that contains the text of a special token would be tokenized to the special id when special parsing is on. For serving, `server.special_in_user_content` is `pass`, `escape` or `refuse`; the default must reproduce the served behaviour (`pass`) and every occurrence is counted in the request log. The choice is **open for the author** (Part 7.16, O10).

**Order of operations when writing them (DECISION, proposed).** (1) UTF-8 and the Unicode tables with NFC; (2) the byte-level mapping and the pre-tokenizer matcher; (3) BPE; (4) added tokens and the streaming detokenizer; (5) check token-id parity on public fixtures, then on the author's dataset with counts only (P-D1); (6) the template lexer, parser and evaluator; (7) rendered-text parity (P-D2); (8) the round-trip check on every assistant turn. **The effort of the tokenizer and the renderer is unmeasured** (D37).

## 6.14 The llama-server catalogue: the specification that was never written

**Why this section exists.** The author's reason for the ground-up ruling (AUTHOR-STATED, 2026-10-24) is that routing through llama.cpp "brought in behaviour that was not accounted for". The proprietary server must therefore reproduce, on purpose and by name, whatever behaviour of llama-server the clients rely on, and drop the rest on purpose. That needs a catalogue. **It does not exist.** The author's question of 08:43 ("whether every parameter is mapped for the server") has the answer: no, and this section is how it becomes yes. The catalogue is held engine work; the line "catalogue llama.cpp's parameters and behaviours as the spec for the Pelias server" has been put to the author twice for the engine hold queue and is unanswered (Part 7.16, O8).

**What has been counted (MEASURED, rough text counts, pinned commit 2ca15f540).** 361 option definitions in `common/arg.cpp`, 139 of them tagged for the server; 36 distinct POST route strings (GET routes not captured); about 86 request-field names read. **Not catalogued:** the API behaviour, the sampler, tool-call and reasoning handling, the prompt cache.

**Launcher parameters the clients rely on today, and what each becomes (DECISION, proposed).** Sources: the Windows handoff of 2026-10-23 (Part 8.8), the runbook launch line, the prototype's launcher defaults.

| llama-server option or environment variable | Value today | In the proprietary runtime |
|---|---|---|
| `-m` | the served GGUF | `model.path` |
| `--alias` | the model alias | `model.alias`, also the id returned by the model list |
| `--host`, `--port` | loopback address, 8012 | `server.bind_address`, `server.port` |
| `--jinja` | on | Always: the renderer is the project's own (6.13) |
| `-np` | 1 | `engine.slots`, default 1 |
| `-c` | 32768 | `kv.context_tokens` |
| `-n` | -1 | `generation.max_tokens`, default unlimited up to the context end |
| `--no-context-shift` | on | `engine.context_overflow` = stop (default); a prompt that does not fit is refused |
| `-ngl` | 99 | Not a parameter: placement is the memory plan (6.9) |
| `-ot` (experts on the CPU) | `.ffn_.*_exps.=CPU` | Not a parameter: the hot-and-cold split is the bank plan |
| `--flash-attn` | on | Not a parameter: the attention kernel is built in |
| `--cache-type-k`, `--cache-type-v` | q8_0 | `kv.type` |
| `--reasoning-format` | deepseek | `api.reasoning_field` = `reasoning_content` |
| `--chat-template-kwargs` | `enable_thinking` true | `template.enable_thinking`, default true, per-request override |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | 1.0, 0.95, 20, 0.0 | `sampling.*` defaults |
| `-t`, `--prio`, `--load-mode` | 7, 2, none (the prototype launcher's defaults, from the roadmap; the current production flags did not have them) | `cpu.threads` (decided in E4, 6.7), `os/threads` priority, load strategy of 6.8; not adopted without a test |
| `PELIAS_HOT_SET` | the hot-set file | `bank.hot_set_file` |
| `PELIAS_TIER_BYPASS_N` | 32 (cell A) or 9 (production) | `prefill.bypass_tokens` |
| `PELIAS_TIER_MAX_MIB` | 7300 in the Windows handoff | `bank.size_gib` (7.5 in the plan) |
| `PELIAS_TIER_VERBOSE`, `PELIAS_TIER_STATS_FILE` | on; a statistics file | `log.level`, `stats.file` |

**The surfaces still to be catalogued.**

| Surface | Counted | Known from the clients today | Open | How it is closed | Module |
|---|---|---|---|---|---|
| Server options | 361 defined, 139 tagged server | The ones in the table above | Which others any client sets, and defaults the clients assume | Read `common/arg.cpp` and the server's argument handling; one row per option with file and line | `config`, `server/` |
| POST routes | 36 distinct strings | The chat endpoint | Which routes a client calls | Read the route table; compare with the clients' request logs | `server/` |
| Request fields | about 86 names | The chat fields the clients send | Meaning, defaults and validation of each | Read the request parser; mark each field ADOPT, CHANGE, DROP or UNKNOWN by client evidence | `server/` |
| Sampler | not counted | Four defaults | The order of stages; penalties; seed semantics | Read the sampler chain construction | `engine/sampler` |
| Reasoning | not counted | The `reasoning_content` field in use | Streaming of reasoning; effect of per-request switches | Read the reasoning-format handling | `server/` |
| Tool calls | not counted | Visual Studio's Agent mode is the only mode that takes a bring-your-own model (Part 8.7), which implies tool calls; which schemas it sends is unmeasured | Streaming form, parallel calls, malformed output, type conversion of arguments | Read the tool-call parsing and the template's tool section | `server/`, `tok/` |
| Prompt cache and slots | not counted | One slot; context 32,768; no context shift | Reuse rules, eviction, queueing behaviour | Read the slot and cache code | `engine/`, `server/` |
| Errors and SSE bytes | not counted | None recorded | Error bodies, status codes, comment lines, final usage chunk | Capture real streams (structure) from each client | `server/` |
| Cross-origin and web UI | not counted | A web UI client exists | Whether it needs cross-origin headers or a served page | Capture its requests | `server/` |

**Method (order of operations; none starts until the author says go).** (1) The author answers whether to add the hold-queue line. (2) A read-only reading of the pinned commit's server and argument sources (not a build), one row per option, route, field and behaviour, each with its file and line [READ-SRC]. (3) Capture real client requests and streams, structure only (roles, tool names, presence of a system prompt, fields, event shapes), from Visual Studio Copilot, odysseus-ai, the web UI and the gauntlet and speed tools; for products named in the clients sheet, the lookup of their published request formats waits for the author's go-ahead (Part 7.16, O6). (4) Classify every row ADOPT (a client needs identical behaviour), CHANGE (different on purpose), DROP (no client needs it) or UNKNOWN. (5) The server's specification is the ADOPT and CHANGE rows. (6) Test by replay: each captured request is sent to the external oracle server and to Pelias and the response structures, finish reasons and event shapes are compared; message text is not compared.

## 6.15 Order of operations for building Part 6 (new text)

The order follows the trainer's rule: build the most uncertain thing first, in the cheapest form that can be tested against something trusted. The engine is HELD; the shared items are written by the trainer's stages first.

| Step | What | Phase or stage | Gate (from the existing ladders) | Depends on |
|---|---|---|---|---|
| 1 | `os/` file mapping, threads, clock; `hal/` device, stream, event, arena, governor, pinned pool with the two host backends | T1 and E2 | A plan above the ceiling is refused; clean build on WSL2 and Windows-native; boundary checks BC1 to BC4 and BC6 | Wave 0 results (done) |
| 2 | `io/` reader with `io/sha256`, `io/json`, layout check | T1 and E3 | Lists all 753 tensors with types; parses a trainer-exported file and its manifest identically | 1 |
| 3 | `quant/` types, references, CPU quantizers | T1b | P-E6: byte-identical to the oracle | 2 |
| 4 | `tok/` tokenizer, then renderer | T2 and E10 (written once, early, for the trainer) | P-D1 and P-D2: zero mismatches | 2 |
| 5 | `kern/` spine references, then fast kernels | T4 to T6, E7 | Rung 1 tolerances; kernel parity in the style of the backend operation tests | 1, 3 |
| 6 | `verify/` comparators and the external oracle harness in `tests/oracle/` | E3 and T12 | The oracle reproduces the stock noise floor (KL 0.017353 at batch 16, 0.018286 at batch 1024) | 2 |
| 7 | `bank/` hot-set solver and I/O, byte-identical to `export_hot_set.py` | E5, P-E14 | Byte-identical output on the same input | 2 |
| 8 | `bus/` threadpool, then doorbell | E4, E6 | E4: throughput at or above llama.cpp's CPU expert path, output matching the oracle. E6: layer latency within the E0 model, handoff p99 no worse than 18.7 µs | 1, 3, 5 |
| 9 | `gpu/` expert kernels and bank; `engine/` layer, then decode loop, then prefill | E5 to E9 | E5, E7, E8 (at least the tier's result, then 55.0 tok/s at 12k), E9 | 5 to 8 |
| 10 | `server/`: `os/net`, HTTP parser and writer, JSON, endpoints, SSE, sampler, tool-call parser | E10 | Drop-in for the launcher; capability probes; a Visual Studio session works; replay of captured requests (6.14) | 4, 9 and the catalogue |
| 11 | `startup/` and `plan/`: the quick probe, plan card, capability probe | E11 | The same code base passes the E8 to E10 gates on WSL2 and Windows-native | 1, 9, 10 |

## 6.16 Gap register for Part 6 (everything this part could not specify, with the label and the way to close it)

| # | Gap | Label | How it is closed | Owner |
|---|---|---|---|---|
| G1 | The llama-server catalogue (6.14) | Not written | Held engine work; needs the author's go and the clients' request captures | Author |
| G2 | The effort of the tokenizer, the renderer, the server and the reader | UNVERIFIED | Measured when each is built; no estimate is given | - |
| G3 | Whether llama-server applies NFC; invalid-UTF-8 policy | UNVERIFIED | P-D1 | Tokenizer |
| G4 | Exact `tojson` output format | UNVERIFIED | P-D2 | Renderer |
| G5 | Whether the doorbell beats stream synchronization with a real CPU workload in the stream | UNVERIFIED | E6 | Doorbell |
| G6 | Whether a resident kernel survives on Windows-native beyond what P-H13 tested | MEASURED for kernels of 30 seconds or less of the probe's load (all completed, no display-driver events); heavier kernels and other display setups unmeasured. The design keeps a resident kernel to one token (6.6), far shorter | Extend P-H13 only if a design needs a longer resident kernel | Doorbell |
| G7 | Whether WSL2 exposes the CPU's sibling-thread pairs; the best thread count | UNVERIFIED | E-A1 in E4 | Threadpool |
| G8 | Acquire and release ordering enough for the flags | UNVERIFIED | E-A2 in E6 | Doorbell |
| G9 | Whether huge pages or an aligned layout help | UNVERIFIED | P-E11 | Reader |
| G10 | Load rate from the WSL2 disk against the Windows mount | Unmeasured | P-M4 | Reader |
| G11 | The margin of the VRAM ceiling; the engine's own peaks; the bank size as a ruling | Unmeasured; open decision | Disturbance test, P-E4, decision O1 | Governor |
| G12 | Whether the 16-byte staged read lifts the Q6_K cold read to 20 to 26.2 GB/s | UNVERIFIED | P-E10 | Kernels |
| G13 | Whether the unused MTP block explains why the VRAM ceiling was not reached | UNVERIFIED | Compare VRAM and speed with the block off and on | Engine |
| G14 | The values of every server limit and timeout | Unmeasured | Chosen when written; tested with oversize, truncated and slow input | Server |
| G15 | Whether own GEMM reaches 0.85 of cuBLASLt | UNVERIFIED | Batch C, P-K1 | Kernels |
| G16 | The Dear ImGui decision | Open | Author | Trainer UI |
| G17 | The start-up probe's running time | Unmeasured | Measured when built | Startup |
| G18 | Windows large pages | Unmeasured | P-E11 | `os/large_pages` |

## 6.17 Acceptance gates for the proprietary runtime (new text; each reuses a gate that exists)

| # | Gate | Evidence | Where defined |
|---|---|---|---|
| PG1 | The boundary holds | BC1 to BC6 pass on both builds | 6.3 |
| PG2 | The reader is correct and safe | Parses the served file and a trainer export with their manifests; rejects a fixed set of truncated and corrupted files without a crash (counts only) | 6.8; E3 |
| PG3 | The tokenizer is the served tokenizer | Zero token-id mismatches in P-D1 | 6.13; trainer master 11.5 |
| PG4 | The renderer is the served renderer | Zero mismatches in P-D2, including the round trip of every assistant turn | 6.13; trainer master 11.5 |
| PG5 | The quantizers are ggml-equal | Byte-identical output in P-E6 | 6.11 |
| PG6 | The memory rules hold | Zero allocations in the token loop (P-M5); a plan above the ceiling is refused | 6.9; master 15.1 |
| PG7 | The handoff beats the synchronized path | E6 gate | 6.6; Part 7.2 |
| PG8 | The server speaks to the clients | Replay of captured client requests: zero structural mismatches; oversize, truncated and slow inputs handled | 6.12, 6.14 |
| PG9 | The model is right and fast enough | K1, K2 and K3 hold (limit set by the float spread, ruling H6); G3, 55.0 tok/s or more at 12k | Part 7.3; blueprint E8 |
| PG10 | It runs without the oracle | BC5 | 6.3 |
