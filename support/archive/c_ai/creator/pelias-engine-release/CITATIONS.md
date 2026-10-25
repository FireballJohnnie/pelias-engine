# Citations and acknowledgements (engine)

**Status: draft, private repository.** Each entry below was checked on **2026-10-20** by opening the page and reading what it says (title, author, date or licence as shown). A "role" says how the source relates to this repository; an entry is never listed as a dependency unless the code or tests use it. Where a role depends on something only the author can say, it is marked **author-stated**.

| Role | Meaning |
|---|---|
| **Built on** | Code or a file format the engine patches, links against, or follows. |
| **Followed in tests** | Documentation the tests and the setup scripts follow. |
| **Related work** | Named in the README (section 10) as a neighbour. No code from it is used, no comparison has been run yet, and this repository makes no novelty claim. |
| **Consulted** | Read while designing; the engine's code does not depend on it (**author-stated**). |
| **Development tool** | Software used to help write and review the work. |

---

## 1. Built on

* **llama.cpp and ggml**, Georgi Gerganov and contributors. <https://github.com/ggml-org/llama.cpp>. MIT licence (README section 11). The engine patches it and uses its CPU kernels and quantisation formats. Stock runs in `benchmarks/` used build `b11399-2ca15f540`.
* **GGUF file format specification**, ggml-org. <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>. Specifies the binary model file format, its metadata and tensor naming. The GGUF file is the interface between the trainer and the engine.
* **k-quants**, Iwan Kawrakow (`ikawrakow`), llama.cpp pull request #1684, merged 2023-06-05. <https://github.com/ggml-org/llama.cpp/pull/1684>. Introduced the 2- to 6-bit quantisation types; the cold-path kernels decode its Q4_K and Q6_K blocks.
  Explainer by a third party: *K-Quants*, `iuliaturc/gguf-docs`. <https://github.com/iuliaturc/gguf-docs/blob/main/k-quants.md>. Describes the double-quantised block scales and offsets with super-block constants.
* **Qwen3.6-35B-A3B**, Qwen Team. <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>. Apache-2.0. The base of the model measured here. Its card states 40 layers, 256 experts, 8 routed plus 1 shared expert active per token, expert intermediate size 512, hidden size 2048, and a layout of 10 x (3 x (Gated DeltaNet to MoE) then 1 x (Gated Attention to MoE)). The geometry used in the README matches the card.

## 2. Followed in tests and setup

* **CUDA Runtime API, Memory Management**, NVIDIA. <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>. Documents `cudaHostRegister` and `cudaHostGetDevicePointer`, the calls for pinning host memory and reading it from the GPU. The page fetched was for CUDA 13.4; the build used toolkit 13.0.88.
* **CUDA Programming Guide, section 5.5 Floating-Point Computation**, NVIDIA. <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>. Table 48 lists a maximum error of 2 ulp for single-precision `expf`, the figure the cold-path test's error bounds assume.
* **Higham, N. J.**, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, SIAM, 2002, section 3.1. The rounding-error constant gamma_n used to derive the test tolerances. (A book; not fetched.)
* **Transparent Hugepage Support**, The Linux Kernel documentation. <https://docs.kernel.org/admin-guide/mm/transhuge.html>. Documents `/sys/kernel/mm/transparent_hugepage/enabled` (`always`, `madvise`, `never`), which the audit script reads and the huge-page test relates to.
* **Advanced settings configuration in WSL**, Microsoft Learn. <https://learn.microsoft.com/en-us/windows/wsl/wsl-config>. Documents the `.wslconfig` keys `memory`, `processors`, `swap` and `swapFile` used on the test machine.
* **CUDA on WSL User Guide**, NVIDIA. <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>. Setting up CUDA on WSL 2.

## 3. Related work (named in README section 10; not used, not yet compared)

* **KTransformers**, `kvcache-ai/ktransformers`. <https://github.com/kvcache-ai/ktransformers>. Apache-2.0. A CPU and GPU split for running and fine-tuning large models. Paper: *KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models*, ACM SIGOPS 31st Symposium on Operating Systems Principles, 2025.
* **Fiddler**, Kamahori, Tang, Gu and two other authors. *Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models*. arXiv:2402.07033, submitted 2024-02-10. <https://arxiv.org/abs/2402.07033>.
* **MoE-Infinity**, Xue, Fu, Lu and two other authors. *MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache* (title of version 3, 2025-03-12). arXiv:2401.14361, first submitted 2024-01-25. <https://arxiv.org/abs/2401.14361>.
* **ik_llama.cpp**, Iwan Kawrakow. <https://github.com/ikawrakow/ik_llama.cpp>. MIT licence. "llama.cpp fork with additional SOTA quants and improved performance"; its README says it started as a fork of llama.cpp in June 2024.
* **llama.cpp pull request #29887**, `am17an`: "llama : add a GPU cache for MoE experts kept in host memory". <https://github.com/ggml-org/llama.cpp/pull/29887>. **Open** when checked on 2026-10-20 (not merged). It ports an expert cache from the qvac-fabric project, runs expert multiplications on the GPU from an LRU cache of experts so that only misses are uploaded, applies to batches of 32 tokens or fewer, and is enabled with `--moe-cache-mib N`.
* **moe-cache**, `IbrahimSiddiqui007`. <https://github.com/IbrahimSiddiqui007/moe-cache>. MIT licence. A plugin for an unmodified `llama-server` that keeps frequently used experts in RAM and reads the rest from an SSD within a memory limit. It addresses models larger than RAM, which is a different case from this engine's.

## 4. Consulted (author-stated)

* **Unsloth Qwen3.6-35B-A3B-MTP-GGUF**, Unsloth AI. <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF>. Apache-2.0. GGUF builds of the same base model with Multi-Token Prediction heads, listed in many quantisations including `UD-Q4_K_M` and `UD-Q4_K_XL`. Read for how those variants are laid out and named.

## 5. Base models of the author's other private models (not part of this repository)

* **Gemma 4 26B A4B**, as uploaded by Unsloth AI. <https://huggingface.co/unsloth/gemma-4-26B-A4B>. The card shows Apache-2.0, lists `google/gemma-4-26B-A4B` as its base, and its MoE table gives 25.2B total and 3.8B active parameters. The 12B variant and the exact Qwen3-Coder model the author used are **not yet confirmed**, so they have no entry here.

## 6. Development tools

* **Claude** (Anthropic): Sonnet, Opus and Fable models, used for coding, review, analysis and documentation. Exact versions are not recorded.
* **Gemini Pro and Gemini 4 Argon** (Google), used for the same kinds of work.
  *Gemini 4 Argon: our next era of frontier intelligence*, Google, 2026-09-30. <https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/>.
  *Gemini 4 Argon is now available for everyone* (headline: "Gemini 4 Argon: Built to solve your hardest problems"), Google, 2026-10-13: available in the Gemini app to Google AI Pro and Google AI Ultra subscribers and in the API from that date. <https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-argon/>.
  API reference: model ID `gemini-4-argon`, context window 1,048,576 tokens. <https://ai.google.dev/gemini-api/docs/models/gemini-4-argon>.

Which tool helped with which part is not recorded, so no per-file attribution is made. The research, the design and the hardware measurements are the author's.
