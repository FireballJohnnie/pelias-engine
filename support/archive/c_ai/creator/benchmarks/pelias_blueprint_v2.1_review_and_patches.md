# Review of Master Blueprint v2.1 and patches toward v2.2

Reviewed 2026-10-20 from the text you pasted. Your v2.1 is untouched; this file holds a verdict per section plus replacement and new text to merge.
I have not seen the Pelias source, so Part 0's code evidence is accepted from your quotes. Nothing here has been built or run.

Labels: [MEASURED] your logs; [PUBLISHED] a source I read (section 4); [DERIVED] arithmetic from measured numbers; [ESTIMATE] my judgment; [HYPOTHESIS] plausible, untested; [UNVERIFIED] I did not check it; [RESEARCH] an idea with no evidence yet that it works.

## 1. Verdict by section

| Section of v2.1 | Verdict | Note |
|---|---|---|
| Part 0: L5 stash, KV probe | Keep | Agree on both. Unit slip: 22 GB/s is 20.5 GiB/s, so the probe ceiling is about 1.28 tok/s, not 1.38. Conclusion unchanged. Move Part 0 to an appendix; it is a review reply, not blueprint. |
| Part I: audit table | Keep | Two rows need the amendments in 2.1 and 2.4. |
| Phase 1 (THP, `--load-mode none`, `-t 7`) | Fix | Changes two variables in one run. See 2.1. |
| Phase 2 (floors) | Keep | |
| Phase 3 (tracer) | Keep | Still a draft until compiled. |
| Phase 4 (simulator) | Fix | Output format for the fork is unverified. See 2.5. |
| Phase 5a / 5b commands | Fix | They drop the launcher's baked flags. See 2.2. |
| Phase 5c command | Fix | Flag names are mixed between mainline and the fork. See 2.3. |
| Phase 5d, 5e | Missing | Prompt-reuse test and MTP acceptance baseline. See 2.6. |
| Track 2 architecture, E0-E7 | Keep, amend | Three amendments in 2.7. |
| Part III Fix 1 | Keep | |
| Part III Fix 2 | Fix | Not a two-line delete. See 2.8. |
| Part III Fix 3 | Fix | Unit slip and extra PCIe traffic. See 2.8. |
| Part III Fix 4 | Fix | "Zero drift" overclaims. See 2.8. |
| Sequencing, pending gates | Missing | See 2.9. |

## 2. Replacement and new text

### 2.1 Phase 1 (replaces 1b)
- **1a** Read-only: `cat /sys/kernel/mm/transparent_hugepage/enabled; grep -i hugepages /proc/meminfo | head -5`
- **1b** `--load-mode none` alone, on top of N=28:
  `llama --stop`
  `EXPERT_FLAGS="--n-cpu-moe 28" llama --serve --stuart-creator --bg -- -c 131072 -ub 2048 --load-mode none -lv 4`
  Then check the log for the host buffer type (`grep -nE "CUDA_Host|buffer size" <log>`) so a silent failure to pin is visible, then run the speed test with its own label.
- **1c** Threads, one change at a time. First read the current count (`grep -i n_threads <log>` [log format UNVERIFIED]), then `-t 8`, `-t 7`, `-t 6`.
- **1d** Only if 1a shows `madvise` or `never` and 1b did not help: the `.wslconfig` `kernelCommandLine` route. It restarts WSL and changes kernel behavior for everything, you run it yourself, and I have not verified that the WSL2 kernel honors it [UNVERIFIED].
- **Gate:** keep a change if prefill or decode improves by at least 5% and the other does not drop by more than 3%, with GPU peak <= 11.2 GiB.

### 2.2 Launch parity rule (new, applies to all of Phase 5)
Calling a binary directly skips the flags the `llama` launcher bakes in. Every Phase 5 launch must reproduce the production settings: KV q8_0, ctx 131072, ub 2048, the chat-template or reasoning setting, one slot, port 8012. If KV falls back to f16 it costs about 1.2 GiB more at 131k (2.5 vs 1.33 GiB) [DERIVED], which breaks the 11.2 GiB gate, and the probes will not be comparable.
Preferred fix: a `LLAMA_BIN` override in the launcher (this edits a script, so it needs your OK). Otherwise copy the full production command line from the top of the running server's log.

### 2.3 Phase 5a and 5c commands
**5a, mainline cache** (needs a build from or after the release that contains PR #29887; confirm with `llama-server --help | grep moe-cache`):
```
~/llama.cpp-moecache/build-wsl/bin/llama-server -m <GGUF> <production flags> \
  --override-tensor ".ffn_.*_exps.=CPU" -ngl 99 --load-mode none --moe-cache-mib 5734 --port 8012
```
Mainline's cache covers batches of up to 32 tokens; larger prompt batches use the normal path [PUBLISHED, secondary article].

**5c, GenerelSchwerz `moe-cache` fork.** v2.1 mixed mainline's `--moe-cache-mib` with the fork's profile flags. Per the fork's wiki the relevant flags are `--moe-hybrid on`, `--moe-expert-cache-mib` (or `--moe-expert-cache-size`, not both), `--moe-gpu-miss-fraction` (default 0.17), `--moe-expert-profile`, `--moe-profile-adapt occurrence`, plus `-ngl all`, `-fit off`, `--load-mode none`. Its cache owns the routed expert weights and overrides `--override-tensor` / CPU-MoE placement, so do not combine them.
Build (wiki's checkpoint): branch `moe-cache`, commit `28d73c87cb90a9b78d4357ff1b456f456bfa1164`, `-DGGML_CUDA=ON`. Use your `build_llama_cuda_v2.sh` pointed at the new folder.
First pass, self-adapting, no profile file:
```
~/llama.cpp-schwerz/build-wsl/bin/llama-server -m <GGUF> <production flags> \
  --moe-hybrid on -ngl all -fit off --load-mode none \
  --moe-expert-cache-mib 5734 --moe-expert-cache-host-pinned-mb 0 \
  --moe-gpu-miss-fraction 0.17 --moe-profile-adapt occurrence -t 8 --port 8012
```
Then sweep the miss fraction (0.0, 0.17, 0.5, 1.0), then try `--phase-aware-workspace` if VRAM is tight. A second pass with a profile file comes after reading its collection command (2.5).
Caveats from the wiki: qualified on Linux with a single CUDA GPU; native Windows hybrid is not qualified; WSL2 is not mentioned [UNVERIFIED]. Its Qwen3.6 test used a 16 GB card and 50% residency with an NVFP4-expert GGUF, so it says little about your 12 GB, 31% case. Single maintainer.

### 2.4 Part I audit table, amended rows
- Phase 5a launcher path: also needs the parity rule (2.2).
- Phase 5c: flag names corrected as above.

### 2.5 Phase 4 output
The fork's wiki lists `--moe-expert-profile` as a model-bound GGUF or a "geometry-compatible ranked STRP file"; JSON is not listed. So v2.1's "export in GenerelSchwerz format" is unverified. Steps: (i) read the fork's Reproduction page, section "collect a model profile"; (ii) keep our own JSON as the primary output; (iii) write a converter only after reading their spec. The wiki says held-out profile quality "has not been established", which is exactly what our Gate G2 supplies.

### 2.6 New Phases 5d and 5e (cheap, run on the current server)
- **5d Prompt reuse (hybrid model).** Send a long prompt, then the same prompt with one changed last line, and look in the server log for "forcing full prompt re-processing". Published reports show this for Qwen3.6 hybrid models. It decides whether multi-turn Visual Studio use pays a full re-read every turn, which matters more than another tok/s. I will give the exact two-request command when you want to run it.
- **5e MTP acceptance baseline** on the current GGUF: `--spec-type draft-mtp --spec-draft-n-max 2`; record the "draft acceptance" line and tok/s against the N=28 baseline. Published context: one report of 35-37% acceptance on a hybrid-memory Qwen3.6-27B at long context with forced full re-processing; another report puts Qwen3.6-27B above 90% on a Mac. So 70% or better with no slowdown means E6 is worth designing; low acceptance means find out why (state rollback vs head misalignment) before building anything. This also gates v0.3 Module 5 (co-training `blk.40`).

### 2.7 Track 2 amendments
- **E0 first, and split GPU kernel time from sync wait.** E4's target of 5.0 ms or less is a stretch: the spine's bytes at 504-540 GB/s are about 3-4 ms [ESTIMATE], and llama.cpp already uses CUDA graphs. Without E0's split, E4 may chase the wrong 2 ms.
- **"Hot-aware MTP drafter" is [RESEARCH].** A drafted token's experts are unknown until layers run, so "biasing drafts toward resident experts" has no mechanism yet. Keep gating by measured draft length and acceptance until one exists.
- **E5/E6: read the fork's wiki first.** Its page list includes "Sparse Recurrent Snapshots", "Retained MTP State" and "Atomic Speculative Verification" (titles only seen; contents UNVERIFIED). Part of E5/E6 may already exist in 5c.
- **Do not hardcode 7 CPU threads.** The sweep in 1c decides. The fork's own example uses a thread count tuned to the core topology.
- **Trigger for Track 2:** define it numerically: best Track 1 result below 55 tok/s at 12k depth, or MTP unusable after 5e.

### 2.8 Part III amendments
- **Fix 2 (zero-sync dispatch).** Deleting `.tolist()` is not enough if the per-chunk slices of the sorted activations are cut with those host integers. Check how `x` is sliced. Cheap safe version: save the host `counts` from the forward pass in the autograd context and reuse them in recompute and backward, which cuts 120 syncs to 40 [DESIGN, unverified against the code]. True zero-sync needs padded fixed-size buffers or one grouped GEMM over all experts.
- **Fix 3 (checkpoint offload).** 40 x T x 2048 x 2 bytes is 1.25 GiB at 8k and 2.5 GiB at 16k (1.34 and 2.68 are GB). Moving it adds about 5 GiB of PCIe traffic per micro-batch at 16k (2.5 each way), about +16% over 30.38 GiB [DERIVED].
- **Fix 4 (native Q4_K/Q6_K).** Identical weights do not give an identical function. llama.cpp's Q4_K/Q6_K dot-product kernels quantize the activations (Q8_K on CPU, q8_1 in the CUDA kernels), which a bf16-dequantized training pass never sees. Expert-side drift shrinks but is not zero. It also needs a Triton dequant for the Q4_K and Q6_K super-block formats [ESTIMATE: days to weeks]. Order: Fixes 1-3 first, and do Fix 4 only if the Q8_0-reference comparison (existing proposal Phase 4) shows drift matters.

### 2.9 Sequencing and pending gates (new)
1. Pending on the current N=28 setting: persona/format probe, capability suite, 120k needle.
2. Phase 1 (one change per run), then 5d and 5e.
3. Run 5a-5c with their own self-adapting or default residency before Phases 3-4. They need no trace from us, and they may reach the target, which would leave Phases 3-4 as quality checks.
4. Phases 3-4 (trace, simulate) if 5a-5c fall short, or to improve their profiles.
5. Track 2 only through its numeric trigger.
6. Training: run the three greps, apply Fixes 1-3, retrain once, then export, trace, build the profile and serve. A retrain shifts routing, so a profile built earlier is stale.
7. Share one Q4_K/Q6_K kernel library between training and serving, tested against ggml's output.

## 3. What I could not verify
- Whether the WSL2 kernel honors the `kernelCommandLine` THP setting, and whether WSL2 pinned-memory limits affect `--load-mode none` or the fork's host pinning.
- The fork's behavior under WSL2, and the exact format and collection command for its profile.
- Whether the Pelias code matches Part 0's quotes (I have not seen it).
- Log line formats (`n_threads`, cache hit-rate line, "forcing full prompt re-processing").
- The contents of the fork's MTP and recurrent-state pages (titles only).

## 4. Sources
- GenerelSchwerz/llama.cpp wiki, pages "Generic Hybrid MoE", "MoE Cache", "CUDA MoE Expert Cache", "12GB VRAM Setup", "FreeToken vs MoE Cache Qwen3.6" (fetched 2026-10-20).
- developersdigest.tech article on `--n-cpu-moe` vs `--moe-cache-mib` (search result, 2026-10-20).
- llama.cpp issue #23322 (low MTP acceptance with hybrid memory, Qwen3.6) and a Hugging Face discussion on MTP acceptance (search results).
- Your v2.1 text, your speed logs, HWiNFO report.
