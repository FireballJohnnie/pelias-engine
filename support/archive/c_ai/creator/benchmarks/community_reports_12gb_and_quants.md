# What people report for Qwen3.6-35B-A3B on 12 GB cards, and published scores for 4-bit and 5-bit GGUFs

Collected 2026-10-19 from public pages (read through a summarising fetch tool, so check a number against its page before quoting it). Everything below is someone else's measurement; nothing was run on this machine. Where a page only had a chart image, the values are not in this file.

## 1. Speed on 12 GB cards (generation tokens/s, own measurements by the authors)

| Setup | Quant | Settings | Result | Source |
|---|---|---|---|---|
| RTX 4070 12 GB, Core i7-9700 (8 cores), DDR4-2667 64 GB, driver 591.59, CUDA 13.1 | UD-Q4_K_M, about 20.6 GiB | `--n-cpu-moe 36 -ngl all -c 65536 -fa on -t 6 -np 1`, thinking off | 24.1 t/s (+/- 0.26) on a clean run; "approx. 28-30 t/s" final; 14.2 then 6.5 t/s while background apps loaded the CPU. Peak VRAM about 7.5 GB at 65K context and 8.4 GB at 131K; peak system RAM about 30 GB. No prompt-processing speed, no accuracy scores | note.com / fika_semla |
| RTX 4070 Super 12 GB, Ryzen 7 9700X, DDR5-6000 48 GB, ik_llama.cpp fork | byteshape IQ4_XS, 4.19 bits per weight | context 131072, Q8 KV cache, multi-token-prediction speculation | 110.2 t/s average on the fork; 79.8 to 97.0 t/s on regular llama.cpp with MTP (same quant) | Reddit post (janvitos) as reported by startupfortune.com |
| 12 GB class (various) | recommends UD-IQ3_XXS, 13.2 GB | `-c 32768 -ctk q4_0 -ctv q4_0 --n-cpu-moe 24 -fa on --batch-size 1024 --ubatch-size 512` | "30-55 t/s" (no source or conditions given); community "10-15 t/s"; the per-GPU tables are images | note.com / zephel01 |
| RTX 3090 24 GB (for scale) | UD-Q4_K_XL | `-c 65536`, all layers on GPU, q8_0 KV | 101.7 / 98.9 / 80.9 t/s (short / medium / long prompt) | aminrj.com |
| GTX 1060 6 GB, i3-8100, 24 GB RAM (for scale) | Q4_K_M | `--n-cpu-moe 36`, `--no-mmap`, `-ngl 99` | 3 -> 17 t/s as tuned (a video demo, relayed by a blog) | jpmrblood.codeberg.page |

Tuning notes from the 4070 report: lowering `--n-cpu-moe` from 40 to 32 made a negligible difference; `-t 6` beat `-t 8` by about 4%; loading without mmap was about 8% faster for about 5 GB more RAM; turning off context checkpoints made the wait for the first token 2.7 times worse.

No report of Q5_K_M speed on a 12 GB card was found. The 4-bit versus 5-bit speed gap on this class of card appears to be unpublished.

## 2. Published scores for 4-bit and 5-bit GGUFs of the base model

| Source and method | Quant | Result |
|---|---|---|
| yrougy/llm-quant-bench (older results, lm-evaluation-harness against llama.cpp, Unsloth UD quants) | UD-Q4_K_M | IFEval 83% / 91%, ARC-Challenge (chat) 95%, HumanEval 59% |
| same | UD-Q4_K_XL | IFEval 84% / 91%, ARC-Challenge 95%, HumanEval 58% |
| same | UD-Q4_K_S, UD-IQ4_NL, UD-IQ4_XS, UD-IQ4_NL_XL, MXFP4_MOE | IFEval 81-83% / 90-91%, ARC-Challenge 95%, HumanEval 56-65% (HumanEval error given as +/- 3.8%; the Q4_K_S 65% is flagged by the author as an anomaly to be redone) |
| same | any 5-bit | none for the 35B-A3B (Q5_K_M appears only for the 27B and a 9B) |
| Unsloth GGUF discussion, SWE-bench Verified, first 100 tasks, mini-swe-agent, single pass | UD-Q8_K_XL / AesSedai Q5_K_M | 53 / 51 resolved of 100 (the poster calls them similar) |
| same discussion, a commenter's HumanEval / HumanEval+ pass@1 | UD-Q6_K_XL | 93.29% / 90.24% |
| localbench (KL divergence vs a BF16 reference over about 250K tokens, top-40 log-probs) | Q8_0 / UD-Q8_K_XL | KL 0.069 / 0.073 (Qwen3.5-35B-A3B Q8_0: 0.121); the Q4 and Q5 values are only in an image |

Two cautions. The two HumanEval figures (about 59% and about 93%) were measured with different harnesses and prompting, which shows how much of such a score is the method and not the quant. And no 4-bit or 5-bit GGUF MMLU-Pro, GPQA or AIME score was found. The only vendor-style quality numbers are for BF16, FP8 and NVFP4 (see `published_reference_numbers.md`).

A newer version of the same repo moved to inspect_ai with BigCodeBench, MUSR and BFCL (tool calling) on 2x RTX 3060 12 GB. Its result files could not be read from here.

## 3. Generic guidance (opinion, not measurement)

One guide calls Q4_K_M the right default for "95% of use cases" and says Q5_K_M gives "a modest gain over Q4 except for code". It is not specific to this model and gives no measurements.

## 4. Sources

- note.com, fika_semla: https://note.com/fika_semla/n/n3a62cf4f2d6f
- startupfortune.com: https://startupfortune.com/110-toks-on-rtx-4070-super-with-qwen36-35b/
- note.com, zephel01: https://note.com/zephel01/n/n4930439f08b2
- aminrj.com: https://aminrj.com/posts/llamacpp-qwen36-35b/
- jpmrblood.codeberg.page: https://jpmrblood.codeberg.page/2026/06/02-llama-cpp-35b-model-6gb-vram-guide/
- yrougy/llm-quant-bench LEGACY.md: https://github.com/yrougy/llm-quant-bench/blob/main/LEGACY.md
- Unsloth GGUF discussion 10: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/10
- localbench: https://localbench.substack.com/p/qwen-36-35b-a3b-gguf-quality-benchmark
- bestllmfor.com: https://bestllmfor.com/guides/qwen3-6-35b-a3b-local-review/
