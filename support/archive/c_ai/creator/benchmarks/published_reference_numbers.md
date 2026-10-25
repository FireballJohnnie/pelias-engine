# Published reference numbers for the base model (Qwen3.6-35B-A3B)

Collected 2026-10-19 by reading the pages below. Values are as the pages returned them (the pages were read through a summarising fetch tool, so check a number against its page before quoting it anywhere else). Nothing here was measured on this machine.

Pages: the model card `Qwen/Qwen3.6-35B-A3B` (Hugging Face), Unsloth's NVFP4 page (`unsloth.ai/docs/basics/nvfp4`), Unsloth's Qwen3.6 tutorial (`unsloth.ai/docs/models/tutorials/qwen3.6`), Unsloth's MLX 8-bit card, Unsloth's Dynamic 3.0 GGUF page.

## 1. The model card (Qwen's own numbers, thinking mode)

| Benchmark | Qwen3.6-35B-A3B | Runnable on this machine? |
|---|---|---|
| MMLU-Pro | 85.2 | yes, as a subset (public dataset); long thinking outputs |
| MMLU-Redux | 93.3 | yes, as a subset |
| SuperGPQA | 64.7 | possible, large set; subset only |
| C-Eval | 90.0 | possible; Chinese-language |
| GPQA | 86.0 | dataset is gated (needs an accepted licence and a login); subset only |
| HLE | 21.4 | no (gated, judge-graded) |
| LiveCodeBench v6 | 80.4 | needs a code-execution sandbox; heavy |
| HMMT Feb 25 / Nov 25 / Feb 26 | 90.7 / 89.1 / 83.6 | small sets; very noisy |
| IMOAnswerBench | 78.9 | possible; small |
| AIME26 | 92.7 | yes; 30 problems, very noisy |
| SWE-bench Verified / Multilingual / Pro | 73.4 / 67.2 / 49.5 | no (agent scaffold, long runs) |
| Terminal-Bench 2.0 | 51.5 | no (harness, 3-hour timeouts, 32 CPUs) |
| TAU3-Bench / MCPMark / MCP-Atlas / WideSearch | 67.2 / 37.0 / 62.8 / 60.1 | no (judge models, tool servers) |
| Claw-Eval Avg / SkillsBench Avg5 / NL2Repo | 68.7 / 28.7 / 29.4 | no |
| Vision and video tables (MMMU 81.7, MathVista 86.4, VideoMME 86.6, ...) | see the card | no: the exported file has no vision part |

How the card says these were run (from its footnotes): thinking mode; temperature 1.0 and top-p 0.95 on the agent tasks; 32,768 output tokens for most queries and 81,920 for complex maths and programming benchmarks; context 128K or more; Terminal-Bench averaged over 5 runs with an 80K output limit; AIME26 is the full 2026 set (I and II).

## 2. Unsloth's quant comparison (same model; MMLU-Pro / GPQA / AIME 2025)

| Variant | MMLU-Pro | GPQA | AIME 2025 |
|---|---|---|---|
| BF16 | 85.75 | 86.36 | 92.50 |
| FP8 | 85.75 | 86.74 | 93.12 |
| Unsloth NVFP4 | 85.85 | 86.74 | 92.29 |
| Unsloth NVFP4-Fast | 85.58 | 87.75 | 91.67 |
| NVIDIA NVFP4 | 85.60 | 87.12 | 91.88 |

The page gives no harness, sampling or hardware for these. The spread between formats is under about 1.5 points on each benchmark, which is the size of the run-to-run noise on sets this small, so read it as "no visible loss", not as a ranking. Speed figures on that page are for a B200 at 128-way concurrency and say nothing about a 12 GB card.

## 3. What is not published as text

- MLX: the KL-divergence and perplexity table covers Qwen3.6-27B only. The 35B-A3B MLX builds (3-, 4-, 8-bit) have no metrics on the pages read.
- GGUF KL divergence for 35B-A3B (the 4- and 5-bit range we care about): shown only in a chart titled "35B-A3B - KLD benchmarks (lower is better)". The values are not in the page text, and the chart image could not be fetched from the workspace. A screenshot of the chart can be read directly.
- Unsloth's KL method (Dynamic 3.0 page, for a different model): 300 held-out prompts, greedy 32 tokens, BF16 as the reference, top-1 accuracy and mean KL. Ours would use a Q8_0 reference (the bf16 file does not fit in memory), so the numbers would not be comparable.

## 4. Recommended sampling (two sources, and they disagree on one value)

| Mode | temp | top-p | top-k | min-p | presence penalty (model card) | presence penalty (Unsloth page) |
|---|---|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 0.0 |
| Thinking, precise coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.5 |

Check the model card directly before settling the thinking-general presence penalty. It is the one sampling lever aimed at repetition loops, and the serve command we print does not set it (the server default is 0). It can be set server-wide with `--presence-penalty`, with no script change.

Other notes from the Unsloth tutorial: `--cache-type-k bf16 --cache-type-v bf16` is its listed fix for gibberish output; multi-token-prediction speculative decoding is `--spec-type draft-mtp --spec-draft-n-max 2`, which it reports as roughly 1.15 to 1.2x faster on this MoE (the GGUF we exported carries the base model's prediction head, untested with the fine-tune).

## 5. What a comparison against these numbers can and cannot say

- The published runs used thinking mode with 32K to 80K output budgets, their own harnesses, and (for Unsloth) data-centre hardware. Ours would use llama.cpp on a quantized file, different prompts and a different harness. A gap of a few points cannot be pinned on the fine-tune.
- The tuned model's thinking has so far been short and fixed in format on easy questions. Whether it reasons at length on hard problems is untested, and it decides whether the reasoning benchmarks (GPQA, AIME, HMMT) are informative. A first check is a few hard problems with a large token limit, before any full run.
- Run time scales with output length: a benchmark of N questions at about T tokens each needs N x T tokens, so at R tokens per second it takes N x T / R seconds. The measured speed and the real reply lengths decide how big a subset is affordable.
- A subset of about 150 items has a 95% interval of roughly 8 points either way; only a drop larger than that can be called.
