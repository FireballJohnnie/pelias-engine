# Windows-native run of the tier build, 2026-10-23 (first runs)

Same machine as the WSL runs (RTX 4070 12 GB, Ryzen 7 5800X3D, 4x16 GB DDR4-3200, overclocked GPU, see `benchmarks/raw/e0-2026-10-22/machine_state/MACHINE_STATE.md`), but the server runs as a Windows program with no WSL layer. Scripts, patches and the Windows copy of the source are not part of this batch.

## Build [MEASURED]
Live tier tree (llama.cpp `2ca15f540` plus the tier patch) copied to a Windows folder, plus one change: the model-file seek in the tier uses `_fseeki64` on Windows (the POSIX `fseeko` with a 32-bit `off_t` cannot address a 20 GiB file). MSVC 19.44.35229 (toolset 14.44), nvcc 13.4.92, CMake 4.4.3, Ninja 1.13.2, CUDA arch 89, Release, AVX2/FMA/F16C/BMI2 (no `-march=native` on MSVC), NCCL off. 358 of 358 steps, no errors (`configure.log`). The WSL builds used nvcc 13.0.88 and gcc, so a Windows against WSL difference is not only the operating system.
The CUDA 13.4 DLLs sit in `bin\x64`; that folder was added to PATH for the server window only.

## Run [MEASURED]
Reference cell: 7.0 GiB bank (7094.2 MiB copied to the GPU, 40 layers), bypass 32, `-b 4096 -ub 4096`, `-t 6 -tb 8`, `--load-mode none`, `-c 32768`, flash attention, q8_0 KV cache, `--n-cpu-moe 40`, the same model file, the same hot set. Timer and hit counter are not in this build. Client: the streaming client of `e0-2026-10-22` (300 tokens at short context, then 300 tokens after an 11,667-token prompt, temperature 0).

| Launch | Short, tok/s | 12k, tok/s | 12k first token | Gap p50 / p99, short | Gap p50 / p99, 12k |
|---|---|---|---|---|---|
| Windows r1 | 47.3 | 53.7 | 4875 ms | invalid (clock ticks) | invalid |
| Windows r2 | 46.8 | 54.1 | 4871 ms | 21.5 / 24.8 ms | 18.3 / 23.1 ms |
| WSL cell A (`e0-2026-10-22`) | 37.9 | 42.1 | 4911 ms | 26.3 / 32.3 ms | 23.3 / 30.9 ms |

- Windows decodes about 25% (short) and 28% (12k) faster than WSL on this client; prefill at 12k is the same (4875 and 4871 ms against 4911 ms) [DERIVED]. Per token that is about 5.2 ms (21.1 against 26.4 ms short, 18.6 against 23.8 ms at 12k), about the size of the timer-free handoff estimate on WSL (at least 4.6 to 5.0 ms per token). That is consistent with the WSL handoff being the difference; it is not proven.
- Windows' gap tail is lighter (p99 about 25 ms against 32 ms), from one WSL launch.
- Run 1 used a client whose clock ticks every 15.6 ms on Windows, so its gap percentiles (16, 31, 32 ms) are invalid; the means are fine (error about 0.25% of the total time). Run 2 used the fixed client.
- Two Windows launches, one WSL launch. Same-configuration launches differ by about 4 tok/s, so this is not yet a claim; three launches each are needed.
- GPU memory in use, whole GPU, Windows `nvidia-smi`: 11,020 MiB with the server up after run 1 (12k context filled); 551 MiB idle with nothing of ours running. Server share about 10,469 MiB [DERIVED]. This is one reading, not a sampled peak, and Windows' idle baseline is about 150 MiB above WSL's.
- The gauntlet (and so gates P1, G3 and H6) is Linux-only, so these runs are not scored against the gates. This client reads lower than the gauntlet on WSL (42.1 here against about 46.0 from the gauntlet at 12k, different launches), so the 54.1 tok/s here must not be read as a G3 result either way.
- `tier_stats.json` is from the clean server exit after run 2.
