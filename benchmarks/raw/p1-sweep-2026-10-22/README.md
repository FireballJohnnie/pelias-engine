# P1 sweep, 2026-10-22: raw results and logs

One machine: RTX 4070 12 GB, Ryzen 7 5800X3D, 4x16 GB DDR4-3200, WSL2 (kernel 6.18.54.1), driver 617.42, llama.cpp `2ca15f540` with the tier patch,
Qwen3.6-35B-A3B Q4_K_M. Nothing here is a claim about other hardware.

## Contents
- `results/` - per-script result TSVs and per-launch logs (series K, A, C to N, J, B, O, P to V, W, X), probe outputs, the U sweep folder.
- `terminal_logs/` - the two pasted terminal sessions cut apart per script, `00_commands_as_typed`, and `MANIFEST.txt` (source hashes).
- `RUN_ORDER.md` - the order the series were run in.
- The account name in paths and prompts was replaced with `user`. Scripts and patches are not part of this batch.

## Reading the numbers
- Same-configuration launches differ by about 4 tok/s on short generation; a difference under that is noise. Claims need at least 3 launches.
- Gate names: P1 needs +5% on short generation, 12k generation or 12k prompt reading, nothing worse than -3%, VRAM at most 11.2 GiB. G3 needs at least 55 tok/s at 12k, at least 1150 prompt tok/s and VRAM at most 11.2 GiB. HOLD means the candidate's launch-to-launch spread was larger than its gain, so the gauntlet does not call it.
- Peak VRAM is read at 0.1 GiB resolution.

## Headline results [MEASURED, means over launches]
| Series, cell | n | Short | 12k gen | Prompt@12k | Peak GiB | P1 / G3 |
|---|---|---|---|---|---|---|
| W, 7.0 GiB bank, base | 3 | 46.9 | 45.4 | 1669 | 10.0 | PASS / FAIL |
| W, 7.0 GiB, `-ub 4096` | 3 | 48.9 | 46.0 | 2255 | 10.7 | PASS / FAIL |
| W + X, 7.0 GiB, MTP draft 2 | 9 | n/a | 51.5 | about 1610 | 11.1 to 11.6 | mixed / FAIL (2 of 9 launches reached 55 at 12k) |
| W, 7.0 GiB, MTP draft 3 | 3 | 49.8 | 52.9 | 1633 | 11.2 | HOLD / FAIL |
| W, 7.0 GiB, MTP 3 + `-ub 4096` | 3 | 49.3 | 50.7 | 1184 | 11.6 | HOLD / FAIL |

No tested cell passes G3 on this machine under WSL2. Other findings (thread and priority settings, CUDA graphs, pinned memory, load modes, patch isolation) are in the TSVs; interpretation is kept in the project notes.
