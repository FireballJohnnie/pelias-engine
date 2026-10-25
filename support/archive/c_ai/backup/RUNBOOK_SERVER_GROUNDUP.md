# RUNBOOK: building the Pelias server ground up (starts the next conversation)

Written 2026-10-24 from master `PELIAS_PROPRIETARY_MASTER_SERVER.md` Parts 6.14, 6.15, 6.17 and 7.1. Nothing here is new design; the master is the source. Labels: DECISION (proposed) means the author has not ruled.

## Step 0: start of the next conversation (5 minutes, no building)

1. The user uploads `HANDOFF_20261024_SERVER_GROUNDUP.md` and this file. Claude reads both. No searches through logs or code.
2. One question call (AskUserQuestion) with the gating decisions O13, O8, O12 and O11 (handoff section 5). Claude states a recommendation in each option; the user answers.
3. If O13 is "the hold lifts": record it as AUTHOR-STATED in the backup RUNBOOK. If not: build only the shared items that the trainer stages need first (below, steps 1 to 4) and say so.

## The build order (master 6.15; each step starts only when the gate of the step before it is met)

| Step | Build | Gate | Depends on |
|---|---|---|---|
| 1 | `os/` (file mapping, threads, clock, sockets later), `hal/` (device, stream, event, arena, governor, pinned pool, two host backends) | A plan above the 10.7 GiB ceiling is refused; clean build on WSL2 and Windows-native; boundary checks BC1 to BC4 and BC6 | Wave 0 results (done) |
| 2 | `io/` reader (direct `mmap` and `MapViewOfFile`), `io/sha256`, `io/json`, layout check | Lists all 753 tensors with types; parses a trainer-exported file and its manifest identically; rejects a fixed set of damaged files without a crash | 1 |
| 3 | `quant/` block formats and CPU quantizers | P-E6: byte-identical to the oracle | 2 |
| 4 | `tok/` tokenizer, then the chat-template renderer | P-D1 and P-D2: zero mismatches | 2 |
| 5 | `kern/` spine references, then fast kernels | Rung 1 tolerances; kernel parity | 1, 3 |
| 6 | `verify/` comparators and the external oracle harness in `tests/oracle/` | The oracle reproduces the stock noise floor (KL 0.017353 at batch 16, 0.018286 at batch 1024) | 2 |
| 7 | `bank/` hot-set solver, byte-identical to `export_hot_set.py` | P-E14 | 2 |
| 8 | `bus/` threadpool, then doorbell | E4 and E6 gates (handoff p99 no worse than 18.7 µs) | 1, 3, 5 |
| 9 | `gpu/` expert kernels and bank; `engine/` layer, decode loop, prefill | E5, E7, E8 (at least the tier's result, then 55.0 tok/s at 12k), E9 | 5 to 8 |
| 10 | `server/`: `os/net`, HTTP parser and writer, JSON, endpoints, SSE, sampler, tool-call parser | Drop-in for the launcher; capability probes; a Visual Studio session works; replay of captured client requests | 4, 9 and the catalogue |
| 11 | `startup/` and `plan/` | The same code base passes the E8 to E10 gates on both platforms | 1, 9, 10 |

Acceptance gates PG1 to PG10 are in master 6.17. The boundary (what is forbidden in the product, what is test-only) is master 6.3.

## Proposed first working session (DECISION, proposed; the user chooses)

- **Track A, read-only, no build: the llama-server catalogue** (master 6.14 method). Needs O8. (1) Re-clone llama.cpp at 2ca15f540, read-only. (2) Read the argument handling and the server sources; one row per option, route, field and behaviour, each with file and line [READ-SRC]. (3) The user captures real client requests and streams (structure only) from Visual Studio Copilot, odysseus-ai, the web UI and the speed tools; client-documentation lookups only with O6. (4) Classify every row ADOPT, CHANGE, DROP or UNKNOWN. The ADOPT and CHANGE rows are the server's specification. (5) Replay test: each captured request goes to the external oracle server and to Pelias; compare response structures, finish reasons and event shapes, not message text.
- **Track B, build: steps 1 and 2 above** in the sandbox with the host-only build framework (g++ against CUDA headers, fake-GPU runtime, clang for PTX where needed); the user runs the real builds on WSL and on Windows. Everything written is tested before it is handed over.
- **Option C (cheapest test of the most uncertain server part):** the HTTP/SSE shell against a fake backend that replays captured streams, so the server's wire behaviour can be tested before the engine exists. Needs the captures from Track A.
- The user has the interim launchers meanwhile (handoff section 2).

## Per-step routine (every step)

1. Claude states in one sentence what is being built and which gate closes it.
2. Claude writes the code in the sandbox, builds host-only, runs the tests, and fixes what fails before handing over.
3. The user runs anything that touches the GPU or the real machine, on WSL and on Windows, and puts results in `C:\AI`. Claude reads them and labels each number MEASURED, with the run it came from.
4. A failed gate stops that branch; the reason is logged in the backup RUNBOOK.
5. Backup: new files into `C:\AI\cl-sndbx-bkp\stuart-creator\<date>\<topic>\`, previous INDEX.md and RUNBOOK.md saved first, entries appended; about every 30 minutes.
6. Commits stay local. Push only when the user says "push".

## Measurements the build will produce (nothing may be quoted before it exists)

The effort of tokenizer, renderer, server and reader (G2); the VRAM peaks and disturbance margin (O1, P-E4); the E0 timeline (P-E2); the load rate from the WSL2 disk against the Windows mount (P-M4); the huge-page effect (P-E11); the wide-load kernel rate (P-E10); own GEMM against cuBLASLt (P-K1 to P-K3); the first non-llama server run, which finally isolates the gap to the 88.2 to 92.0 tok/s ceiling.

## Where to look (read only when the step needs it)

Master Part 6.5 to 6.13 for each module's file roles, parameters and order of operations; 6.16 for the 18 open gaps (G1 to G18); 7.9 for the probe registry; 7.16 for O1 to O14; Part 8 for the history.
