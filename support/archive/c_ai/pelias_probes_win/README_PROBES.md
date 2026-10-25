# pelias_probe - Wave 0 hardware probes (WSL2 and Windows)

Written 2026-10-23. SOURCE ONLY. The CUDA parts have never run on a real GPU
(see "What is and is not verified"). Master chapter 11 defines the probes.

## Files
- main.cpp            subcommands: selftest state list h8 h1 h2 h3 h10 h4 h5 h6 h12 h7 m3 m4 h9 h13
- probes_a.h          P-H8 H1 H2 H3 H10 H4 (clocks, tensor-core rates, canary GEMM, streaming)
- probes_b.h          P-H5 H6 H12 H7 M3 M4 H9 H13 (transfers, pinned limit, soak, model read, graphs, display timeout)
- cudautil.h hostlib.h nvml_dyn.h   helpers (JSON results, memory, NVML loaded at run time)
- kernels.h kernels.cu              the only nvcc file (mma, GEMM, streaming, spin, graph kernels)
- cpu_bw.cpp          AVX2 host read (compiled with -mavx2)
- build.sh            WSL/Linux build (nvcc + g++)
- run_wsl_all.sh      WSL runner: census | quick | full | tdr
- run_windows_all.ps1 Windows runner: -Mode facts | quick | full | tdr (installs nothing; says what is missing)
- dev_fake_gpu/       CPU-emulated CUDA used only to test the host logic in the sandbox. Not needed to run the probes.

## Run (WSL)
    mkdir -p ~/pelias_probes && cp /mnt/c/AI/cl-sndbx-bkp/stuart-creator/2026-10-23/trainer_master/probes/* ~/pelias_probes/ && cd ~/pelias_probes && chmod +x *.sh && ./build.sh
    ./build/pelias_probe selftest --out ./selftest_out
    ./run_wsl_all.sh census        (read-only)
    ./run_wsl_all.sh quick         (about 10 min)
    ./run_wsl_all.sh full          (about 70 min; set PELIAS_GGUF=... for P-M4; close other GPU apps)
    ./run_wsl_all.sh tdr           (display-timeout test; screen may flicker; asks before each stage)
Results: ~/pelias_probe_results/<run>/ and copied to
C:\AI\cl-sndbx-bkp\stuart-creator\2026-10-23\probe_results\wsl\<run>\

## Run (Windows PowerShell)
    powershell -ExecutionPolicy Bypass -File .\run_windows_all.ps1 -Mode facts
    ... -Mode quick | full | tdr
Results: C:\AI\cl-sndbx-bkp\stuart-creator\2026-10-23\probe_results\windows\<run>\

## Results format
One JSON per probe (master 11.11 schema). Never overwrites (adds _2). pass is true, false or null (record only).
Exit codes: 0 ok, 2 error, 3 probe failed its threshold, 4 (selftest only) tensor-core checks failed -> runners skip h3 h10 h4.

## What is and is not verified
- Verified in the sandbox: host code compiles clean (g++, clang) against the real CUDA 13.0.88 headers; device code compiles to PTX with clang 18;
  every probe ran end to end on the fake CPU runtime with valid JSON; ASan/UBSan clean; failure, timeout and skip paths exercised;
  PowerShell script parses (pwsh 7.4.6) and its run logic was tested on Linux with stubs.
- NOT verified: any nvcc compile, any run on a real GPU, the mma fragment layout (a mistake shows as a canary_gemm failure in selftest),
  the Windows branches of hostlib.h, the MSVC build, and the Windows facts cmdlets (Get-CimInstance, Get-WinEvent, Get-Counter).

## Hazards
- P-H7 and P-M3 pin many GiB (stop when MemAvailable gets low). P-H13 may reset the display driver. P-H3/H10/H4 load the card hard (abort at 87 C).
- Nothing here reads or writes the HF token. The model file name is logged as test-model.
