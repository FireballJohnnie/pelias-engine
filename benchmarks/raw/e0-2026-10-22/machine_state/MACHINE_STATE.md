# Machine state during the E0 and KL runs, 2026-10-22

Items marked [AUTHOR-STATED] were reported by the machine's owner and were not read back by the analyst. Items marked [MEASURED] are pasted terminal output.

## Software [MEASURED, pasted `wsl --version`]
WSL 3.0.4.0, kernel 6.18.54.1-1, WSLg 1.0.79, MSRDC 1.2.7391, Direct3D 1.611.1-81528511, DXCore 10.0.26100.1-240331-1435.ge-release, Windows 10.0.26300.9711.
After `wsl --shutdown` and a restart WSL printed `Failed to detect nested virtualization support, error: ERROR_INVALID_DATA`; the owner believes this is a known WSL 3.0.4 bug [AUTHOR-STATED]. The GPU was visible afterwards (RTX 4070, 397 MiB used of 12282 MiB).

## `.wslconfig` [file pasted by the owner]
```
[wsl2]
memory=48GB
processors=16
swap=48GB
swapFile=D:\\AI\\Stuart_Creator\\wsl_swap.vhdx
localhostForwarding=true
kernelCommandLine=transparent_hugepage=always swiotlb=65536
virtiofs=true

[experimental]
autoMemoryReclaim=disabled
sparseVhd=false
```

## GPU and Windows settings [AUTHOR-STATED, not measured by the analyst]
- Overclock in MSI Afterburner: memory +1000 MHz, core about 2790 MHz at 965/970 mV (the owner is not sure of the exact values). **Every speed in this repository was measured on this overclocked card; no stock-clock run exists.** Actual clocks during a run were not recorded.
- Hardware-accelerated GPU scheduling (HAGS): off (turned off when WSL2 was installed).
- NVIDIA Profile Inspector: CUDA Force P2 State off, power management maximum performance, prefer no system-memory fallback.
- Defender excludes the project folder.

## BAR1 readings, Windows-side `nvidia-smi -q -d MEMORY` [MEASURED, pasted]
| Moment | Total | Used | Free |
|---|---|---|---|
| During/after the E0 timeline run (23:13) | 16384 MiB | 16355 MiB | 29 MiB |
| After `wsl --shutdown` (23:27), no WSL process running | 16384 MiB | 16355 MiB | 29 MiB |

The Used figure did not change when WSL was stopped, and GPU memory in use at idle was a few hundred MiB, so the 16355 MiB is not data held by these runs. What holds it is not identified [UNMEASURED]. Windows-side GPU memory by process at 23:28: three processes holding 1835, 123 and 106 MiB.

## Sequential file-read test, `dd bs=16M`, 1 GiB steps, WSL page cache dropped before each run [MEASURED, pasted]
The owner had earlier seen reads over 2 GB start near 2 GB/s and fall toward 200 MB/s. This did not reproduce today. Windows' own file cache state during these reads is unknown, so a cold-Windows-cache case is not excluded.
| File | Size | Speed range per GiB |
|---|---|---|
| WSL disk copy of the Q4_K_M model (first 18 GiB) | 20.2 GiB | 1.5 to 1.6 GB/s, flat |
| Windows-drive copy of the Q4_K_M model (`/mnt/c`) | 20.2 GiB | 0.94 to 1.6 GB/s (dips: 940 MB/s at 8 GiB, 1.1 GB/s at 11 GiB) |
| Windows-drive copy of the Q5_K_M model | 23.6 GiB | 1.1 to 1.6 GB/s, 1.6 GB/s over the last 8 GiB |
