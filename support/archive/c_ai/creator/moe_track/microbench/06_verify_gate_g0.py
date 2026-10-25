#!/usr/bin/env python3
# File: <project>\moe_track\microbench\06_verify_gate_g0.py
"""Gate G0 (is the floor model trustworthy?) plus the numbers that explain it.

    python microbench/06_verify_gate_g0.py --point 40=ALLCPU.json --point 28=N28.json --point 34=N34.json

Reads only. It writes one file, logs/floors/g0_result.json, next to the other floor measurements.

What it does
  1. Fits the token-time model on N=40 and N=28 and PREDICTS the other points (gate G0, tolerance 10%).
  2. Compares the fitted CPU expert-path bandwidth with the RAM micro-benchmark (02), if that file exists:
     a big gap means llama.cpp's CPU path is below what this memory can deliver (headroom for thread/page tuning).
  3. Compares the fixed per-token cost with the measured GPU<->CPU handoff time (03), if that file exists.
  4. Prints a forecast table: tokens/s for other --n-cpu-moe values and for cache hit rates, from the fitted model.
The forecast is only worth reading when G0 passes.

Exit code: 0 PASS, 1 FAIL, 2 input problem (same as lib/gates.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import floor_model  # noqa: E402
import gates  # noqa: E402
import paths  # noqa: E402
from speedlog import SpeedLog, SpeedLogError, load_speedlog  # noqa: E402


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", action="append", required=True, help="N=FILE  (40 = all experts on CPU); at least three points")
    ap.add_argument("--sizes", default=str(paths.FLOORS / "expert_layer_sizes.json"))
    ap.add_argument("--ram", default=str(paths.FLOORS / "ram_bw_floor.json"))
    ap.add_argument("--handoff", default=str(paths.FLOORS / "handoff_latency.json"))
    ap.add_argument("--dual", default=str(paths.FLOORS / "dual_bus_contention.json"))
    ap.add_argument("--out", default=str(paths.FLOORS / "g0_result.json"))
    a = ap.parse_args(argv)

    points: Dict[int, SpeedLog] = {}
    try:
        for s in a.point:
            n, sep, f = s.partition("=")
            if not sep:
                print(f"--point needs N=FILE, got '{s}'", file=sys.stderr)
                return 2
            points[int(n)] = load_speedlog(f)
    except (SpeedLogError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    sizes_path = Path(a.sizes)
    try:
        rc = gates.gate_g0(points, sizes_path)
    except SpeedLogError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if rc == 2:
        return 2

    # ---- same fit again, to report numbers --------------------------------------------------------------------
    sizes = _read_json(sizes_path)
    lb = floor_model.layer_bytes_from_sizes(sizes)
    fit_n = 28 if 28 in points else sorted(n for n in points if n != 40)[0]
    tps = {n: points[n].gen(0) for n in points}
    model = floor_model.fit_two_point(lb, tps[40], tps[fit_n], fit_n)
    desc = floor_model.describe(model)

    result: dict = {
        "gate": "G0",
        "pass": rc == 0,
        "fitted_on": [40, fit_n],
        "measured_tps": {str(n): round(v, 3) for n, v in sorted(tps.items())},
        "predicted_tps": {str(n): round(model.tps_n_cpu_moe(n), 3) for n in sorted(points)},
        "model": desc,
        "sizes_from_gguf_audit": bool(sizes),
    }

    print()
    print("what the fit says")
    print(f"   fixed cost per token (not expert placement): {model.fixed_ms:.2f} ms")
    print(f"   CPU expert path in llama.cpp:                {model.b_cpu_gbs:.1f} GB/s")
    print(f"   all-GPU ceiling of this model:               {desc['tps_all_gpu_ceiling']} tok/s (experts at {model.b_gpu_gbs:.0f} GB/s; they do not all fit in 12 GB)")

    ram = _read_json(Path(a.ram))
    if ram and ram.get("best_expert_gb_s"):
        best = float(ram["best_expert_gb_s"])
        ratio = model.b_cpu_gbs / best
        result["ram_best_expert_gb_s"] = best
        result["fitted_over_ram_expert"] = round(ratio, 3)
        print(f"   RAM micro-benchmark, expert-shaped reads:    {best:.1f} GB/s  -> llama.cpp's path reaches {ratio*100:.0f}% of it")
        if ratio < 0.75:
            print("      llama.cpp's CPU path is well below what this memory can deliver: thread count / page size tuning (Phase 1) may pay.")
        else:
            print("      llama.cpp's CPU path is already near the memory limit: the CPU side has little left to give.")
    else:
        print(f"   (no RAM micro-benchmark at {a.ram}; run microbench/bin/02_ram_bw_test with the server stopped)")

    ho = _read_json(Path(a.handoff))
    if ho and ho.get("handoff_us_per_layer"):
        per_tok = float(ho.get("handoff_ms_per_token_40_layers") or ho["handoff_us_per_layer"] * 40 / 1000.0)
        result["handoff_ms_per_token_40_layers"] = per_tok
        print(f"   GPU<->CPU handoff, {ho['handoff_us_per_layer']:.0f} us per layer -> {per_tok:.2f} ms per token if every layer hands off"
              f" ({per_tok / model.fixed_ms * 100:.0f}% of the fixed cost)")
    else:
        print(f"   (no handoff measurement at {a.handoff}; run microbench/bin/03_handoff_latency)")

    du = _read_json(Path(a.dual))
    if du and du.get("combined_vs_cpu_ratio"):
        result["dual_bus_combined_vs_cpu"] = du["combined_vs_cpu_ratio"]
        print(f"   CPU + GPU reading RAM together:              {du['combined_vs_cpu_ratio']:.2f}x the CPU alone"
              f" (CPU keeps {du.get('concurrent_cpu_plus_h2d', {}).get('cpu_keeps_fraction', float('nan'))*100:.0f}% while copies run)")

    print()
    print("forecast from the fitted model (trust it only if G0 passed)")
    print("   --n-cpu-moe   tok/s   experts on GPU")
    fc_n = {}
    n_layers = len(lb)
    n_exp = int((sizes or {}).get("n_experts_per_layer") or 256)
    topk = int((sizes or {}).get("n_experts_used_per_token") or 8)
    for n in (40, 36, 34, 32, 30, 28, 26, 24, 22, 20):
        t = model.tps_n_cpu_moe(n)
        fc_n[str(n)] = round(t, 2)
        gib = sum(lb[n:]) / topk * n_exp / 2**30      # every expert of the GPU layers (lb is top-k experts' bytes)
        print(f"   {n:>10}   {t:5.1f}   {gib:5.2f} GiB (of ~5.6 GiB budget)")
    result["forecast_n_cpu_moe"] = fc_n
    print("   hit rate    tok/s (misses computed on the CPU path)")
    fc_h = {}
    for h in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95):
        t = model.tps_hit_rate(h)
        fc_h[f"{h:.2f}"] = round(t, 2)
        print(f"   {h:8.2f}   {t:5.1f}")
    result["forecast_hit_rate_cpu_miss"] = fc_h
    result["n_layers_in_model"] = n_layers

    try:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    except OSError as e:
        print(f"\ncould not write {a.out}: {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
