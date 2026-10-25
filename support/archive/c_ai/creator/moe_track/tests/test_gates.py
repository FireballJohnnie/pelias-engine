#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_gates.py
"""Offline tests for lib/speedlog.py, lib/floor_model.py and lib/gates.py.

    python tests/test_gates.py [--real-speed PATH_TO_20261020_032031_q4km-ncmoe28-ctx131k-ub2048.json]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import floor_model  # noqa: E402
import gates  # noqa: E402
from speedlog import load_speedlog  # noqa: E402


def make_speed(path: Path, label: str, gen0: float, gen12: float, pre12: float, vram_peak: float = 10600, cache_n: int = 0) -> Path:
    rows = []
    for depth, g, p in ((0, gen0, 80.0), (4000, (gen0 + gen12) / 2, pre12 * 0.98), (12000, gen12, pre12)):
        for rep, jit in enumerate((0.995, 1.0, 1.005), 1):
            rows.append({"depth": depth, "rep": rep, "prompt_n": 61 if depth == 0 else depth + 14, "cache_n": cache_n,
                         "prompt_tps": p * jit, "gen_tps": g * jit, "ttft": 1.0, "timed": True})
    path.write_text(json.dumps({"meta": {"label": label, "n_ctx": 131072, "vram": {"idle_mib": 10485, "peak_mib": vram_peak, "total_mib": 12282}}, "rows": rows}))
    return path


def run(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-speed", default="")
    args = ap.parse_args()
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("PASS  " if cond else "FAIL  ") + msg)
        fails += 0 if cond else 1

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        if args.real_speed and Path(args.real_speed).exists():
            real = load_speedlog(args.real_speed)
            check(abs(real.gen(0) - 41.14) < 0.1 and abs(real.prefill(12000) - 1192.5) < 0.5, f"real N=28 file parsed: gen0 {real.gen(0):.2f}, prefill12k {real.prefill(12000):.1f}")
            check(abs(real.vram_peak_gib - 10600 / 1024) < 1e-9, "real file VRAM peak read from meta.vram.peak_mib")
            better = make_speed(td / "better.json", "better", real.gen(0), real.gen(12000), real.prefill(12000) * 1.4)
            rc, out = run(gates.gate_p1, real, load_speedlog(better))
            check(rc == 0, "P1 keeps a candidate with +40% prefill and unchanged decode")
            worse = make_speed(td / "worse.json", "worse", real.gen(0) * 0.95, real.gen(12000), real.prefill(12000) * 1.4)
            rc, out = run(gates.gate_p1, real, load_speedlog(worse))
            check(rc == 1 and "regressed" in out, "P1 holds when prefill +40% but decode -5%")

        base = make_speed(td / "base.json", "base", 41.1, 38.8, 1192.0)
        same = make_speed(td / "same.json", "same", 41.2, 38.9, 1195.0)
        rc, _ = run(gates.gate_p1, load_speedlog(base), load_speedlog(same))
        check(rc == 1, "P1 holds an unchanged candidate (inside the 5% improvement threshold)")
        big_vram = make_speed(td / "vram.json", "vram", 41.1, 38.8, 1500.0, vram_peak=12000)
        rc, _ = run(gates.gate_p1, load_speedlog(base), load_speedlog(big_vram))
        check(rc == 1, "P1 holds a faster candidate whose GPU peak is above 11.2 GiB")
        cached = make_speed(td / "cached.json", "cached", 41.1, 38.8, 3000.0, cache_n=5000)
        rc, out = run(gates.gate_p1, load_speedlog(base), load_speedlog(cached))
        check("reused cached prompt tokens" in out, "P1 warns when prompt speed comes from cached tokens")

        # floor model: recover parameters, then G0 out-of-sample
        lb = floor_model.layer_bytes_from_sizes(None)
        truth = floor_model.FloorModel(fixed_ms=7.7, b_cpu_gbs=25.3, b_gpu_gbs=504.0, layer_bytes=lb)
        m = floor_model.fit_two_point(lb, truth.tps_n_cpu_moe(40), truth.tps_n_cpu_moe(28), 28)
        check(abs(m.fixed_ms - 7.7) < 1e-6 and abs(m.b_cpu_gbs - 25.3) < 1e-6, f"two-point fit recovers F={m.fixed_ms:.3f} ms, B_cpu={m.b_cpu_gbs:.2f} GB/s")
        f40 = make_speed(td / "n40.json", "n40", truth.tps_n_cpu_moe(40), truth.tps_n_cpu_moe(40), 900.0)
        f28 = make_speed(td / "n28.json", "n28", truth.tps_n_cpu_moe(28), truth.tps_n_cpu_moe(28), 1190.0)
        f34 = make_speed(td / "n34.json", "n34", truth.tps_n_cpu_moe(34), truth.tps_n_cpu_moe(34), 1000.0)
        pts = {40: load_speedlog(f40), 28: load_speedlog(f28), 34: load_speedlog(f34)}
        rc, out = run(gates.gate_g0, pts, None)
        check(rc == 0, "G0 passes when the third point lies on the fitted line")
        f34bad = make_speed(td / "n34b.json", "n34b", truth.tps_n_cpu_moe(34) * 0.8, 1.0, 1000.0)
        rc, out = run(gates.gate_g0, {40: pts[40], 28: pts[28], 34: load_speedlog(f34bad)}, None)
        check(rc == 1, "G0 fails when the third point is 20% below the prediction")
        rc, out = run(gates.gate_g0, {40: pts[40], 28: pts[28]}, None)
        check(rc == 2, "G0 refuses to run with only two points (it could not fail)")
        # hit-rate formula against hand arithmetic: all bytes 0.608 GB, h=0.79, CPU-miss
        total = sum(lb)
        hand = 7.7 + total * (0.79 / 504e9 + 0.21 / 25.3e9) * 1e3
        check(abs(truth.t_ms_hit_rate(0.79) - hand) < 1e-9, f"hit-rate token time {truth.t_ms_hit_rate(0.79):.2f} ms equals hand arithmetic")

        # G1 / G2 / G3 / H6 / H9 / MTP / 5d on small result files
        (td / "g1.json").write_text(json.dumps({"n_decode_tokens": 120000, "gini_mean": 0.62, "domains": {"chat": 1}}))
        check(run(gates.gate_g1, td / "g1.json")[0] == 0, "G1 passes 120k tokens at Gini 0.62")
        (td / "g1b.json").write_text(json.dumps({"n_decode_tokens": 90000, "gini_mean": 0.62}))
        check(run(gates.gate_g1, td / "g1b.json")[0] == 1, "G1 fails 90k tokens")
        for hit, want in ((0.80, 0), (0.70, 1), (0.60, 1)):
            (td / "g2.json").write_text(json.dumps({"heldout_hit_rate_at_budget": hit, "insample_hit_rate_at_budget": 0.9}))
            check(run(gates.gate_g2, td / "g2.json")[0] == want, f"G2 held-out {hit} -> exit {want}")
        fast = make_speed(td / "fast.json", "fast", 70.0, 60.0, 1300.0)
        check(run(gates.gate_g3, load_speedlog(fast), None)[0] == 0, "G3 passes 60 tok/s at 12k with 1300 tok/s prefill")
        slow_pre = make_speed(td / "slowpre.json", "slowpre", 70.0, 60.0, 900.0)
        check(run(gates.gate_g3, load_speedlog(slow_pre), None)[0] == 1, "G3 fails when prefill drops below 1150")
        (td / "h6.json").write_text(json.dumps({"mean_kl": 0.0004, "top1_agreement": 0.998, "n_tokens": 20000}))
        check(run(gates.gate_h6, td / "h6.json")[0] == 0, "H6 passes tiny KL")
        (td / "h6b.json").write_text(json.dumps({"mean_kl": 0.03, "top1_agreement": 0.95}))
        check(run(gates.gate_h6, td / "h6b.json")[0] == 1, "H6 fails KL 0.03")
        (td / "h9.json").write_text(json.dumps({"miss_reduction_vs_static": 0.24, "recall_at_k": 0.8}))
        check(run(gates.gate_h9, td / "h9.json")[0] == 0, "H9 passes 24% miss reduction")
        (td / "mtp.json").write_text(json.dumps({"acceptance": 0.78, "accepted": 78, "drafted": 100, "gen_tps_median": 50.0, "baseline_gen_tps_median": 41.1}))
        check(run(gates.gate_mtp, td / "mtp.json", None)[0] == 0, "MTP passes 78% acceptance and faster than baseline")
        (td / "mtp2.json").write_text(json.dumps({"acceptance": 0.36, "accepted": 36, "drafted": 100, "gen_tps_median": 30.0, "baseline_gen_tps_median": 41.1}))
        check(run(gates.gate_mtp, td / "mtp2.json", None)[0] == 1, "MTP fails 36% acceptance")
        (td / "reuse.json").write_text(json.dumps({"full_reprocess_second_request": True, "second_prompt_n": 12000, "second_total_tokens": 12010, "second_cache_n": 0, "first_prompt_ms": 10000.0, "second_prompt_ms": 10100.0}))
        check(run(gates.gate_reuse, td / "reuse.json")[0] == 1, "5d reports a full re-read as HOLD")

    print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
