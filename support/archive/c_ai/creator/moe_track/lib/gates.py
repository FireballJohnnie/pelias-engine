#!/usr/bin/env python3
# File: <project>\moe_track\lib\gates.py
"""Every pass/fail gate of the moe_track plan, in one place, reading the real result files.

    python lib/gates.py list
    python lib/gates.py p1  --baseline A.json --candidate B.json
    python lib/gates.py g0  --point 40=ALLCPU.json --point 28=N28.json --point 34=N34.json [--sizes expert_layer_sizes.json]
    python lib/gates.py g1  [--summary logs/moe_trace/trace_summary.json]
    python lib/gates.py g2  [--sim moe_track/sim/sim_result.json]
    python lib/gates.py g3  --candidate FORK.json [--baseline N28.json]
    python lib/gates.py h6  --kl kl_result.json
    python lib/gates.py h9  [--predictor moe_track/sim/predictor_result.json]
    python lib/gates.py mtp --result mtp_5e.json [--baseline N28.json]
    python lib/gates.py reuse --result prompt_reuse_5d.json

Exit code: 0 = PASS, 1 = FAIL or HOLD, 2 = input problem. A gate prints the numbers it used, so the verdict can be checked.
Speed files are the JSON files written by stuart_server_speed.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import floor_model  # noqa: E402
import paths  # noqa: E402
from speedlog import SpeedLog, SpeedLogError, load_speedlog  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# Thresholds (edit here only). Each comes from the blueprint and its 2026-10-20 review.
# ---------------------------------------------------------------------------------------------------------------
GATES: Dict[str, Dict[str, float]] = {
    "p1": {"improve_pct": 5.0, "regress_pct": 3.0, "vram_gib_max": 11.2},
    "g0": {"tolerance": 0.10},
    "g1": {"min_decode_tokens": 100_000, "gini_min": 0.45},
    "g2": {"hit_pass": 0.75, "hit_stop": 0.65, "budget_gib": 5.6},
    "g3": {"gen_tps_12k_min": 55.0, "prefill_tps_12k_min": 1150.0, "vram_gib_max": 11.2},
    "h6": {"mean_kl_max": 0.005, "top1_agree_min": 0.99},
    "h9": {"miss_reduction_min": 0.20},
    "mtp": {"accept_good": 0.70, "accept_poor": 0.50, "speed_floor_ratio": 1.00},
}
KEEP, HOLD = "PASS", "FAIL"


def _out(name: str, verdict: str, lines: List[str]) -> int:
    print(f"[gate {name}] {verdict}")
    for ln in lines:
        print("   " + ln)
    return 0 if verdict == "PASS" else 1


def _need(log: Optional[float], what: str) -> float:
    if log is None:
        raise SpeedLogError(f"missing {what} in the speed file (was depth 12000 run?)")
    return log


# ---------------------------------------------------------------------------------------------------------------
# P1 : keep a single-change launch experiment?
# ---------------------------------------------------------------------------------------------------------------
def gate_p1(baseline: SpeedLog, cand: SpeedLog) -> int:
    th = GATES["p1"]
    metrics: List[Tuple[str, Optional[float], Optional[float]]] = [
        ("generation, short prompt", baseline.gen(0), cand.gen(0)),
        ("generation at 12k depth", baseline.gen(12000), cand.gen(12000)),
        ("prompt reading at 12k", baseline.prefill(12000), cand.prefill(12000)),
    ]
    lines, improved, regressed = [], [], []
    for name, a, b in metrics:
        if a is None or b is None:
            lines.append(f"{name}: not comparable (missing in one file)")
            continue
        pct = (b / a - 1.0) * 100.0
        lines.append(f"{name}: {a:.2f} -> {b:.2f}  ({pct:+.1f}%)")
        if pct >= th["improve_pct"]:
            improved.append(name)
        if pct <= -th["regress_pct"]:
            regressed.append(name)
    for depth in (0, 12000):
        bd, cd = baseline.depths.get(depth), cand.depths.get(depth)
        if bd and cd and bd.gen_spread is not None and cd.gen_spread is not None:
            lines.append(f"generation spread at depth {depth}: baseline {bd.gen_spread*100:.1f}%, candidate {cd.gen_spread*100:.1f}% (a gain smaller than this is noise)")
    for log, tag in ((baseline, "baseline"), (cand, "candidate")):
        d = log.depths.get(12000)
        if d and d.cache_hits:
            lines.append(f"WARNING {tag}: {d.cache_hits} of {d.n} 12k requests reused cached prompt tokens; prompt speed is not comparable")
    vram = cand.vram_peak_gib
    vram_ok = vram is not None and vram <= th["vram_gib_max"]
    lines.append(f"GPU peak {vram:.2f} GiB (limit {th['vram_gib_max']})" if vram is not None else "GPU peak unknown")
    keep = bool(improved) and not regressed and vram_ok
    lines.append(f"improved by >= {th['improve_pct']}%: {improved or 'none'}; regressed by >= {th['regress_pct']}%: {regressed or 'none'}")
    return _out("P1", "PASS" if keep else "HOLD", lines + [("keep this change" if keep else "do not keep this change (or not proven better)")])


# ---------------------------------------------------------------------------------------------------------------
# G0 : out-of-sample floor model
# ---------------------------------------------------------------------------------------------------------------
def gate_g0(points: Dict[int, SpeedLog], sizes_path: Optional[Path]) -> int:
    th = GATES["g0"]
    if 40 not in points or len(points) < 3:
        print("G0 needs the all-CPU point (40) and at least two other --n-cpu-moe points", file=sys.stderr)
        return 2
    sizes = json.loads(sizes_path.read_text()) if sizes_path and sizes_path.exists() else None
    lb = floor_model.layer_bytes_from_sizes(sizes)
    fit_n = 28 if 28 in points else sorted(n for n in points if n != 40)[0]
    pts = {n: points[n].gen(0) for n in points}
    if any(v is None for v in pts.values()):
        raise SpeedLogError("a speed file has no short-prompt (depth 0) generation speed")
    model = floor_model.fit_two_point(lb, pts[40], pts[fit_n], fit_n)
    lines = [f"fitted on N=40 ({pts[40]:.2f} tok/s) and N={fit_n} ({pts[fit_n]:.2f} tok/s): fixed {model.fixed_ms:.2f} ms/token, CPU expert path {model.b_cpu_gbs:.1f} GB/s",
             ("per-layer sizes from the GGUF audit" if sizes else "NOTE: no expert_layer_sizes.json; assumed 1.9 MB per expert in every layer")]
    ok = True
    tested = 0
    for n in sorted(points):
        if n in (40, fit_n):
            continue
        pred = model.tps_n_cpu_moe(n)
        err = abs(pred - pts[n]) / pts[n]
        tested += 1
        lines.append(f"out-of-sample N={n}: predicted {pred:.2f} tok/s, measured {pts[n]:.2f}, error {err*100:.1f}% (limit {th['tolerance']*100:.0f}%)")
        ok &= err <= th["tolerance"]
    return _out("G0", "PASS" if ok and tested else "FAIL", lines + ["the model may be used to forecast cache and placement options" if ok else "the model is NOT trusted; do not use its forecasts"])


# ---------------------------------------------------------------------------------------------------------------
# G1, G2, H9 : read JSON written by the trace tools
# ---------------------------------------------------------------------------------------------------------------
def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SpeedLogError(f"cannot read {p}: {e}")


def gate_g1(summary_path: Path) -> int:
    th = GATES["g1"]
    s = _load_json(summary_path)
    n, g = int(s.get("n_decode_tokens", 0)), float(s.get("gini_mean", 0.0))
    ok = n >= th["min_decode_tokens"] and g > th["gini_min"]
    return _out("G1", "PASS" if ok else "FAIL", [
        f"decode tokens traced {n:,} (need >= {int(th['min_decode_tokens']):,})",
        f"mean per-layer Gini of expert use {g:.3f} (need > {th['gini_min']}; the published figure for this model is 0.65)",
        f"domains: {s.get('domains', {})}"])


def gate_g2(sim_path: Path) -> int:
    th = GATES["g2"]
    s = _load_json(sim_path)
    hit = s.get("heldout_hit_rate_at_budget")
    if hit is None:
        raise SpeedLogError("sim result has no heldout_hit_rate_at_budget; run sim/waterfill_sim.py with --heldout")
    if hit >= th["hit_pass"]:
        verdict, msg = "PASS", "a static, profile-guided hot bank is worth building or testing in a fork"
    elif hit < th["hit_stop"]:
        verdict, msg = "FAIL", "STOP: a static hot bank cannot reach the target; do not build custom code on this track"
    else:
        verdict, msg = "HOLD", "between the stop and pass lines; only a miss-fetch or prediction path could still close the gap"
    lines = [f"held-out hit rate at {s.get('budget_gib', th['budget_gib'])} GiB: {hit:.3f} (pass >= {th['hit_pass']}, stop < {th['hit_stop']})",
             f"in-sample hit rate {s.get('insample_hit_rate_at_budget', float('nan')):.3f}; gap in-sample vs held-out shows how much the profile overfits",
             msg]
    return _out("G2", verdict if verdict != "HOLD" else "HOLD", lines)


def gate_h9(pred_path: Path) -> int:
    th = GATES["h9"]
    s = _load_json(pred_path)
    red = float(s.get("miss_reduction_vs_static", 0.0))
    ok = red >= th["miss_reduction_min"]
    bank_only = s.get("miss_reduction_vs_bank_only")
    lines = [f"miss reduction from PREDICTING the next layer's experts: {red*100:.1f}% (need >= {th['miss_reduction_min']*100:.0f}%), "
             f"measured against a static system that spends the same {s.get('prefetch_slots', '?')} staging slots on profile-ranked experts",
             f"against the bank alone it would read {float(bank_only)*100:.1f}%, but blind fetching earns part of that without any prediction" if bank_only is not None else "bank-alone figure not in the file",
             f"predictor top-K recall {s.get('recall_at_k', float('nan')):.3f}, lead time of one layer"]
    if s.get("slots_that_fit_in_one_layer_window") is not None:
        lines.append(f"slots that fit in one layer's PCIe window: {s['slots_that_fit_in_one_layer_window']} (the result is only usable up to that many)")
    return _out("H9", "PASS" if ok else "FAIL", lines)


# ---------------------------------------------------------------------------------------------------------------
# G3 : adopt a fork / build?
# ---------------------------------------------------------------------------------------------------------------
def gate_g3(cand: SpeedLog, baseline: Optional[SpeedLog]) -> int:
    th = GATES["g3"]
    gen12 = _need(cand.gen(12000), "generation speed at 12k")
    pre12 = _need(cand.prefill(12000), "prompt speed at 12k")
    vram = cand.vram_peak_gib
    ok_g, ok_p = gen12 >= th["gen_tps_12k_min"], pre12 >= th["prefill_tps_12k_min"]
    ok_v = vram is not None and vram <= th["vram_gib_max"]
    lines = [f"generation at 12k {gen12:.2f} tok/s (need >= {th['gen_tps_12k_min']})",
             f"prompt reading at 12k {pre12:.0f} tok/s (need >= {th['prefill_tps_12k_min']:.0f})",
             f"GPU peak {vram:.2f} GiB (limit {th['vram_gib_max']})" if vram is not None else "GPU peak unknown"]
    if baseline:
        b = baseline.gen(12000)
        if b:
            lines.append(f"vs baseline {baseline.label}: {gen12 / b:.2f}x generation at 12k")
    lines.append("accuracy is a separate gate: run the persona probe, capability suite and sim/kl_gate.py on the candidate before adopting it")
    return _out("G3", "PASS" if (ok_g and ok_p and ok_v) else "FAIL", lines)


# ---------------------------------------------------------------------------------------------------------------
# H6 : logit agreement ; MTP (5e) ; prompt reuse (5d)
# ---------------------------------------------------------------------------------------------------------------
def gate_h6(kl_path: Path) -> int:
    th = GATES["h6"]
    s = _load_json(kl_path)
    kl, agree = float(s["mean_kl"]), float(s["top1_agreement"])
    ok = kl <= th["mean_kl_max"] and agree >= th["top1_agree_min"]
    return _out("H6", "PASS" if ok else "FAIL", [
        f"mean KL divergence {kl:.6f} nats (limit {th['mean_kl_max']}), top-1 agreement {agree*100:.2f}% (need >= {th['top1_agree_min']*100:.0f}%)",
        f"scored on {s.get('n_tokens', '?')} tokens; reference = {s.get('reference', '?')}, candidate = {s.get('candidate', '?')}",
        "limits are for placement-only changes (same weights); a re-quantized or retrained file is judged against the Q8_0 reference instead"])


def gate_mtp(result_path: Path, baseline: Optional[SpeedLog]) -> int:
    th = GATES["mtp"]
    s = _load_json(result_path)
    acc, tps = float(s["acceptance"]), float(s["gen_tps_median"])
    base = float(s["baseline_gen_tps_median"]) if s.get("baseline_gen_tps_median") else (baseline.gen(0) if baseline else None)
    lines = [f"draft acceptance {acc*100:.1f}% ({s.get('accepted')}/{s.get('drafted')} drafted tokens)", f"generation {tps:.2f} tok/s" + (f" vs baseline {base:.2f} ({tps/base:.2f}x)" if base else "")]
    faster = base is not None and tps >= base * th["speed_floor_ratio"]
    if acc >= th["accept_good"] and faster:
        v, msg = "PASS", "MTP is worth designing around"
    elif acc < th["accept_poor"]:
        v, msg = "FAIL", "low acceptance: find out why (recurrent-state rollback vs head misalignment) before building on MTP"
    else:
        v, msg = "HOLD", "acceptable but not a clear win; repeat at the long context before deciding"
    return _out("MTP", v, lines + [msg])


def gate_reuse(result_path: Path) -> int:
    s = _load_json(result_path)
    full = bool(s.get("full_reprocess_second_request"))
    lines = [f"second request processed {s.get('second_prompt_n')} of {s.get('second_total_tokens')} prompt tokens (cache reused: {s.get('second_cache_n')})",
             f"first request {s.get('first_prompt_ms'):.0f} ms, second {s.get('second_prompt_ms'):.0f} ms"]
    if full:
        lines.append("RESULT: the server re-reads the whole prompt on a changed last line. Every multi-turn request pays the full prompt-reading time.")
    else:
        lines.append("RESULT: prefix reuse works; multi-turn use does not pay a full re-read.")
    return _out("5d", "PASS" if not full else "HOLD", lines)


# ---------------------------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("p1"); p.add_argument("--baseline", required=True); p.add_argument("--candidate", required=True)
    p = sub.add_parser("g0"); p.add_argument("--point", action="append", required=True, help="N=FILE (40 = all experts on CPU)")
    p.add_argument("--sizes", default=str(paths.FLOORS / "expert_layer_sizes.json"))
    p = sub.add_parser("g1"); p.add_argument("--summary", default=str(paths.MOE_TRACE_DIR / "trace_summary.json"))
    p = sub.add_parser("g2"); p.add_argument("--sim", default=str(paths.SIM_DIR / "sim_result.json"))
    p = sub.add_parser("g3"); p.add_argument("--candidate", required=True); p.add_argument("--baseline")
    p = sub.add_parser("h6"); p.add_argument("--kl", required=True)
    p = sub.add_parser("h9"); p.add_argument("--predictor", default=str(paths.SIM_DIR / "predictor_result.json"))
    p = sub.add_parser("mtp"); p.add_argument("--result", required=True); p.add_argument("--baseline")
    p = sub.add_parser("reuse"); p.add_argument("--result", required=True)
    a = ap.parse_args(argv)

    try:
        if a.cmd == "list":
            print(json.dumps(GATES, indent=2))
            return 0
        if a.cmd == "p1":
            return gate_p1(load_speedlog(a.baseline), load_speedlog(a.candidate))
        if a.cmd == "g0":
            pts: Dict[int, SpeedLog] = {}
            for s in a.point:
                n, _, f = s.partition("=")
                pts[int(n)] = load_speedlog(f)
            return gate_g0(pts, Path(a.sizes))
        if a.cmd == "g1":
            return gate_g1(Path(a.summary))
        if a.cmd == "g2":
            return gate_g2(Path(a.sim))
        if a.cmd == "g3":
            return gate_g3(load_speedlog(a.candidate), load_speedlog(a.baseline) if a.baseline else None)
        if a.cmd == "h6":
            return gate_h6(Path(a.kl))
        if a.cmd == "h9":
            return gate_h9(Path(a.predictor))
        if a.cmd == "mtp":
            return gate_mtp(Path(a.result), load_speedlog(a.baseline) if a.baseline else None)
        if a.cmd == "reuse":
            return gate_reuse(Path(a.result))
    except (SpeedLogError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
