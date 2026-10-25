#!/usr/bin/env python3
# File: <project>\moe_track\pelias_serve\e6_policy_model.py
"""Back-of-the-envelope model for the E6 draft gate: when is a 2-token MTP verification batch worth running, and which min_draft_prob should the gate use?

Everything it prints is an ESTIMATE from the numbers you give it (defaults come from the roadmap's measured N=28 decode step and the guesses marked below). Replace them
with measured values (e0_layer_profiler for the step times, fork/5e_mtp_acceptance.py for acceptance) before believing the answer.

One cycle, starting with a token x that the target has chosen but not yet processed:
    ordinary step :  decode x                                      cost T_step                          emits 1 token
    draft cycle   :  blk.40 drafts d, snapshot the DeltaNet state, verify [x, d] in one batch
        accepted (prob a):  cost T_draft + T_snap + T_verify                                            emits 2 tokens (d and the target's next token)
        rejected (1 - a):   the same, plus restore the state (T_rollback) and decode x again (T_step)    emits 1 token (the target's own token instead of d)
Expected tokens per draft cycle = 1 + a;  expected time = T_draft + T_snap + T_verify + (1 - a) * (T_rollback + T_step).
A draft cycle beats an ordinary step when  a > a* = (C0 + C_r - T_step) / (T_step + C_r)   with C0 = T_draft + T_snap + T_verify, C_r = T_rollback + T_step.

The gate: draft only when the draft's confidence is at least theta (and blk.40's own experts are mostly hot, which is folded into T_draft here). Acceptance probability is
taken from a calibration table  conf -> acceptance  (CSV columns: conf,acceptance,weight). Without one, the model assumes a calibrated drafter: acceptance = conf**gamma
with a made-up confidence distribution; that default only shows the SHAPE of the trade-off.

    python e6_policy_model.py                          defaults
    python e6_policy_model.py --step-ms 24.3 --draft-ms 3 --verify-ms 31 --snap-ms 0.5 --rollback-ms 0.5 --calib calib.csv
    python e6_policy_model.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Costs:
    step_ms: float = 24.33        # one ordinary decode step at N=28 (the roadmap's figure)
    draft_ms: float = 3.0         # blk.40 forward incl. its own experts (GUESS: a fraction of a layer-stack step)
    verify_ms: float = 31.0       # one 2-token verification batch (GUESS: a step plus the extra token's cold experts; measure it)
    snap_ms: float = 0.5          # e5_deltanet_state.cuh snapshot (see estimate_copy_ms: ~0.5 ms, not the 0.095 ms of the draft)
    rollback_ms: float = 0.5      # restore of the same size


def a_star(c: Costs) -> float:
    """Acceptance probability above which a draft cycle yields more tokens per second than an ordinary step. >1 means: never worth it."""
    c0 = c.draft_ms + c.snap_ms + c.verify_ms
    cr = c.rollback_ms + c.step_ms
    return (c0 + cr - c.step_ms) / (c.step_ms + cr)


def cycle_cost_ms(c: Costs, a: float) -> float:
    return c.draft_ms + c.snap_ms + c.verify_ms + (1.0 - a) * (c.rollback_ms + c.step_ms)


def tokens_per_s(c: Costs, table: Sequence[Tuple[float, float, float]], theta: float) -> float:
    """Throughput when the gate drafts for conf >= theta. table rows: (conf, acceptance, weight). theta > max conf -> plain decoding."""
    tot_w = sum(w for _, _, w in table)
    tok = 0.0
    ms = 0.0
    for conf, acc, w in table:
        p = w / tot_w
        if conf >= theta:
            tok += p * (1.0 + acc)
            ms += p * cycle_cost_ms(c, acc)
        else:
            tok += p * 1.0
            ms += p * c.step_ms
    return 1000.0 * tok / ms


def default_table(gamma: float = 1.0, bins: int = 20) -> List[Tuple[float, float, float]]:
    """Made-up confidence distribution (mass towards high confidence, like a decent drafter) with acceptance = conf ** gamma."""
    rows = []
    for i in range(bins):
        conf = (i + 0.5) / bins
        weight = conf ** 1.5 * (1.0 - conf) ** 0.6 + 1e-9
        rows.append((conf, min(1.0, conf ** gamma), weight))
    return rows


def read_table(path: str) -> List[Tuple[float, float, float]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["conf"]), float(r["acceptance"]), float(r.get("weight") or 1.0)))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return sorted(rows)


def sweep(c: Costs, table, thetas: Sequence[float]):
    return [(th, tokens_per_s(c, table, th)) for th in thetas]


def best_theta(c: Costs, table) -> Tuple[float, float]:
    confs = sorted({conf for conf, _, _ in table})
    cands = [0.0] + confs + [max(confs) + 1e-6]
    return max(((th, tokens_per_s(c, table, th)) for th in cands), key=lambda t: t[1])


def selftest() -> int:
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    c = Costs()
    a = a_star(c)
    check("a* is the break-even: at a = a* one draft cycle gives exactly the ordinary rate", abs((1 + a) / cycle_cost_ms(c, a) - 1.0 / c.step_ms) < 1e-9)
    check("above a* a draft cycle is faster, below it slower",
          (1 + a + 0.05) / cycle_cost_ms(c, a + 0.05) > 1.0 / c.step_ms > (1 + a - 0.05) / cycle_cost_ms(c, a - 0.05))
    free = Costs(draft_ms=0, verify_ms=24.33, snap_ms=0, rollback_ms=0)
    check("even with a free draft, free state copies and a verify batch as cheap as one step, a* is 0.5 (a rejected draft costs a re-decode of the accepted token)", abs(a_star(free) - 0.5) < 1e-9)
    check("a very expensive verify batch makes drafting never pay (a* > 1)", a_star(Costs(verify_ms=100.0)) > 1.0)
    tab = default_table()
    plain = tokens_per_s(c, tab, 2.0)
    check("a threshold above every confidence = plain decoding = 1000 / step_ms", abs(plain - 1000.0 / c.step_ms) < 1e-9)
    th, best = best_theta(c, tab)
    check("the best gated rate is never below plain decoding", best >= plain - 1e-12)
    always = tokens_per_s(c, tab, 0.0)
    check("gating beats 'always draft' when the break-even acceptance is above 0 and low-confidence drafts exist", best >= always - 1e-12)
    check("with a calibrated drafter the best threshold sits at the confidence where acceptance crosses a* (within one bin)", abs(th - a_star(c)) <= 0.075 or th > 1.0)
    cheap = Costs(draft_ms=1.0, verify_ms=26.0, snap_ms=0.2, rollback_ms=0.2)
    check("cheaper draft cycles lower the break-even acceptance", a_star(cheap) < a_star(c))
    check("more expensive state copies raise it", a_star(Costs(snap_ms=2.0, rollback_ms=2.0)) > a_star(c))
    sw = sweep(c, tab, [0.0, 0.3, 0.6, 0.9])
    check("sweep returns one row per threshold", len(sw) == 4 and all(v > 0 for _, v in sw))
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "calib.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("conf,acceptance,weight\n0.9,0.95,10\n0.5,0.4,5\n0.2,0.1,5\n")
        t = read_table(p)
        check("calibration CSV is read and sorted", [r[0] for r in t] == [0.2, 0.5, 0.9] and t[2][2] == 10.0)
        th2, b2 = best_theta(c, t)
        check("measured-style table: only the confident bin is worth drafting", th2 == 0.9 or th2 > 0.9)
    print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Costs()
    ap.add_argument("--step-ms", type=float, default=d.step_ms)
    ap.add_argument("--draft-ms", type=float, default=d.draft_ms)
    ap.add_argument("--verify-ms", type=float, default=d.verify_ms)
    ap.add_argument("--snap-ms", type=float, default=d.snap_ms)
    ap.add_argument("--rollback-ms", type=float, default=d.rollback_ms)
    ap.add_argument("--calib", default="", help="CSV with columns conf,acceptance[,weight]")
    ap.add_argument("--gamma", type=float, default=1.0, help="default table only: acceptance = conf ** gamma")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    c = Costs(a.step_ms, a.draft_ms, a.verify_ms, a.snap_ms, a.rollback_ms)
    table = read_table(a.calib) if a.calib else default_table(a.gamma)
    src = a.calib or f"made-up table (acceptance = conf^{a.gamma})"
    print(f"costs (ms): step {c.step_ms}, draft {c.draft_ms}, verify-2 {c.verify_ms}, snapshot {c.snap_ms}, rollback {c.rollback_ms};   table: {src}")
    ast = a_star(c)
    print(f"break-even acceptance a* = {ast:.3f}" + ("   (> 1: drafting cannot pay with these costs)" if ast > 1 else ""))
    print(f"plain decoding: {1000.0 / c.step_ms:.2f} tok/s")
    for th, v in sweep(c, table, [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        print(f"  min_draft_prob {th:.1f}: {v:6.2f} tok/s  ({v / (1000.0 / c.step_ms):.3f}x)")
    th, v = best_theta(c, table)
    print(f"best threshold on this table: {th:.3f} -> {v:.2f} tok/s ({v / (1000.0 / c.step_ms):.3f}x).   ESTIMATE: replace the costs and the table with measured ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
