#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_sim.py
"""Offline tests for sim/waterfill_sim.py on synthetic traces and tiny hand-checkable cases."""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
sys.path.insert(0, str(HERE.parent / "trace"))
sys.path.insert(0, str(HERE.parent / "lib"))
import summarize_traces  # noqa: E402
import waterfill_sim as ws  # noqa: E402
from synth_trace import make_trace  # noqa: E402


def ref_lru_global(experts, size, budget):
    """Plain-list reference for lru_global: hits are refreshed first, then misses are inserted, then evict from the front."""
    T, L, K = experts.shape
    pool = []   # list of keys, oldest first
    used = 0.0
    hit = np.zeros((T, L, K), bool)
    for t in range(T):
        for l in range(L):
            keys = [(l, int(e)) for e in experts[t, l]]
            hs = [k in pool for k in keys]
            for k, h in zip(keys, hs):
                if h:
                    pool.remove(k)
                    pool.append(k)
            for j, (k, h) in enumerate(zip(keys, hs)):
                hit[t, l, j] = h
                if not h:
                    pool.append(k)
                    used += size[l]
            while used > budget and pool:
                old = pool.pop(0)
                used -= size[old[0]]
    return hit


def main() -> int:
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    # --- 1. greedy bank vs exact knapsack
    rng = np.random.default_rng(1)
    def exact(freq, size, budget):
        items = [(freq[l, e] * size[l], int(size[l])) for l in range(freq.shape[0]) for e in range(freq.shape[1]) if freq[l, e] > 0]
        W = int(budget)
        dp = [0.0] * (W + 1)
        for v, w in items:
            for c in range(W, w - 1, -1):
                dp[c] = max(dp[c], dp[c - w] + v)
        return dp[W]

    worst_small = 1.0
    for trial in range(40):          # tiny, awkward instances: sizes 2..5 against budgets of a few items (greedy is only guaranteed to be decent here)
        freq = rng.integers(0, 50, size=(3, 5)).astype(float)
        size = rng.integers(2, 6, size=3).astype(float)
        budget = float(rng.integers(8, 30))
        mask = ws.select_static(freq, size, budget)
        got = float((freq * mask * size[:, None]).sum())
        best = exact(freq, size, budget)
        if float((mask.sum(axis=1) * size).sum()) > budget + 1e-9:
            check(False, "bank exceeded the budget")
        worst_small = min(worst_small, got / best if best else 1.0)
    check(worst_small >= 0.8, f"awkward tiny instances: greedy is at worst {worst_small:.2f} of the exact optimum (>= 0.80)")
    worst_real = 1.0
    for trial in range(20):          # realistic: many experts, sizes within 15% of each other (Q4_K vs Q6_K down projections), generous budget
        freq = np.floor(rng.pareto(1.2, size=(4, 24)) * 20)
        size = np.array([17.0, 20.0, 17.0, 20.0])
        budget = float(rng.integers(200, 500))
        mask = ws.select_static(freq, size, budget)
        got = float((freq * mask * size[:, None]).sum())
        worst_real = min(worst_real, got / exact(freq, size, budget))
    check(worst_real >= 0.985, f"realistic instances: greedy is at worst {worst_real:.4f} of the exact optimum (>= 0.985)")
    freq = rng.integers(1, 99, size=(3, 6)).astype(float)
    size = np.full(3, 1.0)
    m = ws.select_static(freq, size, 7.0)
    top7 = set(np.argsort(-freq.ravel(), kind="stable")[:7].tolist())
    check(set(np.flatnonzero(m.ravel()).tolist()) == top7, "equal sizes: bank = the 7 most-used experts overall (global water level, not per layer)")

    # --- 2. byte-weighted hit rate by hand
    experts = np.array([[[0, 1], [0, 1]], [[0, 2], [3, 1]]], np.uint16)       # T=2, L=2, K=2
    mask = np.zeros((2, 4), bool); mask[0, 0] = mask[1, 1] = True
    size = np.array([10.0, 30.0])
    sc = ws.score_bank(mask, experts, size, 4)
    # hits: t0 l0: [1,0] ; t0 l1: [0,1] ; t1 l0: [1,0] ; t1 l1: [0,1]  -> bytes hit = 10+30+10+30 = 80 of (2*10*2 + 2*30*2)=160
    check(abs(sc["hit_rate_bytes"] - 0.5) < 1e-9 and abs(sc["hit_rate_count"] - 0.5) < 1e-9, "hand-computed hit rates")
    mask2 = np.zeros((2, 4), bool); mask2[1, :] = True
    sc2 = ws.score_bank(mask2, experts, size, 4)
    check(abs(sc2["hit_rate_bytes"] - 120.0 / 160.0) < 1e-9 and abs(sc2["hit_rate_count"] - 0.5) < 1e-9, "byte-weighted differs from count when sizes differ (all of the big layer in the bank)")

    # --- 3. LRU vs reference
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tr = make_trace(d / "a", n_layers=3, n_expert=12, k=3, seqs_per_domain=3, tokens_per_seq=40)
        ex = tr.decode.experts[:120]
        size = np.array([1.0, 2.0, 1.5])
        for budget in (6.0, 20.0, 60.0):
            a = ws.lru_global(ex, size, budget)
            b = ref_lru_global(ex, size, budget)
            if not np.array_equal(a, b):
                check(False, f"LRU global differs from reference at budget {budget}")
                break
        else:
            check(True, "global LRU equals the plain-list reference at three budgets")
        # per-layer LRU: capacity covers all experts -> after warm-up everything hits; capacity 0 -> nothing hits
        full = ws.lru_per_layer(ex, np.array([12, 12, 12]))
        none = ws.lru_per_layer(ex, np.array([0, 0, 0]))
        check(none.sum() == 0 and full[60 - 20:].mean() > 0.9 if len(full) > 60 else none.sum() == 0, "per-layer LRU: zero slots never hit, full slots almost always hit after warm-up")

        # --- 4. full run on a structured synthetic trace
        tr = make_trace(d / "big", n_layers=6, n_expert=32, k=4, seqs_per_domain=24, tokens_per_seq=150)
        sizes = {"layers": [{"layer": l, "role": "spine", "expert_bytes": 1_000_000 + 100_000 * (l % 2)} for l in range(6)]}
        total_gib = sum(1_000_000 + 100_000 * (l % 2) for l in range(6)) * 32 / 2 ** 30
        res = ws.run(tr, sizes, budget_gib=total_gib * 0.3, budgets=[total_gib * f for f in (0.1, 0.2, 0.3, 0.5, 1.0)], g0=None, do_lru=True)
        held = [c["heldout"] for c in res["curve"]]
        check(all(b >= a - 1e-9 for a, b in zip(held, held[1:])), f"held-out hit rate never decreases with budget: {held}")
        check(abs(held[-1] - 1.0) < 1e-3 or held[-1] > 0.97, "budget >= all experts -> hit rate ~1 (experts never seen in the fit set are the only misses)")
        check(res["insample_hit_rate_at_budget"] >= res["heldout_hit_rate_at_budget"] - 0.03, f"in-sample {res['insample_hit_rate_at_budget']:.3f} >= held-out {res['heldout_hit_rate_at_budget']:.3f} (minus noise)")
        check(res["heldout_hit_rate_at_budget"] > 0.45, f"skewed synthetic routing at 30% budget gives a clearly better than uniform (0.30) hit rate: {res['heldout_hit_rate_at_budget']:.3f}")
        check(res["bank_gib_used"] <= total_gib * 0.3 + 1e-9, "bank respects the byte budget")
        check(abs(sum(res["heldout_detail"]["miss_per_step_hist"]) - 1.0) < 1e-3, "miss histogram sums to 1")
        check(res["heldout_dynamic"]["lru_global_pool"] > 0.0 and res["heldout_dynamic"]["hybrid_half_static_half_lru"] > 0.0, "dynamic caches produce hit rates")
        check(set(res["per_domain_heldout"]) == {"code", "persona"}, "per-domain table covers both domains")

        # uniform routing: hit rate tracks the budget fraction
        tru = make_trace(d / "uni", n_layers=4, n_expert=32, k=4, seqs_per_domain=20, tokens_per_seq=150, zipf=0.0, structure=False)
        ru = ws.run(tru, None, budget_gib=(1.9e6 * 4 * 32 * 0.25) / 2 ** 30, budgets=[1.0], g0=None, do_lru=False)
        check(abs(ru["heldout_hit_rate_at_budget"] - 0.25) < 0.06, f"uniform routing, 25% budget -> hit ~0.25: {ru['heldout_hit_rate_at_budget']:.3f}")

        # --- 5. command line and gate G2
        out = d / "sim_result.json"
        only = d / "only_big"
        only.mkdir()
        for suffix in (".pltrace", ".meta.jsonl", ".done"):
            (only / ("big" + suffix)).write_bytes((d / ("big" + suffix)).read_bytes())
        env_args = [sys.executable, str(HERE.parent / "sim" / "waterfill_sim.py"), "--trace-dir", str(only), "--out", str(out), "--sizes", str(d / "nosizes.json"),
                    "--g0", str(d / "nog0.json"), "--budget-gib", "0.02", "--budgets", "0.01,0.02", "--no-lru"]
        p = subprocess.run(env_args, capture_output=True, text=True)
        check(p.returncode == 0 and out.exists(), "command line run writes sim_result.json" + ("" if p.returncode == 0 else "\n" + p.stderr[-500:]))
        if out.exists():
            r = json.loads(out.read_text())
            check("heldout_hit_rate_at_budget" in r and "insample_hit_rate_at_budget" in r, "result has the keys gate G2 reads")
            g = subprocess.run([sys.executable, str(HERE.parent / "lib" / "gates.py"), "g2", "--sim", str(out)], capture_output=True, text=True)
            check(g.returncode in (0, 1) and "[gate G2]" in g.stdout, "gate G2 accepts the file")
    print("FAILURES: %d" % fails if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
