#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_predictor.py
"""Offline tests for sim/predictor_eval.py: it must find real structure, find none in structureless data, and never look ahead."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for sub in ("tests", "sim", "trace", "lib"):
    sys.path.insert(0, str(HERE.parent / sub))
import predictor_eval as pe  # noqa: E402
import pltrace  # noqa: E402
from synth_trace import make_trace  # noqa: E402


def chain_trace(path: Path, n_layers=6, n_expert=32, k=4, seqs=40, toks=100, seed=5):
    """Layer l+1's picks are a fixed function of layer l's picks (shift by 3): perfectly predictable from the previous layer; layer 0 is random."""
    rng = np.random.default_rng(seed)
    blocks, meta = [], []
    for s in range(seqs):
        for t in range(toks):
            row = np.zeros((n_layers, k), np.uint16)
            row[0] = rng.choice(n_expert, k, replace=False)
            for l in range(1, n_layers):
                row[l] = (row[l - 1].astype(int) + 3) % n_expert
            blocks.append({"kind": 1, "seq": s, "pos0": t, "tokens": [0], "experts": row[None]})
        meta.append({"seq": s, "id": f"chain-{s:04d}", "domain": "chain", "prompt_tokens": 5, "generated_tokens": toks, "finish": "length"})
    hdr = {"arch": "synthetic", "n_layers": n_layers, "moe_layers": list(range(n_layers)), "n_expert": n_expert, "n_expert_used": k, "weights": False}
    p = pltrace.write_trace(path, hdr, blocks, meta)
    p.with_suffix(".done").write_text("x\n")
    return pltrace.read_trace(p)


def main() -> int:
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    # --- temporal scores: own picks excluded, decay applied, resets at a new sequence
    pres = np.zeros((5, 4), np.float32)
    pres[0, 1] = 1; pres[1, 2] = 1; pres[3, 0] = 1
    seq = np.array([0, 0, 0, 1, 1])
    sc = pe.temporal_scores(pres, seq)
    check(sc[0].sum() == 0 and sc[1, 1] == 1.0 and abs(sc[2, 1] - pe.DECAY) < 1e-6 and sc[2, 2] == 1.0, "temporal score: nothing from the token itself, earlier picks decayed")
    check(sc[3].sum() == 0 and sc[4, 0] == 1.0 and sc[4, 1] == 0.0, "temporal score resets at a new sequence")

    # --- co-occurrence fit
    a = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    b = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    cn = pe.fit_crosslayer(a, b)
    check(np.allclose(cn[0], [0, 1, 0]) and np.allclose(cn[1], [0, 0, 1]), "cross-layer table: P(next | previous) rows sum to 1 and follow the data")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        budget = 1.9e6 * 6 * 32 * 0.2 / 2 ** 30
        # --- perfectly chained layers: prediction must add a lot beyond the profile-ranked prefetch
        tc = chain_trace(d / "chain")
        r = pe.run(tc, None, budget, [2, 4, 8, 16], 4)
        check(r["miss_reduction_vs_static"] > 0.5, f"chained layers (layer l+1 = f(layer l)): miss reduction vs profile-ranked prefetch {r['miss_reduction_vs_static']:.2f} > 0.5")
        check(r["miss_reduction_by_predictor"]["crosslayer"]["4"] > r["miss_reduction_by_predictor"]["temporal"]["4"], "cross-layer predictor beats temporal on chained data")
        # --- structureless routing: nothing to find
        tu = make_trace(d / "uni", n_layers=6, n_expert=32, k=4, seqs_per_domain=30, tokens_per_seq=120, zipf=0.0, structure=False, stay=0.0)
        ru = pe.run(tu, None, budget, [2, 4, 8, 16], 4)
        check(abs(ru["miss_reduction_vs_static"]) < 0.04, f"structureless routing: miss reduction vs profile-ranked prefetch ~0 ({ru['miss_reduction_vs_static']:.3f})")
        check(ru["miss_reduction_vs_bank_only"] > 0.10, f"... while against the bank alone blind fetching alone 'removes' {ru['miss_reduction_vs_bank_only']:.2f} of the misses (why the gate does not use that)")
        # --- shared hidden topic: real but moderate structure
        ts = make_trace(d / "topic", n_layers=6, n_expert=32, k=4, n_topics=4, seqs_per_domain=30, tokens_per_seq=150, stay=0.95)
        rs = pe.run(ts, None, budget, [2, 4, 8, 16], 4)
        check(rs["miss_reduction_vs_static"] > ru["miss_reduction_vs_static"] + 0.03, f"topic-structured routing finds {rs['miss_reduction_vs_static']:.2f}, more than structureless {ru['miss_reduction_vs_static']:.2f}")
        check(0.0 < rs["recall_at_k"] <= 1.0 and set(rs["miss_reduction_by_slots"]) == {"2", "4", "8", "16"}, "result has recall and all slot counts")

        # --- command line + gate H9 on both outcomes
        only = d / "only_chain"
        only.mkdir()
        for suffix in (".pltrace", ".meta.jsonl", ".done"):
            (only / ("chain" + suffix)).write_bytes((d / ("chain" + suffix)).read_bytes())
        out = d / "pred.json"
        p = subprocess.run([sys.executable, str(HERE.parent / "sim" / "predictor_eval.py"), "--trace-dir", str(only), "--out", str(out), "--sizes", str(d / "none.json"),
                            "--dual", str(d / "none"), "--g0", str(d / "none"), "--budget-gib", str(budget)], capture_output=True, text=True)
        check(p.returncode == 0 and out.exists(), "command line run writes predictor_result.json" + ("" if p.returncode == 0 else "\n" + p.stderr[-400:]))
        g = subprocess.run([sys.executable, str(HERE.parent / "lib" / "gates.py"), "h9", "--predictor", str(out)], capture_output=True, text=True)
        check(g.returncode == 0 and "PASS" in g.stdout, "gate H9 passes the chained trace")
        bad = d / "bad.json"
        bad.write_text(json.dumps({"miss_reduction_vs_static": 0.05, "miss_reduction_vs_bank_only": 0.3, "prefetch_slots": 4, "recall_at_k": 0.3}))
        g = subprocess.run([sys.executable, str(HERE.parent / "lib" / "gates.py"), "h9", "--predictor", str(bad)], capture_output=True, text=True)
        check(g.returncode == 1 and "FAIL" in g.stdout, "gate H9 fails a 5% result")
    print("FAILURES: %d" % fails if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
