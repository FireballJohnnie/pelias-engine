#!/usr/bin/env python3
# File: <project>\moe_track\trace\summarize_traces.py
"""Gate G1 summary: how much was traced, and how skewed is expert use?

    python trace/summarize_traces.py [--trace-dir logs/moe_trace] [--out logs/moe_trace/trace_summary.json] [FILES...]

Reads every finished shard (*.pltrace with its *.meta.jsonl) in the trace folder, or the files given, and writes trace_summary.json:
    n_decode_tokens, n_sequences, gini_mean, gini_by_layer, domains{name: tokens}, split_tokens{fit, heldout},
    top_share{fraction of experts: share of picks}, domain_overlap{pair: Jaccard of each layer's top-20% experts, averaged},
    finish{reason: count}, files[]
Then   python lib/gates.py g1   scores it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402
import pltrace  # noqa: E402

TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)


def gini(counts: np.ndarray) -> float:
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n, s = len(c), c.sum()
    if n == 0 or s <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * c) / (n * s)) - (n + 1.0) / n)


def pick_counts(experts: np.ndarray, n_expert: int) -> np.ndarray:
    """experts uint16 [T, L, K] -> int64 [L, E] number of times each expert was picked in each layer (0xFFFF ignored)."""
    T, L, K = experts.shape
    out = np.zeros((L, n_expert), np.int64)
    for li in range(L):
        v = experts[:, li, :].ravel()
        v = v[v < n_expert]
        out[li] = np.bincount(v, minlength=n_expert)[:n_expert]
    return out


def top_share(counts: np.ndarray, frac: float) -> float:
    """Average over layers of the share of picks that fall on the most-used `frac` of experts."""
    L, E = counts.shape
    k = max(1, int(round(frac * E)))
    shares = []
    for li in range(L):
        c = np.sort(counts[li])[::-1]
        tot = c.sum()
        if tot > 0:
            shares.append(c[:k].sum() / tot)
    return float(np.mean(shares)) if shares else 0.0


def top_sets(counts: np.ndarray, frac: float) -> List[set]:
    L, E = counts.shape
    k = max(1, int(round(frac * E)))
    return [set(np.argsort(-counts[li], kind="stable")[:k].tolist()) for li in range(L)]


def summarize_trace(t: "pltrace.Trace") -> dict:
    L, E = len(t.moe_layers), t.n_expert
    counts = pick_counts(t.decode.experts, E)
    g_layers = [gini(counts[li]) for li in range(L)]
    doms = t.decode_by_domain()
    dom_tokens = {d: int(len(ix)) for d, ix in doms.items()}
    fit = int(t.decode_split_mask("fit").sum())
    held = int(t.decode_split_mask("heldout").sum())
    sets = {d: top_sets(pick_counts(t.decode.experts[ix], E), 0.20) for d, ix in doms.items() if len(ix) >= 500}
    names = sorted(sets)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = [len(sa & sb) / max(1, len(sa | sb)) for sa, sb in zip(sets[a], sets[b])]
            overlap[f"{a}|{b}"] = round(float(np.mean(j)), 4)
    finish: Dict[str, int] = {}
    for m in t.meta:
        finish[m.get("finish", "?")] = finish.get(m.get("finish", "?"), 0) + 1
    return {
        "n_decode_tokens": int(t.decode.n),
        "n_sequences": len(t.meta),
        "n_moe_layers": L,
        "n_expert": E,
        "n_expert_used": t.k,
        "gini_mean": round(float(np.mean(g_layers)), 4) if g_layers else 0.0,
        "gini_by_layer": [round(g, 4) for g in g_layers],
        "domains": dom_tokens,
        "split_tokens": {"fit": fit, "heldout": held},
        "top_share": {f"{f:.2f}": round(top_share(counts, f), 4) for f in TOP_FRACTIONS},
        "domain_overlap_top20_jaccard": overlap,
        "finish": finish,
        "experts_never_picked_fraction": round(float((counts.sum(axis=0) == 0).mean()), 4) if L else 0.0,
    }


def load_done(trace_dir: Path, files: Optional[List[str]] = None) -> List["pltrace.Trace"]:
    if files:
        cand = [Path(f) for f in files]
    else:
        cand = sorted(p for p in trace_dir.glob("*.pltrace") if p.with_suffix(".done").exists())
    out = []
    for p in cand:
        try:
            out.append(pltrace.read_trace(p))
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"skipping {p.name}: {e}", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--trace-dir", default=str(paths.MOE_TRACE_DIR))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    td = Path(a.trace_dir)
    traces = [t for t in load_done(td, a.files) if t.decode.n]
    if not traces:
        print(f"no finished traces in {td} (a shard counts as finished when its .done file exists)", file=sys.stderr)
        return 2
    merged = pltrace.merge_decode(traces) if len(traces) > 1 else traces[0]
    s = summarize_trace(merged)
    s["files"] = [t.source for t in traces]
    out = Path(a.out) if a.out else td / "trace_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"{s['n_decode_tokens']:,} decode tokens in {s['n_sequences']} sequences, {len(traces)} shard(s); mean per-layer Gini {s['gini_mean']:.3f}")
    print("share of picks on the most-used experts (mean over layers):  " + "   ".join(f"{float(k)*100:.0f}% of experts -> {v*100:.0f}%" for k, v in s["top_share"].items()))
    print("   published for this model: 5% -> 33%, 10% -> 48%, 20% -> 67%, 30% -> 79%, 50% -> 92% (workload unstated)")
    for d, n in s["domains"].items():
        print(f"   domain {d:10s} {n:8,d} tokens")
    if s["domain_overlap_top20_jaccard"]:
        print("   hot-set overlap between domains (Jaccard of top 20% experts, mean over layers): " + ", ".join(f"{k} {v:.2f}" for k, v in s["domain_overlap_top20_jaccard"].items()))
    print(f"   fit tokens {s['split_tokens']['fit']:,}, held-out tokens {s['split_tokens']['heldout']:,}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
