#!/usr/bin/env python3
# File: <project>\moe_track\sim\predictor_eval.py
"""Gate H9: can the experts a layer will need be guessed one layer ahead, well enough to cut the misses of a static bank?

    python sim/predictor_eval.py [--trace-dir logs/moe_trace] [--sizes ...] [--budget-gib 5.6] [--prefetch-slots 4] [--out sim/predictor_result.json]

Setting (what a runtime could really know)
  When layer l of token t is about to run, the router of layer l has not run yet. Known: every expert pick of layers 0..l-1 for this token, and all
  picks of earlier tokens of this sequence. From that the predictor names P experts (not already in the static bank) to copy to the GPU while layer
  l-1 is still computing. "Lead time: one layer."
  P is small because the copy has to fit in one layer's compute window: at ~26 GB/s and ~0.4 ms per layer that is ~10 MB, about five 2 MB experts.
  --prefetch-slots sets P (default 4); the result also lists P = 2, 4, 8, 16.

Predictors, all fitted on the fit sequences only and scored on the held-out sequences
  prior       the fit-set frequency of each expert in that layer (what the static bank already uses)
  temporal    experts picked in this layer by recent tokens of the same sequence, exponentially decayed
  crosslayer  co-occurrence: P(expert e in layer l | experts picked in layer l-1 by this token), from the fit set
  blend       weighted sum of the three, weights chosen on a validation slice INSIDE the fit set (never on held-out)

Metrics (held-out layer steps)
  miss_reduction_vs_static    1 - misses(bank + P predicted) / misses(bank + P profile-ranked).  The comparison is with a STATIC system that spends the
                              same P staging slots on the experts the fit set used most (outside the bank), because any P extra fetches remove some misses
                              even if chosen blindly (about P / (experts outside the bank) of them). Only the part that prediction adds counts. Gate H9: >= 0.20.
  miss_reduction_vs_bank_only 1 - misses(bank + P predicted) / misses(bank). Larger, but includes what blind fetching would also give. Reported, not gated.
  recall_at_k                 share of the K actually-picked experts found among the predictor's top-K (unrestricted); a quality number, not the gate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402
import pltrace  # noqa: E402
import summarize_traces  # noqa: E402
import waterfill_sim as ws  # noqa: E402

DECAY = 0.8


def multihot(ex_l: np.ndarray, n_expert: int) -> np.ndarray:
    """ex_l uint16 [T,K] -> float32 [T,E]"""
    T, K = ex_l.shape
    m = np.zeros((T, n_expert), np.float32)
    ok = ex_l < n_expert
    rows = np.repeat(np.arange(T), K).reshape(T, K)
    m[rows[ok], ex_l[ok].astype(np.int64)] = 1.0
    return m


def temporal_scores(pres: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """pres [T,E] multi-hot of each token's picks in one layer; seq [T]. score_t = sum_{j<t, same seq} DECAY^(t-1-j) pres_j."""
    T, E = pres.shape
    out = np.zeros((T, E), np.float32)
    acc = np.zeros(E, np.float32)
    prev_seq = None
    for t in range(T):
        if seq[t] != prev_seq:
            acc = np.zeros(E, np.float32)
            prev_seq = seq[t]
        out[t] = acc
        acc = DECAY * acc + pres[t]            # shifted so that token t's own picks are NOT in its own score
    return out


def fit_crosslayer(pres_prev: np.ndarray, pres_next: np.ndarray) -> np.ndarray:
    """-> [E,E] row-normalised P(next expert | previous expert) from co-occurrence counts."""
    c = pres_prev.T @ pres_next
    rs = c.sum(axis=1, keepdims=True)
    return c / np.maximum(rs, 1e-9)


def _norm_rows(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return x / np.maximum(m, 1e-9)


def prefetch_pick(scores: np.ndarray, bank_l: np.ndarray, p: int, tiebreak: np.ndarray) -> np.ndarray:
    """top-p experts per row, excluding the bank. -> bool [T,E]"""
    T, E = scores.shape
    s = scores + 1e-6 * tiebreak[None, :]
    s = np.where(bank_l[None, :], -np.inf, s)
    p = min(p, int((~bank_l).sum()))
    pred = np.zeros((T, E), bool)
    if p > 0:
        idx = np.argpartition(-s, p - 1, axis=1)[:, :p]
        np.put_along_axis(pred, idx, True, axis=1)
    return pred


def recall_topk(scores: np.ndarray, actual: np.ndarray, k: int, tiebreak: np.ndarray) -> float:
    s = scores + 1e-6 * tiebreak[None, :]
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    pred = np.zeros(scores.shape, bool)
    np.put_along_axis(pred, idx, True, axis=1)
    return float(np.take_along_axis(pred, actual.astype(np.int64), axis=1).mean())


def evaluate(train_ex: np.ndarray, eval_ex: np.ndarray, eval_seq: np.ndarray, train_seq: np.ndarray, bank: np.ndarray, n_expert: int,
             slots: Sequence[int], weight_sets: Sequence[Tuple[float, float, float]]) -> dict:
    """Fit on train_ex [Tt,L,K], score on eval_ex [Te,L,K]. weight_sets are (w_prior, w_temporal, w_cross) for the blend."""
    Te, L, K = eval_ex.shape
    miss_static = 0.0
    miss = {(w, p): 0.0 for w in weight_sets for p in slots}
    kinds = {"prior": (1.0, 0.0, 0.0), "temporal": (0.0, 1.0, 0.0), "crosslayer": (0.0, 0.0, 1.0)}
    miss_kind = {k: {p: 0.0 for p in slots} for k in kinds}
    recall = {k: 0.0 for k in list(kinds) + ["blend0"]}
    steps = 0
    prev_tr = prev_ev = None
    for li in range(L):
        pres_tr = multihot(train_ex[:, li, :], n_expert)
        pres_ev = multihot(eval_ex[:, li, :], n_expert)
        prior = pres_tr.sum(axis=0)
        prior_n = (prior / max(prior.max(), 1e-9)).astype(np.float32)
        actual = eval_ex[:, li, :]
        bank_l = bank[li]
        ok_actual = actual < n_expert
        a_idx = np.where(ok_actual, actual, 0).astype(np.int64)
        miss_static += float((~bank_l[a_idx] & ok_actual).sum())
        steps += Te

        S_prior = np.broadcast_to(prior_n, (Te, n_expert)).astype(np.float32)
        S_temp = _norm_rows(temporal_scores(pres_ev, eval_seq))
        if li > 0:
            cn = fit_crosslayer(prev_tr, pres_tr)
            S_cross = _norm_rows(prev_ev @ cn)
        else:
            S_cross = np.zeros((Te, n_expert), np.float32)
        comps = {"prior": S_prior, "temporal": S_temp, "crosslayer": S_cross}

        for kind, w in kinds.items():
            sc = w[0] * S_prior + w[1] * S_temp + w[2] * S_cross
            for p in slots:
                pred = prefetch_pick(sc, bank_l, p, prior_n)
                miss_kind[kind][p] += float((~bank_l[a_idx] & ~np.take_along_axis(pred, a_idx, 1) & ok_actual).sum())
            recall[kind] += recall_topk(sc, a_idx, K, prior_n) * Te
        for w in weight_sets:
            sc = w[0] * S_prior + w[1] * S_temp + w[2] * S_cross
            for p in slots:
                pred = prefetch_pick(sc, bank_l, p, prior_n)
                miss[(w, p)] += float((~bank_l[a_idx] & ~np.take_along_axis(pred, a_idx, 1) & ok_actual).sum())
        if weight_sets:
            w = weight_sets[0]
            recall["blend0"] += recall_topk(w[0] * S_prior + w[1] * S_temp + w[2] * S_cross, a_idx, K, prior_n) * Te
        prev_tr, prev_ev = pres_tr, pres_ev
    return {
        "miss_static": miss_static,
        "steps": steps,
        "blend_misses": dict(miss),
        "kind_misses": miss_kind,
        "recall": {k: v / (Te * L) for k, v in recall.items()},
    }


def reduction(miss_pred: float, miss_ref: float) -> float:
    return 1.0 - miss_pred / max(miss_ref, 1.0)


def run(trace: "pltrace.Trace", sizes: Optional[dict], budget_gib: float, slots: Sequence[int], headline_slots: int) -> dict:
    L, E, K = len(trace.moe_layers), trace.n_expert, trace.k
    size = ws.expert_bytes_for_layers(sizes, trace.moe_layers)
    ex = trace.decode.experts
    seq = trace.decode.seq
    fit_m, held_m = trace.decode_split_mask("fit"), trace.decode_split_mask("heldout")
    if fit_m.sum() == 0 or held_m.sum() == 0:
        raise SystemExit("need both fit and held-out tokens")
    slots = sorted(set(slots) | {headline_slots})

    # ---- choose the blend weights on a validation slice of the FIT set
    fit_idx = np.flatnonzero(fit_m)
    val_m = (seq[fit_idx] % 3) == 0
    tr_idx, va_idx = fit_idx[~val_m], fit_idx[val_m]
    grid = [w for w in itertools.product((0.0, 0.5, 1.0), (0.0, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0)) if sum(w) > 0]
    best_w = (1.0, 1.0, 1.0)
    if len(va_idx) >= 200 and len(tr_idx) >= 200:
        freq_tr = summarize_traces.pick_counts(ex[tr_idx], E)
        bank_tr = ws.select_static(freq_tr, size, budget_gib * ws.GIB)
        ev = evaluate(ex[tr_idx], ex[va_idx], seq[va_idx], seq[tr_idx], bank_tr, E, [headline_slots], grid)
        best_w = max(grid, key=lambda w: -ev["blend_misses"][(w, headline_slots)])

    # ---- final: fit on all fit tokens, score on held-out
    freq = summarize_traces.pick_counts(ex[fit_m], E)
    bank = ws.select_static(freq, size, budget_gib * ws.GIB)
    final_w = [best_w, (1.0, 1.0, 1.0)]
    res = evaluate(ex[fit_m], ex[held_m], seq[held_m], seq[fit_m], bank, E, slots, final_w)

    ms = res["miss_static"]
    mb = res["blend_misses"]
    mk = res["kind_misses"]
    by_slots = {str(p): round(reduction(mb[(best_w, p)], mk["prior"][p]), 4) for p in slots}
    by_slots_bank_only = {str(p): round(reduction(mb[(best_w, p)], ms), 4) for p in slots}
    kinds = {k: {str(p): round(reduction(v, mk["prior"][p]), 4) for p, v in d.items()} for k, d in mk.items()}
    kinds_bank_only = {k: {str(p): round(reduction(v, ms), 4) for p, v in d.items()} for k, d in mk.items()}
    out = {
        "budget_gib": budget_gib,
        "prefetch_slots": headline_slots,
        "lead_time": "one layer (inputs: this token's picks in earlier layers, earlier tokens of the sequence)",
        "heldout_tokens": int(held_m.sum()),
        "fit_tokens": int(fit_m.sum()),
        "blend_weights_prior_temporal_cross": list(best_w),
        "miss_reduction_vs_static": by_slots[str(headline_slots)],
        "miss_reduction_vs_bank_only": by_slots_bank_only[str(headline_slots)],
        "miss_reduction_by_slots": by_slots,
        "miss_reduction_by_slots_vs_bank_only": by_slots_bank_only,
        "miss_reduction_by_predictor": kinds,
        "miss_reduction_by_predictor_vs_bank_only": kinds_bank_only,
        "static_reference": "bank + the same number of staging slots filled with the fit set's most-used experts outside the bank",
        "recall_at_k": round(res["recall"]["blend0"], 4),
        "recall_at_k_by_predictor": {k: round(v, 4) for k, v in res["recall"].items() if k != "blend0"},
        "misses_static_per_layer_step": round(res["miss_static"] / res["steps"], 4),
        "note": "gate H9 reads miss_reduction_vs_static (vs a profile-ranked prefetch of the same size) at prefetch_slots; blend weights were chosen on a validation slice of the fit set",
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--trace-dir", default=str(paths.MOE_TRACE_DIR))
    ap.add_argument("--sizes", default=str(paths.FLOORS / "expert_layer_sizes.json"))
    ap.add_argument("--budget-gib", type=float, default=5.6)
    ap.add_argument("--prefetch-slots", type=int, default=4)
    ap.add_argument("--dual", default=str(paths.FLOORS / "dual_bus_contention.json"))
    ap.add_argument("--g0", default=str(paths.FLOORS / "g0_result.json"))
    ap.add_argument("--out", default=str(paths.SIM_DIR / "predictor_result.json"))
    a = ap.parse_args(argv)

    trace = ws.merged_trace(Path(a.trace_dir), a.files or None)
    sizes = json.loads(Path(a.sizes).read_text()) if Path(a.sizes).exists() else None
    out = run(trace, sizes, a.budget_gib, [2, 4, 8, 16], a.prefetch_slots)

    # how many slots fit in one layer's window, if the measurements exist
    try:
        du = json.loads(Path(a.dual).read_text()) if Path(a.dual).exists() else None
        g0 = json.loads(Path(a.g0).read_text()) if Path(a.g0).exists() else None
        if du and g0 and du.get("h2d_copy_alone_gb_s"):
            tok_ms = 1000.0 / float(g0["measured_tps"]["28"]) if "28" in g0.get("measured_tps", {}) else None
            if tok_ms:
                window_ms = tok_ms / len(trace.moe_layers)
                eb = float(ws.expert_bytes_for_layers(sizes, trace.moe_layers).mean())
                out["window_ms_per_layer"] = round(window_ms, 3)
                out["slots_that_fit_in_one_layer_window"] = round(float(du["h2d_copy_alone_gb_s"]) * 1e9 * window_ms * 1e-3 / eb, 2)
    except (OSError, ValueError, KeyError):
        pass

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"static bank {a.budget_gib} GiB leaves {out['misses_static_per_layer_step']:.2f} missing experts per layer step (of {trace.k})")
    print(f"prefetch {a.prefetch_slots} experts/layer one layer ahead: miss reduction {out['miss_reduction_vs_static']*100:.1f}% vs a profile-ranked prefetch of the same size"
          f" ({out['miss_reduction_vs_bank_only']*100:.1f}% vs the bank alone)   blend weights prior/temporal/cross {out['blend_weights_prior_temporal_cross']}")
    print("   by slots, vs profile-ranked prefetch: " + ", ".join(f"{p}: {v*100:.1f}%" for p, v in out["miss_reduction_by_slots"].items()))
    print("   by slots, vs bank alone:              " + ", ".join(f"{p}: {v*100:.1f}%" for p, v in out["miss_reduction_by_slots_vs_bank_only"].items()))
    for k, d in out["miss_reduction_by_predictor"].items():
        print(f"   {k:10s} vs profile-ranked: " + ", ".join(f"{p}: {v*100:.1f}%" for p, v in d.items()) + f"   recall@K {out['recall_at_k_by_predictor'][k]:.3f}")
    if "slots_that_fit_in_one_layer_window" in out:
        print(f"   a layer window of {out['window_ms_per_layer']} ms moves about {out['slots_that_fit_in_one_layer_window']} experts over PCIe: slots above that cannot be hidden")
    print(f"wrote {a.out}\nnext:  python lib/gates.py h9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
