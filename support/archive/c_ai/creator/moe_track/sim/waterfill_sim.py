#!/usr/bin/env python3
# File: <project>\moe_track\sim\waterfill_sim.py
"""Step 4 / gate G2: how many expert reads could a VRAM-resident expert bank serve, measured on recorded routing?

    python sim/waterfill_sim.py [--trace-dir logs/moe_trace] [--sizes logs/floors/expert_layer_sizes.json]
                                [--budget-gib 5.6] [--budgets 2,3,4,5,5.6,6,8] [--g0 logs/floors/g0_result.json] [--out sim/sim_result.json]

Method
  fit set      sequences whose prompt id hashes into "fit" (70%); held-out set: the other 30% (trace/pltrace.py split_of). The bank is chosen
               ONLY from the fit set and scored on both: in-sample (fit) and held-out. Gate G2 uses the held-out number.
  static bank  "water-filling": rank every (layer, expert) by how often the fit set used it and keep the most-used ones until the byte budget is
               spent. Time saved per byte is the same in every layer, so ranking by frequency is the right order; experts differ in size
               (Q4_K vs Q6_K down projections), so the budget is counted in bytes and an item that does not fit is skipped, not the whole fill.
  hit rate     byte-weighted: bytes of picked experts found in the bank / bytes of all picked experts (this is what token time depends on);
               the plain count of picks is reported too. The gate reads the byte-weighted held-out value.
  dynamic      LRU caches over the held-out sequences in order, same byte budget: one global pool, and per-layer pools sized like the static
               allocation; and a hybrid (half static, half LRU). A cache sees the held-out stream only, which is how a server would see it.
  misses       per (token, layer) step: share of steps with 0,1,2,... missing experts. A layer that has any miss has to wait for a fetch or a CPU
               computation, so this decides how much of the hit rate turns into speed.
  forecast     if --g0 points at logs/floors/g0_result.json (written by microbench/06_verify_gate_g0.py), tokens/s are forecast with the fitted token-time
               model for misses computed on the CPU path. Trust it only if gate G0 passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
sys.path.insert(0, str(HERE.parent / "lib"))
import floor_model  # noqa: E402
import paths  # noqa: E402
import pltrace  # noqa: E402
import summarize_traces  # noqa: E402

GIB = float(2 ** 30)
DEFAULT_EXPERT_BYTES = 1.9e6


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def expert_bytes_for_layers(sizes: Optional[dict], moe_layers: List[int]) -> np.ndarray:
    """Bytes of ONE expert (gate+up+down) in each traced layer, from the GGUF audit; a default if the audit is missing."""
    if sizes and sizes.get("layers"):
        by_layer = {int(l["layer"]): float(l["expert_bytes"]) for l in sizes["layers"] if l.get("role", "spine") == "spine"}
        if by_layer:
            mean = float(np.mean(list(by_layer.values())))
            return np.array([by_layer.get(int(l), mean) for l in moe_layers], np.float64)
    return np.full(len(moe_layers), DEFAULT_EXPERT_BYTES, np.float64)


def merged_trace(trace_dir: Path, files: Optional[List[str]] = None) -> "pltrace.Trace":
    traces = [t for t in summarize_traces.load_done(trace_dir, files) if t.decode.n]
    if not traces:
        raise SystemExit(f"no finished traces in {trace_dir}; run trace/05_collect_traces.py first")
    return pltrace.merge_decode(traces) if len(traces) > 1 else traces[0]


# ---------------------------------------------------------------------------------------------------------------
# static bank
# ---------------------------------------------------------------------------------------------------------------
def select_static(freq: np.ndarray, size: np.ndarray, budget_bytes: float) -> np.ndarray:
    """freq [L,E] picks per expert in the fit set; size [L] bytes per expert. -> boolean bank mask [L,E].

    Greedy by frequency (value density), skipping items that no longer fit. Experts the fit set never picked are not stored (no evidence)."""
    L, E = freq.shape
    flat_f = freq.ravel().astype(np.float64)
    flat_s = np.repeat(size, E)
    order = np.argsort(-flat_f, kind="stable")
    order = order[flat_f[order] > 0]
    mask = np.zeros(L * E, bool)
    used = 0.0
    cum = np.cumsum(flat_s[order])
    n_prefix = int(np.searchsorted(cum, budget_bytes, side="right"))
    mask[order[:n_prefix]] = True
    used = float(cum[n_prefix - 1]) if n_prefix else 0.0
    smallest = float(size.min())
    for i in order[n_prefix:]:                       # fill the remainder with whatever still fits
        if budget_bytes - used < smallest:
            break
        if flat_s[i] <= budget_bytes - used:
            mask[i] = True
            used += float(flat_s[i])
    return mask.reshape(L, E)


def bank_bytes(mask: np.ndarray, size: np.ndarray) -> float:
    return float((mask.sum(axis=1) * size).sum())


def score_bank(mask: np.ndarray, experts: np.ndarray, size: np.ndarray, n_expert: int) -> dict:
    """experts uint16 [T,L,K]. Returns byte-weighted and count hit rates, per-layer hit rates and the miss-per-step histogram."""
    T, L, K = experts.shape
    if T == 0:
        return {"tokens": 0}
    hit = np.zeros((T, L, K), bool)
    for li in range(L):
        ex = experts[:, li, :]
        ok = ex < n_expert
        h = np.zeros(ex.shape, bool)
        h[ok] = mask[li][ex[ok]]
        hit[:, li, :] = h
    return _score_from_hits(hit, size)


def _score_from_hits(hit: np.ndarray, size: np.ndarray) -> dict:
    T, L, K = hit.shape
    w = size[None, :, None]
    byte_hit = float((hit * w).sum() / (np.ones_like(hit) * w).sum())
    per_layer = (hit.sum(axis=(0, 2)) / (T * K)).tolist()
    miss_per_step = K - hit.sum(axis=2)                                   # [T, L]
    hist = np.bincount(miss_per_step.ravel(), minlength=K + 1)[: K + 1] / miss_per_step.size
    return {"tokens": int(T), "hit_rate_bytes": round(byte_hit, 5), "hit_rate_count": round(float(hit.mean()), 5),
            "per_layer_hit": [round(x, 4) for x in per_layer], "miss_per_step_hist": [round(float(x), 5) for x in hist],
            "steps_with_any_miss": round(float(1.0 - hist[0]), 5)}


# ---------------------------------------------------------------------------------------------------------------
# dynamic caches (LRU), simulated over the held-out stream in order
# ---------------------------------------------------------------------------------------------------------------
def lru_global(experts: np.ndarray, size: np.ndarray, budget_bytes: float, seq_order: Optional[np.ndarray] = None) -> np.ndarray:
    """One LRU pool over all (layer, expert) pairs. Returns hit [T,L,K] bool."""
    T, L, K = experts.shape
    od: "OrderedDict[int, None]" = OrderedDict()
    used = 0.0
    E = 1 << 16
    hit = np.zeros((T, L, K), bool)
    sz = [float(s) for s in size]
    for t in range(T):
        row = experts[t]
        for li in range(L):
            ids = [int(x) for x in row[li]]
            keys = [li * E + e for e in ids]
            hs = [k in od for k in keys]
            for k, h in zip(keys, hs):                      # hits first, so this step's own hits are never the eviction victims
                if h:
                    od.move_to_end(k)
            for j, (k, h) in enumerate(zip(keys, hs)):
                hit[t, li, j] = h
                if not h:
                    if ids[j] >= E - 1:
                        continue
                    od[k] = None
                    used += sz[li]
            while used > budget_bytes and od:
                old, _ = od.popitem(last=False)
                used -= sz[old // E]
    return hit


def lru_per_layer(experts: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Independent LRU per layer with slots[l] experts each. Returns hit [T,L,K]."""
    T, L, K = experts.shape
    pools = [OrderedDict() for _ in range(L)]
    hit = np.zeros((T, L, K), bool)
    for t in range(T):
        row = experts[t]
        for li in range(L):
            cap = int(slots[li])
            if cap <= 0:
                continue
            od = pools[li]
            ids = [int(x) for x in row[li]]
            hs = [e in od for e in ids]
            for e, h in zip(ids, hs):
                if h:
                    od.move_to_end(e)
            for j, (e, h) in enumerate(zip(ids, hs)):
                hit[t, li, j] = h
                if not h:
                    od[e] = None
            while len(od) > cap:
                od.popitem(last=False)
    return hit


def hybrid(experts: np.ndarray, static_mask: np.ndarray, size: np.ndarray, lru_budget_bytes: float) -> np.ndarray:
    """Static bank plus a global LRU for everything outside it. Returns hit [T,L,K]."""
    T, L, K = experts.shape
    n_expert = static_mask.shape[1]
    base = np.zeros((T, L, K), bool)
    for li in range(L):
        ex = experts[:, li, :]
        ok = ex < n_expert
        h = np.zeros(ex.shape, bool)
        h[ok] = static_mask[li][ex[ok]]
        base[:, li, :] = h
    od: "OrderedDict[int, None]" = OrderedDict()
    used = 0.0
    E = 1 << 16
    sz = [float(s) for s in size]
    out = base.copy()
    for t in range(T):
        for li in range(L):
            ids = [int(x) for x in experts[t, li]]
            for j, e in enumerate(ids):
                if base[t, li, j]:
                    continue
                k = li * E + e
                if k in od:
                    od.move_to_end(k)
                    out[t, li, j] = True
                else:
                    od[k] = None
                    used += sz[li]
            while used > lru_budget_bytes and od:
                old, _ = od.popitem(last=False)
                used -= sz[old // E]
    return out


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def run(trace: "pltrace.Trace", sizes: Optional[dict], budget_gib: float, budgets: List[float], g0: Optional[dict], do_lru: bool = True) -> dict:
    L, E, K = len(trace.moe_layers), trace.n_expert, trace.k
    size = expert_bytes_for_layers(sizes, trace.moe_layers)
    ex_all = trace.decode.experts
    fit_m, held_m = trace.decode_split_mask("fit"), trace.decode_split_mask("heldout")
    if held_m.sum() == 0 or fit_m.sum() == 0:
        raise SystemExit(f"need both fit and held-out tokens (have fit {int(fit_m.sum())}, held-out {int(held_m.sum())}); record more prompts")
    fit, held = ex_all[fit_m], ex_all[held_m]
    freq = summarize_traces.pick_counts(fit, E)

    result: dict = {
        "n_decode_tokens": int(trace.decode.n), "fit_tokens": int(fit_m.sum()), "heldout_tokens": int(held_m.sum()),
        "n_moe_layers": L, "n_expert": E, "n_expert_used": K,
        "expert_bytes_per_layer_mean": round(float(size.mean())),
        "all_experts_gib": round(float((size * E).sum() / GIB), 3),
        "budget_gib": budget_gib,
        "hit_rate_definition": "byte-weighted: bytes of picked experts found in the bank / bytes of all picked experts",
    }

    def at(b: float) -> Tuple[np.ndarray, dict, dict]:
        mask = select_static(freq, size, b * GIB)
        return mask, score_bank(mask, fit, size, E), score_bank(mask, held, size, E)

    mask, s_in, s_out = at(budget_gib)
    result["bank_gib_used"] = round(bank_bytes(mask, size) / GIB, 4)
    result["bank_experts"] = int(mask.sum())
    result["bank_experts_per_layer"] = mask.sum(axis=1).astype(int).tolist()
    result["insample_hit_rate_at_budget"] = s_in["hit_rate_bytes"]
    result["heldout_hit_rate_at_budget"] = s_out["hit_rate_bytes"]
    result["insample_hit_rate_count"] = s_in["hit_rate_count"]
    result["heldout_hit_rate_count"] = s_out["hit_rate_count"]
    result["heldout_detail"] = s_out
    result["curve"] = []
    for b in budgets:
        _, a, h = at(b)
        result["curve"].append({"budget_gib": b, "insample": a["hit_rate_bytes"], "heldout": h["hit_rate_bytes"]})

    # per-domain: bank chosen from the OTHER domains' fit tokens (does a profile generalise across kinds of work?)
    dom = trace.decode_domains()
    per_dom = {}
    for d in sorted(set(dom.tolist())):
        m_d = held_m & (dom == d)
        m_other = fit_m & (dom != d)
        if m_d.sum() < 200 or m_other.sum() < 200:
            continue
        mk = select_static(summarize_traces.pick_counts(ex_all[m_other], E), size, budget_gib * GIB)
        mk_own = select_static(summarize_traces.pick_counts(ex_all[fit_m & (dom == d)], E), size, budget_gib * GIB) if (fit_m & (dom == d)).sum() >= 200 else None
        per_dom[d] = {"heldout_tokens": int(m_d.sum()),
                      "bank_from_other_domains": score_bank(mk, ex_all[m_d], size, E)["hit_rate_bytes"],
                      "bank_from_all_fit": score_bank(mask, ex_all[m_d], size, E)["hit_rate_bytes"]}
        if mk_own is not None:
            per_dom[d]["bank_from_own_domain"] = score_bank(mk_own, ex_all[m_d], size, E)["hit_rate_bytes"]
    result["per_domain_heldout"] = per_dom

    if do_lru:
        slots = mask.sum(axis=1)
        g = _score_from_hits(lru_global(held, size, budget_gib * GIB), size)
        pl = _score_from_hits(lru_per_layer(held, slots), size)
        hy = _score_from_hits(hybrid(held, select_static(freq, size, 0.5 * budget_gib * GIB), size, 0.5 * budget_gib * GIB), size)
        result["heldout_dynamic"] = {
            "lru_global_pool": g["hit_rate_bytes"], "lru_per_layer_same_slots": pl["hit_rate_bytes"], "hybrid_half_static_half_lru": hy["hit_rate_bytes"],
            "note": "caches start empty and see the held-out stream in file order; the first tokens pay warm-up misses",
        }

    if g0 and g0.get("model"):
        m = floor_model.FloorModel(g0["model"]["fixed_ms"], g0["model"]["b_cpu_gbs"], g0["model"].get("b_gpu_gbs", 504.0),
                                   floor_model.layer_bytes_from_sizes(sizes, topk=K, n_layers=L))
        if len(m.layer_bytes) == L:
            per_layer = s_out["per_layer_hit"]
            result["forecast"] = {
                "assumes": "misses are computed on the CPU path at the fitted bandwidth; valid only if gate G0 passed",
                "g0_passed": bool(g0.get("pass")),
                "tps_static_bank_heldout": round(m.tps_hit_rate(per_layer), 2),
                "tps_no_bank": round(m.tps_n_cpu_moe(L), 2),
            }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="trace files (default: every finished shard in --trace-dir)")
    ap.add_argument("--trace-dir", default=str(paths.MOE_TRACE_DIR))
    ap.add_argument("--sizes", default=str(paths.FLOORS / "expert_layer_sizes.json"))
    ap.add_argument("--budget-gib", type=float, default=5.6)
    ap.add_argument("--budgets", default="2,3,4,5,5.6,6,8")
    ap.add_argument("--g0", default=str(paths.FLOORS / "g0_result.json"))
    ap.add_argument("--out", default=str(paths.SIM_DIR / "sim_result.json"))
    ap.add_argument("--no-lru", action="store_true", help="skip the slow dynamic-cache simulations")
    a = ap.parse_args(argv)

    trace = merged_trace(Path(a.trace_dir), a.files or None)
    sizes = json.loads(Path(a.sizes).read_text()) if Path(a.sizes).exists() else None
    if sizes is None:
        print(f"note: {a.sizes} not found; assuming {DEFAULT_EXPERT_BYTES/1e6:.1f} MB per expert in every layer (run microbench/01_gguf_expert_sizes.py)", file=sys.stderr)
    g0 = json.loads(Path(a.g0).read_text()) if Path(a.g0).exists() else None
    res = run(trace, sizes, a.budget_gib, [float(x) for x in a.budgets.split(",") if x], g0, not a.no_lru)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"tokens: fit {res['fit_tokens']:,}, held-out {res['heldout_tokens']:,}; all experts {res['all_experts_gib']} GiB, bank {res['bank_gib_used']} GiB ({res['bank_experts']} experts)")
    print(f"static bank at {a.budget_gib} GiB: in-sample hit {res['insample_hit_rate_at_budget']:.3f}, HELD-OUT hit {res['heldout_hit_rate_at_budget']:.3f}  (count-based {res['heldout_hit_rate_count']:.3f})")
    print("budget curve (GiB: in-sample / held-out): " + "  ".join(f"{c['budget_gib']}: {c['insample']:.2f}/{c['heldout']:.2f}" for c in res["curve"]))
    hist = res["heldout_detail"]["miss_per_step_hist"]
    print(f"held-out layer steps with 0 misses {hist[0]*100:.0f}%, 1 miss {hist[1]*100:.0f}%, 2 misses {hist[2]*100:.0f}%, any miss {res['heldout_detail']['steps_with_any_miss']*100:.0f}%")
    for d, v in res["per_domain_heldout"].items():
        print(f"   domain {d:10s} held-out hit with bank from all fit {v['bank_from_all_fit']:.3f}, from other domains only {v['bank_from_other_domains']:.3f}" + (f", from its own {v['bank_from_own_domain']:.3f}" if "bank_from_own_domain" in v else ""))
    if "heldout_dynamic" in res:
        dd = res["heldout_dynamic"]
        print(f"dynamic caches, same budget: global LRU {dd['lru_global_pool']:.3f}, per-layer LRU {dd['lru_per_layer_same_slots']:.3f}, hybrid {dd['hybrid_half_static_half_lru']:.3f}")
    if "forecast" in res:
        f = res["forecast"]
        print(f"forecast: {f['tps_static_bank_heldout']} tok/s with the bank vs {f['tps_no_bank']} with none (misses on the CPU path; G0 passed: {f['g0_passed']})")
    print(f"wrote {out}\nnext:  python lib/gates.py g2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
