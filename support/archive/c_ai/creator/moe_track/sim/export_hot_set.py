#!/usr/bin/env python3
# File: <project>\moe_track\sim\export_hot_set.py
"""Step 6a: turn recorded routing into the hot-set file that the Pelias-Tier llama.cpp build (patches/pelias_tier_v1.patch) reads.

    python sim/export_hot_set.py [--trace-dir logs/moe_trace] [--sizes logs/floors/expert_layer_sizes.json]
                                 [--budget-gib 5.0] [--out logs/pelias_hot_set.txt] [--use-all-tokens]

The experts are chosen exactly like the static bank in sim/waterfill_sim.py (most-used first until the byte budget is spent), from the "fit" share
of the recorded prompts only, so the held-out hit rate printed here is an honest estimate (the same split the gate G2 uses).

Budget accounting. Each layer that gets a bank also gets one extra all-zero "slot 0" (it is what makes a miss cost nothing on the GPU branch), so the
budget given here is the TOTAL bank size: slot-0 bytes are subtracted from it before experts are chosen.

Output (plain text, read by src/llama-pelias-tier.cpp):
    PELIAS_HOT_SET 1
    n_expert 256
    layer 0 : 3 17 22 ...
A "#" starts a comment. The same text is also written as JSON next to it (<out>.json) with the provenance numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402
import summarize_traces  # noqa: E402
import waterfill_sim as ws  # noqa: E402

GIB = float(2 ** 30)


def choose(trace, sizes: Optional[dict], budget_gib: float, use_all: bool) -> dict:
    """Returns {"mask": bool [L,E], "size": float [L], "layers": [layer ids], ...} for the chosen bank."""
    L, E = len(trace.moe_layers), trace.n_expert
    size = ws.expert_bytes_for_layers(sizes, trace.moe_layers)
    ex_all = trace.decode.experts
    fit_m = trace.decode_split_mask("fit")
    held_m = trace.decode_split_mask("heldout")
    if use_all or fit_m.sum() == 0:
        fit = ex_all
    else:
        fit = ex_all[fit_m]
    freq = summarize_traces.pick_counts(fit, E)

    zero_slots = float(size.sum())                      # one all-zero slot per layer that has a bank (worst case: every layer)
    budget = budget_gib * GIB - zero_slots
    if budget <= 0:
        raise SystemExit(f"budget {budget_gib} GiB is smaller than the {zero_slots / GIB:.2f} GiB of zero slots; raise --budget-gib")
    mask = ws.select_static(freq, size, budget)
    # a layer with no chosen expert needs no bank and no zero slot
    used = ws.bank_bytes(mask, size) + float(size[mask.sum(axis=1) > 0].sum())

    res = {
        "mask": mask, "size": size, "layers": list(trace.moe_layers), "n_expert": E,
        "bank_bytes_with_zero_slots": used,
        "insample": ws.score_bank(mask, fit, size, E),
        "heldout": ws.score_bank(mask, ex_all[held_m], size, E) if held_m.sum() and not use_all else None,
        "fit_tokens": int(fit.shape[0]), "heldout_tokens": int(held_m.sum()),
    }
    return res


def render(res: dict, budget_gib: float, source: str) -> str:
    mask, layers, E = res["mask"], res["layers"], res["n_expert"]
    lines = [
        "# Pelias-Tier hot set, written by sim/export_hot_set.py. Do not edit by hand.",
        f"# source: {source}",
        f"# budget {budget_gib:.3f} GiB total (including one all-zero slot per layer), used {res['bank_bytes_with_zero_slots'] / GIB:.3f} GiB, {int(mask.sum())} experts",
        f"# estimated hit rate (byte-weighted): in-sample {res['insample'].get('hit_rate_bytes')}"
        + (f", held-out {res['heldout'].get('hit_rate_bytes')}" if res["heldout"] else ", held-out not measured"),
        "PELIAS_HOT_SET 1",
        f"n_expert {E}",
    ]
    for li, layer in enumerate(layers):
        ids = np.nonzero(mask[li])[0].tolist()
        if ids:
            lines.append(f"layer {int(layer)} : " + " ".join(str(int(e)) for e in ids))
    return "\n".join(lines) + "\n"


def parse_text(text: str, n_expert: int) -> Dict[int, List[int]]:
    """Python mirror of the C++ parser in llama-pelias-tier.cpp (used by the tests to keep the two in step)."""
    out: Dict[int, List[int]] = {}
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].replace(":", " ").replace(",", " ")
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "PELIAS_HOT_SET":
            if len(tok) < 2 or int(tok[1]) != 1:
                raise ValueError(f"line {ln}: unsupported version")
        elif tok[0] == "n_expert":
            if int(tok[1]) != n_expert:
                raise ValueError(f"line {ln}: n_expert mismatch")
        elif tok[0] == "layer":
            il = int(tok[1])
            ids = sorted(set(int(x) for x in tok[2:]))
            if il < 0 or any(e < 0 or e >= n_expert for e in ids):
                raise ValueError(f"line {ln}: out of range")
            out[il] = ids
        else:
            raise ValueError(f"line {ln}: unknown keyword {tok[0]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="trace files (default: every finished shard in --trace-dir)")
    ap.add_argument("--trace-dir", default=str(paths.MOE_TRACE_DIR))
    ap.add_argument("--sizes", default=str(paths.FLOORS / "expert_layer_sizes.json"))
    ap.add_argument("--budget-gib", type=float, default=5.0,
                    help="total VRAM for the bank. Default 5.0: the working N=28 placement uses 10.2-10.4 GiB of the 12 GiB card, of which the 12 GPU-resident layers are 5.58 GiB")
    ap.add_argument("--out", default=str(paths.CREATOR / "logs" / "pelias_hot_set.txt"))
    ap.add_argument("--use-all-tokens", action="store_true", help="choose from every recorded token (no held-out estimate); use for the final production profile")
    a = ap.parse_args(argv)

    trace = ws.merged_trace(Path(a.trace_dir), a.files or None)
    sizes = json.loads(Path(a.sizes).read_text()) if Path(a.sizes).exists() else None
    if sizes is None:
        print(f"note: {a.sizes} not found; assuming {ws.DEFAULT_EXPERT_BYTES / 1e6:.1f} MB per expert (run microbench/01_gguf_expert_sizes.py first)", file=sys.stderr)
    res = choose(trace, sizes, a.budget_gib, a.use_all_tokens)

    src = f"{len(trace.meta)} prompts, {trace.decode.n} decode tokens; traces: {trace.source}"
    text = render(res, a.budget_gib, src)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    meta = {
        "format": "pelias-hot-set", "version": 1, "n_expert": res["n_expert"], "moe_layers": res["layers"],
        "budget_gib": a.budget_gib, "bank_gib_with_zero_slots": round(res["bank_bytes_with_zero_slots"] / GIB, 4),
        "experts_per_layer": res["mask"].sum(axis=1).astype(int).tolist(),
        "insample": res["insample"], "heldout": res["heldout"], "sha256_of_text": hashlib.sha256(text.encode()).hexdigest(),
        "chosen_from": "all tokens" if a.use_all_tokens else "fit split only",
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    held = res["heldout"]["hit_rate_bytes"] if res["heldout"] else None
    print(f"bank: {int(res['mask'].sum())} experts in {int((res['mask'].sum(axis=1) > 0).sum())} layers, {res['bank_bytes_with_zero_slots'] / GIB:.3f} GiB with zero slots")
    print(f"hit rate (byte-weighted): in-sample {res['insample']['hit_rate_bytes']:.3f}" + (f", held-out {held:.3f}" if held is not None else ", held-out not measured"))
    print(f"wrote {out}\nnext:  bash patches/09_build_pelias_tier.sh   then   bash 13_phase6_run.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
