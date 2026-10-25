#!/usr/bin/env python3
# File: <project>\moe_track\microbench\01_gguf_expert_sizes.py
"""Step 2a: exact per-layer expert byte sizes and quantization types, straight from the GGUF header.

Reads only the header (no weights are loaded, no GPU is touched), so it is safe to run while a server is up.
Handles both layouts the pinned llama.cpp accepts:
    separate  blk.N.ffn_gate_exps / blk.N.ffn_up_exps / blk.N.ffn_down_exps
    fused     blk.N.ffn_gate_up_exps / blk.N.ffn_down_exps
and tells the MTP block (blk.N.nextn.* tensors) apart from the 40 spine layers.

Output: logs/floors/expert_layer_sizes.json  (consumed by 06 waterfill simulator and the gate checks)

    python 01_gguf_expert_sizes.py [--gguf PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
from gguf_min import GGUFError, read_gguf  # noqa: E402

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")
SHEXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight$")
BLK_RE = re.compile(r"^blk\.(\d+)\.")


def sidecar_sha256(gguf_path: Path) -> str:
    side = gguf_path.with_name(gguf_path.name + ".sha256")
    if side.exists():
        parts = side.read_text(encoding="utf-8").split()
        if parts:
            return parts[0].strip()
    return "unknown (no .sha256 sidecar next to the file)"


def build_table(gg) -> dict:
    tensors = gg.tensors
    arch = gg.get("general.architecture", "unknown")
    block_count = int(gg.arch_get("block_count", 0) or 0)
    nextn = int(gg.arch_get("nextn_predict_layers", 0) or 0)
    n_used = gg.arch_get("expert_used_count", None)

    mtp_layers = sorted({int(m.group(1)) for t in tensors if (m := BLK_RE.match(t.name)) and ".nextn." in t.name})

    per_layer: dict[int, dict] = {}
    for t in tensors:
        m = EXPERT_RE.match(t.name)
        if not m:
            continue
        layer, kind = int(m.group(1)), m.group(2)
        if len(t.dims) != 3:
            raise GGUFError(f"{t.name}: expected a 3-D expert tensor, got dims {t.dims}")
        per_layer.setdefault(layer, {})[kind] = t

    if not per_layer:
        raise GGUFError("no blk.N.ffn_*_exps.weight tensors found; is this a MoE GGUF?")

    shexp_bytes: dict[int, int] = {}
    for t in tensors:
        m = SHEXP_RE.match(t.name)
        if m:
            shexp_bytes[int(m.group(1))] = shexp_bytes.get(int(m.group(1)), 0) + t.n_bytes

    layers = []
    for layer in sorted(per_layer):
        parts = per_layer[layer]
        if "down" not in parts or not (("gate_up" in parts) or ("gate" in parts and "up" in parts)):
            raise GGUFError(f"layer {layer}: incomplete expert tensor set {sorted(parts)}")
        n_expert = {p.dims[2] for p in parts.values()}
        if len(n_expert) != 1:
            raise GGUFError(f"layer {layer}: expert counts disagree {n_expert}")
        n_expert = n_expert.pop()
        fused = "gate_up" in parts
        detail = {}
        total = 0
        for kind, t in sorted(parts.items()):
            detail[kind] = {"tensor": t.name, "quant": t.type_name, "dims": list(t.dims),
                            "bytes": t.n_bytes, "bytes_per_expert": t.per_slice_bytes(n_expert)}
            total += t.n_bytes
        layers.append({
            "layer": layer,
            "role": "mtp" if layer in mtp_layers else "spine",
            "fused_gate_up": fused,
            "n_expert": n_expert,
            "down_quant": detail["down"]["quant"],
            "gate_up_quant": detail["gate_up"]["quant"] if fused else detail["gate"]["quant"],
            "tensors": detail,
            "layer_routed_bytes": total,
            "layer_routed_mib": round(total / 2**20, 4),
            "expert_bytes": total // n_expert,
            "expert_mb": round(total / n_expert / 1e6, 6),
            "expert_mib": round(total / n_expert / 2**20, 6),
            "shared_expert_bytes": shexp_bytes.get(layer, 0),
        })

    spine = [l for l in layers if l["role"] == "spine"]
    mtp = [l for l in layers if l["role"] == "mtp"]
    n_expert_all = {l["n_expert"] for l in layers}
    if len(n_expert_all) != 1:
        raise GGUFError(f"expert counts differ between layers: {n_expert_all}")
    n_expert = n_expert_all.pop()

    spine_routed = sum(l["layer_routed_bytes"] for l in spine)
    quant_hist: dict[str, int] = {}
    for l in spine:
        quant_hist[l["down_quant"]] = quant_hist.get(l["down_quant"], 0) + 1

    total_tensor_bytes = sum(t.n_bytes for t in tensors)
    mtp_bytes = sum(t.n_bytes for t in tensors if (m := BLK_RE.match(t.name)) and int(m.group(1)) in mtp_layers)
    mtp_tensor_count = sum(1 for t in tensors if (m := BLK_RE.match(t.name)) and int(m.group(1)) in mtp_layers)
    all_routed = sum(l["layer_routed_bytes"] for l in layers)

    topk = int(n_used) if n_used else 8
    active_bytes = sum(l["expert_bytes"] * topk for l in spine)

    summary = {
        "gguf_path": gg.path,
        "gguf_sha256": sidecar_sha256(Path(gg.path)),
        "file_size": gg.file_size,
        "architecture": arch,
        "block_count_kv": block_count,
        "nextn_predict_layers_kv": nextn,
        "n_layers": len(spine),
        "spine_layer_ids": [l["layer"] for l in spine],
        "mtp_layer_ids": mtp_layers,
        "n_experts_per_layer": n_expert,
        "n_experts_used_per_token": topk,
        "expert_layout": "fused gate_up" if any(l["fused_gate_up"] for l in layers) else "separate gate/up",
        "down_quant_layer_counts": quant_hist,
        "q4k_down_layers": quant_hist.get("Q4_K", 0),
        "q6k_down_layers": quant_hist.get("Q6_K", 0),
        "total_routed_bytes": spine_routed,
        "total_routed_gib": round(spine_routed / 2**30, 4),
        "fleet_avg_expert_bytes": spine_routed // (len(spine) * n_expert),
        "fleet_avg_expert_mb": round(spine_routed / (len(spine) * n_expert) / 1e6, 4),
        "active_gb_per_token_topk": round(active_bytes / 1e9, 4),
        "active_gb_per_token_top8": round(active_bytes / 1e9, 4) if topk == 8 else None,
        "non_expert_spine_bytes": total_tensor_bytes - all_routed - sum(
            t.n_bytes for t in tensors if (m := BLK_RE.match(t.name)) and int(m.group(1)) in mtp_layers and not EXPERT_RE.match(t.name)),
        "mtp_blk_tensor_count": mtp_tensor_count,
        "mtp_blk_bytes": mtp_bytes,
        "mtp_routed_bytes": sum(l["layer_routed_bytes"] for l in mtp),
        "layers": layers,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", default=str(paths.GGUF))
    ap.add_argument("--out", default=str(paths.FLOORS / "expert_layer_sizes.json"))
    args = ap.parse_args()

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        print(f"ERROR: GGUF not found at {gguf_path}", file=sys.stderr)
        return 2
    try:
        gg = read_gguf(gguf_path)
        summary = build_table(gg)
    except GGUFError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[2a] wrote {out}")
    print(f"     architecture {summary['architecture']}; {summary['n_layers']} spine layers x {summary['n_experts_per_layer']} experts "
          f"(top-{summary['n_experts_used_per_token']}); layout: {summary['expert_layout']}")
    print(f"     down_proj quant by layer count: {summary['down_quant_layer_counts']}")
    print(f"     routed pool {summary['total_routed_gib']:.2f} GiB; average expert {summary['fleet_avg_expert_mb']:.3f} MB; "
          f"read per token (all experts cold) {summary['active_gb_per_token_topk']:.3f} GB")
    print(f"     MTP block(s) {summary['mtp_layer_ids']}: {summary['mtp_blk_tensor_count']} tensors, {summary['mtp_blk_bytes']/2**20:.1f} MiB "
          f"(not counted in the spine numbers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
