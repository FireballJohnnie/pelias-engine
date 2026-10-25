#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_gguf_min.py
"""Offline tests for lib/gguf_min.py and microbench/01_gguf_expert_sizes.py.

    python tests/test_gguf_min.py [--gguf-py /path/to/llama.cpp/gguf-py] [--model-dir DIR]

--model-dir may hold tiny_f16.gguf (from make_tiny_moe.py) and, if present, tiny_q4km.gguf (from llama-quantize).
These are tests of the tools, run in a sandbox. They are not part of the measurement steps.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "microbench"))

from gguf_min import GGML_TYPES, read_gguf  # noqa: E402


def load_sizes_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sizes", HERE.parent / "microbench" / "01_gguf_expert_sizes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-py", default="")
    ap.add_argument("--model-dir", default="")
    args = ap.parse_args()
    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            fails += 1

    gguf = None
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    try:
        import gguf as gguf  # type: ignore  # noqa: F811
    except Exception as e:  # pragma: no cover
        print(f"NOTE  gguf-py not importable ({e}); comparisons against it are skipped")

    if gguf is not None:
        from gguf.constants import GGML_QUANT_SIZES
        bad = [(int(q), q.name, GGML_TYPES.get(int(q)), (bs, ts)) for q, (bs, ts) in GGML_QUANT_SIZES.items()
               if GGML_TYPES.get(int(q), (None, None, None))[1:] != (bs, ts)]
        check(not bad, f"type/block-size table equals gguf-py's for all {len(GGML_QUANT_SIZES)} types {bad[:3] if bad else ''}")

    sizes = load_sizes_module()
    model_dir = Path(args.model_dir) if args.model_dir else None

    if model_dir and (model_dir / "tiny_f16.gguf").exists() and gguf is not None:
        p = model_dir / "tiny_f16.gguf"
        mine = read_gguf(p)
        ref = gguf.GGUFReader(str(p))
        ref_t = {t.name: t for t in ref.tensors}
        check(len(mine.tensors) == len(ref_t), f"tensor count {len(mine.tensors)} == gguf-py {len(ref_t)}")
        ok = True
        for t in mine.tensors:
            r = ref_t[t.name]
            if not (t.abs_offset == r.data_offset and t.n_bytes == int(r.n_bytes) and t.type_name == r.tensor_type.name):
                ok = False
                print("   mismatch", t.name, t.abs_offset, r.data_offset, t.n_bytes, int(r.n_bytes), t.type_name, r.tensor_type.name)
        check(ok, "every tensor: offset, byte size and type match gguf-py")
        check(mine.arch_get("expert_count") == 16 and mine.arch_get("expert_used_count") == 4, "metadata expert_count/expert_used_count read")

    if model_dir and (model_dir / "tiny_q4km.gguf").exists():
        p = model_dir / "tiny_q4km.gguf"
        gg = read_gguf(p)
        tbl = sizes.build_table(gg)
        kinds = {l["down_quant"] for l in tbl["layers"]}
        check(tbl["n_layers"] == 4 and tbl["n_experts_per_layer"] == 16, f"quantized tiny model: {tbl['n_layers']} layers x {tbl['n_experts_per_layer']} experts")
        check(kinds <= {"Q4_K", "Q6_K", "Q5_K", "Q5_0", "Q8_0"}, f"down_proj quant types read from a real llama-quantize file: {sorted(kinds)}")
        # expert bytes must equal tensor bytes / n_expert and the sum must equal the file's tensor bytes for experts
        exp_sum = sum(t.n_bytes for t in gg.tensors if "_exps." in t.name)
        check(exp_sum == tbl["total_routed_bytes"], f"routed pool {tbl['total_routed_bytes']} bytes == sum of *_exps tensors {exp_sum}")
        for l in tbl["layers"]:
            if l["expert_bytes"] * l["n_expert"] != l["layer_routed_bytes"]:
                check(False, f"layer {l['layer']} expert_bytes x n_expert != layer bytes")
                break
        else:
            check(True, "expert_bytes x n_expert == layer bytes for every layer")

    # fused gate_up layout + MTP block detection, built from scratch with gguf-py
    if gguf is not None:
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fused.gguf"
            w = gguf.GGUFWriter(str(out), "qwen35moe")
            w.add_block_count(3)
            w.add_uint32("qwen35moe.nextn_predict_layers", 1)
            w.add_expert_count(8)
            w.add_expert_used_count(2)
            n_embd, n_ff, n_exp = 256, 256, 8

            def raw(name, shape, q):
                bshape = gguf.quants.quant_shape_to_byte_shape(shape, q)
                w.add_tensor(name, np.zeros(bshape, dtype=np.uint8), raw_dtype=q)

            Q = gguf.GGMLQuantizationType
            for i in range(3):
                raw(f"blk.{i}.ffn_gate_up_exps.weight", (n_exp, 2 * n_ff, n_embd), Q.Q4_K)
                raw(f"blk.{i}.ffn_down_exps.weight", (n_exp, n_embd, n_ff), Q.Q6_K if i % 2 == 0 else Q.Q4_K)
                raw(f"blk.{i}.ffn_gate_shexp.weight", (n_ff, n_embd), Q.Q8_0)
            w.add_tensor("blk.2.nextn.eh_proj.weight", np.zeros((n_embd, 2 * n_embd), dtype=np.float16))
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            tbl = sizes.build_table(read_gguf(out))
            check(tbl["expert_layout"] == "fused gate_up", "fused gate_up layout detected")
            check(tbl["mtp_layer_ids"] == [2] and tbl["n_layers"] == 2, f"MTP block detected (spine {tbl['n_layers']}, mtp {tbl['mtp_layer_ids']})")
            check(tbl["down_quant_layer_counts"] == {"Q6_K": 1, "Q4_K": 1}, f"down quant histogram counts spine layers only {tbl['down_quant_layer_counts']}")
            gate_up = 8 * 2 * n_ff * n_embd // 256 * 144
            down_q6 = 8 * n_embd * n_ff // 256 * 210
            check(tbl["layers"][0]["layer_routed_bytes"] == gate_up + down_q6, "layer 0 bytes = fused gate_up(Q4_K) + down(Q6_K) by hand")
            check(tbl["layers"][0]["expert_bytes"] == (gate_up + down_q6) // 8, "expert bytes = layer bytes / n_expert")

    print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
