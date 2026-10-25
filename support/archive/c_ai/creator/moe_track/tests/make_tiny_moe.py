#!/usr/bin/env python3
# File: <project>\moe_track\tests\make_tiny_moe.py
"""Build a tiny random `qwen3moe` GGUF (same MoE graph code path as the real model) for offline tests.

    python make_tiny_moe.py OUT.gguf [--gguf-py /path/to/llama.cpp/gguf-py] [--layers 4] [--experts 16] [--topk 4]

It is a TEST FIXTURE: random weights, no vocabulary (tokenizer "none") unless --vocab-from copies a real tokenizer. The router is deliberately skewed so
that some experts are used far more than others, like the real model's routing.
Dimensions are multiples of 256 so llama-quantize can make real Q4_K / Q6_K tensors from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gguf-py", default="")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--ff-exp", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skew", type=float, default=1.5, help="router column scale spread; larger = more skewed routing")
    ap.add_argument("--vocab-from", default="", help="copy the tokenizer of this vocab-only GGUF (e.g. llama.cpp/models/ggml-vocab-llama-spm.gguf); sets --vocab to its size")
    args = ap.parse_args()

    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    import gguf  # noqa: E402

    vocab_fields = []
    if args.vocab_from:
        rd = gguf.GGUFReader(args.vocab_from)
        for key, field in rd.fields.items():
            if key.startswith("tokenizer."):
                vocab_fields.append((key, field))
        toks = rd.fields["tokenizer.ggml.tokens"]
        args.vocab = len(toks.data)
    rng = np.random.default_rng(args.seed)
    n_embd, n_ff, n_exp, n_layer, vocab = args.embd, args.ff_exp, args.experts, args.layers, args.vocab
    n_head, n_kv, hd = 4, 2, 64

    w = gguf.GGUFWriter(args.out, "qwen3moe")
    w.add_name("tiny-qwen3moe-test")
    w.add_block_count(n_layer)
    w.add_context_length(2048)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff * args.topk)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_key_length(hd)
    w.add_value_length(hd)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    w.add_expert_count(n_exp)
    w.add_expert_used_count(args.topk)
    w.add_expert_feed_forward_length(n_ff)
    w.add_vocab_size(vocab)
    if vocab_fields:
        for key, field in vocab_fields:
            vt = field.types[0]
            if vt == gguf.GGUFValueType.ARRAY:
                w.add_key_value(key, field.contents(), vt, sub_type=field.types[-1])
            else:
                w.add_key_value(key, field.contents(), vt)
    else:
        w.add_tokenizer_model("none")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    def f32(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def f16(*shape, scale=0.05):
        return f32(*shape, scale=scale).astype(np.float16)

    w.add_tensor("token_embd.weight", f16(vocab, n_embd, scale=0.2))
    w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor("output.weight", f16(vocab, n_embd, scale=0.2))
    col_scale = np.exp(rng.standard_normal(n_exp) * args.skew).astype(np.float32)  # skewed popularity
    for i in range(n_layer):
        b = f"blk.{i}"
        w.add_tensor(f"{b}.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor(f"{b}.attn_q.weight", f16(n_head * hd, n_embd))
        w.add_tensor(f"{b}.attn_k.weight", f16(n_kv * hd, n_embd))
        w.add_tensor(f"{b}.attn_v.weight", f16(n_kv * hd, n_embd))
        w.add_tensor(f"{b}.attn_output.weight", f16(n_embd, n_head * hd))
        w.add_tensor(f"{b}.attn_q_norm.weight", np.ones(hd, dtype=np.float32))
        w.add_tensor(f"{b}.attn_k_norm.weight", np.ones(hd, dtype=np.float32))
        w.add_tensor(f"{b}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        gate_inp = f32(n_exp, n_embd, scale=0.1) * col_scale[:, None]
        w.add_tensor(f"{b}.ffn_gate_inp.weight", gate_inp.astype(np.float32))
        w.add_tensor(f"{b}.ffn_gate_exps.weight", f16(n_exp, n_ff, n_embd))
        w.add_tensor(f"{b}.ffn_up_exps.weight", f16(n_exp, n_ff, n_embd))
        w.add_tensor(f"{b}.ffn_down_exps.weight", f16(n_exp, n_embd, n_ff))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB): {n_layer} layers, {n_exp} experts, top-{args.topk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
