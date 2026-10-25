#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_mtp_torch_fake.py
"""Runs mtp_align/mtp_torch.py (the PyTorch MTP head) on the numpy-backed stand-in for torch (tests/fake_torch.py) and compares it with the numpy reference.
This checks the forward math, indexing and shapes. It cannot check gradients, GPUs or bfloat16; the real `10_train_blk40_mtp_lora.py selftest` does that on your machine.

    python tests/test_mtp_torch_fake.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mtp_align"))
import fake_torch  # noqa: E402

torch = fake_torch.install()
import mtp_block_numpy as M  # noqa: E402
import mtp_torch as MT  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def small_cfg():
    return M.Config(hidden=64, n_heads=4, n_kv_heads=2, head_dim=16, rot_dim=8, rope_base=1.0e4, n_experts=8, top_k=2, ff_exp=16, ff_shared=16)


def to_t(params):
    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in params.items()}


def main() -> int:
    cfg = small_cfg()
    p = M.random_params(cfg, seed=21, scale=0.08)
    r = np.random.default_rng(22)
    T = 8
    h = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e1 = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e2 = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    pos = np.arange(T) + 1
    ref1, ref2 = M.mtp_forward_two_step(p, cfg, h, e1, e2, pos)

    head = MT.MTPHead(to_t(p), cfg, r=4, alpha=8.0, seed=5)
    tt = torch.from_numpy
    O1, K1, V1 = head.step1(tt(h), tt(e1), tt(pos.astype(np.int64)))
    O2 = head.step2(O1, tt(e2), tt(pos.astype(np.int64)), K1, V1)
    check("step 1 equals the numpy reference (LoRA B = 0)", np.allclose(O1.data, ref1, atol=2e-4), f"max diff {np.abs(O1.data - ref1).max():.2e}")
    check("step 2 equals the numpy reference (LoRA B = 0)", np.allclose(O2.data, ref2, atol=2e-4), f"max diff {np.abs(O2.data - ref2).max():.2e}")

    # trainable parameters are exactly the LoRA matrices: 8 modules x (A, B)
    names = [n for n, q in head.named_parameters() if q.requires_grad]
    check("16 trainable tensors (A and B for 8 modules)", len(names) == 16 and all(("lora_A" in n or "lora_B" in n) for n in names), str(names))
    frozen = [n for n, q in head.named_parameters() if not q.requires_grad]
    check("everything else is frozen (8 base weights + norms/router/experts)", len(frozen) == 8 + 11, f"{len(frozen)}")

    # random LoRA -> same output as the numpy model with the merged weights (this is what 11_merge_blk40_and_export.py writes)
    rg = np.random.default_rng(23)
    merged = {k: v.copy() for k, v in p.items()}
    for name, m in head.lin.items():
        m.lora_A.data[...] = rg.standard_normal(m.lora_A.shape).astype(np.float32) * 0.3
        m.lora_B.data[...] = rg.standard_normal(m.lora_B.shape).astype(np.float32) * 0.3
        merged[name] = merged[name] + m.scale * (m.lora_B.data @ m.lora_A.data)
        check(f"merged_weight({name}) = W + scale*B@A", np.allclose(m.merged_weight().data, merged[name], atol=1e-5))
    rm1, rm2 = M.mtp_forward_two_step(merged, cfg, h, e1, e2, pos)
    O1, K1, V1 = head.step1(tt(h), tt(e1), tt(pos.astype(np.int64)))
    O2 = head.step2(O1, tt(e2), tt(pos.astype(np.int64)), K1, V1)
    check("LoRA model == numpy model with merged weights (step 1)", np.allclose(O1.data, rm1, atol=3e-4), f"{np.abs(O1.data - rm1).max():.2e}")
    check("LoRA model == numpy model with merged weights (step 2)", np.allclose(O2.data, rm2, atol=3e-4), f"{np.abs(O2.data - rm2).max():.2e}")
    check("LoRA actually changed the output", not np.allclose(O1.data, ref1, atol=1e-3))

    # lora_state round trip
    st = head.lora_state()
    head2 = MT.MTPHead(to_t(p), cfg, r=4, alpha=8.0, seed=99)
    head2.load_lora_state({k: (tt(a.data), tt(b.data)) for k, (a, b) in st.items()})
    P1, _, _ = head2.step1(tt(h), tt(e1), tt(pos.astype(np.int64)))
    check("adapter state round trip reproduces the output", np.allclose(P1.data, O1.data, atol=1e-6))
    try:
        bad = MT.MTPHead(to_t(p), cfg, r=2, alpha=4.0)
        bad.load_lora_state(st)
        check("adapter with the wrong rank is refused", False)
    except ValueError:
        check("adapter with the wrong rank is refused", True)

    # loss statistics against a plain numpy computation
    V, D = 50, cfg.hidden
    W = (r.standard_normal((V, D)) * 0.2).astype(np.float32)
    hs = r.standard_normal((11, D)).astype(np.float32)
    ht = r.standard_normal((11, D)).astype(np.float32)
    y = r.integers(0, V, size=11)
    st = MT.token_stats(tt(hs), tt(ht), tt(y.astype(np.int64)), tt(W)).data
    zs, zt = hs @ W.T, ht @ W.T
    ls = zs - np.log(np.exp(zs - zs.max(1, keepdims=True)).sum(1, keepdims=True)) - zs.max(1, keepdims=True)
    lt = zt - np.log(np.exp(zt - zt.max(1, keepdims=True)).sum(1, keepdims=True)) - zt.max(1, keepdims=True)
    pt_, ps_ = np.exp(lt), np.exp(ls)
    kl = (pt_ * (lt - ls)).sum(1)
    ce = -ls[np.arange(11), y]
    agree = (zs.argmax(1) == zt.argmax(1)).astype(np.float32)
    ov = np.minimum(pt_, ps_).sum(1)
    check("token_stats: KL", np.allclose(st[:, 0], kl, atol=1e-4))
    check("token_stats: cross-entropy", np.allclose(st[:, 1], ce, atol=1e-4))
    check("token_stats: top-1 agreement", np.allclose(st[:, 2], agree))
    check("token_stats: acceptance overlap", np.allclose(st[:, 3], ov, atol=1e-5))
    tot = MT.run_chunks(tt(hs), tt(ht), tt(y.astype(np.int64)), tt(W), chunk=4, use_checkpoint=True).data
    check("run_chunks sums the chunks", np.allclose(tot, st.sum(0), atol=1e-3))
    same = MT.token_stats(tt(ht), tt(ht), tt(y.astype(np.int64)), tt(W)).data
    check("identical student and teacher: KL 0, agree 1, overlap 1", np.allclose(same[:, 0], 0, atol=1e-5) and np.allclose(same[:, 2], 1) and np.allclose(same[:, 3], 1, atol=1e-5))

    # config from the real config.json layout and tensor shapes
    hf = {"text_config": {"partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6, "num_experts_per_tok": 8,
                          "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}}
    shapes = {"q_norm": (256,), "q_proj": (8192, 2048), "k_proj": (512, 2048), "fc": (2048, 4096), "router": (256, 2048), "down": (256, 2048, 512), "shared_gate": (512, 2048)}
    c, src = MT.hf_cfg_to_config(hf, shapes)
    check("config from the real shapes: 16 heads, 2 kv heads, head 256, rot 64, 256 experts, ff 512",
          (c.n_heads, c.n_kv_heads, c.head_dim, c.rot_dim, c.n_experts, c.ff_exp, c.hidden, c.top_k) == (16, 2, 256, 64, 256, 512, 2048, 8))
    check("rope_theta read from the nested rope_parameters", c.rope_base == 1.0e7 and src["rope_theta"] == "config")
    check("norm_topk_prob reported as a default when the file lacks it", src["norm_topk_prob"] == "default" and c.norm_topk)

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
