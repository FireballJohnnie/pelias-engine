#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_mtp_numpy.py
"""Self-checks of the numpy MTP reference (mtp_align/mtp_block_numpy.py) and of lib/stensors.py. Runs in a second; needs only numpy.

    python tests/test_mtp_numpy.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mtp_align"))
sys.path.insert(0, str(HERE.parent / "lib"))
import mtp_block_numpy as M  # noqa: E402
import stensors as S  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def small_cfg():
    return M.Config(hidden=64, n_heads=4, n_kv_heads=2, head_dim=16, rot_dim=8, rope_base=1.0e4, n_experts=8, top_k=2, ff_exp=16, ff_shared=16)


def test_norm():
    r = np.random.default_rng(0)
    x = r.standard_normal((5, 32)).astype(np.float32)
    y = M.rms_norm(x, np.zeros(32, np.float32), 1e-6, True)
    ref = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
    check("rms_norm with zero weight is the plain RMSNorm (zero-centred)", np.allclose(y, ref, atol=1e-6))
    w = r.standard_normal(32).astype(np.float32)
    y2 = M.rms_norm(x, w, 1e-6, True)
    check("rms_norm scale is 1+w", np.allclose(y2, ref * (1 + w), atol=1e-5))
    y3 = M.rms_norm(x, w, 1e-6, False)
    check("rms_norm plain scale is w", np.allclose(y3, ref * w, atol=1e-5))


def test_rope():
    r = np.random.default_rng(1)
    T, H, D, rd, base = 7, 3, 16, 8, 1.0e4
    x = r.standard_normal((T, H, D)).astype(np.float32)
    pos = np.arange(T) + 5
    y = M.rope_half_split(x, pos, rd, base)
    ok = True
    for t in range(T):
        for i in range(rd // 2):
            ang = pos[t] * base ** (-2.0 * i / rd)
            c, s = np.cos(ang), np.sin(ang)
            a, b = x[t, :, i], x[t, :, i + rd // 2]
            ok &= np.allclose(y[t, :, i], a * c - b * s, atol=1e-5) and np.allclose(y[t, :, i + rd // 2], b * c + a * s, atol=1e-5)
    check("rope pairs (i, i+rot/2) rotate by pos * base^(-2i/rot)", bool(ok))
    check("rope leaves the dims past rot_dim alone", np.array_equal(y[..., rd:], x[..., rd:]))
    n0 = np.linalg.norm(x[..., :rd], axis=-1)
    n1 = np.linalg.norm(y[..., :rd], axis=-1)
    check("rope keeps vector length", np.allclose(n0, n1, atol=1e-4))


def test_forward():
    cfg = small_cfg()
    p = M.random_params(cfg, seed=3, scale=0.08)
    r = np.random.default_rng(4)
    T = 9
    h = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    pos = np.arange(T)
    out, parts = M.mtp_forward(p, cfg, h, e, pos, return_parts=True)
    check("output shape", out.shape == (T, cfg.hidden))
    check("output is finite", bool(np.isfinite(out).all()))
    check("router picks top_k distinct experts per token", parts["moe_ids"].shape == (T, cfg.top_k) and all(len(set(row)) == cfg.top_k for row in parts["moe_ids"]))
    check("top-k weights are renormalised to 1", np.allclose(parts["moe_w"].sum(-1), 1.0, atol=1e-5))

    # causal: changing future inputs does not change earlier outputs
    h2, e2 = h.copy(), e.copy()
    h2[6:] += 1.0
    e2[6:] -= 0.5
    out2 = M.mtp_forward(p, cfg, h2, e2, pos)
    check("causality: positions before the change are untouched", np.allclose(out[:6], out2[:6], atol=1e-5))
    check("causality: positions after the change do change", not np.allclose(out[6:], out2[6:], atol=1e-3))

    # relative position: a uniform shift of the position ids changes nothing
    out3 = M.mtp_forward(p, cfg, h, e, pos + 1000)
    check("uniform position shift is invisible (rotary is relative)", np.allclose(out, out3, atol=2e-3), f"max diff {np.abs(out - out3).max():.2e}")

    # batch == one-at-a-time with the full prefix (what the server does step by step)
    last = M.mtp_forward(p, cfg, h[:5], e[:5], pos[:5])[-1]
    check("a prefix run reproduces the same row as the full run", np.allclose(last, out[4], atol=1e-5))

    # attention gate: with the gate half of q_proj zeroed every gate logit is 0 -> sigmoid = 0.5 -> the output is exactly half of the ungated attention
    p2 = {k: v.copy() for k, v in p.items()}
    H, D = cfg.n_heads, cfg.head_dim
    q = p2["q_proj"].reshape(H, 2 * D, cfg.hidden)
    q[:, D:, :] = 0.0
    p2["q_proj"] = q.reshape(H * 2 * D, cfg.hidden)
    xn = np.random.default_rng(5).standard_normal((4, cfg.hidden)).astype(np.float32)
    gated = M.attention(p2, cfg, xn, np.arange(4))
    real_sigmoid = M.sigmoid
    M.sigmoid = lambda x: np.ones_like(x)
    try:
        ungated = M.attention(p2, cfg, xn, np.arange(4))
    finally:
        M.sigmoid = real_sigmoid
    check("sigmoid output gate: logits 0 -> exactly half of the ungated attention", np.allclose(gated, 0.5 * ungated, atol=1e-6) and np.abs(ungated).max() > 1e-4)

    # shared expert gate: forcing the gate to -inf removes the shared path
    p4 = {k: v.copy() for k, v in p.items()}
    p4["shared_expert_gate"] = np.full_like(p4["shared_expert_gate"], 0.0)
    xs = np.random.default_rng(6).standard_normal((3, cfg.hidden)).astype(np.float32)
    m0, ids, w = M.moe(p4, cfg, xs)
    # with gate logit 0 the sigmoid is exactly 0.5: output = routed + 0.5 * shared
    routed = np.zeros_like(xs)
    for t in range(3):
        for j in range(cfg.top_k):
            e_ = int(ids[t, j])
            gu = p4["gate_up"][e_] @ xs[t]
            hh = M.silu(gu[: cfg.ff_exp]) * gu[cfg.ff_exp:]
            routed[t] += w[t, j] * (p4["down"][e_] @ hh)
    shared = p4["shared_down"] @ (M.silu(p4["shared_gate"] @ xs.T) * (p4["shared_up"] @ xs.T))
    check("moe = routed + sigmoid(gate) * shared (gate logit 0 -> half)", np.allclose(m0, routed + 0.5 * shared.T, atol=1e-5))


def test_two_step():
    """Step 2 compared with a deliberately naive, per-entry implementation (one query at a time, explicit key list)."""
    cfg = small_cfg()
    p = M.random_params(cfg, seed=11, scale=0.08)
    r = np.random.default_rng(12)
    T = 6
    h = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e1 = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e2 = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    pos = np.arange(T) + 3
    O1, O2 = M.mtp_forward_two_step(p, cfg, h, e1, e2, pos)
    O1b, parts = M.mtp_forward(p, cfg, h, e1, pos, return_parts=True)
    check("two-step: step 1 equals the plain forward", np.allclose(O1, O1b, atol=1e-6))
    K1, V1 = parts["k"], parts["v"]
    Hh, KV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    ok = True
    worst = 0.0
    for t in range(T):
        x2 = M._front(p, cfg, O1[t:t + 1], e2[t:t + 1])
        xn = M.rms_norm(x2, p["input_layernorm"], cfg.eps, True)
        q, k, v, gate = M.attn_qkv(p, cfg, xn, np.array([pos[t] + 1]))
        keys = np.concatenate([K1[: t + 1], k], axis=0)                       # [t+2, KV, D]
        vals = np.concatenate([V1[: t + 1], v], axis=0)
        out = np.zeros((Hh, D), dtype=np.float32)
        for hh in range(Hh):
            kv = hh // (Hh // KV)
            sc = (keys[:, kv, :] @ q[0, hh, :]) / np.sqrt(D)
            w = M.softmax(sc.astype(np.float32), axis=-1)
            out[hh] = w @ vals[:, kv, :]
        a = (out.reshape(1, Hh * D) * M.sigmoid(gate)) @ p["o_proj"].T
        ref, _, _ = M._back(p, cfg, x2, a.astype(np.float32))
        d = float(np.abs(ref[0] - O2[t]).max())
        worst = max(worst, d)
        ok &= d < 1e-4
    check("two-step: step 2 matches the naive one-entry-at-a-time version", bool(ok), f"worst {worst:.2e}")
    # step 2 of entry 0 depends on nothing but entry 0
    h2, e12, e22 = h.copy(), e1.copy(), e2.copy()
    h2[1:] += 1.0
    e12[1:] -= 1.0
    e22[1:] += 0.7
    _, O2b = M.mtp_forward_two_step(p, cfg, h2, e12, e22, pos)
    check("two-step: entry 0 does not see later entries", np.allclose(O2[0], O2b[0], atol=1e-5))
    check("two-step: entry 1 does", not np.allclose(O2[1], O2b[1], atol=1e-3))


def test_names():
    cfg = small_cfg()
    p = M.random_params(cfg)
    check("every parameter has exactly one Hugging Face name", sorted(M.HF_NAME_MAP.values()) == sorted(p.keys()))


def test_stensors():
    r = np.random.default_rng(7)
    a = r.standard_normal((5, 3)).astype(np.float32)
    b = r.standard_normal((4,)).astype(np.float32)
    i = np.arange(6, dtype=np.int32).reshape(2, 3)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.safetensors"
        S.save_tensors(p, {"a": a, "b": b, "i": i}, dtypes={"a": "BF16", "b": "F16"}, metadata={"format": "pt"})
        with S.SafeFile(p) as f:
            check("stensors keys", f.keys() == ["a", "b", "i"])
            check("stensors dtype+shape", f.info("a") == ("BF16", (5, 3)) and f.info("b") == ("F16", (4,)) and f.info("i") == ("I32", (2, 3)))
            ga = f.get("a")
            check("bf16 round trip within bf16 precision", np.allclose(ga, a, rtol=8e-3, atol=1e-6))
            check("bf16 round trip is idempotent", np.array_equal(S.bf16_to_f32(S.f32_to_bf16(ga)), ga))
            check("f16 round trip", np.allclose(f.get("b"), b, rtol=1e-3, atol=1e-4))
            check("int tensor exact", np.array_equal(f.get("i"), i))
            check("metadata kept", f.metadata.get("format") == "pt")
        try:
            S.save_tensors(p, {"a": a})
            check("save refuses to overwrite", False)
        except FileExistsError:
            check("save refuses to overwrite", True)
        # a file written by the real library has the same layout: header padded to 8 bytes and sorted names; check the padding rule
        raw = p.read_bytes()
        n = int.from_bytes(raw[:8], "little")
        check("header length is a multiple of 8", n % 8 == 0)
    # bf16 rounding: ties to even
    x = np.array([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, np.nan, -0.0], dtype=np.float32)
    y = S.bf16_to_f32(S.f32_to_bf16(x))
    check("bf16 ties-to-even", y[0] == 1.0 and y[1] == 1.0 + 2 ** -6 and np.isnan(y[2]) and np.signbit(y[3]))


def main() -> int:
    for fn in (test_norm, test_rope, test_forward, test_two_step, test_names, test_stensors):
        fn()
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
