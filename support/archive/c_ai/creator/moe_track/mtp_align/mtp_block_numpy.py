#!/usr/bin/env python3
# File: <project>\moe_track\mtp_align\mtp_block_numpy.py
"""The Qwen3.6-35B-A3B MTP (draft) head in plain numpy: the reference that the PyTorch trainer (10_train_blk40_mtp_lora.py) is checked against.

It follows, line for line, the graph that llama.cpp builds for it (src/models/qwen35moe.cpp, class graph_mtp, commit 2ca15f5) and the way the llama.cpp
converter maps the Hugging Face tensor names (conversion/qwen.py: mtp.fc -> eh_proj, mtp.pre_fc_norm_embedding -> enorm, mtp.pre_fc_norm_hidden -> hnorm,
mtp.norm -> shared_head.norm, every "*norm.weight" gets +1). Real shapes of this model (read from the downloaded checkpoint):

  mtp.fc.weight                              [2048, 4096]     input = concat(norm(embedding of the NEXT token), norm(hidden state of the CURRENT position))
  mtp.pre_fc_norm_embedding.weight           [2048]           RMSNorm, "zero-centred": the scale is (1 + weight)
  mtp.pre_fc_norm_hidden.weight              [2048]
  mtp.layers.0.input_layernorm.weight        [2048]
  mtp.layers.0.self_attn.q_proj.weight       [8192, 2048]     16 heads x (256 query + 256 output-gate), interleaved per head
  mtp.layers.0.self_attn.k_proj.weight       [512, 2048]      2 kv heads x 256
  mtp.layers.0.self_attn.v_proj.weight       [512, 2048]
  mtp.layers.0.self_attn.o_proj.weight       [2048, 4096]
  mtp.layers.0.self_attn.q_norm.weight       [256]            per-head RMSNorm, (1 + weight)
  mtp.layers.0.self_attn.k_norm.weight       [256]
  mtp.layers.0.post_attention_layernorm.weight [2048]
  mtp.layers.0.mlp.gate.weight               [256, 2048]      router: softmax over 256 experts, top 8, weights renormalised to sum 1
  mtp.layers.0.mlp.experts.gate_up_proj      [256, 1024, 2048]  rows 0..511 = gate, rows 512..1023 = up
  mtp.layers.0.mlp.experts.down_proj         [256, 2048, 512]
  mtp.layers.0.mlp.shared_expert.{gate,up}_proj.weight [512, 2048]; down_proj.weight [2048, 512]
  mtp.layers.0.mlp.shared_expert_gate.weight [1, 2048]        sigmoid gate on the shared expert's output
  mtp.norm.weight                            [2048]           final RMSNorm (1 + weight), then the shared lm_head

Rotary embedding: only the first 64 of the 256 head dimensions are rotated (partial_rotary_factor 0.25), pairs (i, i+32) ("half split"), base 1e7. For text the three
multimodal-rope axes carry the same position, so this equals plain rope.

Everything is float32 here (the checkpoint is bfloat16). `Config` holds the sizes so tests can run a small version of the same code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Config:
    hidden: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    rot_dim: int = 64                  # head_dim * partial_rotary_factor
    rope_base: float = 1.0e7
    n_experts: int = 256
    top_k: int = 8
    ff_exp: int = 512
    ff_shared: int = 512
    eps: float = 1.0e-6
    norm_topk: bool = True
    zero_centred_norm: bool = True     # HF stores (scale - 1); the effective scale is 1 + weight


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float, zero_centred: bool = True) -> np.ndarray:
    """RMSNorm over the last axis. x: [..., D]; w: [D]. Zero-centred: y = x / rms(x) * (1 + w)."""
    x = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps))
    scale = (1.0 + w.astype(np.float32)) if zero_centred else w.astype(np.float32)
    return (x * inv * scale).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def rope_half_split(x: np.ndarray, pos: np.ndarray, rot_dim: int, base: float) -> np.ndarray:
    """x: [T, H, D]; rotates the first rot_dim dims with pairs (i, i + rot_dim/2). pos: [T] integer positions."""
    half = rot_dim // 2
    inv_freq = (base ** (-(np.arange(half, dtype=np.float64) * 2.0 / rot_dim))).astype(np.float64)
    ang = pos.astype(np.float64)[:, None] * inv_freq[None, :]            # [T, half]
    cos, sin = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    out = x.copy()
    a, b = x[..., :half], x[..., half:rot_dim]
    out[..., :half] = a * cos - b * sin
    out[..., half:rot_dim] = b * cos + a * sin
    return out


def attn_qkv(p: Dict[str, np.ndarray], cfg: Config, xn: np.ndarray, pos: np.ndarray):
    """Projections of one pass: returns q [T,H,D], k [T,KV,D], v [T,KV,D] (q and k normed and rotated) and the output gate logits [T,H*D]."""
    T = xn.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    qfull = (xn @ p["q_proj"].T).reshape(T, H, 2 * D)
    q, gate = qfull[..., :D], qfull[..., D:]
    k = (xn @ p["k_proj"].T).reshape(T, KV, D)
    v = (xn @ p["v_proj"].T).reshape(T, KV, D)
    q = rms_norm(q, p["q_norm"], cfg.eps, cfg.zero_centred_norm)
    k = rms_norm(k, p["k_norm"], cfg.eps, cfg.zero_centred_norm)
    q = rope_half_split(q, pos, cfg.rot_dim, cfg.rope_base)
    k = rope_half_split(k, pos, cfg.rot_dim, cfg.rope_base)
    return q, k, v, gate.reshape(T, H * D)


def attention(p: Dict[str, np.ndarray], cfg: Config, xn: np.ndarray, pos: np.ndarray, return_kv: bool = False):
    T = xn.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q, k, v, gate = attn_qkv(p, cfg, xn, pos)
    group = H // KV
    out = np.zeros((T, H, D), dtype=np.float32)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)       # causal
    for h in range(H):
        kv = h // group
        s = (q[:, h, :] @ k[:, kv, :].T) / np.float32(np.sqrt(D)) + mask
        out[:, h, :] = softmax(s, axis=-1) @ v[:, kv, :]
    out = out.reshape(T, H * D) * sigmoid(gate)                            # output gate
    res = (out @ p["o_proj"].T).astype(np.float32)
    return (res, k, v) if return_kv else res


def attention_step2(p: Dict[str, np.ndarray], cfg: Config, xn2: np.ndarray, pos2: np.ndarray, K1: np.ndarray, V1: np.ndarray) -> np.ndarray:
    """Second draft step. Entry t is ONE new token at position pos2[t]; it sees the first-pass keys/values of entries 0..t (K1, V1: [T,KV,D], already rotated) and its own."""
    T = xn2.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q, k, v, gate = attn_qkv(p, cfg, xn2, pos2)
    group = H // KV
    out = np.zeros((T, H, D), dtype=np.float32)
    root = np.float32(np.sqrt(D))
    for h in range(H):
        kv = h // group
        s1 = (q[:, h, :] @ K1[:, kv, :].T) / root                          # [T, T]
        s1 = s1 + np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)  # entry t sees first-pass entries 0..t
        s_self = np.sum(q[:, h, :] * k[:, kv, :], axis=-1, keepdims=True) / root
        w = softmax(np.concatenate([s1, s_self], axis=1), axis=-1)
        out[:, h, :] = w[:, :T] @ V1[:, kv, :] + w[:, T:] * v[:, kv, :]
    out = out.reshape(T, H * D) * sigmoid(gate)
    return (out @ p["o_proj"].T).astype(np.float32)


def moe(p: Dict[str, np.ndarray], cfg: Config, xn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (output [T, D], topk ids [T, K], topk weights [T, K])."""
    T = xn.shape[0]
    logits = xn @ p["router"].T
    probs = softmax(logits, axis=-1)
    ids = np.argsort(-probs, axis=-1, kind="stable")[:, : cfg.top_k]
    w = np.take_along_axis(probs, ids, axis=-1)
    if cfg.norm_topk:
        w = w / np.sum(w, axis=-1, keepdims=True)
    F = cfg.ff_exp
    out = np.zeros_like(xn, dtype=np.float32)
    for t in range(T):
        acc = np.zeros(xn.shape[1], dtype=np.float32)
        for j in range(cfg.top_k):
            e = int(ids[t, j])
            gu = p["gate_up"][e] @ xn[t]                                   # [2F]
            h = silu(gu[:F]) * gu[F:]
            acc = acc + w[t, j] * (p["down"][e] @ h)
        out[t] = acc
    sh = p["shared_down"] @ (silu(p["shared_gate"] @ xn.T) * (p["shared_up"] @ xn.T))   # [D, T]
    gate = sigmoid(xn @ p["shared_expert_gate"].reshape(-1))                            # [T]
    out = out + sh.T * gate[:, None]
    return out.astype(np.float32), ids, w.astype(np.float32)


def _front(p, cfg, h_prev, emb_next):
    e = rms_norm(emb_next, p["pre_fc_norm_embedding"], cfg.eps, cfg.zero_centred_norm)
    h = rms_norm(h_prev, p["pre_fc_norm_hidden"], cfg.eps, cfg.zero_centred_norm)
    return np.concatenate([e, h], axis=-1) @ p["fc"].T


def _back(p, cfg, x_attn_in, attn_out):
    x = x_attn_in + attn_out
    xn2 = rms_norm(x, p["post_attention_layernorm"], cfg.eps, cfg.zero_centred_norm)
    m, ids, w = moe(p, cfg, xn2)
    x = x + m
    return rms_norm(x, p["norm"], cfg.eps, cfg.zero_centred_norm), ids, w


def mtp_forward(p: Dict[str, np.ndarray], cfg: Config, h_prev: np.ndarray, emb_next: np.ndarray, pos: np.ndarray,
                return_parts: bool = False):
    """One draft step for a whole sequence at once.
    h_prev:   [T, D]  post-final-norm hidden state of the main model at position t (the state that produced token t+1)
    emb_next: [T, D]  embedding of token t+1 (already sampled/known)
    pos:      [T]     position ids. llama.cpp gives the draft step the position of the TOKEN (t+1); only differences matter (rotary is relative), so any
                      consecutive run works.
    Returns the hidden state after the head's own final norm [T, D]; the logits are that times the (shared) lm_head matrix transposed."""
    x = _front(p, cfg, h_prev, emb_next)
    xn = rms_norm(x, p["input_layernorm"], cfg.eps, cfg.zero_centred_norm)
    a, k, v = attention(p, cfg, xn, pos, return_kv=True)
    out, ids, w = _back(p, cfg, x, a)
    if return_parts:
        return out, {"x_after_fc": x, "moe_ids": ids, "moe_w": w, "k": k, "v": v}
    return out


def mtp_forward_two_step(p: Dict[str, np.ndarray], cfg: Config, h_prev: np.ndarray, emb_next: np.ndarray, emb_next2: np.ndarray, pos: np.ndarray):
    """Both draft steps the server takes when --spec-draft-n-max is 2 (teacher forced: the drafted token is replaced by the true one).
    Step 1 is mtp_forward. Step 2 for entry t: the head's OWN output O1[t] plays the part of the main model's hidden state, the token is emb_next2[t] (token t+2),
    the position is pos[t]+1, and it attends to the step-1 keys/values of entries 0..t plus itself.
    Returns (O1 [T,D], O2 [T,D])."""
    O1, parts = mtp_forward(p, cfg, h_prev, emb_next, pos, return_parts=True)
    x2 = _front(p, cfg, O1, emb_next2)
    xn2 = rms_norm(x2, p["input_layernorm"], cfg.eps, cfg.zero_centred_norm)
    a2 = attention_step2(p, cfg, xn2, pos + 1, parts["k"], parts["v"])
    O2, _, _ = _back(p, cfg, x2, a2)
    return O1, O2


def random_params(cfg: Config, seed: int = 0, scale: float = 0.05) -> Dict[str, np.ndarray]:
    """Small random weights with the right shapes, for tests."""
    r = np.random.default_rng(seed)
    D, H, KV, HD = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    def n(*shape, s=scale):
        return (r.standard_normal(shape) * s).astype(np.float32)

    return {
        "fc": n(D, 2 * D), "pre_fc_norm_embedding": n(D, s=0.2), "pre_fc_norm_hidden": n(D, s=0.2),
        "input_layernorm": n(D, s=0.2), "q_proj": n(H * 2 * HD, D), "k_proj": n(KV * HD, D), "v_proj": n(KV * HD, D), "o_proj": n(D, H * HD),
        "q_norm": n(HD, s=0.2), "k_norm": n(HD, s=0.2), "post_attention_layernorm": n(D, s=0.2),
        "router": n(cfg.n_experts, D, s=0.3), "gate_up": n(cfg.n_experts, 2 * cfg.ff_exp, D), "down": n(cfg.n_experts, D, cfg.ff_exp),
        "shared_gate": n(cfg.ff_shared, D), "shared_up": n(cfg.ff_shared, D), "shared_down": n(D, cfg.ff_shared),
        "shared_expert_gate": n(1, D, s=0.2), "norm": n(D, s=0.2),
    }


# Hugging Face tensor name (after the "mtp." prefix) -> key in the parameter dict above
HF_NAME_MAP = {
    "fc.weight": "fc", "pre_fc_norm_embedding.weight": "pre_fc_norm_embedding", "pre_fc_norm_hidden.weight": "pre_fc_norm_hidden", "norm.weight": "norm",
    "layers.0.input_layernorm.weight": "input_layernorm", "layers.0.self_attn.q_proj.weight": "q_proj", "layers.0.self_attn.k_proj.weight": "k_proj",
    "layers.0.self_attn.v_proj.weight": "v_proj", "layers.0.self_attn.o_proj.weight": "o_proj", "layers.0.self_attn.q_norm.weight": "q_norm",
    "layers.0.self_attn.k_norm.weight": "k_norm", "layers.0.post_attention_layernorm.weight": "post_attention_layernorm",
    "layers.0.mlp.gate.weight": "router", "layers.0.mlp.experts.gate_up_proj": "gate_up", "layers.0.mlp.experts.down_proj": "down",
    "layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate", "layers.0.mlp.shared_expert.up_proj.weight": "shared_up",
    "layers.0.mlp.shared_expert.down_proj.weight": "shared_down", "layers.0.mlp.shared_expert_gate.weight": "shared_expert_gate",
}
