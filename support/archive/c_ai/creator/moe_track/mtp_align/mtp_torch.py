#!/usr/bin/env python3
# File: <project>\moe_track\mtp_align\mtp_torch.py
"""PyTorch version of the Qwen3.6-35B-A3B MTP (draft) head, with LoRA on the eight modules that matter, plus the distillation loss used to train it.

Imported by 10_train_blk40_mtp_lora.py (and by its --selftest). It is the same computation as mtp_block_numpy.py, line for line; the self-test in
10_train_blk40_mtp_lora.py runs both on random weights and demands that they agree.

Only a small, plain set of torch functions is used on purpose (so tests/fake_torch.py can run the forward math without torch installed).

Training view of the head ("teacher forcing"):
  entry t (t = 0 .. T-3):  inputs  = main-model hidden state H[t]  and  the embedding of token ids[t+1]
                           target  = what the main model itself predicts after ids[t+1]  ==  lm_head(H[t+1])      (soft target; KL)
                                     and the real token ids[t+2]                                                  (hard target; small CE weight)
  step 2 (only used when the server drafts 2 tokens): entry t gets the head's OWN output O1[t] instead of H, the token ids[t+2], position +1, and sees the
  step-1 keys/values of entries 0..t; target = lm_head(H[t+2]) / ids[t+3].
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (run_chunks uses torch.utils.checkpoint.checkpoint)

from mtp_block_numpy import Config, HF_NAME_MAP

LORA_TARGETS = ("fc", "q_proj", "k_proj", "v_proj", "o_proj", "shared_gate", "shared_up", "shared_down")


def rms(x, w, eps: float):
    """Zero-centred RMSNorm over the last axis in float32; result in x's dtype. Effective scale = 1 + w."""
    xf = x.float()
    y = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    return y.to(x.dtype)


def rope(x, pos, rot_dim: int, base: float):
    """x [T, H, D]; rotates the first rot_dim dims with pairs (i, i + rot_dim/2), angle = pos * base^(-2i/rot_dim). pos [T] (integer tensor)."""
    half = rot_dim // 2
    inv = base ** (-(torch.arange(half, device=x.device, dtype=torch.float64) * 2.0 / rot_dim))
    ang = pos.to(torch.float64).unsqueeze(1) * inv.unsqueeze(0)               # [T, half]
    cos = ang.cos().float().unsqueeze(1)                                       # [T, 1, half]
    sin = ang.sin().float().unsqueeze(1)
    xf = x.float()
    a = xf[:, :, :half]
    b = xf[:, :, half:rot_dim]
    rest = xf[:, :, rot_dim:]
    out = torch.cat([a * cos - b * sin, b * cos + a * sin, rest], dim=-1)
    return out.to(x.dtype)


class LoRALinear(nn.Module):
    """y = x W^T + scale * (x A^T) B^T, W frozen, A [r, in] and B [out, r] trained in float32. r = 0 gives a plain frozen linear."""

    def __init__(self, weight, r: int, alpha: float, seed: int = 0):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = (self.alpha / self.r) if self.r > 0 else 0.0
        if self.r > 0:
            out_f, in_f = weight.shape
            g = torch.Generator()
            g.manual_seed(int(seed))
            bound = 1.0 / math.sqrt(in_f)                                       # = kaiming_uniform(a=sqrt(5)), the usual LoRA initialisation
            a = (torch.rand(self.r, in_f, generator=g) * 2.0 - 1.0) * bound
            self.lora_A = nn.Parameter(a.float())
            self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, dtype=torch.float32))

    def forward(self, x):
        y = F.linear(x.to(self.weight.dtype), self.weight)
        if self.r > 0:
            d = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
            y = (y.float() + d).to(y.dtype)
        return y

    def merged_weight(self):
        """W + scale * B @ A in float32 (what 11_merge_blk40_and_export.py writes)."""
        w = self.weight.float()
        if self.r > 0:
            w = w + self.scale * torch.matmul(self.lora_B, self.lora_A)
        return w


class MTPHead(nn.Module):
    """params: dict of tensors keyed like mtp_block_numpy.random_params (fc, q_proj, ..., gate_up, down, norm). Everything but the LoRA matrices is frozen."""

    def __init__(self, params: Dict, cfg: Config, r: int = 16, alpha: float = 32.0, targets: Sequence[str] = LORA_TARGETS, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.dtype = params["fc"].dtype
        self.lin = nn.ModuleDict()
        for i, name in enumerate(LORA_TARGETS):
            self.lin[name] = LoRALinear(params[name], r if name in targets else 0, alpha, seed=seed + 101 * (i + 1))
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "input_layernorm", "q_norm", "k_norm", "post_attention_layernorm", "norm",
                     "router", "gate_up", "down", "shared_expert_gate"):
            setattr(self, name, nn.Parameter(params[name], requires_grad=False))

    # ---- pieces --------------------------------------------------------------------------------------------------------------------------------------------
    def _front(self, h_prev, emb_next):
        e = rms(emb_next, self.pre_fc_norm_embedding, self.cfg.eps)
        h = rms(h_prev, self.pre_fc_norm_hidden, self.cfg.eps)
        return self.lin["fc"](torch.cat([e, h], dim=-1))

    def _qkv(self, xn, pos):
        c = self.cfg
        T = xn.shape[0]
        qf = self.lin["q_proj"](xn).reshape(T, c.n_heads, 2 * c.head_dim)
        q = qf[:, :, : c.head_dim]
        gate = qf[:, :, c.head_dim:].reshape(T, c.n_heads * c.head_dim)
        k = self.lin["k_proj"](xn).reshape(T, c.n_kv_heads, c.head_dim)
        v = self.lin["v_proj"](xn).reshape(T, c.n_kv_heads, c.head_dim)
        q = rope(rms(q, self.q_norm, c.eps), pos, c.rot_dim, c.rope_base)
        k = rope(rms(k, self.k_norm, c.eps), pos, c.rot_dim, c.rope_base)
        return q, k, v, gate

    def _attn_full(self, q, k, v):
        """Causal attention over one pass. q [T,H,D], k/v [T,KV,D] -> [T, H*D]."""
        c = self.cfg
        T = q.shape[0]
        g = c.n_heads // c.n_kv_heads
        qh = q.permute(1, 0, 2).unsqueeze(0)                                       # [1,H,T,D]
        kh = k.repeat_interleave(g, dim=1).permute(1, 0, 2).unsqueeze(0)
        vh = v.repeat_interleave(g, dim=1).permute(1, 0, 2).unsqueeze(0)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)             # default scale 1/sqrt(D)
        return o[0].permute(1, 0, 2).reshape(T, c.n_heads * c.head_dim)

    def _attn_step2(self, q2, k2, v2, K1, V1):
        """Entry t: one new key/value (k2[t], v2[t]) plus the step-1 keys/values of entries 0..t. Float32 scores; [T,H,D] in, [T, H*D] out."""
        c = self.cfg
        T = q2.shape[0]
        g = c.n_heads // c.n_kv_heads
        root = math.sqrt(c.head_dim)
        K1e = K1.repeat_interleave(g, dim=1).float()                               # [T,H,D]
        V1e = V1.repeat_interleave(g, dim=1).float()
        k2e = k2.repeat_interleave(g, dim=1).float()
        v2e = v2.repeat_interleave(g, dim=1).float()
        q2f = q2.float()
        s1 = torch.einsum("thd,shd->hts", q2f, K1e) / root                         # [H,T,T]
        causal = torch.triu(torch.ones(T, T, device=q2.device), diagonal=1) > 0.5   # True where s > t
        s1 = s1.masked_fill(causal.unsqueeze(0), float("-inf"))
        s_self = ((q2f * k2e).sum(-1) / root).transpose(0, 1).unsqueeze(-1)         # [H,T,1]
        w = torch.softmax(torch.cat([s1, s_self], dim=-1), dim=-1)                  # [H,T,T+1]
        out1 = torch.einsum("hts,shd->thd", w[:, :, :T], V1e)                       # [T,H,D]
        out2 = w[:, :, T:].permute(1, 0, 2) * v2e                                   # [T,H,1] * [T,H,D]
        return (out1 + out2).reshape(T, c.n_heads * c.head_dim)

    def _attn_proj(self, a, gate):
        """Output gate (sigmoid of the second half of q_proj) then o_proj."""
        a = a.float() * torch.sigmoid(gate.float())
        return self.lin["o_proj"](a.to(self.dtype))

    def _moe(self, xn):
        c = self.cfg
        T, D = xn.shape
        K, Fe = c.top_k, c.ff_exp
        probs = torch.softmax(F.linear(xn.float(), self.router.float()), dim=-1)    # [T, E]
        topw, topi = probs.topk(K, dim=-1)
        if c.norm_topk:
            topw = topw / topw.sum(-1, keepdim=True)
        flat_e = topi.reshape(-1)
        order = torch.argsort(flat_e, stable=True)                                  # token-expert pairs grouped by expert
        tok = order // K                                                            # which token each sorted pair belongs to
        w_sorted = topw.reshape(-1)[order]
        counts = torch.bincount(flat_e, minlength=c.n_experts).tolist()             # one small host sync per forward
        xs = xn[tok]                                                                # [T*K, D]
        out = torch.zeros(T, D, dtype=torch.float32, device=xn.device)
        start = 0
        for e in range(c.n_experts):
            n = counts[e]
            if n == 0:
                continue
            sl = slice(start, start + n)
            start += n
            gu = F.linear(xs[sl], self.gate_up[e])                                  # [n, 2*Fe]  (gate rows first, then up rows)
            h = F.silu(gu[:, :Fe]) * gu[:, Fe:]
            y = F.linear(h, self.down[e])                                           # [n, D]
            out.index_add_(0, tok[sl], y.float() * w_sorted[sl].unsqueeze(1))
        sh = self.lin["shared_down"](F.silu(self.lin["shared_gate"](xn)) * self.lin["shared_up"](xn))
        gate = torch.sigmoid(F.linear(xn.float(), self.shared_expert_gate.float()))  # [T, 1]
        out = out + sh.float() * gate
        return out.to(xn.dtype)

    def _back(self, x, attn_out):
        x = x + attn_out
        xn2 = rms(x, self.post_attention_layernorm, self.cfg.eps)
        x = x + self._moe(xn2)
        return rms(x, self.norm, self.cfg.eps)

    # ---- the two draft steps -------------------------------------------------------------------------------------------------------------------------------
    def step1(self, h_prev, emb_next, pos):
        """h_prev, emb_next: [T, D] (compute dtype); pos: [T] integer tensor. Returns (O1 [T,D], K1 [T,KV,D], V1 [T,KV,D])."""
        x = self._front(h_prev, emb_next)
        xn = rms(x, self.input_layernorm, self.cfg.eps)
        q, k, v, gate = self._qkv(xn, pos)
        a = self._attn_full(q, k, v)
        o = self._back(x, self._attn_proj(a, gate))
        return o, k, v

    def step2(self, o1, emb_next2, pos, K1, V1):
        """o1: step-1 outputs [T,D]; emb_next2: embedding of token t+2 [T,D]; pos as in step 1 (the +1 is added here); K1, V1 from step1 (same T)."""
        x = self._front(o1, emb_next2)
        xn = rms(x, self.input_layernorm, self.cfg.eps)
        q, k, v, gate = self._qkv(xn, pos + 1)
        a = self._attn_step2(q, k, v, K1, V1)
        return self._back(x, self._attn_proj(a, gate))

    # ---- adapter helpers -----------------------------------------------------------------------------------------------------------------------------------
    def lora_state(self) -> Dict[str, Tuple]:
        """{target: (A [r,in] float32 cpu, B [out,r] float32 cpu)} for every target with r > 0."""
        out = {}
        for name, m in self.lin.items():
            if m.r > 0:
                out[name] = (m.lora_A.detach().float().cpu(), m.lora_B.detach().float().cpu())
        return out

    def load_lora_state(self, state: Dict[str, Tuple]) -> None:
        for name, (a, b) in state.items():
            m = self.lin[name]
            if tuple(m.lora_A.shape) != tuple(a.shape) or tuple(m.lora_B.shape) != tuple(b.shape):
                raise ValueError(f"adapter shape mismatch for {name}: have A{tuple(m.lora_A.shape)} B{tuple(m.lora_B.shape)}, file has A{tuple(a.shape)} B{tuple(b.shape)}")
            with torch.no_grad():
                m.lora_A.copy_(a.to(m.lora_A.device))
                m.lora_B.copy_(b.to(m.lora_B.device))

    def trainable_parameters(self) -> List:
        return [p for p in self.parameters() if p.requires_grad]


# ---- loss ---------------------------------------------------------------------------------------------------------------------------------------------------
def token_stats(h_student, h_teacher, y, w_head):
    """Per-token numbers for a chunk of rows. h_student/h_teacher [c, D] (post-norm hidden states), y [c] true next tokens, w_head [V, D] (shared lm_head).
    Returns [c, 4]: KL(teacher || student), cross-entropy of the true token under the student, 1 if both pick the same top token, and the speculative-sampling
    acceptance probability sum_v min(p_teacher, p_student)."""
    zs = F.linear(h_student.to(w_head.dtype), w_head).float()
    with torch.no_grad():
        zt = F.linear(h_teacher.to(w_head.dtype), w_head).float()
    lps = F.log_softmax(zs, dim=-1)
    lpt = F.log_softmax(zt, dim=-1)
    pt = lpt.exp()
    kl = (pt * (lpt - lps)).sum(-1)
    ce = -lps.gather(1, y.unsqueeze(1)).squeeze(1)
    agree = (zs.argmax(-1) == zt.argmax(-1)).float()
    overlap = torch.minimum(pt, lps.exp()).sum(-1)
    return torch.stack([kl, ce, agree, overlap], dim=-1)


def run_chunks(h_student, h_teacher, y, w_head, chunk: int, use_checkpoint: bool):
    """Sum of token_stats over all rows, in chunks (the logits of a chunk are recomputed in the backward pass when use_checkpoint, so memory stays flat).
    Returns a [4] tensor: sums of (KL, CE, agree, overlap)."""
    n = h_student.shape[0]
    total = None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        if use_checkpoint:
            part = torch.utils.checkpoint.checkpoint(token_stats, h_student[s:e], h_teacher[s:e], y[s:e], w_head, use_reentrant=False)
        else:
            part = token_stats(h_student[s:e], h_teacher[s:e], y[s:e], w_head)
        part = part.sum(0)
        total = part if total is None else total + part
    return total


def hf_cfg_to_config(hf: dict, shapes: Dict[str, Tuple[int, ...]]) -> Tuple[Config, Dict[str, str]]:
    """Config from the checkpoint's config.json (nested search) with the sizes taken from the real tensor shapes. Returns (Config, {field: 'config'|'default'})."""
    def find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None
    src: Dict[str, str] = {}

    def get(key, default):
        v = find(hf, key)
        src[key] = "config" if v is not None else "default"
        return v if v is not None else default

    D = int(shapes["q_norm"][0])
    H = int(shapes["q_proj"][0]) // (2 * D)
    KV = int(shapes["k_proj"][0]) // D
    partial = float(get("partial_rotary_factor", 0.25))
    cfg = Config(hidden=int(shapes["fc"][0]), n_heads=H, n_kv_heads=KV, head_dim=D, rot_dim=int(round(D * partial)),
                 rope_base=float(get("rope_theta", 1.0e7)), n_experts=int(shapes["router"][0]), top_k=int(get("num_experts_per_tok", 8)),
                 ff_exp=int(shapes["down"][2]), ff_shared=int(shapes["shared_gate"][0]), eps=float(get("rms_norm_eps", 1e-6)),
                 norm_topk=bool(get("norm_topk_prob", True)), zero_centred_norm=True)
    return cfg, src
