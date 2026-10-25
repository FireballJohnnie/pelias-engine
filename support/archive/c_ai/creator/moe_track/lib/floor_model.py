#!/usr/bin/env python3
# File: <project>\moe_track\lib\floor_model.py
"""Token-time model for expert placement, fitted from measurements instead of assumed.

    T(N)  =  F  +  sum_{l < N} bytes_l / B_cpu  +  sum_{l >= N} bytes_l / B_gpu          (--n-cpu-moe N)
    T(hit) = F  +  sum_l bytes_l * ( hit_l / B_gpu + (1 - hit_l) / B_miss )               (cache with hit rate)

    bytes_l  = top-k * expert_bytes[l]       bytes of routed-expert weights read for one token in layer l
    F        = everything that does not scale with expert placement (attention/DeltaNet, shared experts,
               router, norms, sampling, launch and sync overhead)
    B_cpu    = effective expert-read bandwidth of the CPU path inside llama.cpp (fitted)
    B_gpu    = GPU memory bandwidth for resident experts (spec value, default 504 GB/s for the RTX 4070)
    B_miss   = bandwidth of the miss path: B_cpu when misses are computed on the CPU,
               PCIe bandwidth when misses are copied to the GPU first

Two measured points (all experts on CPU, N=28) fix F and B_cpu. A third point (N=34) is then PREDICTED and compared
with its measurement: that is Gate G0. Fitting and testing on the same two points would always pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class FloorModel:
    fixed_ms: float
    b_cpu_gbs: float
    b_gpu_gbs: float
    layer_bytes: List[float]            # bytes of routed-expert weights read per token in each layer (top-k experts)

    def t_ms_n_cpu_moe(self, n_cpu: int) -> float:
        n_cpu = max(0, min(n_cpu, len(self.layer_bytes)))
        cpu = sum(self.layer_bytes[:n_cpu]) / (self.b_cpu_gbs * 1e9)
        gpu = sum(self.layer_bytes[n_cpu:]) / (self.b_gpu_gbs * 1e9)
        return self.fixed_ms + (cpu + gpu) * 1e3

    def tps_n_cpu_moe(self, n_cpu: int) -> float:
        return 1000.0 / self.t_ms_n_cpu_moe(n_cpu)

    def t_ms_hit_rate(self, hit: float | Sequence[float], miss_gbs: Optional[float] = None) -> float:
        miss = miss_gbs if miss_gbs else self.b_cpu_gbs
        hits = [hit] * len(self.layer_bytes) if isinstance(hit, (int, float)) else list(hit)
        if len(hits) != len(self.layer_bytes):
            raise ValueError("hit-rate list must have one entry per layer")
        t = 0.0
        for b, h in zip(self.layer_bytes, hits):
            t += b * (h / (self.b_gpu_gbs * 1e9) + (1.0 - h) / (miss * 1e9))
        return self.fixed_ms + t * 1e3

    def tps_hit_rate(self, hit: float | Sequence[float], miss_gbs: Optional[float] = None) -> float:
        return 1000.0 / self.t_ms_hit_rate(hit, miss_gbs)


def layer_bytes_from_sizes(sizes: Optional[dict], topk: int = 8, n_layers: int = 40, default_expert_bytes: float = 1.9e6) -> List[float]:
    """Per-layer bytes read per token, from logs/floors/expert_layer_sizes.json (spine layers only)."""
    if sizes and sizes.get("layers"):
        spine = [l for l in sizes["layers"] if l.get("role", "spine") == "spine"]
        k = int(sizes.get("n_experts_used_per_token") or topk)
        spine.sort(key=lambda l: l["layer"])
        return [float(l["expert_bytes"]) * k for l in spine]
    return [default_expert_bytes * topk] * n_layers


def fit_two_point(layer_bytes: Sequence[float], tps_all_cpu: float, tps_at_n: float, n: int, b_gpu_gbs: float = 504.0) -> FloorModel:
    """Fit F and B_cpu from (all experts on CPU) and (first `n` layers on CPU, the rest on GPU)."""
    n_layers = len(layer_bytes)
    if not 0 < n < n_layers:
        raise ValueError(f"n must be between 1 and {n_layers - 1}")
    t40 = 1000.0 / tps_all_cpu
    tn = 1000.0 / tps_at_n
    moved = sum(layer_bytes[n:])                       # bytes that left the CPU path
    moved_gpu_ms = moved / (b_gpu_gbs * 1e9) * 1e3     # what they cost on the GPU
    saved_ms = t40 - tn + moved_gpu_ms                 # = moved / B_cpu in ms
    if saved_ms <= 0:
        raise ValueError("moving layers to the GPU did not reduce token time; the two points cannot define a CPU bandwidth")
    b_cpu = moved / (saved_ms * 1e-3) / 1e9
    fixed = t40 - sum(layer_bytes) / (b_cpu * 1e9) * 1e3
    return FloorModel(fixed_ms=fixed, b_cpu_gbs=b_cpu, b_gpu_gbs=b_gpu_gbs, layer_bytes=list(layer_bytes))


def describe(model: FloorModel) -> Dict[str, float]:
    total = sum(model.layer_bytes)
    return {
        "fixed_ms": round(model.fixed_ms, 3),
        "b_cpu_gbs": round(model.b_cpu_gbs, 2),
        "b_gpu_gbs": model.b_gpu_gbs,
        "gb_per_token_all_experts": round(total / 1e9, 4),
        "ms_all_cpu_expert_path": round(total / (model.b_cpu_gbs * 1e9) * 1e3, 3),
        "tps_all_cpu": round(model.tps_n_cpu_moe(len(model.layer_bytes)), 2),
        "tps_all_gpu_ceiling": round(model.tps_n_cpu_moe(0), 2),
    }
