#!/usr/bin/env python3
# File: <project>\moe_track\lib\speedlog.py
"""Reader for the result files written by stuart_server_speed.py (logs/speed_server/*.json).

Real schema (checked against 20261020_032031_q4km-ncmoe28-ctx131k-ub2048.json):
    meta: label, note, n_ctx, build, depths [0, 4000, 12000], repeats, warmup [..], vram {idle_mib, peak_mib, total_mib}
    rows: [{depth, rep, prompt_n, cache_n, prompt_ms, prompt_tps, gen_n, gen_ms, gen_tps, ttft, finish, timed, vram_used_mib}, ...]
Depth 0 is the short prompt (61 tokens); 4000 and 12000 are filler-prefixed prompts.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

MIB_PER_GIB = 1024.0


class SpeedLogError(RuntimeError):
    pass


@dataclass
class DepthStats:
    depth: int
    n: int
    gen_tps: Optional[float]          # median
    gen_tps_min: Optional[float]
    gen_tps_max: Optional[float]
    prompt_tps: Optional[float]       # median
    prompt_tps_min: Optional[float]
    prompt_tps_max: Optional[float]
    ttft_s: Optional[float]           # median
    cache_hits: int                   # rows where the server reused cached prompt tokens (prompt speed not comparable)

    @property
    def gen_spread(self) -> Optional[float]:
        """(max-min)/median, the run's own noise on generation speed."""
        if self.gen_tps and self.gen_tps_min is not None and self.gen_tps_max is not None:
            return (self.gen_tps_max - self.gen_tps_min) / self.gen_tps
        return None

    @property
    def prompt_spread(self) -> Optional[float]:
        if self.prompt_tps and self.prompt_tps_min is not None and self.prompt_tps_max is not None:
            return (self.prompt_tps_max - self.prompt_tps_min) / self.prompt_tps
        return None


@dataclass
class SpeedLog:
    path: str
    label: str
    n_ctx: Optional[int]
    build: str
    vram_idle_mib: Optional[float]
    vram_peak_mib: Optional[float]
    vram_total_mib: Optional[float]
    depths: Dict[int, DepthStats]
    note: str = ""

    @property
    def vram_peak_gib(self) -> Optional[float]:
        return None if self.vram_peak_mib is None else self.vram_peak_mib / MIB_PER_GIB

    def gen(self, depth: int) -> Optional[float]:
        d = self.depths.get(depth)
        return d.gen_tps if d else None

    def prefill(self, depth: int) -> Optional[float]:
        d = self.depths.get(depth)
        return d.prompt_tps if d else None


def _median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load_speedlog(path: str | Path) -> SpeedLog:
    p = Path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SpeedLogError(f"cannot read {p}: {e}")
    if not isinstance(d, dict) or "rows" not in d or "meta" not in d:
        raise SpeedLogError(f"{p} does not look like a stuart_server_speed.py result (needs 'meta' and 'rows')")
    meta = d["meta"]
    vram = meta.get("vram") or {}
    by_depth: Dict[int, List[dict]] = {}
    for r in d["rows"]:
        if r.get("timed") is False:
            continue
        by_depth.setdefault(int(r["depth"]), []).append(r)
    depths: Dict[int, DepthStats] = {}
    for depth, rows in by_depth.items():
        g = [r.get("gen_tps") for r in rows if r.get("gen_tps")]
        pr = [r.get("prompt_tps") for r in rows if r.get("prompt_tps")]
        tt = [r.get("ttft") for r in rows if r.get("ttft") is not None]
        depths[depth] = DepthStats(
            depth=depth, n=len(rows),
            gen_tps=_median(g), gen_tps_min=min(g) if g else None, gen_tps_max=max(g) if g else None,
            prompt_tps=_median(pr), prompt_tps_min=min(pr) if pr else None, prompt_tps_max=max(pr) if pr else None,
            ttft_s=_median(tt), cache_hits=sum(1 for r in rows if (r.get("cache_n") or 0) > 0))
    return SpeedLog(path=str(p), label=str(meta.get("label", p.stem)), n_ctx=meta.get("n_ctx"),
                    build=str(meta.get("build", "")), note=str(meta.get("note", "")),
                    vram_idle_mib=vram.get("idle_mib"), vram_peak_mib=vram.get("peak_mib"),
                    vram_total_mib=vram.get("total_mib"), depths=depths)
