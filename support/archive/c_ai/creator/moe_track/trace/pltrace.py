#!/usr/bin/env python3
# File: <project>\moe_track\trace\pltrace.py
"""Reader and writer for the .pltrace files written by trace/moe_trace (format documented in moe_trace.cpp).

    from pltrace import read_trace
    t = read_trace("logs/moe_trace/code_0001")        # reads code_0001.pltrace and code_0001.meta.jsonl
    t.decode.experts      # uint16 [n_decode_tokens, n_moe_layers, K]
    t.decode.seq          # uint32 [n_decode_tokens]   sequence id of each token
    t.decode.domain       # list/array of domain names per token (from the meta file)
    t.prefill.experts     # same for traced prefill tokens (the final layer only has the last token of each prompt)

Command line:  python pltrace.py FILE_OR_PREFIX [...]     prints a short summary of each trace.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

MAGIC = b"PLTRACE1"
NOT_RECORDED = 0xFFFF


@dataclass
class Part:
    """All tokens of one kind (prefill or decode), concatenated in file order."""
    experts: np.ndarray                      # uint16 [n, L, K]
    seq: np.ndarray                          # uint32 [n]
    pos: np.ndarray                          # uint32 [n]
    tokens: np.ndarray                       # int32  [n]
    weights: Optional[np.ndarray] = None     # float16 [n, L, K] or None

    @property
    def n(self) -> int:
        return int(self.experts.shape[0])


@dataclass
class Trace:
    header: dict
    prefill: Part
    decode: Part
    meta: List[dict] = field(default_factory=list)
    source: str = ""

    @property
    def moe_layers(self) -> List[int]:
        return list(self.header["moe_layers"])

    @property
    def n_expert(self) -> int:
        return int(self.header["n_expert"])

    @property
    def k(self) -> int:
        return int(self.header["n_expert_used"])

    def seq_domain(self) -> Dict[int, str]:
        return {int(m["seq"]): m.get("domain", "unspecified") for m in self.meta}

    def decode_domains(self) -> np.ndarray:
        m = self.seq_domain()
        return np.array([m.get(int(s), "unspecified") for s in self.decode.seq], dtype=object)

    def seq_split(self) -> Dict[int, str]:
        return {int(m["seq"]): split_of(str(m.get("id", ""))) for m in self.meta}

    def decode_split_mask(self, split: str) -> np.ndarray:
        """Boolean mask over decode tokens: True where the token's prompt belongs to `split` ('fit' or 'heldout')."""
        m = self.seq_split()
        return np.array([m.get(int(s)) == split for s in self.decode.seq], dtype=bool)

    def decode_by_domain(self) -> Dict[str, np.ndarray]:
        dom = self.decode_domains()
        return {d: np.nonzero(dom == d)[0] for d in sorted(set(dom.tolist()))}


HELDOUT_PERCENT = 30


def split_of(prompt_id: str) -> str:
    """Deterministic fit / heldout assignment by prompt id (30% held out). The same prompt always lands in the same split,
    whatever seed or pass produced the trace, so held-out prompts never leak into the profile that is scored on them."""
    h = int(hashlib.sha1(prompt_id.encode("utf-8")).hexdigest()[:8], 16)
    return "heldout" if h % 100 < HELDOUT_PERCENT else "fit"


def _empty(L: int, K: int, weights: bool) -> Part:
    return Part(np.zeros((0, L, K), np.uint16), np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros(0, np.int32),
                np.zeros((0, L, K), np.float16) if weights else None)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix == ".pltrace":
        return p
    q = Path(str(p) + ".pltrace")
    return q if q.exists() else p


def read_trace(path: str | Path) -> Trace:
    p = resolve(path)
    data = p.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{p}: not a pltrace file (bad magic)")
    (hlen,) = struct.unpack_from("<I", data, 8)
    header = json.loads(data[12:12 + hlen].decode("utf-8"))
    L, K, has_w = len(header["moe_layers"]), int(header["n_expert_used"]), bool(header["weights"])
    off = 12 + hlen
    acc = {0: [], 1: []}
    n = len(data)
    while off < n:
        if off + 16 > n:
            raise ValueError(f"{p}: truncated block header at byte {off}")
        kind = data[off]
        seq, nt, pos0 = struct.unpack_from("<III", data, off + 4)
        off += 16
        need = nt * 4 + nt * L * K * 2 + (nt * L * K * 2 if has_w else 0)
        if off + need > n:
            raise ValueError(f"{p}: truncated block body at byte {off} (a recording was probably cut off)")
        toks = np.frombuffer(data, np.int32, nt, off)
        off += nt * 4
        ex = np.frombuffer(data, np.uint16, nt * L * K, off).reshape(nt, L, K)
        off += nt * L * K * 2
        w = None
        if has_w:
            w = np.frombuffer(data, np.float16, nt * L * K, off).reshape(nt, L, K)
            off += nt * L * K * 2
        acc.setdefault(kind, []).append((seq, pos0, toks, ex, w))

    def build(kind: int) -> Part:
        blocks = acc.get(kind, [])
        if not blocks:
            return _empty(L, K, has_w)
        return Part(
            np.concatenate([b[3] for b in blocks]),
            np.concatenate([np.full(len(b[2]), b[0], np.uint32) for b in blocks]),
            np.concatenate([b[1] + np.arange(len(b[2]), dtype=np.uint32) for b in blocks]),
            np.concatenate([b[2] for b in blocks]),
            np.concatenate([b[4] for b in blocks]) if has_w else None,
        )

    meta: List[dict] = []
    mp = Path(str(p)[:-len(".pltrace")] + ".meta.jsonl")
    if mp.exists():
        for line in mp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                meta.append(json.loads(line))
    return Trace(header, build(0), build(1), meta, str(p))


def write_trace(path: str | Path, header: dict, blocks: Iterable[dict], meta: Optional[Sequence[dict]] = None) -> Path:
    """Writes a trace from Python (used for synthetic data in tests). Each block: kind, seq, pos0, tokens[n], experts[n,L,K], weights[n,L,K]|None."""
    p = Path(path)
    if p.suffix != ".pltrace":
        p = Path(str(p) + ".pltrace")
    p.parent.mkdir(parents=True, exist_ok=True)
    h = dict(header)
    h.setdefault("version", 1)
    hb = json.dumps(h).encode("utf-8")
    has_w = bool(h.get("weights"))
    with open(p, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(hb)) + hb)
        for b in blocks:
            toks = np.asarray(b["tokens"], np.int32)
            ex = np.asarray(b["experts"], np.uint16)
            f.write(bytes([int(b["kind"]), 0, 0, 0]) + struct.pack("<III", int(b["seq"]), len(toks), int(b.get("pos0", 0))))
            f.write(toks.tobytes() + ex.tobytes())
            if has_w:
                f.write(np.asarray(b["weights"], np.float16).tobytes())
    if meta is not None:
        mp = Path(str(p)[:-len(".pltrace")] + ".meta.jsonl")
        mp.write_text("".join(json.dumps(m) + "\n" for m in meta), encoding="utf-8")
    return p


def merge_decode(traces: Sequence[Trace]) -> Trace:
    """Concatenates several traces of the same model (e.g. one per domain) into one; sequence ids are renumbered."""
    if not traces:
        raise ValueError("nothing to merge")
    h0 = traces[0].header
    for t in traces[1:]:
        if t.header["moe_layers"] != h0["moe_layers"] or t.header["n_expert"] != h0["n_expert"] or t.header["n_expert_used"] != h0["n_expert_used"]:
            raise ValueError(f"{t.source}: different model geometry than {traces[0].source}")
    base, metas, dec = 0, [], []
    for t in traces:
        remap = {int(m["seq"]): base + i for i, m in enumerate(t.meta)}
        for m in t.meta:
            metas.append(dict(m, seq=remap[int(m["seq"])], source=t.source))
        seq = np.array([remap.get(int(s), base) for s in t.decode.seq], np.uint32)
        dec.append(Part(t.decode.experts, seq, t.decode.pos, t.decode.tokens, t.decode.weights))
        base += len(t.meta)
    has_w = all(d.weights is not None for d in dec)
    part = Part(np.concatenate([d.experts for d in dec]), np.concatenate([d.seq for d in dec]), np.concatenate([d.pos for d in dec]),
                np.concatenate([d.tokens for d in dec]), np.concatenate([d.weights for d in dec]) if has_w else None)
    return Trace(dict(h0), _empty(len(h0["moe_layers"]), int(h0["n_expert_used"]), has_w), part, metas, "merged")


def summarize(t: Trace) -> str:
    L, K, E = len(t.moe_layers), t.k, t.n_expert
    lines = [f"{t.source}: arch {t.header.get('arch')}, {L} MoE layers, {E} experts, top-{K}; decode tokens {t.decode.n}, traced prefill tokens {t.prefill.n}, sequences {len(t.meta)}"]
    if t.decode.n:
        ex = t.decode.experts
        counts = np.zeros((L, E), np.int64)
        for li in range(L):
            v = ex[:, li, :].ravel()
            v = v[v != NOT_RECORDED]
            counts[li] = np.bincount(v, minlength=E)[:E]
        lines.append(f"decode picks per layer: min {counts.sum(1).min()}, max {counts.sum(1).max()}; busiest expert takes {100*counts.max(1).mean()/max(counts.sum(1).mean(),1):.1f}% of a layer's picks on average")
        for d, idx in t.decode_by_domain().items():
            lines.append(f"  domain {d}: {len(idx)} decode tokens")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for a in sys.argv[1:]:
        print(summarize(read_trace(a)))
