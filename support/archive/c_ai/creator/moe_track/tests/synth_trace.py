#!/usr/bin/env python3
# File: <project>\moe_track\tests\synth_trace.py
"""Synthetic router traces with known structure, for testing the simulator and predictor without a model.

Each token has a hidden topic z. Each layer prefers a different, skewed (Zipf) set of experts for each topic, so
  * expert use is skewed (Gini > 0),
  * the experts picked in layer l tell you about layer l+1 (they share z),
  * consecutive tokens share z with probability `stay` (temporal locality),
  * domains favour different topics (a profile from one domain only partly transfers).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
import pltrace  # noqa: E402


def make_trace(path: Path, n_layers=6, n_expert=32, k=4, n_topics=6, domains=("code", "persona"), seqs_per_domain=12, tokens_per_seq=120,
               zipf=1.2, stay=0.9, topic_noise=0.0, seed=3, structure=True) -> pltrace.Trace:
    rng = np.random.default_rng(seed)
    ranks = 1.0 / np.arange(1, n_expert + 1) ** zipf
    ranks /= ranks.sum()
    # pref[l, z] = probability vector over experts
    pref = np.zeros((n_layers, n_topics, n_expert))
    for l in range(n_layers):
        for z in range(n_topics):
            pref[l, z, rng.permutation(n_expert)] = ranks
    if not structure:
        pref[:] = ranks[None, None, :]          # no topic structure: same skew everywhere
    dom_topics = {d: rng.choice(n_topics, size=max(2, n_topics // 2), replace=False) for d in domains}

    blocks, meta = [], []
    seq_id = 0
    for d in domains:
        for s in range(seqs_per_domain):
            z = int(rng.choice(dom_topics[d]))
            rows = []
            for t in range(tokens_per_seq):
                if rng.random() > stay:
                    z = int(rng.choice(dom_topics[d]))
                zz = z if rng.random() > topic_noise else int(rng.integers(n_topics))
                row = np.zeros((n_layers, k), np.uint16)
                for l in range(n_layers):
                    row[l] = rng.choice(n_expert, size=k, replace=False, p=pref[l, zz])
                rows.append(row)
            ex = np.stack(rows)
            for t in range(tokens_per_seq):
                blocks.append({"kind": 1, "seq": seq_id, "pos0": t, "tokens": [0], "experts": ex[t:t + 1]})
            meta.append({"seq": seq_id, "id": f"{d}-{s:04d}", "domain": d, "prompt_tokens": 10, "generated_tokens": tokens_per_seq, "finish": "length"})
            seq_id += 1
    hdr = {"arch": "synthetic", "n_layers": n_layers, "moe_layers": list(range(n_layers)), "n_expert": n_expert, "n_expert_used": k, "weights": False}
    p = pltrace.write_trace(path, hdr, blocks, meta)
    p.with_suffix(".done").write_text("synthetic\n")
    return pltrace.read_trace(p)
