#!/usr/bin/env python3
# File: <project>\moe_track\fork\make_schwerz_corpus.py
"""Turns the trace prompts (trace/build_prompts.py) into the corpus.json that the Schwerz fork's profile collector reads.

    python fork/make_schwerz_corpus.py [--prompts-dir logs/moe_trace/prompts] [--out-dir logs/moe_trace/schwerz] [--per-domain 12] [--max-tokens 256]

Writes <out-dir>/corpus.json (version 1; equal weight for every domain; inline prompts from the FIT split only, so the held-out prompts stay unseen) and
prints the command that collects the profile. The format is the one documented in the fork's wiki page "Generic-Hybrid-MoE-Reproduction":
    {"version": 1, "weights": {"persona": 0.25, ...}, "requests": [{"family": "persona", "prompt": "...", "max_tokens": 256}, ...]}
Limits of the collector: 128 requests, 32 workload labels, 8 MiB file, 4096 tokens per prompt, 2..4096 output tokens. This tool stays inside them.
Reads prompts_*.jsonl; writes only corpus.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402

MAX_REQUESTS, MAX_BYTES, MAX_PROMPT_CHARS = 128, 8 * 1024 * 1024, 14000     # 14000 characters stays well under 4096 tokens for English text and code


def build(prompts_dir: Path, per_domain: int, max_tokens: int) -> dict:
    files = sorted(prompts_dir.glob("prompts_*.jsonl"))
    if not files:
        raise SystemExit(f"no prompts_*.jsonl in {prompts_dir}; run trace/build_prompts.py first")
    by_dom: Dict[str, List[str]] = {}
    for f in files:
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != "fit" or "text" not in r:
                continue
            t = r["text"]
            if 0 < len(t) <= MAX_PROMPT_CHARS:
                by_dom.setdefault(r.get("domain", f.stem), []).append(t)
    if not by_dom:
        raise SystemExit("no usable fit-split prompts found (they must carry 'text' and be at most %d characters)" % MAX_PROMPT_CHARS)
    n_dom = len(by_dom)
    per = max(1, min(per_domain, MAX_REQUESTS // n_dom))
    reqs = []
    for dom in sorted(by_dom):
        step = max(1, len(by_dom[dom]) // per)
        for t in by_dom[dom][::step][:per]:                      # evenly spread through the file, deterministic
            reqs.append({"family": dom, "prompt": t, "max_tokens": int(max_tokens)})
    corpus = {"version": 1, "weights": {d: round(1.0 / n_dom, 6) for d in sorted(by_dom)}, "requests": reqs}
    if len(json.dumps(corpus).encode("utf-8")) > MAX_BYTES:
        raise SystemExit("corpus would exceed 8 MiB; lower --per-domain")
    return corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", default=str(paths.MOE_TRACE_DIR / "prompts"))
    ap.add_argument("--out-dir", default=str(paths.MOE_TRACE_DIR / "schwerz"))
    ap.add_argument("--per-domain", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    corpus = build(Path(a.prompts_dir), a.per_domain, a.max_tokens)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "corpus.json"
    p.write_text(json.dumps(corpus, indent=1), encoding="utf-8")
    print(f"wrote {p}: {len(corpus['requests'])} requests, domains {', '.join(corpus['weights'])}")
    print("collect the profile with:  bash fork/05_phase5_fork_gauntlet.sh profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
