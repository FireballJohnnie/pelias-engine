#!/usr/bin/env python3
# File: <project>\moe_track\trace\05_collect_traces.py
"""Step 3: record the router's expert choices on real traffic, in resumable shards, until enough decode tokens exist (gate G1).

    python trace/05_collect_traces.py plan                 show what would run and a time estimate; runs nothing
    python trace/05_collect_traces.py run                  stop llama-server first (llama --stop); records shard by shard
    python trace/05_collect_traces.py run --target-tokens 120000
    python trace/05_collect_traces.py summary              write logs/moe_trace/trace_summary.json (then: python lib/gates.py g1)

How it works
  * prompts: trace/build_prompts.py (dataset persona + agent rows, built-in code and reasoning requests), rebuilt only if missing
  * a shard = up to --shard-size prompts of one domain, run once with a seed; shards are ordered pass by pass and round-robin across domains,
    so an interrupted run still covers every domain. Each shard writes <domain>_p<pass>_s<NNN>.pltrace + .meta.jsonl, and a .done marker when the
    tracer exits cleanly. A shard that has a .done marker is never repeated: run again to continue.
  * the tracer loads the model itself, so the llama-server must not be running (RAM and VRAM); the script refuses unless --force.
  * the model is loaded with the production placement (--n-cpu-moe 28, KV q8_0, flash attention) so routing is recorded under production math.
Memory/GPU: same as the server. Disk: about 64 MB per 100k decode tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import build_prompts  # noqa: E402
import paths  # noqa: E402

DOMAINS = ("persona", "agent", "code", "reasoning")


def server_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def load_prompts(prompt_dir: Path, args) -> Dict[str, List[dict]]:
    idx = prompt_dir / "prompts_index.json"
    if not idx.exists():
        print(f"building prompts into {prompt_dir}")
        rc = build_prompts.main(["--out-dir", str(prompt_dir), "--dataset", args.dataset, "--quota", args.quota] + (["--fake-tokenize", str(args.fake_tokenize)] if args.fake_tokenize else []))
        if rc != 0:
            raise SystemExit("could not build prompts")
    out: Dict[str, List[dict]] = {}
    for d in DOMAINS:
        p = prompt_dir / f"prompts_{d}.jsonl"
        if p.exists():
            out[d] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not out:
        raise SystemExit(f"no prompt files in {prompt_dir}")
    return out


def lay_out_shards(prompts: Dict[str, List[dict]], shard_size: int, passes: int) -> List[Tuple[str, int, int, List[dict]]]:
    """-> list of (domain, pass, shard_index, prompts) ordered by pass, then shard index, then domain."""
    shards = []
    for p in range(passes):
        per_dom = {d: [recs[i:i + shard_size] for i in range(0, len(recs), shard_size)] for d, recs in prompts.items()}
        longest = max(len(v) for v in per_dom.values())
        for s in range(longest):
            for d in DOMAINS:
                if d in per_dom and s < len(per_dom[d]):
                    shards.append((d, p, s, per_dom[d][s]))
    return shards


def shard_prefix(trace_dir: Path, d: str, p: int, s: int) -> Path:
    return trace_dir / f"{d}_p{p}_s{s:03d}"


def tokens_done(trace_dir: Path) -> int:
    n = 0
    for done in trace_dir.glob("*.done"):
        meta = done.with_suffix(".meta.jsonl")
        if meta.exists():
            for line in meta.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        n += int(json.loads(line).get("generated_tokens", 0))
                    except ValueError:
                        pass
    return n


def tracer_cmd(a, prefix: Path, jsonl: Path, seed: int) -> List[str]:
    cmd = [a.tracer, "-m", a.model, "--out", str(prefix), "--prompts-jsonl", str(jsonl), "--n-predict", str(a.n_predict),
           "--ngl", str(a.ngl), "--ctx", str(a.ctx), "--ubatch", str(a.ubatch), "--kv-type", a.kv_type, "--flash-attn", a.flash_attn,
           "--seed", str(seed), "--quiet"]
    if a.n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(a.n_cpu_moe)]
    if a.threads > 0:
        cmd += ["--threads", str(a.threads)]
    if a.weights:
        cmd += ["--weights"]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "summary"])
    ap.add_argument("--model", default=str(paths.GGUF))
    ap.add_argument("--tracer", default=str(HERE / "bin" / "moe_trace"))
    ap.add_argument("--trace-dir", default=str(paths.MOE_TRACE_DIR))
    ap.add_argument("--dataset", default=str(paths.CREATOR / "dataset" / "<dataset-file>"))
    ap.add_argument("--quota", default="persona=60,agent=40,code=38,reasoning=31")
    ap.add_argument("--fake-tokenize", type=int, default=0, help="tests only")
    ap.add_argument("--target-tokens", type=int, default=120_000, help="stop after this many decode tokens (gate G1 needs 100,000)")
    ap.add_argument("--n-predict", type=int, default=768)
    ap.add_argument("--shard-size", type=int, default=10)
    ap.add_argument("--passes", type=int, default=4, help="how many times to go through the prompt set (a different sampling seed each time)")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-cpu-moe", type=int, default=int(paths.PROD_EXPERT_FLAGS[1]))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--kv-type", default="q8_0")
    ap.add_argument("--flash-attn", default="on")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--weights", action="store_true", help="also record router probabilities (+50% file size)")
    ap.add_argument("--tok-per-sec", type=float, default=20.0, help="assumed speed for the time estimate only")
    ap.add_argument("--max-shards", type=int, default=0, help="run at most this many new shards (0 = until the target)")
    ap.add_argument("--force", action="store_true", help="run even if llama-server is running")
    a = ap.parse_args(argv)

    trace_dir = Path(a.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    if a.action == "summary":
        import summarize_traces
        return summarize_traces.main(["--trace-dir", str(trace_dir)])

    prompts = load_prompts(trace_dir / "prompts", a)
    shards = lay_out_shards(prompts, a.shard_size, a.passes)
    pending = [s for s in shards if not shard_prefix(trace_dir, s[0], s[1], s[2]).with_suffix(".done").exists()]
    have = tokens_done(trace_dir)
    n_prompts = sum(len(v) for v in prompts.values())
    # a finished sequence is on average shorter than n_predict (the model stops on its own); 0.7 is a placeholder until real shards exist
    est_per_shard = a.shard_size * a.n_predict * 0.7
    need = max(0, a.target_tokens - have)
    shards_needed = min(len(pending), math.ceil(need / est_per_shard)) if est_per_shard else len(pending)
    print(f"prompts: {n_prompts} in {len(prompts)} domains ({', '.join(f'{d} {len(v)}' for d, v in prompts.items())}); {len(shards)} shards over {a.passes} pass(es), {len(shards) - len(pending)} already done")
    print(f"decode tokens recorded so far: {have:,}; target {a.target_tokens:,}")
    print(f"about {shards_needed} more shard(s) needed; at {a.tok_per_sec:.0f} tok/s that is roughly {need / max(a.tok_per_sec, 1) / 60:.0f} minutes of decoding plus prompt reading")

    if a.action == "plan":
        for d, p, s, recs in pending[:6]:
            print(f"   next: {d} pass {p} shard {s}: {len(recs)} prompts")
        return 0

    if need == 0:
        print("target already reached; run 'summary'")
        return 0
    if not Path(a.tracer).exists():
        print(f"tracer not built: {a.tracer}\n  build it with:  bash {HERE}/build.sh", file=sys.stderr)
        return 2
    if not Path(a.model).exists():
        print(f"model not found: {a.model}", file=sys.stderr)
        return 2
    if server_running() and not a.force:
        print("llama-server is running and holds the GPU and most of the RAM. Stop it first:  llama --stop   (or pass --force)", file=sys.stderr)
        return 3

    ran = 0
    t0 = time.time()
    for d, p, s, recs in pending:
        if tokens_done(trace_dir) >= a.target_tokens:
            break
        if a.max_shards and ran >= a.max_shards:
            break
        prefix = shard_prefix(trace_dir, d, p, s)
        jsonl = prefix.with_suffix(".prompts.jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        seed = 1000 * p + s + 1
        for stale in (prefix.with_suffix(".pltrace"), prefix.with_suffix(".meta.jsonl")):
            if stale.exists():
                stale.unlink()                      # an interrupted earlier attempt: start the shard over
        cmd = tracer_cmd(a, prefix, jsonl, seed)
        print(f"[shard {d} pass {p} #{s}] {len(recs)} prompts, seed {seed}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"tracer exited with {rc}; shard not marked done. Stopping so the cause can be read above.", file=sys.stderr)
            return rc
        prefix.with_suffix(".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
        ran += 1
        got = tokens_done(trace_dir)
        el = time.time() - t0
        print(f"   total decode tokens {got:,} of {a.target_tokens:,}   elapsed {el/60:.1f} min", flush=True)
    got = tokens_done(trace_dir)
    print(f"done for now: {got:,} decode tokens recorded. " + ("Target reached." if got >= a.target_tokens else "Run again to continue (finished shards are skipped)."))
    import summarize_traces
    summarize_traces.main(["--trace-dir", str(trace_dir)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
