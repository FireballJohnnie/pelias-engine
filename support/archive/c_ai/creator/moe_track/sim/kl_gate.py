#!/usr/bin/env python3
# File: <project>\moe_track\sim\kl_gate.py
"""Gate H6: does a change that should not alter the maths (cache, placement, fork) leave the model's output distribution unchanged?

Uses llama.cpp's own llama-perplexity: it stores the reference model's full log-probabilities over a fixed text once, then scores any candidate
run against them (KL divergence, top-1 agreement). Two steps, each with the server STOPPED:

    python sim/kl_gate.py base    --label ref                         # production build + production placement writes the reference
    python sim/kl_gate.py compare --label fork-cache --base ref --bin /path/to/fork/build/bin --extra "--moe-cache-mib 4096"
    python lib/gates.py h6 --kl logs/moe_trace/kl/fork-cache_vs_ref.json

    python sim/kl_gate.py base --label ref2 ; python sim/kl_gate.py compare --label ref2-again --base ref2     # noise floor of the method itself

Options (both steps)   --model GGUF   --bin DIR (llama-perplexity lives here)   --text FILE   --ctx 1024   --chunks 16   --extra "flags"   --force
  default placement/flags: -ngl 99 --n-cpu-moe 28 -fa on --cache-type-k q8_0 --cache-type-v q8_0 (production); --extra is appended, so a later flag wins.
  default text: dataset/calibration-text.txt (ChatML conversations: persona, code, tool use).
Disk: the reference stores one 16-bit value per vocabulary entry per scored token: about (ctx/2) * chunks * vocab * 2 bytes
  (ctx 1024, 16 chunks, 248k vocabulary: about 4 GB). The script prints the size and checks free space first.
Memory: loads the model like the server does; the server must not be running (--force overrides).
Result JSON (compare): mean_kl, top1_agreement (0..1), n_tokens, reference, candidate, plus perplexity and percentile details.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402

KL_DIR = paths.MOE_TRACE_DIR / "kl"
DEFAULT_FLAGS = ["-ngl", "99", "--n-cpu-moe", str(paths.PROD_EXPERT_FLAGS[1]), "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def server_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def parse_log(text: str, n_ctx: Optional[int] = None) -> dict:
    """Pulls the numbers out of llama-perplexity's KL report. Raises ValueError if the report is not there."""
    def grab(pattern: str, group: int = 1, flags: int = 0) -> Optional[float]:
        m = re.search(pattern, text, flags)
        return float(m.group(group)) if m else None

    mean_kld = grab(r"Mean\s+KLD:\s*" + NUM)
    same_top = grab(r"Same top p:\s*" + NUM)
    if mean_kld is None or same_top is None:
        raise ValueError("no 'KL divergence statistics' in the log (did llama-perplexity fail? read the end of the log)")
    out = {
        "mean_kl": mean_kld,
        "mean_kl_stderr": grab(r"Mean\s+KLD:\s*" + NUM + r"\s*±\s*" + NUM, 2),
        "top1_agreement": same_top / 100.0,
        "ppl_candidate": grab(r"Mean PPL\(Q\)\s*:\s*" + NUM),
        "ppl_reference": grab(r"Mean PPL\(base\)\s*:\s*" + NUM),
        "max_kld": grab(r"Maximum KLD:\s*" + NUM),
        "kld_p99": grab(r"99\.0%\s+KLD:\s*" + NUM),
        "kld_p999": grab(r"99\.9%\s+KLD:\s*" + NUM),
        "kld_median": grab(r"Median\s+KLD:\s*" + NUM),
        "mean_dp_percent": grab(r"Mean\s+Δp:\s*" + NUM),
        "rms_dp_percent": grab(r"RMS Δp\s*:\s*" + NUM),
    }
    m = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", text)
    if m:
        chunks, ctx = int(m.group(1)), int(m.group(2))
        out["n_chunks"], out["n_ctx"] = chunks, ctx
        out["n_tokens"] = chunks * (ctx - 1 - ctx // 2)       # the scorer evaluates the second half of every chunk
    elif n_ctx:
        out["n_ctx"] = n_ctx
    return out


def logits_header(path: Path) -> Optional[dict]:
    """n_ctx, n_vocab, n_chunk of a saved reference file (None if it does not look like one)."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if head != b"_logits_":
                return None
            n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", f.read(12))
        return {"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk}
    except OSError:
        return None


def build_cmd(a, extra_mode: List[str]) -> List[str]:
    exe = Path(a.bin) / "llama-perplexity"
    cmd = [str(exe), "-m", a.model, "-f", a.text, "-c", str(a.ctx), "-b", str(a.ctx), "-ub", str(a.ctx), "--chunks", str(a.chunks)] + DEFAULT_FLAGS
    if a.extra:
        cmd += a.extra.split()
    return cmd + extra_mode


def preflight(a) -> Optional[str]:
    if not (Path(a.bin) / "llama-perplexity").exists():
        return f"llama-perplexity not found in {a.bin}. It is part of the llama.cpp build (target llama-perplexity); for a fork pass --bin <fork build bin>."
    if not Path(a.model).exists():
        return f"model not found: {a.model}"
    if not Path(a.text).exists():
        return f"text file not found: {a.text}"
    if server_running() and not a.force:
        return "llama-server is running; stop it first (llama --stop) or pass --force"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["base", "compare", "parse"])
    ap.add_argument("logfile", nargs="?", help="for 'parse': a saved llama-perplexity log")
    ap.add_argument("--label", default="run")
    ap.add_argument("--base", default="", help="label of the reference written by 'base' (for compare)")
    ap.add_argument("--model", default=str(paths.GGUF))
    ap.add_argument("--bin", default=str(paths.LLAMA_PROD_BIN))
    ap.add_argument("--text", default=str(paths.CREATOR / "dataset" / "calibration-text.txt"))
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--extra", default="")
    ap.add_argument("--kl-dir", default=str(KL_DIR))
    ap.add_argument("--vocab", type=int, default=248320, help="only used for the disk-size estimate before the first run")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    kl_dir = Path(a.kl_dir)

    if a.action == "parse":
        if not a.logfile:
            print("parse needs a log file", file=sys.stderr)
            return 2
        print(json.dumps(parse_log(Path(a.logfile).read_text(encoding="utf-8", errors="replace")), indent=2))
        return 0

    err = preflight(a)
    if err:
        print(err, file=sys.stderr)
        return 3
    kl_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if a.action == "base":
        base_file = kl_dir / f"{a.label}.logits.bin"
        need = (a.ctx - 1 - a.ctx // 2) * a.chunks * a.vocab * 2
        free = shutil.disk_usage(kl_dir).free
        print(f"reference file will be about {need/2**30:.1f} GiB (free on that disk: {free/2**30:.1f} GiB)")
        if need > free * 0.9:
            print("not enough free disk space; lower --chunks or --ctx", file=sys.stderr)
            return 3
        if base_file.exists():
            print(f"{base_file} already exists; choose another --label (nothing is overwritten)", file=sys.stderr)
            return 3
        cmd = build_cmd(a, ["--kl-divergence-base", str(base_file)])
        log = kl_dir / f"{a.label}.base.{stamp}.log"
        print("running:", " ".join(cmd))
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not base_file.exists():
            print(f"llama-perplexity failed (exit {rc}); see {log}", file=sys.stderr)
            return 4
        hdr = logits_header(base_file)
        meta = {"label": a.label, "model": a.model, "bin": a.bin, "cmd": cmd, "text": a.text, "ctx": a.ctx, "chunks": a.chunks, "logits_header": hdr, "stamp": stamp}
        (kl_dir / f"{a.label}.base.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"reference saved: {base_file} ({base_file.stat().st_size/2**30:.2f} GiB)\nnext: python sim/kl_gate.py compare --label <name> --base {a.label} [--bin ... --extra ...]")
        return 0

    # compare
    if not a.base:
        print("compare needs --base <label of the reference>", file=sys.stderr)
        return 2
    base_file = kl_dir / f"{a.base}.logits.bin"
    base_meta_p = kl_dir / f"{a.base}.base.json"
    if not base_file.exists():
        print(f"reference {base_file} not found; run 'base --label {a.base}' first", file=sys.stderr)
        return 3
    if base_meta_p.exists():
        bm = json.loads(base_meta_p.read_text())
        if bm.get("ctx") != a.ctx or bm.get("text") != a.text:
            print(f"NOTE: the reference was made with ctx {bm.get('ctx')} and text {bm.get('text')}; using those", file=sys.stderr)
            a.ctx, a.text = int(bm["ctx"]), bm["text"]
        a.chunks = int(bm.get("chunks", a.chunks))
    cmd = build_cmd(a, ["--kl-divergence", "--kl-divergence-base", str(base_file)])
    log = kl_dir / f"{a.label}_vs_{a.base}.{stamp}.log"
    print("running:", " ".join(cmd))
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text(encoding="utf-8", errors="replace")
    try:
        res = parse_log(text)
    except ValueError as e:
        print(f"{e}; see {log}", file=sys.stderr)
        return 4
    hdr = logits_header(base_file)
    if hdr:   # authoritative: what the reference file holds is what was scored
        res["n_chunks"], res["n_ctx"] = hdr["n_chunk"], hdr["n_ctx"]
        res["n_tokens"] = hdr["n_chunk"] * (hdr["n_ctx"] - 1 - hdr["n_ctx"] // 2)
    res.update({"reference": a.base, "candidate": a.label, "candidate_cmd": cmd, "candidate_model": a.model, "log": str(log), "exit_code": rc})
    out = kl_dir / f"{a.label}_vs_{a.base}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"mean KL {res['mean_kl']:.6f} nats, top-1 agreement {res['top1_agreement']*100:.2f}%, {res.get('n_tokens', '?')} tokens scored; wrote {out}")
    print(f"next: python lib/gates.py h6 --kl {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
