#!/usr/bin/env python3
# File: <project>\moe_track\fork\launch_parity.py
"""Launch parity: an experiment launch must differ from production in exactly the thing under test.

    python fork/launch_parity.py show                      the production command this tool assumes (from lib/paths.py)
    python fork/launch_parity.py live                      the command line of the llama-server that is running right now
    python fork/launch_parity.py diff --candidate "<command or flags>"        what differs from production (live server if there is one)
    python fork/launch_parity.py diff --candidate-file launch.json            same, for a launch record written by gauntlet.py

How the reference is chosen: if a llama-server is running, its real command line (read from /proc) is the reference, so flags the
launcher adds that this repository does not know about (threads, mmap mode, ...) are carried into every experiment. Otherwise the
reference is the flag list in lib/paths.py (PROD_FLAGS + PROD_EXPERT_FLAGS + the model path).

Flags are compared by their long names ('-ngl' and '--n-gpu-layers' are the same flag), and a flag with different values is reported
as 'changed'. Exit code 0 when only the allowed flags differ, 1 otherwise (use --allow FLAG ... to name the flags under test).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402

# short name -> long name (only the ones llama.cpp servers accept and this project uses)
ALIASES = {
    "-m": "--model", "-c": "--ctx-size", "-b": "--batch-size", "-ub": "--ubatch-size", "-n": "--predict", "-np": "--parallel",
    "-ngl": "--n-gpu-layers", "--gpu-layers": "--n-gpu-layers", "-fa": "--flash-attn", "-t": "--threads", "-tb": "--threads-batch",
    "-ctk": "--cache-type-k", "-ctv": "--cache-type-v", "-ncmoe": "--n-cpu-moe", "-cmoe": "--cpu-moe", "-ot": "--override-tensor",
    "-lm": "--load-mode", "-lzm": "--lazy-mode", "-fit": "--fit", "-ctxcp": "--ctx-checkpoints", "--swa-checkpoints": "--ctx-checkpoints",
    "-cram": "--cache-ram", "-sm": "--split-mode", "-ts": "--tensor-split", "-mg": "--main-gpu", "-lv": "--verbosity",
}
# flags that decide where the expert weights live: dropped from the production command before a placement experiment adds its own
PLACEMENT_FLAGS = {"--n-cpu-moe", "--cpu-moe", "--override-tensor", "--moe-cache-mib"}
NUMBER = re.compile(r"^-\d+(\.\d+)?$")


def long_name(flag: str) -> str:
    return ALIASES.get(flag, flag)


def is_flag(tok: str) -> bool:
    return tok.startswith("-") and len(tok) > 1 and not NUMBER.match(tok)


def parse_flags(tokens: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """-> [(long flag name, [values...])] in command order. A token that does not start with '-' (or is a negative number) is a value."""
    out: List[Tuple[str, List[str]]] = []
    i = 0
    toks = list(tokens)
    while i < len(toks):
        t = toks[i]
        if is_flag(t):
            if "=" in t and t.startswith("--"):
                name, _, val = t.partition("=")
                out.append((long_name(name), [val]))
                i += 1
                continue
            vals: List[str] = []
            j = i + 1
            while j < len(toks) and not is_flag(toks[j]):
                vals.append(toks[j])
                j += 1
            out.append((long_name(t), vals))
            i = j
        else:
            i += 1          # the program name or a stray value
    return out


def as_map(tokens: Sequence[str]) -> Dict[str, List[List[str]]]:
    m: Dict[str, List[List[str]]] = {}
    for name, vals in parse_flags(tokens):
        m.setdefault(name, []).append(vals)
    return m


def diff(reference: Sequence[str], candidate: Sequence[str], ignore: Sequence[str] = ()) -> Dict[str, object]:
    ig = {long_name(x) for x in ignore}
    a, b = as_map(reference), as_map(candidate)
    added = sorted(k for k in b if k not in a and k not in ig)
    removed = sorted(k for k in a if k not in b and k not in ig)
    changed = sorted(k for k in a if k in b and k not in ig and a[k] != b[k])
    same = sorted(k for k in a if k in b and a[k] == b[k])
    return {
        "added": {k: b[k] for k in added},
        "removed": {k: a[k] for k in removed},
        "changed": {k: {"production": a[k], "candidate": b[k]} for k in changed},
        "same": same,
    }


def production_tokens() -> List[str]:
    """The reference command from lib/paths.py: model + PROD_FLAGS + placement."""
    return ["llama-server", "-m", str(paths.GGUF)] + list(paths.PROD_FLAGS) + list(paths.PROD_EXPERT_FLAGS)


def live_server_tokens(proc_root: str = "/proc") -> Optional[List[str]]:
    """The argv of a running llama-server (the first one found), or None."""
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return None
    for pid in sorted(pids, key=int):
        try:
            raw = (Path(proc_root) / pid / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [x.decode("utf-8", "replace") for x in raw.split(b"\0") if x]
        if argv and Path(argv[0]).name == "llama-server":
            return argv
    return None


def reference_tokens(proc_root: str = "/proc") -> Tuple[List[str], str]:
    live = live_server_tokens(proc_root)
    if live:
        return live, "the running llama-server"
    return production_tokens(), "lib/paths.py (no llama-server is running)"


def strip_placement(tokens: Sequence[str]) -> List[str]:
    """Remove the expert-placement flags (and their values) from a command, keeping everything else in order."""
    out: List[str] = []
    toks = list(tokens)
    i = 0
    while i < len(toks):
        t = toks[i]
        if is_flag(t) and long_name(t.partition("=")[0] if t.startswith("--") else t) in PLACEMENT_FLAGS:
            i += 1
            if "=" in t and t.startswith("--"):
                continue
            while i < len(toks) and not is_flag(toks[i]):
                i += 1
            continue
        out.append(t)
        i += 1
    return out


def drop_flags(tokens: Sequence[str], names: Sequence[str]) -> List[str]:
    """Remove the named flags (long or short spelling) and their values."""
    drop = {long_name(n) for n in names}
    out: List[str] = []
    toks = list(tokens)
    i = 0
    while i < len(toks):
        t = toks[i]
        if is_flag(t) and long_name(t.partition("=")[0] if t.startswith("--") else t) in drop:
            i += 1
            if "=" in t and t.startswith("--"):
                continue
            while i < len(toks) and not is_flag(toks[i]):
                i += 1
            continue
        out.append(t)
        i += 1
    return out


def translate_load_mode(tokens: Sequence[str], supports_load_mode: bool) -> List[str]:
    """Older trees (the Parmesan fork) have --no-mmap/--mlock instead of --load-mode. 'none' means no mmap, no mlock."""
    if supports_load_mode:
        return list(tokens)
    toks = list(tokens)
    out: List[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if long_name(t) == "--load-mode" and i + 1 < len(toks):
            mode = toks[i + 1]
            out += {"none": ["--no-mmap"], "mmap": [], "mlock": ["--no-mmap", "--mlock"], "mmap+mlock": ["--mlock"], "auto": [], "dio": ["--no-mmap"]}.get(mode, [])
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def build_command(reference: Sequence[str], server_bin: str, placement_and_variant: Sequence[str], model: Optional[str] = None,
                  port: Optional[int] = None, supports_load_mode: bool = True, drop: Sequence[str] = ()) -> List[str]:
    """Reference command with the binary swapped, the placement flags removed and the variant's flags appended."""
    toks = list(reference)
    toks = strip_placement(toks[1:]) if toks and not is_flag(toks[0]) else strip_placement(toks)
    toks = drop_flags(toks, drop)
    toks = translate_load_mode(toks, supports_load_mode)
    cmd = [server_bin] + toks
    if model:
        if "--model" in as_map(cmd):
            cmd = _replace_value(cmd, "--model", model)
        else:
            cmd += ["-m", model]
    if port is not None:
        cmd = _replace_value(cmd, "--port", str(port)) if "--port" in as_map(cmd) else cmd + ["--port", str(port)]
    return cmd + list(placement_and_variant)


def _replace_value(tokens: Sequence[str], name: str, value: str) -> List[str]:
    out: List[str] = []
    toks = list(tokens)
    i = 0
    while i < len(toks):
        t = toks[i]
        if is_flag(t) and long_name(t) == name:
            out += [t, value]
            i += 1
            while i < len(toks) and not is_flag(toks[i]):
                i += 1
            continue
        out.append(t)
        i += 1
    return out


def format_diff(d: Dict[str, object]) -> List[str]:
    def fmt(vals: List[List[str]]) -> str:
        return " | ".join(" ".join(v) if v else "(on)" for v in vals)
    lines: List[str] = []
    for k, v in d["changed"].items():  # type: ignore[union-attr]
        lines.append(f"  changed  {k}: {fmt(v['production'])}  ->  {fmt(v['candidate'])}")
    for k, v in d["added"].items():  # type: ignore[union-attr]
        lines.append(f"  added    {k} {fmt(v)}")
    for k, v in d["removed"].items():  # type: ignore[union-attr]
        lines.append(f"  removed  {k} {fmt(v)}")
    lines.append(f"  unchanged: {len(d['same'])} flags")  # type: ignore[arg-type]
    return lines


def _candidate_tokens(a) -> List[str]:
    if a.candidate_file:
        rec = json.loads(Path(a.candidate_file).read_text(encoding="utf-8"))
        return list(rec["command"])
    if a.candidate:
        return shlex.split(a.candidate)
    raise SystemExit("give --candidate \"...\" or --candidate-file FILE")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["show", "live", "diff"])
    ap.add_argument("--candidate", default="")
    ap.add_argument("--candidate-file", default="")
    ap.add_argument("--allow", nargs="*", default=[], help="flags that are allowed to differ (the ones under test)")
    ap.add_argument("--proc-root", default="/proc", help="tests only")
    a = ap.parse_args(argv)

    if a.action == "show":
        print(shlex.join(production_tokens()))
        return 0
    if a.action == "live":
        live = live_server_tokens(a.proc_root)
        if not live:
            print("no llama-server is running", file=sys.stderr)
            return 2
        print(shlex.join(live))
        return 0
    ref, src = reference_tokens(a.proc_root)
    cand = _candidate_tokens(a)
    d = diff(ref, cand)
    print(f"reference: {src}")
    for ln in format_diff(d):
        print(ln)
    allowed = {long_name(x) for x in a.allow}
    stray = [k for k in list(d["changed"]) + list(d["added"]) + list(d["removed"]) if k not in allowed and k not in ("--model",)]  # type: ignore[arg-type]
    if allowed and stray:
        print("NOT ALLOWED to differ: " + ", ".join(stray))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
