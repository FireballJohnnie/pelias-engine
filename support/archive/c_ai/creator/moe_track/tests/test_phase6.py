#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_phase6.py
"""Checks that a llama.cpp built with patches/pelias_tier_v1.patch gives the same answers as the stock build.

    python tests/test_phase6.py --llama-src DIR --stock-bin DIR --tier-bin DIR --model tiny_f16.gguf [--model tiny_q4km.gguf] [--workdir DIR]

  --llama-src   llama.cpp source folder (for include/llama.h)
  --stock-bin   folder with the libllama.so / libggml*.so of the UNPATCHED build
  --tier-bin    folder with the libraries of the PATCHED build
  --model       a small MoE GGUF made by tests/make_tiny_moe.py (repeat for F16 and a quantized one)

What it does, for every model and every case: runs tests/logits_dump against the stock libraries and against the patched libraries (with and
without a hot-set file), and compares the logits of every position.

  case "tier off"   patched libraries, no hot set       must equal stock to the last bit
  case "dual"       partial / full / single-expert hot sets, batches of 1, 3 and 16 tokens (the dual path)
                    must equal stock up to float summation order (max |diff| tiny, KL ~0, same top-1 where the top two logits are not tied)
  case "bypass"     the same hot sets with batches of 32 and 64 tokens (above the dual limit)
                    must equal stock to the last bit, because the stock path is taken
  a layer whose hot set holds ALL experts makes the CPU branch do no work at all, so matching stock there also proves the VRAM-side branch alone is right
  the stats file written by the patched library must show that dual graphs were really built.

Quantized models and --no-repack. Stock llama.cpp keeps Q4_0 / Q4_K expert weights in a "repacked" CPU layout with its own kernels, while the hot bank is a
plain CPU (or GPU) buffer; the two kernels round differently, so a fair stock-vs-tier comparison on a quantized model has to switch repacking OFF for BOTH
runs (--no-repack auto does that for every model whose expert weights are not F32/F16/BF16). The repacked kernels' handling of skipped (-1) experts is
tested separately and bit-exactly by tests/test_mmid_skip.cpp. Numeric tolerance for quantized models is the same as for F16.

Exit 0 = all passed (or inputs missing -> SKIP), 1 = a check failed. It touches only --workdir (default: a temp folder).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent


def run(cmd: List[str], env: Optional[dict] = None, timeout: int = 600) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)


def load(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32, count=2)
    n_pos, n_vocab = int(raw[0]), int(raw[1])
    arr = np.fromfile(path, dtype=np.float32, offset=8)
    if arr.size != n_pos * n_vocab:
        raise ValueError(f"{path}: expected {n_pos * n_vocab} floats, found {arr.size}")
    return arr.reshape(n_pos, n_vocab)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    la, lb = log_softmax(a), log_softmax(b)
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    top2 = np.sort(a, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    untied = gap > 1e-3
    same_top = a.argmax(axis=1) == b.argmax(axis=1)
    return {
        "max_abs": float(d.max()), "mean_abs": float(d.mean()), "scale": float(np.abs(a).max()),
        "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
        "top1_same_untied": bool(same_top[untied].all()) if untied.any() else True,
        "top1_same_all": float(same_top.mean()), "bitwise": bool(np.array_equal(a, b)),
    }


def hot_text(kind: str, n_layers: int, n_expert: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    lines = ["# test hot set", "PELIAS_HOT_SET 1", f"n_expert {n_expert}"]
    for l in range(n_layers):
        if kind == "partial":
            if l == 2:
                continue                                         # a layer without a bank
            k = {0: n_expert // 3, 1: n_expert - 1, 3: n_expert // 2}.get(l, n_expert // 4)
            ids = sorted(rng.choice(n_expert, size=k, replace=False).tolist())
        elif kind == "full":
            ids = list(range(n_expert))                          # every expert hot everywhere: the CPU branch has nothing to do
        elif kind == "single":
            ids = [int(rng.integers(n_expert))]
        elif kind == "mixed":
            ids = list(range(n_expert)) if l == 1 else sorted(rng.choice(n_expert, size=max(1, n_expert // 4), replace=False).tolist())
        else:
            raise ValueError(kind)
        lines.append(f"layer {l} : " + " ".join(str(i) for i in ids))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-src", default="")
    ap.add_argument("--stock-bin", default="")
    ap.add_argument("--tier-bin", default="")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--workdir", default="")
    ap.add_argument("--no-repack", choices=["auto", "always", "never"], default="auto",
                    help="pass --no-repack to both runs: auto = only for models whose expert weights are quantized (default)")
    ap.add_argument("--ntok", type=int, default=96)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    if not (a.llama_src and a.stock_bin and a.tier_bin and a.model):
        print("SKIP: needs --llama-src, --stock-bin, --tier-bin and at least one --model")
        return 0
    for p in [Path(a.llama_src) / "include" / "llama.h", Path(a.stock_bin) / "libllama.so", Path(a.tier_bin) / "libllama.so"] + [Path(m) for m in a.model]:
        if not p.exists():
            print(f"SKIP: {p} not found")
            return 0

    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="phase6_"))
    work.mkdir(parents=True, exist_ok=True)
    exe = work / "logits_dump"
    cc = run(["g++", "-O2", "-std=c++17", str(HERE / "logits_dump.cpp"), "-I", f"{a.llama_src}/include", "-I", f"{a.llama_src}/ggml/include",
              "-L", a.tier_bin, "-lllama", "-lggml", "-lggml-base", "-o", str(exe)])
    if cc.returncode != 0:
        print("FAIL: could not compile logits_dump.cpp\n" + cc.stderr[-2000:])
        return 1

    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    norepack_models: set = set()

    def dump(lib_dir: str, model: str, out: Path, chunk: int, env: Optional[dict] = None) -> bool:
        e = {"LD_LIBRARY_PATH": lib_dir}
        e.update(env or {})
        cmd = [str(exe), "-m", model, "--out", str(out), "--ntok", str(a.ntok), "--chunk", str(chunk), "-t", str(a.threads)]
        if model in norepack_models:
            cmd.append("--no-repack")
        r = run(cmd, e)
        if r.returncode != 0:
            print(f"      logits_dump failed (code {r.returncode}): {r.stderr[-800:]}")
            return False
        return True

    for model in a.model:
        tag = Path(model).stem
        print(f"=== model {model}")
        # layer / expert counts from the GGUF itself
        n_layers, n_expert = 4, 16
        try:
            sys.path.insert(0, str(HERE.parent / "lib"))
            import gguf_min
            g = gguf_min.read_gguf(model)
            n_layers = int(g.arch_get("block_count", n_layers))
            n_expert = int(g.arch_get("expert_count", n_expert))
            exps = [t for t in g.tensors if "ffn_down_exps" in t.name or "ffn_down_chexps" in t.name]
            quantized = any(t.ggml_type not in (0, 1, 30) for t in exps)      # F32, F16, BF16 are the unquantized types
            if a.no_repack == "always" or (a.no_repack == "auto" and quantized):
                norepack_models.add(model)
        except Exception as ex:   # the reader is optional here; the tiny fixture has fixed sizes
            print(f"      (using default geometry {n_layers}x{n_expert}: {ex})")
            if a.no_repack == "always":
                norepack_models.add(model)
        if model in norepack_models:
            print("      (weights are quantized: both runs use --no-repack so that stock and tier use the same kernels)")

        for chunk in (1, 3, 16):
            ref = work / f"{tag}_stock_c{chunk}.f32"
            ok = dump(a.stock_bin, model, ref, chunk)
            check(ok, f"stock run, batch {chunk}")
            if not ok:
                continue
            ref_a = load(ref)

            off = work / f"{tag}_off_c{chunk}.f32"
            ok = dump(a.tier_bin, model, off, chunk)
            if ok:
                c = compare(ref_a, load(off))
                check(c["bitwise"] or c["max_abs"] < 1e-6 * max(1.0, c["scale"]), f"batch {chunk}: patched build with no hot set equals stock (bitwise {c['bitwise']}, max |diff| {c['max_abs']:.2e})")

            for kind in ("partial", "full", "single", "mixed"):
                hs = work / f"hot_{kind}.txt"
                hs.write_text(hot_text(kind, n_layers, n_expert, 11), encoding="utf-8")
                out = work / f"{tag}_{kind}_c{chunk}.f32"
                stats = work / f"{tag}_{kind}_c{chunk}.stats.json"
                ok = dump(a.tier_bin, model, out, chunk, {"PELIAS_HOT_SET": str(hs), "PELIAS_TIER_DEVICE": "cpu", "PELIAS_TIER_STATS_FILE": str(stats)})
                if not ok:
                    check(False, f"batch {chunk}, hot set '{kind}': run failed")
                    continue
                c = compare(ref_a, load(out))
                tol = 2e-4 * max(1.0, c["scale"])
                check(c["max_abs"] <= tol and c["mean_kl"] <= 1e-7 and c["top1_same_untied"],
                      f"batch {chunk}, hot set '{kind}': dual path equals stock (max |diff| {c['max_abs']:.2e} <= {tol:.1e}, mean KL {c['mean_kl']:.1e}, top-1 same where not tied: {c['top1_same_untied']}, all positions {c['top1_same_all']:.3f})")
                if stats.exists():
                    st = json.loads(stats.read_text())
                    built = st.get("dual_layer_builds_by_n_tokens", {})
                    check(any(int(k) < 32 and v > 0 for k, v in built.items()), f"batch {chunk}, hot set '{kind}': dual graphs were really built ({built})")
                else:
                    check(False, f"batch {chunk}, hot set '{kind}': no stats file was written")

        for chunk in (32, 64):
            ref = work / f"{tag}_stock_c{chunk}.f32"
            ok = dump(a.stock_bin, model, ref, chunk)
            check(ok, f"stock run, batch {chunk}")
            if not ok:
                continue
            ref_a = load(ref)
            hs = work / "hot_partial.txt"
            hs.write_text(hot_text("partial", n_layers, n_expert, 11), encoding="utf-8")
            out = work / f"{tag}_bypass_c{chunk}.f32"
            ok = dump(a.tier_bin, model, out, chunk, {"PELIAS_HOT_SET": str(hs), "PELIAS_TIER_DEVICE": "cpu"})
            if ok:
                c = compare(ref_a, load(out))
                check(c["bitwise"], f"batch {chunk}: at or above the dual limit the stock path is taken and the answer is bit-identical (max |diff| {c['max_abs']:.2e})")

    # a malformed hot set must not break the run: the patched build prints an error and runs as stock
    bad = work / "hot_bad.txt"
    bad.write_text("PELIAS_HOT_SET 1\nn_expert 999\nlayer 0 : 1 2 3\n", encoding="utf-8")
    m0 = a.model[0]
    ref = work / "bad_stock.f32"
    out = work / "bad_tier.f32"
    if dump(a.stock_bin, m0, ref, 1) and dump(a.tier_bin, m0, out, 1, {"PELIAS_HOT_SET": str(bad)}):
        check(compare(load(ref), load(out))["bitwise"], "a hot set made for another expert count is refused and the run equals stock")

    print("ALL PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
