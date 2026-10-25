#!/usr/bin/env python3
# File: <project>\moe_track\pelias_serve\e1_e2_parity_harness.py
"""Stages E1 and E2 of Pelias-Serve: does the new executor compute the same thing as stock llama.cpp?

This script only COMPARES files. Two sets of .npy dumps of the same token sequence are needed:
    reference (stock llama.cpp):   made by pelias_serve/ref_dump.cpp      ref_layer_<L>.npy, ref_logits_last.npy, ref_tokens.txt
    executor under test:           must be written by the Pelias-Serve executor (E4) in the same format and names, prefix `pelias_serve`:
                                   pelias_serve_layer_<L>.npy  (float32 [n_tokens, 2048], the output of decoder layer L),  pelias_serve_logits_last.npy (float32 [K, n_vocab])
The executor (E4) and the dump hook in it are NOT written in this package (see UNFINISHED.md), so until they exist `e1`/`e2` report "missing" for the executor's files and stop.

    python e1_e2_parity_harness.py howto                         the commands that produce the reference dumps (nothing is run)
    python e1_e2_parity_harness.py e1 [--dump-dir D] [--layers 0,3]
    python e1_e2_parity_harness.py e2 [--dump-dir D]
    python e1_e2_parity_harness.py compare A.npy B.npy [--kind hidden|logits]
    python e1_e2_parity_harness.py selftest

Pass criteria (the roadmap's, made scale-aware because the model's hidden states contain a few very large values):
  E1, per layer:  relative L2 error < 1e-3,  max |difference| < 5e-3 x max |reference|,  no NaN/Inf, same shape.
  E2, logits   :  mean KL(reference || executor) over the reference's top-20 tokens < 1e-4 (softmax restricted to those 20, renormalised),  top-1 agreement >= 0.99.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
try:
    import paths  # noqa: E402
    DEFAULT_DUMP_DIR = Path(paths.LOGS) / "floors" / "e1_dumps"
    CREATOR = Path(paths.CREATOR)
    LLAMA_PROD = Path(paths.LLAMA_PROD)
    GGUF = Path(paths.GGUF)
except Exception:  # noqa: BLE001
    CREATOR = Path("<project>")
    DEFAULT_DUMP_DIR = CREATOR / "logs" / "floors" / "e1_dumps"
    LLAMA_PROD = Path("<llama.cpp>")
    GGUF = CREATOR / "gguf" / "test-model-Q4_K_M.gguf"

E1_REL_L2 = 1e-3
E1_MAX_ABS_REL = 5e-3
E2_TOP20_KL = 1e-4
E2_TOP1_AGREE = 0.99
REF, CUR = "ref", "pelias_serve"


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False).astype(np.float32, copy=False)


def compare_hidden(ref: np.ndarray, cur: np.ndarray) -> Tuple[bool, Dict[str, float], str]:
    if ref.shape != cur.shape:
        return False, {}, f"shape {cur.shape} != reference {ref.shape}"
    if not (np.isfinite(ref).all() and np.isfinite(cur).all()):
        return False, {}, "NaN or Inf in a dump"
    diff = ref.astype(np.float64) - cur.astype(np.float64)
    rn = float(np.linalg.norm(ref.astype(np.float64)))
    rel = float(np.linalg.norm(diff) / max(rn, 1e-30))
    scale = float(np.max(np.abs(ref))) or 1.0
    mx = float(np.max(np.abs(diff)))
    ok = rel < E1_REL_L2 and mx < E1_MAX_ABS_REL * scale
    return ok, {"rel_l2": rel, "max_abs": mx, "max_abs_over_scale": mx / scale, "scale": scale}, ""


def topk_kl(ref: np.ndarray, cur: np.ndarray, k: int = 20) -> np.ndarray:
    """KL(ref || cur) per row over the reference's top-k tokens (both softmaxed over those k only)."""
    idx = np.argpartition(-ref, k - 1, axis=-1)[:, :k]
    r = np.take_along_axis(ref, idx, axis=-1).astype(np.float64)
    c = np.take_along_axis(cur, idx, axis=-1).astype(np.float64)
    r -= r.max(-1, keepdims=True)
    c -= c.max(-1, keepdims=True)
    lr = r - np.log(np.exp(r).sum(-1, keepdims=True))
    lc = c - np.log(np.exp(c).sum(-1, keepdims=True))
    return (np.exp(lr) * (lr - lc)).sum(-1)


def compare_logits(ref: np.ndarray, cur: np.ndarray) -> Tuple[bool, Dict[str, float], str]:
    if ref.ndim != 2 or ref.shape != cur.shape:
        return False, {}, f"shape {cur.shape} != reference {ref.shape}"
    if not (np.isfinite(ref).all() and np.isfinite(cur).all()):
        return False, {}, "NaN or Inf in a dump"
    if ref.shape[0] == 0:
        return False, {}, "no positions"
    kl = topk_kl(ref, cur, min(20, ref.shape[1]))
    agree = float(np.mean(ref.argmax(-1) == cur.argmax(-1)))
    ok = float(kl.mean()) < E2_TOP20_KL and agree >= E2_TOP1_AGREE
    return ok, {"top20_kl_mean": float(kl.mean()), "top20_kl_max": float(kl.max()), "top1_agree": agree, "positions": float(ref.shape[0])}, ""


def tokens_match(d: Path) -> Tuple[bool, str]:
    a, b = d / f"{REF}_tokens.txt", d / f"{CUR}_tokens.txt"
    if not a.exists():
        return False, f"{a} is missing (ref_dump writes it)"
    if b.exists() and a.read_text().split() != b.read_text().split():
        return False, "the executor ran a different token sequence than the reference"
    return True, ""


def run_e1(d: Path, layers: List[int]) -> int:
    ok_t, why = tokens_match(d)
    if not ok_t:
        print(f"[E1] {why}")
        return 1
    all_ok = True
    for L in layers:
        kind = "full attention" if L % 4 == 3 else "DeltaNet"
        rp, cp = d / f"{REF}_layer_{L}.npy", d / f"{CUR}_layer_{L}.npy"
        if not rp.exists() or not cp.exists():
            miss = [str(p) for p in (rp, cp) if not p.exists()]
            print(f"[E1] layer {L} ({kind}): missing {', '.join(miss)}")
            all_ok = False
            continue
        ok, m, why = compare_hidden(load(rp), load(cp))
        if why:
            print(f"[E1] layer {L} ({kind}): FAIL  {why}")
        else:
            print(f"[E1] layer {L} ({kind}): rel_L2 {m['rel_l2']:.3e}  max|diff| {m['max_abs']:.3e} ({m['max_abs_over_scale']:.2e} of the reference's largest value {m['scale']:.2f})  -> {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print("[E1] " + ("PASS" if all_ok else "NOT PASSED"))
    return 0 if all_ok else 1


def run_e2(d: Path) -> int:
    ok_t, why = tokens_match(d)
    if not ok_t:
        print(f"[E2] {why}")
        return 1
    rp, cp = d / f"{REF}_logits_last.npy", d / f"{CUR}_logits_last.npy"
    if not rp.exists() or not cp.exists():
        print(f"[E2] missing {', '.join(str(p) for p in (rp, cp) if not p.exists())}")
        return 1
    ok, m, why = compare_logits(load(rp), load(cp))
    if why:
        print(f"[E2] FAIL  {why}")
        return 1
    print(f"[E2] {int(m['positions'])} positions: top-20 KL mean {m['top20_kl_mean']:.3e} (max {m['top20_kl_max']:.3e}), top-1 agreement {m['top1_agree']:.3f}  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def howto(a) -> int:
    d = Path(a.dump_dir)
    print(f"""Reference dumps (stock llama.cpp, your pinned build). Nothing below is run by this script.

  # 1) build the dump tool once (WSL)
  cd {HERE}
  g++ -O2 -std=c++17 ref_dump.cpp -I{LLAMA_PROD}/include -I{LLAMA_PROD}/ggml/include -L{LLAMA_PROD}/build-wsl/bin -lllama -lggml -lggml-base \\
      -Wl,-rpath,{LLAMA_PROD}/build-wsl/bin -o ref_dump

  # 2) dump layer 0 (DeltaNet), layer 3 (full attention) and the last 8 positions' logits for a held-out prompt (the llama-server must be STOPPED: llama --stop)
  mkdir -p {d}
  ./ref_dump -m {GGUF} --out-dir {d} --prompt-file YOUR_HELD_OUT_PROMPT.txt --layers 0,3 --logits-last 8 -t 8 --ngl 99 --n-cpu-moe 28

  # 3) run the same token sequence ({d}/ref_tokens.txt) through the Pelias-Serve executor and let it write
  #    {CUR}_layer_0.npy, {CUR}_layer_3.npy, {CUR}_logits_last.npy into the same folder   (the executor is not part of this package)

  # 4) compare
  python {Path(__file__).name} e1 && python {Path(__file__).name} e2
""")
    return 0


def selftest() -> int:
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    rng = np.random.default_rng(0)
    H = (rng.standard_normal((16, 64)) * 2).astype(np.float32)
    H[3, 5] = 300.0                                                      # a massive activation, as in real hidden states
    ok, m, _ = compare_hidden(H, H.copy())
    check("hidden: identical arrays pass with zero error", ok and m["max_abs"] == 0)
    ok, m, _ = compare_hidden(H, H + rng.standard_normal(H.shape).astype(np.float32) * 1e-3)
    check("hidden: noise of 1e-3 on values of order 1..300 passes", ok)
    ok, m, _ = compare_hidden(H, H * 1.01)
    check("hidden: a 1 % scale error fails", not ok)
    X = H.copy(); X[0, 0] += 5.0
    ok, m, _ = compare_hidden(H, X)
    check("hidden: a single element off by 5 (1.7 % of the largest value) fails the max-abs rule", not ok)
    ok, _, why = compare_hidden(H, H[:, :32])
    check("hidden: shape mismatch is reported", not ok and "shape" in why)
    Y = H.copy(); Y[1, 1] = np.nan
    ok, _, why = compare_hidden(H, Y)
    check("hidden: NaN is reported", not ok and "NaN" in why)

    V = 5000
    Lg = (rng.standard_normal((6, V)) * 3).astype(np.float32)
    ok, m, _ = compare_logits(Lg, Lg.copy())
    check("logits: identical pass with KL 0 and full agreement", ok and m["top20_kl_mean"] < 1e-12 and m["top1_agree"] == 1.0)
    ok, m, _ = compare_logits(Lg, Lg + rng.standard_normal(Lg.shape).astype(np.float32) * 1e-3)
    check("logits: noise of 1e-3 passes", ok)
    ok, m, _ = compare_logits(Lg, Lg + rng.standard_normal(Lg.shape).astype(np.float32) * 0.3)
    check("logits: noise of 0.3 fails", not ok)
    sw = Lg.copy()
    top = sw.argmax(-1)
    sw[np.arange(6), top] -= 50.0
    ok, m, _ = compare_logits(Lg, sw)
    check("logits: a changed top-1 token fails", not ok and m["top1_agree"] < 1.0)
    k = topk_kl(np.array([[0.0, 1.0, 2.0] + [-9.0] * 30]), np.array([[0.0, 1.0, 2.0] + [-9.0] * 30]), 3)
    check("top-k KL of equal rows is 0", abs(float(k[0])) < 1e-12)
    p = np.array([[2.0, 0.0] + [-20.0] * 20]); q = np.array([[0.0, 0.0] + [-20.0] * 20])
    kl = float(topk_kl(p, q, 2)[0])
    pr = np.exp([2.0, 0.0]); pr /= pr.sum()
    check("top-k KL equals the textbook value on two tokens", abs(kl - float((pr * np.log(pr / np.array([0.5, 0.5]))).sum())) < 1e-9)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / f"{REF}_tokens.txt").write_text("1 2 3\n")
        np.save(d / f"{REF}_layer_0.npy", H)
        check("e1: executor files missing -> not passed", run_e1(d, [0]) == 1)
        np.save(d / f"{CUR}_layer_0.npy", H.copy())
        check("e1: matching files pass", run_e1(d, [0]) == 0)
        np.save(d / f"{CUR}_layer_0.npy", H * 1.02)
        check("e1: a 2 % error fails", run_e1(d, [0]) == 1)
        (d / f"{CUR}_tokens.txt").write_text("1 2 4\n")
        check("e1: a different token sequence is refused", run_e1(d, [0]) == 1)
        (d / f"{CUR}_tokens.txt").write_text("1 2 3\n")
        np.save(d / f"{CUR}_layer_0.npy", H.copy())
        np.save(d / f"{REF}_logits_last.npy", Lg)
        np.save(d / f"{CUR}_logits_last.npy", Lg + 1e-4)
        check("e2: matching logits pass", run_e2(d) == 0)
        np.save(d / f"{CUR}_logits_last.npy", sw)
        check("e2: a changed prediction fails", run_e2(d) == 1)
        check("e1: a layer with no dump at all is reported as missing, not as a pass", run_e1(d, [0, 3]) == 1)
    print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["e1", "e2", "compare", "howto", "selftest"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dump-dir", default=str(DEFAULT_DUMP_DIR))
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--kind", choices=["hidden", "logits"], default="hidden")
    a = ap.parse_args(argv)
    if a.action == "selftest":
        return selftest()
    if a.action == "howto":
        return howto(a)
    if a.action == "e1":
        return run_e1(Path(a.dump_dir), [int(x) for x in a.layers.split(",") if x != ""])
    if a.action == "e2":
        return run_e2(Path(a.dump_dir))
    if len(a.files) != 2:
        print("compare needs two .npy files")
        return 2
    fn = compare_hidden if a.kind == "hidden" else compare_logits
    ok, m, why = fn(load(Path(a.files[0])), load(Path(a.files[1])))
    print(why or " ".join(f"{k}={v:.4g}" for k, v in m.items()), "->", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
