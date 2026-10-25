#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_dump_hidden.py
"""Checks mtp_align/dump_hidden.cpp on a tiny model: the hidden states it writes, multiplied by the model's output matrix, must give the logits that
tests/logits_dump.cpp reads out of llama.cpp for the same tokens (to within the bf16 rounding of the stored hidden states).
It also checks the file format and the "refuse to overwrite" rule, and the --in sample-file path.

    python tests/test_dump_hidden.py --llama-src /tmp/src/llama.cpp --lib-dir /tmp/build/stock_bin --model /tmp/tiny/tiny_f16.gguf

Needs g++, a built llama.cpp (libllama.so), the gguf-py package folder of the same llama.cpp source, and numpy. It is a build-and-run test of the C++ tool.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT = HERE.parent
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def build(src: Path, lib: Path, cpp: Path, out: Path):
    cmd = ["g++", "-O2", "-std=c++17", str(cpp), f"-I{src}/include", f"-I{src}/ggml/include", f"-L{lib}", "-lllama", "-lggml", "-lggml-base",
           f"-Wl,-rpath,{lib}", "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"build failed: {cpp.name}")


def read_hidden(path: Path):
    b = path.read_bytes()
    magic, ne, ns = struct.unpack("<3i", b[:12])
    pos = 12
    out = []
    for _ in range(ns):
        (n,) = struct.unpack("<i", b[pos:pos + 4])
        pos += 4
        u = np.frombuffer(b[pos:pos + 2 * n * ne], dtype=np.uint16).reshape(n, ne)
        pos += 2 * n * ne
        out.append((u.astype(np.uint32) << 16).view(np.float32))
    return magic, ne, out, pos == len(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-src", required=True)
    ap.add_argument("--lib-dir", required=True)
    ap.add_argument("--model", required=True, help="tiny F16 GGUF with an untied output.weight")
    a = ap.parse_args()
    src, lib, model = Path(a.llama_src), Path(a.lib_dir), Path(a.model)
    sys.path.insert(0, str(src / "gguf-py"))
    import gguf  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dh, ld = td / "dump_hidden", td / "logits_dump"
        build(src, lib, MT / "mtp_align" / "dump_hidden.cpp", dh)
        build(src, lib, MT / "tests" / "logits_dump.cpp", ld)
        n_tok = 96
        h_path, l_path = td / "h.bin", td / "l.f32"
        r1 = subprocess.run([str(dh), "-m", str(model), "--synthetic", str(n_tok), "--seed", "3", "--out", str(h_path), "-t", "2", "--ubatch", "32"], capture_output=True, text=True)
        r2 = subprocess.run([str(ld), "-m", str(model), "--out", str(l_path), "--ntok", str(n_tok), "--chunk", "32", "-t", "2", "--seed", "3"], capture_output=True, text=True)
        check("dump_hidden ran", r1.returncode == 0, r1.stderr[-400:])
        check("logits_dump ran", r2.returncode == 0, r2.stderr[-400:])
        magic, ne, hs, exact = read_hidden(h_path)
        check("file header magic 'PHID'", magic == 0x50484944)
        check("file size matches its own header", exact)
        rd = gguf.GGUFReader(str(model))
        w = [t for t in rd.tensors if t.name == "output.weight"][0]
        W = np.array(w.data, dtype=np.float32).reshape(w.shape[::-1])
        l = np.fromfile(l_path, dtype=np.float32, offset=8).reshape(n_tok, -1)
        h = hs[0]
        check("hidden has one row per token", h.shape == (n_tok, ne == W.shape[1] and ne))
        pred = h @ W.T
        err = np.abs(pred - l).max()
        scale = np.abs(l).max()
        check("hidden @ output_matrix^T reproduces llama.cpp's logits (bf16 rounding only)", err < 0.01 * scale, f"max err {err:.4f} vs max |logit| {scale:.2f}")
        top_same = (pred.argmax(1) == l.argmax(1)).mean()
        check("same arg-max token at (almost) every position", top_same > 0.97, f"{top_same:.3f}")

        # second run: --in path, two samples (one of them the first 40 tokens of the same sequence); each sample starts from an empty context
        rng = np.random.default_rng(1)
        s1 = rng.integers(100, 31000, size=50, dtype=np.int32)
        s2 = rng.integers(100, 31000, size=33, dtype=np.int32)
        ids = td / "s.ids"
        with open(ids, "wb") as f:
            f.write(struct.pack("<ii", 0x50534944, 2))
            for s in (s1, s2):
                f.write(struct.pack("<i", len(s)))
                f.write(s.tobytes())
        h2 = td / "h2.bin"
        r3 = subprocess.run([str(dh), "-m", str(model), "--in", str(ids), "--out", str(h2), "-t", "2", "--ubatch", "16"], capture_output=True, text=True)
        check("--in run ok", r3.returncode == 0, r3.stderr[-400:])
        _, _, hs2, exact2 = read_hidden(h2)
        check("two samples written with their own lengths", [x.shape[0] for x in hs2] == [50, 33] and exact2)
        # sample 2 alone must give the same rows as inside the two-sample run (no state leaks between samples)
        ids_b = td / "s2.ids"
        with open(ids_b, "wb") as f:
            f.write(struct.pack("<ii", 0x50534944, 1))
            f.write(struct.pack("<i", len(s2)))
            f.write(s2.tobytes())
        h3 = td / "h3.bin"
        subprocess.run([str(dh), "-m", str(model), "--in", str(ids_b), "--out", str(h3), "-t", "2", "--ubatch", "16"], capture_output=True, text=True)
        _, _, hs3, _ = read_hidden(h3)
        check("no state leaks from one sample into the next", np.array_equal(hs3[0], hs2[1]))
        r4 = subprocess.run([str(dh), "-m", str(model), "--in", str(ids), "--out", str(h2)], capture_output=True, text=True)
        check("refuses to overwrite an existing output", r4.returncode == 2)
        bad = td / "bad.ids"
        bad.write_bytes(struct.pack("<ii", 0x50534944, 1) + struct.pack("<i", 2) + struct.pack("<ii", 5, 10 ** 7))
        r5 = subprocess.run([str(dh), "-m", str(model), "--in", str(bad), "--out", str(td / "h5.bin")], capture_output=True, text=True)
        check("rejects a token id outside the vocabulary", r5.returncode == 3)

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
