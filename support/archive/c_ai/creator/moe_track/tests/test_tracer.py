#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_tracer.py
"""Checks trace/moe_trace against the tiny test model. Needs the built tracer and the tiny F16 GGUF; SKIPS (exit 0) if either is missing.

    python tests/test_tracer.py --tracer trace/bin/moe_trace --model tiny_f16.gguf [--gguf-py /path/to/llama.cpp/gguf-py]

Checks
  1. independent: a single-token prompt is run through the whole tiny network in numpy (at position 0 attention reduces to
     the value projection), and the experts the tracer recorded for that token must equal numpy's top-k in every layer;
  2. decode rows hold K distinct in-range experts, router weights are descending and in (0,1];
  3. prefill: only the final token has the last layer recorded; all other layers are complete;
  4. the recording does not depend on the physical batch size (--ubatch 4 versus 32), nor on running twice (greedy);
  5. file round trip through pltrace.write_trace / read_trace.
It touches only a temp folder.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
import pltrace  # noqa: E402


def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean() + eps) * w


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def forward_routing(reader, token: int, n_head=4, n_kv=2, hd=64, topk=4):
    """Routing of every layer for one token at position 0 of an empty context, in float32."""
    t = {x.name: x for x in reader.tensors}
    g = lambda name: np.asarray(t[name].data, dtype=np.float32)   # noqa: E731
    h = g("token_embd.weight")[token].copy()
    n_layers = sum(1 for n in t if n.endswith(".attn_norm.weight"))
    picks, weights = [], []
    for l in range(n_layers):
        b = f"blk.{l}"
        xn = rmsnorm(h, g(f"{b}.attn_norm.weight"))
        v = (g(f"{b}.attn_v.weight") @ xn).reshape(n_kv, hd)
        attn = np.concatenate([v[hh // (n_head // n_kv)] for hh in range(n_head)])   # softmax over one position is 1
        h = h + g(f"{b}.attn_output.weight") @ attn
        xn2 = rmsnorm(h, g(f"{b}.ffn_norm.weight"))
        probs = softmax(g(f"{b}.ffn_gate_inp.weight") @ xn2)
        idx = np.argsort(-probs, kind="stable")[:topk]
        w = probs[idx] / probs[idx].sum()
        gate, up, down = g(f"{b}.ffn_gate_exps.weight"), g(f"{b}.ffn_up_exps.weight"), g(f"{b}.ffn_down_exps.weight")
        out = np.zeros_like(h)
        for e, we in zip(idx, w):
            a = gate[e] @ xn2
            out += we * (down[e] @ ((a / (1.0 + np.exp(-a))) * (up[e] @ xn2)))
        h = h + out
        picks.append(idx.astype(np.int64))
        weights.append(probs[idx])
    return np.array(picks), np.array(weights)


def run_tracer(tracer, model, out, extra):
    cmd = [str(tracer), "-m", str(model), "--out", str(out), "--ctx", "256", "--kv-type", "f16", "--flash-attn", "off", "--threads", "2",
           "--quiet", "--ignore-eos", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default=str(HERE.parent / "trace" / "bin" / "moe_trace"))
    ap.add_argument("--model", default="")
    ap.add_argument("--gguf-py", default="")
    a = ap.parse_args()
    if not Path(a.tracer).exists() or not a.model or not Path(a.model).exists():
        print("SKIP: tracer binary or tiny F16 model not available")
        return 0
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    import gguf  # noqa: E402

    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    reader = gguf.GGUFReader(a.model)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # 1. independent forward pass, single-token prompts
        for tok in (5, 17, 40):
            rc, txt = run_tracer(a.tracer, a.model, d / f"one{tok}", ["--tokens", str(tok), "--n-predict", "1", "--trace-prefill", "--weights", "--greedy"])
            check(rc == 0, f"token {tok}: tracer exits 0" + ("" if rc == 0 else "\n" + txt[-600:]))
            if rc != 0:
                continue
            tr = pltrace.read_trace(d / f"one{tok}")
            exp, wts = forward_routing(reader, tok, topk=tr.k)
            got = tr.prefill.experts[0].astype(np.int64)          # [L, K], position 0
            same_list = np.array_equal(got, exp)
            same_set = all(set(got[l]) == set(exp[l]) for l in range(len(exp)))
            check(same_set, f"token {tok}: recorded experts equal numpy's top-{tr.k} in all {len(exp)} layers (ordered match: {same_list})")
            gw = tr.prefill.weights[0].astype(np.float32)
            check(np.allclose(np.sort(gw, axis=1), np.sort(wts, axis=1), rtol=2e-2, atol=1e-3), f"token {tok}: recorded router probabilities match numpy (F16 activations in llama.cpp, f32 here: loose tolerance)")

        # 2. structure of a longer greedy run
        rc, txt = run_tracer(a.tracer, a.model, d / "long", ["--random-tokens", "12", "--n-predict", "24", "--ubatch", "8", "--trace-prefill", "--weights", "--greedy"])
        check(rc == 0, "long run exits 0")
        tr = pltrace.read_trace(d / "long")
        K, L, E = tr.k, len(tr.moe_layers), tr.n_expert
        check(tr.decode.n == 24 and tr.prefill.n == 12, f"counts: {tr.decode.n} decode, {tr.prefill.n} prefill tokens")
        ex = tr.decode.experts.astype(np.int64)
        check(bool((ex < E).all()) and all(len(set(ex[i, l])) == K for i in range(tr.decode.n) for l in range(L)), "every decode row: K distinct experts, all in range")
        w = tr.decode.weights.astype(np.float32)
        check(bool((w >= 0).all() and (w <= 1.0).all() and (np.diff(w, axis=2) <= 1e-3).all()), "router weights in [0,1] (half precision can underflow tiny ones to 0) and descending within a row")
        pe = tr.prefill.experts
        check(bool((pe[:, :-1, :] != pluser(0xFFFF)).all()) and bool((pe[:-1, -1, :] == 0xFFFF).all()) and bool((pe[-1, -1, :] != 0xFFFF).all()),
              "prefill: last layer only recorded for the final token")

        # 3. independent of physical batch size, repeatable
        outs = {}
        for ub in (4, 32):
            rc, _ = run_tracer(a.tracer, a.model, d / f"ub{ub}", ["--random-tokens", "12", "--n-predict", "24", "--ubatch", str(ub), "--trace-prefill", "--greedy"])
            outs[ub] = pltrace.read_trace(d / f"ub{ub}") if rc == 0 else None
        if outs[4] is not None and outs[32] is not None:
            check(np.array_equal(outs[4].decode.experts, outs[32].decode.experts), "decode routing is identical for --ubatch 4 and 32")
            check(np.array_equal(outs[4].decode.tokens, outs[32].decode.tokens), "greedy generated tokens are identical")
            lastcol = slice(None, -1)
            check(np.array_equal(outs[4].prefill.experts[:, lastcol], outs[32].prefill.experts[:, lastcol]), "prefill routing (layers with full coverage) is identical")
        else:
            check(False, "ubatch comparison runs failed")

        # 4. multiple prompts, jsonl, domains, sampling determinism with a seed
        pj = d / "p.jsonl"
        pj.write_text('{"id":"a","domain":"code","tokens":[3,4,5,6]}\n{"id":"b","domain":"prose","tokens":[7,8,9]}\n', encoding="utf-8")
        for name in ("s1", "s2"):
            run_tracer(a.tracer, a.model, d / name, ["--prompts-jsonl", str(pj), "--n-predict", "10", "--seed", "11"])
        t1, t2 = pltrace.read_trace(d / "s1"), pltrace.read_trace(d / "s2")
        check(t1.decode.n == 20 and sorted(t1.decode_by_domain()) == ["code", "prose"], "two prompts, two domains, 10 decode tokens each")
        check(np.array_equal(t1.decode.experts, t2.decode.experts), "same seed gives the same sampled run")

        # 5. writer / reader round trip
        hdr = {"moe_layers": [0, 1], "n_expert": 8, "n_expert_used": 2, "weights": False, "arch": "synthetic"}
        blk = [{"kind": 1, "seq": 0, "pos0": 3, "tokens": [1, 2], "experts": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], np.uint16)}]
        pltrace.write_trace(d / "rt", hdr, blk, [{"seq": 0, "domain": "x"}])
        rt = pltrace.read_trace(d / "rt")
        check(rt.decode.n == 2 and rt.decode.experts[1, 1, 1] == 7 and rt.decode.pos.tolist() == [3, 4], "round trip through write_trace/read_trace")
    print("FAILURES: %d" % fails if fails else "all passed")
    return 1 if fails else 0


def pluser(v):
    return np.uint16(v)


if __name__ == "__main__":
    sys.exit(main())
