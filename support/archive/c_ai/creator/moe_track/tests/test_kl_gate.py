#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_kl_gate.py
"""Tests sim/kl_gate.py: the log parser on a real llama-perplexity report, and (if the tool and a tokenized tiny model exist) a full base/compare run.

    python tests/test_kl_gate.py [--ppl-bin DIR_WITH_llama-perplexity --model tiny_tok_f16.gguf --text some.txt]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
sys.path.insert(0, str(HERE.parent / "lib"))
import kl_gate  # noqa: E402

SAMPLE = """0.00.102.793 I perplexity: calculating perplexity over 4 chunks, n_ctx=256, batch_size=256, n_seq=1
====== KL divergence statistics ======
Mean    KLD:   0.001234 ±   0.000100
Maximum KLD:   0.020000
99.9%   KLD:   0.015000
99.0%   KLD:   0.009000
Median  KLD:   0.000500
Mean    Δp: -0.010 ± 0.002 %
RMS Δp    :  0.900 ± 0.050 %
Same top p:  99.250 ± 0.100 %
Mean PPL(Q)                   :   5.100000 ±   0.100000
Mean PPL(base)                :   5.000000 ±   0.100000
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl-bin", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--text", default="")
    a = ap.parse_args()
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    r = kl_gate.parse_log(SAMPLE)
    check(abs(r["mean_kl"] - 0.001234) < 1e-9 and abs(r["top1_agreement"] - 0.9925) < 1e-9, "parser: mean KL and top-1 agreement")
    check(r["n_chunks"] == 4 and r["n_ctx"] == 256 and r["n_tokens"] == 4 * 127, "parser: scored token count = chunks * (ctx - 1 - ctx/2)")
    check(abs(r["kld_p99"] - 0.009) < 1e-9 and abs(r["ppl_candidate"] - 5.1) < 1e-9, "parser: percentiles and perplexities")
    try:
        kl_gate.parse_log("nothing here")
        check(False, "parser rejects a log without the report")
    except ValueError:
        check(True, "parser rejects a log without the report")

    if a.ppl_bin and a.model and a.text and (Path(a.ppl_bin) / "llama-perplexity").exists():
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            env = dict(os.environ)
            common = ["--model", a.model, "--bin", a.ppl_bin, "--text", a.text, "--ctx", "256", "--chunks", "4", "--kl-dir", str(d), "--vocab", "32000", "--force"]
            def run(args):
                return subprocess.run([sys.executable, str(HERE.parent / "sim" / "kl_gate.py")] + args, capture_output=True, text=True, env=env)
            # the tiny model has a random router: -ngl 99 / --n-cpu-moe are accepted by a CPU-only build too
            p = run(["base", "--label", "ref"] + common + ["--extra", "-t 2"])
            check(p.returncode == 0 and (d / "ref.logits.bin").exists(), "base run writes the reference logits" + ("" if p.returncode == 0 else "\n" + p.stdout[-300:] + p.stderr[-500:]))
            hdr = kl_gate.logits_header(d / "ref.logits.bin")
            check(hdr is not None and hdr["n_ctx"] == 256 and hdr["n_chunk"] == 4, f"logits header readable: {hdr}")
            p = run(["compare", "--label", "same", "--base", "ref"] + common + ["--extra", "-t 2"])
            check(p.returncode == 0, "compare run succeeds" + ("" if p.returncode == 0 else "\n" + p.stdout[-300:] + p.stderr[-500:]))
            j = d / "same_vs_ref.json"
            if j.exists():
                res = json.loads(j.read_text())
                check(res["mean_kl"] < 1e-4 and res["top1_agreement"] > 0.999 and res["n_tokens"] == 508, f"same model against its own reference: KL {res['mean_kl']:.2e}, top-1 {res['top1_agreement']*100:.2f}%, {res['n_tokens']} tokens")
                g = subprocess.run([sys.executable, str(HERE.parent / "lib" / "gates.py"), "h6", "--kl", str(j)], capture_output=True, text=True)
                check(g.returncode == 0 and "PASS" in g.stdout, "gate H6 passes it")
            else:
                check(False, "compare wrote no result json")
            # a different model: perturbed expert count must differ -> KL clearly above the gate
            p = run(["compare", "--label", "topk2", "--base", "ref"] + common + ["--extra", "-t 2 --override-kv qwen3moe.expert_used_count=int:2"])
            j2 = d / "topk2_vs_ref.json"
            if j2.exists():
                res = json.loads(j2.read_text())
                g = subprocess.run([sys.executable, str(HERE.parent / "lib" / "gates.py"), "h6", "--kl", str(j2)], capture_output=True, text=True)
                check(res["mean_kl"] > 0.005 and g.returncode == 1, f"a model that uses 2 experts instead of 4 is caught: KL {res['mean_kl']:.4f}, gate says {'FAIL' if g.returncode == 1 else 'PASS'}")
            else:
                check(False, "perturbed compare wrote no result" + "\n" + p.stdout[-300:] + p.stderr[-400:])
            p = run(["compare", "--label", "x", "--base", "missing"] + common)
            check(p.returncode == 3, "compare against a missing reference is refused")
    else:
        print("SKIP: end-to-end part (needs --ppl-bin, --model and --text)")
    print("FAILURES: %d" % fails if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
