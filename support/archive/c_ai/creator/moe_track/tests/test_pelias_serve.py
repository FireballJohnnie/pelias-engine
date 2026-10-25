#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_pelias_serve.py
"""One runner for every Track 2 (Pelias-Serve) test: builds and runs the C++ tests of pelias_serve/ on the host (CUDA code through the serial CUDA stand-in),
checks pelias_serve/ref_dump.cpp against tests/logits_dump.cpp on a tiny model, and runs the Python self-tests of the E1/E2 comparison script and the E6 policy model.

    python tests/test_pelias_serve.py --llama-src /tmp/src/llama.cpp --lib-dir /tmp/build/stock_bin --model /tmp/tiny/tiny_f16.gguf [--quick] [--tsan] [--only NAME[,NAME]]

--llama-src  llama.cpp source folder (headers)           --lib-dir  folder with libggml*.so / libllama.so of a build of that source
--model      a tiny F16 GGUF (tests/make_tiny_moe.py makes one)                      --quick   shorter randomised loops    --tsan   also run cold_pool under ThreadSanitizer
Needs g++ and numpy. Compiles into a temporary folder and deletes it afterwards. Exit 0 = everything passed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT = HERE.parent
PS = MT / "pelias_serve"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"\n{detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def sh(cmd, env=None, cwd=None, timeout=900):
    t = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    return r, time.time() - t


def tail(r, n=1800):
    return (r.stdout + r.stderr)[-n:]


def cpp_test(name, cpp, flags, work, env, run_args=(), extra_check=None):
    exe = work / name
    r, t = sh(["g++", *flags, str(cpp), "-I" + str(PS), "-o", str(exe)] if not isinstance(flags, tuple) else ["g++", *flags, str(cpp), "-o", str(exe)])
    check(f"{name}: compiles", r.returncode == 0, tail(r, 3000))
    if r.returncode != 0:
        return None
    r, t = sh([exe, *run_args], env=env, cwd=work)
    out = r.stdout + r.stderr
    n_ok = sum(1 for l in out.splitlines() if l.startswith("ok"))
    n_fail = sum(1 for l in out.splitlines() if l.startswith("FAIL"))
    check(f"{name}: exit 0, {n_ok} checks ok, {n_fail} failed ({t:.0f} s)", r.returncode == 0 and n_fail == 0 and n_ok > 0, tail(r, 3000))
    if extra_check:
        extra_check(work, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-src", required=True)
    ap.add_argument("--lib-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tsan", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    src, lib, model = Path(a.llama_src), Path(a.lib_dir), Path(a.model)
    only = {x for x in a.only.split(",") if x}

    def want(n):
        return not only or n in only

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    ggml_inc = ["-I" + str(src / "ggml" / "include")]
    ggml_lib = ["-L" + str(lib), "-lggml", "-lggml-cpu", "-lggml-base", "-Wl,-rpath," + str(lib)]

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)

        if want("cold_pool"):
            print("== E3 cold pool")
            args = ["--quick"] if a.quick else []
            exe = work / "test_cold_pool"
            r, _ = sh(["g++", "-O2", "-std=c++17", "-pthread", str(HERE / "test_cold_pool.cpp"), "-I" + str(PS), *ggml_inc, *ggml_lib, "-o", str(exe)])
            check("test_cold_pool: compiles", r.returncode == 0, tail(r, 3000))
            if r.returncode == 0:
                r, t = sh([exe, *args], env=env, cwd=work)
                out = r.stdout + r.stderr
                check(f"test_cold_pool: exit 0, no FAIL line ({t:.0f} s)", r.returncode == 0 and "FAIL" not in out and "ok" in out, tail(r, 3000))
            if a.tsan:
                exe2 = work / "test_cold_pool_tsan"
                r, _ = sh(["g++", "-O1", "-g", "-fsanitize=thread", "-std=c++17", "-pthread", str(HERE / "test_cold_pool.cpp"), "-I" + str(PS), *ggml_inc, *ggml_lib, "-o", str(exe2)])
                check("test_cold_pool (thread sanitizer): compiles", r.returncode == 0, tail(r, 3000))
                if r.returncode == 0:
                    r, t = sh([exe2, "--quick"], env=env, cwd=work, timeout=1800)
                    out = r.stdout + r.stderr
                    check(f"test_cold_pool (thread sanitizer): exit 0, no data race reported ({t:.0f} s)", r.returncode == 0 and "WARNING: ThreadSanitizer" not in out and "FAIL" not in out, tail(r, 3000))

        if want("e5"):
            print("== E5 DeltaNet state snapshot")
            cpp_test("test_e5_state", HERE / "test_e5_state.cpp", ("-O2", "-std=c++17", "-DPELIAS_CUDA_SHIM", "-I" + str(PS)), work, env)

        if want("e6"):
            print("== E6 hot-aware draft gate kernel")
            cpp_test("test_e6_gate", HERE / "test_e6_gate.cpp", ("-O2", "-std=c++20", "-pthread", "-DPELIAS_CUDA_SHIM", "-I" + str(PS)), work, env)

        if want("e0"):
            print("== E0 layer profiler")
            def e0_json(work_, out):
                pass
            cpp_test("test_e0_profiler", HERE / "test_e0_profiler.cpp", ("-O2", "-std=c++17", "-Wall", "-Werror", "-DPELIAS_CUDA_SHIM", "-I" + str(PS)), work, env, run_args=[str(work / "e0.json")])
            if (work / "e0.json").exists():
                import json
                try:
                    j = json.loads((work / "e0.json").read_text())
                    check("e0: the profiler's JSON file parses", isinstance(j, (dict, list)) and len(j) > 0)
                except Exception as e:  # noqa: BLE001
                    check("e0: the profiler's JSON file parses", False, str(e))

        if want("ref_dump"):
            print("== reference dump tool vs the existing logits tool")
            rd, ld = work / "ref_dump", work / "logits_dump"
            for cpp, exe in ((PS / "ref_dump.cpp", rd), (HERE / "logits_dump.cpp", ld)):
                r, _ = sh(["g++", "-O2", "-std=c++17", str(cpp), f"-I{src}/include", *ggml_inc, f"-L{lib}", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{lib}", "-o", str(exe)])
                check(f"{cpp.name}: compiles", r.returncode == 0, tail(r, 3000))
                if r.returncode != 0:
                    break
            else:
                n_tok, K = 40, 3
                out_dir = work / "dumps"
                r, _ = sh([rd, "-m", model, "--out-dir", out_dir, "--synthetic", n_tok, "--seed", 3, "--layers", "0,1", "--logits-last", K, "-t", 2, "--ngl", 0], env=env)
                check("ref_dump runs", r.returncode == 0, tail(r))
                r2, _ = sh([rd, "-m", model, "--out-dir", out_dir, "--synthetic", n_tok, "--seed", 3, "--layers", "0", "-t", 2], env=env)
                check("ref_dump refuses to overwrite earlier dumps", r2.returncode != 0 and "already exists" in (r2.stderr + r2.stdout))
                r3, _ = sh([ld, "-m", model, "--out", work / "l.f32", "--ntok", n_tok, "--chunk", n_tok, "-t", 2, "--seed", 3], env=env)
                check("logits_dump runs", r3.returncode == 0, tail(r3))
                if r.returncode == 0 and r3.returncode == 0:
                    l0 = np.load(out_dir / "ref_layer_0.npy")
                    l1 = np.load(out_dir / "ref_layer_1.npy")
                    lg = np.load(out_dir / "ref_logits_last.npy")
                    check("layer dumps are float32 [n_tokens, n_embd] and finite", l0.dtype == np.float32 and l0.shape[0] == n_tok and l0.shape == l1.shape and np.isfinite(l0).all() and np.isfinite(l1).all())
                    check("layer 0 and layer 1 differ (the hook really saw two layers)", not np.array_equal(l0, l1))
                    ref = np.fromfile(work / "l.f32", dtype=np.float32, offset=8).reshape(n_tok, -1)[-K:]
                    check("last-K logits equal logits_dump's (same tokens)", lg.shape == ref.shape and np.abs(lg - ref).max() <= 1e-3 * max(1.0, np.abs(ref).max()), f"{np.abs(lg - ref).max() if lg.shape == ref.shape else lg.shape}")
                    toks = (out_dir / "ref_tokens.txt").read_text().split()
                    check("token file lists every token", len(toks) == n_tok)
                    # the comparison script accepts the reference against itself and rejects a perturbed copy
                    hp = PS / "e1_e2_parity_harness.py"
                    for f in ("layer_0", "layer_1", "logits_last", "tokens"):
                        ext = "txt" if f == "tokens" else "npy"
                        (out_dir / f"pelias_serve_{f}.{ext}").write_bytes((out_dir / f"ref_{f}.{ext}").read_bytes())
                    r4, _ = sh([sys.executable, "-B", hp, "e1", "--dump-dir", out_dir, "--layers", "0,1"])
                    check("harness e1: reference vs identical copy passes", r4.returncode == 0 and "PASS" in r4.stdout, tail(r4))
                    r5, _ = sh([sys.executable, "-B", hp, "e2", "--dump-dir", out_dir])
                    check("harness e2: reference vs identical copy passes", r5.returncode == 0 and "PASS" in r5.stdout, tail(r5))
                    np.save(out_dir / "pelias_serve_layer_0.npy", l0 * 1.03)
                    r6, _ = sh([sys.executable, "-B", hp, "e1", "--dump-dir", out_dir, "--layers", "0"])
                    check("harness e1: a 3 % error is rejected", r6.returncode != 0, tail(r6))

        if want("py"):
            print("== Python self-tests")
            r, _ = sh([sys.executable, "-B", PS / "e1_e2_parity_harness.py", "selftest"])
            check("e1_e2_parity_harness.py selftest", r.returncode == 0 and "ALL PASSED" in r.stdout, tail(r))
            r, _ = sh([sys.executable, "-B", PS / "e6_policy_model.py", "--selftest"])
            check("e6_policy_model.py --selftest", r.returncode == 0, tail(r))
            r, _ = sh([sys.executable, "-B", PS / "e1_e2_parity_harness.py", "howto"])
            check("harness howto prints commands and runs nothing", r.returncode == 0 and "ref_dump" in r.stdout)

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
