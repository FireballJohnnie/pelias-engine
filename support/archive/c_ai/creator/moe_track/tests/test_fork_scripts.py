#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_fork_scripts.py
"""Tests for fork/5d_prompt_reuse.py, fork/5e_mtp_acceptance.py and the 'tier' family of fork/gauntlet.py, against a small fake llama-server.

    python tests/test_fork_scripts.py

The fake server speaks just enough of the llama-server HTTP API (/tokenize, /completion, /v1/chat/completions) to drive both scripts to every verdict:
  mode "reuse"   the server keeps the cached prefix: the second request reads only the changed tail
  mode "full"    the server re-reads the whole prompt every time
  MTP mode       drafting counters in timings, and "speculative.n_max": 0 switches drafting off for a request
Exit 0 = all passed. Uses a temporary LOGS folder; starts no real server and touches nothing outside it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = Path(tempfile.mkdtemp(prefix="forktest_"))
os.environ["LOGS"] = str(TMP / "logs")
os.environ["CREATOR"] = str(TMP / "creator")
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "fork"))

fails = 0


def check(cond: bool, msg: str) -> None:
    global fails
    print(("ok   " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look the module up by name while the class body runs
    spec.loader.exec_module(mod)
    return mod


class Fake(BaseHTTPRequestHandler):
    mode = "reuse"            # reuse | full
    accept_rate = 0.8         # share of drafted tokens that are accepted
    drafting_off_works = True
    cached_text = ""

    def log_message(self, *a):  # silence
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path == "/tokenize":
            toks = body["content"].split()
            self._send({"tokens": list(range(len(toks)))})
        elif self.path == "/completion":
            words = body["prompt"].split()
            total = len(words)
            common = 0
            old = Fake.cached_text.split()
            for x, y in zip(old, words):
                if x != y:
                    break
                common += 1
            cache_n = common if Fake.mode == "reuse" else 0
            if Fake.mode == "reuse" and common == total:
                cache_n = total - 1
            Fake.cached_text = body["prompt"]
            self._send({"tokens_evaluated": total, "tokens_cached": cache_n, "content": "OK",
                        "timings": {"prompt_n": total - cache_n, "cache_n": cache_n, "prompt_ms": (total - cache_n) * 0.9}})
        elif self.path == "/v1/chat/completions":
            n = int(body.get("max_tokens", 100))
            off = body.get("speculative.n_max", None) == 0 and Fake.drafting_off_works
            t = {"predicted_n": n, "predicted_per_second": 40.0 if off else 55.0}
            if not off:
                drafted = n // 2 * 2
                t["draft_n"] = drafted
                t["draft_n_accepted"] = int(drafted * Fake.accept_rate)
            self._send({"choices": [{"message": {"content": "x"}}], "timings": t})
        else:
            self.send_response(404)
            self.end_headers()


def serve():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> int:
    d5 = load_module("d5", ROOT / "fork" / "5d_prompt_reuse.py")
    e5 = load_module("e5", ROOT / "fork" / "5e_mtp_acceptance.py")
    srv, url = serve()
    try:
        # ---- 5d
        Fake.mode = "reuse"
        out = TMP / "r_reuse.json"
        rc = d5.main(["--url", url, "--tokens", "600", "--out", str(out)])
        j = json.loads(out.read_text())
        check(rc == 0 and not j["full_reprocess_second_request"], f"5d, prefix reuse: verdict PASS (second request read {j['second_prompt_n']} of {j['second_total_tokens']} tokens)")
        check(j["second_total_tokens"] >= 500, f"5d builds a prompt of about the asked size ({j['second_total_tokens']} 'tokens' for 600)")
        check(j["second_prompt_n"] <= 10, "5d, prefix reuse: only the changed tail is read again")
        Fake.mode = "full"
        out = TMP / "r_full.json"
        rc = d5.main(["--url", url, "--tokens", "600", "--out", str(out)])
        j = json.loads(out.read_text())
        check(rc == 1 and j["full_reprocess_second_request"], "5d, full re-read: verdict HOLD and flag set")
        check(d5.main(["--url", "http://127.0.0.1:9", "--tokens", "100"]) == 2, "5d, no server: exit code 2")

        # ---- 5e
        Fake.accept_rate, Fake.drafting_off_works = 0.82, True
        out = TMP / "m_good.json"
        rc = e5.main(["run", "--url", url, "--gen", "64", "--out", str(out), "--baseline", str(TMP / "none.json")])
        j = json.loads(out.read_text())
        check(rc == 0 and abs(j["acceptance"] - 0.82) < 0.02, f"5e, good acceptance and faster: PASS (acceptance {j['acceptance']:.2f})")
        check(j.get("baseline_valid") is True and abs(j["baseline_gen_tps_median"] - 40.0) < 1e-6, "5e: paired baseline taken from the same server")
        Fake.accept_rate = 0.30
        rc = e5.main(["run", "--url", url, "--gen", "64", "--out", str(TMP / "m_bad.json"), "--baseline", str(TMP / "none.json")])
        check(rc == 1, "5e, low acceptance: FAIL")
        Fake.accept_rate = 0.60
        rc = e5.main(["run", "--url", url, "--gen", "64", "--out", str(TMP / "m_mid.json"), "--baseline", str(TMP / "none.json")])
        check(rc == 1, "5e, middling acceptance: HOLD (exit 1)")
        Fake.accept_rate, Fake.drafting_off_works = 0.82, False
        out = TMP / "m_nobase.json"
        rc = e5.main(["run", "--url", url, "--gen", "64", "--out", str(out), "--baseline", str(TMP / "none.json")])
        j = json.loads(out.read_text())
        check(j.get("baseline_valid") is False and "baseline_gen_tps_median" not in j, "5e: a server that ignores speculative.n_max=0 gives an invalid paired baseline")
        check(rc == 1, "5e: without any usable baseline the gate cannot say PASS")
        check(e5.main(["run", "--url", "http://127.0.0.1:9"]) == 2, "5e, no server: exit code 2")
    finally:
        srv.shutdown()

    # ---- gauntlet 'tier' family (parsing only; nothing is launched)
    g = load_module("gauntlet_t", ROOT / "fork" / "gauntlet.py")
    v = g.parse_spec("tier")
    check(v.family == "tier" and v.build == "tier" and v.flags[:2] == ["--n-cpu-moe", "40"] and "--fit" in v.flags, "gauntlet: tier defaults to all experts in RAM (n=40), fit off")
    check(v.env["PELIAS_TIER_BYPASS_N"] == "32" and v.env["PELIAS_TIER_MAX_MIB"] == "6144" and v.env["PELIAS_HOT_SET"].endswith("pelias_hot_set.txt"), "gauntlet: tier environment (hot set file, bypass 32, cap 6144 MiB)")
    v2 = g.parse_spec("tier:n=40,hot=/a/b/c.txt,bypass=16,cap=5632,mtp=2")
    check(v2.env["PELIAS_HOT_SET"] == "/a/b/c.txt" and v2.env["PELIAS_TIER_BYPASS_N"] == "16" and "--spec-type" in v2.flags and v2.mtp == 2, "gauntlet: tier with custom hot set, bypass and MTP")
    check(v.label != v2.label, f"gauntlet: distinct labels ({v.label} / {v2.label})")
    try:
        g.parse_spec("tier:mib=5120")
        check(False, "gauntlet: tier rejects a key that belongs to another family")
    except g.SpecError:
        check(True, "gauntlet: tier rejects a key that belongs to another family")
    try:
        g.parse_spec("tier:bypass=0")
        check(False, "gauntlet: tier rejects bypass=0")
    except g.SpecError:
        check(True, "gauntlet: tier rejects bypass=0")
    check(str(g.BUILDS["tier"].server).endswith("llama-pelias-tier/build-wsl/bin/llama-server"), "gauntlet: tier binary is the 09_build_pelias_tier.sh build")

    # ---- gauntlet 'mtpfile' key (separate draft GGUF)
    vm = g.parse_spec("prod:n=28,mtp=2,mtpfile=/x/y/mtp-aligned.gguf")
    i = vm.flags.index("-md") if "-md" in vm.flags else -1
    check(i >= 0 and vm.flags[i + 1] == "/x/y/mtp-aligned.gguf" and "--spec-type" in vm.flags and "-md" in vm.essential, "gauntlet: mtpfile adds -md PATH next to the MTP flags")
    check("mdmtpaligned" in vm.label and vm.label != g.parse_spec("prod:n=28,mtp=2").label, f"gauntlet: mtpfile shows in the label ({vm.label})")
    try:
        g.parse_spec("prod:n=28,mtpfile=/x/y.gguf")
        check(False, "gauntlet: mtpfile without mtp is refused")
    except g.SpecError:
        check(True, "gauntlet: mtpfile without mtp is refused")

    print("ALL PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
