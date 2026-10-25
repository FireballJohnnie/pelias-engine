#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_verify_g0.py
"""Offline test of microbench/06_verify_gate_g0.py with made-up speed files. Touches only a temp folder."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import floor_model  # noqa: E402
from test_gates import make_speed  # noqa: E402

SCRIPT = HERE.parent / "microbench" / "06_verify_gate_g0.py"


def run(args):
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    lb = floor_model.layer_bytes_from_sizes(None)
    truth = floor_model.FloorModel(fixed_ms=7.7, b_cpu_gbs=25.5, b_gpu_gbs=504.0, layer_bytes=lb)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        f40 = make_speed(d / "n40.json", "n40", truth.tps_n_cpu_moe(40), 30, 900)
        f28 = make_speed(d / "n28.json", "n28", truth.tps_n_cpu_moe(28), 38, 1190)
        f34_ok = make_speed(d / "n34ok.json", "n34", truth.tps_n_cpu_moe(34) * 1.03, 33, 1000)
        f34_bad = make_speed(d / "n34bad.json", "n34", truth.tps_n_cpu_moe(34) * 1.25, 33, 1000)
        out = d / "g0.json"
        base = ["--sizes", str(d / "none.json"), "--ram", str(d / "none"), "--handoff", str(d / "none"), "--dual", str(d / "none"), "--out", str(out)]

        rc, txt = run(["--point", f"40={f40}", "--point", f"28={f28}", "--point", f"34={f34_ok}", *base])
        check(rc == 0 and "PASS" in txt, "model-consistent third point passes G0")
        data = json.loads(out.read_text())
        check(data["pass"] is True and data["fitted_on"] == [40, 28], "result json says pass, fitted on 40 and 28")
        check(abs(data["model"]["b_cpu_gbs"] - 25.5) < 0.5, "fit recovers the CPU bandwidth")
        check("forecast_n_cpu_moe" in data and "forecast_hit_rate_cpu_miss" in data, "forecast tables are written")

        rc, txt = run(["--point", f"40={f40}", "--point", f"28={f28}", "--point", f"34={f34_bad}", *base])
        check(rc == 1 and data is not None, "25% off third point fails G0 with exit 1")
        check(json.loads(out.read_text())["pass"] is False, "result json says fail")

        rc, txt = run(["--point", f"40={f40}", "--point", f"28={f28}", *base])
        check(rc == 2, "two points only is an input error (exit 2)")

        # extras present: ram + handoff + dual
        (d / "ram.json").write_text(json.dumps({"best_expert_gb_s": 40.0}))
        (d / "ho.json").write_text(json.dumps({"handoff_us_per_layer": 30.0, "handoff_ms_per_token_40_layers": 1.2}))
        (d / "du.json").write_text(json.dumps({"combined_vs_cpu_ratio": 1.4, "concurrent_cpu_plus_h2d": {"cpu_keeps_fraction": 0.9}}))
        rc, txt = run(["--point", f"40={f40}", "--point", f"28={f28}", "--point", f"34={f34_ok}", "--sizes", str(d / "none.json"),
                       "--ram", str(d / "ram.json"), "--handoff", str(d / "ho.json"), "--dual", str(d / "du.json"), "--out", str(out)])
        check(rc == 0 and "64%" in txt and "1.40x" in txt, "RAM, handoff and dual-bus lines appear")

        rc, txt = run(["--point", f"40={d/'missing.json'}", "--point", f"28={f28}", "--point", f"34={f34_ok}", *base])
        check(rc == 2, "missing speed file is an input error")
    print("FAILURES: %d" % fails if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
