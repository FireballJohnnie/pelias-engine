#!/usr/bin/env python3
# File: <project>\moe_track\fork\5e_mtp_acceptance.py
"""Test 5e: how often does the server accept the MTP head's draft tokens, and is generation faster for it?

    python fork/5e_mtp_acceptance.py run [--url http://127.0.0.1:8012] [--label NAME] [--gen 320] [--greedy] [--no-paired]
                                         [--baseline SPEEDFILE.json] [--out FILE]

The server must have been started with  --spec-type draft-mtp --spec-draft-n-max N  (for example by:  python fork/gauntlet.py run prod:n=28,mtp=2).

For each of 8 fixed prompts (persona chat, agent/tool use, code, reasoning) it asks for --gen tokens through the chat endpoint, with the production sampler
(temperature 1.0, top_p 0.95, top_k 20; --greedy uses temperature 0) and thinking on, and reads the server's own counters from the response:
timings.draft_n (tokens drafted) and timings.draft_n_accepted (tokens kept).
  acceptance = accepted / drafted over all prompts
Paired baseline (default on): each prompt is also sent once with drafting switched off for that request ("speculative.n_max": 0), giving a speed baseline from
the SAME server and prompts. If the server still drafted in those requests the baseline is marked invalid and the speed file given by --baseline (default: the
N=28 production run of 2026-10-20) is used instead.

Result file (default logs/forks/<label>/mtp_5e.json) has the fields lib/gates.py mtp reads:
  acceptance, accepted, drafted, gen_tps_median, baseline_gen_tps_median (when valid), per_prompt[]
Gate: PASS = acceptance >= 70% and not slower than the baseline; FAIL = acceptance < 50%; HOLD in between.
Exit code: 0 PASS, 1 FAIL/HOLD, 2 the server cannot be used or reports no draft counters.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import gates  # noqa: E402
import paths  # noqa: E402
from speedlog import SpeedLogError, load_speedlog  # noqa: E402

DEFAULT_BASELINE = paths.SPEED_DIR / "20261020_032031_q4km-ncmoe28-ctx131k-ub2048.json"

PROMPTS = [
    ("persona", "Tell me, in your own voice, what you like most about helping someone plan a messy week, and how you decide what to do first."),
    ("persona", "I had a rough day and I do not want advice. Just talk with me for a bit about something warm and ordinary."),
    ("agent", "You can call tools. The user says: find the three largest files in my downloads folder and tell me which one I probably do not need. List the tool calls you would make, in order, with their arguments."),
    ("agent", "Plan, step by step, how you would back up a project folder, verify the copy with checksums, and report what changed since yesterday."),
    ("code", "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, then explain its running time."),
    ("code", "Write a C function that reverses a singly linked list in place, and a short test for it."),
    ("reasoning", "A train leaves at 9:40 and travels 195 km at 78 km/h, then rests 25 minutes and travels another 96 km at 64 km/h. When does it arrive? Show the working."),
    ("reasoning", "Explain why the sum of the first n odd numbers is n squared, using a picture-in-words argument and then an induction proof."),
]


def post(url: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ask(base: str, prompt: str, gen: int, greedy: bool, drafting: bool) -> Dict[str, float]:
    body = {
        "model": "stuart-creator", "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "stream": False, "cache_prompt": False,
        "temperature": 0.0 if greedy else 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    }
    if not drafting:
        body["speculative.n_max"] = 0
    r = post(base + "/v1/chat/completions", body)
    t = r.get("timings") or {}
    return {
        "predicted_n": float(t.get("predicted_n", 0)), "gen_tps": float(t.get("predicted_per_second", 0.0)),
        "draft_n": float(t.get("draft_n", 0)), "draft_n_accepted": float(t.get("draft_n_accepted", 0)),
    }


def run(base: str, gen: int, greedy: bool, paired: bool) -> dict:
    ask(base, "Say hello.", 16, greedy, True)                     # warm-up, not scored
    rows: List[dict] = []
    for dom, p in PROMPTS:
        m = ask(base, p, gen, greedy, True)
        row = {"domain": dom, "prompt": p[:60], "gen_tps": m["gen_tps"], "tokens": int(m["predicted_n"]), "drafted": int(m["draft_n"]), "accepted": int(m["draft_n_accepted"])}
        if paired:
            b = ask(base, p, gen, greedy, False)
            row.update({"baseline_gen_tps": b["gen_tps"], "baseline_drafted": int(b["draft_n"])})
        rows.append(row)
    drafted = sum(r["drafted"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    res = {
        "format": "pelias-5e", "url": base, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "greedy": greedy, "gen": gen,
        "drafted": drafted, "accepted": accepted, "acceptance": (accepted / drafted) if drafted else 0.0,
        "gen_tps_median": statistics.median(r["gen_tps"] for r in rows), "per_prompt": rows,
    }
    if paired:
        valid = all(r["baseline_drafted"] == 0 for r in rows) and all(r["baseline_gen_tps"] > 0 for r in rows)
        res["baseline_valid"] = valid
        if valid:
            res["baseline_gen_tps_median"] = statistics.median(r["baseline_gen_tps"] for r in rows)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run"])
    ap.add_argument("--url", default=f"http://{paths.HOST}:{paths.PORT}")
    ap.add_argument("--label", default="")
    ap.add_argument("--gen", type=int, default=320)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-paired", action="store_true")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    base = a.url.rstrip("/")
    try:
        res = run(base, a.gen, a.greedy, not a.no_paired)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"5e: the server at {base} could not be used: {e}", file=sys.stderr)
        return 2
    if res["drafted"] == 0:
        print("5e: the server reported no drafted tokens. Was it started with --spec-type draft-mtp --spec-draft-n-max N, "
              "and does this build put draft_n / draft_n_accepted in the response timings?", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else (paths.LOGS / "forks" / a.label / "mtp_5e.json" if a.label else paths.LOGS / "mtp_5e.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"5e: wrote {out}")
    if res.get("baseline_valid") is False:
        print("    (drafting could not be switched off per request on this server; the speed file below is the baseline instead)")
    base_log = None
    if "baseline_gen_tps_median" not in res:
        try:
            base_log = load_speedlog(a.baseline)
        except (SpeedLogError, OSError, ValueError) as e:
            print(f"    (no baseline speed available: {e})")
    for r in res["per_prompt"]:
        acc = (r["accepted"] / r["drafted"]) if r["drafted"] else 0.0
        print(f"    {r['domain']:9s} {r['gen_tps']:6.1f} tok/s  accepted {r['accepted']}/{r['drafted']} ({acc * 100:4.0f}%)  {r['prompt']}")
    return gates.gate_mtp(out, base_log)


if __name__ == "__main__":
    sys.exit(main())
