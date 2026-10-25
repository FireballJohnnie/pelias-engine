#!/usr/bin/env python3
# File: <project>\moe_track\fork\5d_prompt_reuse.py
"""Test 5d: does the running llama-server reuse the start of a long prompt when only the last line changes?

    python fork/5d_prompt_reuse.py [--url http://127.0.0.1:8012] [--label NAME] [--tokens 8000] [--out FILE]

Why it matters: the model is a hybrid (30 recurrent DeltaNet layers + 10 attention layers). A recurrent layer cannot simply "forget the last few tokens",
so a server can only reuse a cached prompt when it has a saved state (a context checkpoint) from before the point where the new prompt differs.
If it has none, every multi-turn request re-reads the WHOLE conversation. This test shows which one you have.

What it sends (all with cache_prompt on, one generated token, temperature 0):
  request 1   a long prompt of about --tokens tokens that starts with a random line (so nothing from an earlier run can be in the cache)
  request 2   the same prompt with only the LAST line changed          -> how much of it was read again?
  request 3   request 2 again, exactly                                  -> how much does an exact repeat cost?
Verdict (lib/gates.py reuse): full_reprocess_second_request is true when request 2 had to read >= 90% of its prompt again.

Result file (default logs/forks/<label>/prompt_reuse_5d.json, or logs/prompt_reuse_5d.json without --label) has the fields the gate reads:
  first_prompt_n, first_prompt_ms, second_total_tokens, second_prompt_n, second_cache_n, second_prompt_ms, full_reprocess_second_request, repeat_prompt_n
Exit code: 0 = prefix reuse works, 1 = the whole prompt is read again, 2 = the server could not be used.
It only talks to the server on the given URL (read-only apart from filling the server's own prompt cache).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import gates  # noqa: E402
import paths  # noqa: E402

WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nectar orchid pebble quartz ripple saffron timber umber velvet "
         "willow xenon yarrow zephyr anchor bridge candle dagger engine forest glacier hollow island jungle kettle lagoon marble nickel oasis prairie quiver river "
         "summit thicket upland valley wharf yonder zenith arch beacon cobalt dune estuary flint grove haven inlet jetty knoll ledge mesa nook outpost plateau").split()


def post(url: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def count_tokens(base: str, text: str) -> int:
    return len(post(base + "/tokenize", {"content": text}, 120.0)["tokens"])


def make_lines(n: int, rng: random.Random) -> List[str]:
    return [f"Entry {i}: " + " ".join(rng.choice(WORDS) for _ in range(14)) + "." for i in range(n)]


def build_prompt(base: str, target_tokens: int, seed: int) -> List[str]:
    """Lines of text whose total token count is just below target_tokens (the last line is the one that will be changed)."""
    rng = random.Random(seed)
    nonce = f"Session marker {rng.getrandbits(64):016x} {time.time_ns()}"
    sample = make_lines(40, rng)
    per_line = max(1.0, count_tokens(base, "\n".join(sample)) / len(sample))
    n = max(4, int(target_tokens / per_line))
    lines = [nonce] + make_lines(n, rng) + ["Final line: answer with the single word OK."]
    for _ in range(6):                       # settle on the count: shrink/grow by the measured error
        total = count_tokens(base, "\n".join(lines))
        if target_tokens * 0.97 <= total <= target_tokens:
            break
        n = max(4, n + int((target_tokens - total) / per_line))
        lines = [nonce] + make_lines(n, random.Random(seed + 1)) + ["Final line: answer with the single word OK."]
    return lines


def one_request(base: str, text: str) -> Dict[str, float]:
    r = post(base + "/completion", {"prompt": text, "n_predict": 1, "temperature": 0.0, "cache_prompt": True, "stream": False})
    t = r.get("timings") or {}
    prompt_n = int(t.get("prompt_n", r.get("tokens_evaluated", 0)))
    cache_n = int(t.get("cache_n", max(0, int(r.get("tokens_cached", 0)))))
    total = int(r.get("tokens_evaluated", prompt_n + cache_n)) if "tokens_evaluated" in r else prompt_n + cache_n
    if total < prompt_n + cache_n:
        total = prompt_n + cache_n
    return {"total": total, "prompt_n": prompt_n, "cache_n": cache_n, "prompt_ms": float(t.get("prompt_ms", 0.0))}


def run(base: str, tokens: int) -> dict:
    lines = build_prompt(base, tokens, seed=int(time.time()) & 0xFFFF)
    text1 = "\n".join(lines)
    text2 = "\n".join(lines[:-1] + ["Final line: now answer with the single word DONE."])
    a = one_request(base, text1)
    b = one_request(base, text2)
    c = one_request(base, text2)
    full = b["total"] > 0 and b["prompt_n"] >= 0.9 * b["total"]
    return {
        "format": "pelias-5d", "url": base, "target_tokens": tokens, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "first_total_tokens": a["total"], "first_prompt_n": a["prompt_n"], "first_prompt_ms": a["prompt_ms"],
        "second_total_tokens": b["total"], "second_prompt_n": b["prompt_n"], "second_cache_n": b["cache_n"], "second_prompt_ms": b["prompt_ms"],
        "full_reprocess_second_request": bool(full),
        "repeat_total_tokens": c["total"], "repeat_prompt_n": c["prompt_n"], "repeat_cache_n": c["cache_n"], "repeat_prompt_ms": c["prompt_ms"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://{paths.HOST}:{paths.PORT}")
    ap.add_argument("--label", default="")
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    base = a.url.rstrip("/")
    try:
        res = run(base, a.tokens)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"5d: the server at {base} could not be used: {e}", file=sys.stderr)
        return 2
    if a.out:
        out = Path(a.out)
    elif a.label:
        out = paths.LOGS / "forks" / a.label / "prompt_reuse_5d.json"
    else:
        out = paths.LOGS / "prompt_reuse_5d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"5d: wrote {out}")
    print(f"    exact repeat of request 2 read {res['repeat_prompt_n']} of {res['repeat_total_tokens']} prompt tokens again")
    return gates.gate_reuse(out)


if __name__ == "__main__":
    sys.exit(main())
