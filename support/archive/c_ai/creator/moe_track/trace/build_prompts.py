#!/usr/bin/env python3
# File: <project>\moe_track\trace\build_prompts.py
"""Builds the prompt files the trace collector feeds to the model: four domains, ChatML text, thinking switched on.

    python trace/build_prompts.py [--dataset dataset/<file>] [--out-dir logs/moe_trace/prompts]
                                  [--quota persona=60,agent=40,code=100,reasoning=100] [--seed 1] [--fake-tokenize 64]

domains
  persona    dataset rows without tools: the system + user turns before the last assistant turn
  agent      dataset rows that carry a tool list: same, with the tools rendered into the system prompt
  code       built-in coding requests (trace/seed_prompts.py)
  reasoning  built-in maths / logic / explanation requests (trace/seed_prompts.py)

Each output line is  {"id","domain","split","text"}.  "split" (fit or heldout, 70/30 by hash of the id) is what the simulator uses to keep a
held-out set that the cache profile never saw. Prompts end with the assistant header and an open <think> tag, the way the server's template does
with thinking on. The exact template of the served model is not applied here (it needs a Jinja engine); routing statistics do not depend on
small template differences, and the domains' own content is what matters.

--fake-tokenize N writes {"tokens":[...]} (words hashed into 0..N-1) instead of text. Only for tests with a model that has no tokenizer.
Reads the dataset; writes only into --out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import paths  # noqa: E402
import seed_prompts  # noqa: E402
from pltrace import split_of  # noqa: E402

TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)


def _content(m: dict) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, list):   # OpenAI-style parts
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c)


def render_chatml(messages: List[dict], tools: Optional[list] = None, system_default: str = "") -> str:
    msgs = list(messages)
    system = system_default
    if msgs and msgs[0].get("role") == "system":
        system = _content(msgs[0])
        msgs = msgs[1:]
    if tools:
        tl = "\n".join(json.dumps(t if "function" in t or "type" in t else {"type": "function", "function": t}, ensure_ascii=False) for t in tools)
        system = (system + "\n\n" if system else "") + TOOLS_HEADER.format(tools=tl)
    out = []
    if system:
        out.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for m in msgs:
        role = m.get("role")
        if role == "user":
            out.append(f"<|im_start|>user\n{_content(m)}<|im_end|>\n")
        elif role == "assistant":
            body = _content(m)   # earlier reasoning is not shown to the model in later turns
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                body += ("\n" if body else "") + "<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
            out.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        elif role == "tool":
            out.append(f"<|im_start|>user\n<tool_response>\n{_content(m)}\n</tool_response><|im_end|>\n")
        elif role == "system":
            out.append(f"<|im_start|>system\n{_content(m)}<|im_end|>\n")
    out.append("<|im_start|>assistant\n<think>\n")
    return "".join(out)


def prompt_part(messages: List[dict]) -> List[dict]:
    """Everything before the final assistant turn (the target the dataset row would have the model write)."""
    msgs = list(messages)
    while msgs and msgs[-1].get("role") == "assistant":
        msgs.pop()
    return msgs


def fake_tokens(text: str, vocab: int) -> List[int]:
    return [int(hashlib.md5(w.encode("utf-8")).hexdigest()[:6], 16) % vocab for w in re.findall(r"\w+|[^\w\s]", text)] or [1]


def load_rows(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise SystemExit(f"{path} line {i}: {e}")
    return rows


def build(dataset: Optional[Path], quota: Dict[str, int], seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    out: Dict[str, List[dict]] = {d: [] for d in quota}
    if dataset is not None and dataset.exists():
        rows = [r for r in load_rows(dataset) if isinstance(r.get("messages"), list) and prompt_part(r["messages"])]
        rows_tools = [r for r in rows if r.get("tools")]
        rows_plain = [r for r in rows if not r.get("tools")]
        rng.shuffle(rows_tools)
        rng.shuffle(rows_plain)
        for dom, pool in (("persona", rows_plain), ("agent", rows_tools)):
            for i, r in enumerate(pool[: quota.get(dom, 0)]):
                text = render_chatml(prompt_part(r["messages"]), tools=r.get("tools") if dom == "agent" else None)
                out[dom].append({"id": f"{dom}-{i:04d}", "domain": dom, "text": text})
    elif quota.get("persona", 0) or quota.get("agent", 0):
        print(f"note: dataset {dataset} not found; persona and agent prompts skipped", file=sys.stderr)
    for dom, seeds, system in (("code", seed_prompts.CODE, seed_prompts.CODE_SYSTEM), ("reasoning", seed_prompts.REASONING, seed_prompts.REASONING_SYSTEM)):
        n = quota.get(dom, 0)
        for i in range(n):
            user = seeds[i % len(seeds)]
            out[dom].append({"id": f"{dom}-{i:04d}", "domain": dom,
                             "text": render_chatml([{"role": "system", "content": system}, {"role": "user", "content": user}])})
    for dom in out:
        for rec in out[dom]:
            rec["split"] = split_of(rec["id"])
    return out


def parse_quota(s: str) -> Dict[str, int]:
    q = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        if k.strip() not in ("persona", "agent", "code", "reasoning") or not v.strip().isdigit():
            raise SystemExit(f"--quota: bad entry '{part}' (domains: persona, agent, code, reasoning)")
        q[k.strip()] = int(v)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(paths.CREATOR / "dataset" / "<dataset-file>"))
    ap.add_argument("--out-dir", default=str(paths.MOE_TRACE_DIR / "prompts"))
    ap.add_argument("--quota", default="persona=60,agent=40,code=38,reasoning=31", help="prompts per domain (code and reasoning repeat their seed list if the quota is larger)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fake-tokenize", type=int, default=0, help="emit token ids instead of text (tests only)")
    a = ap.parse_args(argv)

    quota = parse_quota(a.quota)
    built = build(Path(a.dataset), quota, a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"seed": a.seed, "dataset": a.dataset, "domains": {}}
    for dom, recs in built.items():
        if not recs:
            continue
        p = out_dir / f"prompts_{dom}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                rec = dict(r)
                if a.fake_tokenize:
                    rec = {k: v for k, v in rec.items() if k != "text"}
                    rec["tokens"] = fake_tokens(r["text"], a.fake_tokenize)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        index["domains"][dom] = {"file": str(p), "prompts": len(recs), "heldout": sum(r["split"] == "heldout" for r in recs),
                                 "mean_chars": round(sum(len(r["text"]) for r in recs) / len(recs))}
        print(f"{dom:10s} {len(recs):4d} prompts ({index['domains'][dom]['heldout']} held out), mean {index['domains'][dom]['mean_chars']} characters -> {p}")
    (out_dir / "prompts_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    if not index["domains"]:
        print("no prompts were built", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
