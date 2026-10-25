#!/usr/bin/env python3
"""E0 per-token latency recorder (Pelias).  Standard library only.

Streams /completion from a llama-server and records the arrival time of every generated token on the CLOCK_MONOTONIC clock (time.perf_counter_ns: on Windows time.monotonic has 15.6 ms ticks, perf_counter does not)
(time.monotonic_ns, the same clock as std::chrono::steady_clock on Linux, so the rows can be lined up with the timeline files
of the timer patch).  Optionally starts the server itself from a gauntlet launch.json, so the run uses exactly the gauntlet's command.

  python3 token_latency.py --launch-json LAUNCH.json --tag graphs_on --out DIR [--env K=V ...] [--ctx 0,12000] [--gen 300]

Writes DIR/<tag>_tokens.tsv   columns: ctx_label, token_index, t_mono_ns, gap_ms   (token_index -1 = request sent)
       DIR/<tag>_summary.txt  p50 / p90 / p99 / max gap and mean tokens/s per context (first 8 tokens excluded from the rates)
Numbers are what the client sees over HTTP: they include the server's sampler, JSON and the local socket (cost not separately measured).
"""
import argparse, json, os, shlex, signal, subprocess, sys, time, urllib.request, urllib.error

PARA = ("The quick survey of the harbour district records that the old customs house, the grain exchange and the "
        "ferry landing share a single drainage channel, which floods whenever the spring tide meets heavy rain. ")


def http_json(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def n_tokens(url, text):
    return len(http_json(url + "/tokenize", {"content": text})["tokens"])


def make_prompt(url, target):
    if target <= 0:
        return "Write a long, detailed story about a lighthouse keeper and the storm that changed her life."
    per = n_tokens(url, PARA)
    reps = max(1, int(target / per))
    return PARA * reps + "\nSummarise the survey above in one long paragraph."


def stream_one(url, label, prompt, gen, rows):
    body = {"prompt": prompt, "n_predict": gen, "stream": True, "cache_prompt": False, "ignore_eos": True, "temperature": 0.0}
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t_sent = time.perf_counter_ns()
    rows.append((label, -1, t_sent, 0.0))
    idx, prev, final = 0, t_sent, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            now = time.perf_counter_ns()
            if ev.get("stop"):
                final = ev
                break
            if ev.get("content", "") != "":
                rows.append((label, idx, now, (now - prev) / 1e6))
                prev = now
                idx += 1
    return idx, final


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))] if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8012")
    ap.add_argument("--launch-json", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", default="0,12000")
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    proc = None
    if a.launch_json:
        if health(a.url):
            sys.exit("STOP: a server already answers on %s; stop it first (llama --stop)" % a.url)
        L = json.load(open(a.launch_json))
        cmd = L["command"]
        env = dict(os.environ)
        bindir = os.path.dirname(cmd[0])
        env["LD_LIBRARY_PATH"] = bindir + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env.update(L.get("env") or {})
        for kv in a.env:
            k, _, v = kv.partition("=")
            env[k] = v
        log = open(os.path.join(a.out, a.tag + "_server.log"), "w")
        log.write("# " + shlex.join(cmd) + "\n# extra env: " + " ".join(a.env) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t0 = time.time()
        while not health(a.url):
            if proc.poll() is not None:
                sys.exit("FAILED: server exited with code %s while loading; see %s_server.log" % (proc.returncode, a.tag))
            if time.time() - t0 > a.ready_timeout:
                sys.exit("FAILED: server not healthy after %.0f s" % a.ready_timeout)
            time.sleep(2)
        print("  server healthy after %.0f s" % (time.time() - t0))
    rows, lines = [], []
    try:
        print("  warm-up (32 tokens, not recorded)")
        stream_one(a.url, "warmup", make_prompt(a.url, 0), 32, [])
        for c in [int(x) for x in a.ctx.split(",") if x.strip() != ""]:
            label = "ctx%d" % c
            prompt = make_prompt(a.url, c)
            mark = len(rows)
            n, fin = stream_one(a.url, label, prompt, a.gen, rows)
            toks = [r for r in rows[mark:] if r[1] >= 0]
            gaps = [r[3] for r in toks[8:]]
            tps = (1000.0 / (sum(gaps) / len(gaps))) if gaps else float("nan")
            first_ms = (toks[0][2] - rows[mark][2]) / 1e6 if toks else float("nan")
            lines.append("%s: prompt tokens %s, generated %d, first token after %.0f ms, mean %.1f tok/s (tokens 9..), gap ms p50 %.2f p90 %.2f p99 %.2f max %.2f"
                         % (label, fin.get("tokens_evaluated", "?"), n, first_ms, tps, pct(gaps, 50), pct(gaps, 90), pct(gaps, 99), max(gaps) if gaps else float("nan")))
            print("  " + lines[-1])
    finally:
        with open(os.path.join(a.out, a.tag + "_tokens.tsv"), "w") as f:
            f.write("ctx\ttoken_index\tt_mono_ns\tgap_ms\n")
            for r in rows:
                f.write("%s\t%d\t%d\t%.3f\n" % r)
        with open(os.path.join(a.out, a.tag + "_summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)


if __name__ == "__main__":
    main()
