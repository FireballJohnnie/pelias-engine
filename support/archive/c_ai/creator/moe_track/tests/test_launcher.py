#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_launcher.py
"""Tests lib/llama_launcher.sh and 12_upgrade_llama_launcher.sh in a throw-away HOME with a fake llama-server (a small shell script that records its arguments and
environment). Nothing outside the temporary folder is touched; the real launcher is never installed by this test.

    python tests/test_launcher.py
"""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MT = HERE.parent
LAUNCHER = MT / "lib" / "llama_launcher.sh"
INSTALLER = MT / "12_upgrade_llama_launcher.sh"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"\n     {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def run(cmd, env, timeout=60):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def flags_of(cmdline: str):
    """The tokens of a 'cmd ...' line printed by --print."""
    assert cmdline.startswith("cmd ")
    return shlex.split(cmdline[4:])


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        home = td / "home"
        (home / ".local" / "bin").mkdir(parents=True)
        creator = td / "Creator"
        (creator / "gguf").mkdir(parents=True)
        (creator / "logs").mkdir()
        model = creator / "gguf" / "test-model-Q4_K_M.gguf"
        model.write_bytes(b"GGUF")
        rec = td / "record.txt"
        fake = td / "fake-llama-server"
        fake.write_text(f"#!/usr/bin/env bash\n{{ echo \"ARGS: $*\"; env | grep -E '^(PELIAS_|GGML_CUDA_DISABLE_FUSION)' | sort; }} > {rec}\nsleep 600\n")
        fake.chmod(0o755)
        tier_src = td / "tier"
        (tier_src / "build-wsl" / "bin").mkdir(parents=True)
        tier_bin = tier_src / "build-wsl" / "bin" / "llama-server"
        tier_bin.write_text(fake.read_text())
        tier_bin.chmod(0o755)
        hot = creator / "logs" / "pelias_hot_set.txt"
        hot.write_text("PELIAS_HOT_SET 1\n")
        port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "PELIAS_", "EXPERT_", "TIER_", "HOT_SET"))}
        env.update(HOME=str(home), LLAMA_CREATOR=str(creator), LLAMA_BIN=str(fake), LLAMA_PORT=str(port), LLAMA_PIDFILE=str(td / "pid"), TIER_SRC=str(tier_src))
        L = ["bash", str(LAUNCHER)]

        print("== syntax and help")
        check("launcher: bash -n", run(["bash", "-n", LAUNCHER], env).returncode == 0)
        check("installer: bash -n", run(["bash", "-n", INSTALLER], env).returncode == 0)
        r = run(L + ["--version"], env)
        check("--version", r.returncode == 0 and "launcher 2.0" in r.stdout, r.stdout + r.stderr)
        r = run(L + ["--help"], env)
        check("--help prints usage", r.returncode == 0 and "--pelias-tier" in r.stdout and "--mtp" in r.stdout)
        r = run(L, env)
        check("no arguments: usage and exit 2", r.returncode == 2)
        r = run(L + ["--bogus"], env)
        check("unknown option refused", r.returncode == 2 and "unknown option" in r.stderr)
        r = run(L + ["--serve", "--stuart-creator", "--profile", "huge", "--print"], env)
        check("unknown profile refused", r.returncode == 2 and "unknown profile" in r.stderr)
        r = run(L + ["--serve", "--stuart-creator", "--mtp-n", "99", "--print"], env)
        check("--mtp-n out of range refused", r.returncode == 2)
        r = run(L + ["--serve", "--stuart-creator", "--profile"], env)
        check("--profile without a value refused", r.returncode == 2)
        r = run(L + ["--serve"], env)
        check("--serve without --stuart-creator refused", r.returncode == 2)

        print("== the command (--print)")
        r = run(L + ["--serve", "--stuart-creator", "--print"], env)
        out = r.stdout
        cmd_line = [l for l in out.splitlines() if l.startswith("cmd ")]
        check("--print succeeds and prints one cmd line", r.returncode == 0 and len(cmd_line) == 1, out + r.stderr)
        f = flags_of(cmd_line[0]) if cmd_line else []

        def val(flag, toks=None):
            toks = f if toks is None else toks
            return toks[toks.index(flag) + 1] if flag in toks else None

        check("binary is the one from LLAMA_BIN", f[:1] == [str(fake)])
        check("model, alias, host, port", val("-m") == str(model) and val("--alias") == "stuart-creator" and val("--host") == "127.0.0.1" and val("--port") == str(port))
        check("long profile: -c 131072 -ub 2048", val("-c") == "131072" and val("-ub") == "2048")
        check("production flags present (jinja, -np 1, -ngl 99, fa on, q8_0 KV, no-context-shift, -n -1)",
              "--jinja" in f and val("-np") == "1" and val("-ngl") == "99" and val("--flash-attn") == "on" and val("--cache-type-k") == "q8_0" and val("--cache-type-v") == "q8_0"
              and "--no-context-shift" in f and val("-n") == "-1")
        check("sampling and reasoning flags", val("--temp") == "1.0" and val("--top-p") == "0.95" and val("--top-k") == "20" and val("--min-p") == "0.0"
              and val("--reasoning-format") == "deepseek" and val("--chat-template-kwargs") == '{"enable_thinking":true}')
        check("-t 7 --prio 2 --load-mode none", val("-t") == "7" and val("--prio") == "2" and val("--load-mode") == "none")
        check("default expert placement --n-cpu-moe 28, no --fit flag", val("--n-cpu-moe") == "28" and "--fit" not in f)
        check("no MTP flags and no PELIAS/GGML envs by default", "--spec-type" not in f and "env  " not in out)

        r = run(L + ["--serve", "--stuart-creator", "--profile", "fast", "--print"], env)
        f2 = flags_of([l for l in r.stdout.splitlines() if l.startswith("cmd ")][0])
        check("fast profile: -c 65536, ub unchanged", val("-c", f2) == "65536" and val("-ub", f2) == "2048")

        e2 = dict(env, LLAMA_CTX="32768", LLAMA_THREADS="8", LLAMA_PRIO="0", LLAMA_UB="1024", EXPERT_FLAGS="-ot .ffn_.*_exps.=CPU")
        r = run(L + ["--serve", "--stuart-creator", "--print"], e2)
        f3 = flags_of([l for l in r.stdout.splitlines() if l.startswith("cmd ")][0])
        check("environment overrides: context, threads, prio, ubatch, expert flags", val("-c", f3) == "32768" and val("-t", f3) == "8" and val("--prio", f3) == "0" and val("-ub", f3) == "1024"
              and val("-ot", f3) == ".ffn_.*_exps.=CPU" and "--n-cpu-moe" not in f3)
        e3 = dict(env, LLAMA_LOAD_MODE="")
        r = run(L + ["--serve", "--stuart-creator", "--print"], e3)
        check("LLAMA_LOAD_MODE='' drops --load-mode", "--load-mode" not in r.stdout)
        r = run(L + ["--serve", "--stuart-creator", "--print", "--", "--temp", "0.3", "--ctx-checkpoints", "8"], env)
        f4 = flags_of([l for l in r.stdout.splitlines() if l.startswith("cmd ")][0])
        check("flags after -- go last (so they win) and stay intact", f4[-4:] == ["--temp", "0.3", "--ctx-checkpoints", "8"])

        print("== --mtp")
        r = run(L + ["--serve", "--stuart-creator", "--mtp", "--print"], env)
        f5 = flags_of([l for l in r.stdout.splitlines() if l.startswith("cmd ")][0])
        check("--mtp: draft-mtp, n-max 2, fusion off", val("--spec-type", f5) == "draft-mtp" and val("--spec-draft-n-max", f5) == "2" and "env  GGML_CUDA_DISABLE_FUSION=1" in r.stdout and "-md" not in f5)
        r = run(L + ["--serve", "--stuart-creator", "--mtp-n", "3", "--mtp-gguf", "/x/draft.gguf", "--print"], env)
        f6 = flags_of([l for l in r.stdout.splitlines() if l.startswith("cmd ")][0])
        check("--mtp-n 3 --mtp-gguf FILE: implies --mtp, adds -md FILE", val("--spec-draft-n-max", f6) == "3" and val("-md", f6) == "/x/draft.gguf" and val("--spec-type", f6) == "draft-mtp")
        r = run(L + ["--serve", "--stuart-creator", "--mtp-gguf", str(td / "missing.gguf")], env)
        check("a missing --mtp-gguf file is refused before starting", r.returncode == 2 and "draft model not found" in r.stderr, r.stderr)

        print("== --pelias-tier")
        et = {k: v for k, v in env.items() if k != "LLAMA_BIN"}
        r = run(L + ["--serve", "--stuart-creator", "--pelias-tier", "--print"], et)
        ft = flags_of([l for l in r.stdout.splitlines() if l.startswith("cmd ")][0])
        check("tier: binary is the tier build", ft[0] == str(tier_bin), ft[:1])
        check("tier: PELIAS_HOT_SET / BYPASS_N / MAX_MIB are exported",
              f"env  PELIAS_HOT_SET={hot}" in r.stdout and "env  PELIAS_TIER_BYPASS_N=32" in r.stdout and "env  PELIAS_TIER_MAX_MIB=6144" in r.stdout, r.stdout)
        check("tier: default placement --n-cpu-moe 40 and --fit off", val("--n-cpu-moe", ft) == "40" and val("--fit", ft) == "off")
        r = run(L + ["--serve", "--stuart-creator", "--pelias-tier", "--mtp", "--print"], et)
        check("tier + mtp together", "GGML_CUDA_DISABLE_FUSION=1" in r.stdout and "PELIAS_HOT_SET" in r.stdout)
        r = run(L + ["--serve", "--stuart-creator", "--pelias-tier"], dict(et, TIER_SRC=str(td / "nope")))
        check("tier: missing build refused with the build command in the message", r.returncode == 2 and "09_build_pelias_tier.sh" in r.stderr, r.stderr)
        hot.rename(hot.with_suffix(".moved"))
        r = run(L + ["--serve", "--stuart-creator", "--pelias-tier"], et)
        check("tier: missing hot-set file refused (would silently be a plain control)", r.returncode == 2 and "hot-set file not found" in r.stderr, r.stderr)
        hot.with_suffix(".moved").rename(hot)

        print("== refusals")
        r = run(L + ["--serve", "--stuart-creator"], dict(env, LLAMA_BIN=str(td / "nobin")))
        check("missing binary refused", r.returncode == 2 and "server binary not found" in r.stderr)
        r = run(L + ["--serve", "--stuart-creator"], dict(env, LLAMA_MODEL=str(td / "nomodel.gguf")))
        check("missing model refused", r.returncode == 2 and "model file not found" in r.stderr)
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        bp = busy.getsockname()[1]
        r = run(L + ["--serve", "--stuart-creator"], dict(env, LLAMA_PORT=str(bp)))
        check("port already in use -> refused, exit 1, nothing started", r.returncode == 1 and "already in use" in r.stderr and not rec.exists(), r.stderr)
        fakebin = td / "fakebin"
        fakebin.mkdir()
        (fakebin / "ss").write_text(f"#!/usr/bin/env bash\necho 'LISTEN 0 1 127.0.0.1:{port} 0.0.0.0:*'\n")
        (fakebin / "ss").chmod(0o755)
        r = run(L + ["--serve", "--stuart-creator"], dict(env, PATH=f"{fakebin}:{env['PATH']}"))
        check("with an ss that reports the port as listening -> refused too", r.returncode == 1 and "already in use" in r.stderr and not rec.exists(), r.stderr)
        busy.close()

        print("== background start / stop")
        r = run(L + ["--serve", "--stuart-creator", "--bg", "--mtp"], env)
        check("--bg starts the fake server", r.returncode == 0 and "started in the background" in r.stdout, r.stdout + r.stderr)
        pidf = td / "pid"
        check("pid file written", pidf.exists() and pidf.read_text().strip().isdigit())
        time.sleep(0.5)
        txt = rec.read_text() if rec.exists() else ""
        check("the server received the flags and GGML_CUDA_DISABLE_FUSION=1", "--spec-type draft-mtp" in txt and "GGML_CUDA_DISABLE_FUSION=1" in txt and "-ub 2048" in txt, txt)
        logs = sorted((creator / "logs").glob("server_Q4_K_M_*.log"))
        check("one log file named server_Q4_K_M_<date_time>.log", len(logs) == 1)
        r = run(L + ["--serve", "--stuart-creator", "--bg"], env)
        check("a second --bg is refused while the first runs", r.returncode == 1 and "already recorded" in r.stderr, r.stdout + r.stderr)
        r = run(L + ["--status"], dict(env, PATH="/usr/bin:/bin"))
        check("--status runs", r.returncode == 0 and "llama-server" in r.stdout or "no llama-server" in r.stdout or "fake-llama-server" in r.stdout, r.stdout + r.stderr)
        pid = int(pidf.read_text().strip())
        r = run(L + ["--stop"], env, timeout=90)
        check("--stop stops it and removes the pid file", r.returncode == 0 and "stopped" in r.stdout and not pidf.exists(), r.stdout + r.stderr)
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        check("the process is gone", not alive)
        r = run(L + ["--stop"], env)
        check("--stop with nothing running is harmless", r.returncode == 0 and "no background server" in r.stdout)
        # a server that dies at once is reported, and no pid file is left
        bad = td / "bad-server"
        bad.write_text("#!/usr/bin/env bash\necho 'boom: cannot load model' >&2\nexit 3\n")
        bad.chmod(0o755)
        time.sleep(1.1)
        r = run(L + ["--serve", "--stuart-creator", "--bg"], dict(env, LLAMA_BIN=str(bad)))
        check("a server that exits at once is reported with its log tail; no pid file left", r.returncode == 1 and "boom" in r.stderr and not pidf.exists(), r.stdout + r.stderr)

        print("== installer")
        bindir = home / ".local" / "bin"
        ie = dict(env, BIN_DIR=str(bindir))
        old = bindir / "llama"
        old.write_text("#!/usr/bin/env bash\necho OLD LAUNCHER\n")
        old.chmod(0o755)
        before = old.read_text()
        r = run(["bash", INSTALLER], ie)
        check("show mode writes nothing", r.returncode == 0 and old.read_text() == before and not list(bindir.glob("llama.*")), r.stdout + r.stderr)
        r = run(["bash", INSTALLER, "install"], ie)
        baks = list(bindir.glob("llama.*.bak"))
        check("install: old launcher backed up byte for byte", r.returncode == 0 and len(baks) == 1 and baks[0].read_text() == before, r.stdout + r.stderr)
        check("install: new launcher in place, executable, identical to lib/llama_launcher.sh", old.read_bytes() == LAUNCHER.read_bytes() and os.access(old, os.X_OK))
        r = run([old, "--version"], env)
        check("installed launcher runs", r.returncode == 0 and "launcher 2.0" in r.stdout)
        r = run(["bash", INSTALLER, "install"], ie)
        check("install again: 'already up to date', no second backup", r.returncode == 0 and "up to date" in r.stdout and len(list(bindir.glob("llama.*.bak"))) == 1)
        time.sleep(1.1)
        r = run(["bash", INSTALLER, "restore"], ie)
        check("restore: old launcher back, replaced one kept", r.returncode == 0 and old.read_text() == before and len(list(bindir.glob("llama.*.replaced"))) == 1, r.stdout + r.stderr)
        check("no stray temp files left in the folder", not list(bindir.glob("llama.new.*")) and not list(bindir.glob("llama.restore.*")))
        fresh = td / "fresh"
        r = run(["bash", INSTALLER, "install"], dict(env, BIN_DIR=str(fresh)))
        check("install into an empty folder just creates it (no backup needed)", r.returncode == 0 and (fresh / "llama").exists() and not list(fresh.glob("*.bak")), r.stdout + r.stderr)

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
