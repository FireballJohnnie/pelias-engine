#!/usr/bin/env python3
# File: <project>\moe_track\fork\gauntlet.py   (WSL: <project>/moe_track/fork/gauntlet.py)
"""Phase 5 gauntlet: start each candidate llama-server (mainline expert cache, the Schwerz fork, the Parmesan fork, or the production build with
other flags), run the same speed test on each, and score the result with the project gates. One variant at a time, one thing changed per variant.

    python fork/gauntlet.py plan                                 the variants worth running and what each costs; runs nothing
    python fork/gauntlet.py builds                               which builds exist
    python fork/gauntlet.py flags SPEC                           the exact command a variant would use, and how it differs from production
    python fork/gauntlet.py run SPEC [SPEC ...] [options]        start, test, stop, score each variant (this STOPS the llama-server on port 8012)
    python fork/gauntlet.py report                               table of every variant measured so far
    python fork/gauntlet.py build-info KEY                       (used by 05_phase5_fork_gauntlet.sh)

SPEC is  family[:key=value,key=value]   (separate keys with commas, no spaces)
    prod      the production build, direct launch.  n=28 (layers whose experts stay on the CPU)   -> the control run, same launcher as the forks
    mcache    mainline master, built-in expert cache.  n=40  mib=5120      (experts on CPU, hot ones cached in VRAM; no fork needed)
    schwerz   GenerelSchwerz fork, hybrid CPU/GPU experts.  mib=5120 | slots=N   profile=none|auto|FILE   adapt=off|occurrence|occurrence-sync
              miss=F (0..1, GPU share of cache misses; omit = the fork's own tuning)   alloc=auto|uniform   workspace=separate|borrow
              overlap=on|off (decode overlap flags, uses backend sampling; default off)   phase=on|off (phase-aware workspace)
    parmesan  ParmesanParty fork, hot-expert bank.  n=40  k=78  rebalance=40  bypass=64   (UNVERIFIED with this GGUF; see README)
    tier      Phase 6 build (production commit + patches/pelias_tier_v1.patch): all experts stay in RAM, the hot ones also sit in a VRAM bank that
              the graph uses for batches below the bypass size.   n=40  hot=FILE (default logs/pelias_hot_set.txt)  bypass=32  cap=6144 (MiB, safety limit)
              Build it first: bash patches/09_build_pelias_tier.sh build      Make the hot set first: python sim/export_hot_set.py
  keys valid for every family:  mtp=N (adds --spec-type draft-mtp --spec-draft-n-max N)   ctxcp=N (--ctx-checkpoints N)   reuse=1 (also run test 5d)
                                mtpfile=PATH (with mtp=N: use this separate draft GGUF, e.g. the one made by mtp_align/11_merge_blk40_and_export.py, as -md PATH;
                                without it the server uses the MTP head inside the main GGUF; no commas in the path)
Examples
    python fork/gauntlet.py run prod:n=28 mcache:n=40,mib=5120 mcache:n=40,mib=4096
    python fork/gauntlet.py run schwerz:mib=5120,profile=auto --sweep miss=0,0.17,0.5,1
    python fork/gauntlet.py run prod:n=28,mtp=3 --tests mtp

Options for run
    --baseline FILE     speed result the gates compare against (default: the N=28 production run of 2026-10-20)
    --depths 0,4000,12000   --repeats N   --gen N   (default: copied from the baseline file so the conditions match)
    --extra "flags"     appended to every launch (shows up in the parity diff)
    --tests reuse,mtp   also run test 5d (prompt reuse) and/or 5e (MTP acceptance) on each running variant
    --ready-timeout S   how long to wait for the model to load (default 900)
    --drop-unsupported  if the binary does not know a parity flag from the production command, drop it instead of stopping
    --restore           at the end start the usual production server again (llama --serve --stuart-creator --bg)
    --dry-run           print the commands and parity diffs, start nothing
    --keep-going        continue with the next variant after a failure (default: stop)
Writes logs/forks/<label>/{launch.json, server.log, check.txt, speed.txt, verdict.json}; speed files go to logs/speed_server as usual.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import gates  # noqa: E402
import launch_parity as LP  # noqa: E402
import paths  # noqa: E402
from speedlog import SpeedLogError, load_speedlog  # noqa: E402

FORKS_DIR = Path(os.environ.get("MOE_FORKS") or (Path.home() / "llama-forks"))
TIER_SRC = Path(os.environ.get("TIER_SRC") or (Path.home() / "llama-pelias-tier"))      # made by patches/09_build_pelias_tier.sh
DEFAULT_HOT_SET = paths.CREATOR / "logs" / "pelias_hot_set.txt"
OUT_DIR = paths.LOGS / "forks"
STORE = OUT_DIR / "prod_cmdline.json"
PY = os.environ.get("PY") or sys.executable
SPEED_SCRIPT = os.environ.get("SPEED_SCRIPT") or str(paths.CREATOR / "stuart_server_speed.py")
PROBE_SCRIPT = os.environ.get("PROBE_SCRIPT") or str(paths.CREATOR / "stuart_server_probe.py")
DEFAULT_BASELINE = paths.SPEED_DIR / "20261020_032031_q4km-ncmoe28-ctx131k-ub2048.json"
N_LAYERS = 40


@dataclass(frozen=True)
class Build:
    key: str
    repo: str
    commit: str
    src: Path
    note: str = ""

    @property
    def bin_dir(self) -> Path:
        return self.src / "build-wsl" / "bin"

    @property
    def server(self) -> Path:
        return self.bin_dir / "llama-server"


BUILDS: Dict[str, Build] = {
    "prod": Build("prod", "", paths.LLAMA_COMMIT, paths.LLAMA_PROD, "the production build (b11399)"),
    "master": Build("master", "https://github.com/ggml-org/llama.cpp", "b38960f52079e41f91863dd0905ec13432966b39",
                    FORKS_DIR / "llama.cpp-master-b38960f", "mainline master of 2026-10-20 (has --moe-cache-mib)"),
    "schwerz": Build("schwerz", "https://github.com/GenerelSchwerz/llama.cpp", "f3f31a788f05d99f9404afe90f9de4baaa2f6e6f",
                     FORKS_DIR / "llama.cpp-schwerz-f3f31a7", "GenerelSchwerz fork of 2026-10-11 (hybrid CPU/GPU experts, profiles)"),
    "parmesan": Build("parmesan", "https://github.com/ParmesanParty/llama.cpp", "39979e84e65846d1aa083b4f8b773d4f1cd3e9e8",
                      FORKS_DIR / "llama.cpp-parmesan-39979e8", "ParmesanParty fork, tag parmesan-b9106 (May 2026 base)"),
    "tier": Build("tier", "", paths.LLAMA_COMMIT, TIER_SRC, "Phase 6: production commit + patches/pelias_tier_v1.patch (VRAM hot bank used by the graph)"),
}

FAMILY_BUILD = {"prod": "prod", "mcache": "master", "schwerz": "schwerz", "parmesan": "parmesan", "tier": "tier"}
FAMILY_KEYS = {
    "prod": {"n"},
    "mcache": {"n", "mib"},
    "schwerz": {"mib", "slots", "profile", "adapt", "miss", "alloc", "workspace", "overlap", "phase"},
    "parmesan": {"n", "k", "rebalance", "bypass"},
    "tier": {"n", "hot", "bypass", "cap"},
}
COMMON_KEYS = {"mtp", "ctxcp", "reuse", "mtpfile"}
DEFAULT_PROFILE = paths.MOE_TRACE_DIR / "schwerz" / "calibration.gguf"


class SpecError(ValueError):
    pass


@dataclass
class Variant:
    spec: str
    family: str
    params: Dict[str, str]
    label: str
    build: str
    flags: List[str]                    # placement + variant flags appended to the production command
    essential: List[str]                # flags the variant cannot work without (never dropped)
    env: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    reuse: bool = False
    mtp: int = 0
    supports_load_mode: bool = True


def _num(params: Dict[str, str], key: str, default, cast=int, lo=None, hi=None):
    raw = params.get(key)
    if raw is None:
        return default
    try:
        v = cast(raw)
    except ValueError:
        raise SpecError(f"{key}={raw!r} is not a number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise SpecError(f"{key}={raw} is out of range ({lo}..{hi})")
    return v


def _tag(x) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "", str(x))


def parse_spec(spec: str, profile_default: Optional[Path] = None) -> Variant:
    family, _, rest = spec.partition(":")
    family = family.strip()
    if family not in FAMILY_BUILD:
        raise SpecError(f"unknown family {family!r}; choose from: {', '.join(FAMILY_BUILD)}")
    params: Dict[str, str] = {}
    for part in [p for p in rest.split(",") if p.strip()]:
        k, eq, v = part.partition("=")
        if not eq or not k.strip():
            raise SpecError(f"bad parameter {part!r} in {spec!r}; use key=value")
        params[k.strip()] = v.strip()
    bad = set(params) - FAMILY_KEYS[family] - COMMON_KEYS
    if bad:
        raise SpecError(f"{family}: unknown key(s) {', '.join(sorted(bad))}; valid: {', '.join(sorted(FAMILY_KEYS[family] | COMMON_KEYS))}")

    flags: List[str] = []
    essential: List[str] = []
    env: Dict[str, str] = {}
    notes: List[str] = []
    parts = [family]
    supports_load_mode = True

    if family == "prod":
        n = _num(params, "n", 28, int, 0, N_LAYERS)
        flags += ["--n-cpu-moe", str(n)] if n > 0 else []
        parts.append(f"n{n}")
    elif family == "mcache":
        n = _num(params, "n", 40, int, 0, N_LAYERS)
        mib = _num(params, "mib", 5120, int, 256, 11000)
        flags += (["--n-cpu-moe", str(n)] if n > 0 else []) + ["--moe-cache-mib", str(mib), "--fit", "off"]
        essential += ["--moe-cache-mib"]
        parts += [f"n{n}", f"mib{mib}"]
        notes.append("mainline cache: experts of the first n layers stay in RAM, the hottest are copied into a VRAM cache; it serves batches of up to 32 tokens")
    elif family == "schwerz":
        if "mib" in params and "slots" in params:
            raise SpecError("schwerz: give mib or slots, not both (the fork rejects both)")
        if "slots" in params:
            slots = _num(params, "slots", 64, int, 1, 4096)
            flags += ["--moe-expert-cache-size", str(slots)]
            essential += ["--moe-expert-cache-size"]
            parts.append(f"slots{slots}")
        else:
            mib = _num(params, "mib", 5120, int, 256, 11000)
            flags += ["--moe-expert-cache-mib", str(mib)]
            essential += ["--moe-expert-cache-mib"]
            parts.append(f"mib{mib}")
        flags = ["--moe-hybrid", "on"] + flags + ["--moe-miss-keep-ranks", "0", "--fit", "off"]
        essential += ["--moe-hybrid", "--moe-miss-keep-ranks"]
        prof = params.get("profile", "none")
        adapt = params.get("adapt", "off")
        if adapt not in ("off", "occurrence", "occurrence-sync"):
            raise SpecError("schwerz: adapt must be off, occurrence or occurrence-sync")
        profile_path: Optional[Path] = None
        if prof == "auto":
            profile_path = profile_default or DEFAULT_PROFILE
        elif prof != "none":
            profile_path = Path(prof)
        if adapt != "off" and profile_path is None:
            raise SpecError("schwerz: adapt needs a profile (profile=auto or profile=FILE)")
        if profile_path is not None:
            flags += ["--moe-expert-profile", str(profile_path)]
            essential += ["--moe-expert-profile"]
            parts.append("prof")
            if adapt != "off":
                flags += ["--moe-profile-adapt", adapt]
                essential += ["--moe-profile-adapt"]
                parts.append("adapt" + {"occurrence": "occ", "occurrence-sync": "sync"}[adapt])
        if "miss" in params:
            miss = _num(params, "miss", 0.17, float, 0.0, 1.0)
            flags += ["--moe-gpu-miss-fraction", f"{miss:g}"]
            essential += ["--moe-gpu-miss-fraction"]
            parts.append(f"miss{miss:g}")
            notes.append("an explicit miss fraction switches the fork's own miss-split tuning off")
        alloc = params.get("alloc")
        if alloc:
            if alloc not in ("auto", "uniform"):
                raise SpecError("schwerz: alloc must be auto or uniform")
            flags += ["--moe-cache-allocation", alloc]
            parts.append("alloc" + alloc)
        ws = params.get("workspace")
        if ws:
            if ws not in ("separate", "borrow"):
                raise SpecError("schwerz: workspace must be separate or borrow")
            flags += ["--moe-prefill-workspace", ws]
            parts.append("ws" + ws)
        if params.get("overlap", "off") == "on":
            flags += ["--decode-overlap", "--decode-boundary-overlap"]
            parts.append("overlap")
            notes.append("decode overlap is documented for backend sampling; the sampler path differs from production (check the persona probe)")
        if params.get("phase", "off") == "on":
            flags += ["--phase-aware-workspace"]
            parts.append("phase")
        notes.append("hybrid execution: cache hits run on the GPU, misses are split between CPU and GPU; --moe-miss-keep-ranks 0 keeps decode exact")
    elif family == "parmesan":
        n = _num(params, "n", 40, int, 0, N_LAYERS)
        k = _num(params, "k", 78, int, 1, 256)
        rb = _num(params, "rebalance", 40, int, 1, 100000)
        bypass = _num(params, "bypass", 64, int, 0, 100000)
        flags += (["--n-cpu-moe", str(n)] if n > 0 else []) + ["--moe-hot-k", str(k), "--moe-hot-rebalance-interval", str(rb), "--fit", "off"]
        essential += ["--moe-hot-k", "--moe-hot-rebalance-interval"]
        env["LLAMA_MOE_HOT_PP_BYPASS_N_TOKENS"] = str(bypass)
        parts += [f"n{n}", f"k{k}", f"rb{rb}", f"by{bypass}"]
        supports_load_mode = False
        notes.append("UNVERIFIED: May 2026 code base; whether it loads this GGUF and its Qwen3.6 architecture is found out by the first start")
    elif family == "tier":
        n = _num(params, "n", 40, int, 0, N_LAYERS)
        bypass = _num(params, "bypass", 32, int, 1, 100000)
        cap = _num(params, "cap", 6144, int, 256, 11000)
        hot = Path(params.get("hot") or DEFAULT_HOT_SET)
        flags += (["--n-cpu-moe", str(n)] if n > 0 else []) + ["--fit", "off"]
        env["PELIAS_HOT_SET"] = str(hot)
        env["PELIAS_TIER_BYPASS_N"] = str(bypass)
        env["PELIAS_TIER_MAX_MIB"] = str(cap)
        env["PELIAS_TIER_VERBOSE"] = "1"
        parts += [f"n{n}", "hot" + (_tag(hot.stem)[-12:] if params.get("hot") else ""), f"by{bypass}"]
        notes.append("Phase 6: the expert files stay in RAM exactly as in the control; the bank is an extra copy of the hot experts in VRAM (cap %d MiB). "
                     "Batches of fewer than %d tokens use bank + RAM, larger ones take the stock path" % (cap, bypass))
        notes.append("the log line 'pelias-tier: hot bank active for N layers' proves the bank exists; 'running without a hot bank' means it was refused and the run is a plain control")
    mtp = _num(params, "mtp", 0, int, 0, 16)
    mtpfile = params.get("mtpfile", "")
    if mtpfile and not mtp:
        raise SpecError("mtpfile=PATH needs mtp=N as well")
    if mtp:
        flags += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp)]
        essential += ["--spec-type", "--spec-draft-n-max"]
        parts.append(f"mtp{mtp}")
        if mtpfile:
            flags += ["-md", mtpfile]
            essential += ["-md"]
            parts.append("md" + _tag(Path(mtpfile).stem)[-16:])
            notes.append(f"separate draft GGUF: {mtpfile}")
    ctxcp = _num(params, "ctxcp", 0, int, 0, 512)
    if ctxcp:
        flags += ["--ctx-checkpoints", str(ctxcp)]
        essential += ["--ctx-checkpoints"]
        parts.append(f"cp{ctxcp}")
    reuse = bool(_num(params, "reuse", 0, int, 0, 1))
    label = "-".join(_tag(p) for p in parts)
    return Variant(spec=spec, family=family, params=params, label=label, build=FAMILY_BUILD[family], flags=flags, essential=essential, env=env,
                   notes=notes, reuse=reuse, mtp=mtp, supports_load_mode=supports_load_mode)


def expand_sweep(specs: Sequence[str], sweep: Optional[str]) -> List[str]:
    if not sweep:
        return list(specs)
    k, eq, vals = sweep.partition("=")
    if not eq or not vals:
        raise SpecError("--sweep needs key=v1,v2,...")
    out: List[str] = []
    for s in specs:
        fam, colon, rest = s.partition(":")
        for v in vals.split(","):
            out.append(f"{fam}:{rest + ',' if rest else ''}{k}={v}")
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# production reference command
# ----------------------------------------------------------------------------------------------------------------------------------
def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def reference_command(proc_root: str = "/proc", store: Path = STORE) -> Tuple[List[str], str]:
    """The production llama-server command: the running one (saved for later), the saved copy, or lib/paths.py as the last resort."""
    live = LP.live_server_tokens(proc_root)
    if live and not _under(Path(live[0]), FORKS_DIR) and not _under(Path(live[0]), TIER_SRC):
        try:
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text(json.dumps({"argv": live, "captured": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
        except OSError:
            pass
        return live, "the running production llama-server (command saved for later runs)"
    if store.exists():
        try:
            rec = json.loads(store.read_text(encoding="utf-8"))
            return list(rec["argv"]), f"the production command saved {rec.get('captured', '?')}"
        except (ValueError, KeyError, OSError):
            pass
    return LP.production_tokens(), "lib/paths.py (no running production server was seen; threads and mmap mode of your launcher are NOT included)"


def full_command(v: Variant, extra: Sequence[str], proc_root: str = "/proc", store: Path = STORE) -> Tuple[List[str], List[str], str]:
    """-> (candidate argv, reference argv, where the reference came from)."""
    ref, src = reference_command(proc_root, store)
    server = str(BUILDS[v.build].server)
    cmd = LP.build_command(ref, server, list(v.flags) + list(extra), model=str(paths.GGUF), port=paths.PORT, supports_load_mode=v.supports_load_mode)
    return cmd, ref, src


# ----------------------------------------------------------------------------------------------------------------------------------
# capability check: does this binary know every flag?
# ----------------------------------------------------------------------------------------------------------------------------------
HELP_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def help_flags(server_bin: str, timeout: float = 90.0) -> Optional[set]:
    """All flag spellings the binary prints in --help, or None if it could not be asked."""
    try:
        p = subprocess.run([server_bin, "--help"], capture_output=True, text=True, timeout=timeout, env=_env_for(server_bin, {}))
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    found = set(HELP_FLAG.findall(text))
    return found if len(found) > 20 else None


def unsupported_flags(cmd: Sequence[str], known: Optional[set]) -> List[str]:
    if known is None:
        return []
    bad = []
    for name, _vals in LP.parse_flags(cmd[1:]):
        # parse_flags returns long names; accept the flag if either its long name or any alias of it is printed in --help
        aliases = {name} | {s for s, l in LP.ALIASES.items() if l == name}
        if not (aliases & known):
            bad.append(name)
    return sorted(set(bad))


def _env_for(server_bin: str, extra_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    bindir = str(Path(server_bin).parent)
    env["LD_LIBRARY_PATH"] = bindir + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.update(extra_env)
    return env


# ----------------------------------------------------------------------------------------------------------------------------------
# server control
# ----------------------------------------------------------------------------------------------------------------------------------
def http_get(url: str, timeout: float = 5.0) -> Tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError, ConnectionError, socket.timeout):
        return 0, ""


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex((host, port)) != 0
    finally:
        s.close()


def wait_port_free(port: int, seconds: int = 90) -> bool:
    for _ in range(seconds):
        if port_free(port):
            return True
        time.sleep(1)
    return False


def stop_production() -> bool:
    """Stop the production server through its own launcher. True when the port is free afterwards."""
    if port_free(paths.PORT) and LP.live_server_tokens() is None:
        return True
    launcher = shutil.which("llama")
    if launcher:
        subprocess.run([launcher, "--stop"], check=False)
        time.sleep(2)
    return wait_port_free(paths.PORT, 90)


def start_server(cmd: Sequence[str], env: Dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    f.write("# " + shlex.join(cmd) + "\n")
    f.flush()
    return subprocess.Popen(list(cmd), stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop_process(proc: Optional[subprocess.Popen], grace: float = 45.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=30)


FAILURE_HINTS = [
    (r"unknown argument|invalid argument|error: unknown|unrecognized (option|argument)", "the binary rejected a flag; see the first lines of the log"),
    (r"unknown model architecture|unsupported architecture|unknown architecture", "this build does not know the model architecture (qwen3.5/3.6 MoE): the fork is too old for this GGUF"),
    (r"out of memory|cudaMalloc failed|failed to allocate|ggml_backend_cuda_buffer_type_alloc_buffer", "ran out of GPU memory: lower mib/slots/k or raise n"),
    (r"illegal memory access|CUDA error", "a CUDA error: read the lines around it in the log"),
    (r"failed to load model|error loading model", "the model failed to load: read the lines just above this one in the log"),
    (r"requires --moe|requires --|cannot be combined|mutually exclusive", "a flag combination was rejected: read the message in the log"),
]


def diagnose(log_text: str) -> str:
    for pat, hint in FAILURE_HINTS:
        if re.search(pat, log_text, re.I):
            return hint
    return "no known pattern; read the end of the log"


def wait_ready(proc: subprocess.Popen, url: str, timeout: float) -> Tuple[bool, float, str]:
    """-> (ok, seconds, reason). Polls /health; returns early if the process exits."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, time.time() - t0, f"the server exited with code {proc.returncode} while loading"
        code, _ = http_get(url + "/health", 3.0)
        if code == 200:
            return True, time.time() - t0, ""
        time.sleep(2.0)
    return False, time.time() - t0, f"not healthy after {timeout:.0f} s"


def vram_used_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def tee_run(cmd: Sequence[str], logfile: Path, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    lines: List[str] = []
    with subprocess.Popen(list(cmd), cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        rc = p.wait()
    text = "".join(lines)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.write_text(text, encoding="utf-8")
    return rc, text


# ----------------------------------------------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------------------------------------------
def _captured(fn, *a) -> Tuple[int, List[str]]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, [ln.rstrip() for ln in buf.getvalue().splitlines()]


def score(baseline_path: Path, cand_path: Path) -> dict:
    base, cand = load_speedlog(baseline_path), load_speedlog(cand_path)
    rc1, l1 = _captured(gates.gate_p1, base, cand)
    try:
        rc3, l3 = _captured(gates.gate_g3, cand, base)
    except SpeedLogError as e:
        rc3, l3 = 1, [f"[gate G3] not scored: {e}"]
    return {
        "baseline": str(baseline_path), "result": str(cand_path),
        "metrics": {"gen_short": cand.gen(0), "gen_12k": cand.gen(12000), "prefill_12k": cand.prefill(12000), "vram_peak_gib": cand.vram_peak_gib,
                    "baseline_gen_short": base.gen(0), "baseline_gen_12k": base.gen(12000), "baseline_prefill_12k": base.prefill(12000)},
        "p1": {"pass": rc1 == 0, "lines": l1}, "g3": {"pass": rc3 == 0, "lines": l3},
    }


def newest_for_label(label: str) -> Optional[Path]:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label)
    cands = sorted(paths.SPEED_DIR.glob(f"*_{safe}.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def baseline_settings(baseline: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"depths": "0,4000,12000", "repeats": 3, "gen": 256, "warmup": 2}
    try:
        meta = json.loads(baseline.read_text(encoding="utf-8")).get("meta", {})
    except (OSError, ValueError):
        return out
    if isinstance(meta.get("depths"), list) and meta["depths"]:
        out["depths"] = ",".join(str(int(d)) for d in meta["depths"])
    for k in ("repeats", "gen"):
        if isinstance(meta.get(k), int) and meta[k] > 0:
            out[k] = meta[k]
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# one variant, start to finish
# ----------------------------------------------------------------------------------------------------------------------------------
def run_variant(v: Variant, a, known_cache: Dict[str, Optional[set]]) -> dict:
    build = BUILDS[v.build]
    rec: dict = {"label": v.label, "spec": v.spec, "family": v.family, "build": v.build, "commit": build.commit, "status": "not started", "notes": v.notes}
    out = OUT_DIR / v.label
    out.mkdir(parents=True, exist_ok=True)
    cmd, ref, ref_src = full_command(v, shlex.split(a.extra) if a.extra else [])
    rec["reference"] = ref_src
    d = LP.diff(ref, cmd)
    launch = {"label": v.label, "command": cmd, "env": v.env, "reference_source": ref_src, "diff_vs_production": d, "build": build.key, "commit": build.commit,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(f"\n=== {v.label} ({v.spec}) ===")
    print(f"  binary: {build.server}\n  reference command: {ref_src}")
    for ln in LP.format_diff(d):
        print(ln)
    for n in v.notes:
        print("  note:", n)
    if v.env:
        print("  env:", " ".join(f"{k}={x}" for k, x in v.env.items()))
    if a.dry_run:
        print("  command:", shlex.join(cmd))
        rec["status"] = "dry run"
        return rec

    if not build.server.exists():
        rec["status"] = "binary missing"
        how = "bash patches/09_build_pelias_tier.sh build" if build.key == "tier" else f"bash fork/05_phase5_fork_gauntlet.sh build {build.key}"
        print(f"  STOP: {build.server} does not exist. Build it first: {how}", file=sys.stderr)
        return rec
    if not paths.GGUF.exists():
        rec["status"] = "model missing"
        print(f"  STOP: model not found: {paths.GGUF}", file=sys.stderr)
        return rec
    for k, x in v.params.items():
        if k == "profile" and x not in ("none", "auto") and not Path(x).exists():
            rec["status"] = "profile missing"
            print(f"  STOP: profile file not found: {x}", file=sys.stderr)
            return rec
    if v.family == "tier":
        hs = Path(v.env.get("PELIAS_HOT_SET", ""))
        if not hs.exists():
            rec["status"] = "hot set missing"
            print(f"  STOP: hot set file not found: {hs}. Make it with: python sim/export_hot_set.py", file=sys.stderr)
            return rec
        v.env["PELIAS_TIER_STATS_FILE"] = str(out / "tier_stats.json")
    if v.family == "schwerz" and v.params.get("profile") == "auto" and not DEFAULT_PROFILE.exists():
        rec["status"] = "profile missing"
        print(f"  STOP: {DEFAULT_PROFILE} does not exist yet. Collect it first: bash fork/05_phase5_fork_gauntlet.sh profile", file=sys.stderr)
        return rec
    if v.mtp and not v.params.get("mtpfile"):
        try:
            sys.path.insert(0, str(paths.MOE_TRACK / "lib"))
            from gguf_min import read_gguf
            g = read_gguf(paths.GGUF, keep_arrays_up_to=0)
            if not any("nextn" in t.name for t in g.tensors):
                rec["status"] = "no MTP head"
                print("  STOP: the GGUF has no nextn/MTP tensors; mtp=N cannot work with this file.", file=sys.stderr)
                return rec
        except Exception as e:  # reading the header is a convenience check only
            print(f"  (could not check the GGUF for an MTP head: {e})")

    key = str(build.server)
    if key not in known_cache:
        print("  asking the binary which flags it knows ...")
        known_cache[key] = help_flags(key)
    bad = unsupported_flags(cmd, known_cache[key])
    if bad:
        ess = {LP.long_name(x) for x in v.essential}
        fatal = [b for b in bad if b in ess]
        print("  this binary does not list: " + ", ".join(bad))
        if fatal or not a.drop_unsupported:
            rec["status"] = "unsupported flags"
            rec["unsupported"] = bad
            print("  STOP: " + ("the variant itself needs " + ", ".join(fatal) if fatal else "rerun with --drop-unsupported to remove the parity flags it does not know"), file=sys.stderr)
            return rec
        cmd = LP.drop_flags(cmd, bad)
        print("  dropped:", ", ".join(bad))
        launch["dropped_flags"] = bad
        launch["command"] = cmd
        (out / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")

    print("  stopping the production server (this frees the GPU) ...")
    if not stop_production():
        rec["status"] = "port busy"
        print(f"  STOP: port {paths.PORT} is still in use; stop whatever holds it and retry", file=sys.stderr)
        return rec

    url = f"http://{paths.HOST}:{paths.PORT}"
    proc: Optional[subprocess.Popen] = None
    try:
        t_start = time.time()
        proc = start_server(cmd, _env_for(key, v.env), out / "server.log")
        ok, secs, why = wait_ready(proc, url, a.ready_timeout)
        if not ok:
            tail = ""
            try:
                tail = (out / "server.log").read_text(encoding="utf-8", errors="replace")[-6000:]
            except OSError:
                pass
            rec["status"] = "did not start"
            rec["reason"] = why
            rec["diagnosis"] = diagnose(tail)
            print(f"  FAILED to start: {why}\n  likely cause: {rec['diagnosis']}\n  log: {out / 'server.log'}", file=sys.stderr)
            return rec
        rec["load_seconds"] = round(secs, 1)
        rec["vram_after_load_mib"] = vram_used_mib()
        print(f"  server healthy after {secs:.0f} s; GPU memory in use {rec['vram_after_load_mib']} MiB")
        if v.family == "tier":
            try:
                log_text = (out / "server.log").read_text(encoding="utf-8", errors="replace")
            except OSError:
                log_text = ""
            m = re.search(r"pelias-tier: hot bank active for (\d+) layers, ([\d.]+) MiB", log_text)
            rec["tier_active"] = bool(m)
            if m:
                rec["tier_layers"], rec["tier_mib"] = int(m.group(1)), float(m.group(2))
                print(f"  hot bank active: {rec['tier_layers']} layers, {rec['tier_mib']:.0f} MiB")
            else:
                rec["status"] = "bank not active"
                why = next((ln for ln in log_text.splitlines() if "pelias-tier" in ln and ("ERROR" in ln or "Running without" in ln or "running without" in ln)), "no pelias-tier line in the log")
                print(f"  STOP: the hot bank was not created, so this would measure the plain build. {why.strip()}", file=sys.stderr)
                return rec

        rc, _ = tee_run([PY, PROBE_SCRIPT, "check", "--url", url], out / "check.txt", cwd=paths.CREATOR)
        rec["check_exit"] = rc
        if rc == 2:
            rec["status"] = "check failed"
            print("  the server does not answer properly; not speed-testing it", file=sys.stderr)
            return rec

        base_path = Path(a.baseline)
        st = baseline_settings(base_path)
        depths = a.depths or str(st["depths"])
        repeats = a.repeats or int(st["repeats"])
        gen = a.gen or int(st["gen"])
        note = f"{v.label}: {build.key} {build.commit[:7]}; " + " ".join(v.flags)
        rc, _ = tee_run([PY, SPEED_SCRIPT, "run", "--label", v.label, "--depths", depths, "--gen", str(gen), "--repeats", str(repeats),
                         "--warmup", str(st["warmup"]), "--url", url, "--note", note[:900]], out / "speed.txt", cwd=paths.CREATOR)
        res = newest_for_label(v.label)
        if rc != 0 or res is None:
            rec["status"] = "speed test failed"
            print("  the speed test did not finish; see speed.txt", file=sys.stderr)
            return rec
        rec["speed_file"] = str(res)
        verdict = score(base_path, res)
        rec.update({"status": "measured", "verdict": verdict})
        for sect in ("p1", "g3"):
            for ln in verdict[sect]["lines"]:
                print("  " + ln)

        wanted = set(a.tests.split(",")) if a.tests else set()
        if v.reuse:
            wanted.add("reuse")
        if v.mtp:
            wanted.add("mtp")
        if "reuse" in wanted:
            rc, _ = tee_run([PY, str(HERE / "5d_prompt_reuse.py"), "--url", url, "--label", v.label], out / "reuse.txt", cwd=paths.CREATOR)
            rec["reuse_exit"] = rc
        if "mtp" in wanted:
            rc, _ = tee_run([PY, str(HERE / "5e_mtp_acceptance.py"), "run", "--url", url, "--label", v.label], out / "mtp.txt", cwd=paths.CREATOR)
            rec["mtp_exit"] = rc
        return rec
    finally:
        stop_process(proc)
        wait_port_free(paths.PORT, 60)
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (out / "verdict.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------------------------
def kl_for(label: str) -> Optional[dict]:
    kl_dir = paths.MOE_TRACE_DIR / "kl"
    for p in sorted(kl_dir.glob(f"{label}_vs_*.json")):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def report_rows(out_dir: Path = OUT_DIR) -> List[dict]:
    rows = []
    for vp in sorted(out_dir.glob("*/verdict.json")):
        try:
            r = json.loads(vp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m = (r.get("verdict") or {}).get("metrics") or {}
        kl = kl_for(r.get("label", ""))
        rows.append({
            "label": r.get("label"), "status": r.get("status"), "gen_short": m.get("gen_short"), "gen_12k": m.get("gen_12k"), "prefill_12k": m.get("prefill_12k"),
            "vram": m.get("vram_peak_gib"), "p1": (r.get("verdict") or {}).get("p1", {}).get("pass"), "g3": (r.get("verdict") or {}).get("g3", {}).get("pass"),
            "kl": kl.get("mean_kl") if kl else None, "top1": kl.get("top1_agreement") if kl else None,
        })
    return rows


def format_report(rows: List[dict]) -> List[str]:
    def f(x, spec):
        return "-" if x is None else format(x, spec)
    def yn(x):
        return "-" if x is None else ("PASS" if x else "no")
    head = f"{'variant':46s} {'status':16s} {'gen':>6s} {'gen12k':>7s} {'pre12k':>7s} {'VRAM':>6s} {'P1':>5s} {'G3':>5s} {'KL':>9s} {'top1':>6s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{str(r['label'])[:46]:46s} {str(r['status'])[:16]:16s} {f(r['gen_short'], '6.1f'):>6s} {f(r['gen_12k'], '7.1f'):>7s} {f(r['prefill_12k'], '7.0f'):>7s} "
                     f"{f(r['vram'], '6.2f'):>6s} {yn(r['p1']):>5s} {yn(r['g3']):>5s} {f(r['kl'], '9.6f'):>9s} {f(None if r['top1'] is None else r['top1'] * 100, '5.1f'):>6s}")
    lines.append("gen = tok/s short prompt, gen12k / pre12k = generation / prompt reading at 12k context; VRAM = GiB peak; KL from sim/kl_gate.py (blank until run)")
    return lines


# ----------------------------------------------------------------------------------------------------------------------------------
DEFAULT_PLAN = [
    ("prod:n=28", "control: production build launched by this tool (same launcher as the forks)"),
    ("mcache:n=40,mib=5120", "mainline expert cache, all experts in RAM + 5 GiB VRAM cache"),
    ("mcache:n=40,mib=4096", "same with a smaller cache (VRAM headroom)"),
    ("schwerz:mib=5120", "Schwerz hybrid, no profile (LRU only)"),
    ("schwerz:mib=5120,profile=auto", "Schwerz hybrid with the profile made from your own traffic"),
    ("schwerz:mib=5120,profile=auto,adapt=occurrence", "same, cache re-ranked while running"),
    ("parmesan:n=40,k=78", "Parmesan hot-expert bank (unverified with this GGUF)"),
]
MINUTES_PER_VARIANT = 15


def cmd_plan(a) -> int:
    print("Phase 5 plan (each variant: load 3-8 min + check 1 min + speed test ~6 min, about %d min; nothing here is run)\n" % MINUTES_PER_VARIANT)
    for i, (spec, why) in enumerate(DEFAULT_PLAN, 1):
        try:
            v = parse_spec(spec)
            b = BUILDS[v.build]
            have = "built" if b.server.exists() else "NOT BUILT"
        except SpecError as e:
            have = f"spec error: {e}"
        print(f"{i}. {spec:52s} [{have}]\n     {why}")
    print("\nThen sweep the best one, for example:  python fork/gauntlet.py run schwerz:mib=5120,profile=auto --sweep miss=0,0.17,0.5,1")
    print("Gates: P1 (+5% on short generation, 12k generation or 12k prompt reading, none worse than -3%, VRAM <= 11.2 GiB), G3 (>= 55 tok/s at 12k, >= 1150 prompt tok/s, VRAM <= 11.2 GiB), H6 (python sim/kl_gate.py).")
    print("Total for the seven variants: about %.1f hours." % (len(DEFAULT_PLAN) * MINUTES_PER_VARIANT / 60.0))
    return 0


def cmd_builds(a) -> int:
    for k, b in BUILDS.items():
        print(f"{k:9s} {'built' if b.server.exists() else 'not built':10s} {b.src}   commit {b.commit[:7]}   {b.note}")
    return 0


def cmd_flags(a) -> int:
    v = parse_spec(a.specs[0])
    cmd, ref, src = full_command(v, shlex.split(a.extra) if a.extra else [])
    print(f"variant {v.label}   binary {BUILDS[v.build].server}")
    print(f"reference: {src}")
    for ln in LP.format_diff(LP.diff(ref, cmd)):
        print(ln)
    if v.env:
        print("env: " + " ".join(f"{k}={x}" for k, x in v.env.items()))
    print("command:\n  " + shlex.join(cmd))
    return 0


def cmd_run(a) -> int:
    specs = expand_sweep(a.specs, a.sweep)
    variants = [parse_spec(s) for s in specs]
    labels = [v.label for v in variants]
    if len(set(labels)) != len(labels):
        raise SpecError("two variants have the same label: " + ", ".join(sorted({x for x in labels if labels.count(x) > 1})))
    if not a.dry_run and not Path(a.baseline).exists():
        print(f"baseline speed file not found: {a.baseline}", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not a.dry_run:
        print("This stops the llama-server on port %d and leaves it stopped when the last variant ends%s." % (paths.PORT, " (then restarts it: --restore)" if a.restore else ""))
    known: Dict[str, Optional[set]] = {}
    results = []
    failed = 0
    for v in variants:
        rec = run_variant(v, a, known)
        results.append(rec)
        if rec["status"] not in ("measured", "dry run"):
            failed += 1
            if not a.keep_going:
                print("\nstopping after the first failure (use --keep-going to continue)", file=sys.stderr)
                break
    if a.restore and not a.dry_run:
        launcher = shutil.which("llama")
        if launcher:
            subprocess.run([launcher, "--serve", "--stuart-creator", "--bg"], check=False)
        else:
            print("cannot restore: the 'llama' launcher is not on PATH", file=sys.stderr)
    elif not a.dry_run:
        print("\nThe production server is STOPPED. Start it again with:  llama --serve --stuart-creator --bg")
    if not a.dry_run:
        print()
        for ln in format_report(report_rows()):
            print(ln)
    return 1 if failed else 0


def cmd_report(a) -> int:
    rows = report_rows()
    if not rows:
        print(f"nothing measured yet under {OUT_DIR}")
        return 0
    lines = format_report(rows)
    for ln in lines:
        print(ln)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "leaderboard.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def cmd_build_info(a) -> int:
    b = BUILDS.get(a.specs[0])
    if not b:
        print(f"unknown build {a.specs[0]!r}", file=sys.stderr)
        return 2
    print(f"{b.repo}\t{b.commit}\t{b.src}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "builds", "flags", "run", "report", "build-info"])
    ap.add_argument("specs", nargs="*")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--depths", default="")
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--tests", default="")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--drop-unsupported", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.action == "plan":
            return cmd_plan(a)
        if a.action == "builds":
            return cmd_builds(a)
        if a.action == "report":
            return cmd_report(a)
        if not a.specs:
            print(f"{a.action} needs at least one SPEC (see --help)", file=sys.stderr)
            return 2
        if a.action == "build-info":
            return cmd_build_info(a)
        if a.action == "flags":
            return cmd_flags(a)
        return cmd_run(a)
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
