#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_pelias_v03.py
"""Tests patches/pelias_v03_patches.py and stuart_pelias0.3.py without torch, on small stand-ins:
  * the peak-VRAM guard (only the first micro-batch of an accumulation group resets; other callers are untouched),
  * the KV-cached probe (same replies as the uncached loop, prompt processed once, the agreement check, both fallbacks),
  * the layer-input offload (copy ordering between streams, buffer reuse, data restored for the recompute, fall-back to the stock checkpoint, install wiring),
  * the launcher shim (flags, finding the 0.2 launcher, wiring into its install_patches).
What it cannot check: real autograd, real CUDA streams, the real transformers cache. Those are exercised by `python stuart_pelias0.3.py smoke` on your machine.

    python tests/test_pelias_v03.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "patches"))
import pelias_v03_patches as V3  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================================================================================================
# a tiny numpy-backed stand-in for the parts of torch these patches touch
# ======================================================================================================================================================
class DT:
    def __init__(self, name, npd, size):
        self.name, self.np, self.size = name, npd, size

    def __repr__(self):
        return self.name


class Ev:
    n = 0

    def __init__(self, log, stream):
        Ev.n += 1
        self.name, self.log, self.stream = f"ev{Ev.n}", log, stream

    def synchronize(self):
        self.log.append(("sync", self.name))


class Stream:
    def __init__(self, st, name):
        self.st, self.name = st, name

    def record_event(self):
        ev = Ev(self.st.log, self.name)
        self.st.log.append(("record", self.name, ev.name))
        return ev

    def wait_event(self, ev):
        self.st.log.append(("wait", self.name, ev.name))


class MiniT:
    def __init__(self, a, dev="cuda", rg=False, st=None):
        self.a, self.dev, self.requires_grad, self.grad, self.st = a, dev, rg, None, st

    shape = property(lambda s: tuple(s.a.shape))
    device = property(lambda s: s.dev)
    is_cuda = property(lambda s: s.dev == "cuda")
    dtype = property(lambda s: {np.dtype(np.uint16): s.st.bfloat16, np.dtype(np.uint8): s.st.uint8, np.dtype(np.float32): s.st.float32}[s.a.dtype])

    def numel(self):
        return int(self.a.size)

    def element_size(self):
        return int(self.a.itemsize)

    def dim(self):
        return self.a.ndim

    def detach(self):
        return MiniT(self.a, self.dev, False, self.st)

    def requires_grad_(self, flag=True):
        self.requires_grad = bool(flag)
        return self

    def record_stream(self, s):
        self.st.log.append(("record_stream", s.name))

    def copy_(self, src, non_blocking=False):
        self.a[...] = src.a
        self.st.log.append(("copy", self.st.current().name, "h2d" if self.is_cuda else "d2h"))
        return self

    def to(self, dev, non_blocking=False):
        self.st.log.append(("to", self.st.current().name, dev))
        return MiniT(self.a.copy(), dev, False, self.st)

    def view(self, x):
        if isinstance(x, DT):
            return MiniT(self.a.view(x.np), self.dev, False, self.st)
        return MiniT(self.a.reshape(x), self.dev, False, self.st)

    def __getitem__(self, sl):
        return MiniT(self.a[sl], self.dev, False, self.st)

    def argmax(self, dim, keepdim=False):
        return MiniT(np.argmax(self.a, axis=dim)[..., None] if keepdim else np.argmax(self.a, axis=dim), self.dev, False, self.st)

    def cpu(self):
        return MiniT(self.a, "cpu", False, self.st)

    def __int__(self):
        return int(self.a.reshape(-1)[0])


class StubTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.log = []
        self.grad_enabled = True
        self.uint8, self.bfloat16, self.float32 = DT("uint8", np.uint8, 1), DT("bfloat16", np.uint16, 2), DT("float32", np.float32, 4)
        self._streams = [Stream(self, "main")]
        self.bw_calls, self.std_calls, self.reset_calls = [], [], []
        st = self

        class Function:
            _last_ctx = None

            @classmethod
            def apply(cls, *args):
                ctx = types.SimpleNamespace()
                prev = st.grad_enabled
                st.grad_enabled = False
                try:
                    out = cls.forward(ctx, *args)
                finally:
                    st.grad_enabled = prev
                cls._last_ctx = ctx
                return out

        class _Ctx:
            def __init__(self, on):
                self.on = on

            def __enter__(self):
                self.prev, st.grad_enabled = st.grad_enabled, self.on

            def __exit__(self, *a):
                st.grad_enabled = self.prev

            def __call__(self, f):                                    # usable as a decorator like torch.no_grad()
                def w(*a, **k):
                    with _Ctx(self.on):
                        return f(*a, **k)
                return w

        class _StreamCtx:
            def __init__(self, s):
                self.s = s

            def __enter__(self):
                st._streams.append(self.s)

            def __exit__(self, *a):
                st._streams.pop()

        def bw(tensors, grads):
            st.bw_calls.append((tensors, grads))
            for t, g in zip(tensors, grads):
                src = getattr(t, "src", None)
                if src is not None and src.requires_grad:
                    src.grad = MiniT(g.a * 2, g.dev, False, st)

        self.cuda = types.SimpleNamespace(Stream=lambda: Stream(st, "side"), stream=_StreamCtx, current_stream=st.current,
                                          reset_peak_memory_stats=lambda device=None: st.reset_calls.append(device))
        self.autograd = types.SimpleNamespace(Function=Function, backward=bw)
        self.no_grad = lambda: _Ctx(False)
        self.enable_grad = lambda: _Ctx(True)
        self.is_grad_enabled = lambda: st.grad_enabled
        self.is_tensor = lambda x: isinstance(x, MiniT)
        self.device = lambda d: d
        utils = types.ModuleType("torch.utils")
        ck = types.ModuleType("torch.utils.checkpoint")
        ck.checkpoint = lambda f, *a, use_reentrant=None, **k: (st.std_calls.append(use_reentrant), f(*a))[1]
        utils.checkpoint = ck
        self.utils = utils
        sys.modules["torch.utils"], sys.modules["torch.utils.checkpoint"] = utils, ck

    def current(self):
        return self._streams[-1]

    def empty(self, *shape, dtype=None, device="cpu", pin_memory=False, requires_grad=False):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        a = np.zeros(shape, dtype=(dtype.np if dtype is not None else np.float32))
        return MiniT(a, device, requires_grad, self)

    def cat(self, ts, dim=0):
        return MiniT(np.concatenate([t.a for t in ts], axis=dim), ts[0].dev, False, self)


# ======================================================================================================================================================
# 1. peak guard
# ======================================================================================================================================================
def test_peak_guard():
    tt = StubTorch()
    src = ("def run_training(args, cuda, n):\n    micro = 0\n    for _ in range(n):\n        cuda.reset_peak_memory_stats()\n        micro += 1\n"
           "def run_training_nomicro(args, cuda, n):\n    for _ in range(n):\n        cuda.reset_peak_memory_stats()\n")
    ns = {}
    exec(compile(src, "/home/u/pelias_src/pelias_v0.2/pelias/train.py", "exec"), ns)
    other = {}
    exec(compile("def run_training(args, cuda, n):\n    micro = 0\n    for _ in range(n):\n        cuda.reset_peak_memory_stats()\n        micro += 1\n", "/x/other.py", "exec"), other)
    check("install returns True the first time", V3.install_peak_guard(tt) is True)
    check("a second install does nothing", V3.install_peak_guard(tt) is False)
    cuda = tt.cuda
    ns["run_training"](types.SimpleNamespace(grad_accum=4), cuda, 12)
    check("run_training: 12 micro-batches, accumulation 4 -> real resets at micro 0, 4, 8 only", len(tt.reset_calls) == 3, str(len(tt.reset_calls)))
    tt.reset_calls.clear()
    ns["run_training"](types.SimpleNamespace(grad_accum=1), cuda, 5)
    check("grad_accum 1: every micro-batch resets (nothing to protect)", len(tt.reset_calls) == 5)
    tt.reset_calls.clear()
    other["run_training"](types.SimpleNamespace(grad_accum=4), cuda, 8)
    check("same function name in another file: not filtered", len(tt.reset_calls) == 8)
    tt.reset_calls.clear()
    ns["run_training_nomicro"](types.SimpleNamespace(grad_accum=4), cuda, 3)
    check("another function of pelias/train.py: not filtered", len(tt.reset_calls) == 3)
    tt.reset_calls.clear()
    ns["run_training"](types.SimpleNamespace(), cuda, 4)
    check("no grad_accum to read: falls through to the real reset", len(tt.reset_calls) == 4)
    tt.reset_calls.clear()
    cuda.reset_peak_memory_stats(7)
    check("called from elsewhere with a device argument: passed through", tt.reset_calls == [7])
    V3.remove_peak_guard(tt)
    check("remove restores the original", not getattr(tt.cuda.reset_peak_memory_stats, "_pelias_v03", False))


# ======================================================================================================================================================
# 2. cached probe
# ======================================================================================================================================================
V = 53
IM_END = 7


def toy_next(tokens):
    return (sum(tokens) * 7 + len(tokens) * 3) % V


class ToyLM:
    """Next token = f(sum, length) of everything seen; the cache holds (sum, length). mode: ok | wrong_cache | raises"""

    def __init__(self, tt, mode="ok"):
        self.tt, self.mode, self.calls = tt, mode, []

    def hidden(self, nxt, n):
        h = np.zeros((1, n, V), dtype=np.float32)
        h[0, -1, nxt % V] = 1.0
        return MiniT(h, "cuda", False, self.tt)

    def __call__(self, inputs_embeds=None, past_key_values=None, use_cache=False):
        ids = [int(v) for v in inputs_embeds.a.reshape(-1)]
        self.calls.append((len(ids), past_key_values is not None, use_cache))
        if use_cache:
            if self.mode == "raises" and past_key_values is not None:
                raise TypeError("cache layout not supported")
            cache = past_key_values if past_key_values is not None else {"s": 0, "n": 0}
            cache["s"] += sum(ids)
            cache["n"] += len(ids)
            nxt = (cache["s"] * 7 + cache["n"] * 3) % V                # same function of (sum, length) as toy_next
            if self.mode == "wrong_cache" and past_key_values is not None:
                nxt = (nxt + 1) % V
            return types.SimpleNamespace(last_hidden_state=self.hidden(nxt, len(ids)), past_key_values=cache)
        return types.SimpleNamespace(last_hidden_state=self.hidden(toy_next(ids), len(ids)), past_key_values=None)


def run_probe(mode, prompts, max_new):
    tt = StubTorch()
    logs, model = [], ToyLM(tt, mode)

    class Base:
        pass

    base = Base()
    base.model = model
    base.lm_head = lambda h: h
    peft_model = types.SimpleNamespace(get_base_model=lambda: base, eval=lambda: None)
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda m, p: peft_model)

    class Tok:
        pad_token_id = 0

        def convert_tokens_to_ids(self, s):
            return IM_END

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[-1]["content"]

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": MiniT(np.array([[ord(c) % V for c in text]], dtype=np.int64), "cpu", False, tt)}

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(int(v)) for v in ids.a.reshape(-1))

    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: Tok())
    sys.modules["peft"], sys.modules["transformers"] = peft, tf

    def orig_probe(args, log):
        raise AssertionError("the original probe must not run")

    T = types.SimpleNamespace(stage_probe=orig_probe, stage_load=lambda a, l: (object(), types.SimpleNamespace(phase=0), None),
                              embed_on_host=lambda m, ids, dev: MiniT(ids.a.astype(np.float32), "cuda", False, tt),
                              resolve_snapshot=lambda p: Path("/snap"))
    new = V3.make_cached_probe(T, tt)
    args = types.SimpleNamespace(device="cuda:0", hf_cache="/x", adapter="/a", probe_file=None, max_new_tokens=max_new)
    import json
    pf = Path(tempfile.mkdtemp()) / "p.json"
    pf.write_text(json.dumps([[{"role": "user", "content": p}] for p in prompts]))
    args.probe_file = str(pf)
    new(args, logs.append)
    return model, logs


def reference_reply(prompt, max_new):
    toks = [ord(c) % V for c in prompt]
    gen = []
    for _ in range(max_new):
        nxt = toy_next(toks + gen)
        gen.append(nxt)
        if nxt == IM_END:
            break
    return " ".join(str(t) for t in gen)


def replies(logs):
    return [l[len("REPLY : "):] for l in logs if l.startswith("REPLY : ")]


def test_probe():
    prompts = ["Who are you", "Boiling point of water"]
    model, logs = run_probe("ok", prompts, 6)
    check("cached probe: replies equal the plain uncached loop", replies(logs) == [reference_reply(p, 6) for p in prompts], str(replies(logs)))
    check("cached probe: the agreement check ran and passed", any("agrees with the uncached pass" in l for l in logs))
    n1 = len(prompts[0])
    cached_calls = [c for c in model.calls if c[2]]
    check("cached probe: each prompt is prefilled once (full length with use_cache) and then fed one token at a time",
          [c[0] for c in cached_calls if c[0] > 1] == [len(p) for p in prompts] and all(c[0] == 1 for c in cached_calls if c[0] <= 1))
    uncached = [c for c in model.calls if not c[2]]
    check("cached probe: exactly one uncached verification pass, over prompt + first token", len(uncached) == 1 and uncached[0][0] == n1 + 1, str(uncached))
    check("cached probe: the PROMPT lines are logged", sum(l.startswith("PROMPT: ") for l in logs) == 2)

    model, logs = run_probe("wrong_cache", prompts, 6)
    check("a cache that disagrees is caught: fallback message, replies still equal the uncached loop",
          any("KV-cache path unusable" in l for l in logs) and replies(logs) == [reference_reply(p, 6) for p in prompts], str(logs))
    check("after the fallback the second prompt does not try the cache again", sum(1 for c in model.calls if c[2] and c[0] == len(prompts[1])) == 0)

    model, logs = run_probe("raises", prompts, 5)
    check("a cache path that raises falls back and the replies are right", any("KV-cache path unusable" in l for l in logs) and replies(logs) == [reference_reply(p, 5) for p in prompts])

    model, logs = run_probe("ok", ["x"], 1)
    check("max_new_tokens = 1 works (no verification needed)", replies(logs) == [reference_reply("x", 1)])
    # stop token: pick a prompt whose first generated token is IM_END
    stopper = None
    for a in range(32, 127):
        if toy_next([ord(chr(a)) % V]) == IM_END:
            stopper = chr(a)
            break
    if stopper:
        model, logs = run_probe("ok", [stopper], 6)
        check("generation stops at <|im_end|> and includes it", replies(logs) == [reference_reply(stopper, 6)] and replies(logs)[0].endswith(str(IM_END)))
    else:
        check("stop-token case (no suitable prompt in the toy model; skipped)", True)


# ======================================================================================================================================================
# 3. offload
# ======================================================================================================================================================
def test_offload():
    tt = StubTorch()
    off = V3.make_offload_checkpoint(tt, hidden_dim=8)
    F = off._Offloaded
    rng = np.random.default_rng(3)

    def bf(shape):
        return MiniT(rng.integers(0, 60000, size=shape).astype(np.uint16), "cuda", False, tt)

    runs = []

    def make_run():
        def run(xx):
            runs.append({"grad_enabled": tt.grad_enabled, "x": xx.a.copy(), "req": xx.requires_grad})
            out = MiniT(xx.a + 1, "cuda", tt.grad_enabled, tt)
            out.src = xx
            return out
        return run

    x = bf((1, 3, 8))
    orig = x.a.copy()
    x.requires_grad = True
    out = off(make_run(), x)
    ctx = F._last_ctx
    check("forward: the layer ran once, with gradients off, on the input", len(runs) == 1 and runs[0]["grad_enabled"] is False and np.array_equal(runs[0]["x"], orig))
    check("forward: the output is the layer output", np.array_equal(out.a, orig + 1))
    ev_ready = [e for e in tt.log if e[0] == "record" and e[1] == "main"][0]
    i_wait = tt.log.index(("wait", "side", ev_ready[2]))
    i_copy = tt.log.index(("copy", "side", "d2h"))
    check("forward: the side stream waits for the stream that produced x BEFORE the copy", i_wait < i_copy)
    check("forward: x is registered with the side stream so its memory is not reused early", ("record_stream", "side") in tt.log)
    check("forward: the host copy holds x exactly", np.array_equal(ctx.buf.a, orig))
    x.a[...] = 0                                                       # the allocator reuses x's memory
    tt.log.clear()
    runs.clear()
    grad = MiniT(np.ones((1, 3, 8), np.float32), "cuda", False, tt)
    res = F.backward(ctx, grad)
    check("backward: waits for the copy event on the compute stream before reading the host buffer", ("wait", "main", ctx.copied.name) in tt.log and
          tt.log.index(("wait", "main", ctx.copied.name)) < tt.log.index(("to", "main", "cuda")))
    check("backward: recomputes the layer with gradients on, on the ORIGINAL input (not the overwritten memory)",
          len(runs) == 1 and runs[0]["grad_enabled"] is True and np.array_equal(runs[0]["x"], orig) and runs[0]["req"] is True)
    check("backward: backpropagates through the recomputed output with the incoming gradient", len(tt.bw_calls) == 1 and tt.bw_calls[0][1][0] is grad)
    check("backward: returns (None, grad of x, None)", len(res) == 3 and res[0] is None and res[2] is None and res[1] is not None and np.allclose(res[1].a, 2.0))
    check("backward: the layer input is not kept afterwards", ctx.buf is None and ctx.raw is None)

    # buffer reuse: the next forward of the same size gets the same pinned buffer after its last read has finished
    tt.log.clear()
    x2 = bf((1, 3, 8))
    off(make_run(), x2)
    check("reuse: the pooled buffer's last-read event is waited on before it is rewritten", any(e[0] == "sync" for e in tt.log))

    # first layer: x does not require grad -> output still requires grad through the dummy, and no grad for x is returned
    runs.clear()
    x3 = bf((1, 3, 8))
    off(make_run(), x3)
    ctx3 = F._last_ctx
    res3 = F.backward(ctx3, grad)
    check("input that does not require grad: backward returns None for it", res3[1] is None and runs[-1]["req"] is False)

    # things it does not handle go to the stock checkpoint
    tt.std_calls.clear()
    cpu = MiniT(np.zeros((1, 3, 8), np.float32), "cpu", False, tt)
    off(lambda a: a, cpu)
    check("a CPU / non-bf16 tensor goes to torch.utils.checkpoint (non-reentrant)", tt.std_calls == [False])
    off(lambda a, b: a, bf((1, 3, 8)), bf((1, 3, 8)))
    check("a second tensor argument goes to torch.utils.checkpoint", len(tt.std_calls) == 2)
    wrong_dim = bf((1, 3, 5))
    off(lambda a: a, wrong_dim)
    check("a tensor whose last dimension is not the hidden size goes to torch.utils.checkpoint", len(tt.std_calls) == 3)
    try:
        off(lambda a: {"x": a}, bf((1, 3, 8)))
        check("an output that is not a tensor raises (loud)", False)
    except RuntimeError as e:
        check("an output that is not a tensor raises (loud)", "only a tensor or a tuple of tensors" in str(e))

    # tuple output
    def run_tuple(xx):
        o = MiniT(xx.a + 1, "cuda", tt.grad_enabled, tt)
        o.src = xx
        return (o, MiniT(xx.a + 2, "cuda", tt.grad_enabled, tt))

    xt = bf((1, 3, 8))
    xt.requires_grad = True
    outs = off(run_tuple, xt)
    cx = F._last_ctx
    tt.bw_calls.clear()
    F.backward(cx, MiniT(np.ones((1, 3, 8), np.float32), "cuda", False, tt), None)
    check("tuple output: forward returns a tuple; backward skips outputs without gradient", isinstance(outs, tuple) and len(tt.bw_calls) == 1 and len(tt.bw_calls[0][0]) == 1)

    # grad disabled (evaluation): stock path
    tt.std_calls.clear()
    tt.grad_enabled = False
    off(lambda a: a, bf((1, 3, 8)))
    tt.grad_enabled = True
    check("with gradients off (evaluation) the stock path is used", tt.std_calls == [False])


def test_install_offload():
    tt = StubTorch()
    DL = type("Qwen3_5MoeDecoderLayer", (), {})

    def layer(gc=True):
        m = DL()
        m.gradient_checkpointing, m._gradient_checkpointing_func = gc, "stock"
        return m

    layers = [layer() for _ in range(3)]
    other = types.SimpleNamespace(gradient_checkpointing=True, _gradient_checkpointing_func="stock")
    model = types.SimpleNamespace(modules=lambda: layers + [other])
    seen = {}
    T = types.SimpleNamespace(attach_lora=lambda m, *a, **k: seen.update(args=a, kw=k) or ("peft_model", 123))
    logs = []
    check("install_offload wires attach_lora", V3.install_offload(T, tt, logs.append) is True)
    r = T.attach_lora(model, 16, 32, False)
    check("the three decoder layers got the offloading function, other modules did not",
          all(getattr(m, "_gradient_checkpointing_func") != "stock" for m in layers) and other._gradient_checkpointing_func == "stock")
    check("the original attach_lora still runs with the same arguments and its result is returned", seen["args"] == (16, 32, False) and r == ("peft_model", 123))
    check("it says how many layers were switched", any("installed on 3 decoder layers" in l for l in logs))
    check("installing twice does nothing", V3.install_offload(T, tt, logs.append) is False)
    model2 = types.SimpleNamespace(modules=lambda: [layer(gc=False)])
    logs.clear()
    T.attach_lora(model2)
    check("no layer with checkpointing enabled: says so and continues", any("NONE found" in l for l in logs))
    V3.uninstall_v03_patches(T, tt)
    check("uninstall restores attach_lora", not getattr(T.attach_lora, "_pelias_v03", False))


def test_install_all():
    tt = StubTorch()
    T = types.SimpleNamespace(stage_probe=lambda a, l: None, attach_lora=lambda m: None)
    logs = []
    done = V3.install_v03_patches(T, torch_module=tt, log=logs.append, offload_xl=False)
    check("default install: peak guard + cached probe, no offload", done.get("peak_guard") is True and done.get("cached_probe") is True and "offload_xl" not in done, str(done))
    check("the probe was replaced", getattr(T.stage_probe, "_pelias_v03", False))
    done2 = V3.install_v03_patches(T, torch_module=tt, log=logs.append, offload_xl=False)
    check("a second install does not wrap again", done2.get("peak_guard") is False and "cached_probe" not in done2)
    V3.uninstall_v03_patches(T, tt)
    os.environ[V3.ENV_OFFLOAD] = "1"
    try:
        T2 = types.SimpleNamespace(stage_probe=lambda a, l: None, attach_lora=lambda m: None)
        d3 = V3.install_v03_patches(T2, torch_module=tt, log=logs.append)
        check("PELIAS_V03_OFFLOAD_XL=1 turns the offload on", d3.get("offload_xl") is True)
    finally:
        del os.environ[V3.ENV_OFFLOAD]
        V3.uninstall_v03_patches(T2, tt)
    T3 = types.SimpleNamespace()
    d4 = V3.install_v03_patches(T3, torch_module=tt, log=logs.append, offload_xl=False)
    check("a module without stage_probe is tolerated", "cached_probe" not in d4)
    V3.uninstall_v03_patches(T3, tt)


# ======================================================================================================================================================
# 4. the shim
# ======================================================================================================================================================
def load_shim():
    spec = importlib.util.spec_from_file_location("shim03", ROOT / "stuart_pelias0.3.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["shim03"] = m
    spec.loader.exec_module(m)
    return m


def test_shim():
    S = load_shim()
    rest, off, no = S.split_flags(["smoke", "--v03-offload-xl", "--grad-accum", "4", "--v03-off"])
    check("split_flags removes only the two v0.3 flags", rest == ["smoke", "--grad-accum", "4"] and off and no)
    rest, off, no = S.split_flags(["train"])
    check("no flags: nothing changes", rest == ["train"] and not off and not no)
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "stuart_pelias0.2.py"
        fake.write_text("calls = []\n\ndef install_patches(T, args, run_dir, gpu_index, out):\n    calls.append(('orig', T))\n\n"
                        "def main(argv=None):\n    calls.append(('main', list(argv)))\n    install_patches('T', None, None, 0, type('O', (), {'log': staticmethod(lambda m: calls.append(('log', m)))})())\n    return 0\n", encoding="utf-8")
        os.environ["STUART_PELIAS02"] = str(fake)
        try:
            check("find_launcher honours STUART_PELIAS02", S.find_launcher() == fake)
            rec = {}
            V3_inst = V3.install_v03_patches
            V3.install_v03_patches = lambda T, **kw: rec.update(T=T, **kw) or {}
            try:
                rc = S.main(["train", "--v03-offload-xl", "--epochs", "1"])
            finally:
                V3.install_v03_patches = V3_inst
            mod = sys.modules["stuart_pelias0_2"]
            check("shim: the 0.2 main() got the command line without the v0.3 flags", ("main", ["train", "--epochs", "1"]) in mod.calls and rc == 0)
            check("shim: the original install_patches ran first, then the v0.3 patches with offload on", mod.calls[1][0] == "orig" and rec.get("offload_xl") is True and rec.get("T") == "T")
            rec.clear()
            V3.install_v03_patches = lambda T, **kw: rec.update(T=T, **kw) or {}
            try:
                S.main(["train"])
            finally:
                V3.install_v03_patches = V3_inst
            check("shim: without the flag the environment variable decides (offload_xl=None is passed)", rec.get("offload_xl", "missing") is None)
            rec.clear()
            V3.install_v03_patches = lambda T, **kw: rec.update(T=T, **kw) or {}
            try:
                S.main(["train", "--v03-off"])
            finally:
                V3.install_v03_patches = V3_inst
            check("shim: --v03-off installs nothing", rec == {})
        finally:
            del os.environ["STUART_PELIAS02"]
    os.environ["STUART_PELIAS02"] = "/nonexistent/stuart_pelias0.2.py"
    try:
        got = S.find_launcher()
        ok = got.is_file() and got.name == "stuart_pelias0.2.py"           # one of the standard places exists on this machine
    except SystemExit as e:
        ok = "cannot find stuart_pelias0.2.py" in str(e) and "STUART_PELIAS02" in str(e)
    finally:
        del os.environ["STUART_PELIAS02"]
    check("find_launcher skips a missing override, tries the standard places, and fails with a clear message when none exists", ok)


def main() -> int:
    for fn in (test_peak_guard, test_probe, test_offload, test_install_offload, test_install_all, test_shim):
        try:
            fn()
        except Exception as e:                                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} ran without an exception", False, repr(e))
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
