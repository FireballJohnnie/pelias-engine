#!/usr/bin/env python3
# File: <project>\moe_track\tests\fake_torch.py
"""A small numpy-backed stand-in for the part of PyTorch that mtp_align/mtp_torch.py uses, so its FORWARD math can be tested on a machine without torch.

    import fake_torch; fake_torch.install()      # puts fake modules named torch, torch.nn, torch.nn.functional, torch.utils.checkpoint into sys.modules

What it is: forward only (no autograd, no GPU, bfloat16 behaves like float32). What it is not: a test of gradients, devices, dtypes or speed. Those only get tested by the
real thing: `python 10_train_blk40_mtp_lora.py selftest` on your machine.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np


class _DType:
    def __init__(self, name, np_dtype):
        self.name, self.np = name, np.dtype(np_dtype)

    def __repr__(self):
        return f"torch.{self.name}"


float32 = _DType("float32", np.float32)
float64 = _DType("float64", np.float64)
bfloat16 = _DType("bfloat16", np.float32)       # stored as float32
float16 = _DType("float16", np.float16)
int64 = _DType("int64", np.int64)
long = int64
bool_ = _DType("bool", np.bool_)


def _dt(arr_dtype) -> _DType:
    for d in (float32, float64, float16, int64, bool_):
        if np.dtype(arr_dtype) == d.np:
            return d
    if np.dtype(arr_dtype) == np.int32:
        return int64
    raise TypeError(arr_dtype)


def _unwrap(x):
    return x.data if isinstance(x, Tensor) else x


class Tensor:
    def __init__(self, data, dtype: _DType | None = None):
        self.data = np.asarray(data)
        self._dtype = dtype or _dt(self.data.dtype)
        self.requires_grad = False
        self.device = "cpu"

    # basic ----------------------------------------------------------------------------------------------------------------------------------------------------
    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self._dtype

    def size(self, d=None):
        return self.shape if d is None else self.shape[d]

    def numel(self):
        return int(self.data.size)

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data.tolist()

    def item(self):
        return self.data.item()

    def float(self):
        return Tensor(self.data.astype(np.float32), float32)

    def to(self, *a, **k):
        dt = None
        for x in list(a) + list(k.values()):
            if isinstance(x, _DType):
                dt = x
            elif isinstance(x, Tensor):
                dt = x.dtype
        if dt is None:
            return self
        return Tensor(self.data.astype(dt.np), dt)

    def cpu(self):
        return self

    def detach(self):
        return Tensor(self.data, self._dtype)

    def clone(self):
        return Tensor(self.data.copy(), self._dtype)

    def contiguous(self):
        return self

    def copy_(self, other):
        self.data[...] = _unwrap(other)
        return self

    # shape ops ------------------------------------------------------------------------------------------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self.data.reshape(shape), self._dtype)

    view = reshape

    def permute(self, *dims):
        return Tensor(np.transpose(self.data, dims), self._dtype)

    def transpose(self, a, b):
        return Tensor(np.swapaxes(self.data, a, b), self._dtype)

    def unsqueeze(self, d):
        return Tensor(np.expand_dims(self.data, d), self._dtype)

    def squeeze(self, d):
        return Tensor(np.squeeze(self.data, axis=d), self._dtype)

    def repeat_interleave(self, n, dim):
        return Tensor(np.repeat(self.data, n, axis=dim), self._dtype)

    def __getitem__(self, idx):
        def conv(i):
            if isinstance(i, Tensor):
                return i.data
            if isinstance(i, tuple):
                return tuple(conv(j) for j in i)
            return i
        return Tensor(self.data[conv(idx)], self._dtype)

    def __setitem__(self, idx, v):
        self.data[idx] = _unwrap(v)

    def __len__(self):
        return self.shape[0]

    # math -----------------------------------------------------------------------------------------------------------------------------------------------------
    def _bin(self, o, f, rev=False):
        a, b = self.data, _unwrap(o)
        r = np.asarray(f(b, a) if rev else f(a, b))
        return Tensor(r)

    def __add__(self, o): return self._bin(o, np.add)
    def __radd__(self, o): return self._bin(o, np.add, True)
    def __sub__(self, o): return self._bin(o, np.subtract)
    def __rsub__(self, o): return self._bin(o, np.subtract, True)
    def __mul__(self, o): return self._bin(o, np.multiply)
    def __rmul__(self, o): return self._bin(o, np.multiply, True)
    def __truediv__(self, o): return self._bin(o, np.true_divide)
    def __rtruediv__(self, o): return self._bin(o, np.true_divide, True)
    def __floordiv__(self, o): return self._bin(o, np.floor_divide)
    def __pow__(self, o): return self._bin(o, np.power)
    def __rpow__(self, o): return self._bin(o, np.power, True)
    def __gt__(self, o): return self._bin(o, np.greater)
    def __eq__(self, o): return self._bin(o, np.equal)
    __hash__ = object.__hash__
    def __neg__(self): return Tensor(-self.data, self._dtype)

    def sum(self, dim=None, keepdim=False):
        return Tensor(np.sum(self.data, axis=dim, keepdims=keepdim), self._dtype)

    def mean(self, dim=None, keepdim=False):
        return Tensor(np.mean(self.data, axis=dim, keepdims=keepdim), self._dtype)

    def cos(self): return Tensor(np.cos(self.data), self._dtype)
    def sin(self): return Tensor(np.sin(self.data), self._dtype)
    def exp(self): return Tensor(np.exp(self.data), self._dtype)

    def argmax(self, dim):
        return Tensor(np.argmax(self.data, axis=dim), int64)

    def topk(self, k, dim=-1):
        idx = np.argsort(-self.data, axis=dim, kind="stable")
        idx = np.take(idx, np.arange(k), axis=dim)
        return Tensor(np.take_along_axis(self.data, idx, axis=dim), self._dtype), Tensor(idx, int64)

    def masked_fill(self, mask, value):
        return Tensor(np.where(mask.data, np.asarray(value, dtype=self.data.dtype), self.data), self._dtype)

    def gather(self, dim, index):
        return Tensor(np.take_along_axis(self.data, index.data, axis=dim), self._dtype)

    def index_add_(self, dim, index, source):
        assert dim == 0
        np.add.at(self.data, index.data, source.data)
        return self

    def matmul(self, o):
        return Tensor(np.matmul(self.data, _unwrap(o)), self._dtype)


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        t = data if isinstance(data, Tensor) else Tensor(data)
        super().__init__(t.data, t.dtype)
        self.requires_grad = requires_grad


# ---- functions --------------------------------------------------------------------------------------------------------------------------------------------
def tensor(x, dtype=None, device=None):
    a = np.asarray(x)
    return Tensor(a.astype(dtype.np) if dtype else a, dtype)


def from_numpy(a): return Tensor(a)
def zeros(*shape, dtype=float32, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=dtype.np), dtype)
def zeros_like(t): return Tensor(np.zeros_like(t.data), t.dtype)
def ones(*shape, device=None, dtype=float32): return Tensor(np.ones(shape, dtype=dtype.np), dtype)
def arange(*args, device=None, dtype=int64): return Tensor(np.arange(*args, dtype=dtype.np), dtype)
def cat(ts, dim=0): return Tensor(np.concatenate([t.data for t in ts], axis=dim), ts[0].dtype)
def stack(ts, dim=0): return Tensor(np.stack([t.data for t in ts], axis=dim), ts[0].dtype)
def rsqrt(x): return Tensor(1.0 / np.sqrt(x.data), x.dtype)
def sigmoid(x): return Tensor((1.0 / (1.0 + np.exp(-x.data))).astype(x.data.dtype), x.dtype)
def softmax(x, dim=-1):
    m = np.max(x.data, axis=dim, keepdims=True)
    e = np.exp(x.data - m)
    return Tensor((e / e.sum(axis=dim, keepdims=True)).astype(x.data.dtype), x.dtype)
def log_softmax(x, dim=-1):
    m = np.max(x.data, axis=dim, keepdims=True)
    z = x.data - m
    return Tensor((z - np.log(np.exp(z).sum(axis=dim, keepdims=True))).astype(x.data.dtype), x.dtype)
def einsum(eq, *ts): return Tensor(np.einsum(eq, *[t.data for t in ts]).astype(ts[0].data.dtype), ts[0].dtype)
def triu(x, diagonal=0): return Tensor(np.triu(x.data, k=diagonal), x.dtype)
def minimum(a, b): return Tensor(np.minimum(a.data, b.data), a.dtype)
def argsort(x, stable=False): return Tensor(np.argsort(x.data, kind="stable"), int64)
def bincount(x, minlength=0): return Tensor(np.bincount(x.data, minlength=minlength), int64)
def matmul(a, b): return Tensor(np.matmul(a.data, b.data), a.dtype)


class Generator:
    def __init__(self): self.seed = 0
    def manual_seed(self, s): self.seed = int(s); return self


def rand(*shape, generator=None):
    rng = np.random.default_rng(generator.seed if generator else None)
    return Tensor(rng.random(shape).astype(np.float32), float32)


class no_grad:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---- nn ---------------------------------------------------------------------------------------------------------------------------------------------------
class Module:
    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_mods", {})

    def __setattr__(self, k, v):
        if isinstance(v, Parameter):
            self._params[k] = v
        elif isinstance(v, Module):
            self._mods[k] = v
        object.__setattr__(self, k, v)

    def named_parameters(self, prefix=""):
        for k, p in self._params.items():
            yield prefix + k, p
        for k, m in self._mods.items():
            yield from m.named_parameters(prefix + k + ".")

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def to(self, *a, **k):
        return self

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class ModuleDict(Module):
    def __setitem__(self, k, m):
        self._mods[k] = m
        object.__setattr__(self, "_item_" + k, m)

    def __getitem__(self, k): return self._mods[k]
    def items(self): return self._mods.items()


def linear(x, w, b=None): return Tensor(np.matmul(x.data, w.data.T).astype(x.data.dtype), x.dtype)
def silu(x): return Tensor((x.data / (1.0 + np.exp(-x.data))).astype(x.data.dtype), x.dtype)


def scaled_dot_product_attention(q, k, v, is_causal=False):
    d = q.shape[-1]
    s = np.matmul(q.data, np.swapaxes(k.data, -1, -2)) / np.sqrt(d)
    if is_causal:
        T = s.shape[-1]
        s = s + np.triu(np.full((T, T), -np.inf, dtype=s.dtype), k=1)
    m = np.max(s, axis=-1, keepdims=True)
    e = np.exp(s - m)
    w = e / e.sum(-1, keepdims=True)
    return Tensor(np.matmul(w, v.data).astype(q.data.dtype), q.dtype)


def checkpoint(fn, *args, use_reentrant=False, **k):
    return fn(*args, **k)


def install():
    me = sys.modules[__name__]
    torch = types.ModuleType("torch")
    for name in ("Tensor", "tensor", "from_numpy", "zeros", "zeros_like", "ones", "arange", "cat", "stack", "rsqrt", "sigmoid", "softmax", "einsum", "triu",
                 "minimum", "argsort", "bincount", "matmul", "Generator", "rand", "no_grad", "float32", "float64", "bfloat16", "float16", "int64", "long"):
        setattr(torch, name, getattr(me, name))
    torch.bool = bool_
    nn = types.ModuleType("torch.nn")
    nn.Module, nn.Parameter, nn.ModuleDict = Module, Parameter, ModuleDict
    F = types.ModuleType("torch.nn.functional")
    F.linear, F.silu, F.scaled_dot_product_attention, F.log_softmax, F.softmax = linear, silu, scaled_dot_product_attention, log_softmax, softmax
    nn.functional = F
    utils = types.ModuleType("torch.utils")
    ck = types.ModuleType("torch.utils.checkpoint")
    ck.checkpoint = checkpoint
    utils.checkpoint = ck
    torch.nn, torch.utils = nn, utils
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": F, "torch.utils": utils, "torch.utils.checkpoint": ck})
    return torch
