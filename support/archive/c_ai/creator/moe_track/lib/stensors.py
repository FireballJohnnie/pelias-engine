#!/usr/bin/env python3
# File: <project>\moe_track\lib\stensors.py
"""Minimal .safetensors reader and writer that needs only numpy (no `safetensors` package, no torch).

Why it exists: the MTP merge script (mtp_align/11_merge_blk40_and_export.py) has to read a few tensors out of 70 GB of checkpoint shards and write a small new one,
and it must work in a plain Python install. bfloat16 is handled by bit manipulation (numpy has no bf16): it is read as uint16 and widened to float32, and written by
rounding float32 to nearest-even bf16.

    from stensors import SafeFile, save_tensors
    with SafeFile(path) as f:
        f.keys(); f.info("name") -> (dtype_str, shape); f.get("name") -> float32 / original-dtype ndarray; f.get_raw("name") -> bytes
    save_tensors(path, {"name": ndarray_float32}, dtypes={"name": "BF16"}, metadata={"format": "pt"})

File format (public, stable): 8 bytes little-endian header length N, then N bytes of JSON {name: {dtype, shape, data_offsets:[begin,end]}, "__metadata__": {...}},
then the raw data; offsets are relative to the end of the header.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

_NP = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
       "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64, "BOOL": np.bool_}
_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2, "U32": 4, "U64": 8, "BOOL": 1}
_NAME = {np.dtype(v): k for k, v in _NP.items()}


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """uint16 holding bfloat16 bits -> float32 (exact)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> uint16 holding bfloat16 bits, round to nearest even; NaN stays NaN."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    nan = np.isnan(x)
    rounding = ((u >> 16) & 1) + np.uint32(0x7FFF)
    out = ((u + rounding) >> 16).astype(np.uint16)
    out[nan] = 0x7FC0
    return out


class SafeFile:
    def __init__(self, path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        size = os.fstat(self._fh.fileno()).st_size
        if size < 8:
            raise ValueError(f"{self.path}: too small to be a safetensors file")
        (n,) = struct.unpack("<Q", self._fh.read(8))
        if n <= 0 or 8 + n > size:
            raise ValueError(f"{self.path}: bad header length {n}")
        self.header: Dict = json.loads(self._fh.read(n).decode("utf-8"))
        self.metadata = self.header.pop("__metadata__", {}) or {}
        self._base = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if size > 0 else None
        for k, v in self.header.items():
            b, e = v["data_offsets"]
            nel = int(np.prod(v["shape"], dtype=np.int64)) if v["shape"] else 1
            if e - b != nel * _SIZE[v["dtype"]] or self._base + e > size:
                raise ValueError(f"{self.path}: tensor {k} has inconsistent size/offsets")

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def keys(self):
        return list(self.header.keys())

    def info(self, name: str) -> Tuple[str, Tuple[int, ...]]:
        v = self.header[name]
        return v["dtype"], tuple(v["shape"])

    def get_raw(self, name: str) -> bytes:
        b, e = self.header[name]["data_offsets"]
        return self._mm[self._base + b: self._base + e]

    def get(self, name: str, as_float32: bool = True) -> np.ndarray:
        """The tensor as an ndarray. BF16 -> float32 (always); F16 -> float32 unless as_float32=False; other dtypes unchanged."""
        dt, shape = self.info(name)
        raw = self.get_raw(name)
        if dt == "BF16":
            return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape).copy()
        arr = np.frombuffer(raw, dtype=_NP[dt]).reshape(shape)
        if as_float32 and dt in ("F16", "F64"):
            return arr.astype(np.float32)
        return arr.copy()


def save_tensors(path, tensors: Dict[str, np.ndarray], dtypes: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, str]] = None,
                 overwrite: bool = False) -> None:
    """Write tensors (name-sorted for a stable file). dtypes[name] in {"BF16","F16","F32"} converts float arrays on the way out; default keeps the array's dtype.
    Refuses to overwrite an existing file unless overwrite=True. Writes to a temporary name and renames, so a crash never leaves a half file under the final name."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    dtypes = dtypes or {}
    hdr: Dict = {}
    blobs = []
    off = 0
    for name in sorted(tensors):
        a = np.asarray(tensors[name])
        want = dtypes.get(name)
        if want == "BF16":
            blob = f32_to_bf16(a.astype(np.float32)).tobytes()
            dt = "BF16"
        elif want in ("F16", "F32"):
            blob = np.ascontiguousarray(a.astype(_NP[want])).tobytes()
            dt = want
        else:
            if np.dtype(a.dtype) not in _NAME:
                raise TypeError(f"{name}: unsupported dtype {a.dtype}")
            dt = _NAME[np.dtype(a.dtype)]
            blob = np.ascontiguousarray(a).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(blob)]}
        blobs.append(blob)
        off += len(blob)
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)                                   # pad the header to 8 bytes like the reference implementation
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def read_index(dir_path) -> Dict[str, str]:
    """weight_map of a sharded checkpoint folder (model.safetensors.index.json); for a single-file checkpoint, builds it from the file."""
    d = Path(dir_path)
    ix = d / "model.safetensors.index.json"
    if ix.exists():
        return dict(json.loads(ix.read_text(encoding="utf-8"))["weight_map"])
    single = d / "model.safetensors"
    if single.exists():
        with SafeFile(single) as f:
            return {k: single.name for k in f.keys()}
    raise FileNotFoundError(f"no model.safetensors.index.json or model.safetensors in {d}")


def iter_selected(dir_path, predicate) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (name, float32 array) for every tensor of a (sharded) checkpoint whose name satisfies predicate(name); opens each shard once."""
    d = Path(dir_path)
    wm = read_index(d)
    by_shard: Dict[str, list] = {}
    for k, s in wm.items():
        if predicate(k):
            by_shard.setdefault(s, []).append(k)
    for shard in sorted(by_shard):
        with SafeFile(d / shard) as f:
            for k in sorted(by_shard[shard]):
                yield k, f.get(k)


def save_stream(path, entries, metadata: Optional[Dict[str, str]] = None, overwrite: bool = False) -> None:
    """Write a safetensors file without holding all tensors in memory. entries: list of (name, dtype_str, shape, nbytes, producer); producer() -> bytes of exactly nbytes.
    Order in the file = sorted by name (the producer for each entry is called once, in that order). Same safety rules as save_tensors (no overwrite, temp name + rename)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    ents = sorted(entries, key=lambda e: e[0])
    hdr: Dict = {}
    off = 0
    for name, dt, shape, nbytes, _ in ents:
        nel = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
        if dt not in _SIZE or nel * _SIZE[dt] != nbytes:
            raise ValueError(f"{name}: {dt} {tuple(shape)} does not occupy {nbytes} bytes")
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, dt, shape, nbytes, prod in ents:
            blob = prod()
            if len(blob) != nbytes:
                raise ValueError(f"{name}: producer returned {len(blob)} bytes, expected {nbytes}")
            f.write(blob)
    os.replace(tmp, path)


def raw_entry(sf: "SafeFile", name: str):
    """(name, dtype_str, shape, nbytes, producer) that copies a tensor's bytes unchanged from an open SafeFile (the file must stay open until save_stream returns)."""
    dt, shape = sf.info(name)
    b, e = sf.header[name]["data_offsets"]
    return (name, dt, shape, e - b, lambda: bytes(sf.get_raw(name)))
