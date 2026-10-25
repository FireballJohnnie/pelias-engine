#!/usr/bin/env python3
# File: <project>\moe_track\lib\gguf_min.py   (WSL: <project>/moe_track/lib/gguf_min.py)
"""Dependency-free GGUF (v2/v3) header reader.

Why this exists: the project venv may not have the `gguf` package, and the tools in moe_track only need the
header (metadata + tensor table), never the tensor data. This reader opens the file, reads the header,
and stops. It never loads weights, so it is safe to run on the 20 GiB Q4_K_M file next to a running server.

Block sizes and type sizes below are checked against llama.cpp's own gguf-py table by
tests/test_gguf_min.py (any mismatch fails the test).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# ggml_type id -> (name, elements per block, bytes per block)
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}

# GGUF metadata value types
_T_U8, _T_I8, _T_U16, _T_I16, _T_U32, _T_I32, _T_F32, _T_BOOL, _T_STR, _T_ARR, _T_U64, _T_I64, _T_F64 = range(13)
_SCALAR_FMT = {
    _T_U8: "<B", _T_I8: "<b", _T_U16: "<H", _T_I16: "<h", _T_U32: "<I", _T_I32: "<i",
    _T_F32: "<f", _T_BOOL: "<?", _T_U64: "<Q", _T_I64: "<q", _T_F64: "<d",
}
_SCALAR_SIZE = {k: struct.calcsize(v) for k, v in _SCALAR_FMT.items()}


class GGUFError(RuntimeError):
    pass


@dataclass
class TensorInfo:
    name: str
    dims: Tuple[int, ...]          # ggml order: dims[0] is the fastest-moving (row length)
    ggml_type: int
    offset: int                    # relative to the start of the tensor data block
    abs_offset: int = 0            # file offset of the first byte
    type_name: str = ""
    n_elements: int = 0
    n_bytes: int = 0

    def per_slice_bytes(self, n_slices: int) -> int:
        """Bytes of one slice along the last dimension (an expert, for a [in, out, n_expert] tensor)."""
        if n_slices <= 0 or self.n_bytes % n_slices:
            raise GGUFError(f"{self.name}: {self.n_bytes} bytes is not divisible by {n_slices} slices")
        return self.n_bytes // n_slices


@dataclass
class GGUFFile:
    path: str
    version: int
    alignment: int
    data_offset: int
    file_size: int
    kv: Dict[str, Any] = field(default_factory=dict)
    tensors: List[TensorInfo] = field(default_factory=list)

    def by_name(self) -> Dict[str, TensorInfo]:
        return {t.name: t for t in self.tensors}

    def get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    def arch_get(self, suffix: str, default: Any = None) -> Any:
        arch = self.kv.get("general.architecture")
        return self.kv.get(f"{arch}.{suffix}", default) if arch else default


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError(f"unexpected end of file (wanted {n} bytes, got {len(b)})")
    return b


def _read_str(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", _read_exact(f, 8))
    if n > (1 << 28):
        raise GGUFError(f"implausible string length {n}")
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_str(f: BinaryIO) -> None:
    (n,) = struct.unpack("<Q", _read_exact(f, 8))
    f.seek(n, 1)


def _read_value(f: BinaryIO, vtype: int, keep_arrays_up_to: int) -> Any:
    if vtype in _SCALAR_FMT:
        return struct.unpack(_SCALAR_FMT[vtype], _read_exact(f, _SCALAR_SIZE[vtype]))[0]
    if vtype == _T_STR:
        return _read_str(f)
    if vtype == _T_ARR:
        (etype,) = struct.unpack("<I", _read_exact(f, 4))
        (count,) = struct.unpack("<Q", _read_exact(f, 8))
        keep = count <= keep_arrays_up_to
        if etype in _SCALAR_FMT:
            size = _SCALAR_SIZE[etype]
            if keep:
                raw = _read_exact(f, size * count)
                code = _SCALAR_FMT[etype][1]
                return list(struct.unpack(f"<{count}{code}", raw))
            f.seek(size * count, 1)
            return {"array_of": etype, "len": count}
        if etype == _T_STR:
            if keep:
                return [_read_str(f) for _ in range(count)]
            for _ in range(count):
                _skip_str(f)
            return {"array_of": etype, "len": count}
        # nested arrays are not used by llama.cpp models; parse generically
        out = [_read_value(f, etype, keep_arrays_up_to) for _ in range(count)]
        return out if keep else {"array_of": etype, "len": count}
    raise GGUFError(f"unknown GGUF metadata value type {vtype}")


def read_gguf(path: str | Path, keep_arrays_up_to: int = 256) -> GGUFFile:
    """Read header + tensor table. Large metadata arrays (tokenizer vocab) are skipped, not stored."""
    p = Path(path)
    file_size = p.stat().st_size
    with open(p, "rb", buffering=1 << 20) as f:
        if _read_exact(f, 4) != GGUF_MAGIC:
            raise GGUFError(f"{p} is not a GGUF file (bad magic)")
        (version,) = struct.unpack("<I", _read_exact(f, 4))
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack("<QQ", _read_exact(f, 16))
        if n_tensors > (1 << 24) or n_kv > (1 << 24):
            raise GGUFError(f"implausible counts: {n_tensors} tensors, {n_kv} kv pairs")

        kv: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", _read_exact(f, 4))
            kv[key] = _read_value(f, vtype, keep_arrays_up_to)

        tensors: List[TensorInfo] = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
            if not 1 <= n_dims <= 8:
                raise GGUFError(f"{name}: implausible n_dims {n_dims}")
            dims = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
            (gtype,) = struct.unpack("<I", _read_exact(f, 4))
            (offset,) = struct.unpack("<Q", _read_exact(f, 8))
            if gtype not in GGML_TYPES:
                raise GGUFError(f"{name}: unknown ggml type id {gtype} (extend GGML_TYPES)")
            tname, bs, ts = GGML_TYPES[gtype]
            n_el = 1
            for d in dims:
                n_el *= d
            if n_el % bs:
                raise GGUFError(f"{name}: {n_el} elements is not a multiple of block size {bs}")
            tensors.append(TensorInfo(name=name, dims=tuple(dims), ggml_type=gtype, offset=offset,
                                      type_name=tname, n_elements=n_el, n_bytes=n_el // bs * ts))

        alignment = int(kv.get("general.alignment", DEFAULT_ALIGNMENT)) or DEFAULT_ALIGNMENT
        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment

    for t in tensors:
        t.abs_offset = data_offset + t.offset
    return GGUFFile(path=str(p), version=version, alignment=alignment, data_offset=data_offset,
                    file_size=file_size, kv=kv, tensors=tensors)
