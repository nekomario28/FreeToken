"""Bounded raw safetensors -> FTW passthrough primitive for low-RAM experiments.

This module does *not* change canonical checkpoint conversion.  It exists to prove a narrow
primitive needed by the experimental Qwen3.5 low-RAM path: when a tensor is known from the
model loader contract to be byte-for-byte passthrough, copy its safetensors payload to FTW in
bounded chunks instead of materialising the complete tensor through ``safe_open().get_tensor``.

The caller remains responsible for proving passthrough eligibility.  This primitive performs
no rename/fusion/quantisation/model logic; it only validates one safetensors entry and preserves
its raw bytes, dtype and shape in the FTW index.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

from freetoken.checkpoint.ftw import ALIGN, FTWWriter, _align_up

_DEFAULT_CHUNK = 8 << 20
_MAX_HEADER = 64 << 20
_ST_TO_FTW_DTYPE = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "U16": "uint16",
    "I16": "int16",
    "F16": "float16",
    "BF16": "bfloat16",
    "U32": "uint32",
    "I32": "int32",
    "F32": "float32",
    "U64": "uint64",
    "I64": "int64",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
}


def _tensor_entry(path: str | os.PathLike[str], name: str) -> tuple[int, int, str, list[int]]:
    """Return absolute file offset, nbytes, FTW dtype string and shape for one entry."""
    p = Path(path)
    with p.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensors header prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen < 2 or hlen > _MAX_HEADER:
            raise ValueError("invalid safetensors header length")
        hraw = f.read(hlen)
        if len(hraw) != hlen:
            raise ValueError("short safetensors header")
    try:
        header = json.loads(hraw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid safetensors header JSON") from exc
    spec = header.get(name) if isinstance(header, dict) else None
    if not isinstance(spec, dict):
        raise KeyError(f"safetensors tensor not found: {name}")
    offsets = spec.get("data_offsets")
    shape = spec.get("shape")
    dtype = str(spec.get("dtype") or "")
    if dtype not in _ST_TO_FTW_DTYPE:
        raise ValueError(f"unsupported passthrough dtype: {dtype!r}")
    if not (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(isinstance(v, int) and v >= 0 for v in offsets)
        and offsets[1] >= offsets[0]
    ):
        raise ValueError("invalid safetensors data_offsets")
    if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
        raise ValueError("invalid safetensors shape")
    start, end = int(offsets[0]), int(offsets[1])
    return 8 + int(hlen) + start, end - start, _ST_TO_FTW_DTYPE[dtype], list(shape)


class BoundedPassthroughFTWWriter(FTWWriter):
    """Experimental FTWWriter with an exact raw-file-range streaming operation.

    Only one bounded byte buffer is alive per read.  The method uses the canonical writer's
    shard/alignment machinery and records the same FTW tensor metadata as ``add_tensor``.
    """

    def add_safetensors_passthrough(
        self,
        *,
        name: str,
        safetensors_path: str | os.PathLike[str],
        safetensors_name: str,
        kind: str = "weight",
        chunk_bytes: int = _DEFAULT_CHUNK,
    ) -> dict[str, Any]:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        file_off, nbytes, dtype, shape = _tensor_entry(safetensors_path, safetensors_name)

        # Match FTWWriter.add_tensor: a tensor that can fit one shard rolls early rather than
        # splitting merely because the current shard is nearly full.
        if self._f is None or (
            nbytes <= self.shard_limit and self._cur + nbytes > self.shard_limit
        ):
            self._roll()
        global_off = self._global
        if global_off % ALIGN:
            raise RuntimeError("FTW tensor start is not aligned")

        fd = os.open(os.fspath(safetensors_path), os.O_RDONLY)
        copied = 0
        max_chunk = 0
        try:
            while copied < nbytes:
                want = min(chunk_bytes, nbytes - copied)
                data = os.pread(fd, want, file_off + copied)
                if len(data) != want:
                    raise OSError(
                        f"short safetensors payload read: {copied + len(data)}/{nbytes} bytes"
                    )
                self._write_raw(memoryview(data))
                copied += want
                max_chunk = max(max_chunk, want)
                try:
                    os.posix_fadvise(fd, file_off + copied - want, want, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        finally:
            os.close(fd)

        entry = {
            "name": name,
            "kind": kind,
            "dtype": dtype,
            "shape": shape,
            "global_off": global_off,
            "nbytes": nbytes,
        }
        self._tensors.append(entry)
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))
        return {
            "entry": dict(entry),
            "payload_bytes": nbytes,
            "max_read_buffer_bytes": max_chunk,
        }


__all__ = ["BoundedPassthroughFTWWriter"]
