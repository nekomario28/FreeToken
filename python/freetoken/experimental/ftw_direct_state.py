"""Experimental destination-aware dense FTW state materializer.

The caller supplies the *complete* loader contract as ``name -> DenseExpectedSpec``.  This
keeps model-specific discovery (reflective keys, loader-only keys, optional extras) separate
from the storage/transfer primitive.  The materializer requires exact key and shape equality,
loads each FTW tensor with the registered-window direct-runtime H2D path, and performs any
required dtype conversion on the GPU.

This module is deliberately opt-in and is not wired into the canonical engine loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from freetoken.checkpoint.mapped_ftw_core import load_ftw_index
from freetoken.experimental.ftw_registered_dense import (
    DEFAULT_WINDOW_BYTES,
    RegisteredDenseTransferReceipt,
    copy_ftw_dense_registered_windows,
)


@dataclass(frozen=True)
class DenseExpectedSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class DirectDenseStateReceipt:
    tensor_count: int
    source_bytes: int
    final_bytes: int
    dtype_cast_count: int
    max_source_tensor_bytes: int
    max_cast_source_bytes: int
    max_cast_final_bytes: int
    window_bytes: int
    transfer_path: str = "file_backed_registered_window_direct_runtime_h2d"
    cast_path: str = "gpu_only"


def _weights(index: dict) -> dict[str, dict]:
    rows = index.get("tensors")
    if not isinstance(rows, list):
        raise ValueError("FTW index is missing tensors list")
    result: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") != "weight":
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("FTW dense index has invalid or duplicate weight name")
        result[name] = row
    return result


def _source_dtype(row: dict) -> torch.dtype:
    raw = row.get("dtype")
    if not isinstance(raw, str) or not hasattr(torch, raw):
        raise ValueError("FTW dense entry has unsupported dtype")
    value = getattr(torch, raw)
    if not isinstance(value, torch.dtype):
        raise ValueError("FTW dense entry dtype does not resolve to torch.dtype")
    return value


def _shape(row: dict) -> tuple[int, ...]:
    raw = row.get("shape")
    if not isinstance(raw, list):
        raise ValueError("FTW dense entry shape must be a list")
    result = tuple(int(dim) for dim in raw)
    if any(dim < 0 for dim in result):
        raise ValueError("FTW dense entry shape dimensions must be non-negative")
    return result


def materialize_ftw_dense_state_direct(
    path: str | Path,
    expected: Mapping[str, DenseExpectedSpec],
    *,
    device: str | torch.device,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> tuple[dict[str, torch.Tensor], DirectDenseStateReceipt]:
    """Materialize the complete dense FTW loader state directly onto ``device``.

    Source-dtype bytes travel FTW -> file-backed mmap -> registered window -> final-or-temporary
    GPU tensor via the direct runtime H2D primitive.  If the loader contract requires a dtype
    change, conversion happens GPU->GPU and the source-dtype temporary is released immediately.
    No tensor-sized anonymous CPU materialization is introduced by this function.
    """

    if not expected:
        raise ValueError("expected dense loader contract must not be empty")
    index = load_ftw_index(path)
    weights = _weights(index)
    expected_keys = set(expected)
    weight_keys = set(weights)
    if weight_keys != expected_keys:
        missing = len(expected_keys - weight_keys)
        unexpected = len(weight_keys - expected_keys)
        raise ValueError(
            f"FTW dense key contract mismatch: missing={missing}, unexpected={unexpected}"
        )

    state: dict[str, torch.Tensor] = {}
    source_bytes = 0
    final_bytes = 0
    dtype_cast_count = 0
    max_source_tensor_bytes = 0
    max_cast_source_bytes = 0
    max_cast_final_bytes = 0

    for row in index["tensors"]:
        if not isinstance(row, dict) or row.get("kind") != "weight":
            continue
        name = row["name"]
        spec = expected[name]
        source_shape = _shape(row)
        if source_shape != tuple(spec.shape):
            raise ValueError("FTW dense shape does not match expected loader contract")
        source_dtype = _source_dtype(row)
        nbytes = int(row.get("nbytes", -1))
        if nbytes < 0:
            raise ValueError("FTW dense entry has invalid nbytes")
        expected_source_bytes = torch.empty((), dtype=source_dtype).element_size()
        for dim in source_shape:
            expected_source_bytes *= dim
        if nbytes != expected_source_bytes:
            raise ValueError("FTW dense nbytes disagrees with shape/dtype")

        loaded, transfer_receipt = copy_ftw_dense_registered_windows(
            path,
            name,
            device=device,
            window_bytes=window_bytes,
        )
        if transfer_receipt.gpu_copy_path != "direct_runtime_h2d":
            raise RuntimeError("direct dense state materializer requires direct runtime H2D")

        source_bytes += nbytes
        max_source_tensor_bytes = max(max_source_tensor_bytes, nbytes)
        final_nbytes = loaded.numel() * torch.empty((), dtype=spec.dtype).element_size()
        if loaded.dtype != spec.dtype:
            dtype_cast_count += 1
            max_cast_source_bytes = max(max_cast_source_bytes, nbytes)
            max_cast_final_bytes = max(max_cast_final_bytes, final_nbytes)
            converted = loaded.to(dtype=spec.dtype)
            if converted.device.type == "cuda":
                torch.cuda.synchronize(device=converted.device)
            loaded = converted
        if tuple(loaded.shape) != tuple(spec.shape) or loaded.dtype != spec.dtype:
            raise RuntimeError("materialized FTW tensor does not satisfy expected contract")
        state[name] = loaded
        final_bytes += final_nbytes

    return state, DirectDenseStateReceipt(
        tensor_count=len(state),
        source_bytes=source_bytes,
        final_bytes=final_bytes,
        dtype_cast_count=dtype_cast_count,
        max_source_tensor_bytes=max_source_tensor_bytes,
        max_cast_source_bytes=max_cast_source_bytes,
        max_cast_final_bytes=max_cast_final_bytes,
        window_bytes=window_bytes,
    )


__all__ = [
    "DenseExpectedSpec",
    "DirectDenseStateReceipt",
    "materialize_ftw_dense_state_direct",
]
