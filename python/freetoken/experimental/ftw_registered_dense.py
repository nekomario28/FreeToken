"""Experimental file-backed FTW dense transfer with bounded host registration.

This module is intentionally not wired into the production FTW loader.  It maps one exact
single-shard ``kind=weight`` FTW entry privately, preallocates its final device tensor, then
registers/copies/unregisters bounded host windows sequentially.  The source stays file-backed;
there is no eager anonymous tensor-sized CPU materialization.

The first consumer is an exact-dtype synthetic integration gate.  Dtype conversion, async
pipeline overlap, cross-shard tensors, model loading, and default-loader integration are
explicitly out of scope until separately validated.
"""
from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path

import torch

from freetoken.checkpoint.mapped_ftw_core import (
    load_ftw_index,
    map_ftw_range_from_index,
    unique_entry,
)
from freetoken.kernel.pinned import host_register, host_unregister

DEFAULT_WINDOW_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class RegisteredDenseTransferReceipt:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    window_bytes: int
    windows: int
    source_storage: str = "file_backed_private_mmap"
    registration_lifetime: str = "one_window_register_copy_sync_unregister"


def _entry_dtype(entry: dict) -> torch.dtype:
    raw = entry.get("dtype")
    if not isinstance(raw, str) or not raw or not hasattr(torch, raw):
        raise ValueError("FTW dense entry has unsupported dtype")
    dtype = getattr(torch, raw)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("FTW dense entry dtype does not resolve to torch.dtype")
    return dtype


def _entry_shape(entry: dict) -> tuple[int, ...]:
    raw = entry.get("shape")
    if not isinstance(raw, list) or not raw:
        raise ValueError("FTW dense entry shape must be a non-empty list")
    shape = tuple(int(value) for value in raw)
    if any(value < 0 for value in shape):
        raise ValueError("FTW dense entry shape dimensions must be non-negative")
    return shape


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= dim
    return value


def _validate_window(window_bytes: int, itemsize: int) -> None:
    if window_bytes <= 0:
        raise ValueError("window_bytes must be positive")
    if window_bytes % mmap.PAGESIZE != 0:
        raise ValueError("window_bytes must be page aligned")
    if window_bytes % itemsize != 0:
        raise ValueError("window_bytes must be divisible by dtype itemsize")


def copy_ftw_dense_registered_windows(
    path: str | Path,
    name: str,
    *,
    device: str | torch.device,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> tuple[torch.Tensor, RegisteredDenseTransferReceipt]:
    """Copy one exact-dtype dense FTW entry into its final device tensor.

    Every host registration is balanced in ``finally`` and synchronized before unregister.
    Only a single-shard ``kind=weight`` entry is accepted.  This function performs no dtype
    conversion and does not alter the FTW checkpoint.
    """

    index = load_ftw_index(path)
    entry = unique_entry(index, name)
    if entry.get("kind") != "weight":
        raise ValueError("registered dense transfer accepts only kind=weight")

    dtype = _entry_dtype(entry)
    shape = _entry_shape(entry)
    itemsize = torch.empty((), dtype=dtype).element_size()
    numel = _numel(shape)
    expected_nbytes = numel * itemsize
    nbytes = int(entry.get("nbytes", -1))
    if expected_nbytes != nbytes:
        raise ValueError("FTW dense entry shape/dtype/nbytes disagree")
    _validate_window(window_bytes, itemsize)

    owner = map_ftw_range_from_index(path, index, name)
    source = None
    try:
        if owner.data_offset % mmap.PAGESIZE != 0:
            raise ValueError("FTW dense mapping data address is not page aligned")
        source = torch.frombuffer(
            owner.mapping,
            dtype=dtype,
            count=numel,
            offset=owner.data_offset,
        ).reshape(shape)
        if source.device.type != "cpu" or not source.is_contiguous():
            raise RuntimeError("FTW mapped dense source must be contiguous CPU storage")

        target = torch.empty(shape, dtype=dtype, device=device)
        source_flat = source.reshape(-1)
        target_flat = target.reshape(-1)
        elements_per_window = window_bytes // itemsize
        windows = 0
        with torch.no_grad():
            for start in range(0, numel, elements_per_window):
                end = min(numel, start + elements_per_window)
                source_window = source_flat[start:end]
                target_window = target_flat[start:end]
                addr = int(source_window.data_ptr())
                bytes_this_window = int(source_window.numel()) * itemsize
                if addr % mmap.PAGESIZE != 0:
                    raise RuntimeError("registered FTW source window address is not page aligned")
                registered = False
                try:
                    host_register(addr, bytes_this_window)
                    registered = True
                    target_window.copy_(source_window, non_blocking=False)
                    if target.device.type == "cuda":
                        torch.cuda.synchronize(device=target.device)
                finally:
                    if registered:
                        host_unregister(addr)
                windows += 1

        receipt = RegisteredDenseTransferReceipt(
            name=name,
            dtype=str(dtype).removeprefix("torch."),
            shape=shape,
            nbytes=nbytes,
            window_bytes=window_bytes,
            windows=windows,
        )
        return target, receipt
    finally:
        source = None
        owner.mapping.close()


__all__ = [
    "DEFAULT_WINDOW_BYTES",
    "RegisteredDenseTransferReceipt",
    "copy_ftw_dense_registered_windows",
]
