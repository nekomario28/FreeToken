"""Experimental file-backed FTW dense transfer with bounded host registration.

This module is intentionally not wired into the production FTW loader. It maps one exact
single-shard ``kind=weight`` FTW entry privately and copies bounded registered host windows
into final device storage. The source stays file-backed; there is no eager anonymous
CPU tensor-sized materialization.

For CUDA/HIP targets the experimental path bypasses ``Tensor.copy_`` and calls the runtime
copy primitive directly. This isolates framework-owned host staging from the storage/transfer
contract while keeping the final destination a normal torch tensor.
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
from freetoken.kernel.pinned import (
    host_register_transfer,
    host_unregister,
    registered_host_to_device_copy,
)

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
    registration_lifetime: str = "one_window_default_register_direct_copy_unregister"
    destination_storage: str = "preallocated_final_tensor"
    gpu_copy_path: str = "direct_runtime_h2d"


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
    if not isinstance(raw, list):
        raise ValueError("FTW dense entry shape must be a list")
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


def _validated_dense_entry(path: str | Path, name: str):
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
    return index, entry, dtype, shape, itemsize, numel, nbytes


def copy_ftw_dense_registered_windows_into(
    path: str | Path,
    name: str,
    target: torch.Tensor,
    *,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> RegisteredDenseTransferReceipt:
    """Copy one exact-dtype dense FTW entry into preallocated final storage.

    Every host registration is balanced in ``finally``. GPU targets use a synchronous direct
    CUDA/HIP runtime H2D copy so registered source pages do not pass through PyTorch's generic
    host-copy path. CPU targets retain ``Tensor.copy_`` for dependency-free structural tests.
    """

    index, _entry, dtype, shape, itemsize, numel, nbytes = _validated_dense_entry(path, name)
    _validate_window(window_bytes, itemsize)
    if target.dtype != dtype:
        raise ValueError("registered dense target dtype must exactly match FTW dtype")
    if tuple(target.shape) != shape:
        raise ValueError("registered dense target shape must exactly match FTW shape")
    if not target.is_contiguous():
        raise ValueError("registered dense target must be contiguous")

    owner = map_ftw_range_from_index(path, index, name)
    source = None
    source_flat = None
    source_window = None
    target_flat = None
    target_window = None
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

        source_flat = source.reshape(-1)
        target_flat = target.reshape(-1)
        elements_per_window = window_bytes // itemsize
        windows = 0
        with torch.no_grad():
            for start in range(0, numel, elements_per_window):
                end = min(numel, start + elements_per_window)
                source_window = source_flat[start:end]
                target_window = target_flat[start:end]
                src_addr = int(source_window.data_ptr())
                bytes_this_window = int(source_window.numel()) * itemsize
                if src_addr % mmap.PAGESIZE != 0:
                    raise RuntimeError("registered FTW source window address is not page aligned")
                registered = False
                try:
                    host_register_transfer(src_addr, bytes_this_window)
                    registered = True
                    if target.device.type == "cuda":
                        registered_host_to_device_copy(
                            int(target_window.data_ptr()), src_addr, bytes_this_window
                        )
                    else:
                        target_window.copy_(source_window)
                finally:
                    if registered:
                        host_unregister(src_addr)
                windows += 1

        return RegisteredDenseTransferReceipt(
            name=name,
            dtype=str(dtype).removeprefix("torch."),
            shape=shape,
            nbytes=nbytes,
            window_bytes=window_bytes,
            windows=windows,
            gpu_copy_path="direct_runtime_h2d" if target.device.type == "cuda" else "torch_cpu_copy",
        )
    finally:
        target_window = None
        source_window = None
        target_flat = None
        source_flat = None
        source = None
        owner.mapping.close()


def copy_ftw_dense_registered_windows(
    path: str | Path,
    name: str,
    *,
    device: str | torch.device,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> tuple[torch.Tensor, RegisteredDenseTransferReceipt]:
    """Allocate final storage, then copy one exact-dtype dense FTW entry into it."""

    _index, _entry, dtype, shape, itemsize, _numel_value, _nbytes = _validated_dense_entry(path, name)
    _validate_window(window_bytes, itemsize)
    target = torch.empty(shape, dtype=dtype, device=device)
    receipt = copy_ftw_dense_registered_windows_into(
        path, name, target, window_bytes=window_bytes
    )
    return target, receipt


__all__ = [
    "DEFAULT_WINDOW_BYTES",
    "RegisteredDenseTransferReceipt",
    "copy_ftw_dense_registered_windows",
    "copy_ftw_dense_registered_windows_into",
]
