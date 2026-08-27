"""Bounded read-only H2D for file-backed PAGEABLE FTW expert banks.

The CPU decode mapping intentionally remains the existing reclaimable file-backed mapping.
For prefill, copying that PAGEABLE tensor with ``Tensor.copy_`` lets the runtime introduce a
large anonymous host staging allocation.  This helper instead opens a temporary read-only view
of the same exact FTW range and transfers it in bounded registered windows directly into an
already-allocated GPU cache tensor.

This module is experimental and opt-in through the file-backed PAGEABLE cache path only.
"""
from __future__ import annotations

from dataclasses import dataclass
import mmap
import warnings

import torch

from freetoken.experimental.ftw_registered_dense import DEFAULT_WINDOW_BYTES, _validate_window
from freetoken.kernel.pinned import (
    host_register_transfer,
    host_unregister,
    registered_host_to_device_copy,
)

_STORAGE_OWNER_ATTR = "_freetoken_mapped_ftw_storage"


@dataclass(frozen=True)
class PageablePrefillTransferReceipt:
    nbytes: int
    window_bytes: int
    windows: int
    source_storage: str = "file_backed_temporary_readonly_mmap"
    registration_lifetime: str = "one_window_register_direct_copy_unregister"
    destination_storage: str = "preallocated_gpu_cache"
    gpu_copy_path: str = "direct_runtime_h2d"


def is_mapped_ftw_tensor(source: torch.Tensor) -> bool:
    """Whether ``source`` retains the mapped-FTW storage owner used by this helper."""
    return getattr(source, _STORAGE_OWNER_ATTR, None) is not None


def copy_mapped_ftw_readonly_windows_into(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> PageablePrefillTransferReceipt:
    """Copy one mapped FTW bank into a preallocated GPU tensor with bounded host residency.

    The temporary transfer mapping is ACCESS_READ rather than the CPU decode mapping's
    MAP_PRIVATE/COW view.  This mirrors the validated direct dense-loader path: each window is
    registered only for the synchronous H2D copy, then immediately unregistered.  No
    tensor-sized anonymous staging buffer is created by this helper.
    """
    owner = getattr(source, _STORAGE_OWNER_ATTR, None)
    if owner is None:
        raise ValueError("source is not backed by mapped FTW storage")
    if source.device.type != "cpu":
        raise ValueError("mapped FTW prefill source must be a CPU tensor")
    if target.device.type != "cuda":
        raise ValueError("mapped FTW prefill destination must be a GPU tensor")
    if not source.is_contiguous() or not target.is_contiguous():
        raise ValueError("mapped FTW prefill source/destination must be contiguous")
    if source.dtype != target.dtype or tuple(source.shape) != tuple(target.shape):
        raise ValueError("mapped FTW prefill source/destination shape and dtype must match")

    itemsize = source.element_size()
    _validate_window(window_bytes, itemsize)
    nbytes = int(source.numel()) * itemsize
    if nbytes != int(owner.nbytes):
        raise ValueError("mapped FTW owner byte count does not match tensor")

    granularity = mmap.ALLOCATIONGRANULARITY
    file_offset = int(owner.file_offset)
    mapping_offset = file_offset - (file_offset % granularity)
    data_offset = file_offset - mapping_offset
    mapping_length = data_offset + nbytes

    source_view = source_flat = source_window = target_flat = target_window = None
    mapping = None
    with owner.shard_path.open("rb") as handle:
        mapping = mmap.mmap(
            handle.fileno(), mapping_length, access=mmap.ACCESS_READ, offset=mapping_offset
        )
    try:
        # Keep the same fail-closed alignment contract as the already-validated dense
        # registered-window loader.  The FTW converter aligns streamed expert entries, so a
        # violation indicates an artifact/layout change that needs an explicit new gate.
        if data_offset % mmap.PAGESIZE != 0:
            raise ValueError("mapped FTW expert transfer address is not page aligned")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable.*",
                category=UserWarning,
            )
            source_view = torch.frombuffer(
                mapping,
                dtype=source.dtype,
                count=source.numel(),
                offset=data_offset,
            ).reshape(tuple(source.shape))
        if source_view.device.type != "cpu" or not source_view.is_contiguous():
            raise RuntimeError("temporary FTW transfer view must be contiguous CPU storage")

        source_flat = source_view.reshape(-1)
        target_flat = target.reshape(-1)
        elements_per_window = window_bytes // itemsize
        windows = 0
        with torch.no_grad():
            for start in range(0, source.numel(), elements_per_window):
                end = min(source.numel(), start + elements_per_window)
                source_window = source_flat[start:end]
                target_window = target_flat[start:end]
                src_addr = int(source_window.data_ptr())
                bytes_this_window = int(source_window.numel()) * itemsize
                if src_addr % mmap.PAGESIZE != 0:
                    raise RuntimeError("mapped FTW expert window address is not page aligned")
                registered = False
                try:
                    host_register_transfer(src_addr, bytes_this_window)
                    registered = True
                    registered_host_to_device_copy(
                        int(target_window.data_ptr()), src_addr, bytes_this_window
                    )
                finally:
                    if registered:
                        host_unregister(src_addr)
                windows += 1

        return PageablePrefillTransferReceipt(
            nbytes=nbytes,
            window_bytes=window_bytes,
            windows=windows,
        )
    finally:
        target_window = None
        source_window = None
        target_flat = None
        source_flat = None
        source_view = None
        if mapping is not None:
            mapping.close()


__all__ = [
    "PageablePrefillTransferReceipt",
    "copy_mapped_ftw_readonly_windows_into",
    "is_mapped_ftw_tensor",
]
