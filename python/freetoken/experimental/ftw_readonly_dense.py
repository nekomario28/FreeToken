"""Read-only file-backed FTW dense transfer for bounded host residency.

Dense H2D loading never writes the checkpoint source.  Keep that source mapped read-only so
HIP host registration cannot turn touched MAP_PRIVATE/COW pages into tensor-sized anonymous
residency.  The destination remains a normal preallocated torch tensor and GPU targets use the
same synchronous direct-runtime H2D primitive as the earlier registered-dense experiment.
"""
from __future__ import annotations

import mmap
import warnings
from pathlib import Path

import torch

from freetoken.checkpoint.mapped_ftw_core import single_shard_range, unique_entry
from freetoken.experimental.ftw_registered_dense import (
    DEFAULT_WINDOW_BYTES,
    RegisteredDenseTransferReceipt,
    _validate_window,
    _validated_dense_entry,
)
from freetoken.kernel.pinned import (
    host_register_transfer,
    host_unregister,
    registered_host_to_device_copy,
)


def _readonly_mapping(path: str | Path, index: dict, name: str) -> tuple[mmap.mmap, int]:
    root = Path(path)
    entry = unique_entry(index, name)
    shard, file_offset = single_shard_range(index, entry)
    nbytes = int(entry.get("nbytes", -1))
    if nbytes <= 0:
        raise ValueError("FTW dense entry has invalid nbytes")

    raw = shard.get("file")
    if not isinstance(raw, str) or not raw:
        raise ValueError("FTW shard has invalid file name")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("FTW shard path must be checkpoint-relative")
    root_real = root.resolve(strict=True)
    shard_path = (root_real / candidate).resolve(strict=True)
    try:
        shard_path.relative_to(root_real)
    except ValueError as exc:
        raise ValueError("FTW shard path escapes checkpoint directory") from exc
    if not shard_path.is_file():
        raise ValueError("FTW shard path is not a regular file")
    if file_offset < 0 or file_offset + nbytes > shard_path.stat().st_size:
        raise ValueError("FTW tensor range exceeds its shard file")

    granularity = mmap.ALLOCATIONGRANULARITY
    mapping_offset = file_offset - (file_offset % granularity)
    data_offset = file_offset - mapping_offset
    mapping_length = data_offset + nbytes
    with shard_path.open("rb") as handle:
        mapping = mmap.mmap(
            handle.fileno(), mapping_length, access=mmap.ACCESS_READ, offset=mapping_offset
        )
    return mapping, data_offset


def copy_ftw_dense_readonly_windows_into(
    path: str | Path,
    name: str,
    target: torch.Tensor,
    *,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> RegisteredDenseTransferReceipt:
    """Copy one exact-dtype FTW entry from a read-only file mapping into ``target``."""
    index, _entry, dtype, shape, itemsize, numel, nbytes = _validated_dense_entry(path, name)
    _validate_window(window_bytes, itemsize)
    if target.dtype != dtype:
        raise ValueError("read-only dense target dtype must exactly match FTW dtype")
    if tuple(target.shape) != shape:
        raise ValueError("read-only dense target shape must exactly match FTW shape")
    if not target.is_contiguous():
        raise ValueError("read-only dense target must be contiguous")

    mapping, data_offset = _readonly_mapping(path, index, name)
    source = source_flat = source_window = target_flat = target_window = None
    try:
        if data_offset % mmap.PAGESIZE != 0:
            raise ValueError("FTW dense mapping data address is not page aligned")
        # PyTorch warns that it cannot enforce the buffer's read-only property.  This source
        # tensor is private to this function and is never written or returned; suppress only
        # that construction warning while retaining the OS-level read-only mapping.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable.*",
                category=UserWarning,
            )
            source = torch.frombuffer(
                mapping, dtype=dtype, count=numel, offset=data_offset
            ).reshape(shape)
        if source.device.type != "cpu" or not source.is_contiguous():
            raise RuntimeError("FTW read-only dense source must be contiguous CPU storage")

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
            source_storage="file_backed_readonly_mmap",
            registration_lifetime="one_window_default_register_direct_copy_unregister",
            destination_storage="preallocated_final_tensor",
            gpu_copy_path="direct_runtime_h2d" if target.device.type == "cuda" else "torch_cpu_copy",
        )
    finally:
        target_window = None
        source_window = None
        target_flat = None
        source_flat = None
        source = None
        mapping.close()


def copy_ftw_dense_readonly_windows(
    path: str | Path,
    name: str,
    *,
    device: str | torch.device,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
) -> tuple[torch.Tensor, RegisteredDenseTransferReceipt]:
    """Allocate exact-dtype final storage and fill it from read-only FTW windows."""
    _index, _entry, dtype, shape, itemsize, _numel, _nbytes = _validated_dense_entry(path, name)
    _validate_window(window_bytes, itemsize)
    target = torch.empty(shape, dtype=dtype, device=device)
    receipt = copy_ftw_dense_readonly_windows_into(
        path, name, target, window_bytes=window_bytes
    )
    return target, receipt


__all__ = [
    "copy_ftw_dense_readonly_windows",
    "copy_ftw_dense_readonly_windows_into",
]
