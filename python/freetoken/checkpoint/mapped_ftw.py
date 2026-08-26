"""Thin PyTorch adapter for file-backed FTW tensor sources.

The storage/range/mmap contract lives in :mod:`mapped_ftw_core`, which is deliberately
standard-library-only so it can be validated on control-plane runners without PyTorch.
This module adds only FTW dtype/shape validation and ``torch.frombuffer`` tensor views.

The returned owner must outlive every tensor view. P0 intentionally has no eager close
API because closing an mmap beneath a live tensor is unsafe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .mapped_ftw_core import MappedFTWRange, map_ftw_range


@dataclass
class MappedFTWEntry:
    """Owner for one torch tensor view over a private file-backed FTW mapping."""

    name: str
    tensor: torch.Tensor
    storage: MappedFTWRange

    @property
    def mapping(self):
        return self.storage.mapping

    @property
    def shard_path(self):
        return self.storage.shard_path

    @property
    def file_offset(self) -> int:
        return self.storage.file_offset

    @property
    def mapping_offset(self) -> int:
        return self.storage.mapping_offset

    @property
    def tensor_offset(self) -> int:
        return self.storage.data_offset

    @property
    def nbytes(self) -> int:
        return self.storage.nbytes


def map_ftw_entry(path: str | os.PathLike[str], name: str) -> MappedFTWEntry:
    """Map one single-shard FTW entry and expose an exact torch tensor view.

    No expert payload is copied into an anonymous HostBank here. The underlying mapping
    is ``mmap.ACCESS_COPY``: clean pages remain file-backed/reclaimable while accidental
    writes become private COW pages and cannot modify the checkpoint.
    """

    storage = map_ftw_range(path, name)
    entry = storage.entry

    dtype_name = entry.get("dtype")
    dtype = getattr(torch, str(dtype_name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported FTW dtype {dtype_name!r}")

    shape = entry.get("shape")
    if not isinstance(shape, list) or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
    ):
        raise ValueError("FTW entry has invalid shape")

    itemsize = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= dim
    if numel * itemsize != storage.nbytes:
        raise ValueError("FTW entry shape/dtype does not match nbytes")

    tensor = torch.frombuffer(
        storage.mapping,
        dtype=dtype,
        count=numel,
        offset=storage.data_offset,
    )
    tensor = tensor.reshape(tuple(shape)) if shape else tensor.reshape(())
    return MappedFTWEntry(name=name, tensor=tensor, storage=storage)


__all__ = ["MappedFTWEntry", "map_ftw_entry"]
