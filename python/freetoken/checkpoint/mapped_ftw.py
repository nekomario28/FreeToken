"""CPU-only proof-of-concept for file-backed FTW tensor sources.

This module deliberately does *not* wire the mappings into the serving engine yet.
It proves the smallest reusable primitive needed by a low-RAM MoE path: an FTW tensor
entry can be exposed as a torch.Tensor whose storage is a private file mapping rather
than a copied anonymous HostBank allocation.

`mmap.ACCESS_COPY` is intentional. Clean pages remain file-backed/reclaimable and a
stray write becomes a private COW page instead of mutating the FTW checkpoint. The
mapping owner must outlive every tensor view; P0 therefore exposes an owner object and
has no eager close API.
"""
from __future__ import annotations

import json
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .ftw import INDEX_NAME


@dataclass
class MappedFTWEntry:
    """Owner for one tensor view over a private file-backed FTW mapping.

    Keep this object alive for at least as long as ``tensor``. Closing an mmap while a
    torch.frombuffer tensor still points into it is unsafe, so P0 intentionally leaves
    teardown to process lifetime / a later owner that can prove all tensor users are gone.
    """

    name: str
    tensor: torch.Tensor
    mapping: mmap.mmap
    shard_path: Path
    file_offset: int
    mapping_offset: int
    tensor_offset: int
    nbytes: int


def _load_index(path: Path) -> dict[str, Any]:
    with (path / INDEX_NAME).open(encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("format") != "freetoken_weight":
        raise ValueError(f"not a FreeToken Weight checkpoint: {path}")
    return index


def _entry(index: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in index.get("tensors", []) if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one FTW tensor named {name!r}, got {len(matches)}")
    return matches[0]


def _single_shard(index: dict[str, Any], entry: dict[str, Any]) -> tuple[dict[str, Any], int]:
    start = int(entry["global_off"])
    nbytes = int(entry["nbytes"])
    if start < 0 or nbytes <= 0:
        raise ValueError("FTW entry has invalid offset/length")
    end = start + nbytes
    containing: list[tuple[dict[str, Any], int]] = []
    for shard in index.get("shards", []):
        shard_start = int(shard["global_off"])
        shard_end = shard_start + int(shard["nbytes"])
        if start >= shard_start and end <= shard_end:
            containing.append((shard, start - shard_start))
    if len(containing) != 1:
        raise ValueError(
            "P0 file-backed FTW mapping requires the tensor entry to fit in exactly one shard"
        )
    return containing[0]


def map_ftw_entry(path: str | os.PathLike[str], name: str) -> MappedFTWEntry:
    """Map one FTW tensor entry privately and return its zero-payload-copy tensor view.

    P0 is intentionally single-shard only. FTW's streaming converter writes per-layer
    expert-bank entries separately; those are the intended first consumer. Entries that
    cross shard boundaries fail closed instead of assuming virtual contiguity across files.
    """

    root = Path(path)
    index = _load_index(root)
    entry = _entry(index, name)
    shard, file_offset = _single_shard(index, entry)

    dtype_name = entry.get("dtype")
    dtype = getattr(torch, str(dtype_name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported FTW dtype {dtype_name!r}")
    shape = entry.get("shape")
    if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError("FTW entry has invalid shape")
    nbytes = int(entry["nbytes"])
    itemsize = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= dim
    if numel * itemsize != nbytes:
        raise ValueError("FTW entry shape/dtype does not match nbytes")

    shard_path = root / str(shard["file"])
    shard_size = shard_path.stat().st_size
    if file_offset + nbytes > shard_size:
        raise ValueError("FTW tensor range exceeds its shard file")

    granularity = mmap.ALLOCATIONGRANULARITY
    mapping_offset = file_offset - (file_offset % granularity)
    tensor_offset = file_offset - mapping_offset
    mapping_length = tensor_offset + nbytes
    with shard_path.open("rb") as handle:
        mapping = mmap.mmap(
            handle.fileno(),
            mapping_length,
            access=mmap.ACCESS_COPY,
            offset=mapping_offset,
        )

    tensor = torch.frombuffer(
        mapping,
        dtype=dtype,
        count=numel,
        offset=tensor_offset,
    )
    tensor = tensor.reshape(tuple(shape)) if shape else tensor.reshape(())
    return MappedFTWEntry(
        name=name,
        tensor=tensor,
        mapping=mapping,
        shard_path=shard_path,
        file_offset=file_offset,
        mapping_offset=mapping_offset,
        tensor_offset=tensor_offset,
        nbytes=nbytes,
    )


__all__ = ["MappedFTWEntry", "map_ftw_entry"]
