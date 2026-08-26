"""Torch-free file-mapping core for the low-RAM FTW expert-source experiment.

This module intentionally depends only on the Python standard library so the storage
contract can be validated even on control-plane runners that do not have PyTorch.
It does not know tensor dtypes/shapes; :mod:`mapped_ftw` is the thin torch adapter.

The index filename is part of the FTW wire format and mirrors ``ftw.INDEX_NAME``.
Keeping this tiny module independent of ``ftw.py`` avoids importing torch merely to
validate private file-mapping semantics.
"""
from __future__ import annotations

import json
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"


@dataclass
class MappedFTWRange:
    """Owner for one private file mapping covering exactly one FTW tensor entry.

    The mapping is ACCESS_COPY (private/COW). Keep this owner alive while any consumer
    view references ``mapping``. P0 deliberately exposes no eager close method because
    closing a buffer beneath a consumer view can invalidate that view unsafely.
    """

    name: str
    entry: dict[str, Any]
    mapping: mmap.mmap
    shard_path: Path
    file_offset: int
    mapping_offset: int
    data_offset: int
    nbytes: int


def load_ftw_index(path: Path) -> dict[str, Any]:
    with (path / INDEX_NAME).open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict) or index.get("format") != FORMAT_TAG:
        raise ValueError(f"not a FreeToken Weight checkpoint: {path}")
    if not isinstance(index.get("tensors"), list) or not isinstance(index.get("shards"), list):
        raise ValueError("FTW index is missing tensors/shards lists")
    return index


def unique_entry(index: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in index["tensors"] if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one FTW tensor named {name!r}, got {len(matches)}")
    return matches[0]


def single_shard_range(
    index: dict[str, Any], entry: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    try:
        start = int(entry["global_off"])
        nbytes = int(entry["nbytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("FTW entry has invalid offset/length") from exc
    if start < 0 or nbytes <= 0:
        raise ValueError("FTW entry has invalid offset/length")
    end = start + nbytes
    containing: list[tuple[dict[str, Any], int]] = []
    for shard in index["shards"]:
        if not isinstance(shard, dict):
            continue
        try:
            shard_start = int(shard["global_off"])
            shard_bytes = int(shard["nbytes"])
        except (KeyError, TypeError, ValueError):
            continue
        if shard_start < 0 or shard_bytes <= 0:
            continue
        shard_end = shard_start + shard_bytes
        if start >= shard_start and end <= shard_end:
            containing.append((shard, start - shard_start))
    if len(containing) != 1:
        raise ValueError(
            "P0 file-backed FTW mapping requires the tensor entry to fit in exactly one shard"
        )
    return containing[0]


def map_ftw_range(path: str | os.PathLike[str], name: str) -> MappedFTWRange:
    """Return a private file mapping over one single-shard FTW tensor byte range.

    No payload copy into anonymous storage is performed here. ``mmap.ACCESS_COPY`` gives
    consumers a writable buffer interface while keeping clean pages file-backed; writes
    become private COW pages and cannot mutate the checkpoint file.
    """

    root = Path(path)
    index = load_ftw_index(root)
    entry = unique_entry(index, name)
    shard, file_offset = single_shard_range(index, entry)
    nbytes = int(entry["nbytes"])

    shard_file = shard.get("file")
    if not isinstance(shard_file, str) or not shard_file:
        raise ValueError("FTW shard has invalid file name")
    shard_path = root / shard_file
    shard_size = shard_path.stat().st_size
    if file_offset < 0 or file_offset + nbytes > shard_size:
        raise ValueError("FTW tensor range exceeds its shard file")

    granularity = mmap.ALLOCATIONGRANULARITY
    mapping_offset = file_offset - (file_offset % granularity)
    data_offset = file_offset - mapping_offset
    mapping_length = data_offset + nbytes
    with shard_path.open("rb") as handle:
        mapping = mmap.mmap(
            handle.fileno(),
            mapping_length,
            access=mmap.ACCESS_COPY,
            offset=mapping_offset,
        )

    return MappedFTWRange(
        name=name,
        entry=entry,
        mapping=mapping,
        shard_path=shard_path,
        file_offset=file_offset,
        mapping_offset=mapping_offset,
        data_offset=data_offset,
        nbytes=nbytes,
    )


__all__ = [
    "FORMAT_TAG",
    "INDEX_NAME",
    "MappedFTWRange",
    "load_ftw_index",
    "map_ftw_range",
    "single_shard_range",
    "unique_entry",
]
