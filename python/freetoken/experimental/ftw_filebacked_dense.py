"""Experimental low-RAM FTW dense-weight iterator backed directly by shard mmaps.

The production FTW iterator reads each weight into transient anonymous buffers and
prefetches ahead.  That is fast, but on a host with only a few GiB of headroom the
largest dense tensors can make startup memory the next bottleneck after expert banks
have become file-backed.

This experiment keeps the FTW file as the host backing store instead:

* weight entries are exposed with ``mmap.ACCESS_COPY`` + ``torch.frombuffer``;
* every accepted tensor must live wholly inside one FTW shard (the normal writer
  already rolls before tensors <= shard_limit would split);
* there is no read-ahead thread and no anonymous tensor-sized staging allocation;
* when the consumer resumes the generator after copying a tensor to its device, the
  shard VMA receives ``MADV_DONTNEED`` so clean pages can be reclaimed immediately;
* mappings are private/COW, so accidental host writes cannot change the checkpoint.

Normal ``iter_ftw_weights`` and ``ft serve`` are untouched.  The experimental
file-backed CPU-MoE launcher installs this iterator only inside its spawned worker.
"""
from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Iterator

import torch

from freetoken.experimental.ftw_filebacked_cpu_moe import (
    _entry_location,
    _load_index,
    _safe_shard,
    _tensor_from_mapping,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Engine construction keeps the copied device tensors, not these host tensors.  The
# mmaps nevertheless stay valid for the process lifetime so a caller retaining a
# yielded tensor cannot observe a dangling buffer after the generator advances.
_LIVE_DENSE_FILE_MAPPINGS: list[mmap.mmap] = []


def _map_dense_shard(path: Path) -> mmap.mmap:
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    try:
        # Dense device copies consume a tensor linearly.  Readahead is useful during
        # the copy; MADV_DONTNEED after each yield bounds how long those pages stay in
        # this process's resident set.
        mm.madvise(mmap.MADV_SEQUENTIAL)
    except (AttributeError, OSError, ValueError):
        pass
    return mm


def _drop_mapping_pages(mm: mmap.mmap) -> None:
    try:
        mm.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError, ValueError):
        pass


def iter_ftw_weights_filebacked(
    path: str,
    *,
    kinds=("weight",),
    workers: int = 8,
    chunk: int = 8 << 20,
    prefetch: int = 2,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield FTW tensors as private file-backed views with no anonymous prefetch.

    The extra arguments intentionally mirror :func:`freetoken.checkpoint.ftw.iter_ftw_weights`
    so this can replace that function at the experimental worker seam.  ``workers``,
    ``chunk`` and ``prefetch`` are ignored: eliminating the staging/prefetch buffers is
    the point of this low-memory path.
    """
    del workers, chunk, prefetch

    root = Path(path)
    index = _load_index(root)
    keep = set(kinds)
    entries = [
        entry
        for entry in index.get("tensors", [])
        if isinstance(entry, dict) and (not keep or entry.get("kind") in keep)
    ]

    mappings: dict[str, mmap.mmap] = {}
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("FTW dense entry has a missing/non-string name")

        shard_name, file_off = _entry_location(index, entry)
        mm = mappings.get(shard_name)
        if mm is None:
            shard = _safe_shard(root, shard_name)
            mm = _map_dense_shard(shard)
            mappings[shard_name] = mm
            _LIVE_DENSE_FILE_MAPPINGS.append(mm)

        tensor = _tensor_from_mapping(mm, entry, file_off)
        if not tensor.is_contiguous():
            raise ValueError(f"FTW dense tensor {name!r} is not contiguous")

        try:
            yield name, tensor
        finally:
            # The production engine's consumer performs a blocking .to(device) for
            # pageable host tensors before requesting the next item.  At resume, the
            # previous tensor's bytes are therefore no longer needed for startup.
            _drop_mapping_pages(mm)

    if mappings:
        logger.info(
            "FTW dense file-backed experiment: %d tensors across %d mapped shard(s); "
            "anonymous prefetch disabled",
            len(entries),
            len(mappings),
        )


__all__ = ["iter_ftw_weights_filebacked"]
