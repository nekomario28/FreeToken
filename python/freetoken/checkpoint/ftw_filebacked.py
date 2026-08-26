"""File-backed FTW expert tensors for CPU-routed MoE prototype paths.

The canonical FTW reader materializes every expert-bank entry into an anonymous HostBank.
That is required for GPU-routed layers because their decode path needs a stable pinned host
pointer / device alias.  CPU-routed layers do not have that requirement: the CPU executor
only needs stable CPU tensor pointers and the existing prefill fallback already accepts
pageable host tensors.

This module maps a *per-layer* FTW ``experts_bank`` entry directly from its shard with a
private copy-on-write mmap and exposes it as a torch Tensor.  Clean pages therefore remain
file-backed/reclaimable; accidental writes are private and never modify the checkpoint.
No function here pins memory or makes a GPU alias.

It is intentionally fail-closed:
- only ``kind=experts_bank`` entries are accepted;
- the entry must fit wholly inside one physical FTW shard;
- shape/dtype bytes must exactly match index ``nbytes``;
- layer mapping requires the converter's ``#Lxxxxx`` per-layer layout.
"""
from __future__ import annotations

import json
import math
import mmap
import os
import re
from pathlib import Path

import torch

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"
_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")
_ALPHA_NAMES = {"gate_up_alpha", "down_alpha"}


def _dtype_of(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown FTW dtype {name!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"FTW dtype {name!r} does not resolve to torch.dtype")
    return dtype


def _load_index(path: str | os.PathLike[str]) -> tuple[Path, dict]:
    directory = Path(path)
    with (directory / INDEX_NAME).open(encoding="utf-8") as f:
        index = json.load(f)
    if index.get("format") != FORMAT_TAG:
        raise ValueError(f"not a {FORMAT_TAG} checkpoint")
    if not isinstance(index.get("tensors"), list) or not isinstance(index.get("shards"), list):
        raise ValueError("FTW index is missing tensors/shards")
    return directory, index


def _entry(index: dict, name: str) -> dict:
    matches = [e for e in index["tensors"] if isinstance(e, dict) and e.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"FTW entry {name!r}: expected exactly one index row, got {len(matches)}")
    entry = matches[0]
    if entry.get("kind") != "experts_bank":
        raise ValueError(f"FTW entry {name!r} is kind={entry.get('kind')!r}, not experts_bank")
    return entry


def _single_shard_piece(index: dict, entry: dict) -> tuple[str, int]:
    start = int(entry["global_off"])
    nbytes = int(entry["nbytes"])
    end = start + nbytes
    containing = []
    for shard in index["shards"]:
        s0 = int(shard["global_off"])
        s1 = s0 + int(shard["nbytes"])
        if start >= s0 and end <= s1:
            containing.append((str(shard["file"]), start - s0))
    if len(containing) != 1:
        raise ValueError(
            f"FTW experts_bank entry {entry.get('name')!r} is not wholly contained in one shard"
        )
    return containing[0]


def map_ftw_expert_entry(
    path: str | os.PathLike[str],
    name: str,
) -> torch.Tensor:
    """Map one single-shard ``experts_bank`` entry as a private file-backed CPU tensor.

    ``torch.frombuffer`` retains the mmap owner for the tensor lifetime, so the mapping stays
    valid after this function closes the file descriptor and returns.  The returned tensor is
    pageable CPU memory by construction and must not be used as a GPU-decode bank source.
    """
    directory, index = _load_index(path)
    entry = _entry(index, name)
    dtype = _dtype_of(str(entry["dtype"]))
    shape = tuple(int(x) for x in entry["shape"])
    if any(x < 0 for x in shape):
        raise ValueError(f"FTW entry {name!r} has a negative shape")
    count = math.prod(shape) if shape else 1
    expected_bytes = count * torch.empty((), dtype=dtype).element_size()
    nbytes = int(entry["nbytes"])
    if expected_bytes != nbytes:
        raise ValueError(
            f"FTW entry {name!r}: shape/dtype imply {expected_bytes} bytes, index says {nbytes}"
        )

    shard_name, file_off = _single_shard_piece(index, entry)
    page = mmap.ALLOCATIONGRANULARITY
    map_off = file_off // page * page
    delta = file_off - map_off
    map_len = delta + nbytes
    fd = os.open(directory / shard_name, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, map_len, access=mmap.ACCESS_COPY, offset=map_off)
    finally:
        os.close(fd)

    # frombuffer holds a reference to ``mm`` for the Storage lifetime.  ACCESS_COPY is
    # writable to PyTorch but private to this process; CPU inference is expected to read only.
    tensor = torch.frombuffer(mm, dtype=dtype, count=count, offset=delta)
    return tensor.reshape(shape) if shape else tensor.reshape(())


def map_ftw_cpu_layer_sources(
    path: str | os.PathLike[str],
    layer_id: int,
    *,
    expected_banks: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Map every per-layer expert bank for ``layer_id`` without materializing HostBanks."""
    if layer_id < 0:
        raise ValueError("layer_id must be non-negative")
    _directory, index = _load_index(path)
    names: dict[str, str] = {}
    for entry in index["tensors"]:
        if not isinstance(entry, dict) or entry.get("kind") != "experts_bank":
            continue
        raw_name = entry.get("name")
        if not isinstance(raw_name, str) or raw_name in _ALPHA_NAMES:
            continue
        match = _LAYER_ENTRY_RE.match(raw_name)
        if match is None or int(match.group("layer")) != layer_id:
            continue
        base = match.group("base")
        if base in names:
            raise ValueError(f"duplicate per-layer FTW bank {base!r} for layer {layer_id}")
        names[base] = raw_name

    actual = set(names)
    if expected_banks is not None and actual != expected_banks:
        raise ValueError(
            f"FTW layer {layer_id} banks {sorted(actual)}, expected {sorted(expected_banks)}"
        )
    if not names:
        raise ValueError(f"FTW layer {layer_id} has no per-layer experts_bank entries")
    return {base: map_ftw_expert_entry(path, raw) for base, raw in sorted(names.items())}


__all__ = ["map_ftw_expert_entry", "map_ftw_cpu_layer_sources"]
