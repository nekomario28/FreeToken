"""Torch-free storage core for file-backed FTW expert-bank experiments.

The serving experiment needs a property that anonymous PAGEABLE HostBanks cannot give:
clean expert pages must remain backed by the checkpoint so the OS may discard/refault them
without first writing the whole bank to swap.  This module proves and owns that storage
contract without importing PyTorch.

Only single-shard tensor entries are mappable in this phase.  FTW's streamed converter writes
per-layer expert-bank entries separately and aligned, so this is sufficient for the intended
CPU-routed first consumer.  Cross-shard entries fail closed instead of inventing virtual
contiguity across files.
"""
from __future__ import annotations

import json
import mmap
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"
_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")
_ALPHA_NAMES = frozenset({"gate_up_alpha", "down_alpha"})


@dataclass
class MappedFTWRange:
    """Owner for one private/COW file mapping covering an FTW tensor entry."""

    name: str
    entry: dict[str, Any]
    mapping: mmap.mmap
    shard_path: Path
    file_offset: int
    mapping_offset: int
    data_offset: int
    nbytes: int


def _as_int(value: Any, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{what} must be >= {minimum}")
    return result


def load_ftw_index(path: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(path)
    with (root / INDEX_NAME).open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict) or index.get("format") != FORMAT_TAG:
        raise ValueError(f"not a FreeToken Weight checkpoint: {root}")
    if not isinstance(index.get("tensors"), list) or not isinstance(index.get("shards"), list):
        raise ValueError("FTW index is missing tensors/shards lists")
    return index


def unique_entry(index: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        item for item in index["tensors"]
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one FTW tensor named {name!r}, got {len(matches)}")
    return matches[0]


def single_shard_range(
    index: dict[str, Any], entry: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    start = _as_int(entry.get("global_off"), "FTW entry global_off")
    nbytes = _as_int(entry.get("nbytes"), "FTW entry nbytes", minimum=1)
    end = start + nbytes
    containing: list[tuple[dict[str, Any], int]] = []
    for shard in index["shards"]:
        if not isinstance(shard, dict):
            continue
        try:
            shard_start = _as_int(shard.get("global_off"), "FTW shard global_off")
            shard_bytes = _as_int(shard.get("nbytes"), "FTW shard nbytes", minimum=1)
        except ValueError:
            continue
        shard_end = shard_start + shard_bytes
        if start >= shard_start and end <= shard_end:
            containing.append((shard, start - shard_start))
    if len(containing) != 1:
        raise ValueError(
            "P0/P1 file-backed FTW mapping requires the tensor entry to fit in exactly one shard"
        )
    return containing[0]


def _safe_shard_path(root: Path, shard: dict[str, Any]) -> Path:
    raw = shard.get("file")
    if not isinstance(raw, str) or not raw:
        raise ValueError("FTW shard has invalid file name")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("FTW shard path must be checkpoint-relative")
    root_real = root.resolve(strict=True)
    shard_real = (root_real / candidate).resolve(strict=True)
    try:
        shard_real.relative_to(root_real)
    except ValueError as exc:
        raise ValueError("FTW shard path escapes checkpoint directory") from exc
    if not shard_real.is_file():
        raise ValueError("FTW shard path is not a regular file")
    return shard_real


def map_ftw_range_from_index(
    path: str | os.PathLike[str], index: dict[str, Any], name: str
) -> MappedFTWRange:
    """Map one FTW entry using an already-validated index.

    ``mmap.ACCESS_COPY`` is MAP_PRIVATE/COW semantics: clean pages stay file-backed and
    writable consumers cannot mutate the checkpoint.  The returned owner must outlive every
    view of ``mapping``; no eager close API is exposed in this experiment.
    """

    root = Path(path)
    entry = unique_entry(index, name)
    shard, file_offset = single_shard_range(index, entry)
    nbytes = _as_int(entry.get("nbytes"), "FTW entry nbytes", minimum=1)
    shard_path = _safe_shard_path(root, shard)
    shard_size = shard_path.stat().st_size
    if file_offset < 0 or file_offset + nbytes > shard_size:
        raise ValueError("FTW tensor range exceeds its shard file")

    granularity = mmap.ALLOCATIONGRANULARITY
    mapping_offset = file_offset - (file_offset % granularity)
    data_offset = file_offset - mapping_offset
    mapping_length = data_offset + nbytes
    with shard_path.open("rb") as handle:
        mapping = mmap.mmap(
            handle.fileno(), mapping_length, access=mmap.ACCESS_COPY, offset=mapping_offset
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


def map_ftw_range(path: str | os.PathLike[str], name: str) -> MappedFTWRange:
    return map_ftw_range_from_index(path, load_ftw_index(path), name)


def group_per_layer_expert_entries(
    index: dict[str, Any],
    num_layers: int,
    *,
    expected_banks: set[str] | frozenset[str] | None = None,
) -> dict[str, list[str]]:
    """Validate and group the converter's ``bank#Lxxxxx`` expert-bank layout.

    Reserved flat alpha vectors are ignored.  Any other flat expert-bank row is rejected:
    file-backed P1 deliberately supports only the streamed per-layer layout, where each
    mapping can be independently reclaimed and where no virtual cross-file tensor is needed.
    """

    num_layers = _as_int(num_layers, "num_layers", minimum=1)
    meta_layers = index.get("expert_bank_num_layers")
    if meta_layers is not None and _as_int(meta_layers, "expert_bank_num_layers", minimum=1) != num_layers:
        raise ValueError(
            f"FTW expert_bank_num_layers={meta_layers!r} does not match requested {num_layers}"
        )

    grouped: dict[str, dict[int, str]] = {}
    flat_rows: list[str] = []
    for entry in index["tensors"]:
        if not isinstance(entry, dict) or entry.get("kind") != "experts_bank":
            continue
        raw_name = entry.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("FTW experts_bank entry has invalid name")
        if raw_name in _ALPHA_NAMES:
            continue
        match = _LAYER_ENTRY_RE.match(raw_name)
        if match is None:
            flat_rows.append(raw_name)
            continue
        layer_id = int(match.group("layer"))
        if layer_id >= num_layers:
            raise ValueError(
                f"FTW expert bank {raw_name!r} targets layer {layer_id}, outside range({num_layers})"
            )
        base = match.group("base")
        by_layer = grouped.setdefault(base, {})
        if layer_id in by_layer:
            raise ValueError(f"duplicate FTW bank {base!r} for layer {layer_id}")
        by_layer[layer_id] = raw_name

    if flat_rows:
        raise ValueError(
            "file-backed FTW expert sources require per-layer entries; flat bank(s): "
            + ", ".join(sorted(flat_rows))
        )
    if not grouped:
        raise ValueError("FTW checkpoint has no per-layer experts_bank entries")

    actual = set(grouped)
    if expected_banks is not None and actual != set(expected_banks):
        raise ValueError(
            f"FTW expert banks {sorted(actual)}, expected {sorted(expected_banks)}"
        )

    expected_layers = list(range(num_layers))
    result: dict[str, list[str]] = {}
    for base in sorted(grouped):
        by_layer = grouped[base]
        if sorted(by_layer) != expected_layers:
            raise ValueError(
                f"FTW bank {base!r} has layers {sorted(by_layer)}, expected {expected_layers}"
            )
        result[base] = [by_layer[layer] for layer in expected_layers]
    return result


__all__ = [
    "FORMAT_TAG",
    "INDEX_NAME",
    "MappedFTWRange",
    "group_per_layer_expert_entries",
    "load_ftw_index",
    "map_ftw_range",
    "map_ftw_range_from_index",
    "single_shard_range",
    "unique_entry",
]
