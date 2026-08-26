"""Experimental zero-copy file-backed FTW expert banks for all-CPU MoE decode.

This module is deliberately narrow. It maps per-layer FTW ``experts_bank`` entries
with ``mmap.ACCESS_COPY`` and exposes them as contiguous CPU tensors, so the kernel
can fault expert pages directly from the checkpoint instead of first copying the
whole expert bank into anonymous HostBanks.

Safety / scope:
* FTW checkpoints only.
* native ``nvfp4`` bank layout only.
* every MoE layer must be CPU-routed (requested residency LOCKED/PAGEABLE).
* per-layer streaming FTW entries only; legacy flat bank entries are rejected.
* mappings are private/COW, so accidental tensor writes cannot modify the FTW file.
* normal FreeToken loaders and ``ft serve`` are untouched.

This is an experiment surface, not a production feature.
"""
from __future__ import annotations

import json
import mmap
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from freetoken.checkpoint.ftw import FORMAT_TAG, FORMAT_VERSION, INDEX_NAME
from freetoken.moe.host_banks import HostResidency

_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")
_NATIVE_NVFP4_BANKS = (
    "gate_up_packed",
    "gate_up_scale",
    "gate_up_global",
    "down_packed",
    "down_scale",
    "down_global",
)

# Process-lifetime ownership mirrors host_banks._LIVE_BUFFERS. OffloadMoeCache keeps
# tensor views, not this result wrapper, so the underlying file mappings must outlive
# the transient ExpertBanks-like object returned during engine construction.
_LIVE_FILE_MAPPINGS: list[mmap.mmap] = []


@dataclass
class FileBackedExpertBanks:
    """Duck-compatible subset of ``ExpertBanks`` plus file-backed telemetry."""

    quant_format: str
    sources: dict[str, list[torch.Tensor]]
    gate_up_alpha: torch.Tensor | None = None
    down_alpha: torch.Tensor | None = None
    layer_residency: list[str] | None = None
    streamed: bool = False
    file_backed_bytes: int = 0
    file_backed_layers: int = 0
    mapped_shards: int = 0
    _mapping_refs: tuple[mmap.mmap, ...] = field(default_factory=tuple, repr=False)


def _dtype_of(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported FTW dtype {name!r}")
    return value


def _load_index(path: Path) -> dict[str, Any]:
    index_path = path / INDEX_NAME
    with index_path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise ValueError("FTW index is not an object")
    if index.get("format") != FORMAT_TAG or index.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported FTW index format/version: "
            f"{index.get('format')!r}/{index.get('version')!r}"
        )
    return index


def _safe_shard(root: Path, name: str) -> Path:
    """Resolve one index-provided shard name without allowing path escape."""

    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"unsafe FTW shard name {name!r}")
    root_real = root.resolve(strict=True)
    candidate = (root_real / name).resolve(strict=True)
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise ValueError(f"FTW shard escapes checkpoint root: {name!r}") from exc
    st = candidate.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"FTW shard is not a regular file: {name!r}")
    return candidate


def _entry_location(index: dict[str, Any], entry: dict[str, Any]) -> tuple[str, int]:
    """Return ``(shard_name, shard_local_offset)`` for one unsplit entry."""

    off = int(entry["global_off"])
    nbytes = int(entry["nbytes"])
    if off < 0 or nbytes <= 0:
        raise ValueError(f"invalid FTW entry range off={off} nbytes={nbytes}")

    matches: list[tuple[str, int]] = []
    for shard in index.get("shards", []):
        if not isinstance(shard, dict):
            continue
        s0 = int(shard.get("global_off", -1))
        sn = int(shard.get("nbytes", -1))
        s1 = s0 + sn
        if s0 >= 0 and sn >= 0 and off >= s0 and off + nbytes <= s1:
            matches.append((str(shard.get("file", "")), off - s0))
    if len(matches) != 1:
        raise ValueError(
            "file-backed prototype requires each per-layer FTW entry to live wholly "
            f"inside exactly one shard; found {len(matches)} matches"
        )
    return matches[0]


def _map_shard(path: Path) -> mmap.mmap:
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    try:
        mm.madvise(mmap.MADV_RANDOM)
    except (AttributeError, OSError, ValueError):
        pass
    return mm


def _tensor_from_mapping(mm: mmap.mmap, entry: dict[str, Any], file_off: int) -> torch.Tensor:
    dtype = _dtype_of(str(entry["dtype"]))
    shape = tuple(int(v) for v in entry["shape"])
    if any(v < 0 for v in shape):
        raise ValueError(f"negative FTW tensor shape: {shape}")
    elsize = torch.empty((), dtype=dtype).element_size()
    nbytes = int(entry["nbytes"])
    if nbytes % elsize:
        raise ValueError(f"FTW tensor byte count {nbytes} is not aligned to dtype size {elsize}")
    count = nbytes // elsize
    expected = 1
    for dim in shape:
        expected *= dim
    if expected != count:
        raise ValueError(
            f"FTW tensor shape {shape} implies {expected} elements but entry stores {count}"
        )
    if file_off < 0 or file_off + nbytes > len(mm):
        raise ValueError("FTW tensor range exceeds mapped shard")
    tensor = torch.frombuffer(mm, dtype=dtype, count=count, offset=file_off)
    return tensor.view(*shape) if shape else tensor.view(())


def _validate_native_nvfp4_schema(sources: dict[str, list[torch.Tensor]]) -> tuple[int, int, int]:
    """Fail before C++ sees raw pointers if the mapped native-NVFP4 geometry is invalid."""

    gup = sources["gate_up_packed"][0]
    gus = sources["gate_up_scale"][0]
    gug = sources["gate_up_global"][0]
    dnp = sources["down_packed"][0]
    dns = sources["down_scale"][0]
    dng = sources["down_global"][0]

    if gup.dtype != torch.uint8 or dnp.dtype != torch.uint8:
        raise ValueError(f"nvfp4 packed banks must be uint8, got {gup.dtype}/{dnp.dtype}")
    if gus.element_size() != 1 or dns.element_size() != 1:
        raise ValueError("nvfp4 block-scale banks must use one-byte elements")
    if gug.dtype != torch.float16 or dng.dtype != torch.float16:
        raise ValueError(f"nvfp4 global banks must be float16, got {gug.dtype}/{dng.dtype}")
    if gup.ndim != 3:
        raise ValueError(f"gate_up_packed must be rank 3, got shape {tuple(gup.shape)}")

    experts = int(gup.shape[0])
    inter = int(gup.shape[1] // 2)
    hidden = int(gup.shape[2] * 2)
    if experts <= 0 or gup.shape[1] != 2 * inter or hidden % 16 or inter % 16:
        raise ValueError(
            f"invalid native nvfp4 dimensions experts={experts} hidden={hidden} inter={inter}"
        )

    expected = {
        "gate_up_packed": (experts, 2 * inter, hidden // 2),
        "gate_up_scale": (experts, 2 * inter, hidden // 16),
        "gate_up_global": (experts, 2 * inter),
        "down_packed": (experts, hidden, inter // 2),
        "down_scale": (experts, hidden, inter // 16),
        "down_global": (experts, hidden),
    }
    for name, shape in expected.items():
        for layer, tensor in enumerate(sources[name]):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"nvfp4 bank {name!r} layer {layer} has shape {tuple(tensor.shape)}, "
                    f"expected {shape}"
                )
    return experts, hidden, inter


def load_ftw_banks_filebacked_cpu(
    path: str,
    *,
    num_layers: int,
    layer_residency: list[str] | None,
) -> FileBackedExpertBanks:
    """Map native-NVFP4 per-layer expert entries directly from an FTW checkpoint.

    ``layer_residency`` is the engine's requested plan. This first prototype refuses
    any PINNED layer and deliberately reports every accepted layer as PAGEABLE because
    its storage is file-backed/reclaimable, even when the engine requested LOCKED.
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if layer_residency is None or len(layer_residency) != num_layers:
        raise ValueError("explicit per-layer residency is required")
    allowed = {HostResidency.LOCKED.value, HostResidency.PAGEABLE.value}
    bad = [i for i, value in enumerate(layer_residency) if value not in allowed]
    if bad:
        raise ValueError(
            "file-backed CPU-MoE prototype requires every layer to be CPU-routed "
            f"(LOCKED/PAGEABLE); incompatible layers: {bad}"
        )

    root = Path(path)
    index = _load_index(root)
    if index.get("quant_format") != "nvfp4":
        raise ValueError(
            "file-backed CPU-MoE prototype currently requires native FTW quant_format "
            f"'nvfp4', got {index.get('quant_format')!r}"
        )
    meta_layers = index.get("expert_bank_num_layers")
    if meta_layers is not None and int(meta_layers) != num_layers:
        raise ValueError(
            f"FTW expert_bank_num_layers={meta_layers} does not match num_layers={num_layers}"
        )

    entries = [
        entry for entry in index.get("tensors", [])
        if isinstance(entry, dict) and entry.get("kind") == "experts_bank"
    ]
    if not entries:
        raise ValueError("FTW checkpoint contains no experts_bank entries")

    groups: dict[str, dict[int, dict[str, Any]]] = {}
    flat: list[str] = []
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError("FTW expert entry has a non-string name")
        match = _LAYER_ENTRY_RE.match(name)
        if match is None:
            flat.append(name)
            continue
        base = match.group("base")
        layer = int(match.group("layer"))
        if layer in groups.setdefault(base, {}):
            raise ValueError(f"duplicate FTW expert entry for {base!r} layer {layer}")
        groups[base][layer] = entry

    if flat:
        raise ValueError(
            "file-backed prototype requires streaming per-layer FTW expert entries; "
            f"found {len(flat)} flat entries"
        )
    if set(groups) != set(_NATIVE_NVFP4_BANKS):
        raise ValueError(
            "FTW native nvfp4 bank schema mismatch: "
            f"found {sorted(groups)}, expected {sorted(_NATIVE_NVFP4_BANKS)}"
        )

    expected_layers = list(range(num_layers))
    for base, by_layer in groups.items():
        if sorted(by_layer) != expected_layers:
            raise ValueError(
                f"FTW bank {base!r} has layers {sorted(by_layer)}, expected {expected_layers}"
            )

    mappings_by_name: dict[str, mmap.mmap] = {}
    mapping_refs: list[mmap.mmap] = []
    sources: dict[str, list[torch.Tensor]] = {name: [] for name in _NATIVE_NVFP4_BANKS}
    file_backed_bytes = 0

    try:
        for base in _NATIVE_NVFP4_BANKS:
            first_shape: tuple[int, ...] | None = None
            first_dtype: torch.dtype | None = None
            for layer in range(num_layers):
                entry = groups[base][layer]
                shard_name, file_off = _entry_location(index, entry)
                shard = _safe_shard(root, shard_name)
                mm = mappings_by_name.get(shard_name)
                if mm is None:
                    mm = _map_shard(shard)
                    mappings_by_name[shard_name] = mm
                    mapping_refs.append(mm)
                tensor = _tensor_from_mapping(mm, entry, file_off)
                if not tensor.is_contiguous():
                    raise ValueError(f"FTW bank {base!r} layer {layer} is not contiguous")
                if tensor.ndim < 1:
                    raise ValueError(f"FTW bank {base!r} layer {layer} has no expert dimension")
                if first_shape is None:
                    first_shape = tuple(tensor.shape)
                    first_dtype = tensor.dtype
                elif tuple(tensor.shape) != first_shape or tensor.dtype != first_dtype:
                    raise ValueError(
                        f"FTW bank {base!r} layer {layer} shape/dtype differs from layer 0"
                    )
                sources[base].append(tensor)
                file_backed_bytes += int(entry["nbytes"])
    except BaseException:
        for mm in mapping_refs:
            try:
                mm.close()
            except BufferError:
                # A tensor may already have been constructed; process teardown will
                # reclaim it. Do not mask the validation error with cleanup failure.
                pass
        raise

    _validate_native_nvfp4_schema(sources)
    _LIVE_FILE_MAPPINGS.extend(mapping_refs)
    return FileBackedExpertBanks(
        quant_format="nvfp4",
        sources=sources,
        layer_residency=[HostResidency.PAGEABLE.value] * num_layers,
        file_backed_bytes=file_backed_bytes,
        file_backed_layers=num_layers,
        mapped_shards=len(mapping_refs),
        _mapping_refs=tuple(mapping_refs),
    )


__all__ = ["FileBackedExpertBanks", "load_ftw_banks_filebacked_cpu"]
