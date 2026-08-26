"""PyTorch adapter for reclaimable file-backed FTW expert sources.

Storage ownership and layout validation live in :mod:`mapped_ftw_core`; this module adds
dtype/shape tensor views, the existing FreeToken source shape
``dict[bank_name, list[Tensor]]``, and an explicit CPU-only cache-attachment boundary.
Nothing here changes the default FTW loader or engine path.

Every returned Tensor also carries its :class:`MappedFTWRange` as a private Python attribute.
That makes mapping lifetime follow the exact tensor objects retained by ``bank_sources`` even
when an intermediate ``ExpertBanks``/bundle object is released by the canonical engine path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .mapped_ftw_core import (
    MappedFTWRange,
    group_per_layer_expert_entries,
    load_ftw_index,
    map_ftw_range,
    map_ftw_range_from_index,
)

PAGEABLE = "pageable"
_STORAGE_OWNER_ATTR = "_freetoken_mapped_ftw_storage"


@dataclass
class MappedFTWEntry:
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


@dataclass
class MappedFTWExpertSources:
    """Owner bundle matching ``ExpertBanks.sources`` without anonymous HostBanks."""

    quant_format: str
    sources: dict[str, list[torch.Tensor]]
    owners: list[MappedFTWEntry]
    layer_residency: list[str]


def _tensor_from_storage(storage: MappedFTWRange) -> torch.Tensor:
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
        storage.mapping, dtype=dtype, count=numel, offset=storage.data_offset
    )
    tensor = tensor.reshape(tuple(shape)) if shape else tensor.reshape(())
    if not tensor.is_contiguous():
        raise ValueError("file-backed FTW tensor view is unexpectedly non-contiguous")
    return tensor


def _owned_entry(name: str, storage: MappedFTWRange) -> MappedFTWEntry:
    """Create a tensor view and make the tensor itself retain the mmap storage owner.

    The bundle-level owner list remains useful for inspection/explicit attachment, but the
    tensor anchor is the lifetime invariant needed by the canonical loader -> cache path:
    ``OffloadMoeCache.bank_sources`` retains these tensors after the temporary loader return
    object is gone.
    """
    tensor = _tensor_from_storage(storage)
    setattr(tensor, _STORAGE_OWNER_ATTR, storage)
    return MappedFTWEntry(name=name, tensor=tensor, storage=storage)


def map_ftw_entry(path: str | os.PathLike[str], name: str) -> MappedFTWEntry:
    storage = map_ftw_range(path, name)
    return _owned_entry(name, storage)


def _map_ftw_entry_from_index(path, index, name: str) -> MappedFTWEntry:
    storage = map_ftw_range_from_index(path, index, name)
    return _owned_entry(name, storage)


def map_ftw_expert_sources(
    path: str | os.PathLike[str],
    num_layers: int,
    *,
    expected_banks: set[str] | frozenset[str] | None = None,
    expected_quant_format: str | None = None,
    num_experts: int | None = None,
) -> MappedFTWExpertSources:
    """Map all streamed per-layer expert banks into the existing source ABI.

    The checkpoint index is parsed once, every non-alpha expert bank must use the converter's
    ``bank#Lxxxxx`` layout, and every layer is labelled PAGEABLE because these mappings are
    neither pinned nor mlocked. The bundle retains all mapping owners explicitly, and each
    returned tensor independently anchors its own storage owner for cache lifetime safety.
    """

    index = load_ftw_index(path)
    quant_format = index.get("quant_format")
    if not isinstance(quant_format, str) or not quant_format:
        raise ValueError("FTW checkpoint is missing quant_format")
    if expected_quant_format is not None and quant_format != expected_quant_format:
        raise ValueError(
            f"FTW quant_format={quant_format!r}, expected {expected_quant_format!r}"
        )
    grouped = group_per_layer_expert_entries(
        index, num_layers, expected_banks=expected_banks
    )

    sources: dict[str, list[torch.Tensor]] = {}
    owners: list[MappedFTWEntry] = []
    for base, names in grouped.items():
        mapped = [_map_ftw_entry_from_index(path, index, name) for name in names]
        tensors = [owner.tensor for owner in mapped]
        head = tensors[0]
        for layer_id, tensor in enumerate(tensors):
            if tensor.shape != head.shape or tensor.dtype != head.dtype:
                raise ValueError(
                    f"FTW bank {base!r} layer {layer_id} shape/dtype differs from layer 0"
                )
            if num_experts is not None and tensor.size(0) != num_experts:
                raise ValueError(
                    f"FTW bank {base!r} layer {layer_id} has {tensor.size(0)} experts, "
                    f"expected {num_experts}"
                )
        sources[base] = tensors
        owners.extend(mapped)

    return MappedFTWExpertSources(
        quant_format=quant_format,
        sources=sources,
        owners=owners,
        layer_residency=[PAGEABLE] * int(num_layers),
    )


def attach_file_backed_ftw_sources(
    cache,
    path: str | os.PathLike[str],
    *,
    num_experts: int | None = None,
) -> MappedFTWExpertSources:
    """Explicitly attach file-backed FTW banks to an all-CPU MoE cache.

    This is an experimental runtime boundary, not a default loader hook. File-backed mappings
    have no device alias, so every layer must already be CPU-routed and prefill overlap must
    be disabled exactly as ``OffloadMoeCache.set_bank_sources`` requires for PAGEABLE banks.
    Mapping owners are retained on both the tensors and the cache so storage outlives all
    decode/prefill users.
    """

    num_layers = int(cache.num_layers)
    expected_cpu_layers = frozenset(range(num_layers))
    if frozenset(cache.cpu_layer_ids) != expected_cpu_layers:
        raise ValueError(
            "file-backed FTW sources currently require every MoE layer to be CPU-routed"
        )
    if bool(cache.prefill_overlap):
        raise ValueError("file-backed PAGEABLE FTW sources require prefill overlap disabled")

    bundle = map_ftw_expert_sources(
        path,
        num_layers,
        expected_banks=set(cache.bank_schema),
        expected_quant_format=str(cache.quant_format),
        num_experts=(int(cache.num_experts) if num_experts is None else int(num_experts)),
    )
    cache.set_bank_sources(bundle.sources, bundle.layer_residency)
    # Experimental owner anchor. No canonical cache ABI change is needed for this P1 boundary.
    cache._file_backed_ftw_owners = tuple(bundle.owners)
    return bundle


__all__ = [
    "MappedFTWEntry",
    "MappedFTWExpertSources",
    "attach_file_backed_ftw_sources",
    "map_ftw_entry",
    "map_ftw_expert_sources",
]
