"""Explicit CPU-only ExpertBanks provider backed directly by streamed FTW mappings.

This module is deliberately opt-in and is not called by the canonical loader. It exists so a
bounded experimental server can exercise the already-validated file-backed source ABI without
changing normal FTW loading or GPU/offload behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.moe.expert_banks import ExpertBanks

from .mapped_ftw import MappedFTWEntry, map_ftw_expert_sources

_NATIVE_NVFP4_BANKS = frozenset({
    "gate_up_packed",
    "gate_up_scale",
    "gate_up_global",
    "down_packed",
    "down_scale",
    "down_global",
})


@dataclass(frozen=True)
class FileBackedExpertBanks(ExpertBanks):
    """Normal ``ExpertBanks`` plus explicit owners for mmap-backed tensor storage."""

    storage_owners: tuple[MappedFTWEntry, ...] = field(default_factory=tuple, repr=False)


def load_file_backed_ftw_cpu_banks(
    model_path: str,
    model_config,
    *,
    decode_target: str,
    layer_residency: list[str] | None,
) -> FileBackedExpertBanks:
    """Return native-NVFP4 file-backed banks for an all-CPU decode experiment.

    The caller must already have selected CPU decode for every MoE layer. Canonical engine
    planning currently requests LOCKED for CPU layers; this provider intentionally returns
    PAGEABLE labels because file mappings are neither pinned nor mlocked. Any PINNED request
    is rejected rather than silently changing a GPU-addressable bank into a CPU-only one.
    """

    if decode_target != "cpu":
        raise ValueError("file-backed FTW provider currently supports decode_target='cpu' only")
    num_layers = int(model_config.num_moe_layers)
    num_experts = int(model_config.num_experts)
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError("file-backed FTW provider requires positive MoE layer/expert counts")
    if layer_residency is not None:
        if len(layer_residency) != num_layers:
            raise ValueError("file-backed FTW layer_residency length mismatch")
        if any(label == "pinned" for label in layer_residency):
            raise ValueError(
                "file-backed FTW provider cannot satisfy PINNED/GPU-addressable residency"
            )

    bundle = map_ftw_expert_sources(
        model_path,
        num_layers,
        expected_banks=_NATIVE_NVFP4_BANKS,
        expected_quant_format="nvfp4",
        num_experts=num_experts,
    )
    return FileBackedExpertBanks(
        quant_format=bundle.quant_format,
        sources=bundle.sources,
        layer_residency=bundle.layer_residency,
        storage_owners=tuple(bundle.owners),
    )


__all__ = ["FileBackedExpertBanks", "load_file_backed_ftw_cpu_banks"]
