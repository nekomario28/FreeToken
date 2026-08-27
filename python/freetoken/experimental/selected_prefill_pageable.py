"""Experimental fixed-capacity PAGEABLE -> device compact prefill bank.

This is deliberately separate from :class:`OffloadMoeCache`.  It exists to test
the DRSF/FreeToken hypothesis without relaxing the production cache invariant
``cache_size >= num_experts`` or changing decode semantics.

The first implementation favors attribution over throughput: selected rows are
copied synchronously one row at a time from CPU sources into a fixed-capacity
buffer.  No full-layer fallback is provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .selected_prefill_compact import SelectedExpertPlan


@dataclass(frozen=True)
class CompactMaterialization:
    views: tuple[torch.Tensor, ...]
    selected_count: int
    allocated_bytes: int
    copied_bytes: int


class CompactPageablePrefillBanks:
    """Reusable per-bank device buffers with a hard expert-row capacity."""

    def __init__(
        self,
        sources: Mapping[str, Sequence[torch.Tensor]],
        *,
        num_experts: int,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if capacity <= 0 or capacity > num_experts:
            raise ValueError(
                f"capacity must be in [1, {num_experts}], got {capacity}"
            )
        if not sources:
            raise ValueError("sources must not be empty")

        self.num_experts = num_experts
        self.capacity = capacity
        self.device = torch.device(device)
        self.names = tuple(sources.keys())
        self.sources: dict[str, tuple[torch.Tensor, ...]] = {}
        self.buffers: dict[str, torch.Tensor] = {}
        self._row_bytes: dict[str, int] = {}

        num_layers: int | None = None
        for name in self.names:
            per_layer = tuple(sources[name])
            if not per_layer:
                raise ValueError(f"bank {name!r} has no layers")
            if num_layers is None:
                num_layers = len(per_layer)
            elif len(per_layer) != num_layers:
                raise ValueError(
                    f"bank {name!r} layer count {len(per_layer)} != {num_layers}"
                )

            head = per_layer[0]
            if head.device.type != "cpu":
                raise ValueError(
                    f"bank {name!r} must be CPU/PAGEABLE source, got {head.device}"
                )
            if head.shape[0] != num_experts:
                raise ValueError(
                    f"bank {name!r} expert dimension {head.shape[0]} != {num_experts}"
                )
            for layer_id, src in enumerate(per_layer):
                if src.device.type != "cpu":
                    raise ValueError(
                        f"bank {name!r} layer {layer_id} must be CPU source, got {src.device}"
                    )
                if not src.is_contiguous():
                    raise ValueError(
                        f"bank {name!r} layer {layer_id} must be contiguous"
                    )
                if src.shape != head.shape or src.dtype != head.dtype:
                    raise ValueError(
                        f"bank {name!r} layer {layer_id} shape/dtype mismatch"
                    )

            self.sources[name] = per_layer
            self.buffers[name] = torch.empty(
                (capacity, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
            self._row_bytes[name] = math.prod(head.shape[1:]) * head.element_size()

        assert num_layers is not None
        self.num_layers = num_layers

    @property
    def bytes_per_expert(self) -> int:
        return sum(self._row_bytes.values())

    @property
    def allocated_bytes(self) -> int:
        return self.capacity * self.bytes_per_expert

    def materialize(
        self,
        layer_id: int,
        plan: SelectedExpertPlan,
    ) -> CompactMaterialization:
        """Copy only the plan's selected rows into compact positions ``[0,U)``.

        The row loop is intentional for the first experiment: it does not create a
        CPU gathered tensor of size ``U`` and it makes copied-byte accounting exact.
        A future optimized mover may batch/coalesce copies only after this path is
        numerically and memory validated.
        """
        if plan.num_experts != self.num_experts:
            raise ValueError(
                f"plan num_experts {plan.num_experts} != bank num_experts {self.num_experts}"
            )
        if plan.selected_count > self.capacity:
            raise ValueError(
                f"selected expert count {plan.selected_count} exceeds capacity {self.capacity}"
            )
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )

        # First bounded prefill is not graph captured; the explicit host sync is
        # therefore part of the preregistered movement path rather than hidden.
        selected = plan.selected_ids.to(device="cpu", dtype=torch.long).tolist()
        for name in self.names:
            src = self.sources[name][layer_id]
            dst = self.buffers[name]
            for compact_row, raw_expert in enumerate(selected):
                dst[compact_row].copy_(src[raw_expert])

        u = plan.selected_count
        return CompactMaterialization(
            views=tuple(self.buffers[name][:u] for name in self.names),
            selected_count=u,
            allocated_bytes=self.allocated_bytes,
            copied_bytes=u * self.bytes_per_expert,
        )


__all__ = ["CompactMaterialization", "CompactPageablePrefillBanks"]
