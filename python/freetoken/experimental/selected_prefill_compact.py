"""Pure planning primitives for selected-expert compact prefill experiments.

This module intentionally does *not* modify OffloadMoeCache or execute any H2D
movement.  It isolates the semantic transform needed by a future PAGEABLE
selected-prefill path:

    raw expert ids -> sorted selected expert set + compact row ids

The resulting compact ids are valid only against banks gathered in exactly the
``selected_ids`` order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class SelectedExpertPlan:
    """A semantics-preserving raw-id -> compact-row routing plan."""

    selected_ids: torch.Tensor
    compact_ids: torch.Tensor
    num_experts: int
    capacity: int

    @property
    def selected_count(self) -> int:
        return self.selected_ids.numel()

    @property
    def route_count(self) -> int:
        return self.compact_ids.numel()


def compact_prefill_capacity(num_experts: int, max_tokens: int, top_k: int) -> int:
    """Fixed compact-row ceiling for one bounded prefill window."""
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k > num_experts:
        raise ValueError(f"top_k {top_k} > num_experts {num_experts}")
    return min(num_experts, max_tokens * top_k)


def build_selected_expert_plan(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    capacity: int | None = None,
) -> SelectedExpertPlan:
    """Deduplicate routed experts and remap every route into compact row space.

    ``selected_ids`` is sorted for deterministic bank ordering. ``compact_ids``
    has the same shape as ``topk_ids`` and satisfies, for every route ``r``::

        selected_ids[compact_ids[r]] == topk_ids[r]

    This function performs fail-closed range/capacity validation.  A CUDA caller
    may incur host synchronization for the validation checks; that is acceptable
    for the first bounded prefill experiment, which is intentionally not graph
    captured.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got shape {tuple(topk_ids.shape)}")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must have integer dtype, got {topk_ids.dtype}")
    if topk_ids.numel() == 0:
        raise ValueError("topk_ids must not be empty")

    ids64 = topk_ids.to(torch.int64)
    invalid = (ids64 < 0) | (ids64 >= num_experts)
    if bool(invalid.any().item()):
        bad = ids64[invalid][0].item()
        raise ValueError(f"expert id {bad} out of range [0, {num_experts})")

    flat = ids64.reshape(-1)
    selected_ids, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    selected_count = selected_ids.numel()

    if capacity is None:
        capacity = min(num_experts, flat.numel())
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    if capacity > num_experts:
        raise ValueError(f"capacity {capacity} > num_experts {num_experts}")
    if selected_count > capacity:
        raise ValueError(
            f"selected expert count {selected_count} exceeds compact capacity {capacity}"
        )

    compact_ids = inverse.reshape_as(topk_ids).to(torch.int32).contiguous()
    return SelectedExpertPlan(
        selected_ids=selected_ids.contiguous(),
        compact_ids=compact_ids,
        num_experts=num_experts,
        capacity=capacity,
    )


def gather_compact_rows(bank: torch.Tensor, plan: SelectedExpertPlan) -> torch.Tensor:
    """Reference gather for tests/prototypes; not the final PAGEABLE H2D mover.

    ``bank`` must have expert id as dimension 0.  The function returns a new
    tensor whose row ``c`` equals raw expert row ``plan.selected_ids[c]``.
    """
    if bank.ndim < 1:
        raise ValueError("bank must have an expert dimension")
    if bank.shape[0] != plan.num_experts:
        raise ValueError(
            f"bank expert dimension {bank.shape[0]} != num_experts {plan.num_experts}"
        )
    ids = plan.selected_ids.to(device=bank.device, dtype=torch.long)
    return bank.index_select(0, ids)


def compact_buffer_bytes(bytes_per_expert: int, capacity: int) -> int:
    """Physical bytes of a fixed-capacity compact expert buffer."""
    if bytes_per_expert <= 0:
        raise ValueError("bytes_per_expert must be > 0")
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    return bytes_per_expert * capacity


__all__ = [
    "SelectedExpertPlan",
    "build_selected_expert_plan",
    "compact_buffer_bytes",
    "compact_prefill_capacity",
    "gather_compact_rows",
]
