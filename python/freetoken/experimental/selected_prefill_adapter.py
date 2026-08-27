"""Prefill-only adapter for fixed-capacity selected-expert NVFP4 banks.

This module is intentionally experimental and does not alter production cache
semantics. It binds only ``OffloadMoELayer._prefill_routed`` and therefore does
not construct an :class:`OffloadMoeCache`, decode slot map, LRU state, or CPU MoE
executor. The target is the bounded one-forward carrier after one-real-layer
validation is green.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch

from freetoken.core import get_global_ctx
from freetoken.moe.offload_cache import iter_offload_moe_layers

from .selected_prefill_compact import build_selected_expert_plan
from .selected_prefill_pageable import CompactPageablePrefillBanks


NVFP4_BANK_ORDER = (
    "gate_up_packed",
    "gate_up_scale",
    "gate_up_global",
    "down_packed",
    "down_scale",
    "down_global",
)


@dataclass(frozen=True)
class CompactPrefillLayerReceipt:
    layer_id: int
    selected_count: int
    copied_bytes: int
    allocated_bytes: int


class NativeNvfp4CompactPrefillAdapter:
    """Shared fixed-capacity bank plus per-layer compact prefill dispatch."""

    def __init__(
        self,
        sources,
        *,
        num_experts: int,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        keys = tuple(sources.keys())
        expected = set(NVFP4_BANK_ORDER)
        observed = set(keys)
        if observed != expected or len(keys) != len(NVFP4_BANK_ORDER):
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"native NVFP4 bank set mismatch: missing={missing}, extra={extra}, keys={keys!r}"
            )

        self.num_experts = num_experts
        self.capacity = capacity
        self.device = torch.device(device)
        # The real file-backed loader currently emits these banks in a different
        # dictionary order. Kernel argument order is the semantic contract, so
        # canonicalize explicitly here rather than making loader insertion order
        # part of the runtime API.
        self.banks = CompactPageablePrefillBanks(
            {name: sources[name] for name in NVFP4_BANK_ORDER},
            num_experts=num_experts,
            capacity=capacity,
            device=self.device,
        )
        if tuple(self.banks.names) != NVFP4_BANK_ORDER:
            raise AssertionError("internal NVFP4 bank canonicalization failed")
        self.receipts: list[CompactPrefillLayerReceipt] = []

    @property
    def allocated_bytes(self) -> int:
        return self.banks.allocated_bytes

    @property
    def bytes_per_expert(self) -> int:
        return self.banks.bytes_per_expert

    def routed_prefill(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        raw_topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """Materialize selected rows and invoke the native NVFP4 prefill kernel."""
        if hidden_states.device != self.device:
            raise ValueError(
                f"hidden_states device {hidden_states.device} != adapter device {self.device}"
            )
        plan = build_selected_expert_plan(
            raw_topk_ids,
            num_experts=self.num_experts,
            capacity=self.capacity,
        )
        materialized = self.banks.materialize(layer_id, plan)
        if materialized.selected_count != plan.selected_count:
            raise AssertionError("materialization selected-count drift")
        if materialized.allocated_bytes != self.allocated_bytes:
            raise AssertionError("compact allocation changed during prefill")

        from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

        out = fused_experts_nvfp4(
            hidden_states,
            *materialized.views,
            topk_weights,
            plan.compact_ids,
            plan.selected_count,
            activation,
            apply_router_weight_on_input,
        )
        self.receipts.append(
            CompactPrefillLayerReceipt(
                layer_id=layer_id,
                selected_count=plan.selected_count,
                copied_bytes=materialized.copied_bytes,
                allocated_bytes=materialized.allocated_bytes,
            )
        )
        return out

    def summary(self) -> dict[str, int | float | None]:
        if not self.receipts:
            return {
                "layer_calls": 0,
                "selected_min": None,
                "selected_max": None,
                "selected_sum": 0,
                "selected_mean": None,
                "copied_bytes": 0,
                "allocated_bytes": self.allocated_bytes,
            }
        counts = [item.selected_count for item in self.receipts]
        copied = sum(item.copied_bytes for item in self.receipts)
        return {
            "layer_calls": len(self.receipts),
            "selected_min": min(counts),
            "selected_max": max(counts),
            "selected_sum": sum(counts),
            "selected_mean": sum(counts) / len(counts),
            "copied_bytes": copied,
            "allocated_bytes": self.allocated_bytes,
        }


def bind_compact_prefill_adapter(model, adapter: NativeNvfp4CompactPrefillAdapter) -> list:
    """Bind the adapter to every generic OffloadMoELayer in ``model``.

    The bound method refuses non-prefill invocation and verifies each layer's
    geometry at binding/call time. Decode methods and ``offload_cache`` remain
    untouched; the bounded carrier must not invoke them.
    """
    from freetoken.layers import OffloadMoELayer

    layers = list(iter_offload_moe_layers(model))
    if not layers:
        raise ValueError("model exposes no offload MoE layers")

    for layer in layers:
        if not isinstance(layer, OffloadMoELayer):
            raise TypeError(
                f"compact prefill adapter supports OffloadMoELayer only, got {type(layer).__name__}"
            )
        if int(layer.num_experts) != adapter.num_experts:
            raise ValueError(
                f"layer {layer.layer_id} num_experts {layer.num_experts} != {adapter.num_experts}"
            )

        def _compact_prefill_routed(
            self,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            *,
            _adapter=adapter,
        ) -> torch.Tensor:
            ctx = get_global_ctx()
            if not ctx.batch.is_prefill:
                raise RuntimeError("compact adapter is prefill-only")
            if self.offload_cache is not None:
                raise RuntimeError(
                    "compact prefill carrier must not attach OffloadMoeCache; "
                    "that would reintroduce a full-layer expert allocation"
                )
            return _adapter.routed_prefill(
                layer_id=int(self.layer_id),
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                raw_topk_ids=topk_ids,
                activation=str(self.activation),
                apply_router_weight_on_input=bool(self.apply_router_weight_on_input),
            )

        layer._prefill_routed = types.MethodType(_compact_prefill_routed, layer)

    return layers


__all__ = [
    "CompactPrefillLayerReceipt",
    "NVFP4_BANK_ORDER",
    "NativeNvfp4CompactPrefillAdapter",
    "bind_compact_prefill_adapter",
]
