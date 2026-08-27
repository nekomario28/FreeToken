from __future__ import annotations

import pytest
import torch

from freetoken.distributed import set_tp_info
from freetoken.experimental.selected_prefill_adapter import (
    NVFP4_BANK_ORDER,
    NativeNvfp4CompactPrefillAdapter,
    bind_compact_prefill_adapter,
)
from freetoken.layers.moe import OffloadMoELayer


def _tiny_sources(order=NVFP4_BANK_ORDER):
    e = 4
    shapes = {
        "gate_up_packed": (e, 8, 4),
        "gate_up_scale": (e, 8, 1),
        "gate_up_global": (e, 8),
        "down_packed": (e, 8, 2),
        "down_scale": (e, 8, 1),
        "down_global": (e, 8),
    }
    dtypes = {
        "gate_up_packed": torch.uint8,
        "gate_up_scale": torch.float32,
        "gate_up_global": torch.float16,
        "down_packed": torch.uint8,
        "down_scale": torch.float32,
        "down_global": torch.float16,
    }
    return {
        name: [torch.zeros(shapes[name], dtype=dtypes[name]).contiguous()]
        for name in order
    }


def _offload_layer(*, num_experts: int) -> OffloadMoELayer:
    set_tp_info(0, 1)
    return OffloadMoELayer(
        layer_id=0,
        num_experts=num_experts,
        top_k=2,
        hidden_size=8,
        intermediate_size=4,
    )


def test_adapter_canonicalizes_real_loader_style_bank_order():
    loader_style = (
        "down_global",
        "down_packed",
        "down_scale",
        "gate_up_global",
        "gate_up_packed",
        "gate_up_scale",
    )
    adapter = NativeNvfp4CompactPrefillAdapter(
        _tiny_sources(loader_style),
        num_experts=4,
        capacity=2,
        device="cpu",
    )
    assert tuple(adapter.banks.names) == NVFP4_BANK_ORDER


def test_adapter_rejects_missing_or_unknown_bank_names():
    missing = _tiny_sources()
    missing.pop("down_global")
    with pytest.raises(ValueError, match="bank set mismatch"):
        NativeNvfp4CompactPrefillAdapter(
            missing,
            num_experts=4,
            capacity=2,
            device="cpu",
        )

    extra = _tiny_sources()
    extra["unexpected"] = [torch.zeros((4, 1), dtype=torch.float32)]
    with pytest.raises(ValueError, match="bank set mismatch"):
        NativeNvfp4CompactPrefillAdapter(
            extra,
            num_experts=4,
            capacity=2,
            device="cpu",
        )


def test_adapter_allocates_only_fixed_compact_capacity_and_starts_empty():
    adapter = NativeNvfp4CompactPrefillAdapter(
        _tiny_sources(),
        num_experts=4,
        capacity=2,
        device="cpu",
    )
    expected = sum(
        int(t[0][0].numel()) * int(t[0][0].element_size())
        for t in _tiny_sources().values()
    ) * 2
    assert adapter.allocated_bytes == expected
    summary = adapter.summary()
    assert summary["layer_calls"] == 0
    assert summary["selected_min"] is None
    assert summary["selected_max"] is None
    assert summary["copied_bytes"] == 0
    assert summary["allocated_bytes"] == expected


def test_binding_does_not_attach_production_offload_cache():
    layer = _offload_layer(num_experts=4)
    adapter = NativeNvfp4CompactPrefillAdapter(
        _tiny_sources(),
        num_experts=4,
        capacity=2,
        device="cpu",
    )
    layers = bind_compact_prefill_adapter(layer, adapter)
    assert layers == [layer]
    assert layer.offload_cache is None
    assert layer._prefill_routed.__self__ is layer


def test_binding_rejects_geometry_drift():
    layer = _offload_layer(num_experts=8)
    adapter = NativeNvfp4CompactPrefillAdapter(
        _tiny_sources(),
        num_experts=4,
        capacity=2,
        device="cpu",
    )
    with pytest.raises(ValueError, match="num_experts"):
        bind_compact_prefill_adapter(layer, adapter)
