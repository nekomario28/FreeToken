from __future__ import annotations

import pytest
import torch

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


def test_adapter_requires_canonical_native_nvfp4_bank_order():
    wrong = tuple(reversed(NVFP4_BANK_ORDER))
    with pytest.raises(ValueError, match="bank order mismatch"):
        NativeNvfp4CompactPrefillAdapter(
            _tiny_sources(wrong),
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
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=4,
    )
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
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=8,
        top_k=2,
        hidden_size=8,
        intermediate_size=4,
    )
    adapter = NativeNvfp4CompactPrefillAdapter(
        _tiny_sources(),
        num_experts=4,
        capacity=2,
        device="cpu",
    )
    with pytest.raises(ValueError, match="num_experts"):
        bind_compact_prefill_adapter(layer, adapter)
