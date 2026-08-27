from __future__ import annotations

import pytest
import torch

from freetoken.experimental.selected_prefill_compact import build_selected_expert_plan
from freetoken.experimental.selected_prefill_pageable import CompactPageablePrefillBanks


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm GPU")


def test_compact_pageable_bank_copies_only_selected_rows_and_reuses_capacity():
    e = 8
    layer0_a = torch.arange(e * 3, dtype=torch.int32).view(e, 3).contiguous()
    layer1_a = (1000 + torch.arange(e * 3, dtype=torch.int32)).view(e, 3).contiguous()
    layer0_b = torch.arange(e * 4, dtype=torch.float32).view(e, 2, 2).contiguous()
    layer1_b = (2000 + torch.arange(e * 4, dtype=torch.float32)).view(e, 2, 2).contiguous()
    sources = {"a": [layer0_a, layer1_a], "b": [layer0_b, layer1_b]}

    banks = CompactPageablePrefillBanks(
        sources,
        num_experts=e,
        capacity=4,
        device="cpu",
    )
    ids = torch.tensor([[7, 1], [1, 4], [7, 4]], dtype=torch.int32)
    plan = build_selected_expert_plan(ids, num_experts=e, capacity=4)
    assert plan.selected_ids.tolist() == [1, 4, 7]

    m0 = banks.materialize(0, plan)
    assert m0.selected_count == 3
    assert torch.equal(m0.views[0], layer0_a[plan.selected_ids])
    assert torch.equal(m0.views[1], layer0_b[plan.selected_ids])
    assert m0.allocated_bytes == 4 * banks.bytes_per_expert
    assert m0.copied_bytes == 3 * banks.bytes_per_expert

    # Same storage is overwritten for the next layer; no extra capacity is allocated.
    m1 = banks.materialize(1, plan)
    assert m1.allocated_bytes == m0.allocated_bytes
    assert torch.equal(m1.views[0], layer1_a[plan.selected_ids])
    assert torch.equal(m1.views[1], layer1_b[plan.selected_ids])


def test_compact_pageable_bank_rejects_capacity_and_geometry_mismatch():
    sources = {"a": [torch.zeros((8, 2), dtype=torch.float32)]}
    banks = CompactPageablePrefillBanks(
        sources,
        num_experts=8,
        capacity=2,
        device="cpu",
    )
    too_wide = build_selected_expert_plan(
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        num_experts=8,
        capacity=3,
    )
    with pytest.raises(ValueError, match="exceeds capacity"):
        banks.materialize(0, too_wide)

    wrong_geometry = build_selected_expert_plan(
        torch.tensor([[0, 1]], dtype=torch.int32),
        num_experts=9,
        capacity=2,
    )
    with pytest.raises(ValueError, match="plan num_experts"):
        banks.materialize(0, wrong_geometry)


def _make_native_nvfp4_sources(*, layers: int, experts: int, h: int, i: int, seed: int):
    """Small PAGEABLE CPU ModelOpt-style NVFP4 banks."""
    g = torch.Generator().manual_seed(seed)
    total = layers * experts

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def rand_scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    flat = {
        "gate_up_packed": rand_u8(total, 2 * i, h // 2),
        "gate_up_scale": rand_scale(total, 2 * i, h // 16),
        "gate_up_global": torch.ones((total, 2 * i), dtype=torch.float16),
        "down_packed": rand_u8(total, h, i // 2),
        "down_scale": rand_scale(total, h, i // 16),
        "down_global": torch.full((total, h), 0.75, dtype=torch.float16),
    }
    # Deliberately leave these PAGEABLE (not pin_memory()).
    return {name: list(t.split(experts)) for name, t in flat.items()}


@cuda
def test_native_nvfp4_full_bank_and_compact_bank_are_numerically_equivalent():
    """P2 synthetic kernel oracle: same kernel, raw/full rows vs compact/remapped rows."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    device = torch.device("cuda")
    layers, e, h, i, topk = 1, 8, 256, 128, 2
    sources = _make_native_nvfp4_sources(layers=layers, experts=e, h=h, i=i, seed=23)

    raw_ids = torch.tensor(
        [[7, 1], [1, 4], [7, 4], [3, 1]],
        dtype=torch.int32,
        device=device,
    )
    plan = build_selected_expert_plan(raw_ids, num_experts=e, capacity=4)
    assert plan.selected_count == 4

    torch.manual_seed(24)
    hidden = torch.randn(4, h, dtype=torch.bfloat16, device=device) / 4
    weights = torch.rand(4, topk, dtype=torch.float32, device=device)

    names = (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    )
    full = [sources[name][0].to(device) for name in names]
    full_out = fused_experts_nvfp4(
        hidden,
        *full,
        weights,
        raw_ids,
        e,
        "silu",
        False,
    )

    compact_banks = CompactPageablePrefillBanks(
        {name: sources[name] for name in names},
        num_experts=e,
        capacity=4,
        device=device,
    )
    materialized = compact_banks.materialize(0, plan)
    assert materialized.selected_count == 4
    assert materialized.allocated_bytes == 4 * compact_banks.bytes_per_expert
    assert materialized.copied_bytes == materialized.allocated_bytes
    assert compact_banks.bytes_per_expert == 56_320

    compact_out = fused_experts_nvfp4(
        hidden,
        *materialized.views,
        weights,
        plan.compact_ids,
        plan.selected_count,
        "silu",
        False,
    )

    # Same routed weights and expert rows, only the expert-id coordinate system changed.
    torch.testing.assert_close(
        compact_out.float(),
        full_out.float(),
        rtol=2e-3,
        atol=2e-3,
    )
