from __future__ import annotations

import pytest
import torch

from freetoken.experimental.selected_prefill_compact import (
    build_selected_expert_plan,
    compact_buffer_bytes,
    compact_prefill_capacity,
    gather_compact_rows,
)


def _assert_lookup_equivalent(topk_ids: torch.Tensor, *, num_experts: int, width: int = 7):
    bank = torch.arange(num_experts * width, dtype=torch.int64).view(num_experts, width)
    plan = build_selected_expert_plan(topk_ids, num_experts=num_experts)
    compact = gather_compact_rows(bank, plan)

    raw_lookup = bank[topk_ids.long()]
    compact_lookup = compact[plan.compact_ids.long()]
    assert torch.equal(raw_lookup, compact_lookup)
    assert torch.equal(
        plan.selected_ids[plan.compact_ids.long()],
        topk_ids.to(torch.int64),
    )
    return plan


def test_duplicate_and_adversarial_order_are_exactly_remapped():
    ids = torch.tensor(
        [[5, 2, 5, 9], [9, 2, 1, 5], [15, 1, 15, 2]],
        dtype=torch.int32,
    )
    plan = _assert_lookup_equivalent(ids, num_experts=16)
    assert plan.selected_ids.tolist() == [1, 2, 5, 9, 15]
    assert plan.selected_count == 5
    assert plan.route_count == 12


def test_disjoint_two_token_top8_hits_capacity_16_and_stays_exact():
    ids = torch.tensor(
        [list(range(0, 8)), list(range(8, 16))],
        dtype=torch.int32,
    )
    plan = build_selected_expert_plan(ids, num_experts=256, capacity=16)
    assert plan.selected_count == 16
    assert plan.selected_ids.tolist() == list(range(16))

    bank = torch.arange(256 * 3, dtype=torch.int64).view(256, 3)
    compact = gather_compact_rows(bank, plan)
    assert torch.equal(bank[ids.long()], compact[plan.compact_ids.long()])


def test_reuse_heavy_two_token_top8_needs_only_eight_rows():
    ids = torch.tensor(
        [list(range(8)), list(reversed(range(8)))],
        dtype=torch.int32,
    )
    plan = build_selected_expert_plan(ids, num_experts=256, capacity=16)
    assert plan.selected_count == 8
    assert torch.equal(
        plan.selected_ids[plan.compact_ids.long()],
        ids.to(torch.int64),
    )


def test_capacity_is_preregistered_geometry_ceiling():
    assert compact_prefill_capacity(256, max_tokens=2, top_k=8) == 16
    assert compact_prefill_capacity(256, max_tokens=64, top_k=8) == 256

    ids = torch.tensor(
        [list(range(0, 8)), list(range(8, 16))],
        dtype=torch.int32,
    )
    with pytest.raises(ValueError, match="exceeds compact capacity"):
        build_selected_expert_plan(ids, num_experts=256, capacity=15)


def test_invalid_ids_fail_closed():
    with pytest.raises(ValueError, match="out of range"):
        build_selected_expert_plan(
            torch.tensor([[0, -1]], dtype=torch.int32),
            num_experts=256,
        )
    with pytest.raises(ValueError, match="out of range"):
        build_selected_expert_plan(
            torch.tensor([[0, 256]], dtype=torch.int32),
            num_experts=256,
        )


def test_non_integer_or_wrong_rank_fails_closed():
    with pytest.raises(TypeError, match="integer dtype"):
        build_selected_expert_plan(torch.tensor([[0.0, 1.0]]), num_experts=8)
    with pytest.raises(ValueError, match="rank-2"):
        build_selected_expert_plan(torch.tensor([0, 1], dtype=torch.int32), num_experts=8)


def test_qwen35_two_token_geometry_distinguishes_best_and_worst_union():
    bytes_per_expert = 1_775_616
    full_layer = 454_557_696
    capacity = compact_prefill_capacity(256, max_tokens=2, top_k=8)

    worst = compact_buffer_bytes(bytes_per_expert, capacity)
    reuse_heavy = compact_buffer_bytes(bytes_per_expert, 8)

    assert worst == 28_409_856
    assert reuse_heavy == 14_204_928
    assert full_layer // worst == 16
    assert full_layer // reuse_heavy == 32
