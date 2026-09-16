from __future__ import annotations

import torch

from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
from freetoken.models.gguf.dequant import GGML_F32, GGML_Q4_K, row_bytes
from freetoken.models.qwen4_exp.gguf import (
    _GGUFMixedLinear,
    _TensorSpec,
    _grouped_to_tiled_last,
    _v_grouped_to_tiled_perm,
    _v_tiled_to_grouped_perm,
)
from freetoken.models.register import get_model_spec


def test_q4_k_row_bytes_matches_ggml_superblock():
    assert row_bytes(256, GGML_Q4_K) == 144
    assert row_bytes(2560, GGML_Q4_K) == 1440


def test_qwen4exp_gguf_registry_contract():
    assert GGUF_ARCH_TO_REGISTRY["qwen4exp"] == "Qwen4ExpGGUFForCausalLM"
    spec = get_model_spec("Qwen4ExpGGUFForCausalLM")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpGGUFForCausalLM"
    assert spec.parse_config == "parse_gguf_config"
    assert spec.iter_weights == "iter_gguf_weights"


def test_mixed_linear_declares_packed_and_dense_part_state():
    parts = (
        _TensorSpec("q", (64, 256), GGML_Q4_K),
        _TensorSpec("bias_like_projection", (8, 256), GGML_F32),
    )
    layer = _GGUFMixedLinear(256, parts, pad=8)
    state = layer.state_dict()

    assert tuple(state["qweight_0"].shape) == (64, 144)
    assert state["qweight_0"].dtype is torch.uint8
    assert tuple(state["weight_1"].shape) == (8, 256)
    assert layer.out_features == 80
    assert "weight_0" not in state
    assert "qweight_1" not in state


def test_gdn_grouped_tiled_permutations_are_exact_inverses():
    # Small analogue of Qwen3.8 K=16, V=48, D=128: 3 V heads per K head.
    k, v, d = 2, 6, 4
    grouped = torch.arange(v * d)
    forward = _v_grouped_to_tiled_perm(k, v, d)
    inverse = _v_tiled_to_grouped_perm(k, v, d)

    tiled = grouped.index_select(0, forward)
    restored = tiled.index_select(0, inverse)

    assert torch.equal(restored, grouped)
    # Explicit semantic order: [K0:v0,v1,v2,K1:v0,v1,v2] ->
    # [v0:K0,K1, v1:K0,K1, v2:K0,K1], preserving D inside each head.
    heads = torch.arange(v).reshape(k, v // k)
    expected_heads = heads.transpose(0, 1).reshape(-1)
    observed_heads = (forward.reshape(-1, d)[:, 0] // d)
    assert torch.equal(observed_heads, expected_heads)


def test_gdn_output_activation_permutation_matches_index_form():
    k, v, d = 2, 6, 4
    x = torch.arange(3 * v * d, dtype=torch.float32).reshape(3, v * d)
    perm = _v_grouped_to_tiled_perm(k, v, d)

    expected = x.index_select(-1, perm)
    actual = _grouped_to_tiled_last(x, k, v, d)

    assert torch.equal(actual, expected)
