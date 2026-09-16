from __future__ import annotations

import torch

from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
from freetoken.models.gguf.dequant import GGML_F32, GGML_Q4_K, row_bytes
from freetoken.models.qwen4_exp.gguf import _GGUFMixedLinear, _TensorSpec
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
