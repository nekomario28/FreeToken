from __future__ import annotations

import pytest

from freetoken.experimental.dense_conversion_envelope import (
    estimate_qwen35_mixed_dense_anonymous_peak,
)


_DTYPE_BYTES = {
    "U8": 1,
    "F8_E4M3": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
}


def _spec(dtype: str, shape: list[int], start: int = 0) -> dict:
    count = 1
    for dim in shape:
        count *= dim
    nbytes = count * _DTYPE_BYTES[dtype]
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, start + nbytes]}


def _config(*, lm_head_nvfp4: bool = True) -> dict:
    layers = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
    }
    if lm_head_nvfp4:
        layers["lm_head"] = {"quant_algo": "W4A16_NVFP4"}
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": layers,
        },
    }


def _mixed_headers() -> dict[str, dict]:
    p = "model.language_model.layers.0"
    return {
        # Large passthrough source: file-backed and therefore not a payload-sized anonymous copy.
        "model.language_model.embed_tokens.weight": _spec("BF16", [100, 10]),
        # q/k/v FP8 fusion: output weight 64 B + fp32 row scales 64 B + tiny act slack 16 B.
        f"{p}.self_attn.q_proj.weight": _spec("F8_E4M3", [8, 4]),
        f"{p}.self_attn.q_proj.weight_scale": _spec("F32", [1]),
        f"{p}.self_attn.k_proj.weight": _spec("F8_E4M3", [4, 4]),
        f"{p}.self_attn.k_proj.weight_scale": _spec("F32", [1]),
        f"{p}.self_attn.v_proj.weight": _spec("F8_E4M3", [4, 4]),
        f"{p}.self_attn.v_proj.weight_scale": _spec("F32", [1]),
        # BF16 GDN b|a fusion: 32 + 32 B output.
        f"{p}.linear_attn.in_proj_b.weight": _spec("BF16", [4, 4]),
        f"{p}.linear_attn.in_proj_a.weight": _spec("BF16", [4, 4]),
        # Native NVFP4 shared gate/up: 16 B packed + 8 B block scales + 16 B input
        # globals + 16 B output globals = 56 B local event. One 8 B first-half global may
        # be retained under adversarial iteration order.
        f"{p}.mlp.shared_expert.gate_proj.weight": _spec("U8", [4, 2]),
        f"{p}.mlp.shared_expert.gate_proj.weight_scale": _spec("F8_E4M3", [4, 1]),
        f"{p}.mlp.shared_expert.gate_proj.weight_scale_2": _spec("F32", [1]),
        f"{p}.mlp.shared_expert.up_proj.weight": _spec("U8", [4, 2]),
        f"{p}.mlp.shared_expert.up_proj.weight_scale": _spec("F8_E4M3", [4, 1]),
        f"{p}.mlp.shared_expert.up_proj.weight_scale_2": _spec("F32", [1]),
        f"{p}.mlp.shared_expert.down_proj.weight": _spec("U8", [4, 2]),
        f"{p}.mlp.shared_expert.down_proj.weight_scale": _spec("F8_E4M3", [4, 1]),
        f"{p}.mlp.shared_expert.down_proj.weight_scale_2": _spec("F32", [1]),
        # Native NVFP4 lm_head standalone; only the per-row global vector is anonymous.
        "lm_head.weight": _spec("U8", [16, 4]),
        "lm_head.weight_scale": _spec("F8_E4M3", [16, 1]),
        "lm_head.weight_scale_2": _spec("F32", [1]),
        # Routed expert payload must be excluded entirely from the dense estimate.
        f"{p}.mlp.experts.0.gate_proj.weight": _spec("U8", [1000, 1000]),
    }


def test_large_file_backed_source_does_not_become_anonymous_peak():
    result = estimate_qwen35_mixed_dense_anonymous_peak(_config(), _mixed_headers())
    assert result["largest_file_backed_source_tensor_bytes"] == 2000
    assert result["max_anonymous_bytes_by_operation_class"]["fp8_fusion"] == 144
    assert result["buffered_native_global_upper_bytes"] == 8
    assert result["anonymous_peak_bytes"] == 152
    assert result["writer_payload_copy_bytes"] == 0


def test_operation_classes_remain_separate():
    result = estimate_qwen35_mixed_dense_anonymous_peak(_config(), _mixed_headers())
    maxima = result["max_anonymous_bytes_by_operation_class"]
    assert maxima["bf16_fusion"] == 64
    assert maxima["nvfp4_gate_up_fusion"] == 56
    assert maxima["nvfp4_standalone"] == 32
    assert result["operation_class_counts"]["passthrough"] >= 1


def test_routed_expert_bytes_are_not_counted_as_dense_source_peak():
    result = estimate_qwen35_mixed_dense_anonymous_peak(_config(), _mixed_headers())
    assert result["largest_file_backed_source_tensor_bytes"] < 1_000_000


def test_incomplete_fp8_fusion_fails_closed():
    headers = _mixed_headers()
    headers.pop("model.language_model.layers.0.self_attn.v_proj.weight")
    headers.pop("model.language_model.layers.0.self_attn.v_proj.weight_scale")
    with pytest.raises(ValueError, match="unsupported/incomplete"):
        estimate_qwen35_mixed_dense_anonymous_peak(_config(), headers)


def test_nvfp4_lm_head_requires_config_contract():
    with pytest.raises(ValueError, match="unsupported/incomplete"):
        estimate_qwen35_mixed_dense_anonymous_peak(
            _config(lm_head_nvfp4=False),
            _mixed_headers(),
        )


def test_wrong_model_family_fails_before_estimation():
    config = _config()
    config["architectures"] = ["OtherArchitecture"]
    with pytest.raises(ValueError, match="supports only"):
        estimate_qwen35_mixed_dense_anonymous_peak(config, _mixed_headers())
