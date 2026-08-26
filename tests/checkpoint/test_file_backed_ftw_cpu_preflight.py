from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/freetoken/experimental/file_backed_ftw_cpu_preflight.py"
SPEC = importlib.util.spec_from_file_location("file_backed_ftw_cpu_preflight_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
preflight = MODULE.preflight_file_backed_ftw_cpu_conversion
layer_bytes = MODULE.native_nvfp4_layer_bytes


def _write_checkpoint(root: Path, *, arch="Qwen3_5MoeForConditionalGeneration", algo="NVFP4"):
    root.mkdir()
    config = {
        "architectures": [arch],
        "quantization_config": {"quant_algo": algo},
        "text_config": {
            "num_hidden_layers": 2,
            "num_experts": 4,
            "hidden_size": 32,
            "moe_intermediate_size": 48,
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "a.safetensors").write_bytes(b"a" * 11)
    (root / "b.safetensors").write_bytes(b"b" * 17)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}),
        encoding="utf-8",
    )


def test_layer_bytes_matches_six_native_bank_formula():
    E, H, I = 4, 32, 48
    expected = (
        E * 2 * I * (H // 2)
        + E * 2 * I * (H // 16)
        + E * 2 * I * 2
        + E * H * (I // 2)
        + E * H * (I // 16)
        + E * H * 2
    )
    assert layer_bytes(num_experts=E, hidden_size=H, intermediate_size=I) == expected
    with pytest.raises(ValueError, match="divisible by 16"):
        layer_bytes(num_experts=E, hidden_size=31, intermediate_size=I)


def test_metadata_only_preflight_reports_shape_capacity_and_no_payload_open(tmp_path):
    source = tmp_path / "model"
    output = tmp_path / "out"
    _write_checkpoint(source)

    result = preflight(str(source), str(output))

    assert result["admission"] == "METADATA_OK_RESOURCE_UNPROVEN"
    assert result["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert result["expert_quant"] == "nvfp4"
    assert result["num_moe_layers"] == 2
    assert result["source_shard_count"] == 2
    assert result["source_shard_bytes"] == 28
    assert result["largest_source_shard_bytes"] == 17
    assert result["native_expert_total_bytes"] == 2 * result["native_expert_layer_bytes"]
    assert result["payload_tensors_opened"] is False
    assert result["blockers"] == []


def test_preflight_blocks_wrong_format_unsafe_output_and_shard_escape(tmp_path):
    source = tmp_path / "model"
    _write_checkpoint(source, arch="OtherModel", algo="FP8")
    result = preflight(str(source), str(source / "out"))
    assert result["admission"] == "BLOCK"
    assert any("output directory" in item for item in result["blockers"])
    assert any("architecture" in item for item in result["blockers"])
    assert any("native ModelOpt NVFP4" in item for item in result["blockers"])

    cfg = json.loads((source / "config.json").read_text())
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    cfg["quantization_config"] = {"quant_algo": "NVFP4"}
    (source / "config.json").write_text(json.dumps(cfg))
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"x")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    result = preflight(str(source), str(tmp_path / "safe-out"))
    assert result["admission"] == "BLOCK"
    assert any("escapes checkpoint root" in item for item in result["blockers"])


def test_preflight_supports_mixed_precision_nvfp4_expert_detection(tmp_path):
    source = tmp_path / "model"
    _write_checkpoint(source, algo="MIXED_PRECISION")
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"]["quantized_layers"] = {
        "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"}
    }
    (source / "config.json").write_text(json.dumps(config))
    result = preflight(str(source), str(tmp_path / "out"))
    assert result["admission"] == "METADATA_OK_RESOURCE_UNPROVEN"
    assert result["expert_quant"] == "nvfp4"
