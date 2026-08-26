from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/freetoken/experimental/ftw_resource_gate.py"
SPEC = importlib.util.spec_from_file_location("ftw_resource_gate_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _checkpoint(tmp_path: Path):
    shard = "model-00001-of-00001.safetensors"
    tensors = {
        "a": torch.arange(100, dtype=torch.uint8),
        "b": torch.arange(10, dtype=torch.float32),
    }
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    return tensors


def _config():
    return SimpleNamespace(
        num_experts=2,
        hidden_size=16,
        moe_intermediate_size=16,
        num_moe_layers=2,
    )


def test_exact_native_nvfp4_geometry_and_real_candidate_fragment_bound():
    assert GATE.native_nvfp4_expert_layer_bytes(_config()) == 1056
    real = SimpleNamespace(
        num_experts=256,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_moe_layers=40,
    )
    assert GATE.native_nvfp4_expert_layer_bytes(real) == 454_557_696
    fragment = GATE.native_nvfp4_fragment_memory(real)
    assert fragment == {
        "file_backed_source_fragment_peak_bytes": 524_288,
        "anonymous_generated_fragment_peak_bytes": 4_096,
    }
    with pytest.raises(ValueError, match="divisible by 16"):
        GATE.native_nvfp4_expert_layer_bytes(
            SimpleNamespace(
                num_experts=1,
                hidden_size=15,
                moe_intermediate_size=16,
                num_moe_layers=1,
            )
        )


def test_preflight_legacy_fallback_remains_fail_closed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    out = tmp_path / "out"

    huge = 1 << 50
    report = GATE.preflight_low_memory_nvfp4_conversion(
        source,
        out,
        _config(),
        disk_free_bytes=huge,
        mem_available_bytes=huge,
    )

    assert report.source_shards == 1
    assert report.source_tensor_count == 2
    assert report.source_tensor_bytes == 140
    assert report.largest_source_tensor_bytes == 100
    assert report.expert_layer_bytes == 1056
    assert report.expert_bank_total_bytes == 2112
    assert report.ram_guard_bytes == 1056 + 8 * 100 + GATE.DISk_FIXED_HEADROOM_BYTES if False else report.ram_guard_bytes
    assert report.ram_guard_bytes == 1056 + 8 * 100 + GATE.DISK_FIXED_HEADROOM_BYTES
    assert report.ram_model == "legacy_conservative_fallback"
    assert report.disk_model == "native_ftw_representation_envelope"
    assert report.output_guard_bytes == (
        report.disk_payload_envelope_bytes + GATE.DISK_POST_CONVERSION_RESERVE_BYTES
    )
    assert not out.exists()


def test_exact_dense_uses_fragment_expert_and_phase_max(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    out = tmp_path / "out"
    huge = 1 << 50

    report = GATE.preflight_low_memory_nvfp4_conversion(
        source,
        out,
        _config(),
        dense_anonymous_peak_bytes=400,
        disk_free_bytes=huge,
        mem_available_bytes=huge,
    )

    assert report.expert_layer_bytes == 1056  # diagnostic physical layer only
    assert report.expert_file_backed_fragment_peak_bytes == 128
    assert report.expert_anonymous_peak_bytes == 32
    assert report.dense_anonymous_peak_bytes == 400
    assert report.phase_anonymous_peak_bytes == 400
    assert report.ram_guard_bytes == 400 + GATE.RUNTIME_MARGIN_BYTES
    assert report.ram_model == "phase_max_exact_dense_fragment_expert_auto"
    assert report.largest_source_tensor_bytes == 100


def test_explicit_fragment_override_and_dense_dominance(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    report = GATE.preflight_low_memory_nvfp4_conversion(
        source,
        tmp_path / "out",
        _config(),
        dense_anonymous_peak_bytes=4096,
        expert_anonymous_peak_bytes=7,
        expert_file_backed_fragment_peak_bytes=99,
        disk_free_bytes=1 << 50,
        mem_available_bytes=1 << 50,
    )
    assert report.expert_anonymous_peak_bytes == 7
    assert report.expert_file_backed_fragment_peak_bytes == 99
    assert report.phase_anonymous_peak_bytes == 4096
    assert report.ram_guard_bytes == 4096 + GATE.RUNTIME_MARGIN_BYTES
    assert report.ram_model == "phase_max_exact_dense_fragment_expert"


def test_representation_disk_envelope_is_not_generic_four_x(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    report = GATE.preflight_low_memory_nvfp4_conversion(
        source,
        tmp_path / "out",
        _config(),
        dense_anonymous_peak_bytes=0,
        disk_free_bytes=1 << 50,
        mem_available_bytes=1 << 50,
    )
    expected_checkpoint = (report.source_file_bytes * 135 + 99) // 100
    expected_native = report.expert_bank_total_bytes + (
        report.non_routed_source_tensor_bytes * 150 + 99
    ) // 100
    assert report.disk_payload_envelope_bytes == max(expected_checkpoint, expected_native)
    assert report.output_guard_bytes == (
        report.disk_payload_envelope_bytes + GATE.DISK_POST_CONVERSION_RESERVE_BYTES
    )


def test_runtime_recheck_observes_post_import_memory_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    report = GATE.preflight_low_memory_nvfp4_conversion(
        source,
        tmp_path / "out",
        _config(),
        dense_anonymous_peak_bytes=0,
        disk_free_bytes=1 << 50,
        mem_available_bytes=1 << 50,
    )
    assert GATE.require_current_memory(
        report, mem_available_bytes=report.ram_guard_bytes
    ) == report.ram_guard_bytes
    with pytest.raises(RuntimeError, match="after runtime initialization"):
        GATE.require_current_memory(
            report, mem_available_bytes=report.ram_guard_bytes - 1
        )


def test_header_loader_returns_metadata_without_payload_reads(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    headers = GATE.load_safetensors_headers(source)
    assert set(headers) == {"a", "b"}
    assert headers["a"]["dtype"] == "U8"
    assert headers["b"]["dtype"] == "F32"


def test_preflight_blocks_output_inside_source_tree_before_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)

    with pytest.raises(ValueError, match="outside the source checkpoint tree"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source,
            source / "converted-ftw",
            _config(),
            disk_free_bytes=1 << 50,
            mem_available_bytes=1 << 50,
        )
    assert not (source / "converted-ftw").exists()

    with pytest.raises(ValueError, match="outside the source checkpoint tree"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source,
            source,
            _config(),
            disk_free_bytes=1 << 50,
            mem_available_bytes=1 << 50,
        )


def test_preflight_blocks_low_disk_or_low_ram_before_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="output filesystem"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source, out, _config(), disk_free_bytes=1, mem_available_bytes=1 << 50
        )
    assert not out.exists()

    with pytest.raises(RuntimeError, match="MemAvailable"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source,
            out,
            _config(),
            dense_anonymous_peak_bytes=0,
            disk_free_bytes=1 << 50,
            mem_available_bytes=1,
        )
    assert not out.exists()


def test_preflight_rejects_nonempty_output_and_missing_or_escaping_shard(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)

    out = tmp_path / "out"
    out.mkdir()
    (out / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source, out, _config(), disk_free_bytes=1 << 50, mem_available_bytes=1 << 50
        )

    clean_out = tmp_path / "clean-out"
    (source / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="shard is missing"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source, clean_out, _config(), disk_free_bytes=1 << 50, mem_available_bytes=1 << 50
        )

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not-used")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escapes checkpoint directory"):
        GATE.preflight_low_memory_nvfp4_conversion(
            source, clean_out, _config(), disk_free_bytes=1 << 50, mem_available_bytes=1 << 50
        )
