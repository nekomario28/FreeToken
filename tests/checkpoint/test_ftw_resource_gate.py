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
    return SimpleNamespace(num_experts=2, hidden_size=16, moe_intermediate_size=16)


def test_exact_native_nvfp4_one_layer_bytes():
    assert GATE.native_nvfp4_expert_layer_bytes(_config()) == 1056
    with pytest.raises(ValueError, match="divisible by 16"):
        GATE.native_nvfp4_expert_layer_bytes(
            SimpleNamespace(num_experts=1, hidden_size=15, moe_intermediate_size=16)
        )


def test_preflight_reads_only_headers_and_returns_conservative_guards(tmp_path):
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
    assert report.output_guard_bytes == (
        4 * 140 + 2 * GATE.ALIGN + GATE.FIXED_HEADROOM_BYTES
    )
    assert report.ram_guard_bytes == (
        1056 + 8 * 100 + GATE.FIXED_HEADROOM_BYTES
    )
    assert report.as_dict()["source_tensor_bytes"] == 140
    assert not out.exists()


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
            source, out, _config(), disk_free_bytes=1 << 50, mem_available_bytes=1
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
