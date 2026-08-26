from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "python" / "freetoken"
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoint"

# Load only the two experiment modules, not freetoken.checkpoint.__init__ and its wider deps.
freetoken_pkg = types.ModuleType("freetoken")
freetoken_pkg.__path__ = [str(PACKAGE_ROOT)]
checkpoint_pkg = types.ModuleType("freetoken.checkpoint")
checkpoint_pkg.__path__ = [str(CHECKPOINT_ROOT)]
sys.modules.setdefault("freetoken", freetoken_pkg)
sys.modules.setdefault("freetoken.checkpoint", checkpoint_pkg)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("freetoken.checkpoint.mapped_ftw_core", CHECKPOINT_ROOT / "mapped_ftw_core.py")
MAPPED = _load("freetoken.checkpoint.mapped_ftw", CHECKPOINT_ROOT / "mapped_ftw.py")


def _write_two_layer_fixture(tmp_path: Path):
    bank_order = ["gate_up_packed", "down_packed"]
    offsets = {}
    raw = bytearray(4 * 4096)
    tensors = []
    values = {}
    slot = 0
    for bank in bank_order:
        for layer in range(2):
            offset = slot * 4096
            payload = bytes([10 * (slot + 1) + i for i in range(8)])
            raw[offset:offset + 8] = payload
            name = f"{bank}#L{layer:05d}"
            offsets[name] = offset
            values[name] = payload
            tensors.append({
                "name": name,
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [2, 4],
                "global_off": offset,
                "nbytes": 8,
            })
            slot += 1
    shard = tmp_path / "freetoken-00000.ftw"
    shard.write_bytes(raw)
    (tmp_path / CORE.INDEX_NAME).write_text(json.dumps({
        "format": "freetoken_weight",
        "version": 1,
        "quant_format": "nvfp4",
        "expert_bank_num_layers": 2,
        "tensors": tensors,
        "shards": [{"file": shard.name, "global_off": 0, "nbytes": len(raw)}],
    }), encoding="utf-8")
    return shard, offsets, values


def test_bundle_matches_existing_bank_sources_contract_and_keeps_owners(tmp_path):
    shard, offsets, values = _write_two_layer_fixture(tmp_path)
    bundle = MAPPED.map_ftw_expert_sources(
        tmp_path,
        2,
        expected_banks={"gate_up_packed", "down_packed"},
        expected_quant_format="nvfp4",
        num_experts=2,
    )

    assert bundle.quant_format == "nvfp4"
    assert bundle.layer_residency == ["pageable", "pageable"]
    assert set(bundle.sources) == {"gate_up_packed", "down_packed"}
    assert all(len(per_layer) == 2 for per_layer in bundle.sources.values())
    assert len(bundle.owners) == 4
    assert all(not owner.mapping.closed for owner in bundle.owners)

    for bank, tensors in bundle.sources.items():
        for layer, tensor in enumerate(tensors):
            name = f"{bank}#L{layer:05d}"
            assert tensor.is_contiguous()
            assert tensor.dtype == torch.uint8
            assert tuple(tensor.shape) == (2, 4)
            assert bytes(tensor.flatten().tolist()) == values[name]

    # The tensor is writable for PyTorch, but ACCESS_COPY must not modify the checkpoint.
    target = bundle.sources["gate_up_packed"][0]
    original_file_byte = shard.read_bytes()[offsets["gate_up_packed#L00000"]]
    target[0, 0] = 255
    assert int(target[0, 0]) == 255
    assert shard.read_bytes()[offsets["gate_up_packed#L00000"]] == original_file_byte


def test_bundle_rejects_wrong_bank_quant_or_expert_contract(tmp_path):
    _write_two_layer_fixture(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        MAPPED.map_ftw_expert_sources(
            tmp_path, 2, expected_banks={"gate_up_packed"}
        )
    with pytest.raises(ValueError, match="quant_format"):
        MAPPED.map_ftw_expert_sources(
            tmp_path, 2, expected_quant_format="nvfp4_marlin"
        )
    with pytest.raises(ValueError, match="experts"):
        MAPPED.map_ftw_expert_sources(tmp_path, 2, num_experts=3)


def test_generic_entry_adapter_checks_shape_dtype_nbytes(tmp_path):
    shard = tmp_path / "freetoken-00000.ftw"
    shard.write_bytes(b"abcdefgh")
    index = {
        "format": "freetoken_weight",
        "version": 1,
        "tensors": [{
            "name": "x", "kind": "experts_bank", "dtype": "uint8",
            "shape": [2, 4], "global_off": 0, "nbytes": 8,
        }],
        "shards": [{"file": shard.name, "global_off": 0, "nbytes": 8}],
    }
    (tmp_path / CORE.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    owner = MAPPED.map_ftw_entry(tmp_path, "x")
    assert tuple(owner.tensor.shape) == (2, 4)
    assert owner.tensor.flatten().tolist() == list(b"abcdefgh")

    index["tensors"][0]["shape"] = [9]
    (tmp_path / CORE.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="shape/dtype"):
        MAPPED.map_ftw_entry(tmp_path, "x")
