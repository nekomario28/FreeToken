from __future__ import annotations

import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "python" / "freetoken"
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoint"
MOE_ROOT = PACKAGE_ROOT / "moe"

# Load only the experiment modules plus the exact offload_cache source under minimal stubs;
# do not import freetoken.checkpoint.__init__ or initialize CUDA/runtime backends.
freetoken_pkg = types.ModuleType("freetoken")
freetoken_pkg.__path__ = [str(PACKAGE_ROOT)]
checkpoint_pkg = types.ModuleType("freetoken.checkpoint")
checkpoint_pkg.__path__ = [str(CHECKPOINT_ROOT)]
moe_pkg = types.ModuleType("freetoken.moe")
moe_pkg.__path__ = [str(MOE_ROOT)]
sys.modules.setdefault("freetoken", freetoken_pkg)
sys.modules.setdefault("freetoken.checkpoint", checkpoint_pkg)
sys.modules.setdefault("freetoken.moe", moe_pkg)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("freetoken.checkpoint.mapped_ftw_core", CHECKPOINT_ROOT / "mapped_ftw_core.py")
MAPPED = _load("freetoken.checkpoint.mapped_ftw", CHECKPOINT_ROOT / "mapped_ftw.py")


def _write_fixture(tmp_path: Path, bank_order: list[str]):
    offsets = {}
    raw = bytearray(len(bank_order) * 2 * 4096)
    tensors = []
    values = {}
    slot = 0
    for bank in bank_order:
        for layer in range(2):
            offset = slot * 4096
            payload = bytes([(10 * (slot + 1) + i) % 256 for i in range(8)])
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


def _write_two_layer_fixture(tmp_path: Path):
    return _write_fixture(tmp_path, ["gate_up_packed", "down_packed"])


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


def test_exact_offload_cache_accepts_file_backed_sources_as_cpu_pageable(tmp_path):
    native_nvfp4 = [
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    ]
    _write_fixture(tmp_path, native_nvfp4)
    bundle = MAPPED.map_ftw_expert_sources(
        tmp_path,
        2,
        expected_banks=set(native_nvfp4),
        expected_quant_format="nvfp4",
        num_experts=2,
    )

    class _Logger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    utils = types.ModuleType("freetoken.utils")
    utils.init_logger = lambda *_args, **_kwargs: _Logger()
    sys.modules["freetoken.utils"] = utils

    flashlib = types.ModuleType("flashlib")
    kernels = types.ModuleType("flashlib.kernels")
    slot_cache = types.ModuleType("flashlib.kernels.slot_cache")
    slot_cache.N_STATS = 4
    slot_cache.Stat = type("Stat", (), {})
    sys.modules["flashlib"] = flashlib
    sys.modules["flashlib.kernels"] = kernels
    sys.modules["flashlib.kernels.slot_cache"] = slot_cache

    class HostResidency(Enum):
        PINNED = "pinned"
        LOCKED = "locked"
        PAGEABLE = "pageable"

    host_banks = types.ModuleType("freetoken.moe.host_banks")
    host_banks.HostResidency = HostResidency
    sys.modules["freetoken.moe.host_banks"] = host_banks

    offload = _load("freetoken.moe.offload_cache", MOE_ROOT / "offload_cache.py")
    cache = offload.OffloadMoeCache(
        num_layers=2,
        num_experts=2,
        cache_size=2,
        device=torch.device("cpu"),
        quant_format="nvfp4",
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0, 1})
    cache.set_bank_sources(bundle.sources, bundle.layer_residency)

    assert cache.layer_residency == ["pageable", "pageable"]
    assert cache._unpinned_layers == frozenset({0, 1})
    assert set(cache.bank_sources) == set(native_nvfp4)
    assert len(cache.banks) == len(native_nvfp4)
    for name in native_nvfp4:
        assert len(cache.bank_sources[name]) == 2
        assert cache.bank_sources[name][0].data_ptr() == bundle.sources[name][0].data_ptr()
