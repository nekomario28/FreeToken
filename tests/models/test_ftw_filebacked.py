from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import pytest
import torch


_MODULE_PATH = Path(__file__).resolve().parents[2] / "python/freetoken/checkpoint/ftw_filebacked.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("freetoken_ftw_filebacked_under_test", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
map_ftw_expert_entry = _MODULE.map_ftw_expert_entry
map_ftw_cpu_layer_sources = _MODULE.map_ftw_cpu_layer_sources
map_ftw_pageable_layer_sources = _MODULE.map_ftw_pageable_layer_sources


def _write_index(tmp_path: Path, *, tensors: list[dict], shards: list[dict]) -> None:
    (tmp_path / "freetoken_weight.json").write_text(
        json.dumps(
            {
                "format": "freetoken_weight",
                "version": 1,
                "align": 4096,
                "shard_limit": 1 << 20,
                "total_bytes": sum(int(s["nbytes"]) for s in shards),
                "tensors": tensors,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )


def _single_entry_fixture(tmp_path: Path, *, kind: str = "experts_bank", nbytes: int = 8):
    shard = tmp_path / "freetoken-00000.ftw"
    payload = bytes(range(1, 9))
    raw = bytearray(8192)
    raw[4096 : 4096 + len(payload)] = payload
    shard.write_bytes(raw)
    entry = {
        "name": "gate_up_packed#L00000",
        "kind": kind,
        "dtype": "uint8",
        "shape": [2, 4],
        "global_off": 4096,
        "nbytes": nbytes,
    }
    _write_index(
        tmp_path,
        tensors=[entry],
        shards=[{"file": shard.name, "global_off": 0, "nbytes": len(raw)}],
    )
    return shard, payload, entry


def _two_layer_fixture(tmp_path: Path):
    shard = tmp_path / "freetoken-00000.ftw"
    raw = bytearray(5 * 4096)
    entries = []
    payloads = {}
    offsets = iter((4096, 8192, 12288, 16384))
    for layer in range(2):
        for bank, base in (("gate_up_packed", 1), ("down_packed", 5)):
            off = next(offsets)
            payload = bytes(base + 10 * layer + i for i in range(4))
            raw[off : off + 4] = payload
            name = f"{bank}#L{layer:05d}"
            payloads[(bank, layer)] = payload
            entries.append(
                {
                    "name": name,
                    "kind": "experts_bank",
                    "dtype": "uint8",
                    "shape": [1, 4],
                    "global_off": off,
                    "nbytes": 4,
                }
            )
    shard.write_bytes(raw)
    _write_index(
        tmp_path,
        tensors=entries,
        shards=[{"file": shard.name, "global_off": 0, "nbytes": len(raw)}],
    )
    return shard, payloads


def test_private_mapping_survives_function_scope_and_does_not_modify_checkpoint(tmp_path):
    shard, payload, _entry = _single_entry_fixture(tmp_path)

    tensor = map_ftw_expert_entry(tmp_path, "gate_up_packed#L00000")
    gc.collect()
    assert tensor.dtype == torch.uint8
    assert tuple(tensor.shape) == (2, 4)
    assert tensor.flatten().tolist() == list(payload)

    tensor[0, 0] = 99
    assert int(tensor[0, 0]) == 99
    assert shard.read_bytes()[4096 : 4096 + len(payload)] == payload


def test_rejects_non_expert_kind_and_shape_nbytes_mismatch(tmp_path):
    _single_entry_fixture(tmp_path, kind="weight")
    with pytest.raises(ValueError, match="not experts_bank"):
        map_ftw_expert_entry(tmp_path, "gate_up_packed#L00000")

    _single_entry_fixture(tmp_path, nbytes=7)
    with pytest.raises(ValueError, match="shape/dtype imply 8 bytes, index says 7"):
        map_ftw_expert_entry(tmp_path, "gate_up_packed#L00000")


def test_rejects_entry_spanning_physical_shards(tmp_path):
    (tmp_path / "freetoken-00000.ftw").write_bytes(bytes(4096))
    (tmp_path / "freetoken-00001.ftw").write_bytes(bytes(4096))
    _write_index(
        tmp_path,
        tensors=[
            {
                "name": "gate_up_packed#L00000",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [8],
                "global_off": 4092,
                "nbytes": 8,
            }
        ],
        shards=[
            {"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 4096},
            {"file": "freetoken-00001.ftw", "global_off": 4096, "nbytes": 4096},
        ],
    )

    with pytest.raises(ValueError, match="not wholly contained in one shard"):
        map_ftw_expert_entry(tmp_path, "gate_up_packed#L00000")


def test_layer_sources_keep_existing_dict_of_layer_tensors_contract(tmp_path):
    shard = tmp_path / "freetoken-00000.ftw"
    raw = bytearray(12288)
    raw[4096:4100] = b"\x01\x02\x03\x04"
    raw[8192:8196] = b"\x05\x06\x07\x08"
    shard.write_bytes(raw)
    _write_index(
        tmp_path,
        tensors=[
            {
                "name": "gate_up_packed#L00000",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [1, 4],
                "global_off": 4096,
                "nbytes": 4,
            },
            {
                "name": "down_packed#L00000",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [1, 4],
                "global_off": 8192,
                "nbytes": 4,
            },
        ],
        shards=[{"file": shard.name, "global_off": 0, "nbytes": len(raw)}],
    )

    sources = map_ftw_cpu_layer_sources(
        tmp_path,
        0,
        expected_banks={"gate_up_packed", "down_packed"},
    )
    assert set(sources) == {"gate_up_packed", "down_packed"}
    assert sources["gate_up_packed"].tolist() == [[1, 2, 3, 4]]
    assert sources["down_packed"].tolist() == [[5, 6, 7, 8]]

    with pytest.raises(ValueError, match="expected"):
        map_ftw_cpu_layer_sources(tmp_path, 0, expected_banks={"gate_up_packed"})
    with pytest.raises(ValueError, match="no per-layer experts_bank entries"):
        map_ftw_cpu_layer_sources(tmp_path, 1)


def test_mixed_residency_overlay_maps_only_pageable_layers(tmp_path):
    shard, payloads = _two_layer_fixture(tmp_path)
    original = shard.read_bytes()

    overlay = map_ftw_pageable_layer_sources(
        tmp_path,
        num_layers=2,
        expected_banks={"gate_up_packed", "down_packed"},
        layer_residency=["pageable", "pinned"],
    )

    assert set(overlay) == {"gate_up_packed", "down_packed"}
    assert overlay["gate_up_packed"][1] is None
    assert overlay["down_packed"][1] is None
    assert overlay["gate_up_packed"][0].tolist() == [list(payloads[("gate_up_packed", 0)])]
    assert overlay["down_packed"][0].tolist() == [list(payloads[("down_packed", 0)])]

    overlay["gate_up_packed"][0][0, 0] = 99
    assert shard.read_bytes() == original


def test_mixed_residency_overlay_leaves_locked_for_hostbank_path(tmp_path):
    _two_layer_fixture(tmp_path)
    overlay = map_ftw_pageable_layer_sources(
        tmp_path,
        num_layers=2,
        expected_banks={"gate_up_packed", "down_packed"},
        layer_residency=["locked", "pageable"],
    )
    assert overlay["gate_up_packed"][0] is None
    assert overlay["down_packed"][0] is None
    assert overlay["gate_up_packed"][1] is not None
    assert overlay["down_packed"][1] is not None


def test_mixed_residency_overlay_fails_closed_on_contract_mismatch(tmp_path):
    _two_layer_fixture(tmp_path)
    with pytest.raises(ValueError, match="layer_residency has 1 labels"):
        map_ftw_pageable_layer_sources(
            tmp_path,
            num_layers=2,
            expected_banks={"gate_up_packed", "down_packed"},
            layer_residency=["pageable"],
        )
    with pytest.raises(ValueError, match="unknown host residency"):
        map_ftw_pageable_layer_sources(
            tmp_path,
            num_layers=2,
            expected_banks={"gate_up_packed", "down_packed"},
            layer_residency=["pageable", "mystery"],
        )
    with pytest.raises(ValueError, match="expected"):
        map_ftw_pageable_layer_sources(
            tmp_path,
            num_layers=2,
            expected_banks={"gate_up_packed"},
            layer_residency=["pageable", "pinned"],
        )


def test_mixed_residency_overlay_rejects_legacy_flat_bank(tmp_path):
    shard = tmp_path / "freetoken-00000.ftw"
    shard.write_bytes(bytes(8192))
    _write_index(
        tmp_path,
        tensors=[
            {
                "name": "gate_up_packed",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [1, 4],
                "global_off": 4096,
                "nbytes": 4,
            }
        ],
        shards=[{"file": shard.name, "global_off": 0, "nbytes": 8192}],
    )
    with pytest.raises(ValueError, match="legacy flat layout"):
        map_ftw_pageable_layer_sources(
            tmp_path,
            num_layers=1,
            expected_banks={"gate_up_packed"},
            layer_residency=["pageable"],
        )
