from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter, INDEX_NAME, layer_bank_entry_name
from freetoken.experimental.ftw_filebacked_cpu_moe import (
    load_ftw_banks_filebacked_cpu,
)
from freetoken.experimental.ftw_filebacked_cpu_moe_server import _requires_all_cpu
from freetoken.moe.host_banks import HostResidency


BANK_SPECS = {
    "gate_up_packed": ((2, 32, 8), torch.uint8),
    "gate_up_scale": ((2, 32, 1), torch.uint8),
    "gate_up_global": ((2, 32), torch.float16),
    "down_packed": ((2, 16, 8), torch.uint8),
    "down_scale": ((2, 16, 1), torch.uint8),
    "down_global": ((2, 16), torch.float16),
}
BANKS = tuple(BANK_SPECS)


def _tensor(shape: tuple[int, ...], dtype: torch.dtype, value: int) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype)


def _write_native_ftw(
    path: Path,
    *,
    flat: bool = False,
    bad_global_dtype: bool = False,
) -> dict:
    writer = FTWWriter(str(path), shard_limit=4096 * 8)
    expected = {}
    for bank_id, (name, (shape, dtype)) in enumerate(BANK_SPECS.items()):
        if bad_global_dtype and name == "gate_up_global":
            dtype = torch.float32
        if flat:
            layer0 = _tensor(shape, dtype, bank_id + 1)
            layer1 = _tensor(shape, dtype, bank_id + 9)
            tensor = torch.cat((layer0, layer1), dim=0)
            writer.add_tensor(name, tensor, kind="experts_bank")
        else:
            for layer in range(2):
                tensor = _tensor(shape, dtype, bank_id + layer + 1)
                writer.add_tensor(
                    layer_bank_entry_name(name, layer),
                    tensor,
                    kind="experts_bank",
                )
                expected[(name, layer)] = tensor.clone()
    index = writer.finalize(
        {
            "quant_format": "nvfp4",
            "expert_bank_num_layers": 2,
        }
    )
    return {"index": index, "expected": expected}


def _entry_file_offset(path: Path, entry_name: str) -> tuple[Path, int]:
    index = json.loads((path / INDEX_NAME).read_text(encoding="utf-8"))
    entry = next(t for t in index["tensors"] if t["name"] == entry_name)
    off = entry["global_off"]
    nbytes = entry["nbytes"]
    for shard in index["shards"]:
        s0 = shard["global_off"]
        s1 = s0 + shard["nbytes"]
        if off >= s0 and off + nbytes <= s1:
            return path / shard["file"], off - s0
    raise AssertionError("entry not contained in one shard")


def test_filebacked_native_nvfp4_maps_cpu_executor_geometry(tmp_path: Path):
    fixture = _write_native_ftw(tmp_path)
    banks = load_ftw_banks_filebacked_cpu(
        str(tmp_path),
        num_layers=2,
        layer_residency=[HostResidency.LOCKED.value] * 2,
    )

    assert banks.quant_format == "nvfp4"
    assert banks.layer_residency == [HostResidency.PAGEABLE.value] * 2
    assert set(banks.sources) == set(BANKS)
    assert banks.file_backed_layers == 2
    assert banks.file_backed_bytes > 0
    assert banks.mapped_shards > 0

    for name, (shape, dtype) in BANK_SPECS.items():
        for layer, tensor in enumerate(banks.sources[name]):
            assert tensor.is_contiguous()
            assert tensor.shape == shape
            assert tensor.dtype == dtype
            assert torch.equal(tensor, fixture["expected"][(name, layer)])


def test_filebacked_mapping_is_private_cow_and_outlives_result_wrapper(tmp_path: Path):
    _write_native_ftw(tmp_path)
    name = layer_bank_entry_name("gate_up_packed", 0)
    shard, offset = _entry_file_offset(tmp_path, name)
    before = shard.read_bytes()[offset]

    banks = load_ftw_banks_filebacked_cpu(
        str(tmp_path),
        num_layers=2,
        layer_residency=[HostResidency.PAGEABLE.value] * 2,
    )
    tensor = banks.sources["gate_up_packed"][0]
    original = int(tensor[0, 0, 0])
    tensor[0, 0, 0] = (original + 1) % 255
    assert int(tensor[0, 0, 0]) != original

    # ACCESS_COPY/MAP_PRIVATE must never mutate the checkpoint.
    after = shard.read_bytes()[offset]
    assert after == before

    # Engine construction drops the ExpertBanks-like wrapper after wiring sources.
    # The tensor remains valid because the experimental module holds mappings for
    # process lifetime.
    del banks
    gc.collect()
    assert int(tensor[0, 0, 0]) != original


def test_filebacked_loader_rejects_any_pinned_layer(tmp_path: Path):
    _write_native_ftw(tmp_path)
    with pytest.raises(ValueError, match="every layer"):
        load_ftw_banks_filebacked_cpu(
            str(tmp_path),
            num_layers=2,
            layer_residency=[
                HostResidency.PAGEABLE.value,
                HostResidency.PINNED.value,
            ],
        )


def test_filebacked_loader_rejects_legacy_flat_entries(tmp_path: Path):
    _write_native_ftw(tmp_path, flat=True)
    with pytest.raises(ValueError, match="per-layer"):
        load_ftw_banks_filebacked_cpu(
            str(tmp_path),
            num_layers=2,
            layer_residency=[HostResidency.PAGEABLE.value] * 2,
        )


def test_filebacked_loader_validates_cpu_executor_nvfp4_dtypes(tmp_path: Path):
    _write_native_ftw(tmp_path, bad_global_dtype=True)
    with pytest.raises(ValueError, match="global banks must be float16"):
        load_ftw_banks_filebacked_cpu(
            str(tmp_path),
            num_layers=2,
            layer_residency=[HostResidency.PAGEABLE.value] * 2,
        )


def test_launcher_first_prototype_requires_all_cpu_fraction():
    assert _requires_all_cpu("1") is True
    assert _requires_all_cpu("1.0") is True
    assert _requires_all_cpu("0.999") is False
    assert _requires_all_cpu("0.5") is False
    assert _requires_all_cpu(None) is False
