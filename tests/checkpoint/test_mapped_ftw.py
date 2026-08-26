from __future__ import annotations

from pathlib import Path

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.checkpoint.mapped_ftw import map_ftw_entry


def _write_one(tmp_path: Path, tensor: torch.Tensor, *, shard_limit: int = 8 << 30):
    writer = FTWWriter(str(tmp_path), shard_limit=shard_limit)
    writer.add_tensor("gate_up_packed#L00000", tensor, kind="experts_bank")
    return writer.finalize({"quant_format": "nvfp4"})


def test_private_mapping_is_tensor_view_without_checkpoint_mutation(tmp_path: Path):
    original = torch.arange(64, dtype=torch.uint8).reshape(8, 8)
    _write_one(tmp_path, original)

    mapped = map_ftw_entry(tmp_path, "gate_up_packed#L00000")
    assert mapped.tensor.device.type == "cpu"
    assert torch.equal(mapped.tensor, original)

    # ACCESS_COPY is a private mapping: writes are visible through the tensor/mapping
    # but must never mutate the FTW source file.
    first = int(original.flatten()[0])
    mapped.tensor.flatten()[0] = 255
    assert int(mapped.tensor.flatten()[0]) == 255
    assert mapped.mapping[mapped.tensor_offset] == 255
    with mapped.shard_path.open("rb") as handle:
        handle.seek(mapped.file_offset)
        assert handle.read(1) == bytes([first])


def test_owner_keeps_mapping_alive_for_tensor_use(tmp_path: Path):
    original = torch.arange(32, dtype=torch.int32)
    _write_one(tmp_path, original)
    owner = map_ftw_entry(tmp_path, "gate_up_packed#L00000")

    # The explicit owner carries the mmap object alongside the tensor. P0 deliberately
    # provides no eager close API because closing storage under torch.frombuffer is unsafe.
    assert owner.mapping.closed is False
    assert int(owner.tensor[-1]) == 31
    assert owner.nbytes == original.numel() * original.element_size()


def test_entry_that_spans_ftw_shards_fails_closed(tmp_path: Path):
    # One entry larger than the shard limit is physically split by FTWWriter. P0 refuses
    # to invent virtual contiguity across files.
    original = torch.arange(8192, dtype=torch.uint8)
    _write_one(tmp_path, original, shard_limit=4096)

    with pytest.raises(ValueError, match="exactly one shard"):
        map_ftw_entry(tmp_path, "gate_up_packed#L00000")


def test_shape_dtype_nbytes_contract_is_checked(tmp_path: Path):
    original = torch.arange(16, dtype=torch.float32)
    _write_one(tmp_path, original)
    owner = map_ftw_entry(tmp_path, "gate_up_packed#L00000")
    assert owner.tensor.dtype == torch.float32
    assert tuple(owner.tensor.shape) == (16,)
    assert torch.equal(owner.tensor, original)


def test_missing_name_fails_closed(tmp_path: Path):
    _write_one(tmp_path, torch.ones(4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="exactly one FTW tensor"):
        map_ftw_entry(tmp_path, "missing#L00000")
