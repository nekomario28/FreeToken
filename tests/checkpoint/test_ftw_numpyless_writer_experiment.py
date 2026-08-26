from __future__ import annotations

import json
from pathlib import Path

import torch

from freetoken.checkpoint.ftw import ALIGN, INDEX_NAME
from freetoken.experimental.ftw_numpyless_writer import NumpylessFTWWriter


def _roundtrip(tmp_path: Path, dtype: torch.dtype) -> None:
    if dtype == torch.uint8:
        source = torch.arange(128, dtype=dtype).reshape(16, 8)
    else:
        source = torch.arange(128, dtype=torch.float32).to(dtype).reshape(16, 8)

    writer = NumpylessFTWWriter(str(tmp_path), shard_limit=ALIGN * 8)
    writer.add_tensor("weight", source, kind="weight")
    index = writer.finalize({})

    entry = index["tensors"][0]
    assert entry["name"] == "weight"
    assert entry["dtype"] == str(dtype).removeprefix("torch.")
    assert entry["shape"] == [16, 8]
    assert entry["global_off"] == 0
    assert entry["nbytes"] == source.numel() * source.element_size()

    shard = tmp_path / index["shards"][0]["file"]
    raw = shard.read_bytes()[: entry["nbytes"]]
    restored = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(source.shape)
    assert torch.equal(restored, source)

    saved = json.loads((tmp_path / INDEX_NAME).read_text(encoding="utf-8"))
    assert saved["total_bytes"] % ALIGN == 0


def test_numpyless_writer_roundtrips_raw_cpu_storage(tmp_path: Path):
    for idx, dtype in enumerate(
        (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.int32)
    ):
        child = tmp_path / str(idx)
        child.mkdir()
        _roundtrip(child, dtype)


def test_numpyless_writer_preserves_production_empty_tensor_contract(tmp_path: Path):
    source = torch.empty((0, 4), dtype=torch.float16)
    writer = NumpylessFTWWriter(str(tmp_path), shard_limit=ALIGN * 8)
    writer.add_tensor("empty", source, kind="weight")
    index = writer.finalize({})
    entry = index["tensors"][0]
    assert entry["nbytes"] == 0
    assert entry["shape"] == [0, 4]
    assert index["total_bytes"] == 0
