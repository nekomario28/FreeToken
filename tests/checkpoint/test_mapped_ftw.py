from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


# Load only the P0 modules under their real package names without importing
# freetoken.checkpoint.__init__ (which would pull unrelated serving dependencies into
# this deliberately minimal CPU smoke).
_ROOT = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "checkpoint"
if "freetoken" not in sys.modules:
    pkg = types.ModuleType("freetoken")
    pkg.__path__ = []
    sys.modules["freetoken"] = pkg
if "freetoken.checkpoint" not in sys.modules:
    pkg = types.ModuleType("freetoken.checkpoint")
    pkg.__path__ = [str(_ROOT)]
    sys.modules["freetoken.checkpoint"] = pkg


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("freetoken.checkpoint.mapped_ftw_core", _ROOT / "mapped_ftw_core.py")
_MAPPED = _load("freetoken.checkpoint.mapped_ftw", _ROOT / "mapped_ftw.py")
map_ftw_entry = _MAPPED.map_ftw_entry


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _write_one(tmp_path: Path, tensor: torch.Tensor, *, shard_limit: int = 8 << 30):
    payload = _tensor_bytes(tensor)
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    entry = {
        "name": "gate_up_packed#L00000",
        "kind": "experts_bank",
        "dtype": dtype_name,
        "shape": list(tensor.shape),
        "global_off": 0,
        "nbytes": len(payload),
    }

    shards = []
    if len(payload) <= shard_limit:
        shard = tmp_path / "freetoken-00000.ftw"
        shard.write_bytes(payload)
        shards.append({"file": shard.name, "global_off": 0, "nbytes": len(payload)})
    else:
        off = 0
        for idx in range(0, len(payload), shard_limit):
            chunk = payload[idx : idx + shard_limit]
            shard = tmp_path / f"freetoken-{idx // shard_limit:05d}.ftw"
            shard.write_bytes(chunk)
            shards.append({"file": shard.name, "global_off": off, "nbytes": len(chunk)})
            off += len(chunk)

    (tmp_path / "freetoken_weight.json").write_text(
        json.dumps(
            {
                "format": "freetoken_weight",
                "version": 1,
                "align": 4096,
                "total_bytes": len(payload),
                "tensors": [entry],
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    return entry


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
    # One entry larger than the shard limit is physically split. P0 refuses to invent
    # virtual contiguity across files.
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
