from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/freetoken/experimental/safetensors_ftw_passthrough.py"

# ftw.py only needs init_logger at import time, but importing freetoken.utils eagerly pulls
# the HF/Transformers stack. Keep this focused smoke on the storage primitive: provide the
# exact tiny dependency during module load, then restore sys.modules immediately.
_previous_utils = sys.modules.get("freetoken.utils")
_logger_stub = types.ModuleType("freetoken.utils")
_logger_stub.init_logger = lambda name: logging.getLogger(name)
sys.modules["freetoken.utils"] = _logger_stub
try:
    SPEC = importlib.util.spec_from_file_location(
        "safetensors_ftw_passthrough_under_test", MODULE_PATH
    )
    assert SPEC is not None and SPEC.loader is not None
    M = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(M)
finally:
    if _previous_utils is None:
        sys.modules.pop("freetoken.utils", None)
    else:
        sys.modules["freetoken.utils"] = _previous_utils

Writer = M.BoundedPassthroughFTWWriter


def _logical_payload(out_dir: Path, index: dict, entry: dict) -> bytes:
    start = int(entry["global_off"])
    end = start + int(entry["nbytes"])
    chunks = []
    for shard in sorted(index["shards"], key=lambda row: row["global_off"]):
        s0 = int(shard["global_off"])
        s1 = s0 + int(shard["nbytes"])
        lo, hi = max(start, s0), min(end, s1)
        if lo >= hi:
            continue
        data = (out_dir / shard["file"]).read_bytes()
        chunks.append(data[lo - s0 : hi - s0])
    return b"".join(chunks)


def test_passthrough_streams_exact_bytes_with_bounded_buffer_across_ftw_shards(tmp_path: Path):
    source_path = tmp_path / "source.safetensors"
    raw_name = "model.language_model.embed_tokens.weight"
    source = torch.arange(24_576, dtype=torch.float32).reshape(6_144, 4).to(torch.bfloat16)
    save_file({raw_name: source}, str(source_path))

    out = tmp_path / "ftw"
    writer = Writer(str(out), shard_limit=16 * 1024)
    receipt = writer.add_safetensors_passthrough(
        name="model.embed_tokens.weight",
        safetensors_path=source_path,
        safetensors_name=raw_name,
        chunk_bytes=4096,
    )
    index = writer.finalize({"synthetic": True})

    entry = index["tensors"][0]
    assert entry["name"] == "model.embed_tokens.weight"
    assert entry["dtype"] == "bfloat16"
    assert entry["shape"] == [6_144, 4]
    assert receipt["max_read_buffer_bytes"] <= 4096
    assert len(index["shards"]) > 1

    source_off, nbytes, _dtype, _shape = M._tensor_entry(source_path, raw_name)
    with source_path.open("rb") as f:
        f.seek(source_off)
        source_payload = f.read(nbytes)
    assert _logical_payload(out, index, entry) == source_payload


def test_passthrough_preserves_alignment_and_early_roll_semantics(tmp_path: Path):
    source_path = tmp_path / "source.safetensors"
    save_file(
        {
            "first": torch.arange(1024, dtype=torch.float32),
            "second": torch.arange(1536, dtype=torch.float32),
        },
        str(source_path),
    )
    out = tmp_path / "ftw"
    writer = Writer(str(out), shard_limit=8192)
    writer.add_safetensors_passthrough(
        name="first", safetensors_path=source_path, safetensors_name="first", chunk_bytes=1024
    )
    writer.add_safetensors_passthrough(
        name="second", safetensors_path=source_path, safetensors_name="second", chunk_bytes=1024
    )
    index = writer.finalize({})

    assert all(int(t["global_off"]) % M.ALIGN == 0 for t in index["tensors"])
    assert index["tensors"][0]["global_off"] == 0
    assert index["tensors"][1]["global_off"] == 4096
    assert len(index["shards"]) == 2
    assert index["shards"][0]["nbytes"] == 4096


def test_passthrough_fails_closed_on_missing_or_invalid_entry(tmp_path: Path):
    source_path = tmp_path / "source.safetensors"
    save_file({"present": torch.arange(16, dtype=torch.float32)}, str(source_path))
    writer = Writer(str(tmp_path / "ftw"))

    with pytest.raises(KeyError, match="tensor not found"):
        writer.add_safetensors_passthrough(
            name="missing", safetensors_path=source_path, safetensors_name="missing"
        )
    with pytest.raises(ValueError, match="chunk_bytes"):
        writer.add_safetensors_passthrough(
            name="present", safetensors_path=source_path, safetensors_name="present", chunk_bytes=0
        )


def test_passthrough_detects_short_payload_read(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "source.safetensors"
    save_file({"present": torch.arange(4096, dtype=torch.float32)}, str(source_path))
    writer = Writer(str(tmp_path / "ftw"))
    real_pread = M.os.pread
    calls = 0

    def short_once(fd, n, offset):
        nonlocal calls
        calls += 1
        data = real_pread(fd, n, offset)
        return data[:-1] if calls == 1 and data else data

    monkeypatch.setattr(M.os, "pread", short_once)
    with pytest.raises(OSError, match="short safetensors payload read"):
        writer.add_safetensors_passthrough(
            name="present",
            safetensors_path=source_path,
            safetensors_name="present",
            chunk_bytes=1024,
        )
