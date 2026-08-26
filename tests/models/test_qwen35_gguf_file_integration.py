"""Real-byte integration coverage for the dense qwen35 GGUF adapter.

The lower-level reader tests already exercise single/sharded GGUF files, while
``test_qwen35moe_gguf.py`` covers the adapter's mapping and geometry rules.  This test
bridges those layers with one tiny independently-written GGUF v3 file so a regression in
metadata parsing, tensor layout, config-shim construction, or the global weight stream
cannot hide behind mocks.

This is intentionally a synthetic fixture, not a usable model checkpoint.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import torch

import freetoken.models.qwen3_5_moe.gguf as qgguf
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata


_U32 = 4
_F32 = 6
_STRING = 8
_GGML_F32 = 0
_ALIGN = 32


def _u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _u64(value: int) -> bytes:
    return struct.pack("<Q", int(value))


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _kv(key: str, value_type: int, value) -> bytes:
    out = _string(key) + _u32(value_type)
    if value_type == _STRING:
        return out + _string(str(value))
    if value_type == _U32:
        return out + _u32(value)
    if value_type == _F32:
        return out + struct.pack("<f", float(value))
    raise AssertionError(value_type)


def _write_tiny_qwen35(path) -> None:
    metadata = [
        _kv("general.architecture", _STRING, "qwen35"),
        _kv("general.alignment", _U32, _ALIGN),
        _kv("qwen35.block_count", _U32, 2),
        _kv("qwen35.embedding_length", _U32, 4),
        _kv("qwen35.attention.head_count", _U32, 1),
        _kv("qwen35.attention.head_count_kv", _U32, 1),
        _kv("qwen35.attention.key_length", _U32, 2),
        _kv("qwen35.attention.layer_norm_rms_epsilon", _F32, 1e-6),
        _kv("qwen35.rope.freq_base", _F32, 10000.0),
        _kv("qwen35.rope.dimension_count", _U32, 2),
        _kv("qwen35.context_length", _U32, 128),
        _kv("qwen35.feed_forward_length", _U32, 8),
        _kv("qwen35.ssm.conv_kernel", _U32, 2),
        _kv("qwen35.ssm.state_size", _U32, 2),
        _kv("qwen35.ssm.group_count", _U32, 1),
        _kv("qwen35.ssm.time_step_rank", _U32, 1),
        _kv("qwen35.ssm.inner_size", _U32, 2),
        _kv("qwen35.full_attention_interval", _U32, 2),
    ]
    specs = [
        ("token_embd.weight", [4, 6]),
        ("output_norm.weight", [4]),
        ("output.weight", [4, 6]),
    ]

    tensors = []
    for seed, (name, dims) in enumerate(specs, 1):
        count = math.prod(dims)
        data = np.arange(seed, seed + count, dtype=np.float32).astype(
            "<f4", copy=False
        ).tobytes()
        tensors.append((name, dims, data))

    header = (
        b"GGUF"
        + _u32(3)
        + _u64(len(tensors))
        + _u64(len(metadata))
        + b"".join(metadata)
    )
    tensor_info = bytearray()
    offsets: list[int] = []
    offset = 0
    for name, dims, data in tensors:
        tensor_info += _string(name)
        tensor_info += _u32(len(dims))
        tensor_info += b"".join(_u64(dim) for dim in dims)
        tensor_info += _u32(_GGML_F32) + _u64(offset)
        offsets.append(offset)
        offset += len(data)
        offset = (offset + _ALIGN - 1) // _ALIGN * _ALIGN

    file_bytes = bytearray(header) + tensor_info
    file_bytes += b"\0" * ((-len(file_bytes)) % _ALIGN)
    data_area = bytearray(offset)
    for (_, _, data), tensor_offset in zip(tensors, offsets):
        data_area[tensor_offset : tensor_offset + len(data)] = data
    path.write_bytes(file_bytes + data_area)


def test_qwen35_real_file_reaches_config_and_global_weight_stream(
    tmp_path, monkeypatch
):
    path = tmp_path / "tiny-qwen35.gguf"
    _write_tiny_qwen35(path)

    metadata = load_gguf_metadata(str(path))
    assert metadata["general.architecture"] == "qwen35"

    raw_tensors = list(iter_gguf_tensors(str(path)))
    assert [tensor.name for tensor in raw_tensors] == [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    ]
    assert raw_tensors[0].shape == (6, 4)

    shim = build_gguf_shim(str(path))
    assert shim.model_type == "qwen35"
    assert shim.architectures == ["Qwen35GGUFForCausalLM"]
    assert shim.vocab_size == 6
    assert shim.tie_word_embeddings is False

    config = qgguf.parse_gguf_config(shim)
    assert config.num_layers == 2
    assert config.hidden_size == 4
    assert config.vocab_size == 6
    assert config.moe_enabled is False
    assert config.gguf_model_path == str(path)

    # The file already produced the exact shim above.  Reuse it so this test owns the
    # GGUF reader/config/adapter seam rather than Hugging Face hub resolution or TP setup.
    import freetoken.utils

    monkeypatch.setattr(freetoken.utils, "cached_load_hf_config", lambda _: shim)
    monkeypatch.setattr(qgguf, "_require_tp1", lambda _what: None)

    weights = dict(
        qgguf.iter_gguf_weights(
            str(path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )
    assert set(weights) == {
        "model.embed_tokens.qweight",
        "model.norm.weight",
        "lm_head.qweight",
    }
    assert weights["model.embed_tokens.qweight"].shape == (6, 16)
    assert weights["model.norm.weight"].shape == (4,)
    assert weights["lm_head.qweight"].shape == (6, 16)
