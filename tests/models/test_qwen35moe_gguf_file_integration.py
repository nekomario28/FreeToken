"""CPU-only integration seam from real GGUF bytes into the Qwen3.5-MoE loader.

This deliberately does not stand in for a production checkpoint or inference test.  It
writes a tiny, structurally coherent GGUF v3 file directly, then exercises the same
config detection and weight iterator used by a bare ``.gguf`` model path.  The purpose is
to make the file-format -> config-shim -> qwen35moe weight-loading seam executable without
network access, a model download, GPU kernels, or a server.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest
import torch

from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights, parse_gguf_config
from freetoken.utils.hf import cached_load_hf_config


# GGUF metadata value tags (gguf.GGUFValueType).
_UINT32 = 4
_FLOAT32 = 6
_STRING = 8
_F32_TENSOR = 0
_ALIGN = 32


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return _u64(len(raw)) + raw


def _kv(key: str, tag: int, value) -> bytes:
    out = _string(key) + _u32(tag)
    if tag == _STRING:
        return out + _string(value)
    if tag == _UINT32:
        return out + _u32(int(value))
    if tag == _FLOAT32:
        return out + struct.pack("<f", float(value))
    raise AssertionError(f"unsupported fixture metadata tag {tag}")


def _write_f32_gguf(
    path: Path,
    kvs: list[bytes],
    tensors: list[tuple[str, tuple[int, ...], list[float]]],
) -> None:
    """Write a minimal GGUF v3 containing F32 tensors.

    ``ggml_shape`` is stored in GGML order exactly as the file format expects.  Tensor
    payload offsets are relative to the aligned start of the tensor-data section.
    """
    header = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(len(kvs))
    header += b"".join(kvs)

    infos = bytearray()
    data = bytearray()
    for name, ggml_shape, values in tensors:
        expected = math.prod(ggml_shape)
        assert len(values) == expected, (name, len(values), expected)

        offset = len(data)
        infos += _string(name)
        infos += _u32(len(ggml_shape))
        for dim in ggml_shape:
            infos += _u64(dim)
        infos += _u32(_F32_TENSOR)
        infos += _u64(offset)

        payload = struct.pack(f"<{expected}f", *values)
        data += payload
        data += b"\0" * ((-len(data)) % _ALIGN)

    body = bytearray(header) + infos
    body += b"\0" * ((-len(body)) % _ALIGN)
    body += data
    path.write_bytes(body)


def _tiny_qwen35moe(path: Path) -> None:
    # One tiny linear-attention decoder layer.  These values satisfy the structural
    # invariants in parse_gguf_config while keeping every tensor small enough for a CPU
    # test.  The fixture intentionally omits tokenizer metadata and routed-expert banks:
    # vocab size comes from token_embd.weight, and this test asks only for non-MoE weights.
    arch = "qwen35moe"
    kvs = [
        _kv("general.architecture", _STRING, arch),
        _kv("general.alignment", _UINT32, _ALIGN),
        _kv(f"{arch}.block_count", _UINT32, 1),
        _kv(f"{arch}.nextn_predict_layers", _UINT32, 0),
        _kv(f"{arch}.embedding_length", _UINT32, 4),
        _kv(f"{arch}.attention.head_count", _UINT32, 1),
        _kv(f"{arch}.attention.head_count_kv", _UINT32, 1),
        _kv(f"{arch}.attention.key_length", _UINT32, 4),
        _kv(f"{arch}.attention.layer_norm_rms_epsilon", _FLOAT32, 1.0e-5),
        _kv(f"{arch}.rope.freq_base", _FLOAT32, 10000.0),
        _kv(f"{arch}.rope.dimension_count", _UINT32, 4),
        _kv(f"{arch}.context_length", _UINT32, 32),
        _kv(f"{arch}.expert_count", _UINT32, 1),
        _kv(f"{arch}.expert_used_count", _UINT32, 1),
        _kv(f"{arch}.expert_feed_forward_length", _UINT32, 4),
        _kv(f"{arch}.expert_shared_feed_forward_length", _UINT32, 4),
        _kv(f"{arch}.feed_forward_length", _UINT32, 4),
        _kv(f"{arch}.ssm.conv_kernel", _UINT32, 2),
        _kv(f"{arch}.ssm.state_size", _UINT32, 2),
        _kv(f"{arch}.ssm.group_count", _UINT32, 1),
        _kv(f"{arch}.ssm.time_step_rank", _UINT32, 1),
        _kv(f"{arch}.ssm.inner_size", _UINT32, 2),
        _kv(f"{arch}.full_attention_interval", _UINT32, 2),
    ]

    tensors = [
        # GGML [hidden, vocab] -> reader's torch shape [vocab, hidden].
        ("token_embd.weight", (4, 8), [float(i) / 16 for i in range(32)]),
        ("output_norm.weight", (4,), [0.25, 0.5, 0.75, 1.0]),
        ("blk.0.attn_norm.weight", (4,), [1.0, 1.25, 1.5, 1.75]),
    ]
    _write_f32_gguf(path, kvs, tensors)


def test_real_gguf_bytes_reach_qwen35moe_config_and_weight_iterator(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny-qwen35moe.gguf"
    _tiny_qwen35moe(model_path)

    # Exercise the normal bare-GGUF dispatch, not a hand-constructed config object.
    shim = cached_load_hf_config(str(model_path))
    assert shim.model_type == "qwen35moe"
    assert shim.architectures == ["Qwen35MoeGGUFForCausalLM"]
    assert shim.vocab_size == 8
    assert shim.tie_word_embeddings is True

    config = parse_gguf_config(shim)
    assert config.num_layers == 1
    assert config.hidden_size == 4
    assert config.vocab_size == 8
    assert config.num_experts == 1

    loaded = dict(
        iter_gguf_weights(
            str(model_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )

    assert set(loaded) == {
        "model.embed_tokens.qweight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
    }

    # Embedding stays in the loader's packed-byte representation; F32 norms are
    # dequantized through the production CPU reference path into bf16.
    assert loaded["model.embed_tokens.qweight"].dtype == torch.uint8
    assert loaded["model.embed_tokens.qweight"].shape == (8, 16)

    assert loaded["model.norm.weight"].dtype == torch.bfloat16
    assert loaded["model.norm.weight"].shape == (4,)
    assert loaded["model.norm.weight"].float().tolist() == pytest.approx(
        [0.25, 0.5, 0.75, 1.0], abs=0.01
    )

    assert loaded["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    assert loaded["model.layers.0.input_layernorm.weight"].shape == (4,)
    assert loaded["model.layers.0.input_layernorm.weight"].float().tolist() == pytest.approx(
        [1.0, 1.25, 1.5, 1.75], abs=0.01
    )
