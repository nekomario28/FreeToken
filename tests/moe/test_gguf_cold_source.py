import enum
import sys
from types import SimpleNamespace

import torch

from freetoken.moe.gguf_cold_source import (
    Qwen4ExpGgufColdExpertSource,
    compile_qwen4exp_gguf_expert_index,
)


class _Q(enum.IntEnum):
    Q4_0 = 2
    IQ2_XXS = 16


class _FakeReader:
    tensors = ()

    def __init__(self, _path):
        self.tensors = type(self).tensors


def _tensor(name, qtype, shape, n_bytes, data_offset):
    return SimpleNamespace(
        name=name,
        tensor_type=qtype,
        shape=shape,
        n_bytes=n_bytes,
        data_offset=data_offset,
    )


def test_qwen4exp_gguf_index_is_metadata_only_and_expert_exact(tmp_path, monkeypatch):
    # Synthetic GGML order:
    # gate/up [fast=8, output_rows=2, experts=3], IQ2 fake block 4 / 2 bytes ->
    # 4 packed bytes per output row, 8 bytes per expert per projection.
    # down [fast=4, output_rows=8, experts=3], Q4 fake block 2 / 1 byte ->
    # 2 packed bytes per output row, 16 bytes per expert.
    gate_off, up_off, down_off = 64, 96, 128
    gate_n, up_n, down_n = 24, 24, 48
    tensors = (
        _tensor("blk.0.ffn_gate_exps.weight", _Q.IQ2_XXS, (8, 2, 3), gate_n, gate_off),
        _tensor("blk.0.ffn_up_exps.weight", _Q.IQ2_XXS, (8, 2, 3), up_n, up_off),
        _tensor("blk.0.ffn_down_exps.weight", _Q.Q4_0, (4, 8, 3), down_n, down_off),
    )
    _FakeReader.tensors = tensors
    fake_gguf = SimpleNamespace(
        GGMLQuantizationType=_Q,
        GGML_QUANT_SIZES={_Q.IQ2_XXS: (4, 2), _Q.Q4_0: (2, 1)},
        GGUFReader=_FakeReader,
    )
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    path = tmp_path / "tiny-qwen4exp.gguf"
    data = bytearray(256)
    # Give every expert/projection a distinct packed byte pattern.
    for expert in range(3):
        data[gate_off + expert * 8 : gate_off + (expert + 1) * 8] = bytes([10 + expert]) * 8
        data[up_off + expert * 8 : up_off + (expert + 1) * 8] = bytes([20 + expert]) * 8
        data[down_off + expert * 16 : down_off + (expert + 1) * 16] = bytes([30 + expert]) * 16
    path.write_bytes(data)

    index = compile_qwen4exp_gguf_expert_index(
        str(path), num_layers=1, num_experts=3
    )

    assert index.gate_layout.shape == (2, 4)
    assert index.up_layout.shape == (2, 4)
    assert index.gate_up_layout.shape == (4, 4)
    assert index.down_layout.shape == (8, 2)
    assert index.ranges[(0, 2, "gate")].offset == gate_off + 16
    assert index.ranges[(0, 2, "up")].offset == up_off + 16
    assert index.ranges[(0, 2, "down")].offset == down_off + 32

    source = Qwen4ExpGgufColdExpertSource(index)
    batch = source.load(0, (2, 0))

    assert batch["gate_up"].shape == (2, 4, 4)
    assert batch["down"].shape == (2, 8, 2)
    assert torch.all(batch["gate_up"][0, :2] == 12)
    assert torch.all(batch["gate_up"][0, 2:] == 22)
    assert torch.all(batch["gate_up"][1, :2] == 10)
    assert torch.all(batch["gate_up"][1, 2:] == 20)
    assert torch.all(batch["down"][0] == 32)
    assert torch.all(batch["down"][1] == 30)


def test_qwen4exp_gguf_index_fails_closed_on_quant_mismatch(tmp_path, monkeypatch):
    _FakeReader.tensors = (
        _tensor("blk.0.ffn_gate_exps.weight", _Q.Q4_0, (8, 2, 3), 12, 64),
        _tensor("blk.0.ffn_up_exps.weight", _Q.IQ2_XXS, (8, 2, 3), 24, 96),
        _tensor("blk.0.ffn_down_exps.weight", _Q.Q4_0, (4, 8, 3), 48, 128),
    )
    fake_gguf = SimpleNamespace(
        GGMLQuantizationType=_Q,
        GGML_QUANT_SIZES={_Q.IQ2_XXS: (4, 2), _Q.Q4_0: (2, 1)},
        GGUFReader=_FakeReader,
    )
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)
    path = tmp_path / "bad.gguf"
    path.write_bytes(bytes(256))

    import pytest

    with pytest.raises(ValueError, match="gate/up must both be IQ2_XXS"):
        compile_qwen4exp_gguf_expert_index(
            str(path), num_layers=1, num_experts=3
        )
