import enum
import sys
from types import SimpleNamespace

import torch

from freetoken.models.qwen4_exp.ple_gguf import (
    GGML_IQ4_NL,
    GgufIq4NlPleTable,
    compile_iq4_nl_ple_index,
)


class _Q(enum.IntEnum):
    IQ4_NL = GGML_IQ4_NL


class _FakeReader:
    tensors = ()

    def __init__(self, _path):
        self.tensors = type(self).tensors


def test_iq4_nl_ple_reads_only_selected_rows_and_restores_order(tmp_path, monkeypatch):
    # 160 values / IQ4_NL row -> 5 blocks * 18 bytes = 90 bytes.
    offset = 64
    rows = 6
    row_bytes = 90
    _FakeReader.tensors = (
        SimpleNamespace(
            name="per_layer_token_embd.weight",
            tensor_type=_Q.IQ4_NL,
            shape=(160, rows),
            n_bytes=rows * row_bytes,
            data_offset=offset,
        ),
    )
    fake_gguf = SimpleNamespace(GGUFReader=_FakeReader)
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    path = tmp_path / "ple.gguf"
    data = bytearray(offset + rows * row_bytes)
    for row in range(rows):
        data[offset + row * row_bytes : offset + (row + 1) * row_bytes] = bytes([row]) * row_bytes
    path.write_bytes(data)

    index = compile_iq4_nl_ple_index(str(path))
    assert index.row_bytes == 90
    table = GgufIq4NlPleTable(index, device=torch.device("cpu"))

    seen = []

    def fake_dequant(packed, quant_type, m, n, dtype):
        assert quant_type == GGML_IQ4_NL
        assert (m, n, dtype) == (2, 160, torch.bfloat16)
        seen.extend(int(v) for v in packed[:, 0])
        # Return each row filled with its packed marker byte.
        return packed[:, :1].to(torch.bfloat16).expand(m, n).clone()

    monkeypatch.setattr("freetoken.kernel.gguf.ggml_dequantize", fake_dequant)
    ids = torch.tensor([[4, 1, 4]], dtype=torch.int64)
    out = table.lookup(ids)

    assert sorted(seen) == [1, 4]  # duplicate row 4 was read/dequantized once
    assert out.shape == (1, 3 * 160)
    assert torch.all(out[0, :160] == 4)
    assert torch.all(out[0, 160:320] == 1)
    assert torch.all(out[0, 320:] == 4)
