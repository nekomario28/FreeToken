from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.checkpoint.low_memory_nvfp4 import stream_nvfp4_layers_serial


_SPEC = SimpleNamespace(
    key_pattern=re.compile(
        r"^layer\.(?P<layer>\d+)\.expert\.(?P<expert>\d+)\."
        r"(?P<proj>gate|up|down)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate": "gate", "up": "up", "down": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="synthetic NVFP4",
)


class _FakeBank:
    def __init__(self, tensor: torch.Tensor, release_cb):
        self.tensor = tensor
        self.nbytes = tensor.numel() * tensor.element_size()
        self._release_cb = release_cb
        self.released = False

    def release(self):
        assert not self.released
        self.released = True
        self._release_cb()


def _checkpoint(tmp_path, *, layers=3, experts=2, hidden=16, inter=16, omit=None):
    tensors = {}
    for layer in range(layers):
        for expert in range(experts):
            base_value = 10 * layer + expert + 1
            for proj in ("gate", "up", "down"):
                wshape = (inter, hidden // 2) if proj in ("gate", "up") else (hidden, inter // 2)
                sshape = (inter, hidden // 16) if proj in ("gate", "up") else (hidden, inter // 16)
                vals = {
                    "weight": torch.full(wshape, base_value, dtype=torch.uint8),
                    "weight_scale": torch.full(sshape, float(base_value), dtype=torch.float32),
                    "weight_scale_2": torch.tensor([float(base_value)], dtype=torch.float32),
                }
                for kind, tensor in vals.items():
                    key = f"layer.{layer}.expert.{expert}.{proj}.{kind}"
                    if key != omit:
                        tensors[key] = tensor

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    config = SimpleNamespace(
        num_moe_layers=layers,
        num_layers=layers,
        first_k_dense_replace=0,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )
    return config


def _tracking_allocator(state):
    def alloc(E, H, I):
        # If the prior sink did not release its layer, a second allocation is a test failure.
        assert state["live_layers"] == 0
        state["live_layers"] += 1
        state["max_live_layers"] = max(state["max_live_layers"], state["live_layers"])
        remaining = 6

        def one_released():
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                state["live_layers"] -= 1

        def bank(shape, dtype):
            return _FakeBank(torch.empty(shape, dtype=dtype), one_released)

        return {
            "gate_up_packed": bank((E, 2 * I, H // 2), torch.uint8),
            "gate_up_scale": bank((E, 2 * I, H // 16), torch.float32),
            "gate_up_global": bank((E, 2 * I), torch.float16),
            "down_packed": bank((E, H, I // 2), torch.uint8),
            "down_scale": bank((E, H, I // 16), torch.float32),
            "down_global": bank((E, H), torch.float16),
        }

    return alloc


def test_streams_exactly_one_expert_layer_at_a_time(tmp_path):
    config = _checkpoint(tmp_path)
    state = {"live_layers": 0, "max_live_layers": 0}
    seen = []

    def sink(layer_id, banks):
        assert state["live_layers"] == 1
        expected = 10 * layer_id + 1
        assert int(banks["gate_up_packed"].tensor[0, 0, 0]) == expected
        assert int(banks["down_packed"].tensor[0, 0, 0]) == expected
        assert float(banks["gate_up_global"].tensor[0, 0]) == float(expected)
        seen.append(layer_id)
        for bank in banks.values():
            bank.release()
        assert state["live_layers"] == 0

    result = stream_nvfp4_layers_serial(
        str(tmp_path),
        config,
        _SPEC,
        drop_page_cache=lambda _path: None,
        layer_sink=sink,
        alloc_layer=_tracking_allocator(state),
    )

    assert seen == [0, 1, 2]
    assert state == {"live_layers": 0, "max_live_layers": 1}
    assert result["layers_streamed"] == 3
    assert result["tensors_read"] == 3 * 2 * 3 * 3
    assert result["expert_bank_bytes_streamed"] > 0


def test_missing_global_scale_fails_closed_before_sink(tmp_path):
    missing = "layer.0.expert.0.gate.weight_scale_2"
    config = _checkpoint(tmp_path, layers=1, omit=missing)
    state = {"live_layers": 0, "max_live_layers": 0}
    calls = []

    with pytest.raises(ValueError, match="missing weight_scale_2|incomplete bank layer"):
        stream_nvfp4_layers_serial(
            str(tmp_path),
            config,
            _SPEC,
            drop_page_cache=lambda _path: None,
            layer_sink=lambda layer_id, banks: calls.append(layer_id),
            alloc_layer=_tracking_allocator(state),
        )

    assert calls == []


def test_missing_entire_layer_fails_before_any_allocation(tmp_path):
    config = _checkpoint(tmp_path, layers=2)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"] = {
        name: shard for name, shard in index["weight_map"].items() if not name.startswith("layer.1.")
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    state = {"live_layers": 0, "max_live_layers": 0}
    with pytest.raises(ValueError, match="expected"):
        stream_nvfp4_layers_serial(
            str(tmp_path),
            config,
            _SPEC,
            drop_page_cache=lambda _path: None,
            layer_sink=lambda *_args: None,
            alloc_layer=_tracking_allocator(state),
        )
    assert state["max_live_layers"] == 0
