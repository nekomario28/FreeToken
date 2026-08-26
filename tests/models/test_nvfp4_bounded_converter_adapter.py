from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "models" / "nvfp4_banks.py"
)


class FakeBank:
    def __init__(self, value: int) -> None:
        self.tensor = torch.tensor([value], dtype=torch.uint8)
        self.released = False

    def release(self) -> None:
        self.released = True


def _load_module(monkeypatch, fake_stream):
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken", freetoken)

    utils = types.ModuleType("freetoken.utils")
    utils.download_hf_weight = lambda path: path
    monkeypatch.setitem(sys.modules, "freetoken.utils", utils)

    checkpoint = types.ModuleType("freetoken.checkpoint")
    checkpoint.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken.checkpoint", checkpoint)

    low_memory = types.ModuleType("freetoken.checkpoint.low_memory_nvfp4")
    low_memory.stream_nvfp4_layers_serial = fake_stream
    monkeypatch.setitem(sys.modules, "freetoken.checkpoint.low_memory_nvfp4", low_memory)

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    module_name = "nvfp4_bounded_converter_adapter_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _config():
    return types.SimpleNamespace(
        num_moe_layers=2,
        num_layers=2,
        first_k_dense_replace=0,
        num_experts=1,
        hidden_size=16,
        moe_intermediate_size=16,
    )


def _source_spec(module):
    return module.Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(r".*"),
        proj_to_role={},
        layer_to_bank=lambda layer, _config: layer,
        desc="synthetic NVFP4",
    )


def test_converter_serial_adapter_preserves_source_abi_and_released_tensor_refs(monkeypatch):
    def fake_stream(_model_path, config, _spec, *, drop_page_cache, layer_sink):
        for layer_id in range(config.num_moe_layers):
            banks = {
                "gate_up_packed": FakeBank(layer_id + 1),
                "down_packed": FakeBank(layer_id + 11),
            }
            layer_sink(layer_id, banks)
            assert all(bank.released for bank in banks.values())
        return {"layers_streamed": config.num_moe_layers, "tensors_read": 18}

    module = _load_module(monkeypatch, fake_stream)
    seen = []

    def sink(layer_id, banks):
        seen.append((layer_id, sorted(banks)))
        for bank in banks.values():
            bank.release()

    sources = module.load_nvfp4_expert_source_banks(
        "unused",
        _config(),
        _source_spec(module),
        drop_page_cache=lambda _path: None,
        primary=False,
        layer_sink=sink,
    )

    assert seen == [
        (0, ["down_packed", "gate_up_packed"]),
        (1, ["down_packed", "gate_up_packed"]),
    ]
    assert [int(t[0]) for t in sources["gate_up_packed"]] == [1, 2]
    assert [int(t[0]) for t in sources["down_packed"]] == [11, 12]


def test_converter_parallel_request_falls_back_to_bounded_serial_and_releases_on_sink_error(
    monkeypatch,
):
    current = {}

    def fake_stream(_model_path, config, _spec, *, drop_page_cache, layer_sink):
        banks = {
            "gate_up_packed": FakeBank(1),
            "down_packed": FakeBank(2),
        }
        current.update(banks)
        layer_sink(0, banks)
        pytest.fail("sink failure should have propagated")
        return {"layers_streamed": config.num_moe_layers, "tensors_read": 0}

    module = _load_module(monkeypatch, fake_stream)
    monkeypatch.delitem(sys.modules, "freetoken.models.weight", raising=False)

    def failing_sink(_layer_id, _banks):
        raise RuntimeError("synthetic sink failure")

    with pytest.raises(RuntimeError, match="synthetic sink failure"):
        module.load_nvfp4_expert_source_banks_parallel(
            "unused",
            _config(),
            _source_spec(module),
            drop_page_cache=lambda _path: None,
            primary=False,
            workers=64,
            layer_sink=failing_sink,
        )

    assert current
    assert all(bank.released for bank in current.values())
    assert "freetoken.models.weight" not in sys.modules


def test_parallel_converter_adapter_composes_with_actual_one_layer_streamer(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    state = {"live_layers": 0, "max_live_layers": 0}
    released = []

    class TrackingBank:
        def __init__(self, tensor):
            self.tensor = tensor
            self.nbytes = tensor.numel() * tensor.element_size()
            self.released = False

        def release(self):
            assert not self.released
            self.released = True
            released.append(self)
            state["live_layers"] -= 1 / 6

    def alloc_banks(specs):
        assert abs(state["live_layers"]) < 1e-9
        state["live_layers"] = 1.0
        state["max_live_layers"] = max(state["max_live_layers"], 1)
        return {
            name: TrackingBank(torch.empty(shape, dtype=dtype))
            for name, (shape, dtype) in specs.items()
        }

    # Load the real one-layer streamer under the package name used by the adapter.
    checkpoint = types.ModuleType("freetoken.checkpoint")
    checkpoint.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken.checkpoint", checkpoint)

    moe = types.ModuleType("freetoken.moe")
    moe.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken.moe", moe)
    host_banks = types.ModuleType("freetoken.moe.host_banks")
    host_banks.alloc_banks = alloc_banks
    monkeypatch.setitem(sys.modules, "freetoken.moe.host_banks", host_banks)

    low_path = (
        Path(__file__).resolve().parents[2]
        / "python" / "freetoken" / "checkpoint" / "low_memory_nvfp4.py"
    )
    low_spec = importlib.util.spec_from_file_location(
        "freetoken.checkpoint.low_memory_nvfp4", low_path
    )
    assert low_spec is not None and low_spec.loader is not None
    low_module = importlib.util.module_from_spec(low_spec)
    monkeypatch.setitem(
        sys.modules, "freetoken.checkpoint.low_memory_nvfp4", low_module
    )
    low_spec.loader.exec_module(low_module)

    # Then load the real generic adapter, preserving the real streamer module above.
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken", freetoken)
    utils = types.ModuleType("freetoken.utils")
    utils.download_hf_weight = lambda path: path
    monkeypatch.setitem(sys.modules, "freetoken.utils", utils)
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    module_name = "nvfp4_bounded_composed_under_test"
    module_spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    module_spec.loader.exec_module(module)

    tensors = {}
    layers, experts, hidden, inter = 2, 2, 16, 16
    for layer in range(layers):
        for expert in range(experts):
            value = 10 * layer + expert + 1
            for proj in ("gate", "up", "down"):
                wshape = (
                    (inter, hidden // 2)
                    if proj in ("gate", "up")
                    else (hidden, inter // 2)
                )
                sshape = (
                    (inter, hidden // 16)
                    if proj in ("gate", "up")
                    else (hidden, inter // 16)
                )
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight"] = torch.full(
                    wshape, value, dtype=torch.uint8
                )
                tensors[
                    f"layer.{layer}.expert.{expert}.{proj}.weight_scale"
                ] = torch.full(sshape, float(value), dtype=torch.float32)
                tensors[
                    f"layer.{layer}.expert.{expert}.{proj}.weight_scale_2"
                ] = torch.tensor([float(value)], dtype=torch.float32)

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        __import__("json").dumps(
            {"weight_map": {name: shard for name in tensors}}
        ),
        encoding="utf-8",
    )
    config = types.SimpleNamespace(
        num_moe_layers=layers,
        num_layers=layers,
        first_k_dense_replace=0,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )
    source_spec = module.Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            r"^layer\.(?P<layer>\d+)\.expert\.(?P<expert>\d+)\."
            r"(?P<proj>gate|up|down)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
        ),
        proj_to_role={"gate": "gate", "up": "up", "down": "down"},
        layer_to_bank=lambda layer, _config: layer,
        desc="synthetic NVFP4",
    )

    seen = []

    def sink(layer_id, banks):
        assert state["live_layers"] == 1.0
        seen.append(layer_id)
        for bank in banks.values():
            bank.release()
        assert abs(state["live_layers"]) < 1e-9

    sources = module.load_nvfp4_expert_source_banks_parallel(
        str(tmp_path),
        config,
        source_spec,
        drop_page_cache=lambda _path: None,
        primary=False,
        workers=64,
        layer_sink=sink,
    )

    assert seen == [0, 1]
    assert state["max_live_layers"] == 1
    assert abs(state["live_layers"]) < 1e-9
    assert len(released) == layers * 6
    assert set(sources) == {
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    }
    assert all(len(per_layer) == layers for per_layer in sources.values())
