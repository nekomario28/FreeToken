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
