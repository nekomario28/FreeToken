from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/freetoken/checkpoint/low_memory_nvfp4.py"
SPEC = importlib.util.spec_from_file_location("low_memory_nvfp4_bridge_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
stream_nvfp4_layers_to_ftw = MODULE.stream_nvfp4_layers_to_ftw

CONVERTER_PATH = ROOT / "python/freetoken/experimental/low_memory_ftw_converter.py"
CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "low_memory_ftw_converter_under_test", CONVERTER_PATH
)
assert CONVERTER_SPEC is not None and CONVERTER_SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(CONVERTER_SPEC)
CONVERTER_SPEC.loader.exec_module(CONVERTER)

SPEC_NVFP4 = SimpleNamespace(
    key_pattern=re.compile(
        r"^layer\.(?P<layer>\d+)\.expert\.(?P<expert>\d+)\."
        r"(?P<proj>gate|up|down)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate": "gate", "up": "up", "down": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="synthetic NVFP4",
)


class FakeBank:
    def __init__(self, tensor: torch.Tensor, release_cb):
        self.tensor = tensor
        self.nbytes = tensor.numel() * tensor.element_size()
        self._release_cb = release_cb
        self.released = False

    def release(self):
        assert not self.released
        self.released = True
        self._release_cb()


class FakeWriter:
    def __init__(self, *, fail_after: int | None = None):
        self.entries = []
        self.fail_after = fail_after

    def add_tensor(self, name, tensor, *, kind):
        if self.fail_after is not None and len(self.entries) >= self.fail_after:
            raise RuntimeError("synthetic writer failure")
        self.entries.append((name, tensor.clone(), kind))


def checkpoint(tmp_path: Path, *, layers=3, experts=2, hidden=16, inter=16):
    tensors = {}
    for layer in range(layers):
        for expert in range(experts):
            value = 10 * layer + expert + 1
            for proj in ("gate", "up", "down"):
                wshape = (inter, hidden // 2) if proj in ("gate", "up") else (hidden, inter // 2)
                sshape = (inter, hidden // 16) if proj in ("gate", "up") else (hidden, inter // 16)
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight"] = torch.full(
                    wshape, value, dtype=torch.uint8
                )
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight_scale"] = torch.full(
                    sshape, float(value), dtype=torch.float32
                )
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight_scale_2"] = torch.tensor(
                    [float(value)], dtype=torch.float32
                )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    return SimpleNamespace(
        num_moe_layers=layers,
        num_layers=layers,
        first_k_dense_replace=0,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )


def tracking_allocator(state, allocations):
    def alloc(E, H, I):
        assert state["live"] == 0
        allocations["count"] += 1
        state["live"] = 1
        state["max_live"] = max(state["max_live"], state["live"])
        remaining = 6

        def released():
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                state["live"] = 0

        def bank(shape, dtype):
            return FakeBank(torch.empty(shape, dtype=dtype), released)

        return {
            "gate_up_packed": bank((E, 2 * I, H // 2), torch.uint8),
            "gate_up_scale": bank((E, 2 * I, H // 16), torch.float32),
            "gate_up_global": bank((E, 2 * I), torch.float16),
            "down_packed": bank((E, H, I // 2), torch.uint8),
            "down_scale": bank((E, H, I // 16), torch.float32),
            "down_global": bank((E, H), torch.float16),
        }

    return alloc


def test_bridge_writes_six_per_layer_entries_and_never_holds_two_layers(tmp_path):
    config = checkpoint(tmp_path)
    writer = FakeWriter()
    state = {"live": 0, "max_live": 0}
    allocations = {"count": 0}

    result = stream_nvfp4_layers_to_ftw(
        str(tmp_path), config, SPEC_NVFP4,
        writer=writer,
        drop_page_cache=lambda _path: None,
        alloc_layer=tracking_allocator(state, allocations),
    )

    assert allocations["count"] == 3
    assert state == {"live": 0, "max_live": 1}
    assert len(writer.entries) == 18
    assert result["ftw_entries_written"] == 18
    assert all(kind == "experts_bank" for _name, _tensor, kind in writer.entries)
    assert {name.rsplit("#L", 1)[1] for name, _tensor, _kind in writer.entries} == {
        "00000", "00001", "00002"
    }
    written_bytes = sum(t.numel() * t.element_size() for _name, t, _kind in writer.entries)
    assert result["ftw_expert_bytes_written"] == written_bytes
    assert result["expert_bank_bytes_streamed"] == written_bytes

    layer0 = {name: tensor for name, tensor, _kind in writer.entries if name.endswith("#L00000")}
    assert int(layer0["gate_up_packed#L00000"][0, 0, 0]) == 1
    assert int(layer0["down_packed#L00000"][0, 0, 0]) == 1
    assert float(layer0["gate_up_global#L00000"][0, 0]) == 1.0


def test_bridge_writer_failure_releases_current_layer_and_never_allocates_next(tmp_path):
    config = checkpoint(tmp_path)
    writer = FakeWriter(fail_after=2)
    state = {"live": 0, "max_live": 0}
    allocations = {"count": 0}

    with pytest.raises(RuntimeError, match="synthetic writer failure"):
        stream_nvfp4_layers_to_ftw(
            str(tmp_path), config, SPEC_NVFP4,
            writer=writer,
            drop_page_cache=lambda _path: None,
            alloc_layer=tracking_allocator(state, allocations),
        )

    assert allocations["count"] == 1
    assert state == {"live": 0, "max_live": 1}
    assert len(writer.entries) == 2


def _qwen35_config():
    return SimpleNamespace(
        architectures=("Qwen3_5MoeForConditionalGeneration",),
        expert_quant="nvfp4",
    )


def test_experimental_loader_routes_only_through_one_layer_stream(monkeypatch):
    calls = []
    sink = object()
    monkeypatch.setattr(CONVERTER, "_source_spec_for_model", lambda _config: SPEC_NVFP4)

    def fake_stream(model_path, model_config, source_spec, layer_sink):
        calls.append((model_path, model_config, source_spec, layer_sink))
        return {"layers_streamed": 2}

    monkeypatch.setattr(CONVERTER, "_stream_one_layer_at_a_time", fake_stream)
    banks = CONVERTER._low_memory_load_expert_banks(
        "/synthetic/model",
        _qwen35_config(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        layer_sink=sink,
    )

    assert calls == [("/synthetic/model", calls[0][1], SPEC_NVFP4, sink)]
    assert banks.quant_format == "nvfp4"
    assert banks.streamed is True
    assert set(banks.sources) == {
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    }
    assert all(per_layer == [] for per_layer in banks.sources.values())


def test_experimental_loader_fails_closed_outside_converter_contract(monkeypatch):
    monkeypatch.setattr(CONVERTER, "_source_spec_for_model", lambda _config: SPEC_NVFP4)
    monkeypatch.setattr(CONVERTER, "_stream_one_layer_at_a_time", lambda *args: None)
    common = dict(
        model_path="/synthetic/model",
        model_config=_qwen35_config(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="requires the canonical converter layer sink"):
        CONVERTER._low_memory_load_expert_banks(**common)
    with pytest.raises(ValueError, match="does not support dummy"):
        CONVERTER._low_memory_load_expert_banks(**common, dummy=True, layer_sink=object())
    with pytest.raises(ValueError, match="owns serial one-layer"):
        CONVERTER._low_memory_load_expert_banks(**common, parallel=True, layer_sink=object())
    with pytest.raises(ValueError, match="does not accept a serving residency plan"):
        CONVERTER._low_memory_load_expert_banks(
            **common, layer_sink=object(), layer_residency=["pageable"]
        )


def test_experimental_converter_patches_only_for_bounded_canonical_call():
    import freetoken.moe.expert_banks as expert_banks_mod

    original = expert_banks_mod.load_expert_banks
    observed = []

    def fake_convert(model_path, out_dir, **kwargs):
        observed.append((model_path, out_dir, kwargs, expert_banks_mod.load_expert_banks))
        assert expert_banks_mod.load_expert_banks is CONVERTER._low_memory_load_expert_banks
        return {"ok": True}

    result = CONVERTER.convert_checkpoint_low_memory_nvfp4(
        "/synthetic/model",
        "/synthetic/out",
        dtype=torch.bfloat16,
        shard_limit=4096,
        device="cpu",
        _convert_fn=fake_convert,
    )

    assert result == {"ok": True}
    assert observed[0][2]["moe_backend"] == "offload"
    assert expert_banks_mod.load_expert_banks is original


def test_experimental_converter_restores_loader_after_failure():
    import freetoken.moe.expert_banks as expert_banks_mod

    original = expert_banks_mod.load_expert_banks

    def fail_convert(*_args, **_kwargs):
        assert expert_banks_mod.load_expert_banks is CONVERTER._low_memory_load_expert_banks
        raise RuntimeError("synthetic canonical conversion failure")

    with pytest.raises(RuntimeError, match="synthetic canonical conversion failure"):
        CONVERTER.convert_checkpoint_low_memory_nvfp4(
            "/synthetic/model",
            "/synthetic/out",
            dtype=torch.bfloat16,
            shard_limit=4096,
            device="cpu",
            _convert_fn=fail_convert,
        )

    assert expert_banks_mod.load_expert_banks is original
