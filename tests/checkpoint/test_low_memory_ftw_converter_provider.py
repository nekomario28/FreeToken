from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/freetoken/experimental/low_memory_ftw_converter.py"
SPEC = importlib.util.spec_from_file_location("low_memory_ftw_converter_provider_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class FakeExpertBanks:
    def __init__(self, quant_format, sources, *, streamed=False, **kwargs):
        self.quant_format = quant_format
        self.sources = sources
        self.streamed = streamed
        self.kwargs = kwargs


def _config():
    return SimpleNamespace(
        architectures=("Qwen3_5MoeForConditionalGeneration",),
        expert_quant="nvfp4",
    )


def _loader_with_fake_stream(calls):
    def fake_stream(model_path, model_config, source_spec, layer_sink):
        calls.append((model_path, model_config, source_spec, layer_sink))
        return {"layers_streamed": 2}

    return CONVERTER._make_low_memory_loader(
        FakeExpertBanks,
        source_spec_for_model=lambda _config: "synthetic-spec",
        stream_one_layer=fake_stream,
    )


def test_provider_routes_only_through_one_layer_stream():
    calls = []
    sink = object()
    config = _config()
    loader = _loader_with_fake_stream(calls)

    banks = loader(
        "/synthetic/model",
        config,
        device="cuda:0",
        dtype="bf16",
        layer_sink=sink,
    )

    assert calls == [("/synthetic/model", config, "synthetic-spec", sink)]
    assert banks.quant_format == "nvfp4"
    assert banks.streamed is True
    assert set(banks.sources) == {
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    }
    assert all(per_layer == [] for per_layer in banks.sources.values())


def test_provider_fails_closed_outside_converter_contract():
    calls = []
    loader = _loader_with_fake_stream(calls)
    common = dict(
        model_path="/synthetic/model",
        model_config=_config(),
        device="cuda:0",
        dtype="bf16",
    )

    with pytest.raises(ValueError, match="requires the canonical converter layer sink"):
        loader(**common)
    with pytest.raises(ValueError, match="does not support dummy"):
        loader(**common, dummy=True, layer_sink=object())
    with pytest.raises(ValueError, match="owns serial one-layer"):
        loader(**common, parallel=True, layer_sink=object())
    with pytest.raises(ValueError, match="does not accept a serving residency plan"):
        loader(**common, layer_sink=object(), layer_residency=["pageable"])
    assert calls == []


def _fake_expert_banks_module():
    def original_loader(*_args, **_kwargs):
        return "original"

    return SimpleNamespace(load_expert_banks=original_loader, ExpertBanks=FakeExpertBanks), original_loader


def test_bounded_patch_restores_provider_after_success():
    fake_mod, original = _fake_expert_banks_module()
    observed = []

    def fake_convert(model_path, out_dir, **kwargs):
        observed.append((model_path, out_dir, kwargs, fake_mod.load_expert_banks))
        assert fake_mod.load_expert_banks is not original
        return {"ok": True}

    result = CONVERTER.convert_checkpoint_low_memory_nvfp4(
        "/synthetic/model",
        "/synthetic/out",
        dtype="bf16",
        shard_limit=4096,
        device="cuda:0",
        _convert_fn=fake_convert,
        _expert_banks_mod=fake_mod,
    )

    assert result == {"ok": True}
    assert observed[0][2]["moe_backend"] == "offload"
    assert fake_mod.load_expert_banks is original


def test_bounded_patch_restores_provider_after_failure():
    fake_mod, original = _fake_expert_banks_module()

    def fail_convert(*_args, **_kwargs):
        assert fake_mod.load_expert_banks is not original
        raise RuntimeError("synthetic canonical conversion failure")

    with pytest.raises(RuntimeError, match="synthetic canonical conversion failure"):
        CONVERTER.convert_checkpoint_low_memory_nvfp4(
            "/synthetic/model",
            "/synthetic/out",
            dtype="bf16",
            shard_limit=4096,
            device="cuda:0",
            _convert_fn=fail_convert,
            _expert_banks_mod=fake_mod,
        )

    assert fake_mod.load_expert_banks is original


def test_non_offload_backend_is_rejected_before_patch():
    fake_mod, original = _fake_expert_banks_module()
    with pytest.raises(ValueError, match="requires moe_backend='offload'"):
        CONVERTER.convert_checkpoint_low_memory_nvfp4(
            "/synthetic/model",
            "/synthetic/out",
            moe_backend="resident",
            _convert_fn=lambda *_args, **_kwargs: None,
            _expert_banks_mod=fake_mod,
        )
    assert fake_mod.load_expert_banks is original
