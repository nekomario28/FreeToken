from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/freetoken/experimental/file_backed_ftw_cpu_convert.py"
SPEC = importlib.util.spec_from_file_location("file_backed_ftw_cpu_convert_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
make_loader = MODULE._make_low_memory_native_nvfp4_loader
temporary_loader = MODULE._temporary_expert_loader


def qwen_config(*, quant="nvfp4"):
    return SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        expert_quant=quant,
        num_moe_layers=2,
        num_experts=4,
    )


def test_adapter_streams_through_canonical_layer_sink_and_returns_native_bundle():
    calls = []
    sink_calls = []
    marker_spec = object()

    def streamer(model_path, config, spec, *, drop_page_cache, layer_sink):
        calls.append((model_path, config, spec, drop_page_cache))
        layer_sink(0, {"synthetic": object()})
        layer_sink(1, {"synthetic": object()})
        return {"layers_streamed": 2, "expert_bank_bytes_streamed": 123}

    def bundle_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    drop = lambda _path: None
    loader = make_loader(
        streamer=streamer,
        spec=marker_spec,
        drop_page_cache=drop,
        bundle_factory=bundle_factory,
    )
    config = qwen_config()
    bundle = loader(
        "/model",
        config,
        device=object(),
        dtype=object(),
        layer_sink=lambda layer_id, banks: sink_calls.append((layer_id, banks)),
    )

    assert calls == [("/model", config, marker_spec, drop)]
    assert [layer_id for layer_id, _banks in sink_calls] == [0, 1]
    assert bundle.quant_format == "nvfp4"
    assert bundle.sources == {}
    assert bundle.streamed is True


def test_adapter_fails_closed_outside_exact_converter_contract():
    loader = make_loader(
        streamer=lambda *_args, **_kwargs: None,
        spec=object(),
        drop_page_cache=lambda _path: None,
        bundle_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    common = dict(device=object(), dtype=object(), layer_sink=lambda *_args: None)

    with pytest.raises(ValueError, match="does not support dummy"):
        loader("/model", qwen_config(), dummy=True, **common)
    with pytest.raises(ValueError, match="requires layer_sink"):
        loader("/model", qwen_config(), device=object(), dtype=object())
    with pytest.raises(ValueError, match="serving residency"):
        loader("/model", qwen_config(), layer_residency=["pageable", "pageable"], **common)

    bad_arch = qwen_config()
    bad_arch.architectures = ["OtherMoeForCausalLM"]
    with pytest.raises(ValueError, match="Qwen3_5MoeForConditionalGeneration"):
        loader("/model", bad_arch, **common)
    with pytest.raises(ValueError, match="requires native ModelOpt NVFP4"):
        loader("/model", qwen_config(quant="fp8_block"), **common)


def test_temporary_loader_restores_original_on_success_and_failure():
    original = object()
    replacement = object()
    module = SimpleNamespace(load_expert_banks=original)

    with temporary_loader(module, replacement):
        assert module.load_expert_banks is replacement
    assert module.load_expert_banks is original

    with pytest.raises(RuntimeError, match="boom"):
        with temporary_loader(module, replacement):
            assert module.load_expert_banks is replacement
            raise RuntimeError("boom")
    assert module.load_expert_banks is original
