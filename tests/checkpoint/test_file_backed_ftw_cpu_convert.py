from __future__ import annotations

import importlib.util
import threading
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
require_preflight = MODULE._require_metadata_preflight


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


def test_adapter_rejects_incomplete_streamer_before_returning_streamed_bundle():
    def streamer(_model_path, _config, _spec, *, drop_page_cache, layer_sink):
        layer_sink(0, {"synthetic": object()})
        return {"layers_streamed": 1, "expert_bank_bytes_streamed": 1}

    loader = make_loader(
        streamer=streamer,
        spec=object(),
        drop_page_cache=lambda _path: None,
        bundle_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    with pytest.raises(RuntimeError, match="incomplete layer set: 1/2"):
        loader(
            "/model",
            qwen_config(),
            device=object(),
            dtype=object(),
            layer_sink=lambda *_args: None,
        )


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


def test_temporary_loader_is_thread_scoped_and_restores_original_on_success_and_failure():
    calls = []

    def original(value):
        calls.append(("original", value))
        return "original"

    def replacement(value):
        calls.append(("replacement", value))
        return "replacement"

    module = SimpleNamespace(load_expert_banks=original)

    with temporary_loader(module, replacement):
        assert module.load_expert_banks("owner") == "replacement"
        other_result = []
        thread = threading.Thread(
            target=lambda: other_result.append(module.load_expert_banks("other"))
        )
        thread.start()
        thread.join()
        assert other_result == ["original"]
    assert module.load_expert_banks is original
    assert calls == [("replacement", "owner"), ("original", "other")]

    with pytest.raises(RuntimeError, match="boom"):
        with temporary_loader(module, replacement):
            assert module.load_expert_banks("owner-2") == "replacement"
            raise RuntimeError("boom")
    assert module.load_expert_banks is original


def test_metadata_preflight_blocks_before_conversion_contract_is_entered():
    calls = []

    def blocked(model_path, out_dir):
        calls.append((model_path, out_dir))
        return {
            "admission": "BLOCK",
            "blockers": ["wrong architecture", "unsafe output"],
        }

    with pytest.raises(ValueError, match="wrong architecture; unsafe output"):
        require_preflight("/model", "/out", preflight_fn=blocked)
    assert calls == [("/model", "/out")]


def test_metadata_preflight_accepts_only_explicit_unproven_state():
    allowed = {"admission": "METADATA_OK_RESOURCE_UNPROVEN", "warnings": ["resource not proven"]}
    assert require_preflight("/model", "/out", preflight_fn=lambda *_args: allowed) is allowed

    with pytest.raises(ValueError, match="unexpected admission"):
        require_preflight("/model", "/out", preflight_fn=lambda *_args: {"admission": "GO"})
