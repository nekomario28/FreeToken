from __future__ import annotations

import importlib.util
import json
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
make_dense_writer = MODULE._make_dense_passthrough_writer
resolve_embedding = MODULE._resolve_embedding_passthrough
temporary_loader = MODULE._temporary_expert_loader
temporary_cpu_device = MODULE._temporary_cpu_conversion_device
temporary_safe_open_skip = MODULE._temporary_safetensors_tensor_skip
temporary_ftw_writer = MODULE._temporary_ftw_writer
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


def test_cpu_conversion_device_shim_is_owner_thread_only_and_preserves_cuda_calls():
    calls = []

    class FakeDevice:
        def __init__(self, value):
            self.type = str(value).split(":", 1)[0]

    class FakeCuda:
        def set_device(self, device):
            calls.append((threading.get_ident(), str(device)))
            return "original"

    fake_torch = SimpleNamespace(device=FakeDevice, cuda=FakeCuda())
    original = fake_torch.cuda.set_device
    owner = threading.get_ident()

    with temporary_cpu_device(fake_torch):
        assert fake_torch.cuda.set_device("cpu") is None
        assert calls == []
        assert fake_torch.cuda.set_device("cuda:0") == "original"
        assert calls == [(owner, "cuda:0")]

        other = []
        thread = threading.Thread(target=lambda: other.append(fake_torch.cuda.set_device("cpu")))
        thread.start()
        thread.join()
        assert other == ["original"]
        assert calls[-1][1] == "cpu"
        assert calls[-1][0] != owner

    assert fake_torch.cuda.set_device.__func__ is original.__func__


def test_cpu_conversion_device_shim_restores_after_failure():
    class FakeDevice:
        def __init__(self, value):
            self.type = str(value).split(":", 1)[0]

    class FakeCuda:
        def set_device(self, _device):
            return "original"

    fake_torch = SimpleNamespace(device=FakeDevice, cuda=FakeCuda())
    original = fake_torch.cuda.set_device
    with pytest.raises(RuntimeError, match="boom"):
        with temporary_cpu_device(fake_torch):
            assert fake_torch.cuda.set_device("cpu") is None
            raise RuntimeError("boom")
    assert fake_torch.cuda.set_device.__func__ is original.__func__


def test_embedding_passthrough_resolver_accepts_only_one_plain_bf16_source(tmp_path: Path):
    raw_name = "model.language_model.embed_tokens.weight"
    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"synthetic")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {raw_name: shard.name}}), encoding="utf-8"
    )

    calls = []

    def tensor_entry(path, name):
        calls.append((Path(path).name, name))
        if name != raw_name:
            raise KeyError(name)
        return 128, 123456, "bfloat16", [32000, 2048]

    result = resolve_embedding(str(tmp_path), tensor_entry_fn=tensor_entry)
    assert result["name"] == "model.embed_tokens.weight"
    assert result["raw_name"] == raw_name
    assert result["payload_bytes"] == 123456
    assert result["dtype"] == "bfloat16"
    assert result["shape"] == [32000, 2048]
    assert Path(result["source_path"]) == shard.resolve()
    assert calls == [(shard.name, raw_name)]


def test_embedding_passthrough_resolver_rejects_quantized_and_ambiguous_sources(tmp_path: Path):
    raw_a = "model.language_model.embed_tokens.weight"
    raw_b = "language_model.embed_tokens.weight"
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"synthetic")

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {raw_a: shard.name, raw_a[:-7] + ".weight_scale": shard.name}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quantized/scaled"):
        resolve_embedding(
            str(tmp_path), tensor_entry_fn=lambda *_args: (0, 1, "bfloat16", [1, 1])
        )

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {raw_a: shard.name, raw_b: shard.name}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one"):
        resolve_embedding(
            str(tmp_path), tensor_entry_fn=lambda *_args: (0, 1, "bfloat16", [1, 1])
        )


def test_embedding_passthrough_resolver_rejects_non_bf16_or_wrong_shape(tmp_path: Path):
    raw_name = "model.language_model.embed_tokens.weight"
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"synthetic")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {raw_name: shard.name}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="requires BF16"):
        resolve_embedding(
            str(tmp_path), tensor_entry_fn=lambda *_args: (0, 8, "float16", [2, 2])
        )
    with pytest.raises(ValueError, match="rank-2"):
        resolve_embedding(
            str(tmp_path), tensor_entry_fn=lambda *_args: (0, 8, "bfloat16", [4])
        )


def test_safe_open_skip_hides_tensor_only_from_owner_and_restores():
    calls = []

    class Handle:
        def keys(self):
            return ["keep", "hidden"]

        def get_tensor(self, name):
            calls.append((threading.get_ident(), name))
            return f"tensor:{name}"

    class Context:
        def __enter__(self):
            return Handle()

        def __exit__(self, *_args):
            return False

    class SafeModule:
        def safe_open(self, path, *args, **kwargs):
            del path, args, kwargs
            return Context()

    module = SafeModule()
    original = module.safe_open
    with temporary_safe_open_skip(module, source_path="/tmp/source.safetensors", raw_name="hidden"):
        with module.safe_open("/tmp/source.safetensors") as handle:
            assert handle.keys() == ["keep"]
            assert handle.get_tensor("keep") == "tensor:keep"
            with pytest.raises(RuntimeError, match="must not be materialized"):
                handle.get_tensor("hidden")

        other = []

        def other_thread():
            with module.safe_open("/tmp/source.safetensors") as handle:
                other.append((handle.keys(), handle.get_tensor("hidden")))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert other == [(["keep", "hidden"], "tensor:hidden")]

    assert module.safe_open.__func__ is original.__func__


def test_dense_passthrough_writer_appends_once_and_corrects_counts():
    target = {
        "name": "model.embed_tokens.weight",
        "raw_name": "model.language_model.embed_tokens.weight",
        "source_path": "/tmp/source.safetensors",
    }

    class FakeWriter:
        def __init__(self):
            self._tensors = []
            self.calls = []

        def add_safetensors_passthrough(self, **kwargs):
            self.calls.append(kwargs)
            self._tensors.append({"name": kwargs["name"]})
            return {"payload_bytes": 1234, "max_read_buffer_bytes": 4096}

        def finalize(self, meta):
            return meta

    Writer = make_dense_writer(FakeWriter, target, chunk_bytes=4096)
    writer = Writer()
    result = writer.finalize({"counts": {"weight": 7, "experts_bank": 4}})
    assert result["counts"] == {"weight": 8, "experts_bank": 4}
    assert result["dense_passthrough"] == {
        "name": target["name"],
        "raw_name": target["raw_name"],
        "payload_bytes": 1234,
        "max_read_buffer_bytes": 4096,
    }
    assert writer.calls[0]["chunk_bytes"] == 4096

    duplicate = Writer()
    duplicate._tensors.append({"name": target["name"]})
    with pytest.raises(RuntimeError, match="already exists"):
        duplicate.finalize({"counts": {"weight": 1}})


def test_temporary_ftw_writer_is_owner_scoped_and_restored():
    class Original:
        pass

    class Replacement:
        pass

    module = SimpleNamespace(FTWWriter=Original)
    with temporary_ftw_writer(module, Replacement):
        assert isinstance(module.FTWWriter(), Replacement)
        other = []
        thread = threading.Thread(target=lambda: other.append(type(module.FTWWriter())))
        thread.start()
        thread.join()
        assert other == [Original]
    assert module.FTWWriter is Original


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
