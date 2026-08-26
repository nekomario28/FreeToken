from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "python/freetoken"
EXPERIMENTAL_ROOT = PACKAGE_ROOT / "experimental"

freetoken_pkg = types.ModuleType("freetoken")
freetoken_pkg.__path__ = [str(PACKAGE_ROOT)]
experimental_pkg = types.ModuleType("freetoken.experimental")
experimental_pkg.__path__ = [str(EXPERIMENTAL_ROOT)]
sys.modules.setdefault("freetoken", freetoken_pkg)
sys.modules.setdefault("freetoken.experimental", experimental_pkg)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("freetoken.experimental.ftw_resource_gate", EXPERIMENTAL_ROOT / "ftw_resource_gate.py")
CONVERT = _load(
    "freetoken.experimental.low_memory_ftw_cpu_convert",
    EXPERIMENTAL_ROOT / "low_memory_ftw_cpu_convert.py",
)


class FakeReport:
    def as_dict(self):
        return {
            "source_shards": 1,
            "source_tensor_bytes": 1234,
            "expert_layer_bytes": 567,
        }


def _config():
    return SimpleNamespace(num_moe_layers=2)


def _good_index():
    return {
        "format": "freetoken_weight",
        "quant_format": "nvfp4",
        "expert_bank_num_layers": 2,
        "counts": {"weight": 3, "experts_bank": 12},
        "fingerprint": "synthetic-fp",
    }


def test_staged_canonical_conversion_publishes_receipt_only_after_success(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"
    calls = []

    def fake_convert(model_path, out_dir, *, shard_limit, device):
        calls.append((model_path, out_dir, shard_limit, device))
        stage = Path(out_dir)
        stage.mkdir(parents=True, exist_ok=False)
        (stage / "freetoken_weight.json").write_text("{}", encoding="utf-8")
        return _good_index()

    index, receipt = CONVERT._atomic_staged_canonical_conversion(
        str(source),
        str(output),
        model_config=_config(),
        preflight_report=FakeReport(),
        convert_fn=fake_convert,
        shard_limit=4096,
        device="cuda:0",
    )

    assert index == _good_index()
    assert output.is_dir()
    receipt_path = output / CONVERT._RECEIPT_NAME
    assert receipt_path.is_file()
    on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert on_disk == receipt
    assert receipt["conversion_target"] == "cpu_file_backed_native_nvfp4"
    assert receipt["quant_format"] == "nvfp4"
    assert receipt["expert_bank_num_layers"] == 2
    assert receipt["device"] == "cuda:0"
    assert calls[0][2:] == (4096, "cuda:0")
    assert not list(tmp_path.glob(".result-ftw.partial-*"))


def test_publish_never_replaces_output_created_during_conversion(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"

    def racing_convert(_model_path, out_dir, **_kwargs):
        stage = Path(out_dir)
        stage.mkdir(parents=True, exist_ok=False)
        (stage / "freetoken_weight.json").write_text("{}", encoding="utf-8")
        output.mkdir()
        (output / "owner-marker").write_text("do-not-replace", encoding="utf-8")
        return _good_index()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        CONVERT._atomic_staged_canonical_conversion(
            str(source),
            str(output),
            model_config=_config(),
            preflight_report=FakeReport(),
            convert_fn=racing_convert,
            shard_limit=4096,
            device="cuda:0",
        )

    assert (output / "owner-marker").read_text(encoding="utf-8") == "do-not-replace"
    assert not list(tmp_path.glob(".result-ftw.partial-*"))


def test_failed_canonical_conversion_never_publishes_and_cleans_partial(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"

    def fail_convert(_model_path, out_dir, **_kwargs):
        stage = Path(out_dir)
        stage.mkdir(parents=True, exist_ok=False)
        (stage / "partial.ftw").write_bytes(b"partial")
        raise RuntimeError("synthetic canonical conversion failure")

    with pytest.raises(RuntimeError, match="synthetic canonical conversion failure"):
        CONVERT._atomic_staged_canonical_conversion(
            str(source),
            str(output),
            model_config=_config(),
            preflight_report=FakeReport(),
            convert_fn=fail_convert,
            shard_limit=4096,
            device="cuda:0",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".result-ftw.partial-*"))


def test_bad_canonical_index_is_not_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def run_with(index, output):
        def fake_convert(_model_path, out_dir, **_kwargs):
            Path(out_dir).mkdir(parents=True, exist_ok=False)
            return index

        return CONVERT._atomic_staged_canonical_conversion(
            str(source),
            str(output),
            model_config=_config(),
            preflight_report=FakeReport(),
            convert_fn=fake_convert,
            shard_limit=4096,
            device="cuda:0",
        )

    wrong_quant = tmp_path / "wrong-quant"
    with pytest.raises(RuntimeError, match="unexpected quant_format"):
        run_with({**_good_index(), "quant_format": "nvfp4_marlin"}, wrong_quant)
    assert not wrong_quant.exists()

    wrong_layers = tmp_path / "wrong-layers"
    with pytest.raises(RuntimeError, match="wrote 1 expert layers"):
        run_with({**_good_index(), "expert_bank_num_layers": 1}, wrong_layers)
    assert not wrong_layers.exists()

    no_experts = tmp_path / "no-experts"
    with pytest.raises(RuntimeError, match="no expert-bank entries"):
        run_with({**_good_index(), "counts": {"weight": 3, "experts_bank": 0}}, no_experts)
    assert not no_experts.exists()


def test_execute_refuses_existing_output_even_when_empty(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"
    output.mkdir()

    with pytest.raises(ValueError, match="non-existing output path"):
        CONVERT._atomic_staged_canonical_conversion(
            str(source),
            str(output),
            model_config=_config(),
            preflight_report=FakeReport(),
            convert_fn=lambda *_args, **_kwargs: _good_index(),
            shard_limit=4096,
            device="cuda:0",
        )


def test_cli_is_preflight_by_default_and_execute_is_explicit():
    parser = CONVERT._build_parser()
    preflight = parser.parse_args(["--model", "/m", "--out", "/o"])
    execute = parser.parse_args(["--model", "/m", "--out", "/o", "--execute"])
    assert preflight.execute is False
    assert execute.execute is True
    assert preflight.shard_limit_gib == 8.0
    assert preflight.device == "cuda:0"


def test_preflight_resolver_reads_local_json_without_freetoken_runtime_import(monkeypatch, tmp_path):
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "quantization_config": {"quant_algo": "NVFP4"},
        "num_experts": 128,
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 40,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "freetoken.utils" or name.startswith("freetoken.models"):
            raise AssertionError(f"preflight imported runtime module {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cfg = CONVERT._resolve_qwen35_preflight_config(str(tmp_path))
    assert cfg.num_experts == 128
    assert cfg.num_moe_layers == 40


def test_lightweight_preflight_config_accepts_pure_and_mixed_nvfp4_without_runtime_imports():
    pure = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        quantization_config={"quant_algo": "NVFP4"},
        num_experts=128,
        hidden_size=2048,
        moe_intermediate_size=768,
        num_hidden_layers=40,
    )
    cfg = CONVERT._preflight_config_from_hf(pure)
    assert (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_moe_layers) == (
        128, 2048, 768, 40
    )

    mixed = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_experts=64,
            hidden_size=1024,
            moe_intermediate_size=512,
            num_hidden_layers=24,
        ),
        quantization_config={
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
                "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
            },
        },
    )
    mixed_cfg = CONVERT._preflight_config_from_hf(mixed)
    assert mixed_cfg.expert_quant == "nvfp4"
    assert mixed_cfg.num_moe_layers == 24


def test_lightweight_preflight_config_fails_closed_on_arch_or_quant_mismatch():
    wrong_arch = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        quantization_config={"quant_algo": "NVFP4"},
    )
    with pytest.raises(ValueError, match="supports .* only"):
        CONVERT._preflight_config_from_hf(wrong_arch)

    wrong_quant = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        quantization_config={"quant_algo": "FP8"},
        num_experts=64,
        hidden_size=1024,
        moe_intermediate_size=512,
        num_hidden_layers=24,
    )
    with pytest.raises(ValueError, match="native NVFP4"):
        CONVERT._preflight_config_from_hf(wrong_quant)
