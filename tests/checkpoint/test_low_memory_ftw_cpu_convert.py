from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


class FakeWriter:
    def __init__(self, out_dir: str, *, shard_limit: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=False)
        self.shard_limit = shard_limit
        self.rows = []
        self._f = (self.out_dir / "freetoken-00000.ftw").open("wb")

    def add_tensor(self, name, tensor, *, kind):
        raw = bytes(tensor.detach().cpu().reshape(-1).view(torch.uint8).tolist())
        self._f.write(raw)
        self.rows.append((name, kind, len(raw)))

    def finalize(self, meta):
        self._f.close()
        self._f = None
        index = {
            "format": "freetoken_weight",
            "tensors": [
                {"name": name, "kind": kind, "nbytes": nbytes}
                for name, kind, nbytes in self.rows
            ],
            **meta,
        }
        (self.out_dir / "freetoken_weight.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        return index


def _config():
    return SimpleNamespace(num_moe_layers=2)


def _metadata_copier(_model_path: str, out_dir: str):
    Path(out_dir, "config.json").write_text("{}", encoding="utf-8")
    return ["config.json"]


def _expert_streamer(model_path, model_config, spec, *, writer, drop_page_cache, alloc_layer):
    assert model_path.endswith("source")
    assert model_config.num_moe_layers == 2
    assert spec == "synthetic-spec"
    assert alloc_layer is None
    drop_page_cache("synthetic-shard")
    written = 0
    nbytes = 0
    for layer in range(2):
        tensor = torch.tensor([layer + 10], dtype=torch.uint8)
        writer.add_tensor(f"gate_up_packed#L{layer:05d}", tensor, kind="experts_bank")
        written += 1
        nbytes += tensor.numel() * tensor.element_size()
    return {
        "layers_streamed": 2,
        "ftw_entries_written": written,
        "ftw_expert_bytes_written": nbytes,
        "expert_bank_bytes_streamed": nbytes,
    }


def test_staged_writer_publishes_only_after_finalize(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"
    dropped = []

    index = CONVERT._write_native_nvfp4_ftw(
        str(source),
        str(output),
        _config(),
        "synthetic-spec",
        dense_weights=lambda: iter([
            ("dense.weight", torch.tensor([1.0, 2.0], dtype=torch.float32)),
        ]),
        expert_streamer=_expert_streamer,
        writer_factory=FakeWriter,
        metadata_copier=_metadata_copier,
        drop_page_cache=dropped.append,
        fingerprint="synthetic-fp",
        shard_limit=4096,
    )

    assert output.is_dir()
    assert (output / "freetoken_weight.json").is_file()
    assert (output / "config.json").is_file()
    assert index["quant_format"] == "nvfp4"
    assert index["expert_bank_num_layers"] == 2
    assert index["conversion_target"] == "cpu_file_backed_native_nvfp4"
    assert index["counts"] == {"weight": 1, "experts_bank": 2}
    assert index["bytes"] == {"weight": 8, "experts_bank": 2}
    assert dropped == ["synthetic-shard"]
    assert not list(tmp_path.glob(".result-ftw.partial-*"))


def test_failed_staged_conversion_never_publishes_and_cleans_partial(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"

    def fail_streamer(*args, writer, **kwargs):
        writer.add_tensor("partial#L00000", torch.tensor([1], dtype=torch.uint8), kind="experts_bank")
        raise RuntimeError("synthetic expert failure")

    with pytest.raises(RuntimeError, match="synthetic expert failure"):
        CONVERT._write_native_nvfp4_ftw(
            str(source),
            str(output),
            _config(),
            "synthetic-spec",
            dense_weights=lambda: iter([]),
            expert_streamer=fail_streamer,
            writer_factory=FakeWriter,
            metadata_copier=_metadata_copier,
            drop_page_cache=lambda _path: None,
            fingerprint=None,
            shard_limit=4096,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".result-ftw.partial-*"))


def test_execute_refuses_existing_output_even_when_empty(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "result-ftw"
    output.mkdir()

    with pytest.raises(ValueError, match="non-existing output path"):
        CONVERT._write_native_nvfp4_ftw(
            str(source), str(output), _config(), "synthetic-spec",
            dense_weights=lambda: iter([]),
            expert_streamer=_expert_streamer,
            writer_factory=FakeWriter,
            metadata_copier=_metadata_copier,
            drop_page_cache=lambda _path: None,
            fingerprint=None,
        )


def test_cli_is_preflight_by_default_and_execute_is_explicit():
    parser = CONVERT._build_parser()
    preflight = parser.parse_args(["--model", "/m", "--out", "/o"])
    execute = parser.parse_args(["--model", "/m", "--out", "/o", "--execute"])
    assert preflight.execute is False
    assert execute.execute is True
    assert preflight.shard_limit_gib == 8.0
