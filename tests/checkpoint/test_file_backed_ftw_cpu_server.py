from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "experimental" / "file_backed_ftw_cpu_server.py"
)
SPEC = importlib.util.spec_from_file_location("file_backed_ftw_cpu_server_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def test_option_parser_accepts_separate_and_equals_forms():
    assert SERVER._option_value(["--moe-backend", "cpu"], "--moe-backend") == "cpu"
    assert SERVER._option_value(["--model", "x", "--moe-backend=cpu"], "--moe-backend") == "cpu"
    assert SERVER._option_value(["--model", "x"], "--moe-backend") is None
    assert SERVER._option_value(["--moe-backend"], "--moe-backend") is None


def test_launcher_requires_explicit_cpu_backend_before_runtime_imports():
    SERVER._require_explicit_cpu_backend(["--moe-backend", "cpu"], prog="experiment")
    for argv in ([], ["--moe-backend", "offload"], ["--moe-backend=hybrid"]):
        with pytest.raises(SystemExit, match="explicit --moe-backend cpu"):
            SERVER._require_explicit_cpu_backend(argv, prog="experiment")


def test_loader_wrapper_forwards_only_file_backed_provider_contract():
    calls = []

    def provider(model_path, model_config, *, decode_target, layer_residency):
        calls.append((model_path, model_config, decode_target, layer_residency))
        return "banks"

    loader = SERVER._make_file_backed_loader(provider)
    config = SimpleNamespace(num_moe_layers=2, num_experts=3)
    residency = ["locked", "locked"]
    result = loader(
        "/checkpoint",
        config,
        device=object(),
        dtype=object(),
        dummy=False,
        parallel=True,
        workers=99,
        chunk=1234,
        decode_target="cpu",
        layer_sink=None,
        layer_residency=residency,
    )
    assert result == "banks"
    assert calls == [("/checkpoint", config, "cpu", residency)]


def test_loader_wrapper_rejects_dummy_and_converter_sink_without_provider_call():
    calls = []

    def provider(*args, **kwargs):
        calls.append((args, kwargs))
        return "unexpected"

    loader = SERVER._make_file_backed_loader(provider)
    common = dict(
        model_path="/checkpoint",
        model_config=SimpleNamespace(),
        device=object(),
        dtype=object(),
        decode_target="cpu",
        layer_residency=["locked"],
    )
    with pytest.raises(ValueError, match="dummy"):
        loader(**common, dummy=True)
    with pytest.raises(ValueError, match="converter sink"):
        loader(**common, layer_sink=object())
    assert calls == []
