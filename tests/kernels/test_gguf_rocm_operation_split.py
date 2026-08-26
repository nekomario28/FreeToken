"""Routing contract for ROCm GGUF operation-split JIT modules.

These tests deliberately do not compile a native extension. Physical gfx1101 coverage
lives in the ROCm validation receipts; this file protects the cheap Python dispatch
contract so a future refactor cannot silently route one public GGUF op back through the
pathological monolithic ROCm translation unit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kernel import gguf


_EXPECTED_SOURCES = {
    "dequant": "gguf_dequant_kernel.cu",
    "mmvq": "gguf_mmvq_kernel.cu",
    "mmq": "gguf_mmq_kernel.cu",
    "moe_vec": "gguf_moe_vec_kernel.cu",
    "moe": "gguf_moe_kernel.cu",
}


def test_rocm_operation_source_contract():
    assert gguf._ROCM_OPERATION_SOURCES == _EXPECTED_SOURCES


def test_operation_module_uses_rocm_family(monkeypatch):
    seen = []
    marker = object()
    monkeypatch.setattr(gguf, "_is_rocm", lambda: True)
    monkeypatch.setattr(gguf, "_rocm_module", lambda operation: seen.append(operation) or marker)
    monkeypatch.setattr(gguf, "_module", lambda: pytest.fail("ROCm dispatch reached monolithic module"))

    assert gguf._operation_module("mmvq") is marker
    assert seen == ["mmvq"]


def test_operation_module_keeps_cuda_monolith(monkeypatch):
    marker = object()
    monkeypatch.setattr(gguf, "_is_rocm", lambda: False)
    monkeypatch.setattr(gguf, "_module", lambda: marker)
    monkeypatch.setattr(gguf, "_rocm_module", lambda operation: pytest.fail(f"CUDA reached ROCm split {operation}"))

    assert gguf._operation_module("dequant") is marker


def test_rocm_loader_rejects_unknown_family_before_build():
    with pytest.raises(ValueError, match="unknown ROCm GGUF operation"):
        gguf._rocm_module("not-an-operation")


def test_public_wrappers_route_to_owned_operation_family(monkeypatch):
    calls = []

    def op(name):
        def invoke(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name
        return invoke

    modules = {
        family: SimpleNamespace(
            ggml_dequantize=op("dequant"),
            ggml_mul_mat_vec_a8=op("mmvq"),
            ggml_mul_mat_a8=op("mmq"),
            ggml_moe_a8_vec=op("moe_vec"),
            ggml_moe_a8=op("moe"),
            ggml_moe_get_block_size=op("moe_block"),
        )
        for family in _EXPECTED_SOURCES
    }
    requested = []

    def load_family(family):
        requested.append(family)
        return modules[family]

    monkeypatch.setattr(gguf, "_operation_module", load_family)

    w = torch.empty(1, dtype=torch.uint8)
    x = torch.empty(1)
    ids = torch.empty(1, dtype=torch.int32)

    assert gguf.ggml_dequantize(w, 2, 1, 32) == "dequant"
    assert gguf.ggml_mul_mat_vec_a8(w, x, 2, 1) == "mmvq"
    assert gguf.ggml_mul_mat_a8(w, x, 2, 1) == "mmq"
    assert gguf.ggml_moe_a8_vec(x, w, ids, 1, 2, 1, 1) == "moe_vec"
    assert gguf.ggml_moe_a8(x, w, ids, ids, ids, 2, 1, 1, 1) == "moe"
    assert gguf.ggml_moe_get_block_size(2) == "moe_block"

    assert requested == ["dequant", "mmvq", "mmq", "moe_vec", "moe", "moe"]
