from __future__ import annotations

from freetoken.experimental.ftw_native_cpu_convert import _make_cpu_decode_expert_loader


def test_native_ftw_conversion_wrapper_forces_cpu_decode_and_preserves_arguments():
    seen = {}

    def original(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "banks"

    wrapped = _make_cpu_decode_expert_loader(original)
    marker = object()
    result = wrapped(
        "model",
        marker,
        device="device-marker",
        dtype="dtype-marker",
        parallel=False,
        workers=3,
        chunk=4096,
        decode_target="gpu",
        layer_sink="sink-marker",
        layer_residency=["locked"],
    )

    assert result == "banks"
    assert seen["args"] == ("model", marker)
    assert seen["kwargs"] == {
        "device": "device-marker",
        "dtype": "dtype-marker",
        "parallel": False,
        "workers": 3,
        "chunk": 4096,
        "decode_target": "cpu",
        "layer_sink": "sink-marker",
        "layer_residency": ["locked"],
    }
