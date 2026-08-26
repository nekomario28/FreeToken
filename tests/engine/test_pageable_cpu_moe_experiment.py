from __future__ import annotations

from freetoken.experimental.pageable_cpu_moe_server import _cpu_layer_spec, _make_pageable_settle


def test_cpu_layer_spec_requires_explicit_surface():
    assert _cpu_layer_spec(["--moe-cpu-layers", "1.0"]) == "1.0"
    assert _cpu_layer_spec(["--model", "x", "--moe-cpu-layers=0.5"]) == "0.5"
    assert _cpu_layer_spec(["--model", "x"]) is None


def test_pageable_settle_only_intercepts_locked():
    calls = []

    def original(bank, residency):
        calls.append((bank, residency))

    settle = _make_pageable_settle(original, locked_value="locked")
    marker = object()
    settle(marker, "locked")
    assert calls == []
    settle(marker, "pinned")
    assert calls == [(marker, "pinned")]
