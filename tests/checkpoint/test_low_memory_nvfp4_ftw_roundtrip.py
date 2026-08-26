from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "python" / "freetoken" / "checkpoint"

SPEC_NVFP4 = SimpleNamespace(
    key_pattern=re.compile(
        r"^layer\.(?P<layer>\d+)\.expert\.(?P<expert>\d+)\."
        r"(?P<proj>gate|up|down)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate": "gate", "up": "up", "down": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="synthetic NVFP4 roundtrip",
)


class FakeBank:
    def __init__(self, tensor: torch.Tensor, release_cb):
        self.tensor = tensor
        self.nbytes = tensor.numel() * tensor.element_size()
        self._release_cb = release_cb
        self.released = False

    def release(self):
        assert not self.released
        self.released = True
        self._release_cb()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path: Path, *, layers=2, experts=1, hidden=16, inter=16):
    tensors = {}
    for layer in range(layers):
        for expert in range(experts):
            value = 10 * layer + expert + 1
            for proj in ("gate", "up", "down"):
                wshape = (inter, hidden // 2) if proj in ("gate", "up") else (hidden, inter // 2)
                sshape = (inter, hidden // 16) if proj in ("gate", "up") else (hidden, inter // 16)
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight"] = torch.full(
                    wshape, value, dtype=torch.uint8
                )
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight_scale"] = torch.full(
                    sshape, float(value), dtype=torch.float32
                )
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight_scale_2"] = torch.tensor(
                    [float(value)], dtype=torch.float32
                )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    return SimpleNamespace(
        num_moe_layers=layers,
        num_layers=layers,
        first_k_dense_replace=0,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )


def _tracking_allocator(state):
    def alloc(E, H, I):
        assert state["live"] == 0
        state["live"] = 1
        state["max_live"] = max(state["max_live"], state["live"])
        remaining = 6

        def released():
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                state["live"] = 0

        def bank(shape, dtype):
            return FakeBank(torch.empty(shape, dtype=dtype), released)

        return {
            "gate_up_packed": bank((E, 2 * I, H // 2), torch.uint8),
            "gate_up_scale": bank((E, 2 * I, H // 16), torch.float32),
            "gate_up_global": bank((E, 2 * I), torch.float16),
            "down_packed": bank((E, H, I // 2), torch.uint8),
            "down_scale": bank((E, H, I // 16), torch.float32),
            "down_global": bank((E, H), torch.float16),
        }

    return alloc


def test_real_ftw_writer_roundtrips_streamed_layers_into_file_backed_sources(tmp_path, monkeypatch):
    low_memory = _load_module(
        "roundtrip_low_memory_nvfp4",
        CHECKPOINT_ROOT / "low_memory_nvfp4.py",
    )

    utils_stub = types.ModuleType("freetoken.utils")
    utils_stub.init_logger = lambda *_args, **_kwargs: SimpleNamespace(
        warning=lambda *_a, **_k: None,
        info=lambda *_a, **_k: None,
    )
    monkeypatch.setitem(sys.modules, "freetoken.utils", utils_stub)
    ftw = _load_module("roundtrip_ftw_writer", CHECKPOINT_ROOT / "ftw.py")

    pkg = types.ModuleType("roundtrip_checkpoint")
    pkg.__path__ = [str(CHECKPOINT_ROOT)]
    monkeypatch.setitem(sys.modules, "roundtrip_checkpoint", pkg)
    core = _load_module(
        "roundtrip_checkpoint.mapped_ftw_core",
        CHECKPOINT_ROOT / "mapped_ftw_core.py",
    )
    mapped = _load_module(
        "roundtrip_checkpoint.mapped_ftw",
        CHECKPOINT_ROOT / "mapped_ftw.py",
    )

    source = tmp_path / "source"
    source.mkdir()
    config = _checkpoint(source)
    out = tmp_path / "ftw"
    writer = ftw.FTWWriter(str(out), shard_limit=4 * 4096)
    state = {"live": 0, "max_live": 0}

    stats = low_memory.stream_nvfp4_layers_to_ftw(
        str(source),
        config,
        SPEC_NVFP4,
        writer=writer,
        drop_page_cache=lambda _path: None,
        alloc_layer=_tracking_allocator(state),
    )
    index = writer.finalize({
        "quant_format": "nvfp4",
        "expert_bank_num_layers": config.num_moe_layers,
    })

    assert state == {"live": 0, "max_live": 1}
    assert stats["layers_streamed"] == 2
    assert stats["ftw_entries_written"] == 12
    expert_entries = [e for e in index["tensors"] if e["kind"] == "experts_bank"]
    assert len(expert_entries) == 12
    assert all("#L" in e["name"] for e in expert_entries)

    bundle = mapped.map_ftw_expert_sources(
        out,
        2,
        expected_banks={
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        },
        expected_quant_format="nvfp4",
        num_experts=1,
    )
    assert bundle.layer_residency == ["pageable", "pageable"]
    assert len(bundle.owners) == 12
    assert all(not owner.mapping.closed for owner in bundle.owners)

    assert int(bundle.sources["gate_up_packed"][0][0, 0, 0]) == 1
    assert int(bundle.sources["down_packed"][0][0, 0, 0]) == 1
    assert float(bundle.sources["gate_up_global"][0][0, 0]) == 1.0
    assert int(bundle.sources["gate_up_packed"][1][0, 0, 0]) == 11
    assert int(bundle.sources["down_packed"][1][0, 0, 0]) == 11
    assert float(bundle.sources["gate_up_global"][1][0, 0]) == 11.0

    # The mapper must resolve the actual FTW files written above, not a hand-built index.
    for owner in bundle.owners:
        assert owner.shard_path.parent == out.resolve()
        assert owner.nbytes == int(core.unique_entry(index, owner.name)["nbytes"])
