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
PACKAGE_ROOT = ROOT / "python/freetoken"
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoint"
EXPERIMENTAL_ROOT = PACKAGE_ROOT / "experimental"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap(monkeypatch):
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = [str(PACKAGE_ROOT)]
    checkpoint = types.ModuleType("freetoken.checkpoint")
    checkpoint.__path__ = [str(CHECKPOINT_ROOT)]
    experimental = types.ModuleType("freetoken.experimental")
    experimental.__path__ = [str(EXPERIMENTAL_ROOT)]
    models = types.ModuleType("freetoken.models")
    models.__path__ = [str(PACKAGE_ROOT / "models")]
    utils = types.ModuleType("freetoken.utils")
    utils.init_logger = lambda *_a, **_k: SimpleNamespace(
        warning=lambda *_a, **_k: None,
        info=lambda *_a, **_k: None,
    )
    loader = types.ModuleType("freetoken.models.loader")
    loader.drop_page_cache = lambda _path: None
    for name, module in {
        "freetoken": freetoken,
        "freetoken.checkpoint": checkpoint,
        "freetoken.experimental": experimental,
        "freetoken.models": models,
        "freetoken.models.loader": loader,
        "freetoken.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    low = _load(
        "freetoken.checkpoint.low_memory_nvfp4",
        CHECKPOINT_ROOT / "low_memory_nvfp4.py",
    )
    ftw = _load("freetoken.checkpoint.ftw", CHECKPOINT_ROOT / "ftw.py")
    fragment = _load(
        "freetoken.experimental.ftw_fragment_stream",
        EXPERIMENTAL_ROOT / "ftw_fragment_stream.py",
    )
    return low, ftw, fragment


SPEC = SimpleNamespace(
    key_pattern=re.compile(
        r"^layer\.(?P<layer>\d+)\.expert\.(?P<expert>\d+)\."
        r"(?P<proj>gate|up|down)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate": "gate", "up": "up", "down": "down"},
    layer_to_bank=lambda layer, _config: layer,
    desc="synthetic fragmented NVFP4",
)


def _checkpoint(root: Path, *, layers=2, experts=5, hidden=32, inter=32):
    tensors = {}
    for layer in range(layers):
        for expert in range(experts):
            value = layer * 20 + expert + 1
            for proj in ("gate", "up", "down"):
                wshape = (inter, hidden // 2) if proj in ("gate", "up") else (hidden, inter // 2)
                sshape = (inter, hidden // 16) if proj in ("gate", "up") else (hidden, inter // 16)
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight"] = torch.full(
                    wshape, value, dtype=torch.uint8
                )
                scale = torch.full(sshape, float(value), dtype=torch.float32).to(torch.float8_e4m3fn)
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight_scale"] = scale
                tensors[f"layer.{layer}.expert.{expert}.{proj}.weight_scale_2"] = torch.tensor(
                    [float(value)], dtype=torch.float32
                )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    config = SimpleNamespace(
        num_moe_layers=layers,
        num_layers=layers,
        first_k_dense_replace=0,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )
    return config, tensors


def _entry_bytes(out: Path, index: dict, entry: dict) -> bytes:
    pos = int(entry["global_off"])
    remaining = int(entry["nbytes"])
    pieces = []
    for shard in sorted(index["shards"], key=lambda item: item["global_off"]):
        start = int(shard["global_off"])
        end = start + int(shard["nbytes"])
        if pos >= end or remaining <= 0:
            continue
        take = min(remaining, end - pos)
        with (out / shard["file"]).open("rb") as handle:
            handle.seek(pos - start)
            pieces.append(handle.read(take))
        pos += take
        remaining -= take
    assert remaining == 0
    return b"".join(pieces)


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def test_fragment_stream_matches_native_bank_layout_without_layer_materialization(monkeypatch, tmp_path):
    _low, ftw, fragment = _bootstrap(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    config, tensors = _checkpoint(source)
    out = tmp_path / "ftw"
    # 4KiB forces gate_up_packed entries to span shards, exercising fragment writes across
    # physical FTW boundaries without assembling the logical tensor first.
    writer = ftw.FTWWriter(str(out), shard_limit=4096)

    class Sink:
        def __init__(self):
            self._writer = writer
            self._seen = set()
            self.n_written = 0
            self.n_bytes = 0

        def __call__(self, layer_id, banks):
            assert layer_id not in self._seen
            self._seen.add(layer_id)
            for bank_name, bank in banks.items():
                self._writer.add_tensor(
                    f"{bank_name}#L{layer_id:05d}", bank.tensor, kind="experts_bank"
                )
                self.n_written += 1
                self.n_bytes += bank.nbytes
                bank.release()

    sink = Sink()
    stats = fragment.stream_nvfp4_fragments_serial(str(source), config, SPEC, sink)
    index = writer.finalize(
        {"quant_format": "nvfp4", "expert_bank_num_layers": config.num_moe_layers}
    )

    assert stats["layers_streamed"] == 2
    assert stats["ftw_entries_written"] == 12
    assert stats["expert_fragment_peak_bytes"] == 512
    assert sink.n_written == 12
    assert len(sink._seen) == 2

    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    for layer in range(config.num_moe_layers):
        gate = [tensors[f"layer.{layer}.expert.{e}.gate.weight"] for e in range(E)]
        up = [tensors[f"layer.{layer}.expert.{e}.up.weight"] for e in range(E)]
        down = [tensors[f"layer.{layer}.expert.{e}.down.weight"] for e in range(E)]
        gate_s = [tensors[f"layer.{layer}.expert.{e}.gate.weight_scale"] for e in range(E)]
        up_s = [tensors[f"layer.{layer}.expert.{e}.up.weight_scale"] for e in range(E)]
        down_s = [tensors[f"layer.{layer}.expert.{e}.down.weight_scale"] for e in range(E)]

        expected = {
            "gate_up_packed": torch.stack(
                [torch.cat([gate[e], up[e]], dim=0) for e in range(E)], dim=0
            ),
            "gate_up_scale": torch.stack(
                [torch.cat([gate_s[e], up_s[e]], dim=0) for e in range(E)], dim=0
            ),
            "gate_up_global": torch.stack(
                [
                    torch.cat(
                        [
                            tensors[f"layer.{layer}.expert.{e}.gate.weight_scale_2"]
                            .to(torch.float16).expand(I),
                            tensors[f"layer.{layer}.expert.{e}.up.weight_scale_2"]
                            .to(torch.float16).expand(I),
                        ]
                    )
                    for e in range(E)
                ],
                dim=0,
            ),
            "down_packed": torch.stack(down, dim=0),
            "down_scale": torch.stack(down_s, dim=0),
            "down_global": torch.stack(
                [
                    tensors[f"layer.{layer}.expert.{e}.down.weight_scale_2"]
                    .to(torch.float16).expand(H)
                    for e in range(E)
                ],
                dim=0,
            ),
        }
        for bank_name, tensor in expected.items():
            name = f"{bank_name}#L{layer:05d}"
            entry = next(item for item in index["tensors"] if item["name"] == name)
            assert entry["shape"] == list(tensor.shape)
            assert _entry_bytes(out, index, entry) == _raw(tensor)


def test_fragment_writer_fails_closed_on_incomplete_or_wrong_dtype(monkeypatch, tmp_path):
    _low, ftw, fragment = _bootstrap(monkeypatch)
    writer = ftw.FTWWriter(str(tmp_path / "out"), shard_limit=4096)

    incomplete = fragment.FragmentTensor(
        torch.uint8,
        (4,),
        lambda: iter([torch.tensor([1, 2], dtype=torch.uint8)]),
    )
    with fragment._fragment_writer_scope(writer):
        try:
            writer.add_tensor("bad", incomplete)
            raise AssertionError("incomplete fragment stream should fail")
        except ValueError as exc:
            assert "incomplete" in str(exc)

    writer2 = ftw.FTWWriter(str(tmp_path / "out2"), shard_limit=4096)
    wrong_dtype = fragment.FragmentTensor(
        torch.uint8,
        (2,),
        lambda: iter([torch.tensor([1, 2], dtype=torch.int8)]),
    )
    with fragment._fragment_writer_scope(writer2):
        try:
            writer2.add_tensor("bad", wrong_dtype)
            raise AssertionError("wrong fragment dtype should fail")
        except ValueError as exc:
            assert "dtype mismatch" in str(exc)
